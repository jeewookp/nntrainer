// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   qwen3_forward.cpp
 * @date   29 May 2026
 * @brief  Paper-aligned GPU-native Qwen3 forward (skeleton commit).
 */

#include "qwen3_forward.h"

#include <blas_kernels.h>
#include <cl_context.h>
#include <cl_tensor_view.h>
#include <engine.h>
#include <rmsnorm.h>
#include <rmsnorm_fp16.h>

#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <memory>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace causallm_gpu {

namespace {
// Q6_K block layout: 256 elements per block, 210 bytes per block
// (see nntrainer/tensor/q6_k_tensor.h:32 — Q6_K_SIZE = 210).
constexpr size_t Q6_K_BLOCK_ELTS = 256;
constexpr size_t Q6_K_BLOCK_BYTES = 210;
} // namespace

Qwen3Forward::Qwen3Forward() = default;

Qwen3Forward::~Qwen3Forward() {
  release_v8c_weight(&layer0_wq_);
  release_v8c_weight(&layer0_wk_);
  release_v8c_weight(&layer0_wv_);
  if (layer0_q_norm_gamma_svm_fp16_ != nullptr && cl_ctx_ != nullptr)
    clSVMFree(cl_ctx_, layer0_q_norm_gamma_svm_fp16_);
  if (layer0_k_norm_gamma_svm_fp16_ != nullptr && cl_ctx_ != nullptr)
    clSVMFree(cl_ctx_, layer0_k_norm_gamma_svm_fp16_);
  if (layer0_attn_norm_gamma_svm_ != nullptr && cl_ctx_ != nullptr) {
    clSVMFree(cl_ctx_, layer0_attn_norm_gamma_svm_);
  }
  if (weight_mmap_ != nullptr && weight_bytes_ > 0) {
    munmap(const_cast<uint8_t *>(weight_mmap_), weight_bytes_);
  }
  if (weight_fd_ >= 0) {
    close(weight_fd_);
  }
}

void Qwen3Forward::release_v8c_weight(V8cFcWeight *w) {
  if (w->scale_buf != nullptr) clReleaseMemObject(w->scale_buf);
  if (w->row_sum_w_int4 != nullptr) clReleaseMemObject(w->row_sum_w_int4);
  // weight_image is owned by the backing's image cache; the backing's
  // destructor releases it. We don't ReleaseMemObject it ourselves.
  if (w->backing != nullptr) {
    delete static_cast<nntrainer::tv::TensorBacking *>(w->backing);
  }
  *w = V8cFcWeight{};
}

size_t Qwen3Forward::embed_table_bytes() const {
  // Q6_K: vocab * hidden / 256 blocks * 210 bytes/block.
  const size_t total_elts =
    static_cast<size_t>(cfg_.vocab_size) * cfg_.hidden_size;
  if ((total_elts % Q6_K_BLOCK_ELTS) != 0) {
    throw std::runtime_error("Q6_K requires vocab*hidden multiple of 256");
  }
  return (total_elts / Q6_K_BLOCK_ELTS) * Q6_K_BLOCK_BYTES;
}

bool Qwen3Forward::init(const Qwen3Config &cfg, const std::string &weight_path) {
  cfg_ = cfg;
  weight_path_ = weight_path;

  weight_fd_ = open(weight_path.c_str(), O_RDONLY);
  if (weight_fd_ < 0) {
    std::fprintf(stderr, "[qwen3-gpu] open(%s) failed: %s\n",
                 weight_path.c_str(), std::strerror(errno));
    return false;
  }
  struct stat st;
  if (fstat(weight_fd_, &st) != 0) {
    std::fprintf(stderr, "[qwen3-gpu] fstat failed: %s\n", std::strerror(errno));
    return false;
  }
  weight_bytes_ = static_cast<size_t>(st.st_size);
  void *m = mmap(nullptr, weight_bytes_, PROT_READ, MAP_PRIVATE, weight_fd_, 0);
  if (m == MAP_FAILED) {
    std::fprintf(stderr, "[qwen3-gpu] mmap failed: %s\n", std::strerror(errno));
    weight_bytes_ = 0;
    return false;
  }
  weight_mmap_ = static_cast<const uint8_t *>(m);

  auto *cl =
    static_cast<nntrainer::ClContext *>(
      nntrainer::Engine::Global().getRegisteredContext("gpu"));
  if (cl == nullptr) {
    std::fprintf(stderr, "[qwen3-gpu] no gpu context registered\n");
    return false;
  }
  cl_ctx_ = cl->context_inst_.GetContext();
  cl_q_ = cl->command_queue_inst_.GetCommandQueue();
  cl_dev_ = cl->context_inst_.GetDeviceId();
  if (cl_ctx_ == nullptr || cl_q_ == nullptr || cl_dev_ == nullptr) {
    std::fprintf(stderr,
                 "[qwen3-gpu] ClContext handles null: ctx=%p q=%p dev=%p\n",
                 cl_ctx_, cl_q_, cl_dev_);
    return false;
  }

  std::fprintf(stderr,
               "[qwen3-gpu] init OK: weights=%s size=%zu MB cl_ctx=%p\n",
               weight_path.c_str(), weight_bytes_ / (1024 * 1024), cl_ctx_);
  std::fprintf(stderr,
               "[qwen3-gpu] cfg: hidden=%u inter=%u d=%u hQ=%u hKV=%u "
               "L=%u vocab=%u S_max=%u\n",
               cfg_.hidden_size, cfg_.intermediate_size, cfg_.head_dim,
               cfg_.num_heads_Q, cfg_.num_heads_KV, cfg_.num_layers,
               cfg_.vocab_size, cfg_.max_seq_len);
  return true;
}

void Qwen3Forward::dump_weight_header(size_t n) {
  if (weight_mmap_ == nullptr) {
    std::fprintf(stderr, "[qwen3-gpu] dump_weight_header: not mmap'd\n");
    return;
  }
  const size_t lim = (n < weight_bytes_) ? n : weight_bytes_;
  std::fprintf(stderr, "[qwen3-gpu] first %zu bytes of %s:\n", lim,
               weight_path_.c_str());
  for (size_t i = 0; i < lim; ++i) {
    std::fprintf(stderr, "%02x ", weight_mmap_[i]);
    if ((i + 1) % 16 == 0) std::fprintf(stderr, "\n");
  }
  if (lim % 16 != 0) std::fprintf(stderr, "\n");
}

bool Qwen3Forward::svm_smoke_test(size_t bytes) {
  if (cl_ctx_ == nullptr) {
    std::fprintf(stderr, "[qwen3-gpu] svm_smoke_test: no cl_ctx\n");
    return false;
  }
  void *svm = clSVMAlloc(cl_ctx_, CL_MEM_READ_WRITE, bytes, /*alignment*/ 0);
  if (svm == nullptr) {
    std::fprintf(stderr,
                 "[qwen3-gpu] clSVMAlloc(%zu) returned null — SVM may be "
                 "unsupported on this device\n", bytes);
    return false;
  }
  // Map for host write (CL_MAP_WRITE) — coarse-grained SVM requires
  // explicit map/unmap; fine-grained also accepts it as a no-op.
  cl_int err = clEnqueueSVMMap(cl_q_, CL_TRUE, CL_MAP_WRITE, svm, bytes, 0,
                               nullptr, nullptr);
  if (err != CL_SUCCESS) {
    std::fprintf(stderr, "[qwen3-gpu] clEnqueueSVMMap(WRITE) err=%d\n", err);
    clSVMFree(cl_ctx_, svm);
    return false;
  }
  uint8_t *p = static_cast<uint8_t *>(svm);
  for (size_t i = 0; i < bytes; ++i) p[i] = static_cast<uint8_t>(i & 0xFF);
  err = clEnqueueSVMUnmap(cl_q_, svm, 0, nullptr, nullptr);
  if (err != CL_SUCCESS) {
    std::fprintf(stderr, "[qwen3-gpu] clEnqueueSVMUnmap(write) err=%d\n", err);
    clSVMFree(cl_ctx_, svm);
    return false;
  }
  clFinish(cl_q_);

  err = clEnqueueSVMMap(cl_q_, CL_TRUE, CL_MAP_READ, svm, bytes, 0, nullptr,
                        nullptr);
  if (err != CL_SUCCESS) {
    std::fprintf(stderr, "[qwen3-gpu] clEnqueueSVMMap(READ) err=%d\n", err);
    clSVMFree(cl_ctx_, svm);
    return false;
  }
  bool ok = true;
  for (size_t i = 0; i < bytes; ++i) {
    if (p[i] != static_cast<uint8_t>(i & 0xFF)) {
      std::fprintf(stderr,
                   "[qwen3-gpu] svm round-trip mismatch at %zu: got 0x%02x\n",
                   i, p[i]);
      ok = false;
      break;
    }
  }
  err = clEnqueueSVMUnmap(cl_q_, svm, 0, nullptr, nullptr);
  if (err != CL_SUCCESS) {
    std::fprintf(stderr, "[qwen3-gpu] clEnqueueSVMUnmap(read) err=%d\n", err);
    ok = false;
  }
  clFinish(cl_q_);
  clSVMFree(cl_ctx_, svm);
  if (ok) {
    std::fprintf(stderr,
                 "[qwen3-gpu] SVM smoke test PASS: %zu bytes round-trip\n",
                 bytes);
  }
  return ok;
}

bool Qwen3Forward::load_layer0_attention_norm_to_svm() {
  if (weight_mmap_ == nullptr || cl_ctx_ == nullptr) {
    std::fprintf(stderr,
                 "[qwen3-gpu] load_layer0_attention_norm: not initialized\n");
    return false;
  }
  if (layer0_attn_norm_gamma_svm_ != nullptr) {
    return true; // already loaded
  }

  const size_t embed_bytes = embed_table_bytes();
  const size_t gamma_bytes =
    static_cast<size_t>(cfg_.hidden_size) * sizeof(float);
  const size_t gamma_offset = embed_bytes;

  if (gamma_offset + gamma_bytes > weight_bytes_) {
    std::fprintf(stderr,
                 "[qwen3-gpu] computed gamma offset %zu + %zu > file %zu\n",
                 gamma_offset, gamma_bytes, weight_bytes_);
    return false;
  }
  const float *gamma_host =
    reinterpret_cast<const float *>(weight_mmap_ + gamma_offset);

  // Sanity log: dump first 8 + last 4 gamma values. For typical LLM RMSNorm
  // the loaded values cluster near 1.0 (initialized as ones, learned to
  // small deviations). Wildly different values strongly suggest the offset
  // is wrong (we landed mid-Q6_K-block or skipped past a wrong tensor).
  std::fprintf(stderr,
               "[qwen3-gpu] layer 0 attention_norm gamma "
               "(host fp32, offset=%zu MB, %u floats):\n  first 8:",
               gamma_offset / (1024 * 1024), cfg_.hidden_size);
  for (int i = 0; i < 8; ++i)
    std::fprintf(stderr, " %f", gamma_host[i]);
  std::fprintf(stderr, "\n  last  4:");
  for (int i = 0; i < 4; ++i)
    std::fprintf(stderr, " %f",
                 gamma_host[cfg_.hidden_size - 4 + i]);
  std::fprintf(stderr, "\n");

  layer0_attn_norm_gamma_svm_ =
    clSVMAlloc(cl_ctx_, CL_MEM_READ_ONLY, gamma_bytes, /*alignment*/ 0);
  if (layer0_attn_norm_gamma_svm_ == nullptr) {
    std::fprintf(stderr,
                 "[qwen3-gpu] clSVMAlloc(%zu) for gamma failed\n",
                 gamma_bytes);
    return false;
  }
  cl_int err = clEnqueueSVMMap(cl_q_, CL_TRUE, CL_MAP_WRITE,
                               layer0_attn_norm_gamma_svm_, gamma_bytes, 0,
                               nullptr, nullptr);
  if (err != CL_SUCCESS) {
    std::fprintf(stderr, "[qwen3-gpu] gamma SVMMap WRITE err=%d\n", err);
    return false;
  }
  std::memcpy(layer0_attn_norm_gamma_svm_, gamma_host, gamma_bytes);
  err = clEnqueueSVMUnmap(cl_q_, layer0_attn_norm_gamma_svm_, 0, nullptr,
                          nullptr);
  if (err != CL_SUCCESS) {
    std::fprintf(stderr, "[qwen3-gpu] gamma SVMUnmap err=%d\n", err);
    return false;
  }
  clFinish(cl_q_);
  std::fprintf(stderr,
               "[qwen3-gpu] layer 0 attention_norm gamma -> SVM ok "
               "(%zu bytes)\n", gamma_bytes);
  return true;
}

bool Qwen3Forward::run_rmsnorm_layer0() {
  if (layer0_attn_norm_gamma_svm_ == nullptr) {
    std::fprintf(stderr,
                 "[qwen3-gpu] run_rmsnorm_layer0: gamma not loaded\n");
    return false;
  }
  const unsigned int W = cfg_.hidden_size;
  const unsigned int H = 1;
  if (W % 4 != 0) {
    std::fprintf(stderr,
                 "[qwen3-gpu] rmsnorm.cl requires hidden %% 4 == 0\n");
    return false;
  }
  const size_t io_bytes = static_cast<size_t>(W) * sizeof(float);

  // Allocate input + output SVM. Input is a known deterministic pattern
  // so we can spot-check the rmsnorm math by hand.
  void *in_svm =
    clSVMAlloc(cl_ctx_, CL_MEM_READ_ONLY, io_bytes, 0);
  void *out_svm =
    clSVMAlloc(cl_ctx_, CL_MEM_WRITE_ONLY, io_bytes, 0);
  if (in_svm == nullptr || out_svm == nullptr) {
    std::fprintf(stderr, "[qwen3-gpu] rmsnorm SVMAlloc failed\n");
    if (in_svm) clSVMFree(cl_ctx_, in_svm);
    if (out_svm) clSVMFree(cl_ctx_, out_svm);
    return false;
  }

  cl_int err = clEnqueueSVMMap(cl_q_, CL_TRUE, CL_MAP_WRITE, in_svm,
                               io_bytes, 0, nullptr, nullptr);
  if (err != CL_SUCCESS) {
    std::fprintf(stderr, "[qwen3-gpu] rmsnorm in SVMMap WRITE err=%d\n", err);
    clSVMFree(cl_ctx_, in_svm); clSVMFree(cl_ctx_, out_svm);
    return false;
  }
  float *in_ptr = static_cast<float *>(in_svm);
  // Pattern: gentle ramp so RMS = sqrt(mean of squares) is computable.
  // Values 0.001 * (i + 1), i in [0, W). RMS = 0.001 * sqrt(sum/W) where
  // sum = W*(W+1)*(2W+1)/6. For W=2560: sum=5604687360, mean=2189330,
  // sqrt(mean)=1479.6, scale = 1/1479.6 ≈ 6.758e-4. After scale, then
  // multiplied by gamma (~1.0), output[0] ≈ 0.001 * 6.758e-4 ≈ 6.758e-7.
  // (Tiny because pattern range >> gamma range; this is just to verify
  // the kernel runs and produces finite numbers.)
  for (unsigned int i = 0; i < W; ++i)
    in_ptr[i] = 0.001f * static_cast<float>(i + 1);
  err = clEnqueueSVMUnmap(cl_q_, in_svm, 0, nullptr, nullptr);
  if (err != CL_SUCCESS) {
    std::fprintf(stderr, "[qwen3-gpu] rmsnorm in SVMUnmap err=%d\n", err);
    clSVMFree(cl_ctx_, in_svm); clSVMFree(cl_ctx_, out_svm);
    return false;
  }
  clFinish(cl_q_);

  // Register + dispatch rmsnorm_cl.
  auto *cl = static_cast<nntrainer::ClContext *>(
    nntrainer::Engine::Global().getRegisteredContext("gpu"));
  auto kp = cl->registerClKernel(nntrainer::rmsnorm_kernel, "rmsnorm_cl");
  if (!kp) {
    std::fprintf(stderr, "[qwen3-gpu] registerClKernel(rmsnorm_cl) failed\n");
    clSVMFree(cl_ctx_, in_svm); clSVMFree(cl_ctx_, out_svm);
    return false;
  }
  if (!kp->SetKernelSVMArguments(0, in_svm) ||
      !kp->SetKernelSVMArguments(1, out_svm) ||
      !kp->SetKernelSVMArguments(2, layer0_attn_norm_gamma_svm_)) {
    std::fprintf(stderr, "[qwen3-gpu] rmsnorm SVM args failed\n");
    clSVMFree(cl_ctx_, in_svm); clSVMFree(cl_ctx_, out_svm);
    return false;
  }
  float eps = cfg_.rms_norm_eps;
  int H_i = static_cast<int>(H), W_i = static_cast<int>(W);
  if (!kp->SetKernelArguments(3, &eps, sizeof(float)) ||
      !kp->SetKernelArguments(4, &H_i, sizeof(int)) ||
      !kp->SetKernelArguments(5, &W_i, sizeof(int))) {
    std::fprintf(stderr, "[qwen3-gpu] rmsnorm scalar args failed\n");
    clSVMFree(cl_ctx_, in_svm); clSVMFree(cl_ctx_, out_svm);
    return false;
  }
  // 1 workgroup per row (h = get_group_id(0)); LWS=64 to match
  // qcom_reqd_sub_group_size("half") on Adreno.
  std::array<size_t, 1> gws = {static_cast<size_t>(H) * 64};
  std::array<size_t, 1> lws = {64};
  cl->command_queue_inst_.enqueueKernel(kp->GetKernel(), 1, gws.data(),
                                        lws.data(), 0, nullptr, nullptr);
  clFinish(cl_q_);

  // Read back + sanity check.
  err = clEnqueueSVMMap(cl_q_, CL_TRUE, CL_MAP_READ, out_svm, io_bytes, 0,
                        nullptr, nullptr);
  if (err != CL_SUCCESS) {
    std::fprintf(stderr, "[qwen3-gpu] rmsnorm out SVMMap READ err=%d\n", err);
    clSVMFree(cl_ctx_, in_svm); clSVMFree(cl_ctx_, out_svm);
    return false;
  }
  const float *out_ptr = static_cast<const float *>(out_svm);
  bool all_finite = true;
  for (unsigned int i = 0; i < W; ++i) {
    if (!std::isfinite(out_ptr[i])) { all_finite = false; break; }
  }
  // Quick host-side reference compute for the first 4 values.
  // input[i] = 0.001*(i+1); RMS = sqrt(mean(input^2)); scale = 1/sqrt(RMS^2+eps)
  // expected[i] = input[i] * scale * gamma[i]
  err = clEnqueueSVMMap(cl_q_, CL_TRUE, CL_MAP_READ,
                        layer0_attn_norm_gamma_svm_, W * sizeof(float), 0,
                        nullptr, nullptr);
  const float *gamma_ptr =
    static_cast<const float *>(layer0_attn_norm_gamma_svm_);
  double ss = 0.0;
  for (unsigned int i = 0; i < W; ++i) {
    const double v = 0.001 * (i + 1);
    ss += v * v;
  }
  const double mean = ss / W;
  const double scale = 1.0 / std::sqrt(mean + eps);
  std::fprintf(stderr,
               "[qwen3-gpu] rmsnorm dispatch: H=%u W=%u eps=%g\n", H, W, eps);
  std::fprintf(stderr,
               "  host-ref mean=%g scale=%g, expected first 4:\n   ", mean,
               scale);
  for (int i = 0; i < 4; ++i) {
    const double expected =
      0.001 * (i + 1) * scale * static_cast<double>(gamma_ptr[i]);
    std::fprintf(stderr, " %g", expected);
  }
  std::fprintf(stderr, "\n  gpu  out first 4:\n   ");
  for (int i = 0; i < 4; ++i)
    std::fprintf(stderr, " %g", static_cast<double>(out_ptr[i]));
  std::fprintf(stderr, "\n  gpu  out last  4:\n   ");
  for (int i = 0; i < 4; ++i)
    std::fprintf(stderr, " %g", static_cast<double>(out_ptr[W - 4 + i]));
  std::fprintf(stderr, "\n  all_finite=%d\n", all_finite ? 1 : 0);

  err = clEnqueueSVMUnmap(cl_q_, layer0_attn_norm_gamma_svm_, 0, nullptr,
                          nullptr);
  err = clEnqueueSVMUnmap(cl_q_, out_svm, 0, nullptr, nullptr);
  clFinish(cl_q_);
  clSVMFree(cl_ctx_, in_svm);
  clSVMFree(cl_ctx_, out_svm);
  return all_finite;
}

bool Qwen3Forward::load_qint4_weight_at(size_t file_offset, unsigned int K,
                                        unsigned int N, V8cFcWeight *out,
                                        const char *tag) {
  if (weight_mmap_ == nullptr || cl_ctx_ == nullptr) {
    std::fprintf(stderr, "[qwen3-gpu] %s: not initialized\n", tag);
    return false;
  }
  if (out->backing != nullptr) return true; // already loaded

  // [qscheme u16][packed K*N/2 bytes][scales N*u16 bytes].
  const size_t packed_bytes = (size_t)K * N / 2;
  const size_t scales_bytes = (size_t)N * sizeof(uint16_t);
  const size_t total_bytes = sizeof(uint16_t) + packed_bytes + scales_bytes;
  if (file_offset + total_bytes > weight_bytes_) {
    std::fprintf(stderr,
                 "[qwen3-gpu] %s offset %zu + size %zu > file %zu\n", tag,
                 file_offset, total_bytes, weight_bytes_);
    return false;
  }
  const uint16_t qscheme =
    *reinterpret_cast<const uint16_t *>(weight_mmap_ + file_offset);
  const uint8_t *section_a = weight_mmap_ + file_offset + sizeof(uint16_t);
  const uint16_t *scales_fp16 =
    reinterpret_cast<const uint16_t *>(section_a + packed_bytes);

  std::fprintf(stderr,
               "[qwen3-gpu] %s off=%zu (~%zu MB) qscheme=%u K=%u N=%u "
               "packed=%zu KB scales=%zu B\n", tag, file_offset,
               file_offset / (1024 * 1024), qscheme, K, N,
               packed_bytes / 1024, scales_bytes);

  cl_mem scale_buf = nullptr;
  cl_mem rsw_buf = nullptr;
  std::unique_ptr<nntrainer::tv::TensorBacking> backing;
  try {
    backing = nntrainer::make_v8c_weight_backing_from_kai_section_a(
      section_a, scales_fp16, N, K, &scale_buf, &rsw_buf);
  } catch (const std::exception &e) {
    std::fprintf(stderr,
                 "[qwen3-gpu] %s make_v8c_weight_backing threw: %s\n",
                 tag, e.what());
    if (scale_buf) clReleaseMemObject(scale_buf);
    if (rsw_buf) clReleaseMemObject(rsw_buf);
    return false;
  }
  nntrainer::tv::ViewSpec ws;
  ws.kind = nntrainer::tv::ViewKind::IMAGE_2D;
  ws.image_channel_order = CL_RGBA;
  ws.image_channel_type = CL_UNSIGNED_INT32;
  ws.width = K / 32;
  ws.height = N;
  ws.row_pitch_bytes = K / 2;
  try {
    out->weight_image = backing->imageView(ws);
  } catch (const std::exception &e) {
    std::fprintf(stderr, "[qwen3-gpu] %s imageView threw: %s\n", tag,
                 e.what());
    clReleaseMemObject(scale_buf);
    clReleaseMemObject(rsw_buf);
    return false;
  }
  out->backing = backing.release();
  out->scale_buf = scale_buf;
  out->row_sum_w_int4 = rsw_buf;
  out->K = K;
  out->N = N;
  return true;
}

// Convert N fp32 values to fp16-bits (uint16). Round-to-nearest-even.
// Minimal correct converter — for one-time small gamma loads, perf isn't
// a concern. Returns 0x7E00 (qNaN) on NaN, ±inf on overflow, denormal
// on underflow.
static uint16_t f2h(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  uint32_t s = (u >> 16) & 0x8000u;
  int32_t e = ((u >> 23) & 0xff) - 127 + 15;
  uint32_t m = u & 0x7fffff;
  if (((u >> 23) & 0xff) == 0xff) {
    // inf / nan
    return (uint16_t)(s | 0x7c00 | (m ? 0x200 : 0));
  }
  if (e >= 31) return (uint16_t)(s | 0x7c00);             // overflow -> inf
  if (e <= 0) {
    // subnormal
    if (e < -10) return (uint16_t)s;
    m |= 0x800000;
    uint32_t shift = (uint32_t)(14 - e);
    uint32_t half = m >> shift;
    if ((m >> (shift - 1)) & 1u) half += 1; // round to nearest
    return (uint16_t)(s | half);
  }
  uint16_t r = (uint16_t)(s | (uint16_t)(e << 10) | (uint16_t)(m >> 13));
  if (m & 0x1000) r += 1; // round-to-nearest-even (simplified)
  return r;
}

bool Qwen3Forward::load_layer0_qk_norm_gammas() {
  if (weight_mmap_ == nullptr || cl_ctx_ == nullptr) {
    std::fprintf(stderr,
                 "[qwen3-gpu] load_layer0_qk_norm_gammas: not initialized\n");
    return false;
  }
  if (layer0_q_norm_gamma_svm_fp16_ != nullptr &&
      layer0_k_norm_gamma_svm_fp16_ != nullptr)
    return true;

  // Per the layer save layout (qkv weights commit), q_norm/k_norm gammas
  // live right after wq/wk respectively. Recompute offsets here so the
  // loader is independent of the wq load step.
  const size_t embed_bytes = embed_table_bytes();
  const size_t attn_norm_bytes =
    static_cast<size_t>(cfg_.hidden_size) * sizeof(float);
  const size_t head_dim_bytes =
    static_cast<size_t>(cfg_.head_dim) * sizeof(float);

  const unsigned int K_hidden = cfg_.hidden_size;
  const unsigned int N_q = cfg_.num_heads_Q * cfg_.head_dim;
  const unsigned int N_kv = cfg_.num_heads_KV * cfg_.head_dim;
  const size_t wq_bytes =
    sizeof(uint16_t) + (size_t)K_hidden * N_q / 2 + (size_t)N_q * 2;
  const size_t wk_bytes =
    sizeof(uint16_t) + (size_t)K_hidden * N_kv / 2 + (size_t)N_kv * 2;

  const size_t wq_off = embed_bytes + attn_norm_bytes;
  const size_t q_norm_off = wq_off + wq_bytes;
  const size_t wk_off = q_norm_off + head_dim_bytes;
  const size_t k_norm_off = wk_off + wk_bytes;

  if (k_norm_off + head_dim_bytes > weight_bytes_) {
    std::fprintf(stderr,
                 "[qwen3-gpu] qk_norm offsets out of range\n");
    return false;
  }

  const float *q_gamma_fp32 =
    reinterpret_cast<const float *>(weight_mmap_ + q_norm_off);
  const float *k_gamma_fp32 =
    reinterpret_cast<const float *>(weight_mmap_ + k_norm_off);

  std::fprintf(stderr,
               "[qwen3-gpu] q_norm off=%zu (~%zu KB) first 4: %g %g %g %g\n"
               "[qwen3-gpu] k_norm off=%zu (~%zu KB) first 4: %g %g %g %g\n",
               q_norm_off, q_norm_off / 1024,
               q_gamma_fp32[0], q_gamma_fp32[1], q_gamma_fp32[2],
               q_gamma_fp32[3],
               k_norm_off, k_norm_off / 1024,
               k_gamma_fp32[0], k_gamma_fp32[1], k_gamma_fp32[2],
               k_gamma_fp32[3]);

  // Convert fp32 -> fp16 and push to SVM. head_dim values × 2 bytes.
  const size_t gamma_fp16_bytes =
    (size_t)cfg_.head_dim * sizeof(uint16_t);
  auto load_one = [this, gamma_fp16_bytes](
                    const float *src_fp32, void **dst_svm,
                    const char *tag) -> bool {
    *dst_svm = clSVMAlloc(cl_ctx_, CL_MEM_READ_ONLY, gamma_fp16_bytes, 0);
    if (*dst_svm == nullptr) {
      std::fprintf(stderr, "[qwen3-gpu] %s SVMAlloc failed\n", tag);
      return false;
    }
    cl_int err = clEnqueueSVMMap(cl_q_, CL_TRUE, CL_MAP_WRITE, *dst_svm,
                                 gamma_fp16_bytes, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
      std::fprintf(stderr, "[qwen3-gpu] %s SVMMap WRITE err=%d\n", tag, err);
      return false;
    }
    uint16_t *p = static_cast<uint16_t *>(*dst_svm);
    for (unsigned int i = 0; i < cfg_.head_dim; ++i)
      p[i] = f2h(src_fp32[i]);
    err = clEnqueueSVMUnmap(cl_q_, *dst_svm, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) return false;
    clFinish(cl_q_);
    return true;
  };
  if (!load_one(q_gamma_fp32, &layer0_q_norm_gamma_svm_fp16_, "q_norm")) return false;
  if (!load_one(k_gamma_fp32, &layer0_k_norm_gamma_svm_fp16_, "k_norm")) return false;
  std::fprintf(stderr,
               "[qwen3-gpu] q_norm + k_norm gammas -> SVM (fp16, %zu B each)\n",
               gamma_fp16_bytes);
  return true;
}

bool Qwen3Forward::load_layer0_qkv_weights() {
  // Layer save order in Qwen3 createTransformerDecoderBlock +
  // Qwen3Transformer::createAttention:
  //   attn_norm -> wq -> q_norm -> wk -> k_norm -> wv -> wo -> ffn_norm -> ...
  // Per-tensor disk size:
  //   fp32 norm gamma:  width * 4
  //   QINT4 FC weight:  2 + (K*N)/2 + N*2
  const size_t embed_bytes = embed_table_bytes();
  const size_t attn_norm_bytes =
    static_cast<size_t>(cfg_.hidden_size) * sizeof(float);
  const size_t q_norm_bytes =
    static_cast<size_t>(cfg_.head_dim) * sizeof(float);
  const size_t k_norm_bytes = q_norm_bytes;

  const unsigned int K_hidden = cfg_.hidden_size;
  const unsigned int N_q = cfg_.num_heads_Q * cfg_.head_dim;
  const unsigned int N_kv = cfg_.num_heads_KV * cfg_.head_dim;
  const size_t wq_bytes =
    sizeof(uint16_t) + (size_t)K_hidden * N_q / 2 + (size_t)N_q * 2;
  const size_t wk_bytes =
    sizeof(uint16_t) + (size_t)K_hidden * N_kv / 2 + (size_t)N_kv * 2;

  const size_t wq_off = embed_bytes + attn_norm_bytes;
  const size_t wk_off = wq_off + wq_bytes + q_norm_bytes;
  const size_t wv_off = wk_off + wk_bytes + k_norm_bytes;

  return load_qint4_weight_at(wq_off, K_hidden, N_q, &layer0_wq_, "wq") &&
         load_qint4_weight_at(wk_off, K_hidden, N_kv, &layer0_wk_, "wk") &&
         load_qint4_weight_at(wv_off, K_hidden, N_kv, &layer0_wv_, "wv");
}

namespace {
// fp16-bits -> fp32 (host-side decode), used for printing GPU outputs.
inline float h2f(uint16_t h) {
  uint32_t s = (uint32_t)(h & 0x8000u) << 16;
  uint32_t e = (h >> 10) & 0x1fu, m = h & 0x3ffu;
  uint32_t o;
  if (e == 0) o = m ? (m << 13) : 0;
  else if (e == 31) o = (m ? 0x7fc00000u : 0x7f800000u);
  else { e += 112; o = (e << 23) | (m << 13); }
  o |= s;
  float f; std::memcpy(&f, &o, 4); return f;
}

// Summary print + finite check for a fp16 cl_mem of length N. Used as a
// quick sanity gate after each GPU FC dispatch.
bool summarize_fp16_buf(cl_command_queue q, cl_mem buf, unsigned int N,
                        const char *tag) {
  std::vector<uint16_t> host(N);
  cl_int err = clEnqueueReadBuffer(q, buf, CL_TRUE, 0,
                                   (size_t)N * sizeof(uint16_t),
                                   host.data(), 0, nullptr, nullptr);
  if (err != CL_SUCCESS) {
    std::fprintf(stderr, "[qwen3-gpu] %s readback err=%d\n", tag, err);
    return false;
  }
  bool all_finite = true;
  float min_v = std::numeric_limits<float>::infinity();
  float max_v = -std::numeric_limits<float>::infinity();
  for (unsigned int n = 0; n < N; ++n) {
    float f = h2f(host[n]);
    if (!std::isfinite(f)) all_finite = false;
    if (f < min_v) min_v = f;
    if (f > max_v) max_v = f;
  }
  std::fprintf(stderr,
               "[qwen3-gpu] %s fp16 N=%u first 8:", tag, N);
  for (int i = 0; i < 8; ++i)
    std::fprintf(stderr, " %g", h2f(host[i]));
  std::fprintf(stderr, "\n  last 4:");
  for (int i = 0; i < 4; ++i)
    std::fprintf(stderr, " %g", h2f(host[N - 4 + i]));
  std::fprintf(stderr, "\n  min=%g max=%g all_finite=%d\n", min_v, max_v,
               all_finite ? 1 : 0);
  return all_finite;
}
} // namespace

bool Qwen3Forward::run_layer0_qkv_projection() {
  if (layer0_wq_.backing == nullptr || layer0_wk_.backing == nullptr ||
      layer0_wv_.backing == nullptr ||
      layer0_attn_norm_gamma_svm_ == nullptr) {
    std::fprintf(stderr,
                 "[qwen3-gpu] qkv proj: weights or gamma not loaded\n");
    return false;
  }
  // M_pad to the v8c kernel tile (TM=4). Single-token forward → M=1.
  const unsigned int M = 1, M_pad = 4;
  const unsigned int K = cfg_.hidden_size;
  const unsigned int N_q = cfg_.num_heads_Q * cfg_.head_dim;
  const unsigned int N_kv = cfg_.num_heads_KV * cfg_.head_dim;

  cl_int err = CL_SUCCESS;
  auto *cl = static_cast<nntrainer::ClContext *>(
    nntrainer::Engine::Global().getRegisteredContext("gpu"));

  // (a) FC input: deterministic ramp pattern (same as step 2/3).
  const size_t in_bytes = (size_t)M_pad * K * sizeof(float);
  std::vector<float> in_host(M_pad * K, 0.0f);
  for (unsigned int k = 0; k < K; ++k)
    in_host[k] = 0.001f * static_cast<float>(k + 1);
  cl_mem in_buf = clCreateBuffer(cl_ctx_,
                                 CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 in_bytes, in_host.data(), &err);
  cl_mem rmsnorm_out_buf =
    clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE, in_bytes, nullptr, &err);

  // (b) rmsnorm.cl on in_buf with SVM gamma. Single call; output shared
  //     by all three FCs (Q/K/V).
  {
    auto kp = cl->registerClKernel(nntrainer::rmsnorm_kernel, "rmsnorm_cl");
    float eps = cfg_.rms_norm_eps;
    int H = static_cast<int>(M_pad), W = static_cast<int>(K);
    if (!kp ||
        !kp->SetKernelArguments(0, &in_buf, sizeof(cl_mem)) ||
        !kp->SetKernelArguments(1, &rmsnorm_out_buf, sizeof(cl_mem)) ||
        !kp->SetKernelSVMArguments(2, layer0_attn_norm_gamma_svm_) ||
        !kp->SetKernelArguments(3, &eps, sizeof(float)) ||
        !kp->SetKernelArguments(4, &H, sizeof(int)) ||
        !kp->SetKernelArguments(5, &W, sizeof(int))) {
      std::fprintf(stderr, "[qwen3-gpu] qkv proj: rmsnorm args failed\n");
      clReleaseMemObject(in_buf); clReleaseMemObject(rmsnorm_out_buf);
      return false;
    }
    std::array<size_t, 1> gws = {(size_t)M_pad * 64};
    std::array<size_t, 1> lws = {64};
    cl->command_queue_inst_.enqueueKernel(kp->GetKernel(), 1, gws.data(),
                                          lws.data(), 0, nullptr, nullptr);
    clFinish(cl_q_);
  }

  // (c) Shared activation quantization (paper §3.6 fused-quant insight):
  //     quantize ONCE; reuse for all three FCs. act_image is also one-time.
  cl_mem act_i8 = clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                                 (size_t)M_pad * K, nullptr, &err);
  cl_mem act_scale = clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                                    sizeof(float) * M_pad, nullptr, &err);
  cl_mem act_zp = clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                                 sizeof(int) * M_pad, nullptr, &err);
  cl_mem act_rs = clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                                 sizeof(int) * M_pad, nullptr, &err);
  if (err != CL_SUCCESS || !act_i8 || !act_scale || !act_zp || !act_rs) {
    std::fprintf(stderr, "[qwen3-gpu] qkv proj scratch alloc failed\n");
    return false;
  }
  nntrainer::quantize_act_v8c_fp32_cl(rmsnorm_out_buf, act_i8, act_scale,
                                      act_zp, act_rs, M_pad, K);

  cl_image_format afmt{CL_RGBA, CL_UNSIGNED_INT32};
  cl_image_desc adesc{};
  adesc.image_type = CL_MEM_OBJECT_IMAGE2D;
  adesc.image_width = K / 16;
  adesc.image_height = M_pad;
  adesc.image_row_pitch = K;
  adesc.buffer = act_i8;
  cl_mem act_image =
    clCreateImage(cl_ctx_, CL_MEM_READ_ONLY, &afmt, &adesc, nullptr, &err);
  if (err != CL_SUCCESS) {
    std::fprintf(stderr, "[qwen3-gpu] qkv proj act image err=%d\n", err);
    return false;
  }

  // (d) Three GEMM dispatches. y_q [M_pad*N_q], y_k [M_pad*N_kv],
  //     y_v [M_pad*N_kv] — each will feed downstream attention.
  cl_mem y_q = clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                              sizeof(uint16_t) * (size_t)M_pad * N_q,
                              nullptr, &err);
  cl_mem y_k = clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                              sizeof(uint16_t) * (size_t)M_pad * N_kv,
                              nullptr, &err);
  cl_mem y_v = clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                              sizeof(uint16_t) * (size_t)M_pad * N_kv,
                              nullptr, &err);
  if (err != CL_SUCCESS) {
    std::fprintf(stderr, "[qwen3-gpu] qkv proj y_* alloc err=%d\n", err);
    return false;
  }

  nntrainer::gemm_int8_v8c_cl(act_image, layer0_wq_.weight_image, act_scale,
                              layer0_wq_.scale_buf, act_rs, act_zp,
                              layer0_wq_.row_sum_w_int4, y_q, M_pad, N_q, K);
  nntrainer::gemm_int8_v8c_cl(act_image, layer0_wk_.weight_image, act_scale,
                              layer0_wk_.scale_buf, act_rs, act_zp,
                              layer0_wk_.row_sum_w_int4, y_k, M_pad, N_kv, K);
  nntrainer::gemm_int8_v8c_cl(act_image, layer0_wv_.weight_image, act_scale,
                              layer0_wv_.scale_buf, act_rs, act_zp,
                              layer0_wv_.row_sum_w_int4, y_v, M_pad, N_kv, K);
  clFinish(cl_q_);

  // (e) Per-head q_norm / k_norm via rmsnorm_cl_fp16 in place. Q is
  //     reshaped [M=1, hQ*d] -> [1, 1, hQ, d] and normed per-head;
  //     same for K with hKV. V is unchanged (Qwen3 has no v_norm).
  //     Kernel signature: rmsnorm_cl_fp16(in, out, alpha, eps_half,
  //     B, C, H, W). For our case B=1, C=1, H=num_heads, W=head_dim.
  //     GWS = (B*C, H) = (1, num_heads); LWS = (1, 1) — no subgroup
  //     reqs in this kernel.
  if (layer0_q_norm_gamma_svm_fp16_ != nullptr &&
      layer0_k_norm_gamma_svm_fp16_ != nullptr) {
    auto dispatch_qk_norm =
      [&](cl_mem io_buf, void *gamma_svm, unsigned int num_heads,
          const char *tag) -> bool {
      auto kp = cl->registerClKernel(nntrainer::rmsnorm_fp16_kernel,
                                     "rmsnorm_cl_fp16");
      if (!kp) {
        std::fprintf(stderr,
                     "[qwen3-gpu] %s register rmsnorm_cl_fp16 failed\n", tag);
        return false;
      }
      uint16_t eps_h = f2h(cfg_.rms_norm_eps);
      int B = 1, C = 1;
      int H = static_cast<int>(num_heads),
          W = static_cast<int>(cfg_.head_dim);
      if (!kp->SetKernelArguments(0, &io_buf, sizeof(cl_mem)) ||
          !kp->SetKernelArguments(1, &io_buf, sizeof(cl_mem)) || // in-place
          !kp->SetKernelSVMArguments(2, gamma_svm) ||
          !kp->SetKernelArguments(3, &eps_h, sizeof(uint16_t)) ||
          !kp->SetKernelArguments(4, &B, sizeof(int)) ||
          !kp->SetKernelArguments(5, &C, sizeof(int)) ||
          !kp->SetKernelArguments(6, &H, sizeof(int)) ||
          !kp->SetKernelArguments(7, &W, sizeof(int))) {
        std::fprintf(stderr, "[qwen3-gpu] %s rmsnorm_fp16 args failed\n", tag);
        return false;
      }
      std::array<size_t, 2> gws = {(size_t)B * C, (size_t)H};
      std::array<size_t, 2> lws = {1, 1};
      cl->command_queue_inst_.enqueueKernel(kp->GetKernel(), 2, gws.data(),
                                            lws.data(), 0, nullptr, nullptr);
      clFinish(cl_q_);
      return true;
    };
    if (!dispatch_qk_norm(y_q, layer0_q_norm_gamma_svm_fp16_,
                          cfg_.num_heads_Q, "q_norm") ||
        !dispatch_qk_norm(y_k, layer0_k_norm_gamma_svm_fp16_,
                          cfg_.num_heads_KV, "k_norm")) {
      // fall through to summarize so we can see partial state
    }
  }

  // (f) Sanity-check each output (M=0 valid row only). Q/K are
  //     post-q_norm/k_norm; V is post-projection only.
  bool ok_q = summarize_fp16_buf(cl_q_, y_q, N_q,  "Q (post q_norm)");
  bool ok_k = summarize_fp16_buf(cl_q_, y_k, N_kv, "K (post k_norm)");
  bool ok_v = summarize_fp16_buf(cl_q_, y_v, N_kv, "V");

  clReleaseMemObject(y_v);
  clReleaseMemObject(y_k);
  clReleaseMemObject(y_q);
  clReleaseMemObject(act_image);
  clReleaseMemObject(act_rs);
  clReleaseMemObject(act_zp);
  clReleaseMemObject(act_scale);
  clReleaseMemObject(act_i8);
  clReleaseMemObject(rmsnorm_out_buf);
  clReleaseMemObject(in_buf);
  return ok_q && ok_k && ok_v;
}

} // namespace causallm_gpu
