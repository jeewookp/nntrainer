// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   qwen3_forward.cpp
 * @date   29 May 2026
 * @brief  Paper-aligned GPU-native Qwen3 forward (skeleton commit).
 */

#include "qwen3_forward.h"

#include <attention_kernels.h>
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

// =============================================================================
// Inline OpenCL kernels for the GPU-native runtime. Strings are passed to
// ClContext::registerClKernel which caches by string identity, so each one
// only compiles once per process. Kept at file scope so any dispatch method
// can reference them regardless of declaration order.
// =============================================================================

// fp16 -> fp32 element-wise convert. One WI per element; gws = (N).
static const std::string kConvertFp16ToFp32Kernel = R"CL(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void cvt_h2f(__global const half *in, __global float *out,
                      const int n) {
  int i = get_global_id(0);
  if (i < n) out[i] = (float)in[i];
}
)CL";

// fp32 element-wise add. out[i] = a[i] + b[i]. One WI per element.
static const std::string kAddFp32Kernel = R"CL(
__kernel void add_fp32(__global const float *a, __global const float *b,
                       __global float *out, const int n) {
  int i = get_global_id(0);
  if (i < n) out[i] = a[i] + b[i];
}
)CL";

// SwiGLU element-wise: out[i] = silu(gate[i]) * up[i]
//   silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
// fp32 throughout (matches the residual stream dtype). One WI per element.
static const std::string kSwigluFp32Kernel = R"CL(
__kernel void swiglu_fp32(__global const float *gate,
                          __global const float *up,
                          __global float *out, const int n) {
  int i = get_global_id(0);
  if (i < n) {
    float g = gate[i];
    float s = g / (1.0f + exp(-g));
    out[i] = s * up[i];
  }
}
)CL";

// fp16 RoPE: in-place rotation of [num_heads, head_dim] fp16 buffer.
// cos/sin tables are doubled-half (cos[k+half] = cos[k], sin[k+half] =
// sin[k]) to match the CPU mha_core convention. GWS = (num_heads,
// head_dim/2); one WI per rotation pair (writes positions k and k+half).
static const std::string kRopeFp16Kernel = R"CL(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void rope_fp16(__global half *xy,
                        __global const half *cos_tbl,
                        __global const half *sin_tbl,
                        const int num_heads,
                        const int half_d) {
  int h = get_global_id(0);
  int k = get_global_id(1);
  if (h >= num_heads || k >= half_d) return;
  int base = h * (half_d * 2);
  half c = cos_tbl[k];
  half s = sin_tbl[k];
  half x_lo = xy[base + k];
  half x_hi = xy[base + k + half_d];
  xy[base + k]          = x_lo * c - x_hi * s;
  xy[base + k + half_d] = x_hi * c + x_lo * s;
}
)CL";

Qwen3Forward::Qwen3Forward() = default;

Qwen3Forward::~Qwen3Forward() {
  // Generic per-layer state (loaded via load_layer).
  for (auto &lw : layers_) {
    if (lw.attn_norm_gamma_svm && cl_ctx_)
      clSVMFree(cl_ctx_, lw.attn_norm_gamma_svm);
    if (lw.q_norm_gamma_svm_fp16 && cl_ctx_)
      clSVMFree(cl_ctx_, lw.q_norm_gamma_svm_fp16);
    if (lw.k_norm_gamma_svm_fp16 && cl_ctx_)
      clSVMFree(cl_ctx_, lw.k_norm_gamma_svm_fp16);
    if (lw.ffn_norm_gamma_svm && cl_ctx_)
      clSVMFree(cl_ctx_, lw.ffn_norm_gamma_svm);
    if (lw.cache_k_svm && cl_ctx_)
      clSVMFree(cl_ctx_, lw.cache_k_svm);
    if (lw.cache_v_svm && cl_ctx_)
      clSVMFree(cl_ctx_, lw.cache_v_svm);
    release_v8c_weight(&lw.wq);
    release_v8c_weight(&lw.wk);
    release_v8c_weight(&lw.wv);
    release_v8c_weight(&lw.wo);
    release_v8c_weight(&lw.ffn_up);
    release_v8c_weight(&lw.ffn_gate);
    release_v8c_weight(&lw.ffn_down);
  }
  if (layer0_output_fp32_ != nullptr)
    clReleaseMemObject(layer0_output_fp32_);
  if (layer0_residual1_fp32_ != nullptr)
    clReleaseMemObject(layer0_residual1_fp32_);
  if (layer0_ffn_norm_gamma_svm_ != nullptr && cl_ctx_ != nullptr)
    clSVMFree(cl_ctx_, layer0_ffn_norm_gamma_svm_);
  release_v8c_weight(&layer0_wq_);
  release_v8c_weight(&layer0_wk_);
  release_v8c_weight(&layer0_wv_);
  release_v8c_weight(&layer0_wo_);
  release_v8c_weight(&layer0_ffn_up_);
  release_v8c_weight(&layer0_ffn_gate_);
  release_v8c_weight(&layer0_ffn_down_);
  if (layer0_cache_k_svm_ != nullptr && cl_ctx_ != nullptr)
    clSVMFree(cl_ctx_, layer0_cache_k_svm_);
  if (layer0_cache_v_svm_ != nullptr && cl_ctx_ != nullptr)
    clSVMFree(cl_ctx_, layer0_cache_v_svm_);
  if (layer0_rope_cos_svm_fp16_ != nullptr && cl_ctx_ != nullptr)
    clSVMFree(cl_ctx_, layer0_rope_cos_svm_fp16_);
  if (layer0_rope_sin_svm_fp16_ != nullptr && cl_ctx_ != nullptr)
    clSVMFree(cl_ctx_, layer0_rope_sin_svm_fp16_);
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

size_t Qwen3Forward::layers_start_offset() const {
  return embed_table_bytes();
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

bool Qwen3Forward::load_layer0_wo() {
  // wo immediately follows wv (no norm in between). See layer save
  // order in load_layer0_qkv_weights for the running offsets.
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
  const size_t wv_bytes = wk_bytes; // same K, N
  const size_t wq_off = embed_bytes + attn_norm_bytes;
  const size_t wk_off = wq_off + wq_bytes + head_dim_bytes;
  const size_t wv_off = wk_off + wk_bytes + head_dim_bytes;
  const size_t wo_off = wv_off + wv_bytes;

  // wo: [K=hQ*d, N=hidden] in the saved tensor (createTransformerDecoder
  // sets the FC unit to DIM=hidden). After v8c packing, K and N match
  // the on-disk save dim.
  const unsigned int wo_K = cfg_.num_heads_Q * cfg_.head_dim;
  const unsigned int wo_N = cfg_.hidden_size;
  return load_qint4_weight_at(wo_off, wo_K, wo_N, &layer0_wo_, "wo");
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

  // (f) RoPE on Q and K (in place). Skipped when rope freqs haven't been
  //     precomputed for any position (precompute_rope_for_position
  //     wasn't called). V has no RoPE.
  if (layer0_rope_position_ >= 0) {
    if (!run_layer0_rope_on_qk(y_q, y_k)) {
      std::fprintf(stderr, "[qwen3-gpu] RoPE dispatch failed\n");
      // fall through to summarize
    } else {
      std::fprintf(stderr,
                   "[qwen3-gpu] RoPE applied at position=%d\n",
                   layer0_rope_position_);
    }
  }

  // (g) Sanity-check each output (M=0 valid row only). Q/K are
  //     post-q_norm/k_norm + optional RoPE; V is post-projection only.
  bool ok_q = summarize_fp16_buf(cl_q_, y_q, N_q,
                                 "Q (post q_norm + RoPE)");
  bool ok_k = summarize_fp16_buf(cl_q_, y_k, N_kv,
                                 "K (post k_norm + RoPE)");
  bool ok_v = summarize_fp16_buf(cl_q_, y_v, N_kv, "V");

  // (h) KV cache write + attention dispatch. Only runs if a position has
  //     been configured via precompute_rope_for_position AND the KV cache
  //     has been allocated. The cache write is row 0 of y_k / y_v
  //     (M_pad=4 but only row 0 is the valid token) into cache[position].
  //     Then dispatch the existing two_conv_attention_prefill_f16_cl
  //     kernel with svm_inputs=true — first time in the codebase this
  //     path runs in production because the existing CausalLM doesn't
  //     SVM-allocate its KV cache.
  bool ok_attn = true;
  if (layer0_cache_k_svm_ != nullptr && layer0_cache_v_svm_ != nullptr &&
      layer0_rope_position_ >= 0) {
    const unsigned int pos = (unsigned int)layer0_rope_position_;
    const size_t kv_row_elts = (size_t)N_kv;          // hKV * d
    const size_t kv_row_bytes = kv_row_elts * sizeof(uint16_t);
    const size_t q_row_bytes  = (size_t)N_q * sizeof(uint16_t);

    // Copy y_k row 0 -> cache_K[pos], y_v row 0 -> cache_V[pos] via
    // map-source + memcpy-into-SVM-region. Both src and dst are tiny
    // (~2 KB each for our 0.6B config) so the host round-trip is
    // negligible. A pure-GPU path (copy kernel) is a follow-up.
    auto map_src = [&](cl_mem src) -> void * {
      cl_int err;
      void *p = clEnqueueMapBuffer(cl_q_, src, CL_TRUE, CL_MAP_READ, 0,
                                   std::max(q_row_bytes, kv_row_bytes), 0,
                                   nullptr, nullptr, &err);
      return err == CL_SUCCESS ? p : nullptr;
    };
    auto write_into_svm_region = [&](void *svm_base, size_t offset_bytes,
                                     const void *src_host, size_t bytes,
                                     const char *tag) -> bool {
      void *dst = static_cast<uint8_t *>(svm_base) + offset_bytes;
      cl_int err = clEnqueueSVMMap(cl_q_, CL_TRUE, CL_MAP_WRITE, dst, bytes,
                                   0, nullptr, nullptr);
      if (err != CL_SUCCESS) {
        std::fprintf(stderr, "[qwen3-gpu] %s SVMMap WRITE err=%d\n", tag,
                     err);
        return false;
      }
      std::memcpy(dst, src_host, bytes);
      err = clEnqueueSVMUnmap(cl_q_, dst, 0, nullptr, nullptr);
      return err == CL_SUCCESS;
    };

    void *p_k = map_src(y_k);
    if (p_k) {
      write_into_svm_region(layer0_cache_k_svm_, pos * kv_row_bytes, p_k,
                            kv_row_bytes, "cache_K");
      clEnqueueUnmapMemObject(cl_q_, y_k, p_k, 0, nullptr, nullptr);
    }
    void *p_v = map_src(y_v);
    if (p_v) {
      write_into_svm_region(layer0_cache_v_svm_, pos * kv_row_bytes, p_v,
                            kv_row_bytes, "cache_V");
      clEnqueueUnmapMemObject(cl_q_, y_v, p_v, 0, nullptr, nullptr);
    }
    clFinish(cl_q_);
    std::fprintf(stderr,
                 "[qwen3-gpu] wrote K/V at cache position %u\n", pos);

    // Allocate Q SVM + O SVM (per-dispatch scratch) and copy Q row 0.
    void *q_svm = clSVMAlloc(cl_ctx_, CL_MEM_READ_ONLY, q_row_bytes, 0);
    void *o_svm = clSVMAlloc(cl_ctx_, CL_MEM_READ_WRITE, q_row_bytes, 0);
    void *p_q = map_src(y_q);
    if (q_svm && o_svm && p_q) {
      write_into_svm_region(q_svm, 0, p_q, q_row_bytes, "Q_svm");
      clEnqueueUnmapMemObject(cl_q_, y_q, p_q, 0, nullptr, nullptr);
      clFinish(cl_q_);

      // Dispatch attention. M=1 query, N_kv = pos + 1 (all positions 0..pos
      // in cache; with our setup pos=0 means N_kv=1 → degenerate single-
      // token attention where softmax of one element is 1.0, so output
      // per head_q is exactly V[head_q / gqa] for this token. Easy to
      // sanity-check vs the post-projection V values.
      const unsigned int N_kv_cache = pos + 1;
      bool ok = nntrainer::two_conv_attention_prefill_f16_cl(
        static_cast<const uint16_t *>(q_svm),
        static_cast<const uint16_t *>(layer0_cache_k_svm_),
        static_cast<const uint16_t *>(layer0_cache_v_svm_),
        static_cast<uint16_t *>(o_svm),
        /*M=*/1, /*N_kv=*/N_kv_cache,
        cfg_.num_heads_Q, cfg_.num_heads_KV, cfg_.head_dim,
        /*causal=*/true, /*svm_inputs=*/true);
      std::fprintf(stderr,
                   "[qwen3-gpu] two_conv_attention_prefill_f16_cl returned "
                   "%d (M=1, N_kv=%u)\n",
                   (int)ok, N_kv_cache);

      if (ok) {
        // Verify: for M=1, N_kv=1, output per head_q ≈ V[head_q / gqa].
        // Print attention output + sample expected V per head.
        cl_int err = clEnqueueSVMMap(cl_q_, CL_TRUE, CL_MAP_READ, o_svm,
                                     q_row_bytes, 0, nullptr, nullptr);
        if (err == CL_SUCCESS) {
          const uint16_t *o = static_cast<const uint16_t *>(o_svm);
          bool finite = true;
          float min_v = std::numeric_limits<float>::infinity();
          float max_v = -std::numeric_limits<float>::infinity();
          for (unsigned int i = 0; i < N_q; ++i) {
            float f = h2f(o[i]);
            if (!std::isfinite(f)) finite = false;
            if (f < min_v) min_v = f;
            if (f > max_v) max_v = f;
          }
          std::fprintf(stderr,
                       "[qwen3-gpu] Attention output (fp16, N=%u) first 8:",
                       N_q);
          for (int i = 0; i < 8; ++i)
            std::fprintf(stderr, " %g", h2f(o[i]));
          std::fprintf(stderr, "\n  last 4:");
          for (int i = 0; i < 4; ++i)
            std::fprintf(stderr, " %g", h2f(o[N_q - 4 + i]));
          std::fprintf(stderr,
                       "\n  min=%g max=%g all_finite=%d\n",
                       min_v, max_v, finite ? 1 : 0);

          // For N_kv=1 each output[hq, :d] ≈ V[hq/gqa, :d]. Spot-check
          // hq=0 (gqa group 0) against V[0, :d] — both should be the
          // same fp16 bit pattern.
          if (N_kv_cache == 1) {
            void *p_v_again = map_src(y_v);
            if (p_v_again) {
              const uint16_t *v_host =
                static_cast<const uint16_t *>(p_v_again);
              std::fprintf(stderr,
                           "  V[head_kv=0] first 8 ref:");
              for (int i = 0; i < 8; ++i)
                std::fprintf(stderr, " %g", h2f(v_host[i]));
              std::fprintf(stderr, "\n  O[head_q=0] first 8 actual:");
              for (int i = 0; i < 8; ++i)
                std::fprintf(stderr, " %g", h2f(o[i]));
              std::fprintf(stderr, "\n");
              clEnqueueUnmapMemObject(cl_q_, y_v, p_v_again, 0, nullptr,
                                      nullptr);
            }
          }
          clEnqueueSVMUnmap(cl_q_, o_svm, 0, nullptr, nullptr);
          ok_attn = finite;
        } else {
          ok_attn = false;
        }
      } else {
        ok_attn = false;
      }
    }

    // (i) wo (attention output projection): O_svm (fp16) -> wo_out fp16
    //     -> residual_1 = x + wo_out (fp32). Inline cvt_h2f + add_fp32
    //     bridge keeps the residual stream in fp32 (matches the existing
    //     rmsnorm.cl dtype). x is in_buf (fp32 ramp pattern).
    if (ok_attn && layer0_wo_.backing != nullptr && o_svm != nullptr) {
      const unsigned int wo_K = N_q;             // hQ * d = 2048
      const unsigned int wo_N = cfg_.hidden_size; // 1024
      // (i.1) Convert attention output O_svm (fp16, [M_pad*hQ*d]) into
      //       a fresh cl_mem fp32 buffer so quantize_act_v8c_fp32_cl
      //       can consume it. Only M=1 valid row is meaningful; we
      //       still allocate M_pad rows of padding (v8c kernel tile).
      const size_t o_fp32_bytes =
        (size_t)M_pad * wo_K * sizeof(float);
      cl_mem o_fp32 = clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                                     o_fp32_bytes, nullptr, &err);
      // Pre-zero padded rows.
      float zero = 0.0f;
      clEnqueueFillBuffer(cl_q_, o_fp32, &zero, sizeof(float), 0,
                          o_fp32_bytes, 0, nullptr, nullptr);
      {
        auto kp = cl->registerClKernel(kConvertFp16ToFp32Kernel,
                                       "cvt_h2f");
        int n = (int)wo_K;
        if (!kp ||
            !kp->SetKernelSVMArguments(0, o_svm) ||
            !kp->SetKernelArguments(1, &o_fp32, sizeof(cl_mem)) ||
            !kp->SetKernelArguments(2, &n, sizeof(int))) {
          std::fprintf(stderr,
                       "[qwen3-gpu] wo cvt_h2f args failed\n");
          ok_attn = false;
        } else {
          std::array<size_t, 1> gws = {(size_t)wo_K};
          std::array<size_t, 1> lws = {64};
          // pad gws[0] up to multiple of lws[0]
          gws[0] = ((gws[0] + lws[0] - 1) / lws[0]) * lws[0];
          cl->command_queue_inst_.enqueueKernel(kp->GetKernel(), 1,
                                                gws.data(), lws.data(), 0,
                                                nullptr, nullptr);
          clFinish(cl_q_);
        }
      }
      // (i.2) v8c FC on (M_pad=4, K=2048) input -> (M_pad, N=1024)
      //       output. Fresh scratch (separate from QKV's act_i8 etc
      //       because K is different).
      cl_mem wo_act_i8 = clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                                        (size_t)M_pad * wo_K, nullptr, &err);
      cl_mem wo_act_scale = clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                                           sizeof(float) * M_pad, nullptr,
                                           &err);
      cl_mem wo_act_zp = clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                                        sizeof(int) * M_pad, nullptr, &err);
      cl_mem wo_act_rs = clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                                        sizeof(int) * M_pad, nullptr, &err);
      cl_mem wo_y_fp16 = clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                                        sizeof(uint16_t) * (size_t)M_pad *
                                          wo_N,
                                        nullptr, &err);
      nntrainer::quantize_act_v8c_fp32_cl(o_fp32, wo_act_i8, wo_act_scale,
                                          wo_act_zp, wo_act_rs, M_pad, wo_K);
      cl_image_format wo_afmt{CL_RGBA, CL_UNSIGNED_INT32};
      cl_image_desc wo_adesc{};
      wo_adesc.image_type = CL_MEM_OBJECT_IMAGE2D;
      wo_adesc.image_width = wo_K / 16;
      wo_adesc.image_height = M_pad;
      wo_adesc.image_row_pitch = wo_K;
      wo_adesc.buffer = wo_act_i8;
      cl_mem wo_act_image = clCreateImage(cl_ctx_, CL_MEM_READ_ONLY,
                                          &wo_afmt, &wo_adesc, nullptr,
                                          &err);
      nntrainer::gemm_int8_v8c_cl(wo_act_image, layer0_wo_.weight_image,
                                  wo_act_scale, layer0_wo_.scale_buf,
                                  wo_act_rs, wo_act_zp,
                                  layer0_wo_.row_sum_w_int4, wo_y_fp16,
                                  M_pad, wo_N, wo_K);
      clFinish(cl_q_);
      bool ok_wo = summarize_fp16_buf(cl_q_, wo_y_fp16, wo_N, "wo (att->h)");
      // (i.3) Convert wo_out fp16 -> fp32, then add to x (in_buf) to form
      //       the post-attention residual. Allocate the persistent
      //       residual_1 cl_mem on first use.
      cl_mem wo_out_fp32 = clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                                          (size_t)wo_N * sizeof(float),
                                          nullptr, &err);
      {
        auto kp = cl->registerClKernel(kConvertFp16ToFp32Kernel, "cvt_h2f");
        int n = (int)wo_N;
        kp->SetKernelArguments(0, &wo_y_fp16, sizeof(cl_mem));
        kp->SetKernelArguments(1, &wo_out_fp32, sizeof(cl_mem));
        kp->SetKernelArguments(2, &n, sizeof(int));
        std::array<size_t, 1> gws = {(size_t)wo_N};
        std::array<size_t, 1> lws = {64};
        gws[0] = ((gws[0] + lws[0] - 1) / lws[0]) * lws[0];
        cl->command_queue_inst_.enqueueKernel(kp->GetKernel(), 1, gws.data(),
                                              lws.data(), 0, nullptr,
                                              nullptr);
        clFinish(cl_q_);
      }
      if (layer0_residual1_fp32_ == nullptr) {
        layer0_residual1_fp32_ =
          clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                         (size_t)wo_N * sizeof(float), nullptr, &err);
      }
      {
        auto kp = cl->registerClKernel(kAddFp32Kernel, "add_fp32");
        int n = (int)wo_N;
        kp->SetKernelArguments(0, &in_buf, sizeof(cl_mem));
        kp->SetKernelArguments(1, &wo_out_fp32, sizeof(cl_mem));
        kp->SetKernelArguments(2, &layer0_residual1_fp32_, sizeof(cl_mem));
        kp->SetKernelArguments(3, &n, sizeof(int));
        std::array<size_t, 1> gws = {(size_t)wo_N};
        std::array<size_t, 1> lws = {64};
        gws[0] = ((gws[0] + lws[0] - 1) / lws[0]) * lws[0];
        cl->command_queue_inst_.enqueueKernel(kp->GetKernel(), 1, gws.data(),
                                              lws.data(), 0, nullptr,
                                              nullptr);
        clFinish(cl_q_);
      }
      // Custom fp32 summary for residual_1 (post wo + residual add).
      {
        std::vector<float> r(wo_N);
        clEnqueueReadBuffer(cl_q_, layer0_residual1_fp32_, CL_TRUE, 0,
                            (size_t)wo_N * sizeof(float), r.data(), 0,
                            nullptr, nullptr);
        bool finite = true;
        float mn = std::numeric_limits<float>::infinity();
        float mx = -mn;
        for (unsigned int i = 0; i < wo_N; ++i) {
          if (!std::isfinite(r[i])) finite = false;
          if (r[i] < mn) mn = r[i];
          if (r[i] > mx) mx = r[i];
        }
        std::fprintf(stderr,
                     "[qwen3-gpu] residual_1 fp32 N=%u first 8:", wo_N);
        for (int i = 0; i < 8; ++i)
          std::fprintf(stderr, " %g", r[i]);
        std::fprintf(stderr, "\n  last 4:");
        for (int i = 0; i < 4; ++i)
          std::fprintf(stderr, " %g", r[wo_N - 4 + i]);
        std::fprintf(stderr,
                     "\n  min=%g max=%g all_finite=%d\n", mn, mx, finite);
        ok_attn = ok_attn && ok_wo && finite;
      }
      clReleaseMemObject(wo_out_fp32);
      clReleaseMemObject(wo_act_image);
      clReleaseMemObject(wo_y_fp16);
      clReleaseMemObject(wo_act_rs);
      clReleaseMemObject(wo_act_zp);
      clReleaseMemObject(wo_act_scale);
      clReleaseMemObject(wo_act_i8);
      clReleaseMemObject(o_fp32);
    }

    if (q_svm) clSVMFree(cl_ctx_, q_svm);
    if (o_svm) clSVMFree(cl_ctx_, o_svm);
  }

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
  return ok_q && ok_k && ok_v && ok_attn;
}

// Inline fp16 RoPE kernel for the GPU-native runtime. Operates in place
// on a [num_heads, head_dim] fp16 buffer. cos/sin tables are
// pre-doubled (cos[k+half] = cos[k], sin[k+half] = sin[k]) to match the
// CPU mha_core convention. Math: for k in [0, half):
//   x_lo = xy[h, k]; x_hi = xy[h, k + half]
//   xy[h, k]        = x_lo * cos[k] - x_hi * sin[k]
//   xy[h, k + half] = x_hi * cos[k] + x_lo * sin[k]
// (compute_rotary_emb_value's transformed_value sign convention).
// GWS = (num_heads, half); one WI per rotation pair.
bool Qwen3Forward::precompute_rope_for_position(unsigned int position) {
  if (cl_ctx_ == nullptr) return false;
  const unsigned int d = cfg_.head_dim;
  const unsigned int half = d / 2;
  const size_t tbl_bytes = (size_t)d * sizeof(uint16_t);

  // (Re)allocate SVM if first time.
  auto ensure_svm = [this, tbl_bytes](void **dst) -> bool {
    if (*dst != nullptr) return true;
    *dst = clSVMAlloc(cl_ctx_, CL_MEM_READ_ONLY, tbl_bytes, 0);
    if (*dst == nullptr) {
      std::fprintf(stderr, "[qwen3-gpu] rope SVMAlloc failed\n");
      return false;
    }
    return true;
  };
  if (!ensure_svm(&layer0_rope_cos_svm_fp16_) ||
      !ensure_svm(&layer0_rope_sin_svm_fp16_))
    return false;

  // Compute thetas[j] = theta ^ (-2j/d) for j in [0, half), then for
  // this position, cos/sin per "doubled half" layout.
  std::vector<float> cos_tmp(d), sin_tmp(d);
  for (unsigned int j = 0; j < half; ++j) {
    float exponent = -2.0f * static_cast<float>(j) / static_cast<float>(d);
    float theta_j = std::pow(cfg_.rope_theta, exponent);
    float angle = static_cast<float>(position) * theta_j;
    float c = std::cos(angle);
    float s = std::sin(angle);
    cos_tmp[j]        = c;
    cos_tmp[j + half] = c;
    sin_tmp[j]        = s;
    sin_tmp[j + half] = s;
  }

  // Push (converted to fp16) into SVM.
  auto write_svm = [&](void *dst, const std::vector<float> &src,
                       const char *tag) -> bool {
    cl_int err = clEnqueueSVMMap(cl_q_, CL_TRUE, CL_MAP_WRITE, dst,
                                 tbl_bytes, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
      std::fprintf(stderr, "[qwen3-gpu] rope %s SVMMap WRITE err=%d\n",
                   tag, err);
      return false;
    }
    uint16_t *p = static_cast<uint16_t *>(dst);
    for (unsigned int i = 0; i < d; ++i) p[i] = f2h(src[i]);
    err = clEnqueueSVMUnmap(cl_q_, dst, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) return false;
    clFinish(cl_q_);
    return true;
  };
  if (!write_svm(layer0_rope_cos_svm_fp16_, cos_tmp, "cos")) return false;
  if (!write_svm(layer0_rope_sin_svm_fp16_, sin_tmp, "sin")) return false;
  layer0_rope_position_ = static_cast<int>(position);
  std::fprintf(stderr,
               "[qwen3-gpu] RoPE freqs precomputed for position=%u: "
               "cos[0..3] = %g %g %g %g, sin[0..3] = %g %g %g %g\n",
               position, cos_tmp[0], cos_tmp[1], cos_tmp[2], cos_tmp[3],
               sin_tmp[0], sin_tmp[1], sin_tmp[2], sin_tmp[3]);
  return true;
}

bool Qwen3Forward::run_layer0_rope_on_qk(cl_mem y_q, cl_mem y_k) {
  if (layer0_rope_cos_svm_fp16_ == nullptr ||
      layer0_rope_sin_svm_fp16_ == nullptr || layer0_rope_position_ < 0) {
    std::fprintf(stderr,
                 "[qwen3-gpu] run_layer0_rope_on_qk: freqs not loaded\n");
    return false;
  }
  auto *cl = static_cast<nntrainer::ClContext *>(
    nntrainer::Engine::Global().getRegisteredContext("gpu"));
  auto kp = cl->registerClKernel(kRopeFp16Kernel, "rope_fp16");
  if (!kp) {
    std::fprintf(stderr, "[qwen3-gpu] rope_fp16 register failed\n");
    return false;
  }
  auto dispatch_one = [&](cl_mem io, unsigned int num_heads,
                          const char *tag) -> bool {
    int nh = static_cast<int>(num_heads);
    int half_d = static_cast<int>(cfg_.head_dim / 2);
    if (!kp->SetKernelArguments(0, &io, sizeof(cl_mem)) ||
        !kp->SetKernelSVMArguments(1, layer0_rope_cos_svm_fp16_) ||
        !kp->SetKernelSVMArguments(2, layer0_rope_sin_svm_fp16_) ||
        !kp->SetKernelArguments(3, &nh, sizeof(int)) ||
        !kp->SetKernelArguments(4, &half_d, sizeof(int))) {
      std::fprintf(stderr, "[qwen3-gpu] %s rope args failed\n", tag);
      return false;
    }
    std::array<size_t, 2> gws = {(size_t)num_heads,
                                 (size_t)(cfg_.head_dim / 2)};
    std::array<size_t, 2> lws = {1, 1};
    cl->command_queue_inst_.enqueueKernel(kp->GetKernel(), 2, gws.data(),
                                          lws.data(), 0, nullptr, nullptr);
    return true;
  };
  if (!dispatch_one(y_q, cfg_.num_heads_Q, "Q")) return false;
  if (!dispatch_one(y_k, cfg_.num_heads_KV, "K")) return false;
  clFinish(cl_q_);
  return true;
}

bool Qwen3Forward::load_layer0_ffn_weights() {
  // After wo there are NO weights for decoder_add. ffn_norm gamma is
  // the next slab, then ffn_up, ffn_gate, ffn_down.
  const size_t embed_bytes = embed_table_bytes();
  const size_t attn_norm_bytes =
    static_cast<size_t>(cfg_.hidden_size) * sizeof(float);
  const size_t head_dim_bytes =
    static_cast<size_t>(cfg_.head_dim) * sizeof(float);
  const unsigned int K_h = cfg_.hidden_size;
  const unsigned int N_q = cfg_.num_heads_Q * cfg_.head_dim;
  const unsigned int N_kv = cfg_.num_heads_KV * cfg_.head_dim;
  const size_t wq_bytes =
    sizeof(uint16_t) + (size_t)K_h * N_q / 2 + (size_t)N_q * 2;
  const size_t wk_bytes =
    sizeof(uint16_t) + (size_t)K_h * N_kv / 2 + (size_t)N_kv * 2;
  const size_t wv_bytes = wk_bytes;
  const size_t wo_K = N_q;
  const size_t wo_N = K_h;
  const size_t wo_bytes =
    sizeof(uint16_t) + wo_K * wo_N / 2 + (size_t)wo_N * 2;

  const size_t wq_off = embed_bytes + attn_norm_bytes;
  const size_t wk_off = wq_off + wq_bytes + head_dim_bytes;
  const size_t wv_off = wk_off + wk_bytes + head_dim_bytes;
  const size_t wo_off = wv_off + wv_bytes;
  const size_t ffn_norm_off = wo_off + wo_bytes;
  // ffn_norm gamma fp32 [hidden] = hidden * 4 bytes
  const size_t ffn_norm_bytes =
    (size_t)cfg_.hidden_size * sizeof(float);
  // ffn_up: K=hidden, N=intermediate
  const unsigned int up_K = cfg_.hidden_size;
  const unsigned int up_N = cfg_.intermediate_size;
  const size_t up_bytes =
    sizeof(uint16_t) + (size_t)up_K * up_N / 2 + (size_t)up_N * 2;
  // ffn_gate: same shape as ffn_up
  const size_t gate_bytes = up_bytes;
  // ffn_down: K=intermediate, N=hidden
  const unsigned int dn_K = cfg_.intermediate_size;
  const unsigned int dn_N = cfg_.hidden_size;

  const size_t ffn_up_off = ffn_norm_off + ffn_norm_bytes;
  const size_t ffn_gate_off = ffn_up_off + up_bytes;
  const size_t ffn_down_off = ffn_gate_off + gate_bytes;

  // (a) ffn_norm gamma load to SVM (fp32 path matches rmsnorm.cl).
  if (layer0_ffn_norm_gamma_svm_ == nullptr) {
    if (ffn_norm_off + ffn_norm_bytes > weight_bytes_) {
      std::fprintf(stderr,
                   "[qwen3-gpu] ffn_norm gamma offset out of range\n");
      return false;
    }
    const float *gp =
      reinterpret_cast<const float *>(weight_mmap_ + ffn_norm_off);
    std::fprintf(stderr,
                 "[qwen3-gpu] ffn_norm off=%zu (~%zu KB) first 8:",
                 ffn_norm_off, ffn_norm_off / 1024);
    for (int i = 0; i < 8; ++i) std::fprintf(stderr, " %g", gp[i]);
    std::fprintf(stderr, "\n");
    layer0_ffn_norm_gamma_svm_ =
      clSVMAlloc(cl_ctx_, CL_MEM_READ_ONLY, ffn_norm_bytes, 0);
    if (layer0_ffn_norm_gamma_svm_ == nullptr) {
      std::fprintf(stderr, "[qwen3-gpu] ffn_norm SVMAlloc failed\n");
      return false;
    }
    cl_int err = clEnqueueSVMMap(cl_q_, CL_TRUE, CL_MAP_WRITE,
                                 layer0_ffn_norm_gamma_svm_, ffn_norm_bytes,
                                 0, nullptr, nullptr);
    if (err != CL_SUCCESS) return false;
    std::memcpy(layer0_ffn_norm_gamma_svm_, gp, ffn_norm_bytes);
    clEnqueueSVMUnmap(cl_q_, layer0_ffn_norm_gamma_svm_, 0, nullptr,
                      nullptr);
    clFinish(cl_q_);
  }

  // (b) Three QINT4 FCs: ffn_up, ffn_gate, ffn_down.
  return load_qint4_weight_at(ffn_up_off, up_K, up_N, &layer0_ffn_up_,
                              "ffn_up") &&
         load_qint4_weight_at(ffn_gate_off, up_K, up_N, &layer0_ffn_gate_,
                              "ffn_gate") &&
         load_qint4_weight_at(ffn_down_off, dn_K, dn_N, &layer0_ffn_down_,
                              "ffn_down");
}

bool Qwen3Forward::run_layer0_ffn() {
  if (layer0_residual1_fp32_ == nullptr ||
      layer0_ffn_norm_gamma_svm_ == nullptr ||
      layer0_ffn_up_.backing == nullptr ||
      layer0_ffn_gate_.backing == nullptr ||
      layer0_ffn_down_.backing == nullptr) {
    std::fprintf(stderr,
                 "[qwen3-gpu] run_layer0_ffn: residual_1 or weights not "
                 "loaded\n");
    return false;
  }
  cl_int err = CL_SUCCESS;
  auto *cl = static_cast<nntrainer::ClContext *>(
    nntrainer::Engine::Global().getRegisteredContext("gpu"));

  const unsigned int M_pad = 4;
  const unsigned int K_h = cfg_.hidden_size;        // 1024
  const unsigned int I = cfg_.intermediate_size;    // 3072

  // (a) ffn_norm.cl on residual_1 (fp32 [K_h]) -> ffn_normed (fp32).
  //     Need M_pad rows for the v8c quantize/GEMM tile alignment, so
  //     allocate a [M_pad * K_h] buffer with residual_1 in row 0 and
  //     zeros elsewhere. residual_1_fp32_ is [K_h]; clEnqueueCopyBuffer
  //     into row 0.
  const size_t row_h_bytes = (size_t)K_h * sizeof(float);
  cl_mem ffn_in_padded =
    clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                   (size_t)M_pad * row_h_bytes, nullptr, &err);
  float zero = 0.0f;
  clEnqueueFillBuffer(cl_q_, ffn_in_padded, &zero, sizeof(float), 0,
                      (size_t)M_pad * row_h_bytes, 0, nullptr, nullptr);
  clEnqueueCopyBuffer(cl_q_, layer0_residual1_fp32_, ffn_in_padded, 0, 0,
                      row_h_bytes, 0, nullptr, nullptr);
  cl_mem ffn_normed = clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                                     (size_t)M_pad * row_h_bytes, nullptr,
                                     &err);
  {
    auto kp = cl->registerClKernel(nntrainer::rmsnorm_kernel, "rmsnorm_cl");
    float eps = cfg_.rms_norm_eps;
    int H = (int)M_pad, W = (int)K_h;
    if (!kp ||
        !kp->SetKernelArguments(0, &ffn_in_padded, sizeof(cl_mem)) ||
        !kp->SetKernelArguments(1, &ffn_normed, sizeof(cl_mem)) ||
        !kp->SetKernelSVMArguments(2, layer0_ffn_norm_gamma_svm_) ||
        !kp->SetKernelArguments(3, &eps, sizeof(float)) ||
        !kp->SetKernelArguments(4, &H, sizeof(int)) ||
        !kp->SetKernelArguments(5, &W, sizeof(int))) {
      std::fprintf(stderr, "[qwen3-gpu] ffn rmsnorm args failed\n");
      return false;
    }
    std::array<size_t, 1> gws = {(size_t)M_pad * 64};
    std::array<size_t, 1> lws = {64};
    cl->command_queue_inst_.enqueueKernel(kp->GetKernel(), 1, gws.data(),
                                          lws.data(), 0, nullptr, nullptr);
    clFinish(cl_q_);
  }

  // (b) Shared activation quant for ffn_up + ffn_gate.
  cl_mem act_i8 = clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                                 (size_t)M_pad * K_h, nullptr, &err);
  cl_mem act_scale = clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                                    sizeof(float) * M_pad, nullptr, &err);
  cl_mem act_zp = clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                                 sizeof(int) * M_pad, nullptr, &err);
  cl_mem act_rs = clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                                 sizeof(int) * M_pad, nullptr, &err);
  nntrainer::quantize_act_v8c_fp32_cl(ffn_normed, act_i8, act_scale, act_zp,
                                      act_rs, M_pad, K_h);
  cl_image_format afmt{CL_RGBA, CL_UNSIGNED_INT32};
  cl_image_desc adesc{};
  adesc.image_type = CL_MEM_OBJECT_IMAGE2D;
  adesc.image_width = K_h / 16;
  adesc.image_height = M_pad;
  adesc.image_row_pitch = K_h;
  adesc.buffer = act_i8;
  cl_mem act_image =
    clCreateImage(cl_ctx_, CL_MEM_READ_ONLY, &afmt, &adesc, nullptr, &err);

  // (c) GEMM ffn_up -> up_fp16, GEMM ffn_gate -> gate_fp16.
  cl_mem up_fp16 = clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                                  sizeof(uint16_t) * (size_t)M_pad * I,
                                  nullptr, &err);
  cl_mem gate_fp16 = clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                                    sizeof(uint16_t) * (size_t)M_pad * I,
                                    nullptr, &err);
  nntrainer::gemm_int8_v8c_cl(act_image, layer0_ffn_up_.weight_image,
                              act_scale, layer0_ffn_up_.scale_buf, act_rs,
                              act_zp, layer0_ffn_up_.row_sum_w_int4,
                              up_fp16, M_pad, I, K_h);
  nntrainer::gemm_int8_v8c_cl(act_image, layer0_ffn_gate_.weight_image,
                              act_scale, layer0_ffn_gate_.scale_buf, act_rs,
                              act_zp, layer0_ffn_gate_.row_sum_w_int4,
                              gate_fp16, M_pad, I, K_h);
  clFinish(cl_q_);

  // (d) Convert up/gate fp16 -> fp32 (for swiglu in fp32) and swiglu.
  cl_mem up_fp32 = clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                                  (size_t)I * sizeof(float), nullptr, &err);
  cl_mem gate_fp32 = clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                                    (size_t)I * sizeof(float), nullptr,
                                    &err);
  auto dispatch_cvt = [&](cl_mem in_fp16, cl_mem out_fp32, unsigned int n,
                          const char *tag) -> bool {
    auto kp = cl->registerClKernel(kConvertFp16ToFp32Kernel, "cvt_h2f");
    int ni = (int)n;
    if (!kp ||
        !kp->SetKernelArguments(0, &in_fp16, sizeof(cl_mem)) ||
        !kp->SetKernelArguments(1, &out_fp32, sizeof(cl_mem)) ||
        !kp->SetKernelArguments(2, &ni, sizeof(int))) {
      std::fprintf(stderr, "[qwen3-gpu] ffn %s cvt args failed\n", tag);
      return false;
    }
    std::array<size_t, 1> gws = {((size_t)n + 63) / 64 * 64};
    std::array<size_t, 1> lws = {64};
    cl->command_queue_inst_.enqueueKernel(kp->GetKernel(), 1, gws.data(),
                                          lws.data(), 0, nullptr, nullptr);
    return true;
  };
  dispatch_cvt(up_fp16, up_fp32, I, "up");
  dispatch_cvt(gate_fp16, gate_fp32, I, "gate");
  clFinish(cl_q_);

  cl_mem swiglu_out = clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                                     (size_t)M_pad * I * sizeof(float),
                                     nullptr, &err);
  clEnqueueFillBuffer(cl_q_, swiglu_out, &zero, sizeof(float), 0,
                      (size_t)M_pad * I * sizeof(float), 0, nullptr, nullptr);
  {
    auto kp = cl->registerClKernel(kSwigluFp32Kernel, "swiglu_fp32");
    int n = (int)I;
    kp->SetKernelArguments(0, &gate_fp32, sizeof(cl_mem));
    kp->SetKernelArguments(1, &up_fp32, sizeof(cl_mem));
    kp->SetKernelArguments(2, &swiglu_out, sizeof(cl_mem));
    kp->SetKernelArguments(3, &n, sizeof(int));
    std::array<size_t, 1> gws = {((size_t)I + 63) / 64 * 64};
    std::array<size_t, 1> lws = {64};
    cl->command_queue_inst_.enqueueKernel(kp->GetKernel(), 1, gws.data(),
                                          lws.data(), 0, nullptr, nullptr);
    clFinish(cl_q_);
  }

  // (e) quantize swiglu_out + v8c(ffn_down) -> ffn_down_out fp16.
  cl_mem dn_act_i8 = clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                                    (size_t)M_pad * I, nullptr, &err);
  cl_mem dn_act_scale = clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                                       sizeof(float) * M_pad, nullptr, &err);
  cl_mem dn_act_zp = clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                                    sizeof(int) * M_pad, nullptr, &err);
  cl_mem dn_act_rs = clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                                    sizeof(int) * M_pad, nullptr, &err);
  nntrainer::quantize_act_v8c_fp32_cl(swiglu_out, dn_act_i8, dn_act_scale,
                                      dn_act_zp, dn_act_rs, M_pad, I);
  cl_image_desc dn_adesc{};
  dn_adesc.image_type = CL_MEM_OBJECT_IMAGE2D;
  dn_adesc.image_width = I / 16;
  dn_adesc.image_height = M_pad;
  dn_adesc.image_row_pitch = I;
  dn_adesc.buffer = dn_act_i8;
  cl_mem dn_act_image =
    clCreateImage(cl_ctx_, CL_MEM_READ_ONLY, &afmt, &dn_adesc, nullptr,
                  &err);
  cl_mem dn_fp16 = clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                                  sizeof(uint16_t) * (size_t)M_pad * K_h,
                                  nullptr, &err);
  nntrainer::gemm_int8_v8c_cl(dn_act_image, layer0_ffn_down_.weight_image,
                              dn_act_scale, layer0_ffn_down_.scale_buf,
                              dn_act_rs, dn_act_zp,
                              layer0_ffn_down_.row_sum_w_int4, dn_fp16,
                              M_pad, K_h, I);
  clFinish(cl_q_);
  summarize_fp16_buf(cl_q_, dn_fp16, K_h, "ffn_down");

  // (f) cvt dn_fp16 -> fp32, then residual_2 = residual_1 + dn_fp32.
  cl_mem dn_fp32 = clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                                  (size_t)K_h * sizeof(float), nullptr,
                                  &err);
  dispatch_cvt(dn_fp16, dn_fp32, K_h, "dn");
  clFinish(cl_q_);

  if (layer0_output_fp32_ == nullptr) {
    layer0_output_fp32_ = clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                                         (size_t)K_h * sizeof(float),
                                         nullptr, &err);
  }
  {
    auto kp = cl->registerClKernel(kAddFp32Kernel, "add_fp32");
    int n = (int)K_h;
    kp->SetKernelArguments(0, &layer0_residual1_fp32_, sizeof(cl_mem));
    kp->SetKernelArguments(1, &dn_fp32, sizeof(cl_mem));
    kp->SetKernelArguments(2, &layer0_output_fp32_, sizeof(cl_mem));
    kp->SetKernelArguments(3, &n, sizeof(int));
    std::array<size_t, 1> gws = {((size_t)K_h + 63) / 64 * 64};
    std::array<size_t, 1> lws = {64};
    cl->command_queue_inst_.enqueueKernel(kp->GetKernel(), 1, gws.data(),
                                          lws.data(), 0, nullptr, nullptr);
    clFinish(cl_q_);
  }

  // (g) Custom fp32 summary for layer 0 output.
  {
    std::vector<float> r(K_h);
    clEnqueueReadBuffer(cl_q_, layer0_output_fp32_, CL_TRUE, 0,
                        (size_t)K_h * sizeof(float), r.data(), 0, nullptr,
                        nullptr);
    bool finite = true;
    float mn = std::numeric_limits<float>::infinity();
    float mx = -mn;
    for (unsigned int i = 0; i < K_h; ++i) {
      if (!std::isfinite(r[i])) finite = false;
      if (r[i] < mn) mn = r[i];
      if (r[i] > mx) mx = r[i];
    }
    std::fprintf(stderr,
                 "[qwen3-gpu] layer0 output fp32 N=%u first 8:", K_h);
    for (int i = 0; i < 8; ++i) std::fprintf(stderr, " %g", r[i]);
    std::fprintf(stderr, "\n  last 4:");
    for (int i = 0; i < 4; ++i) std::fprintf(stderr, " %g", r[K_h - 4 + i]);
    std::fprintf(stderr,
                 "\n  min=%g max=%g all_finite=%d\n", mn, mx, finite);
    if (!finite) return false;
  }

  clReleaseMemObject(dn_fp32);
  clReleaseMemObject(dn_fp16);
  clReleaseMemObject(dn_act_image);
  clReleaseMemObject(dn_act_rs);
  clReleaseMemObject(dn_act_zp);
  clReleaseMemObject(dn_act_scale);
  clReleaseMemObject(dn_act_i8);
  clReleaseMemObject(swiglu_out);
  clReleaseMemObject(gate_fp32);
  clReleaseMemObject(up_fp32);
  clReleaseMemObject(gate_fp16);
  clReleaseMemObject(up_fp16);
  clReleaseMemObject(act_image);
  clReleaseMemObject(act_rs);
  clReleaseMemObject(act_zp);
  clReleaseMemObject(act_scale);
  clReleaseMemObject(act_i8);
  clReleaseMemObject(ffn_normed);
  clReleaseMemObject(ffn_in_padded);
  return true;
}

// Load a [head_dim] fp32 norm gamma at file_offset into SVM, converted
// to fp16 (for the rmsnorm_cl_fp16 kernel). Used by load_layer for
// q_norm / k_norm. Bytes consumed: head_dim * sizeof(float).
static bool load_qk_norm_to_svm_fp16(cl_context cl_ctx, cl_command_queue q,
                                     const float *src_fp32,
                                     unsigned int head_dim,
                                     void **dst_svm, const char *tag) {
  const size_t bytes = (size_t)head_dim * sizeof(uint16_t);
  *dst_svm = clSVMAlloc(cl_ctx, CL_MEM_READ_ONLY, bytes, 0);
  if (*dst_svm == nullptr) {
    std::fprintf(stderr, "[qwen3-gpu] %s SVMAlloc failed\n", tag);
    return false;
  }
  if (clEnqueueSVMMap(q, CL_TRUE, CL_MAP_WRITE, *dst_svm, bytes, 0,
                      nullptr, nullptr) != CL_SUCCESS)
    return false;
  uint16_t *p = static_cast<uint16_t *>(*dst_svm);
  for (unsigned int i = 0; i < head_dim; ++i) p[i] = f2h(src_fp32[i]);
  clEnqueueSVMUnmap(q, *dst_svm, 0, nullptr, nullptr);
  clFinish(q);
  return true;
}

// Load a [hidden] fp32 norm gamma at file_offset into SVM as fp32
// (matches the rmsnorm.cl kernel). Used by load_layer for attn_norm
// and ffn_norm.
static bool load_norm_to_svm_fp32(cl_context cl_ctx, cl_command_queue q,
                                  const float *src_fp32,
                                  unsigned int hidden, void **dst_svm,
                                  const char *tag) {
  const size_t bytes = (size_t)hidden * sizeof(float);
  *dst_svm = clSVMAlloc(cl_ctx, CL_MEM_READ_ONLY, bytes, 0);
  if (*dst_svm == nullptr) {
    std::fprintf(stderr, "[qwen3-gpu] %s SVMAlloc failed\n", tag);
    return false;
  }
  if (clEnqueueSVMMap(q, CL_TRUE, CL_MAP_WRITE, *dst_svm, bytes, 0,
                      nullptr, nullptr) != CL_SUCCESS)
    return false;
  std::memcpy(*dst_svm, src_fp32, bytes);
  clEnqueueSVMUnmap(q, *dst_svm, 0, nullptr, nullptr);
  clFinish(q);
  return true;
}

bool Qwen3Forward::load_layer(unsigned int layer_id, size_t *offset_inout,
                              unsigned int max_seq_len_used) {
  if (weight_mmap_ == nullptr || cl_ctx_ == nullptr) return false;
  if (layers_.size() <= layer_id) layers_.resize(layer_id + 1);
  LayerWeights &lw = layers_[layer_id];
  if (lw.wq.backing != nullptr) return true; // already loaded

  const unsigned int K_h = cfg_.hidden_size;
  const unsigned int N_q = cfg_.num_heads_Q * cfg_.head_dim;
  const unsigned int N_kv = cfg_.num_heads_KV * cfg_.head_dim;
  const unsigned int I = cfg_.intermediate_size;
  const size_t wq_bytes =
    sizeof(uint16_t) + (size_t)K_h * N_q / 2 + (size_t)N_q * 2;
  const size_t wk_bytes =
    sizeof(uint16_t) + (size_t)K_h * N_kv / 2 + (size_t)N_kv * 2;
  const size_t wv_bytes = wk_bytes;
  const size_t wo_bytes =
    sizeof(uint16_t) + (size_t)N_q * K_h / 2 + (size_t)K_h * 2;
  const size_t up_bytes =
    sizeof(uint16_t) + (size_t)K_h * I / 2 + (size_t)I * 2;
  const size_t gate_bytes = up_bytes;
  const size_t down_bytes =
    sizeof(uint16_t) + (size_t)I * K_h / 2 + (size_t)K_h * 2;
  const size_t norm_bytes = (size_t)K_h * sizeof(float);
  const size_t head_norm_bytes = (size_t)cfg_.head_dim * sizeof(float);

  size_t off = *offset_inout;

  // attn_norm gamma -> SVM fp32
  if (!load_norm_to_svm_fp32(
        cl_ctx_, cl_q_,
        reinterpret_cast<const float *>(weight_mmap_ + off), K_h,
        &lw.attn_norm_gamma_svm, "attn_norm")) return false;
  off += norm_bytes;

  // wq -> v8c backing
  if (!load_qint4_weight_at(off, K_h, N_q, &lw.wq, "wq")) return false;
  off += wq_bytes;

  // q_norm gamma -> SVM fp16
  if (!load_qk_norm_to_svm_fp16(
        cl_ctx_, cl_q_,
        reinterpret_cast<const float *>(weight_mmap_ + off),
        cfg_.head_dim, &lw.q_norm_gamma_svm_fp16, "q_norm")) return false;
  off += head_norm_bytes;

  // wk -> v8c backing
  if (!load_qint4_weight_at(off, K_h, N_kv, &lw.wk, "wk")) return false;
  off += wk_bytes;

  // k_norm gamma -> SVM fp16
  if (!load_qk_norm_to_svm_fp16(
        cl_ctx_, cl_q_,
        reinterpret_cast<const float *>(weight_mmap_ + off),
        cfg_.head_dim, &lw.k_norm_gamma_svm_fp16, "k_norm")) return false;
  off += head_norm_bytes;

  // wv -> v8c backing
  if (!load_qint4_weight_at(off, K_h, N_kv, &lw.wv, "wv")) return false;
  off += wv_bytes;

  // wo -> v8c backing
  if (!load_qint4_weight_at(off, N_q, K_h, &lw.wo, "wo")) return false;
  off += wo_bytes;

  // ffn_norm gamma -> SVM fp32
  if (!load_norm_to_svm_fp32(
        cl_ctx_, cl_q_,
        reinterpret_cast<const float *>(weight_mmap_ + off), K_h,
        &lw.ffn_norm_gamma_svm, "ffn_norm")) return false;
  off += norm_bytes;

  // ffn_up, ffn_gate, ffn_down -> v8c backings
  if (!load_qint4_weight_at(off, K_h, I, &lw.ffn_up, "ffn_up"))
    return false;
  off += up_bytes;
  if (!load_qint4_weight_at(off, K_h, I, &lw.ffn_gate, "ffn_gate"))
    return false;
  off += gate_bytes;
  if (!load_qint4_weight_at(off, I, K_h, &lw.ffn_down, "ffn_down"))
    return false;
  off += down_bytes;

  // K, V cache SVM, sized for max_seq_len_used.
  const size_t cache_bytes =
    (size_t)max_seq_len_used * cfg_.num_heads_KV * cfg_.head_dim *
    sizeof(uint16_t);
  lw.cache_k_svm = clSVMAlloc(cl_ctx_, CL_MEM_READ_WRITE, cache_bytes, 0);
  lw.cache_v_svm = clSVMAlloc(cl_ctx_, CL_MEM_READ_WRITE, cache_bytes, 0);
  if (lw.cache_k_svm == nullptr || lw.cache_v_svm == nullptr) {
    std::fprintf(stderr, "[qwen3-gpu] layer %u: KV cache SVMAlloc failed\n",
                 layer_id);
    return false;
  }
  // Zero-init.
  for (void *p : {lw.cache_k_svm, lw.cache_v_svm}) {
    cl_int err = clEnqueueSVMMap(cl_q_, CL_TRUE, CL_MAP_WRITE, p,
                                 cache_bytes, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) return false;
    std::memset(p, 0, cache_bytes);
    clEnqueueSVMUnmap(cl_q_, p, 0, nullptr, nullptr);
  }
  clFinish(cl_q_);

  *offset_inout = off;
  std::fprintf(stderr,
               "[qwen3-gpu] layer %u loaded; advanced offset to %zu "
               "(~%zu MB)\n",
               layer_id, off, off / (1024 * 1024));
  return true;
}

cl_mem Qwen3Forward::forward_one_layer(unsigned int layer_id, cl_mem in_fp32,
                                       unsigned int position) {
  if (layer_id >= layers_.size() || layers_[layer_id].wq.backing == nullptr) {
    std::fprintf(stderr,
                 "[qwen3-gpu] forward_one_layer(%u): not loaded\n", layer_id);
    return nullptr;
  }
  LayerWeights &lw = layers_[layer_id];
  auto *cl = static_cast<nntrainer::ClContext *>(
    nntrainer::Engine::Global().getRegisteredContext("gpu"));
  cl_int err = CL_SUCCESS;

  const unsigned int M_pad = 4;
  const unsigned int K_h = cfg_.hidden_size;
  const unsigned int N_q = cfg_.num_heads_Q * cfg_.head_dim;
  const unsigned int N_kv = cfg_.num_heads_KV * cfg_.head_dim;
  const unsigned int I = cfg_.intermediate_size;

  // (a) Pad in_fp32 [K_h] into [M_pad, K_h] for the v8c tile.
  cl_mem in_padded = clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                                    (size_t)M_pad * K_h * sizeof(float),
                                    nullptr, &err);
  float zero = 0.0f;
  clEnqueueFillBuffer(cl_q_, in_padded, &zero, sizeof(float), 0,
                      (size_t)M_pad * K_h * sizeof(float), 0, nullptr,
                      nullptr);
  clEnqueueCopyBuffer(cl_q_, in_fp32, in_padded, 0, 0,
                      (size_t)K_h * sizeof(float), 0, nullptr, nullptr);
  cl_mem attn_normed =
    clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                   (size_t)M_pad * K_h * sizeof(float), nullptr, &err);
  // attn_norm
  {
    auto kp = cl->registerClKernel(nntrainer::rmsnorm_kernel, "rmsnorm_cl");
    float eps = cfg_.rms_norm_eps;
    int H = (int)M_pad, W = (int)K_h;
    kp->SetKernelArguments(0, &in_padded, sizeof(cl_mem));
    kp->SetKernelArguments(1, &attn_normed, sizeof(cl_mem));
    kp->SetKernelSVMArguments(2, lw.attn_norm_gamma_svm);
    kp->SetKernelArguments(3, &eps, sizeof(float));
    kp->SetKernelArguments(4, &H, sizeof(int));
    kp->SetKernelArguments(5, &W, sizeof(int));
    std::array<size_t, 1> gws = {(size_t)M_pad * 64};
    std::array<size_t, 1> lws = {64};
    cl->command_queue_inst_.enqueueKernel(kp->GetKernel(), 1, gws.data(),
                                          lws.data(), 0, nullptr, nullptr);
    clFinish(cl_q_);
  }

  // (b) Shared act quant for Q/K/V.
  cl_mem act_i8 = clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                                 (size_t)M_pad * K_h, nullptr, &err);
  cl_mem act_scale = clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                                    sizeof(float) * M_pad, nullptr, &err);
  cl_mem act_zp = clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                                 sizeof(int) * M_pad, nullptr, &err);
  cl_mem act_rs = clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                                 sizeof(int) * M_pad, nullptr, &err);
  nntrainer::quantize_act_v8c_fp32_cl(attn_normed, act_i8, act_scale,
                                      act_zp, act_rs, M_pad, K_h);
  cl_image_format afmt{CL_RGBA, CL_UNSIGNED_INT32};
  cl_image_desc adesc{};
  adesc.image_type = CL_MEM_OBJECT_IMAGE2D;
  adesc.image_width = K_h / 16;
  adesc.image_height = M_pad;
  adesc.image_row_pitch = K_h;
  adesc.buffer = act_i8;
  cl_mem act_image =
    clCreateImage(cl_ctx_, CL_MEM_READ_ONLY, &afmt, &adesc, nullptr, &err);

  cl_mem y_q = clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                              sizeof(uint16_t) * (size_t)M_pad * N_q,
                              nullptr, &err);
  cl_mem y_k = clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                              sizeof(uint16_t) * (size_t)M_pad * N_kv,
                              nullptr, &err);
  cl_mem y_v = clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                              sizeof(uint16_t) * (size_t)M_pad * N_kv,
                              nullptr, &err);
  nntrainer::gemm_int8_v8c_cl(act_image, lw.wq.weight_image, act_scale,
                              lw.wq.scale_buf, act_rs, act_zp,
                              lw.wq.row_sum_w_int4, y_q, M_pad, N_q, K_h);
  nntrainer::gemm_int8_v8c_cl(act_image, lw.wk.weight_image, act_scale,
                              lw.wk.scale_buf, act_rs, act_zp,
                              lw.wk.row_sum_w_int4, y_k, M_pad, N_kv, K_h);
  nntrainer::gemm_int8_v8c_cl(act_image, lw.wv.weight_image, act_scale,
                              lw.wv.scale_buf, act_rs, act_zp,
                              lw.wv.row_sum_w_int4, y_v, M_pad, N_kv, K_h);
  clFinish(cl_q_);

  // (c) q_norm / k_norm in place, then RoPE in place (cos/sin tables
  //     are shared across layers — already precomputed for this position).
  auto disp_qk_norm = [&](cl_mem io, void *gamma, unsigned int num_heads) {
    auto kp = cl->registerClKernel(nntrainer::rmsnorm_fp16_kernel,
                                   "rmsnorm_cl_fp16");
    uint16_t eps_h = f2h(cfg_.rms_norm_eps);
    int B = 1, C = 1, H = (int)num_heads, W = (int)cfg_.head_dim;
    kp->SetKernelArguments(0, &io, sizeof(cl_mem));
    kp->SetKernelArguments(1, &io, sizeof(cl_mem));
    kp->SetKernelSVMArguments(2, gamma);
    kp->SetKernelArguments(3, &eps_h, sizeof(uint16_t));
    kp->SetKernelArguments(4, &B, sizeof(int));
    kp->SetKernelArguments(5, &C, sizeof(int));
    kp->SetKernelArguments(6, &H, sizeof(int));
    kp->SetKernelArguments(7, &W, sizeof(int));
    std::array<size_t, 2> gws = {1, (size_t)num_heads};
    std::array<size_t, 2> lws = {1, 1};
    cl->command_queue_inst_.enqueueKernel(kp->GetKernel(), 2, gws.data(),
                                          lws.data(), 0, nullptr, nullptr);
  };
  disp_qk_norm(y_q, lw.q_norm_gamma_svm_fp16, cfg_.num_heads_Q);
  disp_qk_norm(y_k, lw.k_norm_gamma_svm_fp16, cfg_.num_heads_KV);
  clFinish(cl_q_);

  if (layer0_rope_position_ != (int)position) {
    if (!precompute_rope_for_position(position)) return nullptr;
  }
  run_layer0_rope_on_qk(y_q, y_k);

  // (d) Write K, V into this layer's SVM cache at `position`.
  const size_t kv_row_bytes = (size_t)N_kv * sizeof(uint16_t);
  auto copy_cl_to_svm = [&](cl_mem src, void *dst_svm_base,
                            size_t offset_bytes) {
    cl_int e;
    void *p = clEnqueueMapBuffer(cl_q_, src, CL_TRUE, CL_MAP_READ, 0,
                                 kv_row_bytes, 0, nullptr, nullptr, &e);
    if (!p) return false;
    void *dst = static_cast<uint8_t *>(dst_svm_base) + offset_bytes;
    if (clEnqueueSVMMap(cl_q_, CL_TRUE, CL_MAP_WRITE, dst, kv_row_bytes, 0,
                        nullptr, nullptr) != CL_SUCCESS)
      return false;
    std::memcpy(dst, p, kv_row_bytes);
    clEnqueueSVMUnmap(cl_q_, dst, 0, nullptr, nullptr);
    clEnqueueUnmapMemObject(cl_q_, src, p, 0, nullptr, nullptr);
    return true;
  };
  copy_cl_to_svm(y_k, lw.cache_k_svm, (size_t)position * kv_row_bytes);
  copy_cl_to_svm(y_v, lw.cache_v_svm, (size_t)position * kv_row_bytes);
  clFinish(cl_q_);

  // (e) Attention dispatch via SVM.
  const size_t q_row_bytes = (size_t)N_q * sizeof(uint16_t);
  void *q_svm = clSVMAlloc(cl_ctx_, CL_MEM_READ_ONLY, q_row_bytes, 0);
  void *o_svm = clSVMAlloc(cl_ctx_, CL_MEM_READ_WRITE, q_row_bytes, 0);
  {
    cl_int e;
    void *p = clEnqueueMapBuffer(cl_q_, y_q, CL_TRUE, CL_MAP_READ, 0,
                                 q_row_bytes, 0, nullptr, nullptr, &e);
    clEnqueueSVMMap(cl_q_, CL_TRUE, CL_MAP_WRITE, q_svm, q_row_bytes, 0,
                    nullptr, nullptr);
    std::memcpy(q_svm, p, q_row_bytes);
    clEnqueueSVMUnmap(cl_q_, q_svm, 0, nullptr, nullptr);
    clEnqueueUnmapMemObject(cl_q_, y_q, p, 0, nullptr, nullptr);
    clFinish(cl_q_);
  }
  bool attn_ok = nntrainer::two_conv_attention_prefill_f16_cl(
    static_cast<const uint16_t *>(q_svm),
    static_cast<const uint16_t *>(lw.cache_k_svm),
    static_cast<const uint16_t *>(lw.cache_v_svm),
    static_cast<uint16_t *>(o_svm), 1, position + 1, cfg_.num_heads_Q,
    cfg_.num_heads_KV, cfg_.head_dim, true, true);
  if (!attn_ok) {
    std::fprintf(stderr,
                 "[qwen3-gpu] layer %u attention failed\n", layer_id);
  }

  // (f) wo: O_svm fp16 -> wo_out fp16 -> residual_1 fp32 = in + wo_out.
  cl_mem o_fp32 = clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                                 (size_t)M_pad * N_q * sizeof(float),
                                 nullptr, &err);
  clEnqueueFillBuffer(cl_q_, o_fp32, &zero, sizeof(float), 0,
                      (size_t)M_pad * N_q * sizeof(float), 0, nullptr,
                      nullptr);
  {
    auto kp = cl->registerClKernel(kConvertFp16ToFp32Kernel, "cvt_h2f");
    int n = (int)N_q;
    kp->SetKernelSVMArguments(0, o_svm);
    kp->SetKernelArguments(1, &o_fp32, sizeof(cl_mem));
    kp->SetKernelArguments(2, &n, sizeof(int));
    std::array<size_t, 1> gws = {((size_t)N_q + 63) / 64 * 64};
    std::array<size_t, 1> lws = {64};
    cl->command_queue_inst_.enqueueKernel(kp->GetKernel(), 1, gws.data(),
                                          lws.data(), 0, nullptr, nullptr);
    clFinish(cl_q_);
  }
  cl_mem wo_act_i8 = clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                                    (size_t)M_pad * N_q, nullptr, &err);
  cl_mem wo_act_scale = clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                                       sizeof(float) * M_pad, nullptr, &err);
  cl_mem wo_act_zp = clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                                    sizeof(int) * M_pad, nullptr, &err);
  cl_mem wo_act_rs = clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                                    sizeof(int) * M_pad, nullptr, &err);
  cl_mem wo_y_fp16 = clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                                    sizeof(uint16_t) * (size_t)M_pad * K_h,
                                    nullptr, &err);
  nntrainer::quantize_act_v8c_fp32_cl(o_fp32, wo_act_i8, wo_act_scale,
                                      wo_act_zp, wo_act_rs, M_pad, N_q);
  cl_image_desc wo_adesc{};
  wo_adesc.image_type = CL_MEM_OBJECT_IMAGE2D;
  wo_adesc.image_width = N_q / 16;
  wo_adesc.image_height = M_pad;
  wo_adesc.image_row_pitch = N_q;
  wo_adesc.buffer = wo_act_i8;
  cl_mem wo_act_image =
    clCreateImage(cl_ctx_, CL_MEM_READ_ONLY, &afmt, &wo_adesc, nullptr, &err);
  nntrainer::gemm_int8_v8c_cl(wo_act_image, lw.wo.weight_image, wo_act_scale,
                              lw.wo.scale_buf, wo_act_rs, wo_act_zp,
                              lw.wo.row_sum_w_int4, wo_y_fp16, M_pad, K_h, N_q);
  clFinish(cl_q_);
  cl_mem wo_fp32 = clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                                  (size_t)K_h * sizeof(float), nullptr, &err);
  {
    auto kp = cl->registerClKernel(kConvertFp16ToFp32Kernel, "cvt_h2f");
    int n = (int)K_h;
    kp->SetKernelArguments(0, &wo_y_fp16, sizeof(cl_mem));
    kp->SetKernelArguments(1, &wo_fp32, sizeof(cl_mem));
    kp->SetKernelArguments(2, &n, sizeof(int));
    std::array<size_t, 1> gws = {((size_t)K_h + 63) / 64 * 64};
    std::array<size_t, 1> lws = {64};
    cl->command_queue_inst_.enqueueKernel(kp->GetKernel(), 1, gws.data(),
                                          lws.data(), 0, nullptr, nullptr);
    clFinish(cl_q_);
  }
  cl_mem residual_1 = clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                                     (size_t)K_h * sizeof(float), nullptr,
                                     &err);
  {
    auto kp = cl->registerClKernel(kAddFp32Kernel, "add_fp32");
    int n = (int)K_h;
    kp->SetKernelArguments(0, &in_fp32, sizeof(cl_mem));
    kp->SetKernelArguments(1, &wo_fp32, sizeof(cl_mem));
    kp->SetKernelArguments(2, &residual_1, sizeof(cl_mem));
    kp->SetKernelArguments(3, &n, sizeof(int));
    std::array<size_t, 1> gws = {((size_t)K_h + 63) / 64 * 64};
    std::array<size_t, 1> lws = {64};
    cl->command_queue_inst_.enqueueKernel(kp->GetKernel(), 1, gws.data(),
                                          lws.data(), 0, nullptr, nullptr);
    clFinish(cl_q_);
  }

  // (g) ffn_norm + ffn_up/gate (shared quant) + swiglu + ffn_down +
  //     residual_2.
  cl_mem ffn_in_padded =
    clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                   (size_t)M_pad * K_h * sizeof(float), nullptr, &err);
  clEnqueueFillBuffer(cl_q_, ffn_in_padded, &zero, sizeof(float), 0,
                      (size_t)M_pad * K_h * sizeof(float), 0, nullptr,
                      nullptr);
  clEnqueueCopyBuffer(cl_q_, residual_1, ffn_in_padded, 0, 0,
                      (size_t)K_h * sizeof(float), 0, nullptr, nullptr);
  cl_mem ffn_normed = clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                                     (size_t)M_pad * K_h * sizeof(float),
                                     nullptr, &err);
  {
    auto kp = cl->registerClKernel(nntrainer::rmsnorm_kernel, "rmsnorm_cl");
    float eps = cfg_.rms_norm_eps;
    int H = (int)M_pad, W = (int)K_h;
    kp->SetKernelArguments(0, &ffn_in_padded, sizeof(cl_mem));
    kp->SetKernelArguments(1, &ffn_normed, sizeof(cl_mem));
    kp->SetKernelSVMArguments(2, lw.ffn_norm_gamma_svm);
    kp->SetKernelArguments(3, &eps, sizeof(float));
    kp->SetKernelArguments(4, &H, sizeof(int));
    kp->SetKernelArguments(5, &W, sizeof(int));
    std::array<size_t, 1> gws = {(size_t)M_pad * 64};
    std::array<size_t, 1> lws = {64};
    cl->command_queue_inst_.enqueueKernel(kp->GetKernel(), 1, gws.data(),
                                          lws.data(), 0, nullptr, nullptr);
    clFinish(cl_q_);
  }
  cl_mem fa_i8 = clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                                (size_t)M_pad * K_h, nullptr, &err);
  cl_mem fa_sc = clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                                sizeof(float) * M_pad, nullptr, &err);
  cl_mem fa_zp = clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                                sizeof(int) * M_pad, nullptr, &err);
  cl_mem fa_rs = clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                                sizeof(int) * M_pad, nullptr, &err);
  nntrainer::quantize_act_v8c_fp32_cl(ffn_normed, fa_i8, fa_sc, fa_zp, fa_rs,
                                      M_pad, K_h);
  cl_image_desc fa_adesc{};
  fa_adesc.image_type = CL_MEM_OBJECT_IMAGE2D;
  fa_adesc.image_width = K_h / 16;
  fa_adesc.image_height = M_pad;
  fa_adesc.image_row_pitch = K_h;
  fa_adesc.buffer = fa_i8;
  cl_mem fa_image =
    clCreateImage(cl_ctx_, CL_MEM_READ_ONLY, &afmt, &fa_adesc, nullptr, &err);
  cl_mem up_fp16 = clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                                  sizeof(uint16_t) * (size_t)M_pad * I,
                                  nullptr, &err);
  cl_mem gate_fp16 = clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                                    sizeof(uint16_t) * (size_t)M_pad * I,
                                    nullptr, &err);
  nntrainer::gemm_int8_v8c_cl(fa_image, lw.ffn_up.weight_image, fa_sc,
                              lw.ffn_up.scale_buf, fa_rs, fa_zp,
                              lw.ffn_up.row_sum_w_int4, up_fp16, M_pad, I,
                              K_h);
  nntrainer::gemm_int8_v8c_cl(fa_image, lw.ffn_gate.weight_image, fa_sc,
                              lw.ffn_gate.scale_buf, fa_rs, fa_zp,
                              lw.ffn_gate.row_sum_w_int4, gate_fp16, M_pad, I,
                              K_h);
  clFinish(cl_q_);
  cl_mem up_fp32 = clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                                  (size_t)I * sizeof(float), nullptr, &err);
  cl_mem gate_fp32 = clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                                    (size_t)I * sizeof(float), nullptr, &err);
  auto disp_cvt = [&](cl_mem hin, cl_mem fout, unsigned int n) {
    auto kp = cl->registerClKernel(kConvertFp16ToFp32Kernel, "cvt_h2f");
    int ni = (int)n;
    kp->SetKernelArguments(0, &hin, sizeof(cl_mem));
    kp->SetKernelArguments(1, &fout, sizeof(cl_mem));
    kp->SetKernelArguments(2, &ni, sizeof(int));
    std::array<size_t, 1> gws = {((size_t)n + 63) / 64 * 64};
    std::array<size_t, 1> lws = {64};
    cl->command_queue_inst_.enqueueKernel(kp->GetKernel(), 1, gws.data(),
                                          lws.data(), 0, nullptr, nullptr);
  };
  disp_cvt(up_fp16, up_fp32, I);
  disp_cvt(gate_fp16, gate_fp32, I);
  clFinish(cl_q_);
  cl_mem swiglu_out = clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                                     (size_t)M_pad * I * sizeof(float),
                                     nullptr, &err);
  clEnqueueFillBuffer(cl_q_, swiglu_out, &zero, sizeof(float), 0,
                      (size_t)M_pad * I * sizeof(float), 0, nullptr, nullptr);
  {
    auto kp = cl->registerClKernel(kSwigluFp32Kernel, "swiglu_fp32");
    int n = (int)I;
    kp->SetKernelArguments(0, &gate_fp32, sizeof(cl_mem));
    kp->SetKernelArguments(1, &up_fp32, sizeof(cl_mem));
    kp->SetKernelArguments(2, &swiglu_out, sizeof(cl_mem));
    kp->SetKernelArguments(3, &n, sizeof(int));
    std::array<size_t, 1> gws = {((size_t)I + 63) / 64 * 64};
    std::array<size_t, 1> lws = {64};
    cl->command_queue_inst_.enqueueKernel(kp->GetKernel(), 1, gws.data(),
                                          lws.data(), 0, nullptr, nullptr);
    clFinish(cl_q_);
  }
  cl_mem dn_i8 = clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                                (size_t)M_pad * I, nullptr, &err);
  cl_mem dn_sc = clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                                sizeof(float) * M_pad, nullptr, &err);
  cl_mem dn_zp = clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                                sizeof(int) * M_pad, nullptr, &err);
  cl_mem dn_rs = clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                                sizeof(int) * M_pad, nullptr, &err);
  nntrainer::quantize_act_v8c_fp32_cl(swiglu_out, dn_i8, dn_sc, dn_zp, dn_rs,
                                      M_pad, I);
  cl_image_desc dn_adesc{};
  dn_adesc.image_type = CL_MEM_OBJECT_IMAGE2D;
  dn_adesc.image_width = I / 16;
  dn_adesc.image_height = M_pad;
  dn_adesc.image_row_pitch = I;
  dn_adesc.buffer = dn_i8;
  cl_mem dn_image =
    clCreateImage(cl_ctx_, CL_MEM_READ_ONLY, &afmt, &dn_adesc, nullptr, &err);
  cl_mem dn_fp16 = clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                                  sizeof(uint16_t) * (size_t)M_pad * K_h,
                                  nullptr, &err);
  nntrainer::gemm_int8_v8c_cl(dn_image, lw.ffn_down.weight_image, dn_sc,
                              lw.ffn_down.scale_buf, dn_rs, dn_zp,
                              lw.ffn_down.row_sum_w_int4, dn_fp16, M_pad, K_h,
                              I);
  clFinish(cl_q_);
  cl_mem dn_fp32 = clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                                  (size_t)K_h * sizeof(float), nullptr, &err);
  disp_cvt(dn_fp16, dn_fp32, K_h);
  clFinish(cl_q_);
  cl_mem out_fp32 = clCreateBuffer(cl_ctx_, CL_MEM_READ_WRITE,
                                   (size_t)K_h * sizeof(float), nullptr,
                                   &err);
  {
    auto kp = cl->registerClKernel(kAddFp32Kernel, "add_fp32");
    int n = (int)K_h;
    kp->SetKernelArguments(0, &residual_1, sizeof(cl_mem));
    kp->SetKernelArguments(1, &dn_fp32, sizeof(cl_mem));
    kp->SetKernelArguments(2, &out_fp32, sizeof(cl_mem));
    kp->SetKernelArguments(3, &n, sizeof(int));
    std::array<size_t, 1> gws = {((size_t)K_h + 63) / 64 * 64};
    std::array<size_t, 1> lws = {64};
    cl->command_queue_inst_.enqueueKernel(kp->GetKernel(), 1, gws.data(),
                                          lws.data(), 0, nullptr, nullptr);
    clFinish(cl_q_);
  }

  // Cleanup all the per-call scratch.
  clReleaseMemObject(dn_fp32);
  clReleaseMemObject(dn_fp16);
  clReleaseMemObject(dn_image);
  clReleaseMemObject(dn_rs);
  clReleaseMemObject(dn_zp);
  clReleaseMemObject(dn_sc);
  clReleaseMemObject(dn_i8);
  clReleaseMemObject(swiglu_out);
  clReleaseMemObject(gate_fp32);
  clReleaseMemObject(up_fp32);
  clReleaseMemObject(gate_fp16);
  clReleaseMemObject(up_fp16);
  clReleaseMemObject(fa_image);
  clReleaseMemObject(fa_rs);
  clReleaseMemObject(fa_zp);
  clReleaseMemObject(fa_sc);
  clReleaseMemObject(fa_i8);
  clReleaseMemObject(ffn_normed);
  clReleaseMemObject(ffn_in_padded);
  clReleaseMemObject(residual_1);
  clReleaseMemObject(wo_fp32);
  clReleaseMemObject(wo_y_fp16);
  clReleaseMemObject(wo_act_image);
  clReleaseMemObject(wo_act_rs);
  clReleaseMemObject(wo_act_zp);
  clReleaseMemObject(wo_act_scale);
  clReleaseMemObject(wo_act_i8);
  clReleaseMemObject(o_fp32);
  if (q_svm) clSVMFree(cl_ctx_, q_svm);
  if (o_svm) clSVMFree(cl_ctx_, o_svm);
  clReleaseMemObject(y_v);
  clReleaseMemObject(y_k);
  clReleaseMemObject(y_q);
  clReleaseMemObject(act_image);
  clReleaseMemObject(act_rs);
  clReleaseMemObject(act_zp);
  clReleaseMemObject(act_scale);
  clReleaseMemObject(act_i8);
  clReleaseMemObject(attn_normed);
  clReleaseMemObject(in_padded);

  return out_fp32; // caller owns
}

bool Qwen3Forward::allocate_layer0_kv_cache_svm() {
  if (layer0_cache_k_svm_ != nullptr && layer0_cache_v_svm_ != nullptr)
    return true;
  if (cl_ctx_ == nullptr) return false;
  // [max_seq_len * hKV * d] fp16 — concat layout (same as the
  // existing CausalLM KVCacheManager for the non-OHWI path).
  const size_t cache_bytes =
    (size_t)cfg_.max_seq_len * cfg_.num_heads_KV * cfg_.head_dim *
    sizeof(uint16_t);

  auto alloc_one = [this, cache_bytes](void **dst, const char *tag) -> bool {
    if (*dst != nullptr) return true;
    *dst = clSVMAlloc(cl_ctx_, CL_MEM_READ_WRITE, cache_bytes, 0);
    if (*dst == nullptr) {
      std::fprintf(stderr, "[qwen3-gpu] %s SVMAlloc(%zu B) failed\n", tag,
                   cache_bytes);
      return false;
    }
    // Zero-init via SVM map + memset (clEnqueueSVMMemFill is OpenCL 2.0
    // but unavailable through some Adreno OpenCL loaders; map+memset
    // works universally on coarse-grained SVM).
    cl_int err = clEnqueueSVMMap(cl_q_, CL_TRUE, CL_MAP_WRITE, *dst,
                                 cache_bytes, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
      std::fprintf(stderr, "[qwen3-gpu] %s SVMMap zero err=%d\n", tag, err);
      return false;
    }
    std::memset(*dst, 0, cache_bytes);
    err = clEnqueueSVMUnmap(cl_q_, *dst, 0, nullptr, nullptr);
    clFinish(cl_q_);
    return err == CL_SUCCESS;
  };
  if (!alloc_one(&layer0_cache_k_svm_, "cache_K") ||
      !alloc_one(&layer0_cache_v_svm_, "cache_V"))
    return false;

  std::fprintf(stderr,
               "[qwen3-gpu] KV cache SVM allocated: each %zu MB "
               "(max_seq=%u, hKV=%u, d=%u, fp16)\n",
               cache_bytes / (1024 * 1024), cfg_.max_seq_len,
               cfg_.num_heads_KV, cfg_.head_dim);
  return true;
}

} // namespace causallm_gpu
