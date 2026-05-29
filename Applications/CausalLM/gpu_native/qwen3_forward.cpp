// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   qwen3_forward.cpp
 * @date   29 May 2026
 * @brief  Paper-aligned GPU-native Qwen3 forward (skeleton commit).
 */

#include "qwen3_forward.h"

#include <cl_context.h>
#include <engine.h>
#include <rmsnorm.h>

#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace causallm_gpu {

namespace {
// Q6_K block layout: 256 elements per block, 210 bytes per block
// (see nntrainer/tensor/q6_k_tensor.h:32 — Q6_K_SIZE = 210).
constexpr size_t Q6_K_BLOCK_ELTS = 256;
constexpr size_t Q6_K_BLOCK_BYTES = 210;
} // namespace

Qwen3Forward::Qwen3Forward() = default;

Qwen3Forward::~Qwen3Forward() {
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

} // namespace causallm_gpu
