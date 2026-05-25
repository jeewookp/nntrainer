// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Yash Singh <yash.singh@samsung.com>
 *
 * @file	attention_kernels.cpp
 * @date	28 August 2024
 * @brief	Common attention OpenCL kernels
 * @see		https://github.com/nntrainer/nntrainer
 * @author	Yash Singh <yash.singh@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#include "attention_kernels_templates.h"
#include <cl_kernels/flash_attention.h>
#include <cl_kernels/rotary_emb.h>
#include <array>
#include <cmath>
#include <mutex>

namespace nntrainer {

void rotary_emb_cl(float *in, float *out,
                   const std::vector<std::vector<float>> &freqs_cos,
                   const std::vector<std::vector<float>> &freqs_sin,
                   const std::vector<float> &cos_,
                   const std::vector<float> &sin_, unsigned int batch,
                   unsigned int channel, unsigned int height,
                   unsigned int width, unsigned int dim, unsigned int from,
                   unsigned int max_timestep, unsigned int in_size,
                   unsigned int out_size) {
  auto *cl_context =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));

  ClContext::SharedPtrClKernel kernel_rotaryEmb_ptr =
    cl_context->registerClKernel(rotary_emb_kernel, "rotary_emb_cl");
  if (!kernel_rotaryEmb_ptr) {
    return;
  }

  rotary_emb_cl_internal<float>(
    kernel_rotaryEmb_ptr, in, out, freqs_cos, freqs_sin, cos_, sin_, batch,
    channel, height, width, dim, from, max_timestep, in_size, out_size);
}

// =============================================================================
// GPU flash attention scratch + dispatch.
// =============================================================================
namespace {
struct FaScratch {
  cl_mem q_buf = nullptr;
  size_t q_bytes = 0;
  cl_mem kv_buf = nullptr; // single buffer for K+V (loaded sequentially)
  size_t kv_bytes = 0;
  cl_mem k_buf = nullptr;
  size_t k_bytes = 0;
  cl_mem v_buf = nullptr;
  size_t v_bytes = 0;
  cl_mem o_buf = nullptr;
  size_t o_bytes = 0;
};
inline FaScratch &fa_scratch() {
  static FaScratch s;
  return s;
}
inline std::mutex &fa_mtx() {
  static std::mutex m;
  return m;
}
static bool fa_ensure(cl_context ctx, cl_mem *buf, size_t *cap, size_t bytes,
                      cl_mem_flags flags) {
  if (*buf && *cap >= bytes) return true;
  if (*buf) {
    clReleaseMemObject(*buf);
    *buf = nullptr;
    *cap = 0;
  }
  cl_int err = CL_SUCCESS;
  *buf = clCreateBuffer(ctx, flags, bytes, nullptr, &err);
  if (err != CL_SUCCESS || !*buf) {
    *buf = nullptr;
    *cap = 0;
    return false;
  }
  *cap = bytes;
  return true;
}
} // namespace

bool flash_attention_prefill_f16_cl(const uint16_t *Q_host,
                                    const uint16_t *K_host,
                                    const uint16_t *V_host,
                                    uint16_t *O_host, unsigned int M,
                                    unsigned int N_kv,
                                    unsigned int num_heads_Q,
                                    unsigned int num_heads_KV,
                                    unsigned int head_dim, bool causal,
                                    unsigned int cache_from) {
  // Kernel is specialized at compile time on FA_HEAD_DIM=128 and FA_BQ=64.
  constexpr unsigned int FA_HEAD_DIM = 128;
  constexpr unsigned int FA_BQ = 64;
  if (head_dim != FA_HEAD_DIM) return false;
  if (num_heads_KV == 0 || num_heads_Q % num_heads_KV != 0) return false;
  if (M == 0 || N_kv == 0) return false;

  // Round M up to a multiple of FA_BQ so the tail q_block has valid lanes
  // (tail WIs guard themselves via q_in_range==0 inside the kernel).
  const unsigned int num_qb = (M + FA_BQ - 1) / FA_BQ;

  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  cl_context ctx = blas_cc->context_inst_.GetContext();
  cl_command_queue q = blas_cc->command_queue_inst_.GetCommandQueue();

  const size_t HD_Q = (size_t)num_heads_Q * FA_HEAD_DIM;
  const size_t HD_KV = (size_t)num_heads_KV * FA_HEAD_DIM;
  const size_t q_bytes = (size_t)M * HD_Q * sizeof(uint16_t);
  const size_t kv_each = (size_t)N_kv * HD_KV * sizeof(uint16_t);
  const size_t o_bytes = (size_t)M * HD_Q * sizeof(uint16_t);

  std::lock_guard<std::mutex> lock(fa_mtx());
  FaScratch &sc = fa_scratch();
  if (!fa_ensure(ctx, &sc.q_buf, &sc.q_bytes, q_bytes, CL_MEM_READ_ONLY) ||
      !fa_ensure(ctx, &sc.k_buf, &sc.k_bytes, kv_each, CL_MEM_READ_ONLY) ||
      !fa_ensure(ctx, &sc.v_buf, &sc.v_bytes, kv_each, CL_MEM_READ_ONLY) ||
      !fa_ensure(ctx, &sc.o_buf, &sc.o_bytes, o_bytes, CL_MEM_WRITE_ONLY))
    return false;

  if (clEnqueueWriteBuffer(q, sc.q_buf, CL_FALSE, 0, q_bytes, Q_host, 0,
                           nullptr, nullptr) != CL_SUCCESS ||
      clEnqueueWriteBuffer(q, sc.k_buf, CL_FALSE, 0, kv_each, K_host, 0,
                           nullptr, nullptr) != CL_SUCCESS ||
      clEnqueueWriteBuffer(q, sc.v_buf, CL_FALSE, 0, kv_each, V_host, 0,
                           nullptr, nullptr) != CL_SUCCESS)
    return false;

  ClContext::SharedPtrClKernel kp = blas_cc->registerClKernel(
    flash_attention_kernel, "flash_attention_prefill_f16");
  if (!kp) return false;

  int arg = 0;
  if (!kp->SetKernelArguments(arg++, &sc.q_buf, sizeof(cl_mem)) ||
      !kp->SetKernelArguments(arg++, &sc.k_buf, sizeof(cl_mem)) ||
      !kp->SetKernelArguments(arg++, &sc.v_buf, sizeof(cl_mem)) ||
      !kp->SetKernelArguments(arg++, &sc.o_buf, sizeof(cl_mem)))
    return false;
  int Mi = (int)M, Nkvi = (int)N_kv;
  int nhQ = (int)num_heads_Q, nhKV = (int)num_heads_KV;
  int causal_i = causal ? 1 : 0;
  int cache_from_i = (int)cache_from;
  float inv_sqrt_d = 1.0f / std::sqrt((float)FA_HEAD_DIM);
  if (!kp->SetKernelArguments(arg++, &Mi, sizeof(int)) ||
      !kp->SetKernelArguments(arg++, &Nkvi, sizeof(int)) ||
      !kp->SetKernelArguments(arg++, &nhQ, sizeof(int)) ||
      !kp->SetKernelArguments(arg++, &nhKV, sizeof(int)) ||
      !kp->SetKernelArguments(arg++, &causal_i, sizeof(int)) ||
      !kp->SetKernelArguments(arg++, &cache_from_i, sizeof(int)) ||
      !kp->SetKernelArguments(arg++, &inv_sqrt_d, sizeof(float)))
    return false;

  constexpr size_t FA_LWS = 64;
  std::array<size_t, 3> gws = {(size_t)num_heads_Q * num_qb * FA_LWS, 1, 1};
  std::array<size_t, 3> lws = {FA_LWS, 1, 1};
  blas_cc->command_queue_inst_.enqueueKernel(kp->GetKernel(), 1, gws.data(),
                                             lws.data(), 0, nullptr, nullptr);

  if (clEnqueueReadBuffer(q, sc.o_buf, CL_TRUE, 0, o_bytes, O_host, 0, nullptr,
                          nullptr) != CL_SUCCESS)
    return false;
  return true;
}

} // namespace nntrainer
