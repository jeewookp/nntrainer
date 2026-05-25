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
#include <array>
#include <cl_kernels/rotary_emb.h>
#include <cl_kernels/two_conv_attention.h>
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
// Two-1x1-conv attention (paper section 3.7).
// =============================================================================
namespace {
struct TcaScratch {
  // Q/K/V backing buffers used only on the non-SVM fallback.
  cl_mem q_buf = nullptr;
  size_t q_bytes = 0;
  cl_mem k_buf = nullptr;
  size_t k_bytes = 0;
  cl_mem v_buf = nullptr;
  size_t v_bytes = 0;
  cl_mem o_buf = nullptr;
  size_t o_bytes = 0;
  // Score matrix - always cl_mem, never SVM. Shape [H, M, N_kv] fp16.
  cl_mem scores = nullptr;
  size_t scores_bytes = 0;
};
inline TcaScratch &tca_scratch() {
  static TcaScratch s;
  return s;
}
inline std::mutex &tca_mtx() {
  static std::mutex m;
  return m;
}
static bool tca_ensure(cl_context ctx, cl_mem *buf, size_t *cap, size_t bytes,
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

bool two_conv_attention_prefill_f16_cl(const uint16_t *Q_host,
                                       const uint16_t *K_host,
                                       const uint16_t *V_host,
                                       uint16_t *O_host, unsigned int M,
                                       unsigned int N_kv,
                                       unsigned int num_heads_Q,
                                       unsigned int num_heads_KV,
                                       unsigned int head_dim, bool causal,
                                       bool svm_inputs) {
  if (head_dim == 0 || M == 0 || N_kv == 0) return false;
  if (num_heads_KV == 0 || num_heads_Q % num_heads_KV != 0) return false;
  // Match the kernel tile defaults; relaxing requires re-defining TM/TN.
  constexpr unsigned int TM_QK = 4, TN_QK = 8;
  constexpr unsigned int TM_SV = 4, TD_SV = 8;
  constexpr unsigned int SOFTMAX_LWS = 64;
  // d must be tile-aligned for the SV kernel; both M and N_kv get
  // tile-rounding by the kernel itself (tail-WI guards inside).
  if (head_dim % TD_SV != 0) return false;

  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  cl_context ctx = blas_cc->context_inst_.GetContext();
  cl_command_queue q = blas_cc->command_queue_inst_.GetCommandQueue();

  const size_t HD_Q = (size_t)num_heads_Q * head_dim;
  const size_t HD_KV = (size_t)num_heads_KV * head_dim;
  const size_t q_bytes = (size_t)M * HD_Q * sizeof(uint16_t);
  const size_t k_bytes = (size_t)N_kv * HD_KV * sizeof(uint16_t);
  const size_t v_bytes = k_bytes;
  const size_t o_bytes = (size_t)M * HD_Q * sizeof(uint16_t);
  const size_t scores_bytes =
    (size_t)num_heads_Q * M * N_kv * sizeof(uint16_t);

  std::lock_guard<std::mutex> lock(tca_mtx());
  TcaScratch &sc = tca_scratch();
  if (!tca_ensure(ctx, &sc.scores, &sc.scores_bytes, scores_bytes,
                  CL_MEM_READ_WRITE))
    return false;

  cl_mem q_arg = nullptr, k_arg = nullptr, v_arg = nullptr, o_arg = nullptr;
  if (!svm_inputs) {
    if (!tca_ensure(ctx, &sc.q_buf, &sc.q_bytes, q_bytes, CL_MEM_READ_ONLY) ||
        !tca_ensure(ctx, &sc.k_buf, &sc.k_bytes, k_bytes, CL_MEM_READ_ONLY) ||
        !tca_ensure(ctx, &sc.v_buf, &sc.v_bytes, v_bytes, CL_MEM_READ_ONLY) ||
        !tca_ensure(ctx, &sc.o_buf, &sc.o_bytes, o_bytes, CL_MEM_WRITE_ONLY))
      return false;
    if (clEnqueueWriteBuffer(q, sc.q_buf, CL_FALSE, 0, q_bytes, Q_host, 0,
                             nullptr, nullptr) != CL_SUCCESS ||
        clEnqueueWriteBuffer(q, sc.k_buf, CL_FALSE, 0, k_bytes, K_host, 0,
                             nullptr, nullptr) != CL_SUCCESS ||
        clEnqueueWriteBuffer(q, sc.v_buf, CL_FALSE, 0, v_bytes, V_host, 0,
                             nullptr, nullptr) != CL_SUCCESS)
      return false;
    q_arg = sc.q_buf;
    k_arg = sc.k_buf;
    v_arg = sc.v_buf;
    o_arg = sc.o_buf;
  }

  // ---- K1: QK matmul ----
  {
    ClContext::SharedPtrClKernel kp =
      blas_cc->registerClKernel(two_conv_attention_kernel, "qk_matmul_f16");
    if (!kp) return false;
    if (svm_inputs) {
      if (!kp->SetKernelSVMArguments(0, const_cast<uint16_t *>(Q_host)) ||
          !kp->SetKernelSVMArguments(1, const_cast<uint16_t *>(K_host)))
        return false;
    } else {
      if (!kp->SetKernelArguments(0, &q_arg, sizeof(cl_mem)) ||
          !kp->SetKernelArguments(1, &k_arg, sizeof(cl_mem)))
        return false;
    }
    if (!kp->SetKernelArguments(2, &sc.scores, sizeof(cl_mem))) return false;
    int Mi = (int)M, Nkvi = (int)N_kv, di = (int)head_dim;
    int hdq = (int)HD_Q, hdkv = (int)HD_KV;
    int gqa = (int)(num_heads_Q / num_heads_KV);
    int causal_i = causal ? 1 : 0;
    float scale = 1.0f / std::sqrt((float)head_dim);
    if (!kp->SetKernelArguments(3, &Mi, sizeof(int)) ||
        !kp->SetKernelArguments(4, &Nkvi, sizeof(int)) ||
        !kp->SetKernelArguments(5, &di, sizeof(int)) ||
        !kp->SetKernelArguments(6, &hdq, sizeof(int)) ||
        !kp->SetKernelArguments(7, &hdkv, sizeof(int)) ||
        !kp->SetKernelArguments(8, &gqa, sizeof(int)) ||
        !kp->SetKernelArguments(9, &causal_i, sizeof(int)) ||
        !kp->SetKernelArguments(10, &scale, sizeof(float)))
      return false;
    const size_t nx = (N_kv + TN_QK - 1) / TN_QK;
    const size_t mx = (M + TM_QK - 1) / TM_QK;
    std::array<size_t, 3> gws = {nx, mx, num_heads_Q};
    blas_cc->command_queue_inst_.enqueueKernel(kp->GetKernel(), 3, gws.data(),
                                               nullptr, 0, nullptr, nullptr);
  }

  // ---- K2: row softmax over N_kv ----
  {
    ClContext::SharedPtrClKernel kp =
      blas_cc->registerClKernel(two_conv_attention_kernel, "softmax_row_f16");
    if (!kp) return false;
    if (!kp->SetKernelArguments(0, &sc.scores, sizeof(cl_mem))) return false;
    int Mi = (int)M, Nkvi = (int)N_kv;
    if (!kp->SetKernelArguments(1, &Mi, sizeof(int)) ||
        !kp->SetKernelArguments(2, &Nkvi, sizeof(int)))
      return false;
    std::array<size_t, 3> gws = {SOFTMAX_LWS, M, num_heads_Q};
    std::array<size_t, 3> lws = {SOFTMAX_LWS, 1, 1};
    blas_cc->command_queue_inst_.enqueueKernel(kp->GetKernel(), 3, gws.data(),
                                               lws.data(), 0, nullptr, nullptr);
  }

  // ---- K3: scores @ V -> O ----
  {
    ClContext::SharedPtrClKernel kp =
      blas_cc->registerClKernel(two_conv_attention_kernel, "sv_matmul_f16");
    if (!kp) return false;
    if (!kp->SetKernelArguments(0, &sc.scores, sizeof(cl_mem))) return false;
    if (svm_inputs) {
      if (!kp->SetKernelSVMArguments(1, const_cast<uint16_t *>(V_host)) ||
          !kp->SetKernelSVMArguments(2, O_host))
        return false;
    } else {
      if (!kp->SetKernelArguments(1, &v_arg, sizeof(cl_mem)) ||
          !kp->SetKernelArguments(2, &o_arg, sizeof(cl_mem)))
        return false;
    }
    int Mi = (int)M, Nkvi = (int)N_kv, di = (int)head_dim;
    int hdq = (int)HD_Q, hdkv = (int)HD_KV;
    int gqa = (int)(num_heads_Q / num_heads_KV);
    if (!kp->SetKernelArguments(3, &Mi, sizeof(int)) ||
        !kp->SetKernelArguments(4, &Nkvi, sizeof(int)) ||
        !kp->SetKernelArguments(5, &di, sizeof(int)) ||
        !kp->SetKernelArguments(6, &hdq, sizeof(int)) ||
        !kp->SetKernelArguments(7, &hdkv, sizeof(int)) ||
        !kp->SetKernelArguments(8, &gqa, sizeof(int)))
      return false;
    const size_t dx = (head_dim + TD_SV - 1) / TD_SV;
    const size_t mx = (M + TM_SV - 1) / TM_SV;
    std::array<size_t, 3> gws = {dx, mx, num_heads_Q};
    blas_cc->command_queue_inst_.enqueueKernel(kp->GetKernel(), 3, gws.data(),
                                               nullptr, 0, nullptr, nullptr);
  }

  if (svm_inputs) {
    clFinish(q);
  } else {
    if (clEnqueueReadBuffer(q, sc.o_buf, CL_TRUE, 0, o_bytes, O_host, 0,
                            nullptr, nullptr) != CL_SUCCESS)
      return false;
  }
  return true;
}

} // namespace nntrainer
