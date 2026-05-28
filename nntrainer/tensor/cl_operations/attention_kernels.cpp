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
#include <chrono>
#include <cl_kernels/rotary_emb.h>
#include <cl_kernels/two_conv_attention.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
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
  // int8-KV variant: separate scale buffers; K/V byte buffers reuse k_buf/v_buf
  // (size halved relative to the fp16 path).
  cl_mem k_scale_buf = nullptr;
  size_t k_scale_bytes = 0;
  cl_mem v_scale_buf = nullptr;
  size_t v_scale_bytes = 0;
  // Image2d_from_buffer cache (image variant). Views over q_buf/k_buf/v_buf,
  // valid as long as shape and underlying buffer don't change. Recreate when
  // (M, N_kv, HD_Q, HD_KV) shift.
  cl_mem q_image = nullptr;
  cl_mem k_image = nullptr;
  cl_mem v_image = nullptr;
  unsigned int img_M = 0, img_N_kv = 0;
  unsigned int img_HD_Q = 0, img_HD_KV = 0;
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
// =============================================================================
// Per-kernel timing for the buffer-path two_conv attention. Env-gated
// NNTR_MHA_PROFILE=1. Reports K1 (QK matmul) / K2 (softmax) / K3 (SV matmul)
// wall-clock split + read_output time. Adds a clFinish between each kernel
// when enabled — slows the chain (sync overhead) but isolates per-step time.
// =============================================================================
struct McaProf {
  long long k1_ns = 0;  // QK matmul
  long long k2_ns = 0;  // softmax
  long long k3_ns = 0;  // SV matmul
  long long upload_ns = 0;
  long long read_ns = 0;
  long long calls = 0;
  long long m_sum = 0;
  ~McaProf() {
    if (!calls) return;
    std::FILE *f = std::fopen("/data/local/tmp/qwen3_qint4/mca_prof.log", "a");
    if (!f) f = stderr;
    auto pct = [&](long long x) {
      long long tot = k1_ns + k2_ns + k3_ns + upload_ns + read_ns;
      return tot > 0 ? (double)x * 100.0 / (double)tot : 0.0;
    };
    std::fprintf(f,
                 "\n[MCA-PROF] %lld calls (avg M=%.1f)\n"
                 "  upload    %.2f ms (%.1f%%)\n"
                 "  K1 QK     %.2f ms (%.1f%%)\n"
                 "  K2 sftmx  %.2f ms (%.1f%%)\n"
                 "  K3 SV     %.2f ms (%.1f%%)\n"
                 "  read_out  %.2f ms (%.1f%%)\n"
                 "  total     %.2f ms\n",
                 calls, (double)m_sum / (double)calls,
                 upload_ns / 1e6, pct(upload_ns),
                 k1_ns / 1e6, pct(k1_ns),
                 k2_ns / 1e6, pct(k2_ns),
                 k3_ns / 1e6, pct(k3_ns),
                 read_ns / 1e6, pct(read_ns),
                 (k1_ns + k2_ns + k3_ns + upload_ns + read_ns) / 1e6);
    std::fflush(f);
    if (f != stderr) std::fclose(f);
  }
};
static McaProf &mca_prof() {
  static McaProf p;
  return p;
}
static bool mca_prof_enabled() {
  static int cached = -1;
  if (cached < 0)
    cached = std::getenv("NNTR_MHA_PROFILE") != nullptr ? 1 : 0;
  return cached != 0;
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
  // KNOWN BUG (session 2026-05-28): the qk_matmul_f16 kernel produces
  // numerically wrong scores when it actually runs (output text
  // becomes "aines"-style garbage instead of the expected response).
  // Previously the half8 subscript build error masked this — kp was
  // null, so the function returned false and the caller fell back to
  // CPU mha. After 4c824219 unblocked the build, the underlying
  // correctness bug surfaced.
  //
  // Until the kernel correctness is debugged (need CPU-vs-GPU score
  // diff harness), unconditionally return false here so the call site
  // falls back to CPU mha (~295 TPS, coherent output) instead of
  // running the broken GPU path (~138 TPS, garbage output).
  //
  // To re-test the kernel, set NNTR_MHA_GPU_FORCE_BROKEN=1.
  if (std::getenv("NNTR_MHA_GPU_FORCE_BROKEN") == nullptr) {
    static int warned = 0;
    if (!warned && std::getenv("NNTR_MHA_GPU") != nullptr) {
      warned = 1;
      std::fprintf(stderr,
                   "[NOTE] NNTR_MHA_GPU=1 requested but the GPU mha "
                   "kernel has a known correctness bug; using CPU mha "
                   "instead. Set NNTR_MHA_GPU_FORCE_BROKEN=1 to test "
                   "the GPU kernel anyway (output will be garbled).\n");
      std::fflush(stderr);
    }
    return false;
  }

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

  // Pre-K1 sync point for profiling. Always cheap when env unset.
  const bool _prof = std::getenv("NNTR_MHA_PROFILE") != nullptr;
  std::chrono::steady_clock::time_point _t0, _t1;
  auto NOW = []() { return std::chrono::steady_clock::now(); };
  auto NS = [](auto t1, auto t0) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0)
      .count();
  };
  if (_prof) {
    clFinish(q);
    _t0 = NOW();
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
    // Pack WIs into Adreno-sized workgroups so the 64-wide subgroup is
    // actually fed. Without an explicit lws the driver defaults to 1
    // WI per WG, which on Adreno 830 leaves the wave/subgroup empty
    // and pegs the kernel to ~1/64 of compute peak. Pad gws[0] up to a
    // multiple of LWS_QK[0]; the kernel's `if (m0 >= M || n0 >= N_kv)
    // return` handles the padded slots harmlessly.
    constexpr size_t LWS_QK_X = 64;
    const size_t nx_pad = ((nx + LWS_QK_X - 1) / LWS_QK_X) * LWS_QK_X;
    std::array<size_t, 3> gws = {nx_pad, mx, num_heads_Q};
    std::array<size_t, 3> lws = {LWS_QK_X, 1, 1};
    blas_cc->command_queue_inst_.enqueueKernel(kp->GetKernel(), 3, gws.data(),
                                               lws.data(), 0, nullptr, nullptr);
  }
  if (_prof) {
    clFinish(q);
    _t1 = NOW();
    mca_prof().k1_ns += NS(_t1, _t0);
    _t0 = NOW();
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
  if (_prof) {
    clFinish(q);
    _t1 = NOW();
    mca_prof().k2_ns += NS(_t1, _t0);
    _t0 = NOW();
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
    // Same fix as QK: explicit lws to fill the 64-wide Adreno subgroup
    // instead of relying on driver-default lws=1. Kernel has bounds
    // check so padded slots harmlessly return.
    constexpr size_t LWS_SV_X = 64;
    const size_t dx_pad = ((dx + LWS_SV_X - 1) / LWS_SV_X) * LWS_SV_X;
    std::array<size_t, 3> gws = {dx_pad, mx, num_heads_Q};
    std::array<size_t, 3> lws = {LWS_SV_X, 1, 1};
    blas_cc->command_queue_inst_.enqueueKernel(kp->GetKernel(), 3, gws.data(),
                                               lws.data(), 0, nullptr, nullptr);
  }
  if (_prof) {
    clFinish(q);
    _t1 = NOW();
    mca_prof().k3_ns += NS(_t1, _t0);
    _t0 = NOW();
  }

  if (svm_inputs) {
    clFinish(q);
  } else {
    if (clEnqueueReadBuffer(q, sc.o_buf, CL_TRUE, 0, o_bytes, O_host, 0,
                            nullptr, nullptr) != CL_SUCCESS)
      return false;
  }
  if (_prof) {
    _t1 = NOW();
    mca_prof().read_ns += NS(_t1, _t0);
    mca_prof().calls += 1;
    mca_prof().m_sum += M;
    // Periodic dump every N calls (don't rely on dtor firing).
    if (mca_prof().calls % 28 == 0) {
      const McaProf &p = mca_prof();
      long long tot = p.k1_ns + p.k2_ns + p.k3_ns + p.read_ns;
      std::fprintf(stderr,
                   "[MCA-PROF] %lld calls (avg M=%.1f) "
                   "K1=%.1fms K2=%.1fms K3=%.1fms read=%.1fms total=%.1fms\n",
                   p.calls, (double)p.m_sum / (double)p.calls,
                   p.k1_ns / 1e6, p.k2_ns / 1e6, p.k3_ns / 1e6,
                   p.read_ns / 1e6, tot / 1e6);
      std::fflush(stderr);
    }
  }
  return true;
}

// =============================================================================
// int8-KV variant. Mirrors two_conv_attention_prefill_f16_cl but binds the
// int8 K/V byte buffers + their FP16 scale buffers, and dispatches the
// qk_matmul_f16_kvi8 / sv_matmul_f16_kvi8 kernels. Softmax kernel is
// shared with the fp16 variant since it operates only on the score buffer.
// =============================================================================
bool two_conv_attention_prefill_f16_kvi8_cl(
  const uint16_t *Q_host, const int8_t *K_i8_host, const int8_t *V_i8_host,
  const uint16_t *K_scale_host, const uint16_t *V_scale_host,
  uint16_t *O_host, unsigned int M, unsigned int N_kv, unsigned int num_heads_Q,
  unsigned int num_heads_KV, unsigned int head_dim, bool causal,
  bool svm_inputs) {
  if (head_dim == 0 || M == 0 || N_kv == 0) return false;
  if (num_heads_KV == 0 || num_heads_Q % num_heads_KV != 0) return false;
  constexpr unsigned int TM_QK = 4, TN_QK = 8;
  constexpr unsigned int TM_SV = 4, TD_SV = 8;
  constexpr unsigned int SOFTMAX_LWS = 64;
  if (head_dim % TD_SV != 0) return false;

  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  cl_context ctx = blas_cc->context_inst_.GetContext();
  cl_command_queue q = blas_cc->command_queue_inst_.GetCommandQueue();

  const size_t HD_Q = (size_t)num_heads_Q * head_dim;
  const size_t HD_KV = (size_t)num_heads_KV * head_dim;
  const size_t q_bytes = (size_t)M * HD_Q * sizeof(uint16_t);
  const size_t k_i8_bytes = (size_t)N_kv * HD_KV * sizeof(int8_t);
  const size_t v_i8_bytes = k_i8_bytes;
  const size_t kscale_bytes =
    (size_t)N_kv * num_heads_KV * sizeof(uint16_t);
  const size_t vscale_bytes = kscale_bytes;
  const size_t o_bytes = (size_t)M * HD_Q * sizeof(uint16_t);
  const size_t scores_bytes =
    (size_t)num_heads_Q * M * N_kv * sizeof(uint16_t);

  std::lock_guard<std::mutex> lock(tca_mtx());
  TcaScratch &sc = tca_scratch();
  if (!tca_ensure(ctx, &sc.scores, &sc.scores_bytes, scores_bytes,
                  CL_MEM_READ_WRITE))
    return false;

  cl_mem q_arg = nullptr, k_arg = nullptr, v_arg = nullptr, o_arg = nullptr;
  cl_mem k_scale_arg = nullptr, v_scale_arg = nullptr;
  if (!svm_inputs) {
    if (!tca_ensure(ctx, &sc.q_buf, &sc.q_bytes, q_bytes, CL_MEM_READ_ONLY) ||
        !tca_ensure(ctx, &sc.k_buf, &sc.k_bytes, k_i8_bytes,
                    CL_MEM_READ_ONLY) ||
        !tca_ensure(ctx, &sc.v_buf, &sc.v_bytes, v_i8_bytes,
                    CL_MEM_READ_ONLY) ||
        !tca_ensure(ctx, &sc.k_scale_buf, &sc.k_scale_bytes, kscale_bytes,
                    CL_MEM_READ_ONLY) ||
        !tca_ensure(ctx, &sc.v_scale_buf, &sc.v_scale_bytes, vscale_bytes,
                    CL_MEM_READ_ONLY) ||
        !tca_ensure(ctx, &sc.o_buf, &sc.o_bytes, o_bytes, CL_MEM_WRITE_ONLY))
      return false;
    if (clEnqueueWriteBuffer(q, sc.q_buf, CL_FALSE, 0, q_bytes, Q_host, 0,
                             nullptr, nullptr) != CL_SUCCESS ||
        clEnqueueWriteBuffer(q, sc.k_buf, CL_FALSE, 0, k_i8_bytes, K_i8_host, 0,
                             nullptr, nullptr) != CL_SUCCESS ||
        clEnqueueWriteBuffer(q, sc.v_buf, CL_FALSE, 0, v_i8_bytes, V_i8_host, 0,
                             nullptr, nullptr) != CL_SUCCESS ||
        clEnqueueWriteBuffer(q, sc.k_scale_buf, CL_FALSE, 0, kscale_bytes,
                             K_scale_host, 0, nullptr, nullptr) != CL_SUCCESS ||
        clEnqueueWriteBuffer(q, sc.v_scale_buf, CL_FALSE, 0, vscale_bytes,
                             V_scale_host, 0, nullptr, nullptr) != CL_SUCCESS)
      return false;
    q_arg = sc.q_buf;
    k_arg = sc.k_buf;
    v_arg = sc.v_buf;
    o_arg = sc.o_buf;
    k_scale_arg = sc.k_scale_buf;
    v_scale_arg = sc.v_scale_buf;
  }

  // ---- K1: QK matmul (int8 K + scale) ----
  {
    ClContext::SharedPtrClKernel kp = blas_cc->registerClKernel(
      two_conv_attention_kernel, "qk_matmul_f16_kvi8");
    if (!kp) return false;
    if (svm_inputs) {
      if (!kp->SetKernelSVMArguments(0, const_cast<uint16_t *>(Q_host)) ||
          !kp->SetKernelSVMArguments(1, const_cast<int8_t *>(K_i8_host)) ||
          !kp->SetKernelSVMArguments(2, const_cast<uint16_t *>(K_scale_host)))
        return false;
    } else {
      if (!kp->SetKernelArguments(0, &q_arg, sizeof(cl_mem)) ||
          !kp->SetKernelArguments(1, &k_arg, sizeof(cl_mem)) ||
          !kp->SetKernelArguments(2, &k_scale_arg, sizeof(cl_mem)))
        return false;
    }
    if (!kp->SetKernelArguments(3, &sc.scores, sizeof(cl_mem))) return false;
    int Mi = (int)M, Nkvi = (int)N_kv, di = (int)head_dim;
    int hdq = (int)HD_Q, hdkv = (int)HD_KV;
    int gqa = (int)(num_heads_Q / num_heads_KV);
    int nhkv = (int)num_heads_KV;
    int causal_i = causal ? 1 : 0;
    float scale = 1.0f / std::sqrt((float)head_dim);
    if (!kp->SetKernelArguments(4, &Mi, sizeof(int)) ||
        !kp->SetKernelArguments(5, &Nkvi, sizeof(int)) ||
        !kp->SetKernelArguments(6, &di, sizeof(int)) ||
        !kp->SetKernelArguments(7, &hdq, sizeof(int)) ||
        !kp->SetKernelArguments(8, &hdkv, sizeof(int)) ||
        !kp->SetKernelArguments(9, &gqa, sizeof(int)) ||
        !kp->SetKernelArguments(10, &nhkv, sizeof(int)) ||
        !kp->SetKernelArguments(11, &causal_i, sizeof(int)) ||
        !kp->SetKernelArguments(12, &scale, sizeof(float)))
      return false;
    const size_t nx = (N_kv + TN_QK - 1) / TN_QK;
    const size_t mx = (M + TM_QK - 1) / TM_QK;
    constexpr size_t LWS_QK_X = 64;
    const size_t nx_pad = ((nx + LWS_QK_X - 1) / LWS_QK_X) * LWS_QK_X;
    std::array<size_t, 3> gws = {nx_pad, mx, num_heads_Q};
    std::array<size_t, 3> lws = {LWS_QK_X, 1, 1};
    blas_cc->command_queue_inst_.enqueueKernel(kp->GetKernel(), 3, gws.data(),
                                               lws.data(), 0, nullptr, nullptr);
  }

  // ---- K2: row softmax over N_kv (shared with fp16 path) ----
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

  // ---- K3: scores @ V (int8 V + scale) -> O ----
  {
    ClContext::SharedPtrClKernel kp = blas_cc->registerClKernel(
      two_conv_attention_kernel, "sv_matmul_f16_kvi8");
    if (!kp) return false;
    if (!kp->SetKernelArguments(0, &sc.scores, sizeof(cl_mem))) return false;
    if (svm_inputs) {
      if (!kp->SetKernelSVMArguments(1, const_cast<int8_t *>(V_i8_host)) ||
          !kp->SetKernelSVMArguments(2, const_cast<uint16_t *>(V_scale_host)) ||
          !kp->SetKernelSVMArguments(3, O_host))
        return false;
    } else {
      if (!kp->SetKernelArguments(1, &v_arg, sizeof(cl_mem)) ||
          !kp->SetKernelArguments(2, &v_scale_arg, sizeof(cl_mem)) ||
          !kp->SetKernelArguments(3, &o_arg, sizeof(cl_mem)))
        return false;
    }
    int Mi = (int)M, Nkvi = (int)N_kv, di = (int)head_dim;
    int hdq = (int)HD_Q, hdkv = (int)HD_KV;
    int gqa = (int)(num_heads_Q / num_heads_KV);
    int nhkv = (int)num_heads_KV;
    if (!kp->SetKernelArguments(4, &Mi, sizeof(int)) ||
        !kp->SetKernelArguments(5, &Nkvi, sizeof(int)) ||
        !kp->SetKernelArguments(6, &di, sizeof(int)) ||
        !kp->SetKernelArguments(7, &hdq, sizeof(int)) ||
        !kp->SetKernelArguments(8, &hdkv, sizeof(int)) ||
        !kp->SetKernelArguments(9, &gqa, sizeof(int)) ||
        !kp->SetKernelArguments(10, &nhkv, sizeof(int)))
      return false;
    const size_t dx = (head_dim + TD_SV - 1) / TD_SV;
    const size_t mx = (M + TM_SV - 1) / TM_SV;
    // Same fix as QK: explicit lws to fill the 64-wide Adreno subgroup
    // instead of relying on driver-default lws=1. Kernel has bounds
    // check so padded slots harmlessly return.
    constexpr size_t LWS_SV_X = 64;
    const size_t dx_pad = ((dx + LWS_SV_X - 1) / LWS_SV_X) * LWS_SV_X;
    std::array<size_t, 3> gws = {dx_pad, mx, num_heads_Q};
    std::array<size_t, 3> lws = {LWS_SV_X, 1, 1};
    blas_cc->command_queue_inst_.enqueueKernel(kp->GetKernel(), 3, gws.data(),
                                               lws.data(), 0, nullptr, nullptr);
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

// =============================================================================
// image2d_from_buffer variant. Reads Q/K/V via 16-byte texels (8 halves per
// texel) — same trick that gave v8c FC kernel 87% of Adreno 830 peak. 8x
// fewer memory transactions per WI in the d-axis reduction. Non-SVM only:
// image2d_from_buffer requires a cl_mem, so SVM inputs are first copied to
// the scratch cl_mems (kept in TcaScratch alongside the fp16 wrapper's).
// =============================================================================
bool two_conv_attention_prefill_f16_img_cl(
  const uint16_t *Q_host, const uint16_t *K_host, const uint16_t *V_host,
  uint16_t *O_host, unsigned int M, unsigned int N_kv, unsigned int num_heads_Q,
  unsigned int num_heads_KV, unsigned int head_dim, bool causal) {
  if (head_dim == 0 || M == 0 || N_kv == 0) return false;
  if (num_heads_KV == 0 || num_heads_Q % num_heads_KV != 0) return false;
  // Smaller image-variant tiles to avoid register spill on Adreno.
  constexpr unsigned int TM_IMG = 2, TN_IMG = 4;
  constexpr unsigned int TM_SV_IMG = 2;
  constexpr unsigned int SOFTMAX_LWS = 64;
  // image2d packing requires d-multiple-of-8 + HD multiples-of-8.
  if (head_dim % 8 != 0) return false;

  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  cl_context ctx = blas_cc->context_inst_.GetContext();
  cl_command_queue q = blas_cc->command_queue_inst_.GetCommandQueue();

  const size_t HD_Q = (size_t)num_heads_Q * head_dim;
  const size_t HD_KV = (size_t)num_heads_KV * head_dim;
  if (HD_Q % 8 != 0 || HD_KV % 8 != 0) return false;
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

  // Build image2d views over the buffers (cached across layers — same shape
  // is reused across all 28 transformer blocks during a prefill).
  // RGBA UINT32 = 16 bytes = 8 halves per texel.
  cl_int err = CL_SUCCESS;
  const bool shape_changed = sc.img_M != M || sc.img_N_kv != N_kv ||
                             sc.img_HD_Q != HD_Q || sc.img_HD_KV != HD_KV ||
                             !sc.q_image || !sc.k_image || !sc.v_image;
  if (shape_changed) {
    if (sc.q_image) { clReleaseMemObject(sc.q_image); sc.q_image = nullptr; }
    if (sc.k_image) { clReleaseMemObject(sc.k_image); sc.k_image = nullptr; }
    if (sc.v_image) { clReleaseMemObject(sc.v_image); sc.v_image = nullptr; }

    cl_image_format img_fmt{CL_RGBA, CL_UNSIGNED_INT32};
    cl_image_desc qd{};
    qd.image_type = CL_MEM_OBJECT_IMAGE2D;
    qd.image_width = HD_Q / 8;
    qd.image_height = M;
    qd.image_row_pitch = HD_Q * sizeof(uint16_t);
    qd.buffer = sc.q_buf;
    sc.q_image =
      clCreateImage(ctx, CL_MEM_READ_ONLY, &img_fmt, &qd, nullptr, &err);
    if (err != CL_SUCCESS || !sc.q_image) return false;

    cl_image_desc kd{};
    kd.image_type = CL_MEM_OBJECT_IMAGE2D;
    kd.image_width = HD_KV / 8;
    kd.image_height = N_kv;
    kd.image_row_pitch = HD_KV * sizeof(uint16_t);
    kd.buffer = sc.k_buf;
    sc.k_image =
      clCreateImage(ctx, CL_MEM_READ_ONLY, &img_fmt, &kd, nullptr, &err);
    if (err != CL_SUCCESS || !sc.k_image) return false;

    cl_image_desc vd = kd;
    vd.buffer = sc.v_buf;
    sc.v_image =
      clCreateImage(ctx, CL_MEM_READ_ONLY, &img_fmt, &vd, nullptr, &err);
    if (err != CL_SUCCESS || !sc.v_image) return false;

    sc.img_M = M;
    sc.img_N_kv = N_kv;
    sc.img_HD_Q = HD_Q;
    sc.img_HD_KV = HD_KV;
  }
  cl_mem q_image = sc.q_image;
  cl_mem k_image = sc.k_image;
  cl_mem v_image = sc.v_image;
  auto cleanup = []() {};  // images are cached, no per-call release

  // ---- K1: QK matmul (image2d Q, K) ----
  {
    ClContext::SharedPtrClKernel kp = blas_cc->registerClKernel(
      two_conv_attention_kernel, "qk_matmul_f16_img");
    if (!kp) { cleanup(); return false; }
    if (!kp->SetKernelArguments(0, &q_image, sizeof(cl_mem)) ||
        !kp->SetKernelArguments(1, &k_image, sizeof(cl_mem)) ||
        !kp->SetKernelArguments(2, &sc.scores, sizeof(cl_mem))) {
      cleanup(); return false;
    }
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
        !kp->SetKernelArguments(10, &scale, sizeof(float))) {
      cleanup(); return false;
    }
    const size_t nx = (N_kv + TN_IMG - 1) / TN_IMG;
    const size_t mx = (M + TM_IMG - 1) / TM_IMG;
    // Fair comparison to the buffer variant: same lws=64 fix
    // (see aad66ab5). Without this the image kernel was dispatched
    // with driver-default lws=1, leaving the 64-wide Adreno subgroup
    // empty and tanking image2d perf below buffer perf.
    constexpr size_t LWS_QK_X = 64;
    const size_t nx_pad = ((nx + LWS_QK_X - 1) / LWS_QK_X) * LWS_QK_X;
    std::array<size_t, 3> gws = {nx_pad, mx, num_heads_Q};
    std::array<size_t, 3> lws = {LWS_QK_X, 1, 1};
    blas_cc->command_queue_inst_.enqueueKernel(kp->GetKernel(), 3, gws.data(),
                                               lws.data(), 0, nullptr, nullptr);
  }

  // ---- K2: softmax (shared with the scalar fp16 path) ----
  {
    ClContext::SharedPtrClKernel kp =
      blas_cc->registerClKernel(two_conv_attention_kernel, "softmax_row_f16");
    if (!kp) { cleanup(); return false; }
    if (!kp->SetKernelArguments(0, &sc.scores, sizeof(cl_mem))) {
      cleanup(); return false;
    }
    int Mi = (int)M, Nkvi = (int)N_kv;
    if (!kp->SetKernelArguments(1, &Mi, sizeof(int)) ||
        !kp->SetKernelArguments(2, &Nkvi, sizeof(int))) {
      cleanup(); return false;
    }
    std::array<size_t, 3> gws = {SOFTMAX_LWS, M, num_heads_Q};
    std::array<size_t, 3> lws = {SOFTMAX_LWS, 1, 1};
    blas_cc->command_queue_inst_.enqueueKernel(kp->GetKernel(), 3, gws.data(),
                                               lws.data(), 0, nullptr, nullptr);
  }

  // ---- K3: SV matmul (image2d V) ----
  {
    ClContext::SharedPtrClKernel kp = blas_cc->registerClKernel(
      two_conv_attention_kernel, "sv_matmul_f16_img");
    if (!kp) { cleanup(); return false; }
    if (!kp->SetKernelArguments(0, &sc.scores, sizeof(cl_mem)) ||
        !kp->SetKernelArguments(1, &v_image, sizeof(cl_mem)) ||
        !kp->SetKernelArguments(2, &sc.o_buf, sizeof(cl_mem))) {
      cleanup(); return false;
    }
    int Mi = (int)M, Nkvi = (int)N_kv, di = (int)head_dim;
    int hdq = (int)HD_Q, hdkv = (int)HD_KV;
    int gqa = (int)(num_heads_Q / num_heads_KV);
    if (!kp->SetKernelArguments(3, &Mi, sizeof(int)) ||
        !kp->SetKernelArguments(4, &Nkvi, sizeof(int)) ||
        !kp->SetKernelArguments(5, &di, sizeof(int)) ||
        !kp->SetKernelArguments(6, &hdq, sizeof(int)) ||
        !kp->SetKernelArguments(7, &hdkv, sizeof(int)) ||
        !kp->SetKernelArguments(8, &gqa, sizeof(int))) {
      cleanup(); return false;
    }
    const size_t dx = head_dim / 8;
    const size_t mx = (M + TM_SV_IMG - 1) / TM_SV_IMG;
    // Same lws=64 fix as the buffer variant. For Qwen3 head_dim=128,
    // dx=16 — padded up to 64; WIs 16..63 early-out via `x0 >= d`.
    constexpr size_t LWS_SV_X = 64;
    const size_t dx_pad = ((dx + LWS_SV_X - 1) / LWS_SV_X) * LWS_SV_X;
    std::array<size_t, 3> gws = {dx_pad, mx, num_heads_Q};
    std::array<size_t, 3> lws = {LWS_SV_X, 1, 1};
    blas_cc->command_queue_inst_.enqueueKernel(kp->GetKernel(), 3, gws.data(),
                                               lws.data(), 0, nullptr, nullptr);
  }

  if (clEnqueueReadBuffer(q, sc.o_buf, CL_TRUE, 0, o_bytes, O_host, 0, nullptr,
                          nullptr) != CL_SUCCESS) {
    cleanup();
    return false;
  }
  cleanup();
  return true;
}

// =============================================================================
// Step #1 (skeleton) of the GPU mha migration. flash-attention single-
// kernel prefill. See attention_kernels.h declaration + flash_attention.cl
// for the kernel-side design + memory budget. This wrapper is the host
// dispatch point that mha_core.cpp will call once Step #2 fills in the
// kernel body.
//
// Step status:
//   - .cl file is registered, kernel compiles into the registry.
//   - this wrapper validates shapes, sets up cl_mem bindings, dispatches
//     the kernel, and reads the result back. Body of the kernel is a
//     zero-fill stub, so the output is meaningless. Wrapper returns
//     `false` regardless of dispatch outcome so the caller's existing
//     CPU path runs. The point is to exercise the build + symbol +
//     env-gate plumbing.
// =============================================================================
namespace {

static bool gpu_mha_env_enabled() {
  static int cached = -1;
  if (cached < 0)
    cached = std::getenv("NNTR_GPU_MHA") != nullptr ? 1 : 0;
  return cached != 0;
}

} // anonymous namespace

bool flash_attention_prefill_f16_cl(const uint16_t *Q_host,
                                    const uint16_t *K_host,
                                    const uint16_t *V_host,
                                    uint16_t *O_host, unsigned int M,
                                    unsigned int N_kv,
                                    unsigned int num_heads_Q,
                                    unsigned int num_heads_KV,
                                    unsigned int head_dim, bool causal,
                                    bool svm_inputs) {
  if (!gpu_mha_env_enabled())
    return false;
  if (num_heads_Q == 0 || num_heads_KV == 0 || head_dim == 0 || M == 0 ||
      N_kv == 0)
    return false;
  if (num_heads_Q % num_heads_KV != 0)
    return false;

  // Step 1 stub log so we can confirm the dispatch path is reachable
  // on real device traffic when NNTR_GPU_MHA=1 + NNTR_GPU_MHA_TRIP=1.
  static int logged_trip = 0;
  if (!logged_trip && std::getenv("NNTR_GPU_MHA_TRIP") != nullptr) {
    logged_trip = 1;
    std::fprintf(stderr,
                 "[GPU-MHA] flash_attention stub reached: M=%u N_kv=%u "
                 "hq=%u hkv=%u d=%u causal=%d svm=%d\n",
                 M, N_kv, num_heads_Q, num_heads_KV, head_dim,
                 (int)causal, (int)svm_inputs);
    std::fflush(stderr);
  }

  // Silence unused-param warnings while body is a stub.
  (void)Q_host; (void)K_host; (void)V_host; (void)O_host;

  // Stub: return false so the caller's existing CPU mha path runs.
  // Step #2 of the GPU mha plan will set up the cl_mem bindings,
  // dispatch the real kernel, and flip the return to true.
  return false;
}

} // namespace nntrainer
