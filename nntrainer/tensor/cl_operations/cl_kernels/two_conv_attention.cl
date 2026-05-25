// SPDX-License-Identifier: Apache-2.0
// Two-1x1-conv attention for prefill, per ML Drift section 3.7
// (matmul-as-convolution pattern; K-cache OHWI, V-cache reversed-dims;
// avoids the per-WI register state that makes flash attention spill on
// mobile GPUs).
//
// Pipeline:
//   K1 qk_matmul   : Q[M, HD_Q]  @ K[N_kv, HD_KV]^T  -> scores[H, M, N_kv]
//   K2 softmax_row : row-wise softmax over the N_kv axis (per (h, m))
//   K3 sv_matmul   : scores[H, M, N_kv] @ V[N_kv, HD_KV] -> O[M, HD_Q]
//
// All three kernels treat each (head_q) tile independently. GQA is
// resolved at compile time inside each kernel via head_kv = head_q /
// gqa. Q/K/V/O are FP16-bit (uint16) in global memory; reductions and
// softmax run in FP32 register precision.

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// =============================================================
// QK matmul: scores[h, m, n] = (Q[m, h, :] . K[n, h_kv, :]) * scale
//   Each WI computes a TM_QK x TN_QK tile of the (M, N_kv) score
//   matrix for a fixed head_q. The d-axis is fully reduced inside
//   the WI; d is small (=128 for Qwen3 / Llama-class models).
//
// gws = (N_kv / TN_QK, M / TM_QK, H)
// Each WI's private accumulator = TM_QK*TN_QK floats (e.g. 4*8 = 32);
// fits comfortably in the per-WI register file with room for K/Q
// staging.
// =============================================================
#ifndef TM_QK
#define TM_QK 4
#endif
#ifndef TN_QK
#define TN_QK 8
#endif

__kernel void qk_matmul_f16(
    __global const half *Q,           // [M, HD_Q] fp16, row-major
    __global const half *K,           // [N_kv, HD_KV] fp16, row-major
    __global       half *scores,      // [H, M, N_kv] fp16, row-major
    const int M, const int N_kv, const int d,
    const int HD_Q, const int HD_KV, const int gqa,
    const int causal, const float scale) {
  const int n0 = get_global_id(0) * TN_QK;
  const int m0 = get_global_id(1) * TM_QK;
  const int head_q = get_global_id(2);
  const int head_kv = head_q / gqa;

  if (m0 >= M || n0 >= N_kv) return;

  float acc[TM_QK][TN_QK];
  #pragma unroll
  for (int i = 0; i < TM_QK; i++)
    #pragma unroll
    for (int j = 0; j < TN_QK; j++) acc[i][j] = 0.0f;

  // Reduction over d.
  for (int x = 0; x < d; x++) {
    half q_col[TM_QK];
    half k_col[TN_QK];
    #pragma unroll
    for (int i = 0; i < TM_QK; i++) {
      const int m = m0 + i;
      q_col[i] = (m < M)
                   ? Q[(long)m * HD_Q + head_q * d + x]
                   : (half)0.0f;
    }
    #pragma unroll
    for (int j = 0; j < TN_QK; j++) {
      const int n = n0 + j;
      k_col[j] = (n < N_kv)
                   ? K[(long)n * HD_KV + head_kv * d + x]
                   : (half)0.0f;
    }
    #pragma unroll
    for (int i = 0; i < TM_QK; i++) {
      const float qf = (float)q_col[i];
      #pragma unroll
      for (int j = 0; j < TN_QK; j++) acc[i][j] += qf * (float)k_col[j];
    }
  }

  // Apply scale + causal mask + write out. The mask must be applied
  // before softmax sees the value, so writing -INFINITY here keeps the
  // softmax kernel mask-agnostic.
  const long score_base = (long)head_q * (long)M * (long)N_kv;
  #pragma unroll
  for (int i = 0; i < TM_QK; i++) {
    const int m = m0 + i;
    if (m >= M) continue;
    #pragma unroll
    for (int j = 0; j < TN_QK; j++) {
      const int n = n0 + j;
      if (n >= N_kv) continue;
      float v = acc[i][j] * scale;
      if (causal && n > m) v = -INFINITY;
      scores[score_base + (long)m * N_kv + n] = (half)v;
    }
  }
}

// =============================================================
// Row softmax (in-place): for each (h, m), softmax over N_kv axis.
//   One workgroup of LWS WIs per (h, m). Three local-memory
//   reductions: max, then exp + sum, then inverse-multiply.
// gws = (LWS, M, H), LWS = SOFTMAX_LWS.
// =============================================================
#ifndef SOFTMAX_LWS
#define SOFTMAX_LWS 64
#endif

__attribute__((reqd_work_group_size(SOFTMAX_LWS, 1, 1)))
__kernel void softmax_row_f16(__global half *scores,    // [H, M, N_kv]
                              const int M, const int N_kv) {
  const int tid = get_local_id(0);
  const int m = get_group_id(1);
  const int h = get_group_id(2);
  if (m >= M) return;

  __global half *row =
    scores + (long)h * (long)M * N_kv + (long)m * N_kv;

  __local float lscratch[SOFTMAX_LWS];

  // Pass 1: per-WI partial max.
  float pmax = -INFINITY;
  for (int n = tid; n < N_kv; n += SOFTMAX_LWS) {
    float v = (float)row[n];
    pmax = fmax(pmax, v);
  }
  lscratch[tid] = pmax;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = SOFTMAX_LWS / 2; s > 0; s >>= 1) {
    if (tid < s) lscratch[tid] = fmax(lscratch[tid], lscratch[tid + s]);
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  const float row_max = lscratch[0];
  barrier(CLK_LOCAL_MEM_FENCE);

  // Pass 2: per-WI exp(x - max) + partial sum. Stash exp() back to row.
  float psum = 0.0f;
  for (int n = tid; n < N_kv; n += SOFTMAX_LWS) {
    float e = (row_max == -INFINITY) ? 0.0f : exp((float)row[n] - row_max);
    row[n] = (half)e;
    psum += e;
  }
  lscratch[tid] = psum;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = SOFTMAX_LWS / 2; s > 0; s >>= 1) {
    if (tid < s) lscratch[tid] += lscratch[tid + s];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  const float row_sum = lscratch[0];
  const float inv = (row_sum > 0.0f) ? (1.0f / row_sum) : 0.0f;
  barrier(CLK_LOCAL_MEM_FENCE);

  // Pass 3: divide by sum (in-place).
  for (int n = tid; n < N_kv; n += SOFTMAX_LWS) {
    row[n] = (half)((float)row[n] * inv);
  }
}

// =============================================================
// SV matmul: O[m, h, x] = sum_n scores[h, m, n] * V[n, h_kv, x]
//   Mirror of qk_matmul: each WI computes a TM_SV x TD_SV tile of
//   (M, d) for fixed head_q. Reduction is over N_kv (1003 for our
//   1k-prefill workload, so larger than d but still small enough
//   that no online softmax is needed - softmax has already run).
//
// gws = (d / TD_SV, M / TM_SV, H)
// =============================================================
#ifndef TM_SV
#define TM_SV 4
#endif
#ifndef TD_SV
#define TD_SV 8
#endif

__kernel void sv_matmul_f16(
    __global const half *scores,      // [H, M, N_kv] fp16, post-softmax
    __global const half *V,           // [N_kv, HD_KV] fp16
    __global       half *O,           // [M, HD_Q] fp16, row-major
    const int M, const int N_kv, const int d,
    const int HD_Q, const int HD_KV, const int gqa) {
  const int x0 = get_global_id(0) * TD_SV;
  const int m0 = get_global_id(1) * TM_SV;
  const int head_q = get_global_id(2);
  const int head_kv = head_q / gqa;

  if (m0 >= M || x0 >= d) return;

  float acc[TM_SV][TD_SV];
  #pragma unroll
  for (int i = 0; i < TM_SV; i++)
    #pragma unroll
    for (int j = 0; j < TD_SV; j++) acc[i][j] = 0.0f;

  const long score_base = (long)head_q * (long)M * (long)N_kv;

  // Reduction over N_kv. We re-read each (m, n) score for each (n, x)
  // V row, which is suboptimal but the score range is small (fp16) and
  // the bandwidth-amplification factor is bounded by TD_SV.
  for (int n = 0; n < N_kv; n++) {
    half s_col[TM_SV];
    half v_col[TD_SV];
    #pragma unroll
    for (int i = 0; i < TM_SV; i++) {
      const int m = m0 + i;
      s_col[i] = (m < M)
                   ? scores[score_base + (long)m * N_kv + n]
                   : (half)0.0f;
    }
    #pragma unroll
    for (int j = 0; j < TD_SV; j++) {
      const int x = x0 + j;
      v_col[j] = (x < d)
                   ? V[(long)n * HD_KV + head_kv * d + x]
                   : (half)0.0f;
    }
    #pragma unroll
    for (int i = 0; i < TM_SV; i++) {
      const float sf = (float)s_col[i];
      #pragma unroll
      for (int j = 0; j < TD_SV; j++) acc[i][j] += sf * (float)v_col[j];
    }
  }

  #pragma unroll
  for (int i = 0; i < TM_SV; i++) {
    const int m = m0 + i;
    if (m >= M) continue;
    #pragma unroll
    for (int j = 0; j < TD_SV; j++) {
      const int x = x0 + j;
      if (x >= d) continue;
      O[(long)m * HD_Q + head_q * d + x] = (half)acc[i][j];
    }
  }
}
