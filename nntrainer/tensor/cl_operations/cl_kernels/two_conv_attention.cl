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
// §3.8 OHWI K-cache variant of qk_matmul_f16.
// K is laid out as [H_kv, S_max, d] (per-head contiguous over the
// S/d axes — the OHWI "convolution weight" form), not the default
// row-major [N_kv, H_kv * d]. The kernel only reads N_kv rows out
// of the S_max allocated; S_max is the head stride.
//
// Element index:  K[(long)head_kv * (long)S_max * d + n * d + x]
//
// Per-head contiguous d-axis reads (stride 1 inside the inner d loop)
// are more cache-line friendly than the concat layout's strided reads
// (stride = HD_KV per token), so the scalar kernel should match or
// beat the concat variant. Image2d OHWI variant is a follow-up.
// =============================================================
__kernel void qk_matmul_f16_ohwi(
    __global const half *Q,           // [M, HD_Q] fp16, row-major (unchanged)
    __global const half *K,           // [H_kv, S_max, d] fp16, OHWI
    __global       half *scores,      // [H, M, N_kv] fp16, row-major
    const int M, const int N_kv, const int d,
    const int HD_Q, const int S_max, const int gqa,
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

  const long k_head_base = (long)head_kv * (long)S_max * (long)d;

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
                   ? K[k_head_base + (long)n * d + x]
                   : (half)0.0f;
    }
    #pragma unroll
    for (int i = 0; i < TM_QK; i++) {
      const float qf = (float)q_col[i];
      #pragma unroll
      for (int j = 0; j < TN_QK; j++) acc[i][j] += qf * (float)k_col[j];
    }
  }

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

// =============================================================
// int8-KV variants (paper §3.7 int8 KV path).
// K/V cache is stored as signed int8 bytes; a per-(token, head)
// FP16 amax scale lifts the int8 values back to fp16 range.
// Same memory layout as the f16 kernels above:
//   K_i8[n, head_kv, x]   -> n*HD_KV + head_kv*d + x (1 byte each)
//   K_scale[n, head_kv]   -> n*num_heads_KV + head_kv (fp16)
// The scale is constant across the d-axis for a given (n, head_kv),
// so we multiply once per (n, head_kv) outside the inner d-loop.
// =============================================================

__kernel void qk_matmul_f16_kvi8(
    __global const half *Q,           // [M, HD_Q] fp16
    __global const char *K_i8,        // [N_kv, HD_KV] int8 (signed)
    __global const half *K_scale,     // [N_kv, num_heads_KV] fp16
    __global       half *scores,      // [H, M, N_kv] fp16
    const int M, const int N_kv, const int d,
    const int HD_Q, const int HD_KV, const int gqa,
    const int num_heads_KV,
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

  // Reduction over d. Q in fp16, K in int8; convert int8 -> fp32 at
  // load and accumulate in fp32 (matches CPU helper's precision).
  for (int x = 0; x < d; x++) {
    half q_col[TM_QK];
    float k_col[TN_QK];
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
                   ? convert_float(K_i8[(long)n * HD_KV + head_kv * d + x])
                   : 0.0f;
    }
    #pragma unroll
    for (int i = 0; i < TM_QK; i++) {
      const float qf = (float)q_col[i];
      #pragma unroll
      for (int j = 0; j < TN_QK; j++) acc[i][j] += qf * k_col[j];
    }
  }

  // Apply per-(n, head_kv) scale, the softmax scale, causal mask, and
  // write out. Scale is constant in d for each (n, head_kv) so it
  // factors out of the inner reduction.
  float ks[TN_QK];
  #pragma unroll
  for (int j = 0; j < TN_QK; j++) {
    const int n = n0 + j;
    ks[j] = (n < N_kv)
              ? (float)K_scale[(long)n * num_heads_KV + head_kv]
              : 0.0f;
  }

  const long score_base = (long)head_q * (long)M * (long)N_kv;
  #pragma unroll
  for (int i = 0; i < TM_QK; i++) {
    const int m = m0 + i;
    if (m >= M) continue;
    #pragma unroll
    for (int j = 0; j < TN_QK; j++) {
      const int n = n0 + j;
      if (n >= N_kv) continue;
      float v = acc[i][j] * ks[j] * scale;
      if (causal && n > m) v = -INFINITY;
      scores[score_base + (long)m * N_kv + n] = (half)v;
    }
  }
}

__kernel void sv_matmul_f16_kvi8(
    __global const half *scores,      // [H, M, N_kv] fp16 post-softmax
    __global const char *V_i8,        // [N_kv, HD_KV] int8
    __global const half *V_scale,     // [N_kv, num_heads_KV] fp16
    __global       half *O,           // [M, HD_Q] fp16
    const int M, const int N_kv, const int d,
    const int HD_Q, const int HD_KV, const int gqa,
    const int num_heads_KV) {
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

  // Reduction over N_kv. The V_scale is constant in d for a given
  // (n, head_kv) so we fold it into the score multiplier once per n.
  for (int n = 0; n < N_kv; n++) {
    const float vs = (float)V_scale[(long)n * num_heads_KV + head_kv];
    half s_col[TM_SV];
    float v_col[TD_SV];
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
                   ? convert_float(V_i8[(long)n * HD_KV + head_kv * d + x])
                   : 0.0f;
    }
    #pragma unroll
    for (int i = 0; i < TM_SV; i++) {
      const float sf = (float)s_col[i] * vs;
      #pragma unroll
      for (int j = 0; j < TD_SV; j++) acc[i][j] += sf * v_col[j];
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

// =============================================================
// Packed (image2d_from_buffer) variants of qk_matmul / sv_matmul.
// Pattern reused from v8c FC kernel (87% Adreno 830 peak): view the
// fp16 buffer as RGBA UINT32 image2d (16 bytes = 8 halves per texel),
// read via read_imageui returning uint4, reinterpret as half8 with
// as_half8(). 8x fewer memory transactions than scalar half loads.
// d, HD_Q, HD_KV must be multiples of 8.
// =============================================================

// Smaller tile (TM_IMG=2, TN_IMG=4 = 8 acc) to keep register pressure low.
// half8 staging × (TM_IMG+TN_IMG) = 6 half8 = 48 halves = 96 bytes plus
// the 8 float acc = 32 bytes. Fits comfortably in the per-WI register
// file even with private temporaries.
#ifndef TM_IMG
#define TM_IMG 2
#endif
#ifndef TN_IMG
#define TN_IMG 4
#endif

__kernel void qk_matmul_f16_img(
    __read_only image2d_t Q_img,      // width=HD_Q/8 texels, height=M
    __read_only image2d_t K_img,      // width=HD_KV/8 texels, height=N_kv
    __global       half *scores,      // [H, M, N_kv] fp16, row-major
    const int M, const int N_kv, const int d,
    const int HD_Q, const int HD_KV, const int gqa,
    const int causal, const float scale) {
  const int n0 = get_global_id(0) * TN_IMG;
  const int m0 = get_global_id(1) * TM_IMG;
  const int head_q = get_global_id(2);
  const int head_kv = head_q / gqa;

  if (m0 >= M || n0 >= N_kv) return;

  float acc[TM_IMG][TN_IMG];
  #pragma unroll
  for (int i = 0; i < TM_IMG; i++)
    #pragma unroll
    for (int j = 0; j < TN_IMG; j++) acc[i][j] = 0.0f;

  const sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP |
                        CLK_FILTER_NEAREST;
  const int q_tx0 = (head_q * d) >> 3;   // 1 texel = 8 halves
  const int k_tx0 = (head_kv * d) >> 3;
  const int d_tex = d >> 3;

  for (int xt = 0; xt < d_tex; xt++) {
    half8 q_pack[TM_IMG];
    half8 k_pack[TN_IMG];

    #pragma unroll
    for (int i = 0; i < TM_IMG; i++) {
      const int m = m0 + i;
      const int my = (m < M) ? m : 0;
      const uint4 v = read_imageui(Q_img, smp, (int2)(q_tx0 + xt, my));
      const half8 hp = as_half8(v);
      q_pack[i] = (m < M) ? hp : (half8)((half)0.0h);
    }
    #pragma unroll
    for (int j = 0; j < TN_IMG; j++) {
      const int n = n0 + j;
      const int ny = (n < N_kv) ? n : 0;
      const uint4 v = read_imageui(K_img, smp, (int2)(k_tx0 + xt, ny));
      const half8 hp = as_half8(v);
      k_pack[j] = (n < N_kv) ? hp : (half8)((half)0.0h);
    }

    // Per (i, j): compute one half-precision dot via two half4 dot()
    // builtins. The compiler maps each dot(float4,float4) to fma chains.
    #pragma unroll
    for (int i = 0; i < TM_IMG; i++) {
      const float4 qlo = convert_float4(q_pack[i].s0123);
      const float4 qhi = convert_float4(q_pack[i].s4567);
      #pragma unroll
      for (int j = 0; j < TN_IMG; j++) {
        const float4 klo = convert_float4(k_pack[j].s0123);
        const float4 khi = convert_float4(k_pack[j].s4567);
        acc[i][j] += dot(qlo, klo) + dot(qhi, khi);
      }
    }
  }

  // Apply scale + causal mask + write.
  const long score_base = (long)head_q * (long)M * (long)N_kv;
  #pragma unroll
  for (int i = 0; i < TM_IMG; i++) {
    const int m = m0 + i;
    if (m >= M) continue;
    #pragma unroll
    for (int j = 0; j < TN_IMG; j++) {
      const int n = n0 + j;
      if (n >= N_kv) continue;
      float v = acc[i][j] * scale;
      if (causal && n > m) v = -INFINITY;
      scores[score_base + (long)m * N_kv + n] = (half)v;
    }
  }
}

// SV with V packed as image2d. Score is small per-n (TM_SV scalars)
// and stays as a plain buffer; only V benefits from the texel pack.
// Each WI computes a TM_SV_IMG x 8 tile of (M, d) for fixed head_q.
#ifndef TM_SV_IMG
#define TM_SV_IMG 2
#endif
__kernel void sv_matmul_f16_img(
    __global const half *scores,      // [H, M, N_kv] post-softmax fp16
    __read_only image2d_t V_img,      // width=HD_KV/8 texels, height=N_kv
    __global       half *O,           // [M, HD_Q] fp16
    const int M, const int N_kv, const int d,
    const int HD_Q, const int HD_KV, const int gqa) {
  const int x_tex = get_global_id(0);           // 1 texel = 8 halves of d
  const int m0 = get_global_id(1) * TM_SV_IMG;
  const int head_q = get_global_id(2);
  const int head_kv = head_q / gqa;
  const int x0 = x_tex * 8;

  if (m0 >= M || x0 >= d) return;

  float8 acc[TM_SV_IMG];
  #pragma unroll
  for (int i = 0; i < TM_SV_IMG; i++) acc[i] = (float8)(0.0f);

  const sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP |
                        CLK_FILTER_NEAREST;
  const long score_base = (long)head_q * (long)M * (long)N_kv;
  const int v_tx_x = ((head_kv * d) >> 3) + x_tex;

  // Reduction over N_kv. Per n: TM_SV_IMG score scalars + one 8-half V texel.
  for (int n = 0; n < N_kv; n++) {
    const uint4 vv = read_imageui(V_img, smp, (int2)(v_tx_x, n));
    const float8 v_pack = convert_float8(as_half8(vv));

    #pragma unroll
    for (int i = 0; i < TM_SV_IMG; i++) {
      const int m = m0 + i;
      const float sf = (m < M)
                         ? (float)scores[score_base + (long)m * N_kv + n]
                         : 0.0f;
      acc[i] = mad((float8)sf, v_pack, acc[i]);
    }
  }

  #pragma unroll
  for (int i = 0; i < TM_SV_IMG; i++) {
    const int m = m0 + i;
    if (m >= M) continue;
    const half8 oh = convert_half8(acc[i]);
    // Adreno's OpenCL compiler rejects subscript on vector types
    // (`oh[e]`); spill to a half[8] array and index that instead.
    half oh_arr[8];
    vstore8(oh, 0, oh_arr);
    #pragma unroll
    for (int e = 0; e < 8; e++) {
      const int x = x0 + e;
      if (x >= d) continue;
      O[(long)m * HD_Q + head_q * d + x] = oh_arr[e];
    }
  }
}
