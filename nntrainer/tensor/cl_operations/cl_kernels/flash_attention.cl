// SPDX-License-Identifier: Apache-2.0
// Flash-attention-style single-kernel prefill / decode attention for
// Adreno (paper ML Drift §3.6 fusion + Dao et al. 2022 online softmax).
//
// REPLACES: the three-kernel two_conv_attention.cl path which is
// (a) slower than CPU NEON on Adreno 830 due to VGPR spill (each WI's
// TM_QK*TN_QK accumulator floods the register file), and (b) routes
// the full scores[H, M, N_kv] tensor through global memory between
// kernels, which is ~6.7 MB bandwidth per prefill that we don't need.
//
// PIPELINE (one kernel):
//   For each (head_q, query_row):
//     - online maximum / sum / output accumulator (head_dim FP32 reg)
//     - serial loop over K/V rows n=0..N_kv-1 (causal: n<=m):
//       * s = scale * dot(Q[m,head_q], K[n,head_kv])   (fp32 accum)
//       * m_new = max(m_i, s); alpha = exp(m_i - m_new); p = exp(s - m_new)
//       * l_i = alpha*l_i + p
//       * acc[:] = alpha*acc[:] + p*V[n,head_kv,:]
//       * m_i = m_new
//     - write O[query_row, head_q*d : (head_q+1)*d] = acc / l_i
//
// CORRECTNESS-FIRST DESIGN (Step #2):
//   ONE work-item per (head_q, query_row). The full d=128 fp32 output
//   accumulator (512 B) + Q row (256 B fp16) live in private memory.
//   No LDS, no inter-WI reduction, no scores DRAM materialization — the
//   whole point (removing the [H,M,N_kv] global traffic) is achieved by
//   the single-WI serial form. Tiling/LDS cooperation is a follow-up.
//   gws = (num_heads_q * M,); lws chosen by host (small, e.g. 64).
//
// K-LAYOUT NOTE: the gpu_native NNTR_OHWI_IMG=0 path stores cache_k_svm
//   in OHWI form  K[head_kv * S_max * d + n * d + x]  (qk_matmul_f16_ohwi
//   layout, NOT the pure concat [N_kv, HD_KV]). V stays concat
//   V[n * HD_KV + head_kv * d + x] (sv_matmul_f16 layout). To feed the
//   EXACT SAME buffers as the 3-kernel _ohwi_cl fallback, this kernel
//   takes a k_stride param: if k_stride > 0, K is OHWI with that S_max
//   row-stride; if k_stride <= 0, K is pure concat (HD_KV stride). Q and
//   O are always concat. This keeps the flash path bit-comparable to the
//   baseline it replaces.
//
// QWEN3-0.6B SHAPES (for reference; kernel is parameterized):
//   M = step_size (prefill: input length; decode: 1)
//   N_kv = cache_to (running cache fill, <= MAX_SEQ_LEN)
//   d = head_dim = 128
//   num_heads_q = 16, num_heads_kv = 8 (GQA = 2)
//   HD_Q = num_heads_q * d = 2048
//   HD_KV = num_heads_kv * d = 1024

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifndef FLASH_BLOCK_KV
#define FLASH_BLOCK_KV 32
#endif

#ifndef FLASH_MAX_D
#define FLASH_MAX_D 128
#endif

// Single-WI-per-(head_q, query_row) flash attention prefill. Online
// softmax (Dao et al.) so scores are never materialized to global memory.
// Signature matches the host wrapper flash_attention_prefill_f16_cl.
__kernel void flash_attention_prefill_f16_skeleton(
    __global const half *Q,           // [M, HD_Q] fp16, row-major (concat)
    __global const half *K,           // OHWI [H_kv,S_max,d] or concat [N_kv,HD_KV]
    __global const half *V,           // [N_kv, HD_KV] fp16, row-major (concat)
    __global       half *O,           // [M, HD_Q] fp16, row-major (concat)
    const int M,
    const int N_kv,
    const int d,                       // head_dim
    const int HD_Q,                    // num_heads_q * d
    const int HD_KV,                   // num_heads_kv * d
    const int gqa,                     // num_heads_q / num_heads_kv
    const int is_causal,
    const float scale,                 // 1 / sqrt(d), precomputed
    const int k_stride                 // >0: K OHWI S_max row-stride; <=0: concat
) {
  const int gid = get_global_id(0);     // decodes to (head_q, query_row)
  const int head_q = gid / M;
  const int m = gid % M;
  if (m >= M) return;
  const int total = (HD_Q / d) * M;     // num_heads_q * M
  if (gid >= total) return;

  const int head_kv = head_q / gqa;

  // K base offset for this (head_kv). OHWI: head_kv*S_max*d + n*d + x.
  // concat: n*HD_KV + head_kv*d + x. We fold the per-n stride into k_row.
  const long k_head_base =
      (k_stride > 0) ? ((long)head_kv * (long)k_stride * (long)d)
                     : ((long)head_kv * (long)d);
  const long k_row_stride = (k_stride > 0) ? (long)d : (long)HD_KV;

  const long q_base = (long)m * HD_Q + (long)head_q * d;

  // Load this query row into private fp32 registers.
  float q_row[FLASH_MAX_D];
  for (int x = 0; x < d; ++x)
    q_row[x] = (float)Q[q_base + x];

  // Online-softmax state.
  float m_i = -INFINITY;
  float l_i = 0.0f;
  float acc[FLASH_MAX_D];
  for (int x = 0; x < d; ++x)
    acc[x] = 0.0f;

  // Causal: key n is masked when is_causal && n > m. So the valid range
  // is n in [0, n_last] where n_last = is_causal ? min(N_kv-1, m) : N_kv-1.
  const int n_last = is_causal ? min(N_kv - 1, m) : (N_kv - 1);

  for (int n = 0; n <= n_last; ++n) {
    const long k_base = k_head_base + (long)n * k_row_stride;
    const long v_base = (long)n * HD_KV + (long)head_kv * d;

    // s = scale * dot(Q[m,head_q], K[n,head_kv]) in fp32.
    float dot = 0.0f;
    for (int x = 0; x < d; ++x)
      dot += q_row[x] * (float)K[k_base + x];
#ifdef FLASH_FP16_SCORE
    // Match the 3-kernel baseline, which writes scores as fp16 before
    // softmax (qk_matmul_f16 stores (half)(acc*scale)). Truncating here
    // makes the flash path bit-comparable to that baseline.
    const float s = (float)((half)(scale * dot));
#else
    const float s = scale * dot;
#endif

    // Online softmax update (Dao et al.).
    const float m_new = fmax(m_i, s);
    const float alpha = exp(m_i - m_new);   // m_i==-inf, m_new finite => 0
    const float p = exp(s - m_new);
    l_i = alpha * l_i + p;
    for (int x = 0; x < d; ++x)
      acc[x] = alpha * acc[x] + p * (float)V[v_base + x];
    m_i = m_new;
  }

  // Normalize and write out. l_i == 0 only when no key was attended
  // (e.g. causal row with N_kv==0); guard to avoid NaN.
  const float inv = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;
  const long o_base = (long)m * HD_Q + (long)head_q * d;
  for (int x = 0; x < d; ++x)
    O[o_base + x] = (half)(acc[x] * inv);
}
