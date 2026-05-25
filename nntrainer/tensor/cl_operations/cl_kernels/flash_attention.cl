// SPDX-License-Identifier: Apache-2.0
// GPU port of MHACoreLayer::gemm_attention (Applications/CausalLM/layers/
// mha_core.cpp), which implements a flash-attention-style tiled compute with
// online softmax + causal masking + GQA. Algorithm mirrors the CPU NEON
// reference; numerical convention matches the FP16-throughout ARM path
// (FP16 storage, FP32 accumulators inside the kernel).
//
// One workgroup of LWS work-items computes one (head_q, q_block) tile:
//   - Each WI owns one Q row in the block
//   - All WIs in the WG cooperatively load K, V blocks into local memory
//   - Each WI maintains private (m_max, l_sum, O_row[d]) running state
//
// Constraints:
//   - d (head_dim) must be a compile-time constant (FA_HEAD_DIM) so O_row
//     and scores fit in private memory
//   - LWS == FA_BQ
//   - FA_BK must divide N_kv? No — bk = min(FA_BK, N_kv - kb) handled at runtime
//   - N_kv >= 1 (no degenerate empty cache)

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifndef FA_LWS
#define FA_LWS 64
#endif
#ifndef FA_BQ
#define FA_BQ 64
#endif
#ifndef FA_BK
#define FA_BK 32
#endif
#ifndef FA_HEAD_DIM
#define FA_HEAD_DIM 128
#endif

__attribute__((reqd_work_group_size(FA_LWS, 1, 1)))
__kernel void flash_attention_prefill_f16(
    __global const half  *Q,           // [M, HD_Q] fp16 row-major
    __global const half  *K,           // [N_kv, HD_KV] fp16 row-major
    __global const half  *V,           // [N_kv, HD_KV] fp16 row-major
    __global       half  *O,           // [M, HD_Q] fp16 row-major
    const int M, const int N_kv,
    const int num_heads_Q, const int num_heads_KV,
    const int causal,                   // 1 = causal lower-triangular mask
    const int cache_from,               // absolute Q row offset for causal mask
    const float inv_sqrt_d) {
  const int tid = get_local_id(0);
  const int wg_id = get_group_id(0);
  // Workgroup tiling: gws = num_heads_Q * num_qb * LWS in dim 0.
  // wg_id = head_q * num_qb + qb_idx.
  const int num_qb = (M + FA_BQ - 1) / FA_BQ;
  const int head_q = wg_id / num_qb;
  const int qb_idx = wg_id % num_qb;
  const int q_row_local = tid;
  const int q_row_global = qb_idx * FA_BQ + q_row_local;
  const int gqa = num_heads_Q / num_heads_KV;
  const int head_kv = head_q / gqa;
  const int HD_Q  = num_heads_Q * FA_HEAD_DIM;
  const int HD_KV = num_heads_KV * FA_HEAD_DIM;
  const int q_abs = cache_from + q_row_global;
  const int q_in_range = (q_row_global < M) ? 1 : 0;

  // Load Q row into private memory (promote to fp32 for the inner FMA).
  // Tail WIs (q_in_range==0) keep Q_row=0 so they participate in collective
  // loads below without affecting any real output.
  float Q_row[FA_HEAD_DIM];
  if (q_in_range) {
    const __global half *q_src =
      Q + (long)q_row_global * HD_Q + head_q * FA_HEAD_DIM;
    for (int x = 0; x < FA_HEAD_DIM; ++x) Q_row[x] = (float)q_src[x];
  } else {
    for (int x = 0; x < FA_HEAD_DIM; ++x) Q_row[x] = 0.0f;
  }

  // Per-WI online softmax state.
  float O_row[FA_HEAD_DIM];
  for (int x = 0; x < FA_HEAD_DIM; ++x) O_row[x] = 0.0f;
  float m_max = -3.0e38f;
  float l_sum = 0.0f;

  // Shared K/V tile (reused: load K, compute QK, then overwrite with V,
  // compute SV). FA_BK * FA_HEAD_DIM fp16s = 32*128*2 = 8 KB per WG.
  __local half lKV[FA_BK * FA_HEAD_DIM];

  for (int kb = 0; kb < N_kv; kb += FA_BK) {
    const int bk = min((int)FA_BK, N_kv - kb);

    // ---- Cooperatively load K[kb..kb+FA_BK, head_kv, :] into local. ----
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int i = tid; i < FA_BK * FA_HEAD_DIM; i += FA_LWS) {
      const int k_off = i / FA_HEAD_DIM;
      const int d_off = i % FA_HEAD_DIM;
      lKV[i] = (k_off < bk)
                 ? K[(long)(kb + k_off) * HD_KV + head_kv * FA_HEAD_DIM + d_off]
                 : (half)0.0f;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // ---- Q_row · K_block^T (private scores[FA_BK], scaled by inv_sqrt_d). ----
    float scores[FA_BK];
    for (int k = 0; k < FA_BK; ++k) {
      float s = 0.0f;
      const __local half *k_row = lKV + k * FA_HEAD_DIM;
      for (int x = 0; x < FA_HEAD_DIM; ++x) s += Q_row[x] * (float)k_row[x];
      scores[k] = s * inv_sqrt_d;
    }

    // Mask out-of-range (k >= bk) and causal-forbidden positions.
    for (int k = 0; k < FA_BK; ++k) {
      const int forbid = (k >= bk) || (causal && (kb + k > q_abs));
      if (forbid) scores[k] = -3.0e38f;
    }

    // ---- Online softmax update. ----
    float block_max = -3.0e38f;
    for (int k = 0; k < FA_BK; ++k) block_max = fmax(block_max, scores[k]);

    const float new_max = fmax(m_max, block_max);
    const float c = (m_max > -3.0e38f) ? exp(m_max - new_max) : 0.0f;
    float block_sum = 0.0f;
    for (int k = 0; k < FA_BK; ++k) {
      scores[k] = (new_max > -3.0e38f) ? exp(scores[k] - new_max) : 0.0f;
      block_sum += scores[k];
    }
    l_sum = l_sum * c + block_sum;
    m_max = new_max;

    // Rescale running output.
    for (int x = 0; x < FA_HEAD_DIM; ++x) O_row[x] *= c;

    // ---- Cooperatively load V[kb..kb+FA_BK, head_kv, :] (overwrites K). ----
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int i = tid; i < FA_BK * FA_HEAD_DIM; i += FA_LWS) {
      const int k_off = i / FA_HEAD_DIM;
      const int d_off = i % FA_HEAD_DIM;
      lKV[i] = (k_off < bk)
                 ? V[(long)(kb + k_off) * HD_KV + head_kv * FA_HEAD_DIM + d_off]
                 : (half)0.0f;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // ---- O_row += scores · V_block. ----
    for (int x = 0; x < FA_HEAD_DIM; ++x) {
      float acc = 0.0f;
      for (int k = 0; k < FA_BK; ++k) {
        acc += scores[k] * (float)lKV[k * FA_HEAD_DIM + x];
      }
      O_row[x] += acc;
    }
  }

  if (!q_in_range) return;
  const float inv_l = (l_sum > 0.0f) ? (1.0f / l_sum) : 0.0f;
  __global half *o_dst = O + (long)q_row_global * HD_Q + head_q * FA_HEAD_DIM;
  for (int x = 0; x < FA_HEAD_DIM; ++x) o_dst[x] = (half)(O_row[x] * inv_l);
}
