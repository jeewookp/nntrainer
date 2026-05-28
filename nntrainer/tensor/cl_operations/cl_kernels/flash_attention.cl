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
//     - load Q[query_row, head_q*d : (head_q+1)*d] into local memory
//     - online maximum / sum / output accumulator (head_dim FP32 reg)
//     - tile-loop over K/V in BLOCK_KV-sized chunks:
//       * cooperatively load K[block], V[block] into LDS
//       * compute s[i] = Q · K[i] / sqrt(d), apply causal mask
//       * online softmax update: m_new = max, alpha = exp(m_old-m_new),
//         sum_new = alpha*sum_old + sum(exp(s - m_new))
//         out_new = alpha*out_old + sum_i (exp(s[i]-m_new) * V[i,:])
//     - write O[query_row, head_q*d : (head_q+1)*d] = out / sum
//
// MEMORY BUDGET (Adreno 830 has 32KB LDS per workgroup):
//   K tile: BLOCK_KV * d * sizeof(half)  e.g. 32*128*2 = 8KB
//   V tile: same                                       = 8KB
//   Q row:  d * sizeof(half)              e.g. 128*2   = 256B
//   sum/max/output FP32 in registers (no LDS for these)
//   Total LDS: ~17KB. Fits.
//
// GROUP SIZING:
//   gws = (1, 1, num_heads_q * M)   — one WG per (head_q, query_row)
//   lws = SUBGROUP_SIZE (64 on Adreno "half")
//   Within a WG, 64 WIs cooperatively (a) load K/V blocks, (b) compute
//   the d-axis reduction for one (query_row, head_q, key_row) score,
//   (c) reduce the BLOCK_KV per-key partial outputs into one row.
//
// STATUS:
//   Step #1 SKELETON. Kernel body is a stub (zero-fills output) so the
//   build + dispatch + env-gate plumbing is exercised end-to-end. The
//   actual flash math lands in a follow-up commit after the host
//   dispatch path + correctness-validation harness are in place.
//
// QWEN3-0.6B SHAPES (for reference; kernel is parameterized):
//   M = step_size (prefill: input length; decode: 1)
//   N_kv = cache_to (running cache fill, ≤ MAX_SEQ_LEN)
//   d = head_dim = 128
//   num_heads_q = 16, num_heads_kv = 8 (GQA = 2)
//   HD_Q = num_heads_q * d = 2048
//   HD_KV = num_heads_kv * d = 1024

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifndef FLASH_BLOCK_KV
#define FLASH_BLOCK_KV 32
#endif

// Skeleton entry. Step #1 of GPU mha migration: prove build + dispatch.
// The signature here is the final-form contract that the host wrapper
// (flash_attention_prefill_f16_cl in attention_kernels.cpp) will call.
// Following step replaces the body with the online-softmax flash loop.
__kernel void flash_attention_prefill_f16_skeleton(
    __global const half *Q,           // [M, HD_Q] fp16, row-major
    __global const half *K,           // [N_kv, HD_KV] fp16, row-major
    __global const half *V,           // [N_kv, HD_KV] fp16, row-major
    __global       half *O,           // [M, HD_Q] fp16, row-major
    const int M,
    const int N_kv,
    const int d,                       // head_dim
    const int HD_Q,                    // num_heads_q * d
    const int HD_KV,                   // num_heads_kv * d
    const int gqa,                     // num_heads_q / num_heads_kv
    const int is_causal,
    const float scale                  // 1 / sqrt(d), precomputed
) {
  // Skeleton: gws = (1, 1, num_heads_q * M). global_id(2) decodes to
  // (head_q, query_row). Local size = SUBGROUP_SIZE.
  const int gid2 = get_global_id(2);
  const int head_q = gid2 / M;
  const int m = gid2 % M;
  const int lid = get_local_id(0);

  // Only WI 0 of each WG zero-fills the output row, to keep the stub
  // deterministic without race. Real kernel will do cooperative writes.
  if (lid != 0) return;
  if (m >= M) return;

  for (int x = 0; x < d; ++x) {
    O[(long)m * HD_Q + head_q * d + x] = (half)0.0f;
  }

  // Silence unused-param warnings during skeleton phase.
  (void)Q; (void)K; (void)V;
  (void)N_kv; (void)HD_KV; (void)gqa; (void)is_causal; (void)scale;
}
