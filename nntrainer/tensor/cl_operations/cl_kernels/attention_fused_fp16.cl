// Fused FlashAttention-style Qwen attention, v5 (per-thread Q-tile).
//
// Prior attempts and what they taught:
//   V0   WG=(128,1,1), TQ=1                1794 ms   correct (baseline)
//   V1   work_group_reduce_add             2397 ms   128x redundant exp
//   V2   V1 + d==0 exp + __local           2464 ms   extra barrier
//   V3a  WG=(128,8,1) sub_group_reduce     1988 ms   WRONG (lane map)
//   V3b  WG=(128,8,1) tree reduce            17 ms   NOOP (launch limit)
//   V3c  WG=(128,4,1) tree reduce          2307 ms   correct, WG-barrier bound
//   V3d  WG=(128,4,1) sub_group             ---     WRONG (lane map)
//   V4   V0 + image1d_buffer K/V            noop   CL_INVALID_VALUE
//                                                   (SVM pool sub-offsets
//                                                    can't be host_ptr)
// NEON reference                             866 ms
//
// The V3 family tried to get FlashAttention's Q-tile reuse by widening
// the workgroup in the Q dimension (WG = (HD, TM, 1)), but:
//   - Multi-wavefront WG barriers are MUCH more expensive on Adreno
//     than V0's single-wavefront ones -> V3c's extra 7 barriers/iter
//     eat the ILP win.
//   - sub_group_reduce would bypass that cost, but Qualcomm's 2D WG
//     sub-group formation is NOT linear-id row-major -> wrong output.
//
// V5 keeps V0's single-wavefront WG (HD, 1, 1) — so barriers stay
// cheap — and moves the Q-tile INTO the thread's private registers
// instead of into extra work-items:
//
//   Each of the 128 threads holds TQ private Q values (one per Q
//   position in the tile), TQ running softmax states, and TQ accs.
//   For every kk the WG loads K[kk] (one 256 B cache line shared
//   across TQ rows) and V[kk] (another), computes TQ partials per
//   thread, parallel-reduces all TQ rows in one pass through
//   __local[TQ][HD] (same 7 barriers as V0 — each barrier handles
//   TQ reductions at once), broadcasts TQ rescale / score_exp
//   scalars through __local[TQ], and updates TQ accs.
//
// Effect on memory traffic:
//   V0 has num_WGs = num_heads_Q * M.  Each WG re-reads the full K /
//   V sequence for its one Q position.
//   V5 has num_WGs = num_heads_Q * ceil(M / TQ).  Each WG still
//   re-reads K / V once, but now one read feeds TQ Q positions —
//   effective K / V traffic drops by TQ.
//   Theoretical memory floor for attention: 56 GB (V0 no-cache) /
//   TQ = 14 GB at TQ=4 -> ~233 ms at 60 GB/s (vs. V0's 933 ms).
//
// TQ=4 is a conservative first step.  Private register pressure at
// TQ=4 is ~7 floats/row x 4 = 28 scalars/thread + scratch ~ well
// inside the 64 KB WG register file.  __local footprint is
// TQ * HD * 4 = 2 KB.  Can push to TQ=8 if the driver copes.

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define HD 128
#define TQ 4

__kernel __attribute__((reqd_work_group_size(HD, 1, 1)))
void attention_fused_fp16(
    __global const half *Q,
    __global const half *K_cache,
    __global const half *V_cache,
    __global half *out,
    const int M,
    const int T,
    const int from,
    const int num_heads_Q,
    const int gqa_size,
    const int is_causal,
    const float scale) {

  const int d      = get_local_id(0);         // 0..HD-1
  const int h      = get_group_id(1);         // 0..num_heads_Q-1
  const int m_base = get_group_id(2) * TQ;    // first Q row of this tile

  const int num_heads_KV = num_heads_Q / gqa_size;
  const int h_kv = h / gqa_size;
  const int W_q  = num_heads_Q  * HD;
  const int W_k  = num_heads_KV * HD;

  // Per-thread state for TQ rows.  q_val is loaded once; the softmax
  // running state is maintained by d==0 of the WG, but we declare
  // the arrays on every thread because the compiler keeps them in
  // regs regardless.
  float q_val      [TQ];
  float acc        [TQ];
  float running_max[TQ];
  float running_sum[TQ];
  int   kk_end_row [TQ];
  bool  row_valid  [TQ];

  int kk_end_tile = 0;

  #pragma unroll
  for (int q = 0; q < TQ; ++q) {
    const int m = m_base + q;
    row_valid  [q] = (m < M);
    q_val      [q] = row_valid[q] ? (float)Q[m * W_q + h * HD + d] : 0.0f;
    acc        [q] = 0.0f;
    running_max[q] = -INFINITY;
    running_sum[q] = 0.0f;
    kk_end_row [q] = row_valid[q]
                     ? ((is_causal != 0) ? (from + m + 1) : T)
                     : 0;
    if (kk_end_row[q] > kk_end_tile) kk_end_tile = kk_end_row[q];
  }

  // Row-parallel reduce scratch + per-row softmax publishers.
  __local float reduce_buf [TQ][HD];
  __local float rescale_l  [TQ];
  __local float score_exp_l[TQ];
  __local float sum_l      [TQ];

  for (int kk = 0; kk < kk_end_tile; ++kk) {
    const float k_val = (float)K_cache[kk * W_k + h_kv * HD + d];

    // Stage TQ partials in __local, one row at a time via q index.
    #pragma unroll
    for (int q = 0; q < TQ; ++q) {
      const bool row_active = row_valid[q] && (kk < kk_end_row[q]);
      reduce_buf[q][d] = row_active ? (q_val[q] * k_val) : 0.0f;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Tree-reduce all TQ rows in parallel.  Each thread updates TQ
    // slots per step; total barrier count is log2(HD) = 7, the same
    // as V0 — just with TQ × more flops per barrier.
    for (int stride = HD / 2; stride > 0; stride >>= 1) {
      if (d < stride) {
        #pragma unroll
        for (int q = 0; q < TQ; ++q) {
          reduce_buf[q][d] += reduce_buf[q][d + stride];
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }

    // d==0 advances the softmax running state for all TQ rows and
    // publishes per-row rescale / score_exp.
    if (d == 0) {
      #pragma unroll
      for (int q = 0; q < TQ; ++q) {
        const bool row_active = row_valid[q] && (kk < kk_end_row[q]);
        if (row_active) {
          const float score    = reduce_buf[q][0] * scale;
          const float prev_max = running_max[q];
          const float new_max  = fmax(prev_max, score);
          const float rescale  = (isinf(prev_max) && prev_max < 0.0f)
                                 ? 0.0f
                                 : exp(prev_max - new_max);
          const float score_exp = exp(score - new_max);
          rescale_l  [q] = rescale;
          score_exp_l[q] = score_exp;
          running_max[q] = new_max;
          running_sum[q] = running_sum[q] * rescale + score_exp;
        } else {
          rescale_l  [q] = 1.0f;
          score_exp_l[q] = 0.0f;
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const float v_val = (float)V_cache[kk * W_k + h_kv * HD + d];

    #pragma unroll
    for (int q = 0; q < TQ; ++q) {
      const bool row_active = row_valid[q] && (kk < kk_end_row[q]);
      if (row_active) {
        acc[q] = acc[q] * rescale_l[q] + score_exp_l[q] * v_val;
      }
    }
  }

  // Publish final running_sum for the normalization step.  Only
  // d==0's private copy is the authoritative one.
  if (d == 0) {
    #pragma unroll
    for (int q = 0; q < TQ; ++q) sum_l[q] = running_sum[q];
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  #pragma unroll
  for (int q = 0; q < TQ; ++q) {
    if (row_valid[q]) {
      const int m = m_base + q;
      const float s = sum_l[q];
      const float inv = (s > 0.0f) ? (1.0f / s) : 0.0f;
      out[m * W_q + h * HD + d] = (half)(acc[q] * inv);
    }
  }
}
