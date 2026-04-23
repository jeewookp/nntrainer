// Fused FlashAttention-style Qwen attention, v3b (multi-Q per WG,
// manual per-row tree reduce).
//
// Timings (Qwen3-4B prefill, 36 layers, M=437, num_heads=32, HD=128):
//   V0  tree-reduce + __local broadcast       1794 ms
//   V1  work_group_reduce_add barrier-free    2397 ms
//   V2  work_group_reduce_add + d==0 exp      2464 ms
//   V3a sub_group_reduce_add + TM=8           wrong output (lane-map
//                                             assumption bad) — 1988 ms
// NEON baseline                                866 ms
//
// V3a tried to exploit ILP by packing TM=8 Q positions into a WG so
// the scheduler had 8 wavefronts to round-robin, and used
// sub_group_reduce_add with qcom_reqd_sub_group_size("full") to reduce
// each row independently. The output came back garbage because the
// Adreno driver's 2D lane-to-sub-group mapping is NOT simply
// row-major: lanes 0..127 did not all correspond to m_local=0 of a
// single row, so the per-row dot product mixed rows.
//
// V3b keeps the same dispatch (WG = (HD, TM, 1)) and the same ILP
// recipe, but replaces the sub_group reduce with an EXPLICIT per-row
// tree reduce through __local[TM][HD]. Index arithmetic is by
// (m_local, d) so correctness is independent of any lane-ordering
// assumption. The cost is log2(HD) = 7 barriers per kk iteration, but
// the barriers are now cheap relative to ILP-hidden memory latency
// (they only sync 1024 threads within a WG, not across WGs; the
// scheduler can still overlap the 8 wavefronts' work between them).

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define HD 128
// TM is the number of Q positions packed into a single WG. TM=8
// (1024 threads / WG) returned zeros / stale memory on Adreno 830 —
// likely hit a driver limit (max WG size, register pressure, or
// __local budget) and silently nooped. TM=4 (512 threads, 2KB
// __local) keeps the ILP win from multi-wavefront scheduling while
// staying well under any Adreno 830 launch threshold.
#define TM 4

__kernel __attribute__((reqd_work_group_size(HD, TM, 1)))
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

  const int d       = get_local_id(0);     // 0..HD-1
  const int m_local = get_local_id(1);     // 0..TM-1
  const int h       = get_group_id(1);     // 0..num_heads_Q-1
  const int m_base  = get_group_id(2) * TM;
  const int m       = m_base + m_local;
  const bool row_valid = (m < M);

  const int num_heads_KV = num_heads_Q / gqa_size;
  const int h_kv = h / gqa_size;
  const int W_q  = num_heads_Q  * HD;
  const int W_k  = num_heads_KV * HD;

  const float q_val = row_valid ? (float)Q[m * W_q + h * HD + d] : 0.0f;

  // Per-row tree-reduce scratch. [m_local][d].
  __local float reduce_buf[TM][HD];

  // Per-row softmax publishers (written by d==0 of each row).
  __local float rescale_l  [TM];
  __local float score_exp_l[TM];

  float running_max = -INFINITY;
  float running_sum = 0.0f;
  float acc         = 0.0f;

  // Causal: each row's own kk_end is from+m+1; the WG loops to the
  // max so every row reaches the barriers inside the loop.
  int kk_end = T;
  if (is_causal != 0) {
    const int m_last = (m_base + TM - 1 < M) ? (m_base + TM - 1) : (M - 1);
    kk_end = from + m_last + 1;
  }
  const int kk_end_row = (is_causal != 0) ? (from + m + 1) : T;

  for (int kk = 0; kk < kk_end; ++kk) {
    const bool row_active = row_valid && (kk < kk_end_row);

    const float k_val = (float)K_cache[kk * W_k + h_kv * HD + d];
    const float partial = row_active ? (q_val * k_val) : 0.0f;

    // Stage partial in local memory; one per (m_local, d) slot.
    reduce_buf[m_local][d] = partial;
    barrier(CLK_LOCAL_MEM_FENCE);

    // In-row tree reduce across d. Bank conflicts are 2-way
    // (stride=64 hits same bank as stride=0 under 32-bank layout),
    // acceptable for this working-set size.
    for (int stride = HD / 2; stride > 0; stride >>= 1) {
      if (d < stride) {
        reduce_buf[m_local][d] += reduce_buf[m_local][d + stride];
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }

    const float score = reduce_buf[m_local][0] * scale;

    if (d == 0) {
      if (row_active) {
        const float prev_max = running_max;
        const float new_max  = fmax(prev_max, score);
        const float rescale  = (isinf(prev_max) && prev_max < 0.0f)
                               ? 0.0f
                               : exp(prev_max - new_max);
        const float score_exp = exp(score - new_max);
        rescale_l  [m_local] = rescale;
        score_exp_l[m_local] = score_exp;
        running_max = new_max;
        running_sum = running_sum * rescale + score_exp;
      } else {
        rescale_l  [m_local] = 1.0f;
        score_exp_l[m_local] = 0.0f;
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const float rescale   = rescale_l  [m_local];
    const float score_exp = score_exp_l[m_local];
    const float v_val = (float)V_cache[kk * W_k + h_kv * HD + d];
    if (row_active) {
      acc = acc * rescale + score_exp * v_val;
    }
  }

  __local float sum_l[TM];
  if (d == 0) sum_l[m_local] = running_sum;
  barrier(CLK_LOCAL_MEM_FENCE);

  if (row_valid) {
    const float s = sum_l[m_local];
    const float inv = (s > 0.0f) ? (1.0f / s) : 0.0f;
    out[m * W_q + h * HD + d] = (half)(acc * inv);
  }
}
