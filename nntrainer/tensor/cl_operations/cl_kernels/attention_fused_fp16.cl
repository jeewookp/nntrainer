// Fused FlashAttention-style Qwen attention, v3 (multi-Q per WG).
//
// V0..V2 timings:
//   V0  tree-reduce + __local broadcast       1794 ms
//   V1  work_group_reduce_add barrier-free    2397 ms (128x redundant exp)
//   V2  work_group_reduce_add + d==0 exp      2464 ms (barrier + serial lane)
// NEON baseline for the same workload         866 ms
//
// Root cause of V0..V2 regression: WG = 128 = 1 wavefront, and the kk
// loop has a serial dependency through running_max/running_sum/acc.  The
// GPU has nothing to schedule while that single wavefront stalls on
// K/V memory, so compute ran at ~3% of ALU peak.  The fix is structural,
// not a tuning knob: pack multiple Q positions into the same WG so
// the wavefront scheduler has multiple wavefronts to round-robin.
//
// V3 layout:
//   WG = (HD, TM, 1) = (128, 8, 1) = 8 wavefronts, 1024 threads.
//   Each WG handles (1 head h) x (TM consecutive Q positions).
//   global = (HD, TM * num_heads_Q, ceil(M / TM)).
//
// ILP recipe:
//   * sub-group size is forced to the native wavefront width (128) via
//     qcom_reqd_sub_group_size("full").  With the default row-major
//     flattening of a 2D WG (local_id(0) varies fastest), lanes
//     0..127 = row m_local=0, lanes 128..255 = row m_local=1, etc.
//   * sub_group_reduce_add therefore reduces within a single row
//     (across d), which is exactly the Q.K dot for that row.
//   * Each row keeps its own running_max / running_sum / acc in private
//     registers.  Only the rescale / score_exp scalars per row are
//     published to __local[TM], broadcast-read by the other 127 lanes
//     of the same row.
//   * K and V are loaded once per kk and reused across all TM rows — a
//     TMx reduction in global memory traffic.
//
// Tail handling:
//   M is not necessarily divisible by TM (Qwen3-4B prefill is M=437).
//   Rows whose m_global >= M must NOT exit early (would break barriers).
//   Instead all threads stay through every barrier; invalid rows skip
//   the d==0 softmax update, publish neutral rescale=1 / score_exp=0,
//   and write nothing at the end.  For causal attention the loop bound
//   is the max kk_end across the valid rows in the WG, so the longest
//   row determines the iteration count.

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable

#define HD 128
#define TM 8

__attribute__((qcom_reqd_sub_group_size("full")))
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

  // Out-of-bounds rows must still participate in every barrier, so we
  // only mask the global memory read.
  const float q_val = row_valid ? (float)Q[m * W_q + h * HD + d] : 0.0f;

  // Per-row softmax state — publishers are d==0 of each row.
  __local float rescale_l  [TM];
  __local float score_exp_l[TM];

  float running_max = -INFINITY;
  float running_sum = 0.0f;
  float acc         = 0.0f;

  // Loop to the MAX kk across all rows in this WG (causal: last valid
  // row has the largest bound; non-causal: every row goes to T).
  int kk_end = T;
  if (is_causal != 0) {
    const int m_last = (m_base + TM - 1 < M) ? (m_base + TM - 1) : (M - 1);
    kk_end = from + m_last + 1;
  }
  // Each row's own termination bound (causal = from + m + 1).
  const int kk_end_row = (is_causal != 0) ? (from + m + 1) : T;

  for (int kk = 0; kk < kk_end; ++kk) {
    const bool row_active = row_valid && (kk < kk_end_row);

    const float k_val = (float)K_cache[kk * W_k + h_kv * HD + d];
    const float partial = row_active ? (q_val * k_val) : 0.0f;

    // Row-wise reduction: sub_group_size == HD == 128 == one wavefront,
    // so this reduces across d for a single m_local.  Barrier-free
    // inside the sub-group.
    const float score = sub_group_reduce_add(partial) * scale;

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
        // Neutral values so the other 127 lanes of this row read
        // something defined, but acc stays unchanged below.
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

  // Publish the final sum for the normalization step.
  __local float sum_l[TM];
  if (d == 0) sum_l[m_local] = running_sum;
  barrier(CLK_LOCAL_MEM_FENCE);

  if (row_valid) {
    const float s = sum_l[m_local];
    // Guard s==0 (empty causal row kk_end_row==0) — shouldn't happen
    // for valid rows with from+m+1 >= 1 but be defensive.
    const float inv = (s > 0.0f) ? (1.0f / s) : 0.0f;
    out[m * W_q + h * HD + d] = (half)(acc * inv);
  }
}
