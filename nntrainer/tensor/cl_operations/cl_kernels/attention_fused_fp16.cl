// Fused FlashAttention-style Qwen attention, v3d (multi-Q per WG,
// sub_group_reduce_add, TM=4).
//
// Timings (Qwen3-4B prefill, 36 layers, M=437, num_heads=32, HD=128):
//   V0  tree reduce, TM=1                            1794 ms  correct
//   V1  work_group_reduce_add barrier-free           2397 ms  correct
//   V2  work_group_reduce_add + d==0 exp             2464 ms  correct
//   V3a sub_group_reduce_add, TM=8                   1988 ms  WRONG
//   V3b tree reduce, TM=8                              17 ms  NOOPED (launch limit)
//   V3c tree reduce, TM=4                            2307 ms  correct
// NEON baseline                                       866 ms
//
// V3c proved the ILP hypothesis isn't free on Adreno: WG-level
// barriers for the per-row tree reduce stall all four wavefronts
// every 7 times per kk iteration, which eats the win from having
// four independent softmax states.  V0's single-wavefront WG keeps
// its 7 barriers/iter cheap because there's nothing to sync across.
//
// V3a got the barrier economics right (sub_group_reduce_add is
// wavefront-internal, so ~0-cost) but produced wrong output — we
// assumed TM=8 was the culprit (driver lane reassignment under
// pressure).  V3d tests whether TM=4 + sub_group_reduce gives BOTH
// correctness and speed:
//   - OpenCL 2.0 spec guarantees sub-groups are built from
//     consecutive work-items in LINEAR local-id order
//     (local_id(0) inner), so lanes 0..127 = m_local=0, lanes
//     128..255 = m_local=1, etc., which is exactly one row per
//     sub-group when sub_group_size = HD = 128.
//   - qcom_reqd_sub_group_size("full") forces the native wavefront
//     width (128 on Adreno 830).
//   - With TM=4, WG = 512 threads = 4 wavefronts — well under the
//     launch limit that tripped V3b.
// Expected: if the OpenCL 2.0 linear-subgroup guarantee holds under
// these params, we get correct output AND barrier-free per-row
// reduces across 4 wavefronts of ILP.  If wrong output: the
// guarantee is violated on this driver and we fall back to V3c.

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable

#define HD 128
#define TM 4

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

  const float q_val = row_valid ? (float)Q[m * W_q + h * HD + d] : 0.0f;

  // Per-row softmax publishers (written by d==0 of each row).
  __local float rescale_l  [TM];
  __local float score_exp_l[TM];

  float running_max = -INFINITY;
  float running_sum = 0.0f;
  float acc         = 0.0f;

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

    // Per-row sub-group reduction: sub_group_size = HD = 128 = one
    // wavefront, and OpenCL 2.0 sub-groups follow linear-id order
    // (local_id(0) inner), so lanes 0..127 = m_local=0 row, lanes
    // 128..255 = m_local=1 row, ..., giving us one sub-group per
    // row.  Barrier-free within the wavefront.
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
