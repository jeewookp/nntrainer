// Step 3b / incremental: reduce via sub_group_reduce_add (1D WG).
//
// Step 3 (work_group_reduce_add) added 1531 ms over step 2 across
// 36 layers.  That's ~85% of V0's 1794 ms concentrated in a single
// intrinsic call.  On a 1-wavefront WG (HD=128, the Adreno native
// wavefront width) sub_group_reduce_add is logically equivalent to
// work_group_reduce_add but stays on a single wavefront the driver
// already has in flight, skipping the workgroup-barrier machinery.
//
// The V3a/V3d sub-group experiments failed at TM>=4 because 2D WGs
// have non-linear lane ordering on Qualcomm.  Here WG is 1D
// ((HD, 1, 1)), so lanes 0..127 ARE threads 0..127 in both
// local-id and sub-group-id spaces — no ambiguity.
//
// If step 3b lands well below step 3 (say ~500-800 ms qk), the
// bottleneck of V0 really is the workgroup-level reduce intrinsic,
// and swapping to sub_group_reduce is the hot fix for attention.
// From there, the V5-style per-thread Q-tile (TQ=4) on top of
// sub_group_reduce would be step 6.

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable

#define HD 128

__attribute__((qcom_reqd_sub_group_size("full")))
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

  const int d = get_local_id(0);
  const int h = get_group_id(1);
  const int m = get_group_id(2);
  if (h >= num_heads_Q || m >= M) return;

  const int num_heads_KV = num_heads_Q / gqa_size;
  const int h_kv = h / gqa_size;
  const int W_q  = num_heads_Q  * HD;
  const int W_k  = num_heads_KV * HD;

  const float q_val = (float)Q[m * W_q + h * HD + d];

  const int kk_end = (is_causal != 0) ? (from + m + 1) : T;

  float acc = 0.0f;
  for (int kk = 0; kk < kk_end; ++kk) {
    const float k_val = (float)K_cache[kk * W_k + h_kv * HD + d];
    const float v_val = (float)V_cache[kk * W_k + h_kv * HD + d];

    const float score = sub_group_reduce_add(q_val * k_val) * scale;

    acc += score * v_val;
  }

  out[m * W_q + h * HD + d] = (half)acc;
}
