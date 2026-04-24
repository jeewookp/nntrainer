// Step 5 / incremental: WG=(64,1,1) = 1 true Adreno wavefront.
//
// DEBUG PROBE FINDING (commit b5ba0f7d):
//   Adreno 830 reports sub_group_size = 64 under
//   qcom_reqd_sub_group_size("full") — NOT 128 as I had assumed.
//   WG = 128 threads therefore splits into 2 sub-groups of 64, and
//   sub_group_reduce_add only sees half the partials.  That's why
//   step 4 / step 4b / V3a / V3d all produced garbage despite the
//   softmax math being algorithmically correct.  The earlier "2D
//   WG lane-mapping" explanation was wrong — it was the width all
//   along.
//
// Step 5 fixes the premise by shrinking the WG to 64 threads = one
// real wavefront.  HD=128 is still covered; each of the 64 threads
// now handles TWO d values (d_a = d, d_b = d + 64) in its private
// registers.
//
// Per kk:
//   * two K half-loads (d_a and d_b of the same cache line)
//   * two V half-loads
//   * two MACs into a local partial (q_a*k_a + q_b*k_b)
//   * one sub_group_reduce_add across all 64 lanes -> score
//   * all 64 lanes compute the same softmax step (no divergence,
//     no barrier, no __local broadcast)
//   * two acc updates (acc_a, acc_b)
//
// Expected: correct output AND low cost.  The reduce now covers the
// full HD (since partial already sums the two d lanes per thread
// before reducing across threads), softmax is branchless (same
// design as step 4b), and every barrier except the trivial
// wavefront-internal reduce is gone.  Target: step 3b-ish (~1000
// ms) or lower — beats NEON 866 with TQ tiling layered on top
// later.
//
// Dispatch side needs the matching shape change: local = (64,1,1),
// global = (64, num_heads_Q, M).

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable

#define HD 128
#define WG 64            // Adreno 830 native wavefront width.
#define DPT (HD / WG)    // d values per thread = 2

__attribute__((qcom_reqd_sub_group_size("full")))
__kernel __attribute__((reqd_work_group_size(WG, 1, 1)))
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

  const int d = get_local_id(0);            // 0..WG-1 (64)
  const int h = get_group_id(1);
  const int m = get_group_id(2);
  if (h >= num_heads_Q || m >= M) return;

  const int num_heads_KV = num_heads_Q / gqa_size;
  const int h_kv = h / gqa_size;
  const int W_q  = num_heads_Q  * HD;
  const int W_k  = num_heads_KV * HD;

  // Two d values handled by this thread: d_a = d, d_b = d + WG.
  // Q is loaded once outside the kk loop.
  float q_val[DPT];
  #pragma unroll
  for (int i = 0; i < DPT; ++i) {
    const int dd = d + i * WG;
    q_val[i] = (float)Q[m * W_q + h * HD + dd];
  }

  float running_max = -INFINITY;
  float running_sum = 0.0f;
  float acc[DPT] = {0.0f, 0.0f};

  const int kk_end = (is_causal != 0) ? (from + m + 1) : T;

  for (int kk = 0; kk < kk_end; ++kk) {
    // Local partial = sum over the DPT d values this thread owns.
    float partial = 0.0f;
    float k_val[DPT];
    float v_val[DPT];
    #pragma unroll
    for (int i = 0; i < DPT; ++i) {
      const int dd = d + i * WG;
      k_val[i] = (float)K_cache[kk * W_k + h_kv * HD + dd];
      v_val[i] = (float)V_cache[kk * W_k + h_kv * HD + dd];
      partial += q_val[i] * k_val[i];
    }

    // One reduce covers the full HD because the per-thread partial
    // already sums the DPT lanes, and sub_group_reduce_add covers
    // all WG=64 lanes of the true wavefront.
    const float score = sub_group_reduce_add(partial) * scale;

    // Every lane recomputes softmax redundantly (branchless, no
    // barrier, no __local).  Each lane gets the identical score,
    // so running_max / running_sum / rescale / score_exp stay in
    // sync across the wavefront.
    const float prev_max = running_max;
    const float new_max  = fmax(prev_max, score);
    const float rescale  = (isinf(prev_max) && prev_max < 0.0f)
                           ? 0.0f
                           : exp(prev_max - new_max);
    const float score_exp = exp(score - new_max);

    #pragma unroll
    for (int i = 0; i < DPT; ++i) {
      acc[i] = acc[i] * rescale + score_exp * v_val[i];
    }
    running_sum = running_sum * rescale + score_exp;
    running_max = new_max;
  }

  // Write both d lanes per thread.
  #pragma unroll
  for (int i = 0; i < DPT; ++i) {
    const int dd = d + i * WG;
    out[m * W_q + h * HD + dd] = (half)(acc[i] / running_sum);
  }
}
