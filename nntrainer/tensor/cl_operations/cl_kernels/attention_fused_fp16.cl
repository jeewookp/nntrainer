// Step 4b / incremental: step 3b + branchless all-lane softmax.
//
// Step 4 (SGR + V0's if(d==0) + __local rescale/score_exp broadcast +
// barrier) regressed to qk=1414 ms AND produced garbage output.  On
// the Qualcomm driver the combo of sub_group_reduce_add + a
// divergent if(d==0) + a workgroup barrier apparently breaks
// something — either a scheduling interaction with the SGR internal
// fence, or the compiler reordering the __local write across the
// barrier.
//
// V1 tried the all-lane softmax (no barrier, every thread
// computes exp redundantly) with WGR and came in at 2397 ms because
// 128 redundant exp()s ran on a workgroup-reduce-serialized
// wavefront — each exp waited on the next.  With SGR the picture
// inverts: all 128 lanes of the wavefront are already issuing in
// parallel inside a single wavefront, so 128 independent fp32 exp()
// ops cost roughly the same as 1 exp() broadcast — the wavefront
// already has 128 ALUs in flight.
//
// Step 4b therefore:
//   * keeps SGR for the Q.K dot across d (cheap on 1-wavefront WG)
//   * drops the barrier + if(d==0) + __local broadcast entirely
//   * has every lane carry its own running_max / running_sum / acc
//     in private regs.  All lanes get the same score from SGR and
//     compute the same softmax state, so the "per-lane" running
//     state is identical across the wavefront — no need to reconcile.
// Result: correct output (same math as V0) and no barrier cost.
//
// Expected: ~1000-1100 ms (step 3b + a handful of per-lane ALU ops
// for softmax), and correct output.

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

  float running_max = -INFINITY;
  float running_sum = 0.0f;
  float acc         = 0.0f;

  const int kk_end = (is_causal != 0) ? (from + m + 1) : T;

  for (int kk = 0; kk < kk_end; ++kk) {
    const float k_val = (float)K_cache[kk * W_k + h_kv * HD + d];
    const float v_val = (float)V_cache[kk * W_k + h_kv * HD + d];

    const float score = sub_group_reduce_add(q_val * k_val) * scale;

    // Every lane redoes the softmax math on the SAME score.  Free
    // on a 1-wavefront WG: those 128 exp()s issue in parallel.
    const float prev_max = running_max;
    const float new_max  = fmax(prev_max, score);
    const float rescale  = (isinf(prev_max) && prev_max < 0.0f)
                           ? 0.0f
                           : exp(prev_max - new_max);
    const float score_exp = exp(score - new_max);

    acc         = acc         * rescale + score_exp * v_val;
    running_sum = running_sum * rescale + score_exp;
    running_max = new_max;
  }

  // running_sum is identical on every lane; no broadcast needed.
  out[m * W_q + h * HD + d] = (half)(acc / running_sum);
}
