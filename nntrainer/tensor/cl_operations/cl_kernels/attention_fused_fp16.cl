// Fused FlashAttention-style Qwen attention, v2.
//
// V0 used a manual tree reduction (7 barriers/iter) for the Q.K dot and
// a __local broadcast of the softmax state (2 more barriers), so 9
// barriers per kk iteration.  Measured 1794 ms fused on Qwen3-4B prefill.
//
// V1 replaced the reduction with work_group_reduce_add and made every
// thread compute the softmax state from the broadcast score, removing
// all explicit barriers — but every thread re-does the fmax+2*exp math
// 128x redundantly, and the transcendental cost dominated: measured
// 2397 ms (worse than V0).
//
// V2 keeps the barrier-free reduction (work_group_reduce_add) but
// confines the fmax / exp math to d == 0 like V0, broadcasting the
// rescale + score_exp scalars via __local.  That's one reduce + one
// __local round-trip per iter, and no redundant transcendentals:
//
//   partial = q_val * k_val
//   score   = work_group_reduce_add(partial) * scale           // barrier-free
//   if d==0: update running_max, running_sum, publish rescale / score_exp
//   barrier
//   acc = acc * rescale + score_exp * v_val
//
// Layout / dispatch unchanged from V0:
//   Q / K_cache / V_cache / out : fp16, GQA (h_kv = h / gqa_size),
//   HD hard-coded = 128, WG = (HD, 1, 1), global = (HD, num_heads_Q, M).

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define HD 128

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
  const int W_q = num_heads_Q  * HD;
  const int W_k = num_heads_KV * HD;

  const float q_val = (float)Q[m * W_q + h * HD + d];

  // Scalars published by d==0 for the other threads to broadcast-read.
  __local float rescale_l;
  __local float score_exp_l;

  // Running softmax state — lives entirely on d == 0.  Other threads
  // only need the per-iter rescale / score_exp that come through
  // __local, and the final sum which we publish once at the end.
  float running_max = -INFINITY;
  float running_sum = 0.0f;
  float acc = 0.0f;

  const int kk_end = (is_causal != 0) ? (from + m + 1) : T;

  for (int kk = 0; kk < kk_end; ++kk) {
    const float k_val = (float)K_cache[kk * W_k + h_kv * HD + d];
    const float partial = q_val * k_val;

    // Single-intrinsic workgroup reduction; all threads receive the
    // same scalar with no explicit barrier in our kernel (the driver
    // handles any internal sync inside the intrinsic).
    const float score = work_group_reduce_add(partial) * scale;

    if (d == 0) {
      const float prev_max = running_max;
      const float new_max = fmax(prev_max, score);
      const float rescale = (isinf(prev_max) && prev_max < 0.0f)
                            ? 0.0f
                            : exp(prev_max - new_max);
      const float score_exp = exp(score - new_max);
      rescale_l = rescale;
      score_exp_l = score_exp;
      running_max = new_max;
      running_sum = running_sum * rescale + score_exp;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const float rescale   = rescale_l;
    const float score_exp = score_exp_l;
    const float v_val = (float)V_cache[kk * W_k + h_kv * HD + d];
    acc = acc * rescale + score_exp * v_val;
  }

  // Publish the final sum for the normalisation step.
  __local float sum_l;
  if (d == 0) sum_l = running_sum;
  barrier(CLK_LOCAL_MEM_FENCE);
  out[m * W_q + h * HD + d] = (half)(acc / sum_l);
}
