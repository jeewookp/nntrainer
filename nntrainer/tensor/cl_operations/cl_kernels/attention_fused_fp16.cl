// Fused FlashAttention-style Qwen attention in a single kernel.
// Replaces the NEON triple (compute_kcaches + softmax_triangle +
// compute_fp16vcache_transposed) with one dispatch that keeps the
// online-softmax running stats in registers and never materialises
// the M x T attention matrix.
//
// v1: barrier-free inner loop
//   - work_group_reduce_add collapses the per-kk dot product to a
//     single workgroup-wide value (all threads read the same result,
//     no __local scratch, no explicit barrier).
//   - running_max / running_sum are held in private registers of
//     every thread; they're redundant copies of the same scalar, but
//     because each thread computes them from the broadcast score all
//     copies stay in sync without __local writes or barriers.
//   - Previous (V0) version barriered 9+ times per kk iteration for
//     the tree reduction + __local broadcast.  At M=437, T=437,
//     num_heads_Q=32 that was ~30M barriers per prefill layer; with
//     v1 it drops to zero.
//
// Layout unchanged from V0:
//   Q        : fp16 [M, num_heads_Q,  head_dim]
//   K_cache  : fp16 [T_max, num_heads_KV, head_dim]  (only [0, T) valid)
//   V_cache  : fp16 [T_max, num_heads_KV, head_dim]  (only [0, T) valid)
//   out      : fp16 [M, num_heads_Q, head_dim]
// GQA : h_kv = h / gqa_size.
//
// Dispatch:
//   global = (HD, num_heads_Q, M)    local = (HD, 1, 1)
//   local_id(0) = d  (0..HD-1)
//   group_id(1) = h, group_id(2) = m

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

  // Per-thread copies of the running softmax state.  Every thread
  // mutates them in lockstep from the broadcast score, so they stay
  // identical across the workgroup without __local writes or barriers.
  float running_max = -INFINITY;
  float running_sum = 0.0f;
  float acc = 0.0f;

  const int kk_end = (is_causal != 0) ? (from + m + 1) : T;

  for (int kk = 0; kk < kk_end; ++kk) {
    const float k_val = (float)K_cache[kk * W_k + h_kv * HD + d];
    const float partial = q_val * k_val;

    // One-call workgroup reduction: every thread gets the same score.
    const float score = work_group_reduce_add(partial) * scale;

    const float prev_max = running_max;
    const float new_max = fmax(prev_max, score);
    const float rescale = (isinf(prev_max) && prev_max < 0.0f)
                          ? 0.0f
                          : exp(prev_max - new_max);
    const float score_exp = exp(score - new_max);

    const float v_val = (float)V_cache[kk * W_k + h_kv * HD + d];
    acc         = acc         * rescale + score_exp * v_val;
    running_sum = running_sum * rescale + score_exp;
    running_max = new_max;
  }

  out[m * W_q + h * HD + d] = (half)(acc / running_sum);
}
