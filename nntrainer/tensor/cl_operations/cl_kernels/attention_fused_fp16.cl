// Fused FlashAttention-style Qwen attention in a single kernel.
// Replaces the NEON triple (compute_kcaches + softmax_triangle +
// compute_fp16vcache_transposed) with one dispatch that keeps the
// online-softmax running stats in registers / __local and never
// materialises the M x T attention matrix.
//
// Layout:
//   Q        : fp16 [M, num_heads_Q,  head_dim]
//   K_cache  : fp16 [T_max, num_heads_KV, head_dim]  (only [0, T) valid)
//   V_cache  : fp16 [T_max, num_heads_KV, head_dim]  (only [0, T) valid)
//   out      : fp16 [M, num_heads_Q, head_dim]
//
// GQA: each Q head h maps to KV head h_kv = h / gqa_size.
// head_dim is fixed to HD (128 on Qwen3-4B); the host gate enforces this.
//
// Dispatch:
//   global = (HD, num_heads_Q, M)    local = (HD, 1, 1)
//   local_id(0) = d  (0..HD-1), one thread per output-dim element
//   group_id(1) = h, group_id(2) = m
//
// Online softmax:
//   running_max <- -inf
//   running_sum <- 0
//   acc[d]      <- 0
//   for kk in [0, kk_end):                    (kk_end = from+m+1 if causal else T)
//     score = (Q[m,h,:] . K[kk,h_kv,:]) / sqrt(HD)
//     new_max   = max(running_max, score)
//     rescale   = exp(running_max - new_max)
//     score_exp = exp(score - new_max)
//     acc[d]       = acc[d] * rescale + score_exp * V[kk,h_kv,d]
//     running_sum  = running_sum * rescale + score_exp
//     running_max  = new_max
//   out[m,h,d] = acc[d] / running_sum

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

  // Cache Q[m,h,d] in register (same Q row for all kk iterations).
  const float q_val = (float)Q[m * W_q + h * HD + d];

  __local float partial[HD];  // scratch for the per-kk dot reduction
  __local float score_broadcast;
  __local float rescale_broadcast;
  __local float score_exp_broadcast;
  __local float running_max_shared;
  __local float running_sum_shared;

  if (d == 0) {
    running_max_shared = -INFINITY;
    running_sum_shared = 0.0f;
  }
  // Per-thread accumulator (one slot of the output vector).
  float acc = 0.0f;

  const int kk_end = (is_causal != 0) ? (from + m + 1) : T;

  for (int kk = 0; kk < kk_end; ++kk) {
    barrier(CLK_LOCAL_MEM_FENCE);

    const float k_val = (float)K_cache[kk * W_k + h_kv * HD + d];
    partial[d] = q_val * k_val;

    barrier(CLK_LOCAL_MEM_FENCE);

    // Tree reduction across HD threads: partial[0] ends up with the dot.
    for (int s = HD / 2; s > 0; s >>= 1) {
      if (d < s) partial[d] += partial[d + s];
      barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (d == 0) {
      const float score = partial[0] * scale;
      const float prev_max = running_max_shared;
      const float new_max = fmax(prev_max, score);
      rescale_broadcast   = (isinf(prev_max) && prev_max < 0.0f) ? 0.0f
                                                                  : exp(prev_max - new_max);
      score_exp_broadcast = exp(score - new_max);
      score_broadcast     = score;
      running_sum_shared  = running_sum_shared * rescale_broadcast
                           + score_exp_broadcast;
      running_max_shared  = new_max;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Update acc for this thread's dim d.
    const float v_val = (float)V_cache[kk * W_k + h_kv * HD + d];
    acc = acc * rescale_broadcast + score_exp_broadcast * v_val;
  }

  barrier(CLK_LOCAL_MEM_FENCE);
  const float denom = running_sum_shared;
  out[m * W_q + h * HD + d] = (half)(acc / denom);
}
