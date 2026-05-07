// 2x KK-unrolled M=1 attention. Same online-softmax algorithm and
// dispatch shape as attention_fused_fp16_decode, but the kk loop
// processes two K/V positions per iteration:
//   * Loads K[kk] / K[kk+1] and V[kk] / V[kk+1] together (better
//     cache locality across adjacent KV cache rows; 2x register tile
//     for V values across the softmax update).
//   * Two independent partial dot reductions per iter (subgroup
//     reduce_add pipelines two scores).
//   * Single online-softmax fold absorbs both scores at once:
//     new_max  = max(prev_max, score0, score1)
//     rescale  = exp(prev_max - new_max)
//     exp_i    = exp(score_i - new_max)
//     acc      = acc * rescale + exp_0 * V[kk] + exp_1 * V[kk+1]
//     sum      = sum * rescale + exp_0 + exp_1
//
// Tail (kk_end odd) is handled by a single-kk step at the end.
//
// avg_gpu currently 600us per call on Adreno 830 with kk_end ≈ 470
// (Qwen3-4B context); s2start only 100us, so this kernel is
// compute-bound and unroll savings translate directly to wall time.

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable

#define HD  128
#define WG  64
#define DPT (HD / WG)  // 2

__attribute__((qcom_reqd_sub_group_size("full")))
__kernel __attribute__((reqd_work_group_size(WG, 1, 1)))
void attention_fused_fp16_decode_v2(
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

  const int num_heads_KV = num_heads_Q / gqa_size;
  const int h_kv = h / gqa_size;
  const int W_q  = num_heads_Q  * HD;
  const int W_k  = num_heads_KV * HD;

  const int kk_end = (is_causal != 0) ? (from + 1) : T;

  float q_val[DPT];
  #pragma unroll
  for (int i = 0; i < DPT; ++i) {
    const int dd = d + i * WG;
    q_val[i] = (float)Q[h * HD + dd];
  }

  float running_max = -INFINITY;
  float running_sum = 0.0f;
  float acc[DPT];
  #pragma unroll
  for (int i = 0; i < DPT; ++i) acc[i] = 0.0f;

  const int kk_pairs = kk_end >> 1;
  const int kk_tail  = kk_end & 1;

  int kk = 0;
  for (int p = 0; p < kk_pairs; ++p) {
    float partial0 = 0.0f;
    float partial1 = 0.0f;
    float v_val0[DPT];
    float v_val1[DPT];
    #pragma unroll
    for (int i = 0; i < DPT; ++i) {
      const int dd = d + i * WG;
      const int base0 = kk * W_k + h_kv * HD + dd;
      const int base1 = base0 + W_k;  // = (kk+1) * W_k + h_kv * HD + dd
      const float k_v0 = (float)K_cache[base0];
      const float k_v1 = (float)K_cache[base1];
      v_val0[i] = (float)V_cache[base0];
      v_val1[i] = (float)V_cache[base1];
      partial0 += q_val[i] * k_v0;
      partial1 += q_val[i] * k_v1;
    }
    const float score0 = sub_group_reduce_add(partial0) * scale;
    const float score1 = sub_group_reduce_add(partial1) * scale;

    const float prev_max = running_max;
    const float new_max  = fmax(fmax(prev_max, score0), score1);
    const float rescale  = (isinf(prev_max) && prev_max < 0.0f)
                           ? 0.0f
                           : exp(prev_max - new_max);
    const float exp_0 = exp(score0 - new_max);
    const float exp_1 = exp(score1 - new_max);

    #pragma unroll
    for (int i = 0; i < DPT; ++i) {
      acc[i] = acc[i] * rescale + exp_0 * v_val0[i] + exp_1 * v_val1[i];
    }
    running_sum = running_sum * rescale + exp_0 + exp_1;
    running_max = new_max;
    kk += 2;
  }

  // Tail: single odd kk position.
  if (kk_tail) {
    float partial = 0.0f;
    float v_val[DPT];
    #pragma unroll
    for (int i = 0; i < DPT; ++i) {
      const int dd = d + i * WG;
      const float k_v = (float)K_cache[kk * W_k + h_kv * HD + dd];
      v_val[i]        = (float)V_cache[kk * W_k + h_kv * HD + dd];
      partial += q_val[i] * k_v;
    }
    const float score = sub_group_reduce_add(partial) * scale;

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

  const float inv = (running_sum > 0.0f) ? (1.0f / running_sum) : 0.0f;
  #pragma unroll
  for (int i = 0; i < DPT; ++i) {
    const int dd = d + i * WG;
    out[h * HD + dd] = (half)(acc[i] * inv);
  }
}
