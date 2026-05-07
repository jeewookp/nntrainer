// M=1 specialization of attention_fused_fp16. The general kernel
// packs TQ=4 consecutive Q rows per WG to amortize K/V loads during
// prefill (M = seq_len). For decode M=1, three of four packed rows
// are masked off but still consume cycles via branchless conditional
// selects (sub_group_reduce_add + exp(...) execute both branches).
// This kernel drops the TQ dimension so each WG handles exactly one
// Q row -- the only one that exists in decode.
//
// Algorithm: same online-softmax / FlashAttention update as the
// general kernel. Dispatch shape:
//   global = (WG, num_heads_Q, 1)    -- one WG per Q head
//   local  = (WG, 1, 1)
// Q is the single new token's projection (1, num_heads_Q, HD).
// out is its attention result (1, num_heads_Q, HD).
// K_cache/V_cache contain T entries (from + 1 valid for causal).

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable

#define HD  128
#define WG  64
#define DPT (HD / WG)  // 2

__attribute__((qcom_reqd_sub_group_size("full")))
__kernel __attribute__((reqd_work_group_size(WG, 1, 1)))
void attention_fused_fp16_decode(
    __global const half *Q,
    __global const half *K_cache,
    __global const half *V_cache,
    __global half *out,
    const int M,        // = 1 for this kernel; ignored
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

  // For decode the single query is at m=0 within the call. The causal
  // bound for that row is `from + 1` (positions 0..from inclusive),
  // and `from` is the absolute index of the new token.
  const int kk_end = (is_causal != 0) ? (from + 1) : T;

  // Q for this head, persistent across kk loop.
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

  for (int kk = 0; kk < kk_end; ++kk) {
    float partial = 0.0f;
    float v_val[DPT];
    #pragma unroll
    for (int i = 0; i < DPT; ++i) {
      const int dd = d + i * WG;
      const float k_val = (float)K_cache[kk * W_k + h_kv * HD + dd];
      v_val[i]          = (float)V_cache[kk * W_k + h_kv * HD + dd];
      partial += q_val[i] * k_val;
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
