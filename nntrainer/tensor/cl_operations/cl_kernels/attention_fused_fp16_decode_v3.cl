// 4x KK-unrolled M=1 attention with batched online softmax.
//
// v3 extends v2's 2-way KK-unroll to 4-way: each iter loads K and V
// for 4 consecutive kk positions, computes 4 dot products and
// reductions, then folds all 4 scores into the online softmax with
// a single rescale (vs 1 rescale per pair in v2).
//
// Per kk-equivalent op count:
//   v2  (KK=2):  1 reduction, 0.5 rescale-exp, 1.0 score-exp = 1.5 exps/kk
//   v3  (KK=4):  1 reduction, 0.25 rescale-exp, 1.0 score-exp = 1.25 exps/kk
//   -> v3 saves ~17% exp ops.
//
// native_exp() replaces exp() throughout. Adreno's hardware
// transcendental unit gives single-cycle native_exp for the typical
// score magnitudes here (well within fp16 range bounds), at the
// cost of ~1 ULP relative error.
//
// Register budget per lane (fp32):
//   q_val[DPT]                                = 2
//   acc[DPT]                                  = 2
//   running_max, running_sum                  = 2
//   k_v[4][DPT], v_v[4][DPT]                  = 16
//   partials[4] (reused for scores then exps) = 4
//   transients (rescale, max)                 = ~3
//   total ~29 fp32 / lane.
// v2 lives at ~10. The added pressure may cost some occupancy on
// Adreno but should fit comfortably in the per-lane register file.

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable

#define HD  128
#define WG  64
#define DPT (HD / WG)  // 2
#define KK_BLOCK 4

__attribute__((qcom_reqd_sub_group_size("full")))
__kernel __attribute__((reqd_work_group_size(WG, 1, 1)))
void attention_fused_fp16_decode_v3(
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
  const int W_k = num_heads_KV * HD;

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

  const int kk_blocks = kk_end / KK_BLOCK;
  const int kk_tail = kk_end & (KK_BLOCK - 1);

  int kk = 0;
  for (int p = 0; p < kk_blocks; ++p) {
    // Load K and V for the 4 kk positions in this block.
    float k_v[KK_BLOCK][DPT];
    float v_v[KK_BLOCK][DPT];
    #pragma unroll
    for (int b = 0; b < KK_BLOCK; ++b) {
      #pragma unroll
      for (int i = 0; i < DPT; ++i) {
        const int dd = d + i * WG;
        const int base = (kk + b) * W_k + h_kv * HD + dd;
        k_v[b][i] = (float)K_cache[base];
        v_v[b][i] = (float)V_cache[base];
      }
    }

    // Compute the 4 partials and reduce to scores. The reductions
    // are issued back-to-back; Adreno's wavefront subgroup_reduce
    // pipelines them through the cluster reduce unit.
    float partials[KK_BLOCK];
    #pragma unroll
    for (int b = 0; b < KK_BLOCK; ++b) {
      float partial = 0.0f;
      #pragma unroll
      for (int i = 0; i < DPT; ++i) partial += q_val[i] * k_v[b][i];
      partials[b] = sub_group_reduce_add(partial) * scale;
    }

    // Block max and online softmax fold (single rescale for the
    // whole block).
    const float prev_max = running_max;
    float block_max = partials[0];
    #pragma unroll
    for (int b = 1; b < KK_BLOCK; ++b) block_max = fmax(block_max, partials[b]);
    const float new_max = fmax(prev_max, block_max);
    const float rescale = (isinf(prev_max) && prev_max < 0.0f)
                            ? 0.0f
                            : native_exp(prev_max - new_max);

    // Compute exps in place (reuse partials[] storage).
    float block_sum = 0.0f;
    #pragma unroll
    for (int b = 0; b < KK_BLOCK; ++b) {
      partials[b] = native_exp(partials[b] - new_max);
      block_sum += partials[b];
    }

    // Update accumulators: single rescale, then add weighted V's.
    #pragma unroll
    for (int i = 0; i < DPT; ++i) {
      float a = acc[i] * rescale;
      #pragma unroll
      for (int b = 0; b < KK_BLOCK; ++b) {
        a += partials[b] * v_v[b][i];
      }
      acc[i] = a;
    }
    running_sum = running_sum * rescale + block_sum;
    running_max = new_max;

    kk += KK_BLOCK;
  }

  // Tail: 0..3 leftover kk positions. Process individually.
  for (int t = 0; t < kk_tail; ++t) {
    float partial = 0.0f;
    float v_val[DPT];
    #pragma unroll
    for (int i = 0; i < DPT; ++i) {
      const int dd = d + i * WG;
      const int base = (kk + t) * W_k + h_kv * HD + dd;
      const float k_v_t = (float)K_cache[base];
      v_val[i]          = (float)V_cache[base];
      partial += q_val[i] * k_v_t;
    }
    const float score = sub_group_reduce_add(partial) * scale;
    const float prev_max = running_max;
    const float new_max  = fmax(prev_max, score);
    const float rescale  = (isinf(prev_max) && prev_max < 0.0f)
                             ? 0.0f
                             : native_exp(prev_max - new_max);
    const float score_exp = native_exp(score - new_max);
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
