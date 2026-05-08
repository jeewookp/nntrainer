// Subgroup-half / 2-Q-per-WG. Hybrid of v3's KK-block batched
// softmax fold and v6's 2-Q packing, but with the second Q head
// served by a SECOND SUBGROUP within the same WG instead of by
// extra per-lane state.
//
// Per-CU vs per-WG:
//   v3:  32 WGs x 64 lanes = 32 wavefronts (1 per WG)
//   v6:  16 WGs x 64 lanes = 16 wavefronts (1 per WG, 2x work each)
//        -> half the in-flight wavefronts -> latency-hiding worse
//        -> measured -30% (488us -> 633us).
//   v8:  16 WGs x 128 lanes = 32 wavefronts (2 per WG, 1 head each)
//        -> SAME wavefront count as v3.
//        -> 2 wavefronts per WG run on the same CU, so the K/V
//           cache lines that subgroup 0 loads stay in L1 for
//           subgroup 1 to hit. Fully amortizes the gqa K/V load
//           redundancy without doubling per-lane reg pressure.
//
// Per-lane register footprint (matches v3):
//   q_val[2], acc[2], running_max, running_sum, partials[KK_BLOCK]
//   ~= 10 fp32 stable, ~14 transient peak (during phase 3 V load).
//   No occupancy regression vs v3.
//
// Memory model per WG per kk:
//   1 K row read by lane subgroup_lane in [0,64). Both subgroup 0
//   and subgroup 1 issue the same address; first read populates
//   L1, second read is a cache hit. So per-WG K traffic is half
//   what v3 incurred from two sibling Q-head WGs.
//
// Math is bit-identical to v7 (pre-scaled Q + K-then-V phasing +
// KK_BLOCK=4 batched softmax fold).
//
// Dispatch (caller side):
//   g = { 128, num_heads_Q / 2, 1 }
//   l = { 128,                  1, 1 }
//   Requires gqa_size >= 2 (so two consecutive Q heads share kv-head)
//   and num_heads_Q even.

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable

#define HD       128
#define WG       128
#define SUB      64
#define DPT      (HD / SUB)   // 2
#define KK_BLOCK 4

__attribute__((qcom_reqd_sub_group_size("full")))
__kernel __attribute__((reqd_work_group_size(WG, 1, 1)))
void attention_fused_fp16_decode_v8(
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

  const int d_local  = get_local_id(0);
  const int sub_id   = d_local / SUB;     // 0 or 1
  const int sub_lane = d_local % SUB;     // 0..63 within subgroup
  const int hg       = get_group_id(1);
  const int h0       = hg * 2;            // first Q head in WG
  const int h        = h0 + sub_id;       // this subgroup's Q head

  const int num_heads_KV = num_heads_Q / gqa_size;
  const int h_kv = h / gqa_size;          // both subgroups share kv-head
  const int W_k  = num_heads_KV * HD;

  const int kk_end = (is_causal != 0) ? (from + 1) : T;

  // Pre-scaled Q so per-kk `* scale` mul drops out.
  float q_val[DPT];
  #pragma unroll
  for (int i = 0; i < DPT; ++i) {
    const int dd = sub_lane + i * SUB;
    q_val[i] = (float)Q[h * HD + dd] * scale;
  }

  float running_max = -INFINITY;
  float running_sum = 0.0f;
  float acc[DPT];
  #pragma unroll
  for (int i = 0; i < DPT; ++i) acc[i] = 0.0f;

  const int kk_blocks = kk_end / KK_BLOCK;
  const int kk_tail   = kk_end & (KK_BLOCK - 1);

  int kk = 0;
  for (int p = 0; p < kk_blocks; ++p) {
    // ----- PHASE 1: K + partial compute -----
    float partials[KK_BLOCK];
    #pragma unroll
    for (int b = 0; b < KK_BLOCK; ++b) {
      float local_p = 0.0f;
      #pragma unroll
      for (int i = 0; i < DPT; ++i) {
        const int dd   = sub_lane + i * SUB;
        const int base = (kk + b) * W_k + h_kv * HD + dd;
        local_p += q_val[i] * (float)K_cache[base];
      }
      partials[b] = sub_group_reduce_add(local_p);
    }

    // ----- PHASE 2: online softmax fold -----
    const float prev_max = running_max;
    float bm = partials[0];
    #pragma unroll
    for (int b = 1; b < KK_BLOCK; ++b) bm = fmax(bm, partials[b]);
    const float new_max = fmax(prev_max, bm);
    const float rescale = (isinf(prev_max) && prev_max < 0.0f)
                            ? 0.0f
                            : native_exp(prev_max - new_max);
    float bsum = 0.0f;
    #pragma unroll
    for (int b = 0; b < KK_BLOCK; ++b) {
      partials[b] = native_exp(partials[b] - new_max);
      bsum += partials[b];
    }

    #pragma unroll
    for (int i = 0; i < DPT; ++i) acc[i] = acc[i] * rescale;

    // ----- PHASE 3: V + acc update -----
    #pragma unroll
    for (int b = 0; b < KK_BLOCK; ++b) {
      const float pb = partials[b];
      #pragma unroll
      for (int i = 0; i < DPT; ++i) {
        const int dd   = sub_lane + i * SUB;
        const int base = (kk + b) * W_k + h_kv * HD + dd;
        acc[i] += pb * (float)V_cache[base];
      }
    }

    running_sum = running_sum * rescale + bsum;
    running_max = new_max;
    kk += KK_BLOCK;
  }

  // Tail: 0..3 leftover kk positions.
  for (int t = 0; t < kk_tail; ++t) {
    float local_p = 0.0f;
    float v_val[DPT];
    #pragma unroll
    for (int i = 0; i < DPT; ++i) {
      const int dd   = sub_lane + i * SUB;
      const int base = (kk + t) * W_k + h_kv * HD + dd;
      local_p += q_val[i] * (float)K_cache[base];
      v_val[i] = (float)V_cache[base];
    }
    const float score = sub_group_reduce_add(local_p);

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
    const int dd = sub_lane + i * SUB;
    out[h * HD + dd] = (half)(acc[i] * inv);
  }
}
