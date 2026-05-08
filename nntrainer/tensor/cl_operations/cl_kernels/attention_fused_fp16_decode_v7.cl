// v3 layout (1 Q head per WG, KK_BLOCK=4, native_exp batch fold)
// with EXPLICIT K-then-V load phasing to halve peak register
// pressure per lane.
//
// Why this is the actual lever:
//   * v3 attn compute = 261us per WG, identical across v3, v5, v6
//     -> the *per-WG kernel time* is what bottlenecks, not the WG
//     count or memory-amortization tricks (v6 proved this regress).
//   * v3 keeps k_v[4][2] AND v_v[4][2] alive across the whole
//     KK_BLOCK iteration. Reg footprint ~29 fp32 / lane.
//   * Adreno per-lane register file caps wavefronts-per-CU at 2-3
//     when reg pressure is ~29. v7 phases K loads and V loads into
//     non-overlapping live ranges so peak reg footprint drops to
//     ~14-15 fp32 / lane -> 4+ wavefronts/CU possible -> better
//     latency hiding -> lower wall.
//
// Per-iter phases:
//   PHASE 1 (K + partials):  load k_v_b for each b, compute partial,
//                            drop k_v_b. Live: q_val, acc, partials[].
//   PHASE 2 (softmax fold):  block_max + rescale + exp(partials).
//                            Live: q_val, acc, partials[] (post-exp).
//   PHASE 3 (V + acc):       load v_v_b for each b, acc += p*v.
//                            Live: q_val, acc.
//
// Stable: q_val[2] + acc[2] + (max,sum) = 6 fp32
// Phase 1 peak: 6 + partials[4] + k_v_b[2] + p = ~13
// Phase 2 peak: 6 + partials[4] + transients = ~15
// Phase 3 peak: 6 + partials[4] + v_v_b[2] = ~12
// -> ~half of v3's 29.
//
// Algorithm and outputs are bit-equivalent to v3 (same fmadd order,
// same native_exp call sites, same KK_BLOCK=4 batch fold). The only
// difference is that K and V live ranges no longer overlap.

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable

#define HD       128
#define WG       64
#define DPT      (HD / WG)   // 2
#define KK_BLOCK 4

__attribute__((qcom_reqd_sub_group_size("full")))
__kernel __attribute__((reqd_work_group_size(WG, 1, 1)))
void attention_fused_fp16_decode_v7(
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
  const int W_k  = num_heads_KV * HD;

  const int kk_end = (is_causal != 0) ? (from + 1) : T;

  // Pre-scale Q so the per-kk `* scale` mul drops out.
  float q_val[DPT];
  #pragma unroll
  for (int i = 0; i < DPT; ++i) {
    const int dd = d + i * WG;
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
    // ----- PHASE 1: K + partial compute (k_v_b dies each iter) -----
    float partials[KK_BLOCK];
    #pragma unroll
    for (int b = 0; b < KK_BLOCK; ++b) {
      float local_p = 0.0f;
      // K loaded scalar, used immediately, no array kept alive.
      #pragma unroll
      for (int i = 0; i < DPT; ++i) {
        const int dd   = d + i * WG;
        const int base = (kk + b) * W_k + h_kv * HD + dd;
        local_p += q_val[i] * (float)K_cache[base];
      }
      partials[b] = sub_group_reduce_add(local_p);
    }

    // ----- PHASE 2: online softmax fold (no extra live regs) -----
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

    // Pre-rescale acc so the V-update phase needs only `+= p*v`.
    #pragma unroll
    for (int i = 0; i < DPT; ++i) acc[i] = acc[i] * rescale;

    // ----- PHASE 3: V + acc update (v_v_b dies each iter) -----
    #pragma unroll
    for (int b = 0; b < KK_BLOCK; ++b) {
      const float pb = partials[b];
      #pragma unroll
      for (int i = 0; i < DPT; ++i) {
        const int dd   = d + i * WG;
        const int base = (kk + b) * W_k + h_kv * HD + dd;
        acc[i] += pb * (float)V_cache[base];
      }
    }

    running_sum = running_sum * rescale + bsum;
    running_max = new_max;
    kk += KK_BLOCK;
  }

  // Tail: 0..3 leftover kk positions. Single per-kk fold (low reg
  // pressure already, no need to phase).
  for (int t = 0; t < kk_tail; ++t) {
    float local_p = 0.0f;
    float v_val[DPT];
    #pragma unroll
    for (int i = 0; i < DPT; ++i) {
      const int dd   = d + i * WG;
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
    const int dd = d + i * WG;
    out[h * HD + dd] = (half)(acc[i] * inv);
  }
}
