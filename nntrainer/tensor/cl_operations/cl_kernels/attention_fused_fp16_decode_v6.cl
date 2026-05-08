// 2 Q heads per WG with shared K/V loads. Builds on v3's KK-block
// online-softmax fold; halves K/V global-memory traffic by having
// the same WG service two consecutive Q heads that fall in the same
// gqa group.
//
// Why this is the next lever:
//   * v3 attn compute = 261us, BW estimate = 50us (75 GB/s ceil),
//     so ~210us is latency/issue overhead -- not compute-bound.
//   * gqa_size=4 in Qwen3-4B means 4 Q heads share each kv-head's
//     K/V. v3 uses 1 WG per Q head (32 WGs) -> 4x redundant K/V
//     loads per kv-head. v6 packs 2 Q heads per WG, halving the
//     redundancy.
//   * Adreno's L2 caches the kv-head's K rows when 2 sibling Q-head
//     WGs run on adjacent CUs, but in-WG sharing eliminates the
//     duplicate fetch entirely.
//   * Phase K-attn earlier tried 4-Q-per-WG (full GQA) and tanked
//     occupancy (32 fp32/lane). v6 stays at 2-Q so reg pressure
//     stays at ~24 fp32/lane (LOWER than v3's 29).
//
// Per-lane register footprint (fp32):
//   q_val0[2], q_val1[2]                = 4
//   acc0[2],   acc1[2]                  = 4
//   running_{max,sum}_{0,1}             = 4
//   partials0[KK_BLOCK], partials1[..]  = 8
//   k_v_b[2] / v_v_b[2] (transient)     = 2 peak
//   p0, p1, fold transients             = ~3
//   total peak                          ~24-26 fp32
//
// Memory model per WG per kk:
//   K row: 1 vector load (256 bytes) reused by 2 Q heads
//   V row: 1 vector load reused by 2 Q heads
//   (vs v3: each WG loads its own copy; with 4 Q heads in same kv
//    group there were 4 redundant copies in flight, now 2)
//
// Pre-scaling: q_val *= scale at load time so the inner reduce can
// drop the per-kk `* scale` mul. Saves kk_end / KK_BLOCK muls per
// Q head; tiny but free.
//
// K-then-V phasing: load K_block, compute both Q heads' partials,
// drop K_block; later load V_block to update accs. Keeps register
// peak below the all-at-once load+compute pattern.

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable

#define HD       128
#define WG       64
#define DPT      (HD / WG)   // 2
#define KK_BLOCK 4

__attribute__((qcom_reqd_sub_group_size("full")))
__kernel __attribute__((reqd_work_group_size(WG, 1, 1)))
void attention_fused_fp16_decode_v6(
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

  const int d  = get_local_id(0);
  const int hg = get_group_id(1);   // 0 .. (num_heads_Q/2 - 1)
  const int h0 = hg * 2;            // first Q head in this WG
  const int h1 = h0 + 1;            // second Q head (same kv group when gqa>=2)

  const int num_heads_KV = num_heads_Q / gqa_size;
  const int h_kv = h0 / gqa_size;   // h0 and h0+1 share kv-head (gqa>=2)
  const int W_k  = num_heads_KV * HD;

  const int kk_end = (is_causal != 0) ? (from + 1) : T;

  // Pre-scaled Q: drop per-kk `* scale`.
  float q_val0[DPT];
  float q_val1[DPT];
  #pragma unroll
  for (int i = 0; i < DPT; ++i) {
    const int dd = d + i * WG;
    q_val0[i] = (float)Q[h0 * HD + dd] * scale;
    q_val1[i] = (float)Q[h1 * HD + dd] * scale;
  }

  float running_max0 = -INFINITY, running_sum0 = 0.0f;
  float running_max1 = -INFINITY, running_sum1 = 0.0f;
  float acc0[DPT];
  float acc1[DPT];
  #pragma unroll
  for (int i = 0; i < DPT; ++i) { acc0[i] = 0.0f; acc1[i] = 0.0f; }

  const int kk_blocks = kk_end / KK_BLOCK;
  const int kk_tail   = kk_end & (KK_BLOCK - 1);

  int kk = 0;
  for (int p = 0; p < kk_blocks; ++p) {
    // ----- Phase 1: load K, compute 4*2 partials, drop K -----
    float partials0[KK_BLOCK];
    float partials1[KK_BLOCK];
    #pragma unroll
    for (int b = 0; b < KK_BLOCK; ++b) {
      float k_v_b[DPT];
      #pragma unroll
      for (int i = 0; i < DPT; ++i) {
        const int dd   = d + i * WG;
        const int base = (kk + b) * W_k + h_kv * HD + dd;
        k_v_b[i] = (float)K_cache[base];
      }
      float p0 = 0.0f, p1 = 0.0f;
      #pragma unroll
      for (int i = 0; i < DPT; ++i) {
        p0 += q_val0[i] * k_v_b[i];
        p1 += q_val1[i] * k_v_b[i];
      }
      partials0[b] = sub_group_reduce_add(p0);
      partials1[b] = sub_group_reduce_add(p1);
    }

    // ----- Phase 2: online softmax fold for both heads -----
    // Head 0
    const float prev_max0 = running_max0;
    float bm0 = partials0[0];
    #pragma unroll
    for (int b = 1; b < KK_BLOCK; ++b) bm0 = fmax(bm0, partials0[b]);
    const float new_max0 = fmax(prev_max0, bm0);
    const float rescale0 = (isinf(prev_max0) && prev_max0 < 0.0f)
                             ? 0.0f
                             : native_exp(prev_max0 - new_max0);
    float bsum0 = 0.0f;
    #pragma unroll
    for (int b = 0; b < KK_BLOCK; ++b) {
      partials0[b] = native_exp(partials0[b] - new_max0);
      bsum0 += partials0[b];
    }
    // Head 1
    const float prev_max1 = running_max1;
    float bm1 = partials1[0];
    #pragma unroll
    for (int b = 1; b < KK_BLOCK; ++b) bm1 = fmax(bm1, partials1[b]);
    const float new_max1 = fmax(prev_max1, bm1);
    const float rescale1 = (isinf(prev_max1) && prev_max1 < 0.0f)
                             ? 0.0f
                             : native_exp(prev_max1 - new_max1);
    float bsum1 = 0.0f;
    #pragma unroll
    for (int b = 0; b < KK_BLOCK; ++b) {
      partials1[b] = native_exp(partials1[b] - new_max1);
      bsum1 += partials1[b];
    }

    // Pre-rescale the running accs.
    #pragma unroll
    for (int i = 0; i < DPT; ++i) {
      acc0[i] = acc0[i] * rescale0;
      acc1[i] = acc1[i] * rescale1;
    }

    // ----- Phase 3: load V, update accs -----
    // V is shared across heads, so load once per b.
    #pragma unroll
    for (int b = 0; b < KK_BLOCK; ++b) {
      float v_v_b[DPT];
      #pragma unroll
      for (int i = 0; i < DPT; ++i) {
        const int dd   = d + i * WG;
        const int base = (kk + b) * W_k + h_kv * HD + dd;
        v_v_b[i] = (float)V_cache[base];
      }
      #pragma unroll
      for (int i = 0; i < DPT; ++i) {
        acc0[i] += partials0[b] * v_v_b[i];
        acc1[i] += partials1[b] * v_v_b[i];
      }
    }

    running_sum0 = running_sum0 * rescale0 + bsum0;
    running_max0 = new_max0;
    running_sum1 = running_sum1 * rescale1 + bsum1;
    running_max1 = new_max1;

    kk += KK_BLOCK;
  }

  // Tail: handle 0..3 leftover kk positions (single per-kk fold,
  // shared K and V load across both heads).
  for (int t = 0; t < kk_tail; ++t) {
    float k_v_t[DPT], v_v_t[DPT];
    #pragma unroll
    for (int i = 0; i < DPT; ++i) {
      const int dd   = d + i * WG;
      const int base = (kk + t) * W_k + h_kv * HD + dd;
      k_v_t[i] = (float)K_cache[base];
      v_v_t[i] = (float)V_cache[base];
    }
    float p0 = 0.0f, p1 = 0.0f;
    #pragma unroll
    for (int i = 0; i < DPT; ++i) {
      p0 += q_val0[i] * k_v_t[i];
      p1 += q_val1[i] * k_v_t[i];
    }
    const float s0 = sub_group_reduce_add(p0);
    const float s1 = sub_group_reduce_add(p1);

    const float pm0 = running_max0;
    const float nm0 = fmax(pm0, s0);
    const float r0  = (isinf(pm0) && pm0 < 0.0f) ? 0.0f
                                                  : native_exp(pm0 - nm0);
    const float e0  = native_exp(s0 - nm0);
    const float pm1 = running_max1;
    const float nm1 = fmax(pm1, s1);
    const float r1  = (isinf(pm1) && pm1 < 0.0f) ? 0.0f
                                                  : native_exp(pm1 - nm1);
    const float e1  = native_exp(s1 - nm1);

    #pragma unroll
    for (int i = 0; i < DPT; ++i) {
      acc0[i] = acc0[i] * r0 + e0 * v_v_t[i];
      acc1[i] = acc1[i] * r1 + e1 * v_v_t[i];
    }
    running_sum0 = running_sum0 * r0 + e0;
    running_max0 = nm0;
    running_sum1 = running_sum1 * r1 + e1;
    running_max1 = nm1;
  }

  const float inv0 = (running_sum0 > 0.0f) ? (1.0f / running_sum0) : 0.0f;
  const float inv1 = (running_sum1 > 0.0f) ? (1.0f / running_sum1) : 0.0f;
  #pragma unroll
  for (int i = 0; i < DPT; ++i) {
    const int dd = d + i * WG;
    out[h0 * HD + dd] = (half)(acc0[i] * inv0);
    out[h1 * HD + dd] = (half)(acc1[i] * inv1);
  }
}
