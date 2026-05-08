// v3 with explicit `prefetch()` hints to issue async L2 loads
// ahead of when the inner kk loop actually consumes them.
//
// Why this lever:
//   v3 / v8 plateau at s2start = 262us per dispatch. v8 (sub-group
//   sharing) didn't help because Adreno's L2 already caches sibling-
//   kv-head loads automatically. So the bottleneck isn't cache
//   redundancy; it's the L2->L1 fetch latency stalling each kk's
//   load chain. ~470 kk x 4 cache lines x ~30 cycles = ~70us of
//   pure miss-stall on the critical path, plus serial softmax fold
//   dependencies.
//
//   prefetch(__global ptr, size) is a non-blocking hint that the
//   data should be brought into the cache hierarchy before it's
//   accessed. On Adreno it warms L2 (and L1 if the line is small
//   enough) so subsequent reads hit instead of stalling.
//
// We prefetch PREFETCH_AHEAD blocks ahead at the top of each outer
// iteration. Each prefetch covers KK_BLOCK * HD * 2 = 1024 bytes of
// K and the same for V. With Adreno's prefetcher able to track
// multiple in-flight prefetches, this should fully hide L1 miss
// latency by the time the actual load happens.
//
// If Adreno silently no-ops prefetch, this kernel runs identical
// to v3 (no regression).

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable

#define HD              128
#define WG              64
#define DPT             (HD / WG)   // 2
#define KK_BLOCK        4
#define PREFETCH_AHEAD  2

__attribute__((qcom_reqd_sub_group_size("full")))
__kernel __attribute__((reqd_work_group_size(WG, 1, 1)))
void attention_fused_fp16_decode_v9(
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

  // Warm-up prefetches for the first PREFETCH_AHEAD blocks before
  // the main loop begins. This way every outer iter's actual loads
  // hit L2/L1 instead of stalling on cold cache lines.
  #pragma unroll
  for (int pre = 0; pre < PREFETCH_AHEAD; ++pre) {
    if (pre < kk_blocks) {
      const int kk_warm = pre * KK_BLOCK;
      prefetch(K_cache + kk_warm * W_k + h_kv * HD,
               (size_t)(KK_BLOCK * HD));
      prefetch(V_cache + kk_warm * W_k + h_kv * HD,
               (size_t)(KK_BLOCK * HD));
    }
  }

  int kk = 0;
  for (int p = 0; p < kk_blocks; ++p) {
    // Prefetch the block PREFETCH_AHEAD iterations from now.
    if (p + PREFETCH_AHEAD < kk_blocks) {
      const int kk_pf = (p + PREFETCH_AHEAD) * KK_BLOCK;
      prefetch(K_cache + kk_pf * W_k + h_kv * HD,
               (size_t)(KK_BLOCK * HD));
      prefetch(V_cache + kk_pf * W_k + h_kv * HD,
               (size_t)(KK_BLOCK * HD));
    }

    // ----- PHASE 1: K + partial compute -----
    float partials[KK_BLOCK];
    #pragma unroll
    for (int b = 0; b < KK_BLOCK; ++b) {
      float local_p = 0.0f;
      #pragma unroll
      for (int i = 0; i < DPT; ++i) {
        const int dd   = d + i * WG;
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
        const int dd   = d + i * WG;
        const int base = (kk + b) * W_k + h_kv * HD + dd;
        acc[i] += pb * (float)V_cache[base];
      }
    }

    running_sum = running_sum * rescale + bsum;
    running_max = new_max;
    kk += KK_BLOCK;
  }

  // Tail
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
