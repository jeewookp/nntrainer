// Flash Attention-style M=1 attention with local-memory KV chunk tile.
//
// Each WG cooperatively loads CHUNK_KK kk-positions of K and V into
// __local memory (one tile each), then iterates through the chunk
// reading from local mem (~1-cycle latency vs ~100 cycles for L2).
// After the chunk is exhausted the inner KV is bumped and the next
// chunk loaded.
//
// Register pressure per lane is LOWER than v3 because K/V no longer
// live in registers across the kk loop (~14 fp32 vs v3's 29). Frees
// occupancy for Adreno to keep more wavefronts co-resident, hiding
// reduce/exp latency.
//
// Local memory budget per WG (HD=128, fp16):
//   K_local[CHUNK_KK][HD] = CHUNK_KK * 256 bytes
//   V_local[CHUNK_KK][HD] = same
// CHUNK_KK=16 -> 8 KB per tile, 16 KB total. Adreno's per-WG SLM
// cap is typically 32 KB so this fits with margin.
//
// Cooperative load mapping (64 lanes, 16 kk-positions x 128 d each):
//   1024 fp16 per tile / 64 lanes = 16 fp16 per lane per tile.
//   lane lid loads:
//     load_kk      = lid / 8        (0..7,  8 lanes per kk-row)
//     load_d_start = (lid % 8) * 16 (0,16,32,...,112)
//   each lane reads 16 fp16 from K_cache and writes to K_local.
// CHUNK_KK=16: covers 2 kk per pass, 4 passes total (= 8 lid groups
// over 16 kk positions).
//
// Wait -- with WG=64 and CHUNK_KK=16, mapping is:
//   load_kk      = lid / 4         (0..15)
//   load_d_start = (lid % 4) * 32  (0,32,64,96)
//   each lane reads 32 fp16. Heavy per-lane work but covers tile
//   in one pass.
//
// We'll use CHUNK_KK=8 for simpler 1:1 lane mapping:
//   1024 fp16 / 64 lanes = 16 fp16 per lane (8 kk x 128/64=2 d) too
//   small. Use 16 fp16 = 1 kk row coverage per lane.
//   Mapping: load_kk = lid/16, load_d = (lid%16)*8. Each lane loads
//   8 fp16. 64 lanes cover 8 kk * 16 d-blocks = 8 kk * 128 d = full
//   chunk in 1 pass. 16 fp16 left? No: 8 kk * 16 lane-groups = 128
//   lanes needed, but we have 64. So each lane covers 2 (kk,
//   d-block) pairs.

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable

#define HD 128
#define WG 64
#define DPT (HD / WG)  // 2
#define CHUNK_KK 8

__attribute__((qcom_reqd_sub_group_size("full")))
__kernel __attribute__((reqd_work_group_size(WG, 1, 1)))
void attention_fused_fp16_decode_v5(
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

  // Local memory tile for one CHUNK_KK worth of K and V.
  // Flat 1D to avoid Adreno OpenCL multi-dim __local + vstore8 quirk
  // ("Attempt to use subscript to obtain element of 'half4'" build err).
  __local half K_local[CHUNK_KK * HD];
  __local half V_local[CHUNK_KK * HD];

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

  // Cooperative-load mapping. Each lane covers d-positions
  // {d, d + WG} (matches the inner loop's DPT mapping). Outer loop
  // walks chunk_eff kk-rows so each pass loads 1 kk row of K/V into
  // SLM. Per chunk: chunk_eff rows x DPT=2 d-positions per lane = up
  // to 16 scalar half loads/stores per lane (kept simple and avoids
  // half8 / multi-dim local-array compiler quirks on Adreno).

  for (int kk_base = 0; kk_base < kk_end; kk_base += CHUNK_KK) {
    // Effective chunk size (last tile may be partial).
    const int chunk_eff = min(CHUNK_KK, kk_end - kk_base);

    // Cooperative load (DPT-aligned, scalar).
    for (int kk_l = 0; kk_l < chunk_eff; ++kk_l) {
      const int g_base = (kk_base + kk_l) * W_k + h_kv * HD;
      const int l_base = kk_l * HD;
      #pragma unroll
      for (int i = 0; i < DPT; ++i) {
        const int dd = d + i * WG;
        K_local[l_base + dd] = K_cache[g_base + dd];
        V_local[l_base + dd] = V_cache[g_base + dd];
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Inner kk loop reads K/V from local memory.
    for (int kk_l = 0; kk_l < chunk_eff; ++kk_l) {
      const int l_base = kk_l * HD;
      float partial = 0.0f;
      float v_val[DPT];
      #pragma unroll
      for (int i = 0; i < DPT; ++i) {
        const int dd = d + i * WG;
        const float k_v = (float)K_local[l_base + dd];
        v_val[i]        = (float)V_local[l_base + dd];
        partial += q_val[i] * k_v;
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

    barrier(CLK_LOCAL_MEM_FENCE);  // before reusing K_local / V_local
  }

  const float inv = (running_sum > 0.0f) ? (1.0f / running_sum) : 0.0f;
  #pragma unroll
  for (int i = 0; i < DPT; ++i) {
    const int dd = d + i * WG;
    out[h * HD + dd] = (half)(acc[i] * inv);
  }
}
