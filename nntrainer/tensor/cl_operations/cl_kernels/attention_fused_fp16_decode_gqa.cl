// GQA-aware M=1 attention kernel.
//
// Qwen3-4B has num_heads_Q=32 and num_heads_KV=8 (GQA group size 4
// = 32/8). The existing decode kernel (attention_fused_fp16_decode_v2)
// dispatches 32 WGs, one per Q head; the 4 Q heads sharing each h_kv
// each independently fetch the same K/V cache region from main
// memory. With T~470 and HD=128 fp16, that's 4x duplicated KV reads
// across the 32 WG grid: ~7.7 MB raw vs 1.92 MB if shared.
//
// This kernel batches gqa_size Q heads per WG. Dispatch:
//   global = (WG, num_heads_KV, 1)   = (64, 8, 1)
//   local  = (WG, 1, 1)
// Each WG holds 4 Q heads' running state in registers, shares the
// per-kk K and V loads across them, and produces 4 head outputs.
//
// Register budget at WG=64, GQA=4, DPT=HD/WG=2:
//   q_val[4][2]      = 8 fp32
//   acc[4][2]        = 8 fp32
//   running_max[4]   = 4 fp32
//   running_sum[4]   = 4 fp32
//   k/v_val[2]       = 4 fp32
//   transient (score, exp, rescale) ~4 fp32
//   total ~32 fp32 = 128 bytes / lane -- well under Adreno's reg file.
//
// Trade-off: 32 WGs -> 8 WGs (4x less parallelism) BUT each WG does
// 4x more work and reads 4x less KV data. Net: BW reduced ~50% if
// KV was the bottleneck; compute-bound case slower due to fewer WGs
// hiding latency.

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable

#define HD 128
#define WG 64
#define DPT (HD / WG)  // 2
#define GQA 4          // Qwen3-4B specific: num_heads_Q / num_heads_KV

__attribute__((qcom_reqd_sub_group_size("full")))
__kernel __attribute__((reqd_work_group_size(WG, 1, 1)))
void attention_fused_fp16_decode_gqa(
    __global const half *Q,
    __global const half *K_cache,
    __global const half *V_cache,
    __global half *out,
    const int M,
    const int T,
    const int from,
    const int num_heads_Q,
    const int gqa_size,    // must equal GQA at compile time
    const int is_causal,
    const float scale) {

  const int d = get_local_id(0);
  const int h_kv = get_group_id(1);

  const int num_heads_KV = num_heads_Q / GQA;
  const int W_q = num_heads_Q * HD;
  const int W_k = num_heads_KV * HD;

  const int kk_end = (is_causal != 0) ? (from + 1) : T;

  // Load Q for all GQA Q heads in this h_kv group.
  float q_val[GQA][DPT];
  #pragma unroll
  for (int g = 0; g < GQA; ++g) {
    const int h_q = h_kv * GQA + g;
    #pragma unroll
    for (int i = 0; i < DPT; ++i) {
      const int dd = d + i * WG;
      q_val[g][i] = (float)Q[h_q * HD + dd];
    }
  }

  float running_max[GQA];
  float running_sum[GQA];
  float acc[GQA][DPT];
  #pragma unroll
  for (int g = 0; g < GQA; ++g) {
    running_max[g] = -INFINITY;
    running_sum[g] = 0.0f;
    #pragma unroll
    for (int i = 0; i < DPT; ++i) acc[g][i] = 0.0f;
  }

  for (int kk = 0; kk < kk_end; ++kk) {
    // Single shared K/V load for all GQA heads.
    float k_val[DPT];
    float v_val[DPT];
    #pragma unroll
    for (int i = 0; i < DPT; ++i) {
      const int dd = d + i * WG;
      k_val[i] = (float)K_cache[kk * W_k + h_kv * HD + dd];
      v_val[i] = (float)V_cache[kk * W_k + h_kv * HD + dd];
    }

    #pragma unroll
    for (int g = 0; g < GQA; ++g) {
      float partial = 0.0f;
      #pragma unroll
      for (int i = 0; i < DPT; ++i) partial += q_val[g][i] * k_val[i];
      const float score = sub_group_reduce_add(partial) * scale;

      const float prev_max = running_max[g];
      const float new_max = fmax(prev_max, score);
      const float rescale = (isinf(prev_max) && prev_max < 0.0f)
                              ? 0.0f
                              : exp(prev_max - new_max);
      const float score_exp = exp(score - new_max);

      #pragma unroll
      for (int i = 0; i < DPT; ++i) {
        acc[g][i] = acc[g][i] * rescale + score_exp * v_val[i];
      }
      running_sum[g] = running_sum[g] * rescale + score_exp;
      running_max[g] = new_max;
    }
  }

  #pragma unroll
  for (int g = 0; g < GQA; ++g) {
    const int h_q = h_kv * GQA + g;
    const float inv = (running_sum[g] > 0.0f) ? (1.0f / running_sum[g]) : 0.0f;
    #pragma unroll
    for (int i = 0; i < DPT; ++i) {
      const int dd = d + i * WG;
      out[h_q * HD + dd] = (half)(acc[g][i] * inv);
    }
  }
}
