// Step 6 / incremental: step 5 + TQ=2 per-thread Q-tile.
//
// Step 5 landed at qk = 748 ms (correct output) — already beats NEON's
// 866 ms by 14%.  Breakdown:
//   ~340 ms  memory   (K+V+Q loads, from step 2 measurement)
//   ~410 ms  reduce + softmax + acc
//
// Step 6 attacks the ~340 ms memory component by packing TQ=2
// consecutive Q positions into the same WG.  Each thread still
// handles DPT=2 d values per row (HD / WG = 128 / 64 = 2), now
// across TQ=2 rows -> four fp32 q_val slots per thread.  One K/V
// fetch per kk feeds TQ=2 rows, halving K/V reads.
//
// Per kk inner loop:
//   * Load K[kk, h_kv, d_a], K[kk, h_kv, d_b]        (shared across rows)
//   * Load V[kk, h_kv, d_a], V[kk, h_kv, d_b]        (shared across rows)
//   * Per row q in [0, TQ):
//       partial[q] = q_val[q][0]*k_val[0] + q_val[q][1]*k_val[1]
//       score[q]   = sub_group_reduce_add(partial[q]) * scale
//       (branchless softmax update for running_max[q] / running_sum[q])
//       acc[q][0]  = acc[q][0]*rescale + score_exp*v_val[0]
//       acc[q][1]  = acc[q][1]*rescale + score_exp*v_val[1]
//
// Register budget at TQ=2 DPT=2:
//   q_val[2][2] = 4 fp32
//   acc[2][2]   = 4 fp32
//   running_max[2], running_sum[2] = 4 fp32
//   + a few scratches -> ~16 regs/thread = well under the wavefront
//   register file.
//
// Tail (M not divisible by TQ) is handled per-row via kk_end_row[q]
// gating: each row loops only up to its own causal end, other rows
// contribute a zero-partial / neutral softmax step (rescale=1,
// score_exp=0 -> acc unchanged).
//
// Expected: qk ~500-600 ms if memory fraction scales cleanly.
// Dispatch shape changes: global = (64, num_heads_Q, ceil(M/TQ)).

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable

#define HD  128
#define WG  64
#define DPT (HD / WG)          // 2
#define TQ  2

__attribute__((qcom_reqd_sub_group_size("full")))
__kernel __attribute__((reqd_work_group_size(WG, 1, 1)))
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

  const int d      = get_local_id(0);         // 0..WG-1
  const int h      = get_group_id(1);
  const int m_base = get_group_id(2) * TQ;

  const int num_heads_KV = num_heads_Q / gqa_size;
  const int h_kv = h / gqa_size;
  const int W_q  = num_heads_Q  * HD;
  const int W_k  = num_heads_KV * HD;

  // Per-row validity and causal bounds.
  int  kk_end_row[TQ];
  bool row_valid [TQ];
  int  kk_end_tile = 0;
  #pragma unroll
  for (int q = 0; q < TQ; ++q) {
    const int m = m_base + q;
    row_valid [q] = (m < M);
    kk_end_row[q] = row_valid[q]
                    ? ((is_causal != 0) ? (from + m + 1) : T)
                    : 0;
    if (kk_end_row[q] > kk_end_tile) kk_end_tile = kk_end_row[q];
  }

  // Load Q once per row × DPT.
  float q_val[TQ][DPT];
  #pragma unroll
  for (int q = 0; q < TQ; ++q) {
    const int m = m_base + q;
    #pragma unroll
    for (int i = 0; i < DPT; ++i) {
      const int dd = d + i * WG;
      q_val[q][i] = row_valid[q] ? (float)Q[m * W_q + h * HD + dd] : 0.0f;
    }
  }

  float running_max[TQ] = {-INFINITY, -INFINITY};
  float running_sum[TQ] = { 0.0f,      0.0f     };
  float acc        [TQ][DPT] = { {0.0f, 0.0f}, {0.0f, 0.0f} };

  for (int kk = 0; kk < kk_end_tile; ++kk) {
    // Shared per-kk K/V loads (DPT lanes per thread).
    float k_val[DPT];
    float v_val[DPT];
    #pragma unroll
    for (int i = 0; i < DPT; ++i) {
      const int dd = d + i * WG;
      k_val[i] = (float)K_cache[kk * W_k + h_kv * HD + dd];
      v_val[i] = (float)V_cache[kk * W_k + h_kv * HD + dd];
    }

    #pragma unroll
    for (int q = 0; q < TQ; ++q) {
      const bool row_active = row_valid[q] && (kk < kk_end_row[q]);

      // Local partial over DPT lanes, masked on inactive rows.
      float partial = 0.0f;
      #pragma unroll
      for (int i = 0; i < DPT; ++i) partial += q_val[q][i] * k_val[i];
      if (!row_active) partial = 0.0f;

      const float score = sub_group_reduce_add(partial) * scale;

      // Neutral softmax step on inactive rows: rescale=1,
      // score_exp=0 -> acc / running_sum unchanged.
      const float prev_max = running_max[q];
      const float new_max  = row_active ? fmax(prev_max, score) : prev_max;
      const float rescale  = (!row_active)
                             ? 1.0f
                             : ((isinf(prev_max) && prev_max < 0.0f)
                                ? 0.0f
                                : exp(prev_max - new_max));
      const float score_exp = row_active ? exp(score - new_max) : 0.0f;

      #pragma unroll
      for (int i = 0; i < DPT; ++i) {
        acc[q][i] = acc[q][i] * rescale + score_exp * v_val[i];
      }
      running_sum[q] = running_sum[q] * rescale + score_exp;
      running_max[q] = new_max;
    }
  }

  // Write all valid rows × DPT.
  #pragma unroll
  for (int q = 0; q < TQ; ++q) {
    if (!row_valid[q]) continue;
    const int m = m_base + q;
    const float s = running_sum[q];
    const float inv = (s > 0.0f) ? (1.0f / s) : 0.0f;
    #pragma unroll
    for (int i = 0; i < DPT; ++i) {
      const int dd = d + i * WG;
      out[m * W_q + h * HD + dd] = (half)(acc[q][i] * inv);
    }
  }
}
