// Step 2 / incremental: K load + Q load + per-lane partial.
//
// Step 1 measurement: qk = 170 ms (K-only, 36 layers).  That's only
// 9.5% of V0's 1794 ms — L2 cache across 14000 WGs absorbs most of
// the K traffic.  Memory is NOT the main attention bottleneck on
// this hardware; reduce/softmax/V/acc together account for the
// remaining 1624 ms.
//
// Step 2 adds:
//   * one Q load per WG (outside the kk loop, so ~1 load / WG)
//   * one V load per kk (same pattern as K — adds memory cost
//     comparable to K itself ~170 ms)
//   * per-lane partial = q_val * k_val accumulated to sink
// Still no reduce, no softmax, no real score.  Expected time:
// Step 1 + ~170 ms V read + tiny Q overhead -> ~340-400 ms.
//
// If Step 2 lands near 350 ms, the K + V + Q path costs ~350 ms
// total and ALL the remaining 1400 ms in V0 must come from the
// reduce + softmax + acc path — i.e. work_group_reduce_add or the
// per-kk __local broadcast of rescale / score_exp is the real
// bottleneck, not memory.
//
// Sink strategy: acc keeps per-lane running q*k + v so neither the
// K nor the V read is dead code.  Scale applied on write-out.
//
// Kernel signature and dispatch shape still match V0.

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define HD 128

__kernel __attribute__((reqd_work_group_size(HD, 1, 1)))
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

  const int d = get_local_id(0);
  const int h = get_group_id(1);
  const int m = get_group_id(2);
  if (h >= num_heads_Q || m >= M) return;

  const int num_heads_KV = num_heads_Q / gqa_size;
  const int h_kv = h / gqa_size;
  const int W_q  = num_heads_Q  * HD;
  const int W_k  = num_heads_KV * HD;

  const float q_val = (float)Q[m * W_q + h * HD + d];

  const int kk_end = (is_causal != 0) ? (from + m + 1) : T;

  // Per-lane sink: accumulate q*k + v for every kk.  Each thread
  // keeps its own running acc so the compiler can't hoist the loop.
  float acc = 0.0f;
  for (int kk = 0; kk < kk_end; ++kk) {
    const float k_val = (float)K_cache[kk * W_k + h_kv * HD + d];
    const float v_val = (float)V_cache[kk * W_k + h_kv * HD + d];
    acc += q_val * k_val + v_val;
  }

  out[m * W_q + h * HD + d] = (half)(acc * scale);
}
