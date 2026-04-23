// Step 1 / incremental: K-only memory-bandwidth probe.
//
// Full V5 with TQ=4 landed at MHA=1284 ms / qk=1204 ms (vs V0 1794 ms
// and NEON 866 ms).  Better than V0, still ~5x off the theoretical
// memory floor.  To see where the rest of the 1204 ms actually goes
// we strip the kernel back to just the K read + a scalar sink, so the
// measured time is (dispatch + scheduler + pure K global-memory load)
// with no Q, no V, no softmax, no reduce.
//
// Kernel signature is kept identical to V0..V5 (Q, K, V, out, plus
// scalars) so the C++ dispatch side doesn't change.  Unused arguments
// are quietly ignored.  out is filled with a deterministic, K-derived
// value so the compiler cannot dead-code-eliminate the inner loop.
//
// Incremental plan (each step keeps previous measurement points
// valid so we can spot which addition costs what):
//   Step 1:  K load only, sink to out                            <-- here
//   Step 2:  + Q load + partial = q * k, sink
//   Step 3:  + tree reduce across d for score
//   Step 4:  + online softmax (running_max / running_sum / rescale)
//   Step 5:  + V load + acc update
//   Step 6:  + TQ-tile register-held Q (V5 design)
//
// Dispatch is V0 shape: WG=(128,1,1), global=(128, num_heads_Q, M).

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

  (void)Q;
  (void)V_cache;

  const int d = get_local_id(0);
  const int h = get_group_id(1);
  const int m = get_group_id(2);
  if (h >= num_heads_Q || m >= M) return;

  const int num_heads_KV = num_heads_Q / gqa_size;
  const int h_kv = h / gqa_size;
  const int W_q  = num_heads_Q  * HD;
  const int W_k  = num_heads_KV * HD;

  const int kk_end = (is_causal != 0) ? (from + m + 1) : T;

  // Pure K read loop.  acc must feed the output to survive the
  // compiler's dead-code pass.  Same K access pattern as V0 / V5:
  //   one contiguous HD-half cache line per WG per kk.
  float acc = 0.0f;
  for (int kk = 0; kk < kk_end; ++kk) {
    const float k_val = (float)K_cache[kk * W_k + h_kv * HD + d];
    acc += k_val;
  }

  // Store something derived from every loaded K so the compiler
  // can't elide the loop.  `scale` is a runtime scalar so it
  // defeats constant folding too.
  out[m * W_q + h * HD + d] = (half)(acc * scale);
}
