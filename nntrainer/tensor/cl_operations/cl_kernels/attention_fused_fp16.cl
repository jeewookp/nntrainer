// Step 3 / incremental: + work_group_reduce_add across d per kk.
//
// Step 1 qk =  170 ms  (K-only load)
// Step 2 qk =  341 ms  (+ V load + Q load + per-lane partial q*k+v)
// V0 qk    = 1794 ms  (full kernel: reduce + softmax + V acc update)
//
// So memory (K+V+Q+partial) costs ~340 ms, and the remaining
// ~1450 ms of V0 must be in reduce + softmax + acc update.
//
// Step 3 isolates the REDUCE cost by adding one work_group_reduce_add
// per kk on (q_val * k_val), multiplying by scale to form the score,
// and sinking the score into a per-lane accumulator against v_val.
// Still no softmax, no online-max / -sum, no rescale.  The per-lane
// acc += score * v keeps every memory and reduce op live against
// dead-code elimination.
//
// Expected cost split:
//   Step 3 - Step 2  = per-kk work_group_reduce_add cost
// If step 3 comes in around 800-1200 ms, reduce is the dominant
// component of V0's 1794 ms and we need a different reduce strategy
// (sub_group_reduce, tree reduce with fewer rows, ...).
// If step 3 comes in around 400-600 ms, reduce is cheap and the
// missing cost lives in softmax / acc (step 4-5).
//
// Kernel signature + V0 dispatch unchanged.

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

  float acc = 0.0f;
  for (int kk = 0; kk < kk_end; ++kk) {
    const float k_val = (float)K_cache[kk * W_k + h_kv * HD + d];
    const float v_val = (float)V_cache[kk * W_k + h_kv * HD + d];

    // The reduce we want to cost in isolation.  Every thread sees
    // the same score after the intrinsic — no explicit barrier
    // needed.
    const float score = work_group_reduce_add(q_val * k_val) * scale;

    // Per-lane sink.  score * v_val keeps both the reduce output
    // and the V load alive for the compiler.
    acc += score * v_val;
  }

  out[m * W_q + h * HD + d] = (half)acc;
}
