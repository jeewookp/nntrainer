// Q dot K^T per-head attention scoring for Qwen-style GQA.  Replaces
// the NEON compute_kcaches inner loop on the FP16 prefill path.
//
//   Q     : fp16 [M, num_heads_Q,  head_dim]
//   K     : fp16 [T_max, num_heads_KV, head_dim]   (cache; only [0, T) valid)
//   out   : fp16 attention scores in causal-triangle layout
//
// Output indexing (matches the NEON impl in mha_core.cpp's compute_kcaches
// caller -- output_addr = out + out_start_row * num_heads_Q):
//
//   if is_causal:
//     out_start_row(i) = (from + i)*(from + i + 1)/2 - from*(from + 1)/2
//     out[(out_start_row(i) + kk) * num_heads_Q + h] = score(i, kk, h)
//     for kk in [0, from + i + 1)
//
//   else (full M x T):
//     out[(i * T + kk) * num_heads_Q + h] = score(i, kk, h)
//
// score(i, kk, h) = (sum_d Q[i, h, d] * K[kk, h_kv, d]) / sqrt(head_dim)
//                   with h_kv = h / gqa_size  (GQA broadcast).
//
// Dispatch:
//   global = (T_round, num_heads_Q, M)        local = (64, 1, 1)
//   each work-item computes ONE score for (i = gid.z, h = gid.y, kk = gid.x).
//   causal-masked threads early-return (about half the grid for prefill).

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void compute_kcaches_qwen_fp16(
    __global const half *q,
    __global const half *k_cache,
    __global half *out,
    const int M,
    const int T,
    const int from,
    const int num_heads_Q,
    const int gqa_size,
    const int head_dim,
    const int is_causal,
    const float scale) {

  const int kk = get_global_id(0);
  const int h  = get_global_id(1);
  const int i  = get_global_id(2);
  if (i >= M || h >= num_heads_Q || kk >= T) return;

  // Causal mask: kk valid only up to from + i.
  if (is_causal != 0 && kk > from + i) return;

  const int num_heads_KV = num_heads_Q / gqa_size;
  const int W_q = num_heads_Q  * head_dim;
  const int W_k = num_heads_KV * head_dim;
  const int h_kv = h / gqa_size;

  const __global half *q_ptr = q       + i  * W_q + h    * head_dim;
  const __global half *k_ptr = k_cache + kk * W_k + h_kv * head_dim;

  // fp32 accumulator to match the NEON path's numerical behaviour.
  float acc = 0.0f;
  for (int d = 0; d < head_dim; ++d) {
    acc += (float)q_ptr[d] * (float)k_ptr[d];
  }

  int out_start_row;
  if (is_causal != 0) {
    const int A = (from + i) * (from + i + 1) / 2;
    const int B = from * (from + 1) / 2;
    out_start_row = A - B;
  } else {
    out_start_row = i * T;
  }

  out[(out_start_row + kk) * num_heads_Q + h] = (half)(acc * scale);
}
