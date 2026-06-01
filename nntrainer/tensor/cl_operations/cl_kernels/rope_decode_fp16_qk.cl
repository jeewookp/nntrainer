// Fused Q+K decode RoPE (M=1, FP16).
//
// Combines the separate rope_q and rope_k dispatches into a single
// clEnqueueNDRangeKernel call, saving one kernel dispatch overhead
// (~18 us × 2304 calls ≈ 41 ms per generation run).
//
// Global:  { head_dim/2, num_heads_Q + num_heads_K, 1 }
// Local:   { head_dim/2, 1, 1 }
//
// Work groups h = 0 .. num_heads_Q-1        → Q RoPE (in-place, q_in == q_out)
// Work groups h = num_heads_Q .. total-1    → K RoPE (k_in → kc_out = K_cache slice)
//
// Algorithm is identical to rope_decode_fp16 (q6k / rope_decode paths).

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void rope_decode_fp16_qk(
    __global const half *q_in,   __global half *q_out,
    __global const half *k_in,   __global half *kc_out,
    const int num_heads_Q,
    const int num_heads_K,
    const int head_dim,
    const int position,
    const float exponent_base) {

  const int d        = get_local_id(0);
  const int h        = get_group_id(1);
  const int half_dim = head_dim / 2;
  if (d >= half_dim) return;

  const float theta_d = native_exp((float)d * exponent_base);
  const float angle   = (float)position * theta_d;
  const float c = cos(angle);
  const float s = sin(angle);

  if (h < num_heads_Q) {
    const int base = h * head_dim;
    const float x0 = (float)q_in[base + d];
    const float x1 = (float)q_in[base + d + half_dim];
    q_out[base + d]            = (half)(x0 * c - x1 * s);
    q_out[base + d + half_dim] = (half)(x1 * c + x0 * s);
  } else {
    const int hk = h - num_heads_Q;
    if (hk >= num_heads_K) return;
    const int base = hk * head_dim;
    const float x0 = (float)k_in[base + d];
    const float x1 = (float)k_in[base + d + half_dim];
    kc_out[base + d]            = (half)(x0 * c - x1 * s);
    kc_out[base + d + half_dim] = (half)(x1 * c + x0 * s);
  }
}
