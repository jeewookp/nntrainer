// Fused Q+K prefill RoPE (M>1, FP16).
//
// Extends rope_decode_fp16_qk to handle M>1 tokens in one dispatch.
// Each WG covers one (token, head) pair; the token dimension is the
// third global/group dimension.
//
// Global:  { head_dim/2, num_heads_Q + num_heads_K, M }
// Local:   { head_dim/2, 1, 1 }
//
// Work groups h = 0 .. num_heads_Q-1     -> Q RoPE (in-place, q_in == q_out)
// Work groups h = num_heads_Q .. total-1 -> K RoPE (k_in -> kc_out = K_cache slice)
// Work groups m = 0 .. M-1              -> token index within prefill chunk
//
// Memory layout: [M, num_heads * head_dim] (tokens outer, heads inner)
// Matches the layout of query_step / key_step / b_cache_key_step in
// one_batch_incremental_forwarding.

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__attribute__((reqd_work_group_size(64, 1, 1)))
__kernel void rope_prefill_fp16_qk(
    __global const half *q_in,   __global half *q_out,
    __global const half *k_in,   __global half *kc_out,
    const int num_heads_Q,
    const int num_heads_K,
    const int head_dim,
    const int M,
    const int from,
    const float exponent_base) {

  const int d        = get_local_id(0);
  const int h        = get_group_id(1);
  const int m        = get_group_id(2);  // token index 0..M-1
  const int half_dim = head_dim / 2;
  if (d >= half_dim) return;
  if (m >= M)        return;

  const int position  = from + m;
  const float theta_d = native_exp((float)d * exponent_base);
  const float angle   = (float)position * theta_d;
  const float c = cos(angle);
  const float s = sin(angle);

  if (h < num_heads_Q) {
    const int base = m * num_heads_Q * head_dim + h * head_dim;
    const float x0 = (float)q_in[base + d];
    const float x1 = (float)q_in[base + d + half_dim];
    q_out[base + d]            = (half)(x0 * c - x1 * s);
    q_out[base + d + half_dim] = (half)(x1 * c + x0 * s);
  } else {
    const int hk = h - num_heads_Q;
    if (hk >= num_heads_K) return;
    const int base_k = m * num_heads_K * head_dim + hk * head_dim;
    const float x0 = (float)k_in[base_k + d];
    const float x1 = (float)k_in[base_k + d + half_dim];
    kc_out[base_k + d]            = (half)(x0 * c - x1 * s);
    kc_out[base_k + d + half_dim] = (half)(x1 * c + x0 * s);
  }
}
