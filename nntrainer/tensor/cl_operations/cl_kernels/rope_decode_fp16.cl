// Decode-specific in-kernel RoPE for FP16 (M=1).
//
// Replaces the CPU-side apply_rotary_emb_tensor_v2 in MHACoreLayer
// so that the entry-side SVMMap drain (currently 526us / call wait
// for upstream qkv_norm GPU writes to be CPU-visible) becomes
// unnecessary -- the producer (qkv_norm) and consumer (this kernel
// + attention_fused_fp16_decode) are then both GPU kernels on the
// same blas_cc queue, so OpenCL same-queue ordering covers the
// read-after-write coherence.
//
// Algorithm (matches MHACoreLayer::apply_rotary_emb_tensor_v2 fp16):
//   For each head h, for each pair (d, d+half_dim):
//     theta_d = 1 / pow(theta_base, 2d/head_dim)
//             = exp(d * (-2 * ln(theta_base) / head_dim))
//             = native_exp(d * exponent_base)        // exponent_base passed in
//     angle   = position * theta_d
//     c, s    = cos(angle), sin(angle)               // precise math (large arg)
//     out[d]            = in[d]      * c - in[d+half] * s
//     out[d+half]       = in[d+half] * c + in[d]     * s
//
// In-place when out == in; out-of-place K-cache write when out
// points to a slice of cache_key.
//
// Dispatch: WG = head_dim/2, one WG per head.
//   g = { head_dim/2, num_heads, 1 }
//   l = { head_dim/2, 1,         1 }
//
// Notes:
//   * `cos` / `sin` are spec-quality (libm range reduction) -- needed
//     because for theta_d=1 (d=0) and position up to 2048, angle can
//     reach ~2048 rad which native_cos may handle poorly on Adreno.
//   * native_exp is fine: exponent stays in [-2*ln(theta_base), 0],
//     i.e. [-27.6, 0] for theta_base = 1e6.

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__attribute__((reqd_work_group_size(64, 1, 1)))
__kernel void rope_decode_fp16(
    __global const half *in,
    __global half *out,
    const int num_heads,
    const int head_dim,
    const int position,
    const float exponent_base) {
  const int half_dim = head_dim / 2;
  const int d = get_local_id(0);
  const int h = get_group_id(1);
  if (d >= half_dim || h >= num_heads) return;

  const float exponent = (float)d * exponent_base;
  const float theta_d = native_exp(exponent);
  const float angle   = (float)position * theta_d;
  const float c = cos(angle);
  const float s = sin(angle);

  const int base = h * head_dim;
  const float x0 = (float)in[base + d];
  const float x1 = (float)in[base + d + half_dim];

  out[base + d]            = (half)(x0 * c - x1 * s);
  out[base + d + half_dim] = (half)(x1 * c + x0 * s);
}
