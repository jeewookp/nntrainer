// Multi-output int4 GEMV (M=1) on RGBA32UI weight image2d, 2x K-unrolled.
//
// Same algorithm as gpu_fused_gemv_int4_image2d but applies the per-FC
// v3 win: 8 K-positions per uint pixel (vs 4 K-pos per ushort pixel)
// halves the imageread count, and 2x K-loop unroll (16 K-positions /
// iter, 64 MACs / iter) exposes more instruction-level parallelism.
// The combo lifted the per-FC kernel from 230 us → 167 us at K=2560.
//
// Image2D layout (per partition, identical to int4_gemv_weight_image2d_v2):
//   width = N_part / 4, height = K / 8
//   pixel = uint4 = 4 output channels × 8 K-position nibbles each
//
// For the gate_up case (N_v == 0) the caller passes the q_image as
// the v_image dummy slot; the kernel never enters the v branch when
// N_v == 0, so the image is never read.

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__constant sampler_t fused_w_smp_v2 = CLK_NORMALIZED_COORDS_FALSE |
                                       CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

__attribute__((reqd_work_group_size(64, 1, 1)))
kernel void
gpu_fused_gemv_int4_image2d_v2(__global const half *input,
                                __read_only image2d_t weights_q,
                                __global const half *scales_q,
                                __global half *q_out,
                                __read_only image2d_t weights_k,
                                __global const half *scales_k,
                                __global half *k_out,
                                __read_only image2d_t weights_v,
                                __global const half *scales_v,
                                __global half *v_out,
                                const int K,
                                const int N_q,
                                const int N_k,
                                const int N_v) {
  const int gid = get_global_id(0);
  const int n   = gid * 4;
  const int total_n = N_q + N_k + N_v;
  if (n >= total_n) return;

  __global const half *scales;
  __global half       *out;
  int local_n;
  int N_part;
  if (n < N_q) {
    scales = scales_q; out = q_out;
    local_n = n;        N_part = N_q;
  } else if (n < N_q + N_k) {
    scales = scales_k; out = k_out;
    local_n = n - N_q;  N_part = N_k;
  } else {
    scales = scales_v; out = v_out;
    local_n = n - N_q - N_k;
    N_part = N_v;
  }
  if (local_n >= N_part) return;

  const int n_pixel = local_n / 4;

  float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
  // 2x K-unroll: read 2 weight pixels (16 K-positions) and 16 input
  // halves per iteration. K is assumed multiple of 16 (per-FC and
  // fused weights for Qwen3-4B all have K%16==0).
  for (int k = 0; k < K; k += 16) {
    const half8 in_lo = vload8(0, input + k);
    const half8 in_hi = vload8(0, input + k + 8);
    uint4 packed_lo;
    uint4 packed_hi;
    if (n < N_q) {
      packed_lo = read_imageui(weights_q, fused_w_smp_v2,
                               (int2)(n_pixel, k / 8));
      packed_hi = read_imageui(weights_q, fused_w_smp_v2,
                               (int2)(n_pixel, k / 8 + 1));
    } else if (n < N_q + N_k) {
      packed_lo = read_imageui(weights_k, fused_w_smp_v2,
                               (int2)(n_pixel, k / 8));
      packed_hi = read_imageui(weights_k, fused_w_smp_v2,
                               (int2)(n_pixel, k / 8 + 1));
    } else {
      packed_lo = read_imageui(weights_v, fused_w_smp_v2,
                               (int2)(n_pixel, k / 8));
      packed_hi = read_imageui(weights_v, fused_w_smp_v2,
                               (int2)(n_pixel, k / 8 + 1));
    }

#define STEP(packed, in_v, lane_idx, shift)                            \
  do {                                                                 \
    const float in_k = (float)in_v.s##lane_idx;                        \
    acc0 += in_k * (float)((int)((packed.s0 >> shift) & 0xF) - 8);     \
    acc1 += in_k * (float)((int)((packed.s1 >> shift) & 0xF) - 8);     \
    acc2 += in_k * (float)((int)((packed.s2 >> shift) & 0xF) - 8);     \
    acc3 += in_k * (float)((int)((packed.s3 >> shift) & 0xF) - 8);     \
  } while (0)

    // K positions [k+0..k+7] from packed_lo / in_lo
    STEP(packed_lo, in_lo, 0,  0);
    STEP(packed_lo, in_lo, 1,  4);
    STEP(packed_lo, in_lo, 2,  8);
    STEP(packed_lo, in_lo, 3, 12);
    STEP(packed_lo, in_lo, 4, 16);
    STEP(packed_lo, in_lo, 5, 20);
    STEP(packed_lo, in_lo, 6, 24);
    STEP(packed_lo, in_lo, 7, 28);
    // K positions [k+8..k+15] from packed_hi / in_hi
    STEP(packed_hi, in_hi, 0,  0);
    STEP(packed_hi, in_hi, 1,  4);
    STEP(packed_hi, in_hi, 2,  8);
    STEP(packed_hi, in_hi, 3, 12);
    STEP(packed_hi, in_hi, 4, 16);
    STEP(packed_hi, in_hi, 5, 20);
    STEP(packed_hi, in_hi, 6, 24);
    STEP(packed_hi, in_hi, 7, 28);
#undef STEP
  }

  const half4 scale = vload4(0, scales + local_n);
  out[local_n + 0] = (half)(acc0 * (float)scale.s0);
  out[local_n + 1] = (half)(acc1 * (float)scale.s1);
  out[local_n + 2] = (half)(acc2 * (float)scale.s2);
  out[local_n + 3] = (half)(acc3 * (float)scale.s3);
}
