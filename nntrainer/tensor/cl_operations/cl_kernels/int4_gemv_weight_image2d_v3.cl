// 2x K-unrolled int4 GEMV (M=1) on RGBA32UI weight image2d.
//
// Same algorithm and memory layout as gpu_int4_gemv_weight_image2d_v2
// (8 K-positions per pixel per channel), but the K loop is unrolled 2x
// to read 2 weight pixels and 16 K input positions per iteration.
//
// Per-iter work: 16 K-positions × 4 N-channels = 64 MACs (vs 32 in v2).
// Halves the loop trip count, exposes more instruction-level
// parallelism between weight loads and MAC chains, keeps the same
// 64-lane / 256-N-per-WG dispatch geometry so cache behavior is
// identical to v2.
//
// Dispatch (host -- identical to v2):
//   global = align(N, 256) / 4
//   local  = (64, 1, 1)

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__constant sampler_t weight_smp_v3 = CLK_NORMALIZED_COORDS_FALSE |
                                      CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

kernel void
gpu_int4_gemv_weight_image2d_v3(__constant half *input,
                                 __global const half *scales,
                                 __global half *output,
                                 __read_only image2d_t weights,
                                 const int K,
                                 const int N) {
  const int n = get_global_id(0) * 4;
  if (n >= N)
    return;

  float acc0 = 0.0f;
  float acc1 = 0.0f;
  float acc2 = 0.0f;
  float acc3 = 0.0f;

  const int n_pixel = n / 4;
  // K assumed multiple of 16 (per-FC gemv: K in {2560, 4096, 9728},
  // all %16 == 0). Tail handling omitted.
  for (int k = 0; k < K; k += 16) {
    const half8 in_lo = vload8(0, input + k);
    const half8 in_hi = vload8(0, input + k + 8);
    const uint4 packed_lo = read_imageui(weights, weight_smp_v3,
                                          (int2)(n_pixel, k / 8));
    const uint4 packed_hi = read_imageui(weights, weight_smp_v3,
                                          (int2)(n_pixel, k / 8 + 1));

#define STEP(packed, in_v, lane_idx, shift)                            \
  do {                                                                 \
    const float in_k = (float)in_v.s##lane_idx;                        \
    acc0 += in_k * (float)((int)((packed.s0 >> shift) & 0xF) - 8);     \
    acc1 += in_k * (float)((int)((packed.s1 >> shift) & 0xF) - 8);     \
    acc2 += in_k * (float)((int)((packed.s2 >> shift) & 0xF) - 8);     \
    acc3 += in_k * (float)((int)((packed.s3 >> shift) & 0xF) - 8);     \
  } while (0)

    // K positions [k, k+7] from packed_lo / in_lo
    STEP(packed_lo, in_lo, 0,  0);
    STEP(packed_lo, in_lo, 1,  4);
    STEP(packed_lo, in_lo, 2,  8);
    STEP(packed_lo, in_lo, 3, 12);
    STEP(packed_lo, in_lo, 4, 16);
    STEP(packed_lo, in_lo, 5, 20);
    STEP(packed_lo, in_lo, 6, 24);
    STEP(packed_lo, in_lo, 7, 28);
    // K positions [k+8, k+15] from packed_hi / in_hi
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

  const half4 scale = vload4(0, scales + n);
  output[n + 0] = (half)(acc0 * (float)scale.s0);
  output[n + 1] = (half)(acc1 * (float)scale.s1);
  output[n + 2] = (half)(acc2 * (float)scale.s2);
  output[n + 3] = (half)(acc3 * (float)scale.s3);
}
