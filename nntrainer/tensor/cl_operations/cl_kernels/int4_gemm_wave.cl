#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// Int4 GEMM — delegate thread layout + vectorized reads + full unroll.
// No wave memory (avoids segfault from constant_load offset issues).
// Each thread: 8 output slices × 4 channels = 32 outputs.
// Inner loop: 2 src slices (8 k values) per iteration, fully unrolled.

__constant sampler_t smp_zero = CLK_NORMALIZED_COORDS_FALSE |
                                 CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

__kernel void gemm_int4_wave(
    __global const ushort* restrict weights,
    __global const ushort* restrict unused,
    __global const half* restrict scales,
    __write_only image2d_t dst_image,
    __read_only image2d_t src_image,
    const int M,
    const int N,
    const int K) {

  const int src_slices = K >> 2;
  const int dst_slices = N >> 2;

  int X = get_group_id(1) * get_local_size(0) + get_local_id(0);
  int Y = get_group_id(2) * get_local_size(1) + get_local_id(1);
  int Z = get_group_id(0) * get_local_size(2) + get_local_id(2);
  if (X >= M || Y >= 1) return;
  if (Z * 8 >= dst_slices) return;

  half4 r0 = (half4)(0.0h), r1 = (half4)(0.0h);
  half4 r2 = (half4)(0.0h), r3 = (half4)(0.0h);
  half4 r4 = (half4)(0.0h), r5 = (half4)(0.0h);
  half4 r6 = (half4)(0.0h), r7 = (half4)(0.0h);

  int base_n = Z * 32;

  // Preload scales
  half4 sc0 = vload4(0, scales + base_n);
  half4 sc1 = vload4(0, scales + base_n + 4);
  half4 sc2 = vload4(0, scales + base_n + 8);
  half4 sc3 = vload4(0, scales + base_n + 12);
  half4 sc4 = vload4(0, scales + base_n + 16);
  half4 sc5 = vload4(0, scales + base_n + 20);
  half4 sc6 = vload4(0, scales + base_n + 24);
  half4 sc7 = vload4(0, scales + base_n + 28);

#define DQ4(PW, SHIFT, SC) ((half4)( \
  ((half)((int)((PW.x >> SHIFT) & 0xF) - 8)) * SC.x, \
  ((half)((int)((PW.y >> SHIFT) & 0xF) - 8)) * SC.y, \
  ((half)((int)((PW.z >> SHIFT) & 0xF) - 8)) * SC.z, \
  ((half)((int)((PW.w >> SHIFT) & 0xF) - 8)) * SC.w))

  int coord_s = 0;
  do {
    half4 src0 = read_imageh(src_image, smp_zero,
                             (int2)(X, coord_s));
    half4 src1 = read_imageh(src_image, smp_zero,
                             (int2)(X, coord_s + 1));

    // Row 0: weights at packed row coord_s, 32 consecutive n values
    int row0 = coord_s * N + base_n;
    ushort4 pw;

    // Output group 0 (channels base_n+0..3)
    pw = vload4(0, weights + row0);
    r0 += src0.x * DQ4(pw, 0, sc0);
    r0 += src0.y * DQ4(pw, 4, sc0);
    r0 += src0.z * DQ4(pw, 8, sc0);
    r0 += src0.w * DQ4(pw, 12, sc0);

    // Output group 1
    pw = vload4(0, weights + row0 + 4);
    r1 += src0.x * DQ4(pw, 0, sc1);
    r1 += src0.y * DQ4(pw, 4, sc1);
    r1 += src0.z * DQ4(pw, 8, sc1);
    r1 += src0.w * DQ4(pw, 12, sc1);

    // Output group 2
    pw = vload4(0, weights + row0 + 8);
    r2 += src0.x * DQ4(pw, 0, sc2);
    r2 += src0.y * DQ4(pw, 4, sc2);
    r2 += src0.z * DQ4(pw, 8, sc2);
    r2 += src0.w * DQ4(pw, 12, sc2);

    // Output group 3
    pw = vload4(0, weights + row0 + 12);
    r3 += src0.x * DQ4(pw, 0, sc3);
    r3 += src0.y * DQ4(pw, 4, sc3);
    r3 += src0.z * DQ4(pw, 8, sc3);
    r3 += src0.w * DQ4(pw, 12, sc3);

    // Output group 4
    pw = vload4(0, weights + row0 + 16);
    r4 += src0.x * DQ4(pw, 0, sc4);
    r4 += src0.y * DQ4(pw, 4, sc4);
    r4 += src0.z * DQ4(pw, 8, sc4);
    r4 += src0.w * DQ4(pw, 12, sc4);

    // Output group 5
    pw = vload4(0, weights + row0 + 20);
    r5 += src0.x * DQ4(pw, 0, sc5);
    r5 += src0.y * DQ4(pw, 4, sc5);
    r5 += src0.z * DQ4(pw, 8, sc5);
    r5 += src0.w * DQ4(pw, 12, sc5);

    // Output group 6
    pw = vload4(0, weights + row0 + 24);
    r6 += src0.x * DQ4(pw, 0, sc6);
    r6 += src0.y * DQ4(pw, 4, sc6);
    r6 += src0.z * DQ4(pw, 8, sc6);
    r6 += src0.w * DQ4(pw, 12, sc6);

    // Output group 7
    pw = vload4(0, weights + row0 + 28);
    r7 += src0.x * DQ4(pw, 0, sc7);
    r7 += src0.y * DQ4(pw, 4, sc7);
    r7 += src0.z * DQ4(pw, 8, sc7);
    r7 += src0.w * DQ4(pw, 12, sc7);

    // Row 1: src1
    int row1 = (coord_s + 1) * N + base_n;

    pw = vload4(0, weights + row1);
    r0 += src1.x * DQ4(pw, 0, sc0);
    r0 += src1.y * DQ4(pw, 4, sc0);
    r0 += src1.z * DQ4(pw, 8, sc0);
    r0 += src1.w * DQ4(pw, 12, sc0);

    pw = vload4(0, weights + row1 + 4);
    r1 += src1.x * DQ4(pw, 0, sc1);
    r1 += src1.y * DQ4(pw, 4, sc1);
    r1 += src1.z * DQ4(pw, 8, sc1);
    r1 += src1.w * DQ4(pw, 12, sc1);

    pw = vload4(0, weights + row1 + 8);
    r2 += src1.x * DQ4(pw, 0, sc2);
    r2 += src1.y * DQ4(pw, 4, sc2);
    r2 += src1.z * DQ4(pw, 8, sc2);
    r2 += src1.w * DQ4(pw, 12, sc2);

    pw = vload4(0, weights + row1 + 12);
    r3 += src1.x * DQ4(pw, 0, sc3);
    r3 += src1.y * DQ4(pw, 4, sc3);
    r3 += src1.z * DQ4(pw, 8, sc3);
    r3 += src1.w * DQ4(pw, 12, sc3);

    pw = vload4(0, weights + row1 + 16);
    r4 += src1.x * DQ4(pw, 0, sc4);
    r4 += src1.y * DQ4(pw, 4, sc4);
    r4 += src1.z * DQ4(pw, 8, sc4);
    r4 += src1.w * DQ4(pw, 12, sc4);

    pw = vload4(0, weights + row1 + 20);
    r5 += src1.x * DQ4(pw, 0, sc5);
    r5 += src1.y * DQ4(pw, 4, sc5);
    r5 += src1.z * DQ4(pw, 8, sc5);
    r5 += src1.w * DQ4(pw, 12, sc5);

    pw = vload4(0, weights + row1 + 24);
    r6 += src1.x * DQ4(pw, 0, sc6);
    r6 += src1.y * DQ4(pw, 4, sc6);
    r6 += src1.z * DQ4(pw, 8, sc6);
    r6 += src1.w * DQ4(pw, 12, sc6);

    pw = vload4(0, weights + row1 + 28);
    r7 += src1.x * DQ4(pw, 0, sc7);
    r7 += src1.y * DQ4(pw, 4, sc7);
    r7 += src1.z * DQ4(pw, 8, sc7);
    r7 += src1.w * DQ4(pw, 12, sc7);

#undef DQ4

    coord_s += 2;
  } while (coord_s < src_slices);

  int out_s = Z * 8;
  if (out_s + 0 < dst_slices) write_imageh(dst_image, (int2)(X, out_s + 0), r0);
  if (out_s + 1 < dst_slices) write_imageh(dst_image, (int2)(X, out_s + 1), r1);
  if (out_s + 2 < dst_slices) write_imageh(dst_image, (int2)(X, out_s + 2), r2);
  if (out_s + 3 < dst_slices) write_imageh(dst_image, (int2)(X, out_s + 3), r3);
  if (out_s + 4 < dst_slices) write_imageh(dst_image, (int2)(X, out_s + 4), r4);
  if (out_s + 5 < dst_slices) write_imageh(dst_image, (int2)(X, out_s + 5), r5);
  if (out_s + 6 < dst_slices) write_imageh(dst_image, (int2)(X, out_s + 6), r6);
  if (out_s + 7 < dst_slices) write_imageh(dst_image, (int2)(X, out_s + 7), r7);
}
