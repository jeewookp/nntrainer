#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// Int4 GEMM — delegate layout + vectorized reads + ACC_ROW macro.
// Each thread: 8 output slices × 4 channels = 32 outputs.

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

  half4 sc0 = vload4(0, scales + base_n);
  half4 sc1 = vload4(0, scales + base_n + 4);
  half4 sc2 = vload4(0, scales + base_n + 8);
  half4 sc3 = vload4(0, scales + base_n + 12);
  half4 sc4 = vload4(0, scales + base_n + 16);
  half4 sc5 = vload4(0, scales + base_n + 20);
  half4 sc6 = vload4(0, scales + base_n + 24);
  half4 sc7 = vload4(0, scales + base_n + 28);

#define DQ(V, S, SC) (((half)((int)((V >> S) & 0xFu) - 8)) * SC)

#define ACC4(R, SRC, PW, SC) \
  R.x += SRC.x*DQ(PW.x,0,SC.x) + SRC.y*DQ(PW.x,4,SC.x) + SRC.z*DQ(PW.x,8,SC.x) + SRC.w*DQ(PW.x,12,SC.x); \
  R.y += SRC.x*DQ(PW.y,0,SC.y) + SRC.y*DQ(PW.y,4,SC.y) + SRC.z*DQ(PW.y,8,SC.y) + SRC.w*DQ(PW.y,12,SC.y); \
  R.z += SRC.x*DQ(PW.z,0,SC.z) + SRC.y*DQ(PW.z,4,SC.z) + SRC.z*DQ(PW.z,8,SC.z) + SRC.w*DQ(PW.z,12,SC.z); \
  R.w += SRC.x*DQ(PW.w,0,SC.w) + SRC.y*DQ(PW.w,4,SC.w) + SRC.z*DQ(PW.w,8,SC.w) + SRC.w*DQ(PW.w,12,SC.w);

  int coord_s = 0;
  do {
    half4 src0 = read_imageh(src_image, smp_zero, (int2)(X, coord_s));
    half4 src1 = read_imageh(src_image, smp_zero, (int2)(X, coord_s + 1));

    int row0 = coord_s * N + base_n;
    ushort4 pw;

    pw = vload4(0, weights + row0);      ACC4(r0, src0, pw, sc0)
    pw = vload4(0, weights + row0 + 4);  ACC4(r1, src0, pw, sc1)
    pw = vload4(0, weights + row0 + 8);  ACC4(r2, src0, pw, sc2)
    pw = vload4(0, weights + row0 + 12); ACC4(r3, src0, pw, sc3)
    pw = vload4(0, weights + row0 + 16); ACC4(r4, src0, pw, sc4)
    pw = vload4(0, weights + row0 + 20); ACC4(r5, src0, pw, sc5)
    pw = vload4(0, weights + row0 + 24); ACC4(r6, src0, pw, sc6)
    pw = vload4(0, weights + row0 + 28); ACC4(r7, src0, pw, sc7)

    int row1 = (coord_s + 1) * N + base_n;

    pw = vload4(0, weights + row1);      ACC4(r0, src1, pw, sc0)
    pw = vload4(0, weights + row1 + 4);  ACC4(r1, src1, pw, sc1)
    pw = vload4(0, weights + row1 + 8);  ACC4(r2, src1, pw, sc2)
    pw = vload4(0, weights + row1 + 12); ACC4(r3, src1, pw, sc3)
    pw = vload4(0, weights + row1 + 16); ACC4(r4, src1, pw, sc4)
    pw = vload4(0, weights + row1 + 20); ACC4(r5, src1, pw, sc5)
    pw = vload4(0, weights + row1 + 24); ACC4(r6, src1, pw, sc6)
    pw = vload4(0, weights + row1 + 28); ACC4(r7, src1, pw, sc7)

#undef DQ
#undef ACC4

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
