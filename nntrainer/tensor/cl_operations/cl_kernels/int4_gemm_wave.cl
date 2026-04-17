#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// Int4 GEMM — delegate layout + texture cached weights + full unroll.
// Weights as image1d_buffer_t for texture cache (vs __global ushort*).
// Each thread: 8 output slices × 4 channels = 32 outputs.

__constant sampler_t smp_zero = CLK_NORMALIZED_COORDS_FALSE |
                                 CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

__kernel void gemm_int4_wave(
    __read_only image1d_buffer_t weights_img,
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

  // Weight image: RGBA ushort, width = N/4, pixel[i] = 4 ushorts
  // Index: row * (N/4) + col_group, where col_group = n/4
  // Each pixel has .x .y .z .w = weights for 4 consecutive n channels
  int w_col[8];
  for (int s = 0; s < 8; ++s)
    w_col[s] = (base_n + s * 4) / 4;

  int N4 = N / 4;

  int coord_s = 0;
  do {
    half4 src0 = read_imageh(src_image, smp_zero, (int2)(X, coord_s));
    half4 src1 = read_imageh(src_image, smp_zero, (int2)(X, coord_s + 1));

    // Row 0: packed weights at row coord_s
    int row0 = coord_s * N4;

    // Read 8 weight pixels from image (8 groups × 4 channels)
    // read_imageui returns uint4, .x/.y/.z/.w are the 4 ushort values
    uint4 pw0 = read_imageui(weights_img, row0 + w_col[0]);
    uint4 pw1 = read_imageui(weights_img, row0 + w_col[1]);
    uint4 pw2 = read_imageui(weights_img, row0 + w_col[2]);
    uint4 pw3 = read_imageui(weights_img, row0 + w_col[3]);
    uint4 pw4 = read_imageui(weights_img, row0 + w_col[4]);
    uint4 pw5 = read_imageui(weights_img, row0 + w_col[5]);
    uint4 pw6 = read_imageui(weights_img, row0 + w_col[6]);
    uint4 pw7 = read_imageui(weights_img, row0 + w_col[7]);

#define DQ(PW_COMP, SHIFT, SC_COMP) \
    (((half)((int)((PW_COMP >> SHIFT) & 0xFu) - 8)) * SC_COMP)

#define ACC_ROW(R, SRC, PW, SC) \
    R.x += SRC.x * DQ(PW.x, 0, SC.x) + SRC.y * DQ(PW.x, 4, SC.x) + SRC.z * DQ(PW.x, 8, SC.x) + SRC.w * DQ(PW.x, 12, SC.x); \
    R.y += SRC.x * DQ(PW.y, 0, SC.y) + SRC.y * DQ(PW.y, 4, SC.y) + SRC.z * DQ(PW.y, 8, SC.y) + SRC.w * DQ(PW.y, 12, SC.y); \
    R.z += SRC.x * DQ(PW.z, 0, SC.z) + SRC.y * DQ(PW.z, 4, SC.z) + SRC.z * DQ(PW.z, 8, SC.z) + SRC.w * DQ(PW.z, 12, SC.z); \
    R.w += SRC.x * DQ(PW.w, 0, SC.w) + SRC.y * DQ(PW.w, 4, SC.w) + SRC.z * DQ(PW.w, 8, SC.w) + SRC.w * DQ(PW.w, 12, SC.w);

    ACC_ROW(r0, src0, pw0, sc0)
    ACC_ROW(r1, src0, pw1, sc1)
    ACC_ROW(r2, src0, pw2, sc2)
    ACC_ROW(r3, src0, pw3, sc3)
    ACC_ROW(r4, src0, pw4, sc4)
    ACC_ROW(r5, src0, pw5, sc5)
    ACC_ROW(r6, src0, pw6, sc6)
    ACC_ROW(r7, src0, pw7, sc7)

    // Row 1
    int row1 = (coord_s + 1) * N4;
    pw0 = read_imageui(weights_img, row1 + w_col[0]);
    pw1 = read_imageui(weights_img, row1 + w_col[1]);
    pw2 = read_imageui(weights_img, row1 + w_col[2]);
    pw3 = read_imageui(weights_img, row1 + w_col[3]);
    pw4 = read_imageui(weights_img, row1 + w_col[4]);
    pw5 = read_imageui(weights_img, row1 + w_col[5]);
    pw6 = read_imageui(weights_img, row1 + w_col[6]);
    pw7 = read_imageui(weights_img, row1 + w_col[7]);

    ACC_ROW(r0, src1, pw0, sc0)
    ACC_ROW(r1, src1, pw1, sc1)
    ACC_ROW(r2, src1, pw2, sc2)
    ACC_ROW(r3, src1, pw3, sc3)
    ACC_ROW(r4, src1, pw4, sc4)
    ACC_ROW(r5, src1, pw5, sc5)
    ACC_ROW(r6, src1, pw6, sc6)
    ACC_ROW(r7, src1, pw7, sc7)

#undef DQ
#undef ACC_ROW

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
