// delegate_conv_wave.cl의 EXACT 구조를 유지하면서 weight만 int4로 변경.
// 변경점: load 8 half8 (128B) instead of 32 (512B), then unpack int4 nibbles.
// 모든 나머지: 동일한 wave memory, 동일한 thread mapping, 동일한 asm.

#define MAIN_FUNCTION __kernel void gemm_int4_wave
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__constant sampler_t smp_zero = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
#pragma OPENCL EXTENSION cl_qcom_subgroup_uniform_load: enable
#pragma OPENCL EXTENSION cl_qcom_subgroup_constant_load: enable
#pragma OPENCL EXTENSION cl_qcom_inline_asm : enable

__attribute__((qcom_max_concurrent_subgroups(12)))
MAIN_FUNCTION(
  __constant half8* weights_buffer __attribute__((sub_group_uniform)),
  __constant half8* xmem_buffer __attribute__((max_constant_size((6144)))),
  __global const half* scales,
  __write_only image2d_t dst_tensor_image2d,
  __read_only image2d_t src_tensor_image2d,
  int4 shared_int4_0,
  int4 shared_int4_1,
  int4 shared_int4_2) {

  // EXACT delegate thread mapping
  int X = get_group_id(1) * get_local_size(0) + get_local_id(0);
  int Y = get_group_id(2) * get_local_size(1) + get_local_id(1);
  int Z = get_group_id(0) * get_local_size(2) + get_local_id(2);
  if (X >= shared_int4_0.z || Y >= shared_int4_0.x) return;
  if (Z * 8 >= shared_int4_0.y) return;

  half4 r0 = (half4)(0.0h);
  half4 r1 = (half4)(0.0h);
  half4 r2 = (half4)(0.0h);
  half4 r3 = (half4)(0.0h);
  half4 r4 = (half4)(0.0h);
  half4 r5 = (half4)(0.0h);
  half4 r6 = (half4)(0.0h);
  half4 r7 = (half4)(0.0h);

  // Per-channel scales (32 channels for this Z group)
  int base_n = Z * 32;
  half4 sc0 = vload4(0, scales + base_n);
  half4 sc1 = vload4(0, scales + base_n + 4);
  half4 sc2 = vload4(0, scales + base_n + 8);
  half4 sc3 = vload4(0, scales + base_n + 12);
  half4 sc4 = vload4(0, scales + base_n + 16);
  half4 sc5 = vload4(0, scales + base_n + 20);
  half4 sc6 = vload4(0, scales + base_n + 24);
  half4 sc7 = vload4(0, scales + base_n + 28);

  // EXACT delegate wave memory setup
  int x_coord = mad24(X, shared_int4_2.x, shared_int4_1.y);
  int y_coord = mad24(Y, shared_int4_2.y, shared_int4_1.z);
  int subgroup_id = (int)((0x1F & qcom_get_physical_sub_group_id()));
  subgroup_id = subgroup_id % 12;
  int c_offset = mul24(subgroup_id, shared_int4_0.w);
  short a0 = c_offset * 8;

  __asm__ __volatile__(
    "mova a0, 256;"
    "mova a0, %[a0];"
    "(rpt5)nop ;"
    :
    : [a0] "r" (a0)
    :
  );

  // View wave memory as ushort* for int4 unpacking
  __constant ushort* w_us = (__constant ushort*)&xmem_buffer[c_offset];

  // f_offset in half8 units. Int4 data is packed into the half8 buffer:
  // Per iteration: 64 ushorts = 128 bytes = 8 half8 (vs 32 for fp16)
  int f_offset = Z * shared_int4_1.x * 8;

  int coord_y = Y;
  int coord_x = X;
  int coord_s = 0;
  do {
    // Read src (identical to delegate)
    half4 src0 = read_imageh(src_tensor_image2d, smp_zero,
                             (int2)(coord_x, coord_y * shared_int4_1.w + coord_s));
    coord_s++;
    half4 src1 = read_imageh(src_tensor_image2d, smp_zero,
                             (int2)(coord_x, coord_y * shared_int4_1.w + coord_s));
    coord_s++;

    // Load 8 half8 of packed int4 data via wave memory
    qcom_sub_group_constant_load8(xmem_buffer, weights_buffer,
                                   c_offset, f_offset >> 1, 8);
    f_offset += 16;
    qcom_sub_group_sync(QCOM_CLK_CONST_LOAD_SYNC);

    // w_us[0..63]: 64 ushorts = 2 rows × 32 channels
    // Row 0 (w_us[0..31]): src0 k-values, Row 1 (w_us[32..63]): src1

#define DQ(IDX, SHIFT, SC) (((half)((int)((w_us[IDX] >> SHIFT) & 0xFu) - 8)) * SC)

    // Row 0: src0 × dequant(row0)
    r0.x += src0.x*DQ(0,0,sc0.x) + src0.y*DQ(0,4,sc0.x) + src0.z*DQ(0,8,sc0.x) + src0.w*DQ(0,12,sc0.x);
    r0.y += src0.x*DQ(1,0,sc0.y) + src0.y*DQ(1,4,sc0.y) + src0.z*DQ(1,8,sc0.y) + src0.w*DQ(1,12,sc0.y);
    r0.z += src0.x*DQ(2,0,sc0.z) + src0.y*DQ(2,4,sc0.z) + src0.z*DQ(2,8,sc0.z) + src0.w*DQ(2,12,sc0.z);
    r0.w += src0.x*DQ(3,0,sc0.w) + src0.y*DQ(3,4,sc0.w) + src0.z*DQ(3,8,sc0.w) + src0.w*DQ(3,12,sc0.w);

    r1.x += src0.x*DQ(4,0,sc1.x) + src0.y*DQ(4,4,sc1.x) + src0.z*DQ(4,8,sc1.x) + src0.w*DQ(4,12,sc1.x);
    r1.y += src0.x*DQ(5,0,sc1.y) + src0.y*DQ(5,4,sc1.y) + src0.z*DQ(5,8,sc1.y) + src0.w*DQ(5,12,sc1.y);
    r1.z += src0.x*DQ(6,0,sc1.z) + src0.y*DQ(6,4,sc1.z) + src0.z*DQ(6,8,sc1.z) + src0.w*DQ(6,12,sc1.z);
    r1.w += src0.x*DQ(7,0,sc1.w) + src0.y*DQ(7,4,sc1.w) + src0.z*DQ(7,8,sc1.w) + src0.w*DQ(7,12,sc1.w);

    r2.x += src0.x*DQ(8,0,sc2.x) + src0.y*DQ(8,4,sc2.x) + src0.z*DQ(8,8,sc2.x) + src0.w*DQ(8,12,sc2.x);
    r2.y += src0.x*DQ(9,0,sc2.y) + src0.y*DQ(9,4,sc2.y) + src0.z*DQ(9,8,sc2.y) + src0.w*DQ(9,12,sc2.y);
    r2.z += src0.x*DQ(10,0,sc2.z) + src0.y*DQ(10,4,sc2.z) + src0.z*DQ(10,8,sc2.z) + src0.w*DQ(10,12,sc2.z);
    r2.w += src0.x*DQ(11,0,sc2.w) + src0.y*DQ(11,4,sc2.w) + src0.z*DQ(11,8,sc2.w) + src0.w*DQ(11,12,sc2.w);

    r3.x += src0.x*DQ(12,0,sc3.x) + src0.y*DQ(12,4,sc3.x) + src0.z*DQ(12,8,sc3.x) + src0.w*DQ(12,12,sc3.x);
    r3.y += src0.x*DQ(13,0,sc3.y) + src0.y*DQ(13,4,sc3.y) + src0.z*DQ(13,8,sc3.y) + src0.w*DQ(13,12,sc3.y);
    r3.z += src0.x*DQ(14,0,sc3.z) + src0.y*DQ(14,4,sc3.z) + src0.z*DQ(14,8,sc3.z) + src0.w*DQ(14,12,sc3.z);
    r3.w += src0.x*DQ(15,0,sc3.w) + src0.y*DQ(15,4,sc3.w) + src0.z*DQ(15,8,sc3.w) + src0.w*DQ(15,12,sc3.w);

    r4.x += src0.x*DQ(16,0,sc4.x) + src0.y*DQ(16,4,sc4.x) + src0.z*DQ(16,8,sc4.x) + src0.w*DQ(16,12,sc4.x);
    r4.y += src0.x*DQ(17,0,sc4.y) + src0.y*DQ(17,4,sc4.y) + src0.z*DQ(17,8,sc4.y) + src0.w*DQ(17,12,sc4.y);
    r4.z += src0.x*DQ(18,0,sc4.z) + src0.y*DQ(18,4,sc4.z) + src0.z*DQ(18,8,sc4.z) + src0.w*DQ(18,12,sc4.z);
    r4.w += src0.x*DQ(19,0,sc4.w) + src0.y*DQ(19,4,sc4.w) + src0.z*DQ(19,8,sc4.w) + src0.w*DQ(19,12,sc4.w);

    r5.x += src0.x*DQ(20,0,sc5.x) + src0.y*DQ(20,4,sc5.x) + src0.z*DQ(20,8,sc5.x) + src0.w*DQ(20,12,sc5.x);
    r5.y += src0.x*DQ(21,0,sc5.y) + src0.y*DQ(21,4,sc5.y) + src0.z*DQ(21,8,sc5.y) + src0.w*DQ(21,12,sc5.y);
    r5.z += src0.x*DQ(22,0,sc5.z) + src0.y*DQ(22,4,sc5.z) + src0.z*DQ(22,8,sc5.z) + src0.w*DQ(22,12,sc5.z);
    r5.w += src0.x*DQ(23,0,sc5.w) + src0.y*DQ(23,4,sc5.w) + src0.z*DQ(23,8,sc5.w) + src0.w*DQ(23,12,sc5.w);

    r6.x += src0.x*DQ(24,0,sc6.x) + src0.y*DQ(24,4,sc6.x) + src0.z*DQ(24,8,sc6.x) + src0.w*DQ(24,12,sc6.x);
    r6.y += src0.x*DQ(25,0,sc6.y) + src0.y*DQ(25,4,sc6.y) + src0.z*DQ(25,8,sc6.y) + src0.w*DQ(25,12,sc6.y);
    r6.z += src0.x*DQ(26,0,sc6.z) + src0.y*DQ(26,4,sc6.z) + src0.z*DQ(26,8,sc6.z) + src0.w*DQ(26,12,sc6.z);
    r6.w += src0.x*DQ(27,0,sc6.w) + src0.y*DQ(27,4,sc6.w) + src0.z*DQ(27,8,sc6.w) + src0.w*DQ(27,12,sc6.w);

    r7.x += src0.x*DQ(28,0,sc7.x) + src0.y*DQ(28,4,sc7.x) + src0.z*DQ(28,8,sc7.x) + src0.w*DQ(28,12,sc7.x);
    r7.y += src0.x*DQ(29,0,sc7.y) + src0.y*DQ(29,4,sc7.y) + src0.z*DQ(29,8,sc7.y) + src0.w*DQ(29,12,sc7.y);
    r7.z += src0.x*DQ(30,0,sc7.z) + src0.y*DQ(30,4,sc7.z) + src0.z*DQ(30,8,sc7.z) + src0.w*DQ(30,12,sc7.z);
    r7.w += src0.x*DQ(31,0,sc7.w) + src0.y*DQ(31,4,sc7.w) + src0.z*DQ(31,8,sc7.w) + src0.w*DQ(31,12,sc7.w);

    // Row 1: src1
    r0.x += src1.x*DQ(32,0,sc0.x) + src1.y*DQ(32,4,sc0.x) + src1.z*DQ(32,8,sc0.x) + src1.w*DQ(32,12,sc0.x);
    r0.y += src1.x*DQ(33,0,sc0.y) + src1.y*DQ(33,4,sc0.y) + src1.z*DQ(33,8,sc0.y) + src1.w*DQ(33,12,sc0.y);
    r0.z += src1.x*DQ(34,0,sc0.z) + src1.y*DQ(34,4,sc0.z) + src1.z*DQ(34,8,sc0.z) + src1.w*DQ(34,12,sc0.z);
    r0.w += src1.x*DQ(35,0,sc0.w) + src1.y*DQ(35,4,sc0.w) + src1.z*DQ(35,8,sc0.w) + src1.w*DQ(35,12,sc0.w);

    r1.x += src1.x*DQ(36,0,sc1.x) + src1.y*DQ(36,4,sc1.x) + src1.z*DQ(36,8,sc1.x) + src1.w*DQ(36,12,sc1.x);
    r1.y += src1.x*DQ(37,0,sc1.y) + src1.y*DQ(37,4,sc1.y) + src1.z*DQ(37,8,sc1.y) + src1.w*DQ(37,12,sc1.y);
    r1.z += src1.x*DQ(38,0,sc1.z) + src1.y*DQ(38,4,sc1.z) + src1.z*DQ(38,8,sc1.z) + src1.w*DQ(38,12,sc1.z);
    r1.w += src1.x*DQ(39,0,sc1.w) + src1.y*DQ(39,4,sc1.w) + src1.z*DQ(39,8,sc1.w) + src1.w*DQ(39,12,sc1.w);

    r2.x += src1.x*DQ(40,0,sc2.x) + src1.y*DQ(40,4,sc2.x) + src1.z*DQ(40,8,sc2.x) + src1.w*DQ(40,12,sc2.x);
    r2.y += src1.x*DQ(41,0,sc2.y) + src1.y*DQ(41,4,sc2.y) + src1.z*DQ(41,8,sc2.y) + src1.w*DQ(41,12,sc2.y);
    r2.z += src1.x*DQ(42,0,sc2.z) + src1.y*DQ(42,4,sc2.z) + src1.z*DQ(42,8,sc2.z) + src1.w*DQ(42,12,sc2.z);
    r2.w += src1.x*DQ(43,0,sc2.w) + src1.y*DQ(43,4,sc2.w) + src1.z*DQ(43,8,sc2.w) + src1.w*DQ(43,12,sc2.w);

    r3.x += src1.x*DQ(44,0,sc3.x) + src1.y*DQ(44,4,sc3.x) + src1.z*DQ(44,8,sc3.x) + src1.w*DQ(44,12,sc3.x);
    r3.y += src1.x*DQ(45,0,sc3.y) + src1.y*DQ(45,4,sc3.y) + src1.z*DQ(45,8,sc3.y) + src1.w*DQ(45,12,sc3.y);
    r3.z += src1.x*DQ(46,0,sc3.z) + src1.y*DQ(46,4,sc3.z) + src1.z*DQ(46,8,sc3.z) + src1.w*DQ(46,12,sc3.z);
    r3.w += src1.x*DQ(47,0,sc3.w) + src1.y*DQ(47,4,sc3.w) + src1.z*DQ(47,8,sc3.w) + src1.w*DQ(47,12,sc3.w);

    r4.x += src1.x*DQ(48,0,sc4.x) + src1.y*DQ(48,4,sc4.x) + src1.z*DQ(48,8,sc4.x) + src1.w*DQ(48,12,sc4.x);
    r4.y += src1.x*DQ(49,0,sc4.y) + src1.y*DQ(49,4,sc4.y) + src1.z*DQ(49,8,sc4.y) + src1.w*DQ(49,12,sc4.y);
    r4.z += src1.x*DQ(50,0,sc4.z) + src1.y*DQ(50,4,sc4.z) + src1.z*DQ(50,8,sc4.z) + src1.w*DQ(50,12,sc4.z);
    r4.w += src1.x*DQ(51,0,sc4.w) + src1.y*DQ(51,4,sc4.w) + src1.z*DQ(51,8,sc4.w) + src1.w*DQ(51,12,sc4.w);

    r5.x += src1.x*DQ(52,0,sc5.x) + src1.y*DQ(52,4,sc5.x) + src1.z*DQ(52,8,sc5.x) + src1.w*DQ(52,12,sc5.x);
    r5.y += src1.x*DQ(53,0,sc5.y) + src1.y*DQ(53,4,sc5.y) + src1.z*DQ(53,8,sc5.y) + src1.w*DQ(53,12,sc5.y);
    r5.z += src1.x*DQ(54,0,sc5.z) + src1.y*DQ(54,4,sc5.z) + src1.z*DQ(54,8,sc5.z) + src1.w*DQ(54,12,sc5.z);
    r5.w += src1.x*DQ(55,0,sc5.w) + src1.y*DQ(55,4,sc5.w) + src1.z*DQ(55,8,sc5.w) + src1.w*DQ(55,12,sc5.w);

    r6.x += src1.x*DQ(56,0,sc6.x) + src1.y*DQ(56,4,sc6.x) + src1.z*DQ(56,8,sc6.x) + src1.w*DQ(56,12,sc6.x);
    r6.y += src1.x*DQ(57,0,sc6.y) + src1.y*DQ(57,4,sc6.y) + src1.z*DQ(57,8,sc6.y) + src1.w*DQ(57,12,sc6.y);
    r6.z += src1.x*DQ(58,0,sc6.z) + src1.y*DQ(58,4,sc6.z) + src1.z*DQ(58,8,sc6.z) + src1.w*DQ(58,12,sc6.z);
    r6.w += src1.x*DQ(59,0,sc6.w) + src1.y*DQ(59,4,sc6.w) + src1.z*DQ(59,8,sc6.w) + src1.w*DQ(59,12,sc6.w);

    r7.x += src1.x*DQ(60,0,sc7.x) + src1.y*DQ(60,4,sc7.x) + src1.z*DQ(60,8,sc7.x) + src1.w*DQ(60,12,sc7.x);
    r7.y += src1.x*DQ(61,0,sc7.y) + src1.y*DQ(61,4,sc7.y) + src1.z*DQ(61,8,sc7.y) + src1.w*DQ(61,12,sc7.y);
    r7.z += src1.x*DQ(62,0,sc7.z) + src1.y*DQ(62,4,sc7.z) + src1.z*DQ(62,8,sc7.z) + src1.w*DQ(62,12,sc7.z);
    r7.w += src1.x*DQ(63,0,sc7.w) + src1.y*DQ(63,4,sc7.w) + src1.z*DQ(63,8,sc7.w) + src1.w*DQ(63,12,sc7.w);

#undef DQ

  } while (coord_s < shared_int4_1.w);

  // Write output (identical to delegate)
  int coord_s_out = Z * 8;
  if (coord_s_out < shared_int4_0.y)
    write_imageh(dst_tensor_image2d, (int2)(X, Y * shared_int4_0.y + coord_s_out), r0);
  coord_s_out++;
  if (coord_s_out < shared_int4_0.y)
    write_imageh(dst_tensor_image2d, (int2)(X, Y * shared_int4_0.y + coord_s_out), r1);
  coord_s_out++;
  if (coord_s_out < shared_int4_0.y)
    write_imageh(dst_tensor_image2d, (int2)(X, Y * shared_int4_0.y + coord_s_out), r2);
  coord_s_out++;
  if (coord_s_out < shared_int4_0.y)
    write_imageh(dst_tensor_image2d, (int2)(X, Y * shared_int4_0.y + coord_s_out), r3);
  coord_s_out++;
  if (coord_s_out < shared_int4_0.y)
    write_imageh(dst_tensor_image2d, (int2)(X, Y * shared_int4_0.y + coord_s_out), r4);
  coord_s_out++;
  if (coord_s_out < shared_int4_0.y)
    write_imageh(dst_tensor_image2d, (int2)(X, Y * shared_int4_0.y + coord_s_out), r5);
  coord_s_out++;
  if (coord_s_out < shared_int4_0.y)
    write_imageh(dst_tensor_image2d, (int2)(X, Y * shared_int4_0.y + coord_s_out), r6);
  coord_s_out++;
  if (coord_s_out < shared_int4_0.y)
    write_imageh(dst_tensor_image2d, (int2)(X, Y * shared_int4_0.y + coord_s_out), r7);
}
