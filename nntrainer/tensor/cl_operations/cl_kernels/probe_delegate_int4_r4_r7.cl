// r4..r7 probe — second half of the full int4-dense compute.
//
// Pair with probe_delegate_int4_r0_r3.cl (first half) to realise the
// full 8-accumulator output by dispatching twice on the same weights,
// sidestepping the register-pressure cliff observed at 8 live acc.
//
// Same dense 64-ushort/block layout — per iter loads ushorts[0..63] via
// count=8 but only uses [16..31] (src0) and [48..63] (src1), producing
// output channels z*8+4..z*8+7.

#define MAIN_FUNCTION __kernel void delegate_conv_int4_dense_r4_r7
#define bool2 uchar2
#define bool3 uchar3
#define bool4 uchar4
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__constant sampler_t smp_none = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;
__constant sampler_t smp_zero = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
#pragma OPENCL EXTENSION cl_qcom_subgroup_uniform_load: enable
#pragma OPENCL EXTENSION cl_qcom_subgroup_constant_load: enable
#pragma OPENCL EXTENSION cl_qcom_inline_asm : enable

__attribute__((qcom_max_concurrent_subgroups(12)))
MAIN_FUNCTION(__constant half8* weights_buffer  __attribute__((sub_group_uniform)),
  __constant half8* xmem_buffer  __attribute__((max_constant_size((6144)))),
  __global const half* restrict scales,
  __read_only image2d_t biases_image2d,
  __write_only image2d_t dst_tensor_image2d,
  __read_only image2d_t src_tensor_image2d,
  int4 shared_int4_0,
  int4 shared_int4_1,
  int4 shared_int4_2) {
  int X = get_group_id(1) * get_local_size(0) + get_local_id(0);
  int Y = get_group_id(2) * get_local_size(1) + get_local_id(1);
  int Z = get_group_id(0) * get_local_size(2) + get_local_id(2);
  if (X >= shared_int4_0.z || Y >= shared_int4_0.x) return;
  if (Z * 8 + 4 >= shared_int4_0.y) return;

  half4 r4 = (half4)(0.f);
  half4 r5 = (half4)(0.f);
  half4 r6 = (half4)(0.f);
  half4 r7 = (half4)(0.f);

  const int base_n = Z * 32;
  const half4 sc4 = vload4(0, scales + base_n + 16);
  const half4 sc5 = vload4(0, scales + base_n + 20);
  const half4 sc6 = vload4(0, scales + base_n + 24);
  const half4 sc7 = vload4(0, scales + base_n + 28);

  int coord_x, coord_y, coord_s;
  int subgroup_id = (int)((0x1F & qcom_get_physical_sub_group_id()));
  subgroup_id = subgroup_id % 12;
  int c_offset = mul24(subgroup_id, shared_int4_0.w);
  short a0 = c_offset * 8;

__asm__ __volatile__(
  "mova a0, 256;"
  "mova a0, %[a0];"
  "(rpt5)nop ;"
  :
  : [a0]  "r" (a0)
  :
);
  __constant ushort* w_cache = (__constant ushort*)&xmem_buffer[c_offset];

  int f_offset = Z * shared_int4_1.x * 8;

  coord_y = Y;
  coord_x = X;
      coord_s = 0;
      do {
        half4 src0 = read_imageh(src_tensor_image2d, smp_zero, (int2)((coord_x), ((coord_y) * shared_int4_1.w + (coord_s))));
        coord_s++;
        half4 src1 = read_imageh(src_tensor_image2d, smp_zero, (int2)((coord_x), ((coord_y) * shared_int4_1.w + (coord_s))));
        coord_s++;
        qcom_sub_group_constant_load8(xmem_buffer, weights_buffer, c_offset, f_offset >> 1, 8);
        f_offset += 16;
        qcom_sub_group_sync(QCOM_CLK_CONST_LOAD_SYNC);

#define DQ(I,S,SC) (((half)((int)((w_cache[I] >> S) & 0xFu) - 8)) * SC)

  // Row 0 (src0): r4..r7 using ushorts[16..31].
  r4 += src0.x * (half4)(DQ(16, 0,sc4.x), DQ(17, 0,sc4.y), DQ(18, 0,sc4.z), DQ(19, 0,sc4.w));
  r4 += src0.y * (half4)(DQ(16, 4,sc4.x), DQ(17, 4,sc4.y), DQ(18, 4,sc4.z), DQ(19, 4,sc4.w));
  r4 += src0.z * (half4)(DQ(16, 8,sc4.x), DQ(17, 8,sc4.y), DQ(18, 8,sc4.z), DQ(19, 8,sc4.w));
  r4 += src0.w * (half4)(DQ(16,12,sc4.x), DQ(17,12,sc4.y), DQ(18,12,sc4.z), DQ(19,12,sc4.w));
  r5 += src0.x * (half4)(DQ(20, 0,sc5.x), DQ(21, 0,sc5.y), DQ(22, 0,sc5.z), DQ(23, 0,sc5.w));
  r5 += src0.y * (half4)(DQ(20, 4,sc5.x), DQ(21, 4,sc5.y), DQ(22, 4,sc5.z), DQ(23, 4,sc5.w));
  r5 += src0.z * (half4)(DQ(20, 8,sc5.x), DQ(21, 8,sc5.y), DQ(22, 8,sc5.z), DQ(23, 8,sc5.w));
  r5 += src0.w * (half4)(DQ(20,12,sc5.x), DQ(21,12,sc5.y), DQ(22,12,sc5.z), DQ(23,12,sc5.w));
  r6 += src0.x * (half4)(DQ(24, 0,sc6.x), DQ(25, 0,sc6.y), DQ(26, 0,sc6.z), DQ(27, 0,sc6.w));
  r6 += src0.y * (half4)(DQ(24, 4,sc6.x), DQ(25, 4,sc6.y), DQ(26, 4,sc6.z), DQ(27, 4,sc6.w));
  r6 += src0.z * (half4)(DQ(24, 8,sc6.x), DQ(25, 8,sc6.y), DQ(26, 8,sc6.z), DQ(27, 8,sc6.w));
  r6 += src0.w * (half4)(DQ(24,12,sc6.x), DQ(25,12,sc6.y), DQ(26,12,sc6.z), DQ(27,12,sc6.w));
  r7 += src0.x * (half4)(DQ(28, 0,sc7.x), DQ(29, 0,sc7.y), DQ(30, 0,sc7.z), DQ(31, 0,sc7.w));
  r7 += src0.y * (half4)(DQ(28, 4,sc7.x), DQ(29, 4,sc7.y), DQ(30, 4,sc7.z), DQ(31, 4,sc7.w));
  r7 += src0.z * (half4)(DQ(28, 8,sc7.x), DQ(29, 8,sc7.y), DQ(30, 8,sc7.z), DQ(31, 8,sc7.w));
  r7 += src0.w * (half4)(DQ(28,12,sc7.x), DQ(29,12,sc7.y), DQ(30,12,sc7.z), DQ(31,12,sc7.w));

  // Row 1 (src1): r4..r7 using ushorts[48..63].
  r4 += src1.x * (half4)(DQ(48, 0,sc4.x), DQ(49, 0,sc4.y), DQ(50, 0,sc4.z), DQ(51, 0,sc4.w));
  r4 += src1.y * (half4)(DQ(48, 4,sc4.x), DQ(49, 4,sc4.y), DQ(50, 4,sc4.z), DQ(51, 4,sc4.w));
  r4 += src1.z * (half4)(DQ(48, 8,sc4.x), DQ(49, 8,sc4.y), DQ(50, 8,sc4.z), DQ(51, 8,sc4.w));
  r4 += src1.w * (half4)(DQ(48,12,sc4.x), DQ(49,12,sc4.y), DQ(50,12,sc4.z), DQ(51,12,sc4.w));
  r5 += src1.x * (half4)(DQ(52, 0,sc5.x), DQ(53, 0,sc5.y), DQ(54, 0,sc5.z), DQ(55, 0,sc5.w));
  r5 += src1.y * (half4)(DQ(52, 4,sc5.x), DQ(53, 4,sc5.y), DQ(54, 4,sc5.z), DQ(55, 4,sc5.w));
  r5 += src1.z * (half4)(DQ(52, 8,sc5.x), DQ(53, 8,sc5.y), DQ(54, 8,sc5.z), DQ(55, 8,sc5.w));
  r5 += src1.w * (half4)(DQ(52,12,sc5.x), DQ(53,12,sc5.y), DQ(54,12,sc5.z), DQ(55,12,sc5.w));
  r6 += src1.x * (half4)(DQ(56, 0,sc6.x), DQ(57, 0,sc6.y), DQ(58, 0,sc6.z), DQ(59, 0,sc6.w));
  r6 += src1.y * (half4)(DQ(56, 4,sc6.x), DQ(57, 4,sc6.y), DQ(58, 4,sc6.z), DQ(59, 4,sc6.w));
  r6 += src1.z * (half4)(DQ(56, 8,sc6.x), DQ(57, 8,sc6.y), DQ(58, 8,sc6.z), DQ(59, 8,sc6.w));
  r6 += src1.w * (half4)(DQ(56,12,sc6.x), DQ(57,12,sc6.y), DQ(58,12,sc6.z), DQ(59,12,sc6.w));
  r7 += src1.x * (half4)(DQ(60, 0,sc7.x), DQ(61, 0,sc7.y), DQ(62, 0,sc7.z), DQ(63, 0,sc7.w));
  r7 += src1.y * (half4)(DQ(60, 4,sc7.x), DQ(61, 4,sc7.y), DQ(62, 4,sc7.z), DQ(63, 4,sc7.w));
  r7 += src1.z * (half4)(DQ(60, 8,sc7.x), DQ(61, 8,sc7.y), DQ(62, 8,sc7.z), DQ(63, 8,sc7.w));
  r7 += src1.w * (half4)(DQ(60,12,sc7.x), DQ(61,12,sc7.y), DQ(62,12,sc7.z), DQ(63,12,sc7.w));

#undef DQ
      } while (coord_s < shared_int4_1.w);

  coord_s = mul24(Z, 8) + 4;
  coord_x = X;
  coord_y = Y;
  if (coord_s < shared_int4_0.y) {
    half4 res = r4;
    res += read_imageh(biases_image2d, smp_zero, (int2)((coord_s), 0));
    write_imageh(dst_tensor_image2d, (int2)((coord_x), ((coord_y) * shared_int4_0.y + (coord_s))), res);
    coord_s++;
  }
  if (coord_s < shared_int4_0.y) {
    half4 res = r5;
    res += read_imageh(biases_image2d, smp_zero, (int2)((coord_s), 0));
    write_imageh(dst_tensor_image2d, (int2)((coord_x), ((coord_y) * shared_int4_0.y + (coord_s))), res);
    coord_s++;
  }
  if (coord_s < shared_int4_0.y) {
    half4 res = r6;
    res += read_imageh(biases_image2d, smp_zero, (int2)((coord_s), 0));
    write_imageh(dst_tensor_image2d, (int2)((coord_x), ((coord_y) * shared_int4_0.y + (coord_s))), res);
    coord_s++;
  }
  if (coord_s < shared_int4_0.y) {
    half4 res = r7;
    res += read_imageh(biases_image2d, smp_zero, (int2)((coord_s), 0));
    write_imageh(dst_tensor_image2d, (int2)((coord_x), ((coord_y) * shared_int4_0.y + (coord_s))), res);
  }
}
