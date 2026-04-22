// r0-only probe of delegate_conv_int4_dense.cl — 1/8 of the compute block.
//
// Diagnostic: the full 8-accumulator kernel runs 800x slower than the fp16
// delegate reference on Adreno 830 despite matching math (rel_l2=0), which
// strongly suggests register pressure / spill rather than ALU throughput.
//
// This kernel is IDENTICAL to delegate_conv_int4_dense.cl except:
//   - Only r0 is accumulated (weights ushorts[0..3] and [32..35] are used;
//     ushorts[4..31] and [36..63] are still LOADED by count=8 so the load
//     cost is preserved — only compute is shrunk).
//   - Only sc0 is preloaded (saves 7 half4 registers).
//   - Output writes only coord_s = Z*8 (first out_ch in the 8-lane block).
//
// Interpretation:
//   us_r0 ~ full_us / 8   → linear scaling, basic-block size is the knob
//   us_r0 << full_us / 8  → register-pressure cliff at the full kernel
//
// Correctness is checked only on out_ch where (oc % 8 == 0); others are
// left zero by this kernel.

#define MAIN_FUNCTION __kernel void delegate_conv_int4_dense_r0
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
  if (Z * 8 >= shared_int4_0.y) return;

  half4 r0 = (half4)(0.f);

  const int base_n = Z * 32;
  const half4 sc0 = vload4(0, scales + base_n + 0);

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

  // r0 only, src0 (ushorts[0..3]).
  r0 += src0.x * (half4)(DQ( 0, 0,sc0.x), DQ( 1, 0,sc0.y), DQ( 2, 0,sc0.z), DQ( 3, 0,sc0.w));
  r0 += src0.y * (half4)(DQ( 0, 4,sc0.x), DQ( 1, 4,sc0.y), DQ( 2, 4,sc0.z), DQ( 3, 4,sc0.w));
  r0 += src0.z * (half4)(DQ( 0, 8,sc0.x), DQ( 1, 8,sc0.y), DQ( 2, 8,sc0.z), DQ( 3, 8,sc0.w));
  r0 += src0.w * (half4)(DQ( 0,12,sc0.x), DQ( 1,12,sc0.y), DQ( 2,12,sc0.z), DQ( 3,12,sc0.w));
  // r0 only, src1 (ushorts[32..35]).
  r0 += src1.x * (half4)(DQ(32, 0,sc0.x), DQ(33, 0,sc0.y), DQ(34, 0,sc0.z), DQ(35, 0,sc0.w));
  r0 += src1.y * (half4)(DQ(32, 4,sc0.x), DQ(33, 4,sc0.y), DQ(34, 4,sc0.z), DQ(35, 4,sc0.w));
  r0 += src1.z * (half4)(DQ(32, 8,sc0.x), DQ(33, 8,sc0.y), DQ(34, 8,sc0.z), DQ(35, 8,sc0.w));
  r0 += src1.w * (half4)(DQ(32,12,sc0.x), DQ(33,12,sc0.y), DQ(34,12,sc0.z), DQ(35,12,sc0.w));

#undef DQ
      } while (coord_s < shared_int4_1.w);

  coord_s = mul24(Z, 8);
  coord_x = X;
  coord_y = Y;
  if (coord_s < shared_int4_0.y) {
    half4 res = r0;
    res += read_imageh(biases_image2d, smp_zero, (int2)((coord_s), 0));
    write_imageh(dst_tensor_image2d, (int2)((coord_x), ((coord_y) * shared_int4_0.y + (coord_s))), res);
  }
}
