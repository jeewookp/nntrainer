// r0..r3 with dequant-to-local phase separation.
//
// Same compute shape as probe_delegate_int4_r0_r3.cl (4 acc, 1/2 full work)
// but the per-iter body is split so the compiler sees a clean dep graph:
//
//   Phase 1: dequant the 32 half4 weight vectors this iter needs into
//            named private variables (w_s0_r0_0 ... w_s1_r3_c). Depends
//            only on w_cache and scales, independent of accumulators.
//
//   Phase 2: 32 pure half4 FMAs updating r0..r3. Reads the dequanted
//            names, no shift/AND/cast ops mixed in.
//
// Goal: beat the 2001 us baseline of the inline-DQ r0..r3 probe. If the
// compiler pipelines phase 1 behind phase 2's FMA latency we get closer
// to the fp16 delegate's ALU efficiency.
//
// Peak live halves in private regs during phase 2:
//   32 half4 weights (128) + 4 acc (16) + 4 scales (16) + src0+src1 (8)
//   = ~168 halves. Inside Adreno 830's per-thread register budget.

#define MAIN_FUNCTION __kernel void delegate_conv_int4_dense_r0_r3_phased
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
  half4 r1 = (half4)(0.f);
  half4 r2 = (half4)(0.f);
  half4 r3 = (half4)(0.f);

  const int base_n = Z * 32;
  const half4 sc0 = vload4(0, scales + base_n +  0);
  const half4 sc1 = vload4(0, scales + base_n +  4);
  const half4 sc2 = vload4(0, scales + base_n +  8);
  const half4 sc3 = vload4(0, scales + base_n + 12);

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
#define DQ4(I0,I1,I2,I3,S,SCx,SCy,SCz,SCw) \
  (half4)(DQ(I0,S,SCx), DQ(I1,S,SCy), DQ(I2,S,SCz), DQ(I3,S,SCw))

  // ===== Phase 1: dequant all 32 half4 into named private regs =====
  // src0 side — ushorts[0..15], one half4 per (r,shift).
  half4 w_s0_r0_0 = DQ4( 0, 1, 2, 3,  0, sc0.x, sc0.y, sc0.z, sc0.w);
  half4 w_s0_r0_4 = DQ4( 0, 1, 2, 3,  4, sc0.x, sc0.y, sc0.z, sc0.w);
  half4 w_s0_r0_8 = DQ4( 0, 1, 2, 3,  8, sc0.x, sc0.y, sc0.z, sc0.w);
  half4 w_s0_r0_c = DQ4( 0, 1, 2, 3, 12, sc0.x, sc0.y, sc0.z, sc0.w);
  half4 w_s0_r1_0 = DQ4( 4, 5, 6, 7,  0, sc1.x, sc1.y, sc1.z, sc1.w);
  half4 w_s0_r1_4 = DQ4( 4, 5, 6, 7,  4, sc1.x, sc1.y, sc1.z, sc1.w);
  half4 w_s0_r1_8 = DQ4( 4, 5, 6, 7,  8, sc1.x, sc1.y, sc1.z, sc1.w);
  half4 w_s0_r1_c = DQ4( 4, 5, 6, 7, 12, sc1.x, sc1.y, sc1.z, sc1.w);
  half4 w_s0_r2_0 = DQ4( 8, 9,10,11,  0, sc2.x, sc2.y, sc2.z, sc2.w);
  half4 w_s0_r2_4 = DQ4( 8, 9,10,11,  4, sc2.x, sc2.y, sc2.z, sc2.w);
  half4 w_s0_r2_8 = DQ4( 8, 9,10,11,  8, sc2.x, sc2.y, sc2.z, sc2.w);
  half4 w_s0_r2_c = DQ4( 8, 9,10,11, 12, sc2.x, sc2.y, sc2.z, sc2.w);
  half4 w_s0_r3_0 = DQ4(12,13,14,15,  0, sc3.x, sc3.y, sc3.z, sc3.w);
  half4 w_s0_r3_4 = DQ4(12,13,14,15,  4, sc3.x, sc3.y, sc3.z, sc3.w);
  half4 w_s0_r3_8 = DQ4(12,13,14,15,  8, sc3.x, sc3.y, sc3.z, sc3.w);
  half4 w_s0_r3_c = DQ4(12,13,14,15, 12, sc3.x, sc3.y, sc3.z, sc3.w);
  // src1 side — ushorts[32..47].
  half4 w_s1_r0_0 = DQ4(32,33,34,35,  0, sc0.x, sc0.y, sc0.z, sc0.w);
  half4 w_s1_r0_4 = DQ4(32,33,34,35,  4, sc0.x, sc0.y, sc0.z, sc0.w);
  half4 w_s1_r0_8 = DQ4(32,33,34,35,  8, sc0.x, sc0.y, sc0.z, sc0.w);
  half4 w_s1_r0_c = DQ4(32,33,34,35, 12, sc0.x, sc0.y, sc0.z, sc0.w);
  half4 w_s1_r1_0 = DQ4(36,37,38,39,  0, sc1.x, sc1.y, sc1.z, sc1.w);
  half4 w_s1_r1_4 = DQ4(36,37,38,39,  4, sc1.x, sc1.y, sc1.z, sc1.w);
  half4 w_s1_r1_8 = DQ4(36,37,38,39,  8, sc1.x, sc1.y, sc1.z, sc1.w);
  half4 w_s1_r1_c = DQ4(36,37,38,39, 12, sc1.x, sc1.y, sc1.z, sc1.w);
  half4 w_s1_r2_0 = DQ4(40,41,42,43,  0, sc2.x, sc2.y, sc2.z, sc2.w);
  half4 w_s1_r2_4 = DQ4(40,41,42,43,  4, sc2.x, sc2.y, sc2.z, sc2.w);
  half4 w_s1_r2_8 = DQ4(40,41,42,43,  8, sc2.x, sc2.y, sc2.z, sc2.w);
  half4 w_s1_r2_c = DQ4(40,41,42,43, 12, sc2.x, sc2.y, sc2.z, sc2.w);
  half4 w_s1_r3_0 = DQ4(44,45,46,47,  0, sc3.x, sc3.y, sc3.z, sc3.w);
  half4 w_s1_r3_4 = DQ4(44,45,46,47,  4, sc3.x, sc3.y, sc3.z, sc3.w);
  half4 w_s1_r3_8 = DQ4(44,45,46,47,  8, sc3.x, sc3.y, sc3.z, sc3.w);
  half4 w_s1_r3_c = DQ4(44,45,46,47, 12, sc3.x, sc3.y, sc3.z, sc3.w);

  // ===== Phase 2: pure FMA =====
  r0 += src0.x * w_s0_r0_0;
  r0 += src0.y * w_s0_r0_4;
  r0 += src0.z * w_s0_r0_8;
  r0 += src0.w * w_s0_r0_c;
  r1 += src0.x * w_s0_r1_0;
  r1 += src0.y * w_s0_r1_4;
  r1 += src0.z * w_s0_r1_8;
  r1 += src0.w * w_s0_r1_c;
  r2 += src0.x * w_s0_r2_0;
  r2 += src0.y * w_s0_r2_4;
  r2 += src0.z * w_s0_r2_8;
  r2 += src0.w * w_s0_r2_c;
  r3 += src0.x * w_s0_r3_0;
  r3 += src0.y * w_s0_r3_4;
  r3 += src0.z * w_s0_r3_8;
  r3 += src0.w * w_s0_r3_c;
  r0 += src1.x * w_s1_r0_0;
  r0 += src1.y * w_s1_r0_4;
  r0 += src1.z * w_s1_r0_8;
  r0 += src1.w * w_s1_r0_c;
  r1 += src1.x * w_s1_r1_0;
  r1 += src1.y * w_s1_r1_4;
  r1 += src1.z * w_s1_r1_8;
  r1 += src1.w * w_s1_r1_c;
  r2 += src1.x * w_s1_r2_0;
  r2 += src1.y * w_s1_r2_4;
  r2 += src1.z * w_s1_r2_8;
  r2 += src1.w * w_s1_r2_c;
  r3 += src1.x * w_s1_r3_0;
  r3 += src1.y * w_s1_r3_4;
  r3 += src1.z * w_s1_r3_8;
  r3 += src1.w * w_s1_r3_c;

#undef DQ
#undef DQ4
      } while (coord_s < shared_int4_1.w);

  coord_s = mul24(Z, 8);
  coord_x = X;
  coord_y = Y;
  if (coord_s < shared_int4_0.y) {
    half4 res = r0;
    res += read_imageh(biases_image2d, smp_zero, (int2)((coord_s), 0));
    write_imageh(dst_tensor_image2d, (int2)((coord_x), ((coord_y) * shared_int4_0.y + (coord_s))), res);
    coord_s++;
  }
  if (coord_s < shared_int4_0.y) {
    half4 res = r1;
    res += read_imageh(biases_image2d, smp_zero, (int2)((coord_s), 0));
    write_imageh(dst_tensor_image2d, (int2)((coord_x), ((coord_y) * shared_int4_0.y + (coord_s))), res);
    coord_s++;
  }
  if (coord_s < shared_int4_0.y) {
    half4 res = r2;
    res += read_imageh(biases_image2d, smp_zero, (int2)((coord_s), 0));
    write_imageh(dst_tensor_image2d, (int2)((coord_x), ((coord_y) * shared_int4_0.y + (coord_s))), res);
    coord_s++;
  }
  if (coord_s < shared_int4_0.y) {
    half4 res = r3;
    res += read_imageh(biases_image2d, smp_zero, (int2)((coord_s), 0));
    write_imageh(dst_tensor_image2d, (int2)((coord_x), ((coord_y) * shared_int4_0.y + (coord_s))), res);
  }
}
