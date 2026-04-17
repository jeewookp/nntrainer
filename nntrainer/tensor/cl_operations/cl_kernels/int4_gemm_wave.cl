// Int4 GEMM using delegate's wave memory at FULL fp16 load speed.
//
// Key insight: load 32 half8 (512 bytes) per iteration — SAME as fp16 delegate.
// But 512 bytes of int4 = 1024 nibbles = 32 input channels × 32 output channels.
// That's 4x more K per iteration → inner loop runs K/32 times (vs K/8 for fp16).
//
// Weight layout (repacked): for each (Z, k_block):
//   32 half8 = 256 halfs = 512 bytes = 256 ushorts
//   These 256 ushorts contain 1024 int4 nibbles:
//     ushort[s*32 + n] packs 4 nibbles for output_ch=base_n+n, k=k_block*32+s*4..+3
//   where s=0..7 (8 groups of 4 k-values), n=0..31 (32 output channels)

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

  int X = get_group_id(1) * get_local_size(0) + get_local_id(0);
  int Y = get_group_id(2) * get_local_size(1) + get_local_id(1);
  int Z = get_group_id(0) * get_local_size(2) + get_local_id(2);
  if (X >= shared_int4_0.z || Y >= shared_int4_0.x) return;
  if (Z * 8 >= shared_int4_0.y) return;

  half4 r0=(half4)(0.0h), r1=(half4)(0.0h), r2=(half4)(0.0h), r3=(half4)(0.0h);
  half4 r4=(half4)(0.0h), r5=(half4)(0.0h), r6=(half4)(0.0h), r7=(half4)(0.0h);

  int base_n = Z * 32;
  half4 sc0=vload4(0,scales+base_n), sc1=vload4(0,scales+base_n+4);
  half4 sc2=vload4(0,scales+base_n+8), sc3=vload4(0,scales+base_n+12);
  half4 sc4=vload4(0,scales+base_n+16), sc5=vload4(0,scales+base_n+20);
  half4 sc6=vload4(0,scales+base_n+24), sc7=vload4(0,scales+base_n+28);

  // Wave memory setup (identical to delegate)
  int subgroup_id = (int)((0x1F & qcom_get_physical_sub_group_id()));
  subgroup_id = subgroup_id % 12;
  int c_offset = mul24(subgroup_id, shared_int4_0.w);
  short a0 = c_offset * 8;
  __asm__ __volatile__(
    "mova a0, 256;" "mova a0, %[a0];" "(rpt5)nop ;"
    : : [a0] "r" (a0) :
  );

  // View xmem as ushort for int4 unpacking (256 ushorts per load)
  __constant ushort* w = (__constant ushort*)&xmem_buffer[c_offset];

  // f_offset: same as delegate (32 half8 per iteration)
  int f_offset = Z * shared_int4_1.x * 32;

  // Inner loop: process 32 k-values per iteration (8 src slices)
  // vs delegate's 8 k-values (2 src slices). 4x fewer iterations!
  int src_slices = shared_int4_1.w;
  int coord_s = 0;

  do {
    // Load 32 half8 of packed int4 via wave memory (FULL speed, same as fp16)
    qcom_sub_group_constant_load8(xmem_buffer, weights_buffer,
                                   c_offset, f_offset >> 1, 32);
    f_offset += 64;
    qcom_sub_group_sync(QCOM_CLK_CONST_LOAD_SYNC);

    // Process 8 src slices (32 input channels) from this load
    // w[0..255] = 256 ushorts, layout:
    //   For src_slice s (0..7), output channel n (0..31):
    //     w[s*32 + n] has 4 nibbles for k = coord_s*4 + s*4 + 0..3

#define DQ(IDX,SH,SC) (((half)((int)((w[IDX]>>SH)&0xFu)-8))*SC)

#define DO_SLICE(SRC_S, W_BASE) { \
  half4 src = read_imageh(src_tensor_image2d, smp_zero, \
    (int2)(X, Y * src_slices + coord_s + SRC_S)); \
  r0.x+=src.x*DQ(W_BASE,0,sc0.x)+src.y*DQ(W_BASE,4,sc0.x)+src.z*DQ(W_BASE,8,sc0.x)+src.w*DQ(W_BASE,12,sc0.x); \
  r0.y+=src.x*DQ(W_BASE+1,0,sc0.y)+src.y*DQ(W_BASE+1,4,sc0.y)+src.z*DQ(W_BASE+1,8,sc0.y)+src.w*DQ(W_BASE+1,12,sc0.y); \
  r0.z+=src.x*DQ(W_BASE+2,0,sc0.z)+src.y*DQ(W_BASE+2,4,sc0.z)+src.z*DQ(W_BASE+2,8,sc0.z)+src.w*DQ(W_BASE+2,12,sc0.z); \
  r0.w+=src.x*DQ(W_BASE+3,0,sc0.w)+src.y*DQ(W_BASE+3,4,sc0.w)+src.z*DQ(W_BASE+3,8,sc0.w)+src.w*DQ(W_BASE+3,12,sc0.w); \
  r1.x+=src.x*DQ(W_BASE+4,0,sc1.x)+src.y*DQ(W_BASE+4,4,sc1.x)+src.z*DQ(W_BASE+4,8,sc1.x)+src.w*DQ(W_BASE+4,12,sc1.x); \
  r1.y+=src.x*DQ(W_BASE+5,0,sc1.y)+src.y*DQ(W_BASE+5,4,sc1.y)+src.z*DQ(W_BASE+5,8,sc1.y)+src.w*DQ(W_BASE+5,12,sc1.y); \
  r1.z+=src.x*DQ(W_BASE+6,0,sc1.z)+src.y*DQ(W_BASE+6,4,sc1.z)+src.z*DQ(W_BASE+6,8,sc1.z)+src.w*DQ(W_BASE+6,12,sc1.z); \
  r1.w+=src.x*DQ(W_BASE+7,0,sc1.w)+src.y*DQ(W_BASE+7,4,sc1.w)+src.z*DQ(W_BASE+7,8,sc1.w)+src.w*DQ(W_BASE+7,12,sc1.w); \
  r2.x+=src.x*DQ(W_BASE+8,0,sc2.x)+src.y*DQ(W_BASE+8,4,sc2.x)+src.z*DQ(W_BASE+8,8,sc2.x)+src.w*DQ(W_BASE+8,12,sc2.x); \
  r2.y+=src.x*DQ(W_BASE+9,0,sc2.y)+src.y*DQ(W_BASE+9,4,sc2.y)+src.z*DQ(W_BASE+9,8,sc2.y)+src.w*DQ(W_BASE+9,12,sc2.y); \
  r2.z+=src.x*DQ(W_BASE+10,0,sc2.z)+src.y*DQ(W_BASE+10,4,sc2.z)+src.z*DQ(W_BASE+10,8,sc2.z)+src.w*DQ(W_BASE+10,12,sc2.z); \
  r2.w+=src.x*DQ(W_BASE+11,0,sc2.w)+src.y*DQ(W_BASE+11,4,sc2.w)+src.z*DQ(W_BASE+11,8,sc2.w)+src.w*DQ(W_BASE+11,12,sc2.w); \
  r3.x+=src.x*DQ(W_BASE+12,0,sc3.x)+src.y*DQ(W_BASE+12,4,sc3.x)+src.z*DQ(W_BASE+12,8,sc3.x)+src.w*DQ(W_BASE+12,12,sc3.x); \
  r3.y+=src.x*DQ(W_BASE+13,0,sc3.y)+src.y*DQ(W_BASE+13,4,sc3.y)+src.z*DQ(W_BASE+13,8,sc3.y)+src.w*DQ(W_BASE+13,12,sc3.y); \
  r3.z+=src.x*DQ(W_BASE+14,0,sc3.z)+src.y*DQ(W_BASE+14,4,sc3.z)+src.z*DQ(W_BASE+14,8,sc3.z)+src.w*DQ(W_BASE+14,12,sc3.z); \
  r3.w+=src.x*DQ(W_BASE+15,0,sc3.w)+src.y*DQ(W_BASE+15,4,sc3.w)+src.z*DQ(W_BASE+15,8,sc3.w)+src.w*DQ(W_BASE+15,12,sc3.w); \
  r4.x+=src.x*DQ(W_BASE+16,0,sc4.x)+src.y*DQ(W_BASE+16,4,sc4.x)+src.z*DQ(W_BASE+16,8,sc4.x)+src.w*DQ(W_BASE+16,12,sc4.x); \
  r4.y+=src.x*DQ(W_BASE+17,0,sc4.y)+src.y*DQ(W_BASE+17,4,sc4.y)+src.z*DQ(W_BASE+17,8,sc4.y)+src.w*DQ(W_BASE+17,12,sc4.y); \
  r4.z+=src.x*DQ(W_BASE+18,0,sc4.z)+src.y*DQ(W_BASE+18,4,sc4.z)+src.z*DQ(W_BASE+18,8,sc4.z)+src.w*DQ(W_BASE+18,12,sc4.z); \
  r4.w+=src.x*DQ(W_BASE+19,0,sc4.w)+src.y*DQ(W_BASE+19,4,sc4.w)+src.z*DQ(W_BASE+19,8,sc4.w)+src.w*DQ(W_BASE+19,12,sc4.w); \
  r5.x+=src.x*DQ(W_BASE+20,0,sc5.x)+src.y*DQ(W_BASE+20,4,sc5.x)+src.z*DQ(W_BASE+20,8,sc5.x)+src.w*DQ(W_BASE+20,12,sc5.x); \
  r5.y+=src.x*DQ(W_BASE+21,0,sc5.y)+src.y*DQ(W_BASE+21,4,sc5.y)+src.z*DQ(W_BASE+21,8,sc5.y)+src.w*DQ(W_BASE+21,12,sc5.y); \
  r5.z+=src.x*DQ(W_BASE+22,0,sc5.z)+src.y*DQ(W_BASE+22,4,sc5.z)+src.z*DQ(W_BASE+22,8,sc5.z)+src.w*DQ(W_BASE+22,12,sc5.z); \
  r5.w+=src.x*DQ(W_BASE+23,0,sc5.w)+src.y*DQ(W_BASE+23,4,sc5.w)+src.z*DQ(W_BASE+23,8,sc5.w)+src.w*DQ(W_BASE+23,12,sc5.w); \
  r6.x+=src.x*DQ(W_BASE+24,0,sc6.x)+src.y*DQ(W_BASE+24,4,sc6.x)+src.z*DQ(W_BASE+24,8,sc6.x)+src.w*DQ(W_BASE+24,12,sc6.x); \
  r6.y+=src.x*DQ(W_BASE+25,0,sc6.y)+src.y*DQ(W_BASE+25,4,sc6.y)+src.z*DQ(W_BASE+25,8,sc6.y)+src.w*DQ(W_BASE+25,12,sc6.y); \
  r6.z+=src.x*DQ(W_BASE+26,0,sc6.z)+src.y*DQ(W_BASE+26,4,sc6.z)+src.z*DQ(W_BASE+26,8,sc6.z)+src.w*DQ(W_BASE+26,12,sc6.z); \
  r6.w+=src.x*DQ(W_BASE+27,0,sc6.w)+src.y*DQ(W_BASE+27,4,sc6.w)+src.z*DQ(W_BASE+27,8,sc6.w)+src.w*DQ(W_BASE+27,12,sc6.w); \
  r7.x+=src.x*DQ(W_BASE+28,0,sc7.x)+src.y*DQ(W_BASE+28,4,sc7.x)+src.z*DQ(W_BASE+28,8,sc7.x)+src.w*DQ(W_BASE+28,12,sc7.x); \
  r7.y+=src.x*DQ(W_BASE+29,0,sc7.y)+src.y*DQ(W_BASE+29,4,sc7.y)+src.z*DQ(W_BASE+29,8,sc7.y)+src.w*DQ(W_BASE+29,12,sc7.y); \
  r7.z+=src.x*DQ(W_BASE+30,0,sc7.z)+src.y*DQ(W_BASE+30,4,sc7.z)+src.z*DQ(W_BASE+30,8,sc7.z)+src.w*DQ(W_BASE+30,12,sc7.z); \
  r7.w+=src.x*DQ(W_BASE+31,0,sc7.w)+src.y*DQ(W_BASE+31,4,sc7.w)+src.z*DQ(W_BASE+31,8,sc7.w)+src.w*DQ(W_BASE+31,12,sc7.w); \
}

    // Process first 4 src slices from this load
    DO_SLICE(0, 0)
    DO_SLICE(1, 32)
    DO_SLICE(2, 64)
    DO_SLICE(3, 96)

    // Process next 4 src slices (second half of loaded data)
    DO_SLICE(4, 128)
    DO_SLICE(5, 160)
    DO_SLICE(6, 192)
    DO_SLICE(7, 224)

#undef DQ
#undef DO_SLICE

    coord_s += 8;
  } while (coord_s < src_slices);

  int cs = Z * 8;
  if (cs<shared_int4_0.y) write_imageh(dst_tensor_image2d,(int2)(X,Y*shared_int4_0.y+cs),r0); cs++;
  if (cs<shared_int4_0.y) write_imageh(dst_tensor_image2d,(int2)(X,Y*shared_int4_0.y+cs),r1); cs++;
  if (cs<shared_int4_0.y) write_imageh(dst_tensor_image2d,(int2)(X,Y*shared_int4_0.y+cs),r2); cs++;
  if (cs<shared_int4_0.y) write_imageh(dst_tensor_image2d,(int2)(X,Y*shared_int4_0.y+cs),r3); cs++;
  if (cs<shared_int4_0.y) write_imageh(dst_tensor_image2d,(int2)(X,Y*shared_int4_0.y+cs),r4); cs++;
  if (cs<shared_int4_0.y) write_imageh(dst_tensor_image2d,(int2)(X,Y*shared_int4_0.y+cs),r5); cs++;
  if (cs<shared_int4_0.y) write_imageh(dst_tensor_image2d,(int2)(X,Y*shared_int4_0.y+cs),r6); cs++;
  if (cs<shared_int4_0.y) write_imageh(dst_tensor_image2d,(int2)(X,Y*shared_int4_0.y+cs),r7);
}
