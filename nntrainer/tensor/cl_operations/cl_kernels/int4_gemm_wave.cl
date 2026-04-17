// delegate_conv_wave.cl에서 weight만 int4로 변경한 버전.
// 동일한 wave memory, 동일한 thread 매핑, 동일한 qcom extensions.
// 변경: half8 weight → int4 packed (4x 적은 메모리 로드)

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
  const int M,
  const int N,
  const int K) {

  const int src_slices = K / 4;
  const int dst_slices = N / 4;

  int X = get_group_id(1) * get_local_size(0) + get_local_id(0);
  int Y = get_group_id(2) * get_local_size(1) + get_local_id(1);
  int Z = get_group_id(0) * get_local_size(2) + get_local_id(2);
  if (X >= M || Y >= 1) return;
  if (Z * 8 >= dst_slices) return;

  half4 r0 = (half4)(0.0h);
  half4 r1 = (half4)(0.0h);
  half4 r2 = (half4)(0.0h);
  half4 r3 = (half4)(0.0h);
  half4 r4 = (half4)(0.0h);
  half4 r5 = (half4)(0.0h);
  half4 r6 = (half4)(0.0h);
  half4 r7 = (half4)(0.0h);

  // Load per-channel scales for 32 output channels (8 slices × 4 ch)
  int base_n = Z * 32;
  half4 sc0 = vload4(0, scales + base_n);
  half4 sc1 = vload4(0, scales + base_n + 4);
  half4 sc2 = vload4(0, scales + base_n + 8);
  half4 sc3 = vload4(0, scales + base_n + 12);
  half4 sc4 = vload4(0, scales + base_n + 16);
  half4 sc5 = vload4(0, scales + base_n + 20);
  half4 sc6 = vload4(0, scales + base_n + 24);
  half4 sc7 = vload4(0, scales + base_n + 28);

  // Wave memory setup (identical to delegate)
  int subgroup_id = (int)((0x1F & qcom_get_physical_sub_group_id()));
  subgroup_id = subgroup_id % 12;
  int c_offset = mul24(subgroup_id, 32);  // 32 half8 = 512 bytes per subgroup
  short a0 = (short)(c_offset * 8);

  __asm__ __volatile__(
    "mova a0, 256;"
    "mova a0, %[a0];"
    "(rpt5)nop ;"
    :
    : [a0] "r" (a0)
    :
  );

  // View xmem as ushort* for int4 unpacking
  __constant ushort* w_ushort = (__constant ushort*)&xmem_buffer[c_offset];

  // Int4 weight layout: weights are packed as ushort in the half8 buffer.
  // weights_buffer contains packed int4 data reinterpreted as half8.
  // Per iteration: 32 output channels × 8 input channels (2 src slices)
  //   = 256 int4 values = 128 bytes = 8 half8 elements
  //
  // We load 8 half8 via constant_load (vs 32 for fp16 delegate).
  // f_offset is in half8 units, incremented by 8 per iteration (was 64).
  //
  // The int4 packed data is stored as:
  //   Row layout: for each packed_row (= k/4), 32 ushorts (one per output channel)
  //   2 packed_rows per iteration (8 k values from 2 src slices)
  //   = 64 ushorts = 128 bytes = 8 half8

  int f_offset = Z * src_slices * 4;  // half8 units: Z * (K/4) * (N/16)...
  // Actually: weight buffer layout matches packed weights laid out as
  // sequential ushorts reinterpreted as half8.
  // For Z output group, iteration i: offset = Z * (src_slices/2) * 8 + i * 8
  // in half8 units.
  f_offset = Z * src_slices * 4;

  int coord_s = 0;
  do {
    half4 src0 = read_imageh(src_tensor_image2d, smp_zero,
                             (int2)(X, Y * src_slices + coord_s));
    half4 src1 = read_imageh(src_tensor_image2d, smp_zero,
                             (int2)(X, Y * src_slices + coord_s + 1));

    // Load 8 half8 (128 bytes) of packed int4 weights via wave memory
    qcom_sub_group_constant_load8(xmem_buffer, weights_buffer,
                                   c_offset, f_offset >> 1, 8);
    f_offset += 16;
    qcom_sub_group_sync(QCOM_CLK_CONST_LOAD_SYNC);

    // w_ushort[0..63] now contains 64 ushorts:
    //   w_ushort[0..31]  = row 0 (32 channels, packed k0..k3 from src0)
    //   w_ushort[32..63] = row 1 (32 channels, packed k4..k7 from src1)

#define DQ(W_IDX, SHIFT, SC) \
    (((half)((int)((w_ushort[W_IDX] >> SHIFT) & 0xFu) - 8)) * SC)

#define ACC_SLICE(R, SRC, OFF, SC) \
    R.x += SRC.x*DQ(OFF,0,SC.x) + SRC.y*DQ(OFF,4,SC.x) + SRC.z*DQ(OFF,8,SC.x) + SRC.w*DQ(OFF,12,SC.x); \
    R.y += SRC.x*DQ(OFF+1,0,SC.y) + SRC.y*DQ(OFF+1,4,SC.y) + SRC.z*DQ(OFF+1,8,SC.y) + SRC.w*DQ(OFF+1,12,SC.y); \
    R.z += SRC.x*DQ(OFF+2,0,SC.z) + SRC.y*DQ(OFF+2,4,SC.z) + SRC.z*DQ(OFF+2,8,SC.z) + SRC.w*DQ(OFF+2,12,SC.z); \
    R.w += SRC.x*DQ(OFF+3,0,SC.w) + SRC.y*DQ(OFF+3,4,SC.w) + SRC.z*DQ(OFF+3,8,SC.w) + SRC.w*DQ(OFF+3,12,SC.w);

    // Row 0 (src0: 4 k values per ushort)
    ACC_SLICE(r0, src0, 0,  sc0)
    ACC_SLICE(r1, src0, 4,  sc1)
    ACC_SLICE(r2, src0, 8,  sc2)
    ACC_SLICE(r3, src0, 12, sc3)
    ACC_SLICE(r4, src0, 16, sc4)
    ACC_SLICE(r5, src0, 20, sc5)
    ACC_SLICE(r6, src0, 24, sc6)
    ACC_SLICE(r7, src0, 28, sc7)

    // Row 1 (src1)
    ACC_SLICE(r0, src1, 32, sc0)
    ACC_SLICE(r1, src1, 36, sc1)
    ACC_SLICE(r2, src1, 40, sc2)
    ACC_SLICE(r3, src1, 44, sc3)
    ACC_SLICE(r4, src1, 48, sc4)
    ACC_SLICE(r5, src1, 52, sc5)
    ACC_SLICE(r6, src1, 56, sc6)
    ACC_SLICE(r7, src1, 60, sc7)

#undef DQ
#undef ACC_SLICE

    coord_s += 2;
  } while (coord_s < src_slices);

  // Write output (identical to delegate)
  int out_s = Z * 8;
  if (out_s + 0 < dst_slices) write_imageh(dst_tensor_image2d, (int2)(X, Y * dst_slices + out_s + 0), r0);
  if (out_s + 1 < dst_slices) write_imageh(dst_tensor_image2d, (int2)(X, Y * dst_slices + out_s + 1), r1);
  if (out_s + 2 < dst_slices) write_imageh(dst_tensor_image2d, (int2)(X, Y * dst_slices + out_s + 2), r2);
  if (out_s + 3 < dst_slices) write_imageh(dst_tensor_image2d, (int2)(X, Y * dst_slices + out_s + 3), r3);
  if (out_s + 4 < dst_slices) write_imageh(dst_tensor_image2d, (int2)(X, Y * dst_slices + out_s + 4), r4);
  if (out_s + 5 < dst_slices) write_imageh(dst_tensor_image2d, (int2)(X, Y * dst_slices + out_s + 5), r5);
  if (out_s + 6 < dst_slices) write_imageh(dst_tensor_image2d, (int2)(X, Y * dst_slices + out_s + 6), r6);
  if (out_s + 7 < dst_slices) write_imageh(dst_tensor_image2d, (int2)(X, Y * dst_slices + out_s + 7), r7);
}
