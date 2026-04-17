#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_qcom_subgroup_uniform_load: enable
#pragma OPENCL EXTENSION cl_qcom_subgroup_constant_load: enable

// Int4 GEMM with wave memory (Adreno 830 optimized).
// Uses qcom_sub_group_constant_load for int4 weight loading.
// Weight layout: channel-wise int4, packed[(k/4)*N + n].
// Per iteration: reads 2 src slices (8 input ch) × 32 output ch.
// Weight load: 2 × 32 ushorts = 128 bytes/iter (vs 512 for fp16).

__constant sampler_t smp_zero = CLK_NORMALIZED_COORDS_FALSE |
                                 CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

__attribute__((qcom_max_concurrent_subgroups(12)))
__kernel void gemm_int4_wave(
    __constant ushort* weights_buffer __attribute__((sub_group_uniform)),
    __constant ushort* xmem_buffer __attribute__((max_constant_size((6144)))),
    __global const half* scales,
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

  half4 r0 = (half4)(0.0h);
  half4 r1 = (half4)(0.0h);
  half4 r2 = (half4)(0.0h);
  half4 r3 = (half4)(0.0h);
  half4 r4 = (half4)(0.0h);
  half4 r5 = (half4)(0.0h);
  half4 r6 = (half4)(0.0h);
  half4 r7 = (half4)(0.0h);

  int base_n = Z * 32;

  // Load per-channel scales for 32 output channels
  half4 sc0 = vload4(0, scales + base_n);
  half4 sc1 = vload4(0, scales + base_n + 4);
  half4 sc2 = vload4(0, scales + base_n + 8);
  half4 sc3 = vload4(0, scales + base_n + 12);
  half4 sc4 = vload4(0, scales + base_n + 16);
  half4 sc5 = vload4(0, scales + base_n + 20);
  half4 sc6 = vload4(0, scales + base_n + 24);
  half4 sc7 = vload4(0, scales + base_n + 28);

  // Wave memory setup
  int subgroup_id = (int)((0x1F & qcom_get_physical_sub_group_id()));
  subgroup_id = subgroup_id % 12;
  // Each subgroup gets 6144/12 = 512 bytes = 256 ushorts in xmem
  int c_offset = subgroup_id * 256;

  __constant ushort* w_cache = &xmem_buffer[c_offset];

  int coord_s = 0;
  do {
    // Read 2 src slices (8 input channels)
    half4 src0 = read_imageh(src_image, smp_zero,
                             (int2)(X, Y * src_slices + coord_s));
    half4 src1 = read_imageh(src_image, smp_zero,
                             (int2)(X, Y * src_slices + coord_s + 1));

    // Load int4 packed weights via constant load.
    // Row 0: weights[coord_s * N + base_n .. +31] — 32 ushorts for k=4*coord_s..+3
    // Row 1: weights[(coord_s+1) * N + base_n .. +31] — 32 ushorts for k=4*(coord_s+1)..+3
    // Total: 64 ushorts = 128 bytes → fits in one load of 64 ushort elements
    int w_off0 = coord_s * N + base_n;
    int w_off1 = (coord_s + 1) * N + base_n;

    // Load both rows: 32 ushorts each at c_offset and c_offset+32
    qcom_sub_group_constant_load8(xmem_buffer, weights_buffer,
                                   c_offset, w_off0, 32);
    qcom_sub_group_constant_load8(xmem_buffer, weights_buffer,
                                   c_offset + 32, w_off1, 32);
    qcom_sub_group_sync(QCOM_CLK_CONST_LOAD_SYNC);

    // Unpack row 0 (src0: k = coord_s*4 + 0..3) for 8 output slice groups
    // w_cache[0..31] = row0: packed weights for 32 output channels
    // w_cache[s*4+j] = packed weights for output channel base_n+s*4+j

#define UNPACK_AND_ACC(R, SRC_CH, ROW_OFF, SHIFT, SC) \
    { \
      half4 dq; \
      dq.x = ((half)((int)((w_cache[ROW_OFF + 0] >> SHIFT) & 0xF) - 8)) * SC.x; \
      dq.y = ((half)((int)((w_cache[ROW_OFF + 1] >> SHIFT) & 0xF) - 8)) * SC.y; \
      dq.z = ((half)((int)((w_cache[ROW_OFF + 2] >> SHIFT) & 0xF) - 8)) * SC.z; \
      dq.w = ((half)((int)((w_cache[ROW_OFF + 3] >> SHIFT) & 0xF) - 8)) * SC.w; \
      R += SRC_CH * dq; \
    }

    // Row 0: 4 k values (nibbles 0,1,2,3) × 8 output groups
    UNPACK_AND_ACC(r0, src0.x, 0,  0, sc0)  // k+0, out 0..3
    UNPACK_AND_ACC(r0, src0.y, 0,  4, sc0)  // k+1
    UNPACK_AND_ACC(r0, src0.z, 0,  8, sc0)  // k+2
    UNPACK_AND_ACC(r0, src0.w, 0, 12, sc0)  // k+3

    UNPACK_AND_ACC(r1, src0.x, 4,  0, sc1)
    UNPACK_AND_ACC(r1, src0.y, 4,  4, sc1)
    UNPACK_AND_ACC(r1, src0.z, 4,  8, sc1)
    UNPACK_AND_ACC(r1, src0.w, 4, 12, sc1)

    UNPACK_AND_ACC(r2, src0.x, 8,  0, sc2)
    UNPACK_AND_ACC(r2, src0.y, 8,  4, sc2)
    UNPACK_AND_ACC(r2, src0.z, 8,  8, sc2)
    UNPACK_AND_ACC(r2, src0.w, 8, 12, sc2)

    UNPACK_AND_ACC(r3, src0.x, 12,  0, sc3)
    UNPACK_AND_ACC(r3, src0.y, 12,  4, sc3)
    UNPACK_AND_ACC(r3, src0.z, 12,  8, sc3)
    UNPACK_AND_ACC(r3, src0.w, 12, 12, sc3)

    UNPACK_AND_ACC(r4, src0.x, 16,  0, sc4)
    UNPACK_AND_ACC(r4, src0.y, 16,  4, sc4)
    UNPACK_AND_ACC(r4, src0.z, 16,  8, sc4)
    UNPACK_AND_ACC(r4, src0.w, 16, 12, sc4)

    UNPACK_AND_ACC(r5, src0.x, 20,  0, sc5)
    UNPACK_AND_ACC(r5, src0.y, 20,  4, sc5)
    UNPACK_AND_ACC(r5, src0.z, 20,  8, sc5)
    UNPACK_AND_ACC(r5, src0.w, 20, 12, sc5)

    UNPACK_AND_ACC(r6, src0.x, 24,  0, sc6)
    UNPACK_AND_ACC(r6, src0.y, 24,  4, sc6)
    UNPACK_AND_ACC(r6, src0.z, 24,  8, sc6)
    UNPACK_AND_ACC(r6, src0.w, 24, 12, sc6)

    UNPACK_AND_ACC(r7, src0.x, 28,  0, sc7)
    UNPACK_AND_ACC(r7, src0.y, 28,  4, sc7)
    UNPACK_AND_ACC(r7, src0.z, 28,  8, sc7)
    UNPACK_AND_ACC(r7, src0.w, 28, 12, sc7)

    // Row 1 (src1: k = (coord_s+1)*4 + 0..3), cache offset +32
    UNPACK_AND_ACC(r0, src1.x, 32,  0, sc0)
    UNPACK_AND_ACC(r0, src1.y, 32,  4, sc0)
    UNPACK_AND_ACC(r0, src1.z, 32,  8, sc0)
    UNPACK_AND_ACC(r0, src1.w, 32, 12, sc0)

    UNPACK_AND_ACC(r1, src1.x, 36,  0, sc1)
    UNPACK_AND_ACC(r1, src1.y, 36,  4, sc1)
    UNPACK_AND_ACC(r1, src1.z, 36,  8, sc1)
    UNPACK_AND_ACC(r1, src1.w, 36, 12, sc1)

    UNPACK_AND_ACC(r2, src1.x, 40,  0, sc2)
    UNPACK_AND_ACC(r2, src1.y, 40,  4, sc2)
    UNPACK_AND_ACC(r2, src1.z, 40,  8, sc2)
    UNPACK_AND_ACC(r2, src1.w, 40, 12, sc2)

    UNPACK_AND_ACC(r3, src1.x, 44,  0, sc3)
    UNPACK_AND_ACC(r3, src1.y, 44,  4, sc3)
    UNPACK_AND_ACC(r3, src1.z, 44,  8, sc3)
    UNPACK_AND_ACC(r3, src1.w, 44, 12, sc3)

    UNPACK_AND_ACC(r4, src1.x, 48,  0, sc4)
    UNPACK_AND_ACC(r4, src1.y, 48,  4, sc4)
    UNPACK_AND_ACC(r4, src1.z, 48,  8, sc4)
    UNPACK_AND_ACC(r4, src1.w, 48, 12, sc4)

    UNPACK_AND_ACC(r5, src1.x, 52,  0, sc5)
    UNPACK_AND_ACC(r5, src1.y, 52,  4, sc5)
    UNPACK_AND_ACC(r5, src1.z, 52,  8, sc5)
    UNPACK_AND_ACC(r5, src1.w, 52, 12, sc5)

    UNPACK_AND_ACC(r6, src1.x, 56,  0, sc6)
    UNPACK_AND_ACC(r6, src1.y, 56,  4, sc6)
    UNPACK_AND_ACC(r6, src1.z, 56,  8, sc6)
    UNPACK_AND_ACC(r6, src1.w, 56, 12, sc6)

    UNPACK_AND_ACC(r7, src1.x, 60,  0, sc7)
    UNPACK_AND_ACC(r7, src1.y, 60,  4, sc7)
    UNPACK_AND_ACC(r7, src1.z, 60,  8, sc7)
    UNPACK_AND_ACC(r7, src1.w, 60, 12, sc7)

#undef UNPACK_AND_ACC

    coord_s += 2;
  } while (coord_s < src_slices);

  // Write output
  int out_s = Z * 8;
  if (out_s + 0 < dst_slices) write_imageh(dst_image, (int2)(X, Y * dst_slices + out_s + 0), r0);
  if (out_s + 1 < dst_slices) write_imageh(dst_image, (int2)(X, Y * dst_slices + out_s + 1), r1);
  if (out_s + 2 < dst_slices) write_imageh(dst_image, (int2)(X, Y * dst_slices + out_s + 2), r2);
  if (out_s + 3 < dst_slices) write_imageh(dst_image, (int2)(X, Y * dst_slices + out_s + 3), r3);
  if (out_s + 4 < dst_slices) write_imageh(dst_image, (int2)(X, Y * dst_slices + out_s + 4), r4);
  if (out_s + 5 < dst_slices) write_imageh(dst_image, (int2)(X, Y * dst_slices + out_s + 5), r5);
  if (out_s + 6 < dst_slices) write_imageh(dst_image, (int2)(X, Y * dst_slices + out_s + 6), r6);
  if (out_s + 7 < dst_slices) write_imageh(dst_image, (int2)(X, Y * dst_slices + out_s + 7), r7);
}
