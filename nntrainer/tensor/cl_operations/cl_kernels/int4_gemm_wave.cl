#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_qcom_subgroup_uniform_load: enable
#pragma OPENCL EXTENSION cl_qcom_subgroup_constant_load: enable
#pragma OPENCL EXTENSION cl_qcom_inline_asm : enable

// Int4 version of delegate's conv_wave_memory kernel.
// Uses the same wave memory architecture (qcom_sub_group_constant_load)
// but reads int4 packed weights instead of fp16, doing dequant in-kernel.
//
// Weight layout: channel-wise int4, packed 4 nibbles per ushort
//   weights[(k/4) * N + n] = nibble(k) | nibble(k+1)<<4 | nibble(k+2)<<8 | nibble(k+3)<<12
//   dequant: ((nibble - 8) * scale[n])
//
// Per iteration: load 8 input channels (2 src slices) × 32 output channels
// from int4 weights via constant load, dequant to fp16, accumulate.

__constant sampler_t smp_zero = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

__attribute__((qcom_max_concurrent_subgroups(12)))
__kernel void gemm_int4_wave(
    __constant ushort* weights_buffer __attribute__((sub_group_uniform)),
    __constant ushort* xmem_buffer __attribute__((max_constant_size((6144)))),
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

  // Load scales for this Z group (8 output slices × 4 channels = 32 channels)
  int base_n = Z * 32;
  half scale_arr[32];
  for (int i = 0; i < 32 && base_n + i < N; ++i)
    scale_arr[i] = scales[base_n + i];

  int subgroup_id = (int)((0x1F & qcom_get_physical_sub_group_id()));
  subgroup_id = subgroup_id % 12;
  int c_offset = subgroup_id * 32;
  short a0 = (short)(c_offset * 2);

  __asm__ __volatile__(
    "mova a0, 256;"
    "mova a0, %[a0];"
    "(rpt5)nop ;"
    :
    : [a0] "r" (a0)
    :
  );

  // Int4 weights: packed 4 nibbles per ushort
  // For 32 output channels × 4 input channels = 128 weight values
  // = 128 nibbles = 32 ushorts per inner iteration (loading 4 k values)
  //
  // But we process 8 input channels per iteration (2 src slices),
  // so we load 64 ushorts = 32 output channels × 8 k values / 4 per ushort = 64

  __constant ushort* w_cache = &xmem_buffer[c_offset];

  // Weight offset: Z group × K/4 × N (int4 packed layout)
  int w_base = (base_n);  // column offset in packed weight matrix

  int coord_s = 0;
  do {
    // Read 2 src slices (8 input channels)
    half4 src0 = read_imageh(src_tensor_image2d, smp_zero,
                             (int2)(X, Y * src_slices + coord_s));
    coord_s++;
    half4 src1 = read_imageh(src_tensor_image2d, smp_zero,
                             (int2)(X, Y * src_slices + coord_s));
    coord_s++;

    // Load int4 packed weights for these 8 input channels × 32 output channels
    // Each ushort packs 4 k values for one n: weights[(k/4)*N + n]
    // For k_base = (coord_s-2)*4, we need 2 rows of packed weights:
    //   row0: weights[((coord_s-2)*4/4)*N + base_n ... +31] = 32 ushorts (k0..k3)
    //   row1: weights[((coord_s-1)*4/4)*N + base_n ... +31] = 32 ushorts (k4..k7)
    int k_row0 = (coord_s - 2);
    int k_row1 = (coord_s - 1);

    // Load via constant load: 32 ushorts per row = 64 bytes
    // Total: 64 ushorts = 128 bytes
    qcom_sub_group_constant_load8(xmem_buffer, weights_buffer,
                                   c_offset, (k_row0 * N + w_base) / 8, 8);
    qcom_sub_group_sync(QCOM_CLK_CONST_LOAD_SYNC);

    // Unpack and accumulate for first 4 input channels (src0)
    for (int s = 0; s < 8 && base_n + s * 4 < N; ++s) {
      ushort4 pw = (ushort4)(w_cache[s * 4], w_cache[s * 4 + 1],
                             w_cache[s * 4 + 2], w_cache[s * 4 + 3]);

      half4 sc = (half4)(scale_arr[s * 4], scale_arr[s * 4 + 1],
                         scale_arr[s * 4 + 2], scale_arr[s * 4 + 3]);

      // k+0
      half4 dq;
      dq.x = ((half)((int)(pw.x & 0xF) - 8)) * sc.x;
      dq.y = ((half)((int)(pw.y & 0xF) - 8)) * sc.y;
      dq.z = ((half)((int)(pw.z & 0xF) - 8)) * sc.z;
      dq.w = ((half)((int)(pw.w & 0xF) - 8)) * sc.w;
      if (s < 2) { r0 += src0.x * dq; }
      else if (s < 4) { r2 += src0.x * dq; }
      else if (s < 6) { r4 += src0.x * dq; }
      else { r6 += src0.x * dq; }

      // k+1
      dq.x = ((half)((int)((pw.x >> 4) & 0xF) - 8)) * sc.x;
      dq.y = ((half)((int)((pw.y >> 4) & 0xF) - 8)) * sc.y;
      dq.z = ((half)((int)((pw.z >> 4) & 0xF) - 8)) * sc.z;
      dq.w = ((half)((int)((pw.w >> 4) & 0xF) - 8)) * sc.w;
      if (s < 2) { r0 += src0.y * dq; }
      else if (s < 4) { r2 += src0.y * dq; }
      else if (s < 6) { r4 += src0.y * dq; }
      else { r6 += src0.y * dq; }

      // k+2
      dq.x = ((half)((int)((pw.x >> 8) & 0xF) - 8)) * sc.x;
      dq.y = ((half)((int)((pw.y >> 8) & 0xF) - 8)) * sc.y;
      dq.z = ((half)((int)((pw.z >> 8) & 0xF) - 8)) * sc.z;
      dq.w = ((half)((int)((pw.w >> 8) & 0xF) - 8)) * sc.w;
      if (s < 2) { r0 += src0.z * dq; }
      else if (s < 4) { r2 += src0.z * dq; }
      else if (s < 6) { r4 += src0.z * dq; }
      else { r6 += src0.z * dq; }

      // k+3
      dq.x = ((half)((int)((pw.x >> 12) & 0xF) - 8)) * sc.x;
      dq.y = ((half)((int)((pw.y >> 12) & 0xF) - 8)) * sc.y;
      dq.z = ((half)((int)((pw.z >> 12) & 0xF) - 8)) * sc.z;
      dq.w = ((half)((int)((pw.w >> 12) & 0xF) - 8)) * sc.w;
      if (s < 2) { r0 += src0.w * dq; }
      else if (s < 4) { r2 += src0.w * dq; }
      else if (s < 6) { r4 += src0.w * dq; }
      else { r6 += src0.w * dq; }
    }

    // Second row (src1, k4..k7)
    qcom_sub_group_constant_load8(xmem_buffer, weights_buffer,
                                   c_offset, (k_row1 * N + w_base) / 8, 8);
    qcom_sub_group_sync(QCOM_CLK_CONST_LOAD_SYNC);

    for (int s = 0; s < 8 && base_n + s * 4 < N; ++s) {
      ushort4 pw = (ushort4)(w_cache[s * 4], w_cache[s * 4 + 1],
                             w_cache[s * 4 + 2], w_cache[s * 4 + 3]);

      half4 sc = (half4)(scale_arr[s * 4], scale_arr[s * 4 + 1],
                         scale_arr[s * 4 + 2], scale_arr[s * 4 + 3]);

      half4 dq;
      dq.x = ((half)((int)(pw.x & 0xF) - 8)) * sc.x;
      dq.y = ((half)((int)(pw.y & 0xF) - 8)) * sc.y;
      dq.z = ((half)((int)(pw.z & 0xF) - 8)) * sc.z;
      dq.w = ((half)((int)(pw.w & 0xF) - 8)) * sc.w;
      if (s < 2) { r1 += src1.x * dq; }
      else if (s < 4) { r3 += src1.x * dq; }
      else if (s < 6) { r5 += src1.x * dq; }
      else { r7 += src1.x * dq; }

      dq.x = ((half)((int)((pw.x >> 4) & 0xF) - 8)) * sc.x;
      dq.y = ((half)((int)((pw.y >> 4) & 0xF) - 8)) * sc.y;
      dq.z = ((half)((int)((pw.z >> 4) & 0xF) - 8)) * sc.z;
      dq.w = ((half)((int)((pw.w >> 4) & 0xF) - 8)) * sc.w;
      if (s < 2) { r1 += src1.y * dq; }
      else if (s < 4) { r3 += src1.y * dq; }
      else if (s < 6) { r5 += src1.y * dq; }
      else { r7 += src1.y * dq; }

      dq.x = ((half)((int)((pw.x >> 8) & 0xF) - 8)) * sc.x;
      dq.y = ((half)((int)((pw.y >> 8) & 0xF) - 8)) * sc.y;
      dq.z = ((half)((int)((pw.z >> 8) & 0xF) - 8)) * sc.z;
      dq.w = ((half)((int)((pw.w >> 8) & 0xF) - 8)) * sc.w;
      if (s < 2) { r1 += src1.z * dq; }
      else if (s < 4) { r3 += src1.z * dq; }
      else if (s < 6) { r5 += src1.z * dq; }
      else { r7 += src1.z * dq; }

      dq.x = ((half)((int)((pw.x >> 12) & 0xF) - 8)) * sc.x;
      dq.y = ((half)((int)((pw.y >> 12) & 0xF) - 8)) * sc.y;
      dq.z = ((half)((int)((pw.z >> 12) & 0xF) - 8)) * sc.z;
      dq.w = ((half)((int)((pw.w >> 12) & 0xF) - 8)) * sc.w;
      if (s < 2) { r1 += src1.w * dq; }
      else if (s < 4) { r3 += src1.w * dq; }
      else if (s < 6) { r5 += src1.w * dq; }
      else { r7 += src1.w * dq; }
    }
  } while (coord_s < src_slices);

  // Write output
  int out_s = Z * 8;
  if (out_s + 0 < dst_slices)
    write_imageh(dst_tensor_image2d, (int2)(X, Y * dst_slices + out_s + 0), r0);
  if (out_s + 1 < dst_slices)
    write_imageh(dst_tensor_image2d, (int2)(X, Y * dst_slices + out_s + 1), r1);
  if (out_s + 2 < dst_slices)
    write_imageh(dst_tensor_image2d, (int2)(X, Y * dst_slices + out_s + 2), r2);
  if (out_s + 3 < dst_slices)
    write_imageh(dst_tensor_image2d, (int2)(X, Y * dst_slices + out_s + 3), r3);
  if (out_s + 4 < dst_slices)
    write_imageh(dst_tensor_image2d, (int2)(X, Y * dst_slices + out_s + 4), r4);
  if (out_s + 5 < dst_slices)
    write_imageh(dst_tensor_image2d, (int2)(X, Y * dst_slices + out_s + 5), r5);
  if (out_s + 6 < dst_slices)
    write_imageh(dst_tensor_image2d, (int2)(X, Y * dst_slices + out_s + 6), r6);
  if (out_s + 7 < dst_slices)
    write_imageh(dst_tensor_image2d, (int2)(X, Y * dst_slices + out_s + 7), r7);
}
