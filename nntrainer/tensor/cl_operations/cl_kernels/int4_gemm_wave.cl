#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// Int4 GEMM with delegate-style thread/output layout.
// Phase 1: correctness-first (no wave memory yet, plain __constant reads).
// Each thread computes 8 output slices × 4 channels = 32 output elements.
// Weight: channel-wise int4, packed[(k/4)*N + n], 4 nibbles per ushort.

__constant sampler_t smp_zero = CLK_NORMALIZED_COORDS_FALSE |
                                 CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

__kernel void gemm_int4_wave(
    __global const ushort* weights,
    __global const ushort* unused_xmem,
    __global const half* scales,
    __write_only image2d_t dst_image,
    __read_only image2d_t src_image,
    const int M,
    const int N,
    const int K) {

  const int src_slices = K / 4;
  const int dst_slices = N / 4;

  // Same thread mapping as delegate kernel
  int X = get_group_id(1) * get_local_size(0) + get_local_id(0);
  int Y = get_group_id(2) * get_local_size(1) + get_local_id(1);
  int Z = get_group_id(0) * get_local_size(2) + get_local_id(2);
  if (X >= M || Y >= 1) return;
  if (Z * 8 >= dst_slices) return;

  // 8 output slices × 4 channels each
  half4 r[8];
  for (int i = 0; i < 8; ++i)
    r[i] = (half4)(0.0h);

  int base_n = Z * 32;  // first output channel for this thread

  // Inner loop over K
  for (int k = 0; k < K; ++k) {
    // Read src activation
    int src_s = k / 4;
    int src_c = k % 4;
    half4 src_pixel = read_imageh(src_image, smp_zero,
                                   (int2)(X, Y * src_slices + src_s));
    half src_val;
    if (src_c == 0) src_val = src_pixel.x;
    else if (src_c == 1) src_val = src_pixel.y;
    else if (src_c == 2) src_val = src_pixel.z;
    else src_val = src_pixel.w;

    // For each of 8 output slices (32 output channels)
    for (int s = 0; s < 8; ++s) {
      int n_base = base_n + s * 4;
      if (n_base >= N) break;

      // Read 4 packed weights (4 output channels, same k)
      ushort4 pw;
      int pk = k / 4;
      pw.x = weights[pk * N + n_base + 0];
      pw.y = weights[pk * N + n_base + 1];
      pw.z = weights[pk * N + n_base + 2];
      pw.w = (n_base + 3 < N) ? weights[pk * N + n_base + 3] : (ushort)0;

      // Extract nibble for this k
      int shift = (k % 4) * 4;
      half4 dq;
      dq.x = ((half)((int)((pw.x >> shift) & 0xF) - 8)) * scales[n_base + 0];
      dq.y = ((half)((int)((pw.y >> shift) & 0xF) - 8)) * scales[n_base + 1];
      dq.z = ((half)((int)((pw.z >> shift) & 0xF) - 8)) * scales[n_base + 2];
      dq.w = ((half)((int)((pw.w >> shift) & 0xF) - 8)) * scales[n_base + 3];

      r[s] += src_val * dq;
    }
  }

  // Write output
  int out_s = Z * 8;
  for (int s = 0; s < 8; ++s) {
    if (out_s + s < dst_slices)
      write_imageh(dst_image, (int2)(X, Y * dst_slices + out_s + s), r[s]);
  }
}
