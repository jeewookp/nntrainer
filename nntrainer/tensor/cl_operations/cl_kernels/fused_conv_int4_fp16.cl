// Fused int4 → fp16 convolution (1×1 matmul).
//
// Replaces the (dequant_int4_to_delegate_fp16 → captured delegate_conv_wave)
// pipeline with a single kernel that reads packed int4 weights + per-output-
// channel scales, decodes nibbles inline, and does the full M×N GEMM via
// image2d src + image2d dst (same interfaces as the captured conv).
//
// Layout:
//   packed_weights: global ushort [K/4 * N]
//       each ushort packs 4 nibbles for 4 consecutive k positions
//       at column n: packed_weights[(k/4)*N + n]
//       nibble at position (k%4) = ((packed >> ((k%4)*4)) & 0xF) - 8
//   scales: global half [N]  (per-output-channel)
//   src: image2d (M, K/4) fp16 RGBA  (fp16 activation)
//   bias: image2d (N/4, 1) fp16 RGBA
//   dst: image2d (M, N/4) fp16 RGBA
//
// Each work-item computes one output pixel (half4 = 4 output channels)
// at (m, n_slice) by iterating the K dimension in groups of 4.
//
// Validity: requires N % 4 == 0 and K % 4 == 0. Delegate contract already
// enforces N % 32 == 0 and K % 8 == 0, so this is always satisfied for
// model shapes we care about.

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__constant sampler_t smp_zero = CLK_NORMALIZED_COORDS_FALSE |
                                CLK_ADDRESS_CLAMP |
                                CLK_FILTER_NEAREST;

__kernel void fused_conv_int4_fp16(
    __global const ushort* restrict packed_weights,
    __global const half*   restrict scales,
    __read_only  image2d_t src,
    __read_only  image2d_t bias,
    __write_only image2d_t dst,
    const int M,
    const int N,
    const int K) {

  const int m = get_global_id(0);
  const int n_slice = get_global_id(1);
  if (m >= M) return;
  if (n_slice * 4 >= N) return;

  const int src_slices = K / 4;
  const int n_base = n_slice * 4;

  // Per-output-channel scales (4 halves, one per output channel).
  const half4 scale4 = vload4(0, scales + n_base);

  half4 acc = read_imageh(bias, smp_zero, (int2)(n_slice, 0));

  for (int k4 = 0; k4 < src_slices; ++k4) {
    // Read 4 src values (half4 = 4 consecutive k positions starting at k4*4).
    const half4 src4 = read_imageh(src, smp_zero, (int2)(m, k4));

    // Read 4 packed ushorts, one per output channel at this k4.
    // Each packs nibbles for k = k4*4..k4*4+3.
    const ushort4 packed4 = vload4(0, packed_weights + k4 * N + n_base);

    // Expand the 16 nibbles into 4 half4's (one per k_sub) and FMA.
    // Compiler unrolls and folds the shift into an immediate.
    #pragma unroll
    for (int k_sub = 0; k_sub < 4; ++k_sub) {
      const int shift = k_sub * 4;
      half4 w;
      w.x = (half)((int)((packed4.x >> shift) & 0xF) - 8) * scale4.x;
      w.y = (half)((int)((packed4.y >> shift) & 0xF) - 8) * scale4.y;
      w.z = (half)((int)((packed4.z >> shift) & 0xF) - 8) * scale4.z;
      w.w = (half)((int)((packed4.w >> shift) & 0xF) - 8) * scale4.w;

      // src.sN with dynamic N isn't allowed; unroll picks the right lane
      // via constant index at each iteration.
      half s;
      if (k_sub == 0) s = src4.s0;
      else if (k_sub == 1) s = src4.s1;
      else if (k_sub == 2) s = src4.s2;
      else                 s = src4.s3;

      acc += s * w;
    }
  }

  write_imageh(dst, (int2)(m, n_slice), acc);
}
