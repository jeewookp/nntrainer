// Channel-wise int4 GEMV (M=1) with RGBA32UI weight image2d.
//
// Variant of gpu_int4_gemv_weight_image2d that doubles K-bandwidth per
// imageread by packing 8 K-positions per channel into a single uint
// (instead of 4 K-positions per ushort). Halves the imageread count
// and halves the loop trip-count for the same FLOPs.
//
// Weight image layout:
//   CL_RGBA + UNSIGNED_INT32, width = N/4, height = K/8.
//   pixel.{x,y,z,w} = uint packed nibbles for output channels
//                     [n*4 + 0/1/2/3] at K positions [k*8..k*8+7].
//   Each uint packs 8 K-nibbles for one output channel:
//     bits  [0..3]   -> K position k*8+0
//     bits  [4..7]   -> K position k*8+1
//     bits  [8..11]  -> K position k*8+2
//     bits [12..15]  -> K position k*8+3
//     bits [16..19]  -> K position k*8+4
//     bits [20..23]  -> K position k*8+5
//     bits [24..27]  -> K position k*8+6
//     bits [28..31]  -> K position k*8+7
//
// Construction from the existing ushort[(K/4)*N] SVM weight:
//   uint = ((uint)src[(2j+1)*N + n] << 16) | (uint)src[2j*N + n]
//   stored at dst[j*N + n].
//
// Per-WI work: 4 output channels (n .. n+3), full K scan, 8 K per iter.
//
// Activation/scales/output stay SVM (matches the v1 helper signature).
//
// Dispatch:
//   global = align(N, 256) / 4   (each WI handles 4 channels)
//   local  = (64, 1, 1)

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__constant sampler_t weight_smp_v2 = CLK_NORMALIZED_COORDS_FALSE |
                                      CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

kernel void
gpu_int4_gemv_weight_image2d_v2(__constant half *input,
                                 __global const half *scales,
                                 __global half *output,
                                 __read_only image2d_t weights,
                                 const int K,
                                 const int N) {
  const int n = get_global_id(0) * 4;
  if (n >= N)
    return;

  float acc0 = 0.0f;
  float acc1 = 0.0f;
  float acc2 = 0.0f;
  float acc3 = 0.0f;

  const int n_pixel = n / 4;
  for (int k = 0; k < K; k += 8) {
    // Load 8 inputs at once (16 bytes coalesced).
    const half8 in_v = vload8(0, input + k);
    // Load weight pixel: 4 N channels × 8 K-nibbles each = 16 bytes.
    const uint4 packed = read_imageui(weights, weight_smp_v2,
                                      (int2)(n_pixel, k / 8));

    // 8 K-positions × 4 N-channels = 32 MACs per iter.
    // s0
    {
      const float in_k = (float)in_v.s0;
      acc0 += in_k * (float)((int)((packed.s0 >>  0) & 0xF) - 8);
      acc1 += in_k * (float)((int)((packed.s1 >>  0) & 0xF) - 8);
      acc2 += in_k * (float)((int)((packed.s2 >>  0) & 0xF) - 8);
      acc3 += in_k * (float)((int)((packed.s3 >>  0) & 0xF) - 8);
    }
    // s1
    {
      const float in_k = (float)in_v.s1;
      acc0 += in_k * (float)((int)((packed.s0 >>  4) & 0xF) - 8);
      acc1 += in_k * (float)((int)((packed.s1 >>  4) & 0xF) - 8);
      acc2 += in_k * (float)((int)((packed.s2 >>  4) & 0xF) - 8);
      acc3 += in_k * (float)((int)((packed.s3 >>  4) & 0xF) - 8);
    }
    // s2
    {
      const float in_k = (float)in_v.s2;
      acc0 += in_k * (float)((int)((packed.s0 >>  8) & 0xF) - 8);
      acc1 += in_k * (float)((int)((packed.s1 >>  8) & 0xF) - 8);
      acc2 += in_k * (float)((int)((packed.s2 >>  8) & 0xF) - 8);
      acc3 += in_k * (float)((int)((packed.s3 >>  8) & 0xF) - 8);
    }
    // s3
    {
      const float in_k = (float)in_v.s3;
      acc0 += in_k * (float)((int)((packed.s0 >> 12) & 0xF) - 8);
      acc1 += in_k * (float)((int)((packed.s1 >> 12) & 0xF) - 8);
      acc2 += in_k * (float)((int)((packed.s2 >> 12) & 0xF) - 8);
      acc3 += in_k * (float)((int)((packed.s3 >> 12) & 0xF) - 8);
    }
    // s4
    {
      const float in_k = (float)in_v.s4;
      acc0 += in_k * (float)((int)((packed.s0 >> 16) & 0xF) - 8);
      acc1 += in_k * (float)((int)((packed.s1 >> 16) & 0xF) - 8);
      acc2 += in_k * (float)((int)((packed.s2 >> 16) & 0xF) - 8);
      acc3 += in_k * (float)((int)((packed.s3 >> 16) & 0xF) - 8);
    }
    // s5
    {
      const float in_k = (float)in_v.s5;
      acc0 += in_k * (float)((int)((packed.s0 >> 20) & 0xF) - 8);
      acc1 += in_k * (float)((int)((packed.s1 >> 20) & 0xF) - 8);
      acc2 += in_k * (float)((int)((packed.s2 >> 20) & 0xF) - 8);
      acc3 += in_k * (float)((int)((packed.s3 >> 20) & 0xF) - 8);
    }
    // s6
    {
      const float in_k = (float)in_v.s6;
      acc0 += in_k * (float)((int)((packed.s0 >> 24) & 0xF) - 8);
      acc1 += in_k * (float)((int)((packed.s1 >> 24) & 0xF) - 8);
      acc2 += in_k * (float)((int)((packed.s2 >> 24) & 0xF) - 8);
      acc3 += in_k * (float)((int)((packed.s3 >> 24) & 0xF) - 8);
    }
    // s7
    {
      const float in_k = (float)in_v.s7;
      acc0 += in_k * (float)((int)((packed.s0 >> 28) & 0xF) - 8);
      acc1 += in_k * (float)((int)((packed.s1 >> 28) & 0xF) - 8);
      acc2 += in_k * (float)((int)((packed.s2 >> 28) & 0xF) - 8);
      acc3 += in_k * (float)((int)((packed.s3 >> 28) & 0xF) - 8);
    }
  }

  const half4 scale = vload4(0, scales + n);
  output[n + 0] = (half)(acc0 * (float)scale.s0);
  output[n + 1] = (half)(acc1 * (float)scale.s1);
  output[n + 2] = (half)(acc2 * (float)scale.s2);
  output[n + 3] = (half)(acc3 * (float)scale.s3);
}
