// Channel-wise int4 GEMV kernel for Adreno (M = 1) -- weights in image2d.
//
// Same algorithm as gpu_int4_gemv_adreno_v3 (Phase A: __constant
// activation), but reads the int4 packed weight matrix from a 2D
// image instead of __global SVM. Goal: leverage Adreno's texture
// cache (separate from L1 data cache) for the weight memory traffic
// that dominates M=1 GEMV bandwidth.
//
// Weight image2d layout:
//   CL_RGBA16UI, width = N / 4, height = K / 4.
//   pixel.{x,y,z,w} = ushort packed nibbles for output channels
//                     [n*4 + 0/1/2/3] at K positions [k*4..k*4+3].
//   Each ushort packs 4 K-nibbles for one output channel:
//     bits  [0..3]  -> K position k
//     bits  [4..7]  -> K position k+1
//     bits  [8..11] -> K position k+2
//     bits [12..15] -> K position k+3
//   Same byte layout as the existing SVM ushort[(K/4) * N] array;
//   the helper just creates an image2d view via clCreateImage with
//   CL_MEM_COPY_HOST_PTR initialised from the SVM data, cached
//   per-weight-pointer at first dispatch.
//
// Per-WI work: 4 output channels (n .. n+3), full K scan.
//
// Activation/scales/output stay SVM (matches gpu_int4_gemv_adreno_v3
// signature so caller can A/B switch via env-gate).
//
// Dispatch (host):
//   global = align(N, 256) / 4   (each WI handles 4 channels)
//   local  = (64, 1, 1)

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__constant sampler_t weight_smp = CLK_NORMALIZED_COORDS_FALSE |
                                   CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

kernel void
gpu_int4_gemv_weight_image2d(__constant half *input,
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
  for (int k = 0; k < K; k += 4) {
    const half4 in_v = vload4(0, input + k);
    const uint4 packed_u = read_imageui(weights, weight_smp,
                                        (int2)(n_pixel, k / 4));
    const ushort4 packed = (ushort4)((ushort)packed_u.x, (ushort)packed_u.y,
                                     (ushort)packed_u.z, (ushort)packed_u.w);

    // Lane k+0 (low 4 bits)
    const float in0 = (float)in_v.s0;
    acc0 += in0 * (float)((int)(packed.s0 & 0x000F) - 8);
    acc1 += in0 * (float)((int)(packed.s1 & 0x000F) - 8);
    acc2 += in0 * (float)((int)(packed.s2 & 0x000F) - 8);
    acc3 += in0 * (float)((int)(packed.s3 & 0x000F) - 8);

    // Lane k+1
    const float in1 = (float)in_v.s1;
    acc0 += in1 * (float)((int)((packed.s0 & 0x00F0) >> 4) - 8);
    acc1 += in1 * (float)((int)((packed.s1 & 0x00F0) >> 4) - 8);
    acc2 += in1 * (float)((int)((packed.s2 & 0x00F0) >> 4) - 8);
    acc3 += in1 * (float)((int)((packed.s3 & 0x00F0) >> 4) - 8);

    // Lane k+2
    const float in2 = (float)in_v.s2;
    acc0 += in2 * (float)((int)((packed.s0 & 0x0F00) >> 8) - 8);
    acc1 += in2 * (float)((int)((packed.s1 & 0x0F00) >> 8) - 8);
    acc2 += in2 * (float)((int)((packed.s2 & 0x0F00) >> 8) - 8);
    acc3 += in2 * (float)((int)((packed.s3 & 0x0F00) >> 8) - 8);

    // Lane k+3 (high 4 bits)
    const float in3 = (float)in_v.s3;
    acc0 += in3 * (float)((int)((packed.s0 & 0xF000) >> 12) - 8);
    acc1 += in3 * (float)((int)((packed.s1 & 0xF000) >> 12) - 8);
    acc2 += in3 * (float)((int)((packed.s2 & 0xF000) >> 12) - 8);
    acc3 += in3 * (float)((int)((packed.s3 & 0xF000) >> 12) - 8);
  }

  const half4 scale = vload4(0, scales + n);
  output[n + 0] = (half)(acc0 * (float)scale.s0);
  output[n + 1] = (half)(acc1 * (float)scale.s1);
  output[n + 2] = (half)(acc2 * (float)scale.s2);
  output[n + 3] = (half)(acc3 * (float)scale.s3);
}
