// __global uint-packed int4 GEMV (M=1) without image2d.
//
// Same memory layout as gpu_int4_gemv_weight_image2d_v2 (8 K-positions
// per uint per channel, K/8 row-major × N/4 column groups), but reads
// the weight via __global const uint4 * instead of image2d_t. Use case:
// FCs with N too large for the device's CL_DEVICE_IMAGE2D_MAX_WIDTH
// (typically 16384, so N/4 must be ≤ 16384 → N ≤ 65536). For Qwen3-4B
// the lm_head weight has N=152064 (vocab) and falls through here.
//
// Per-iter work: 8 K-positions × 4 N-channels = 32 MACs (same as v2).
//
// Memory saved vs the legacy ushort SVM kernel (gpu_int4_gemv_adreno_v3):
//   ushort layout:  K/4 row-major rows × N ushorts → 16-byte read per
//                   iter holds 4 K × 4 N = 16 nibbles
//   uint layout:    K/8 row-major rows × N uints → same 16-byte read
//                   holds 8 K × 4 N = 32 nibbles (2x fewer reads)

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

kernel void
gpu_int4_gemv_adreno_v4(__constant half *input,
                         __global const half *scales,
                         __global half *output,
                         __global const uint4 *weights,
                         const int K,
                         const int N) {
  const int n = get_global_id(0) * 4;
  if (n >= N)
    return;

  float acc0 = 0.0f;
  float acc1 = 0.0f;
  float acc2 = 0.0f;
  float acc3 = 0.0f;

  const int n_quad = n / 4;
  const int row_quads = N / 4;
  for (int k = 0; k < K; k += 8) {
    const half8 in_v = vload8(0, input + k);
    const uint4 packed = weights[(k / 8) * row_quads + n_quad];

    {
      const float in_k = (float)in_v.s0;
      acc0 += in_k * (float)((int)((packed.s0 >>  0) & 0xF) - 8);
      acc1 += in_k * (float)((int)((packed.s1 >>  0) & 0xF) - 8);
      acc2 += in_k * (float)((int)((packed.s2 >>  0) & 0xF) - 8);
      acc3 += in_k * (float)((int)((packed.s3 >>  0) & 0xF) - 8);
    }
    {
      const float in_k = (float)in_v.s1;
      acc0 += in_k * (float)((int)((packed.s0 >>  4) & 0xF) - 8);
      acc1 += in_k * (float)((int)((packed.s1 >>  4) & 0xF) - 8);
      acc2 += in_k * (float)((int)((packed.s2 >>  4) & 0xF) - 8);
      acc3 += in_k * (float)((int)((packed.s3 >>  4) & 0xF) - 8);
    }
    {
      const float in_k = (float)in_v.s2;
      acc0 += in_k * (float)((int)((packed.s0 >>  8) & 0xF) - 8);
      acc1 += in_k * (float)((int)((packed.s1 >>  8) & 0xF) - 8);
      acc2 += in_k * (float)((int)((packed.s2 >>  8) & 0xF) - 8);
      acc3 += in_k * (float)((int)((packed.s3 >>  8) & 0xF) - 8);
    }
    {
      const float in_k = (float)in_v.s3;
      acc0 += in_k * (float)((int)((packed.s0 >> 12) & 0xF) - 8);
      acc1 += in_k * (float)((int)((packed.s1 >> 12) & 0xF) - 8);
      acc2 += in_k * (float)((int)((packed.s2 >> 12) & 0xF) - 8);
      acc3 += in_k * (float)((int)((packed.s3 >> 12) & 0xF) - 8);
    }
    {
      const float in_k = (float)in_v.s4;
      acc0 += in_k * (float)((int)((packed.s0 >> 16) & 0xF) - 8);
      acc1 += in_k * (float)((int)((packed.s1 >> 16) & 0xF) - 8);
      acc2 += in_k * (float)((int)((packed.s2 >> 16) & 0xF) - 8);
      acc3 += in_k * (float)((int)((packed.s3 >> 16) & 0xF) - 8);
    }
    {
      const float in_k = (float)in_v.s5;
      acc0 += in_k * (float)((int)((packed.s0 >> 20) & 0xF) - 8);
      acc1 += in_k * (float)((int)((packed.s1 >> 20) & 0xF) - 8);
      acc2 += in_k * (float)((int)((packed.s2 >> 20) & 0xF) - 8);
      acc3 += in_k * (float)((int)((packed.s3 >> 20) & 0xF) - 8);
    }
    {
      const float in_k = (float)in_v.s6;
      acc0 += in_k * (float)((int)((packed.s0 >> 24) & 0xF) - 8);
      acc1 += in_k * (float)((int)((packed.s1 >> 24) & 0xF) - 8);
      acc2 += in_k * (float)((int)((packed.s2 >> 24) & 0xF) - 8);
      acc3 += in_k * (float)((int)((packed.s3 >> 24) & 0xF) - 8);
    }
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
