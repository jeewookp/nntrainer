// Step 5 / incremental: + vload4 input/weight + 4 channels per WI.
//
// Step 4 result: gpu_kernel = 2912 ms, correct output.  Almost
// identical to baseline math but arranged as WG=64, 1 channel/WI
// with scalar loads.  Step 5 moves to 4 channels per WI matching
// the baseline's packing strategy, with vload4 on input and
// weights:
//
//   const half4   in_v   = vload4(0, input + k);
//   const ushort4 packed = vload4(0, weights + (k/4) * N + n);
//
// Memory: 4x fewer weight loads per k-block (ushort4 vs 4 scalar
// ushorts), and one half4 load covers 4 k-lanes of input.  Inside
// the iteration: 16 MACs arranged per (k_lane, channel) so the
// compiler can schedule them with 4-way ILP on the accumulator
// chain.
//
// Dispatch: local = {64, 1, 1}, global = align(N, 256) / 4.  WG
// of 64 covers 4*64 = 256 output channels.  All Qwen3-4B FC N
// values (1024 / 2560 / 4096 / 9728) are multiples of 256.
//
// Expected: gpu_kernel drops toward the ~500 us/call compute
// floor if the 4-way ILP + coalesced loads materialize.

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__attribute__((reqd_work_group_size(64, 1, 1)))
kernel void
gpu_int4_gemv_image_v5(__global const half *input,
                       __global const half *scales,
                       __global half *output,
                       __global const ushort *weights,
                       const int K,
                       const int N) {
  const int n = get_global_id(0) * 4;
  if (n >= N) return;

  float acc0 = 0.0f;
  float acc1 = 0.0f;
  float acc2 = 0.0f;
  float acc3 = 0.0f;

  for (int k = 0; k < K; k += 4) {
    const half4 in_v = vload4(0, input + k);
    const ushort4 packed = vload4(0, weights + (k / 4) * N + n);

    const float in0 = (float)in_v.s0;
    acc0 += in0 * (float)((int)(packed.s0 & 0x000F) - 8);
    acc1 += in0 * (float)((int)(packed.s1 & 0x000F) - 8);
    acc2 += in0 * (float)((int)(packed.s2 & 0x000F) - 8);
    acc3 += in0 * (float)((int)(packed.s3 & 0x000F) - 8);

    const float in1 = (float)in_v.s1;
    acc0 += in1 * (float)((int)((packed.s0 & 0x00F0) >> 4) - 8);
    acc1 += in1 * (float)((int)((packed.s1 & 0x00F0) >> 4) - 8);
    acc2 += in1 * (float)((int)((packed.s2 & 0x00F0) >> 4) - 8);
    acc3 += in1 * (float)((int)((packed.s3 & 0x00F0) >> 4) - 8);

    const float in2 = (float)in_v.s2;
    acc0 += in2 * (float)((int)((packed.s0 & 0x0F00) >> 8) - 8);
    acc1 += in2 * (float)((int)((packed.s1 & 0x0F00) >> 8) - 8);
    acc2 += in2 * (float)((int)((packed.s2 & 0x0F00) >> 8) - 8);
    acc3 += in2 * (float)((int)((packed.s3 & 0x0F00) >> 8) - 8);

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
