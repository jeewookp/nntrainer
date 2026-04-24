
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define CEIL_DIV(a, b) (((a) + (b)-1) / (b))
#define ALIGN(a, b) (CEIL_DIV(a, b) * (b))

// Channel-wise int4 GEMV kernel for Adreno (M = 1 path).
//
// Memory layout (must match Int4Utils::convertKaiToChannelwise -- the
// same layout consumed by gpu_int4_gemm_adreno):
//
//   weights : ushort[(K/4) * N]
//             Each ushort packs 4 unsigned bias-8 nibbles for one channel
//             at K positions [k, k+1, k+2, k+3]:
//               bits  [0..3]  -> k
//               bits  [4..7]  -> k+1
//               bits  [8..11] -> k+2
//               bits [12..15] -> k+3
//             dequantized weight = ((nibble) - 8) * scale.
//
//   scales  : half[align(N, 32)]
//             one fp16 scale per output channel (per-channel quantization).
//
//   input   : half[K]
//             single-token activation (M = 1).
//
//   output  : half[N]
//             y[n] = scale[n] * sum_k ((nibble[k,n] - 8) * input[k])
//
// Each work-item produces 4 output channels (n .. n+3) by scanning the
// full K dimension. There is no cross-work-item reduction; outputs are
// independent across N. Float accumulators avoid half-precision overflow
// when K is large.
//
// Implementation note: kept to the same scalar element-wise style as
// gpu_int4_gemm_adreno (per-element ops on .s0/.s1/.s2/.s3) because the
// fancier vector convert_*/bit-mask form was rejected at runtime by the
// Adreno OpenCL compiler.
//
// Dispatch from the host:
//   global = align_N / 4   (each WI handles 4 channels)
//   local  = {16, 1, 1}    (any size that divides global cleanly works,
//                          the kernel does not use any sub-group ops)
kernel void
gpu_int4_gemv_adreno(__global const half *input,
                     __global const half *scales,
                     __global half *output,
                     __global const ushort *weights,
                     const int K,
                     const int N) {
  const int n = get_global_id(0) * 4;
  // Defensive guard for align_N > N (currently unused since all model N
  // values are multiples of 32, but keeps the kernel correct if that
  // assumption ever changes).
  if (n >= N)
    return;

  float acc0 = 0.0f;
  float acc1 = 0.0f;
  float acc2 = 0.0f;
  float acc3 = 0.0f;

  for (int k = 0; k < K; k += 4) {
    const half4 in_v = vload4(0, input + k);
    const ushort4 packed = vload4(0, weights + (k / 4) * N + n);

    // Lane k+0 (low 4 bits of each ushort)
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
