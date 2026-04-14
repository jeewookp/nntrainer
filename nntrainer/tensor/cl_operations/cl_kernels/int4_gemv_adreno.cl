
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
// independent across N. Float accumulator is used to avoid half overflow
// on long K (max abs term per K-step is ~7 * |input|, which can sum well
// past 65504 over thousands of K steps).
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

  float4 acc = (float4)(0.0f, 0.0f, 0.0f, 0.0f);

  for (int k = 0; k < K; k += 4) {
    const half4 in_v = vload4(0, input + k);
    const ushort4 packed = vload4(0, weights + (k / 4) * N + n);

    // Lane k+0 (low 4 bits of each ushort)
    const float4 w0 = convert_float4(
      convert_short4(packed & (ushort4)(0x000F)) - (short4)(8));
    acc += (float)in_v.s0 * w0;

    // Lane k+1
    const float4 w1 = convert_float4(
      convert_short4((packed >> 4) & (ushort4)(0x000F)) - (short4)(8));
    acc += (float)in_v.s1 * w1;

    // Lane k+2
    const float4 w2 = convert_float4(
      convert_short4((packed >> 8) & (ushort4)(0x000F)) - (short4)(8));
    acc += (float)in_v.s2 * w2;

    // Lane k+3 (high 4 bits)
    const float4 w3 = convert_float4(
      convert_short4((packed >> 12) & (ushort4)(0x000F)) - (short4)(8));
    acc += (float)in_v.s3 * w3;
  }

  const float4 scale = convert_float4(vload4(0, scales + n));
  const float4 result = acc * scale;
  vstore4(convert_half4(result), 0, output + n);
}
