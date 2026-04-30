// Fused SwiGLU + int4 GEMV for M=1 decode (down projection in MLP).
//
// Replaces the 2-dispatch sequence:
//   swiglu  : tmp[k] = silu(gate[k]) * up[k]            -- per-elem
//   down FC : out[n] = scale[n] * sum_k weight[n][k] * tmp[k]
// with a single dispatch that streams the silu*up computation
// in-register without ever materialising tmp in SVM.  Saves:
//   - 1 GPU dispatch per layer (swiglu kernel)
//   - 1 SVMMap drain on swiglu output (if any)
//   - 1 SVM write + 1 SVM read of the K-element tmp buffer
// For Qwen3-4B K=intermediate=9728, that's ~38 KB of SVM traffic
// avoided per layer.
//
// Memory layout matches int4_gemv_adreno.cl:
//   weights : ushort[(K/4) * N], 4 nibbles per ushort (channel-wise int4)
//   scales  : half[align(N, 32)]
//   gate, up: half[K], single-token activations (M = 1)
//   output  : half[N]
//
// Each WI produces 4 output channels (n .. n+3) by scanning K.
//
// Dispatch:
//   global = align(N, 256) / 4
//   local  = (64, 1, 1)

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define CEIL_DIV(a, b) (((a) + (b)-1) / (b))
#define ALIGN(a, b) (CEIL_DIV(a, b) * (b))

kernel void
gpu_swiglu_down_int4(__global const half *gate,
                     __global const half *up,
                     __global const half *scales,
                     __global half *output,
                     __global const ushort *weights,
                     const int K,
                     const int N) {
  const int n = get_global_id(0) * 4;
  if (n >= N)
    return;

  float acc0 = 0.0f;
  float acc1 = 0.0f;
  float acc2 = 0.0f;
  float acc3 = 0.0f;

  for (int k = 0; k < K; k += 4) {
    // Load gate and up; compute silu(gate)*up in fp32 once.
    const half4 g_v = vload4(0, gate + k);
    const half4 u_v = vload4(0, up + k);

    const float g0 = (float)g_v.s0;
    const float g1 = (float)g_v.s1;
    const float g2 = (float)g_v.s2;
    const float g3 = (float)g_v.s3;
    const float u0 = (float)u_v.s0;
    const float u1 = (float)u_v.s1;
    const float u2 = (float)u_v.s2;
    const float u3 = (float)u_v.s3;

    // silu(x) = x / (1 + exp(-x)).  Match swiglu CPU helper math.
    const float in0 = (g0 / (1.0f + native_exp(-g0))) * u0;
    const float in1 = (g1 / (1.0f + native_exp(-g1))) * u1;
    const float in2 = (g2 / (1.0f + native_exp(-g2))) * u2;
    const float in3 = (g3 / (1.0f + native_exp(-g3))) * u3;

    const ushort4 packed = vload4(0, weights + (k / 4) * N + n);

    // Lane k+0 (low 4 bits)
    acc0 += in0 * (float)((int)(packed.s0 & 0x000F) - 8);
    acc1 += in0 * (float)((int)(packed.s1 & 0x000F) - 8);
    acc2 += in0 * (float)((int)(packed.s2 & 0x000F) - 8);
    acc3 += in0 * (float)((int)(packed.s3 & 0x000F) - 8);

    // Lane k+1
    acc0 += in1 * (float)((int)((packed.s0 & 0x00F0) >> 4) - 8);
    acc1 += in1 * (float)((int)((packed.s1 & 0x00F0) >> 4) - 8);
    acc2 += in1 * (float)((int)((packed.s2 & 0x00F0) >> 4) - 8);
    acc3 += in1 * (float)((int)((packed.s3 & 0x00F0) >> 4) - 8);

    // Lane k+2
    acc0 += in2 * (float)((int)((packed.s0 & 0x0F00) >> 8) - 8);
    acc1 += in2 * (float)((int)((packed.s1 & 0x0F00) >> 8) - 8);
    acc2 += in2 * (float)((int)((packed.s2 & 0x0F00) >> 8) - 8);
    acc3 += in2 * (float)((int)((packed.s3 & 0x0F00) >> 8) - 8);

    // Lane k+3 (high 4 bits)
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
