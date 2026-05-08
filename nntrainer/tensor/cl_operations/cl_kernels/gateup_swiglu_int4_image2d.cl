// Phase L-1: gate_up + swiglu fused into a single kernel.
//
// Each lane computes gate[n..n+3] AND up[n..n+3] (same N indices)
// instead of either-or partitions. After the K loop, applies scales,
// computes silu(gate) * up inline, writes swiglu output.
//
// Eliminates:
//   * The separate swiglu kernel dispatch (~30us host overhead +
//     drain wait per layer per token = 36 * 64 = 2304 dispatches /
//     decode)
//   * The 2 x 9728 fp16 SVM writes (gate_out + up_out staged) -- now
//     just 1 x 9728 (swiglu_out).
//
// Memory traffic per iter: 2 image2d reads (gate + up) + 1 input
// vload. Same input cache reuse as fused_gemv. Each lane holds 8
// fp32 accumulators (4 gate + 4 up), well under register pressure
// limit on Adreno.
//
// Image2D layout (per partition, identical to v2 image2d_v2):
//   CL_RGBA + UNSIGNED_INT32, width = N/4, height = K/8.
//   pixel.{x,y,z,w} = uint packed 8 K-nibbles for output channels
//                     [n*4 + 0/1/2/3] at K positions [k*8..k*8+7].

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__constant sampler_t gateup_swiglu_smp =
  CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

__attribute__((reqd_work_group_size(64, 1, 1)))
kernel void
gpu_gateup_swiglu_int4_image2d(__global const half *input,
                                __read_only image2d_t weights_g,
                                __global const half *scales_g,
                                __read_only image2d_t weights_u,
                                __global const half *scales_u,
                                __global half *swiglu_out,
                                const int K,
                                const int N,
                                const int M) {
  const int n = get_global_id(0) * 4;
  const int m = get_global_id(1);
  if (n >= N || m >= M) return;

  const int n_pixel = n / 4;
  const int input_off = m * K;
  const int output_off = m * N;

  float g0 = 0.0f, g1 = 0.0f, g2 = 0.0f, g3 = 0.0f;
  float u0 = 0.0f, u1 = 0.0f, u2 = 0.0f, u3 = 0.0f;

  for (int k = 0; k < K; k += 16) {
    const half8 in_lo = vload8(0, input + input_off + k);
    const half8 in_hi = vload8(0, input + input_off + k + 8);
    const uint4 g_lo = read_imageui(weights_g, gateup_swiglu_smp,
                                     (int2)(n_pixel, k / 8));
    const uint4 g_hi = read_imageui(weights_g, gateup_swiglu_smp,
                                     (int2)(n_pixel, k / 8 + 1));
    const uint4 u_lo = read_imageui(weights_u, gateup_swiglu_smp,
                                     (int2)(n_pixel, k / 8));
    const uint4 u_hi = read_imageui(weights_u, gateup_swiglu_smp,
                                     (int2)(n_pixel, k / 8 + 1));

#define STEP(packed, in_v, lane_idx, shift, a0, a1, a2, a3)            \
  do {                                                                 \
    const float in_k = (float)in_v.s##lane_idx;                        \
    a0 += in_k * (float)((int)((packed.s0 >> shift) & 0xF) - 8);       \
    a1 += in_k * (float)((int)((packed.s1 >> shift) & 0xF) - 8);       \
    a2 += in_k * (float)((int)((packed.s2 >> shift) & 0xF) - 8);       \
    a3 += in_k * (float)((int)((packed.s3 >> shift) & 0xF) - 8);       \
  } while (0)

    // 8 K positions (k+0..k+7) from packed_lo / in_lo, for gate AND up.
    STEP(g_lo, in_lo, 0,  0, g0, g1, g2, g3);
    STEP(u_lo, in_lo, 0,  0, u0, u1, u2, u3);
    STEP(g_lo, in_lo, 1,  4, g0, g1, g2, g3);
    STEP(u_lo, in_lo, 1,  4, u0, u1, u2, u3);
    STEP(g_lo, in_lo, 2,  8, g0, g1, g2, g3);
    STEP(u_lo, in_lo, 2,  8, u0, u1, u2, u3);
    STEP(g_lo, in_lo, 3, 12, g0, g1, g2, g3);
    STEP(u_lo, in_lo, 3, 12, u0, u1, u2, u3);
    STEP(g_lo, in_lo, 4, 16, g0, g1, g2, g3);
    STEP(u_lo, in_lo, 4, 16, u0, u1, u2, u3);
    STEP(g_lo, in_lo, 5, 20, g0, g1, g2, g3);
    STEP(u_lo, in_lo, 5, 20, u0, u1, u2, u3);
    STEP(g_lo, in_lo, 6, 24, g0, g1, g2, g3);
    STEP(u_lo, in_lo, 6, 24, u0, u1, u2, u3);
    STEP(g_lo, in_lo, 7, 28, g0, g1, g2, g3);
    STEP(u_lo, in_lo, 7, 28, u0, u1, u2, u3);
    // 8 K positions (k+8..k+15) from packed_hi / in_hi.
    STEP(g_hi, in_hi, 0,  0, g0, g1, g2, g3);
    STEP(u_hi, in_hi, 0,  0, u0, u1, u2, u3);
    STEP(g_hi, in_hi, 1,  4, g0, g1, g2, g3);
    STEP(u_hi, in_hi, 1,  4, u0, u1, u2, u3);
    STEP(g_hi, in_hi, 2,  8, g0, g1, g2, g3);
    STEP(u_hi, in_hi, 2,  8, u0, u1, u2, u3);
    STEP(g_hi, in_hi, 3, 12, g0, g1, g2, g3);
    STEP(u_hi, in_hi, 3, 12, u0, u1, u2, u3);
    STEP(g_hi, in_hi, 4, 16, g0, g1, g2, g3);
    STEP(u_hi, in_hi, 4, 16, u0, u1, u2, u3);
    STEP(g_hi, in_hi, 5, 20, g0, g1, g2, g3);
    STEP(u_hi, in_hi, 5, 20, u0, u1, u2, u3);
    STEP(g_hi, in_hi, 6, 24, g0, g1, g2, g3);
    STEP(u_hi, in_hi, 6, 24, u0, u1, u2, u3);
    STEP(g_hi, in_hi, 7, 28, g0, g1, g2, g3);
    STEP(u_hi, in_hi, 7, 28, u0, u1, u2, u3);
#undef STEP
  }

  // Apply scales and SwiGLU inline:  silu(g) * u  =  g / (1 + exp(-g)) * u
  // SiLU stays in fp32 for stability; final cast to half on output.
  const half4 sg = vload4(0, scales_g + n);
  const half4 su = vload4(0, scales_u + n);

  const float g0s = g0 * (float)sg.s0;
  const float g1s = g1 * (float)sg.s1;
  const float g2s = g2 * (float)sg.s2;
  const float g3s = g3 * (float)sg.s3;
  const float u0s = u0 * (float)su.s0;
  const float u1s = u1 * (float)su.s1;
  const float u2s = u2 * (float)su.s2;
  const float u3s = u3 * (float)su.s3;

  const float silu0 = g0s / (1.0f + native_exp(-g0s));
  const float silu1 = g1s / (1.0f + native_exp(-g1s));
  const float silu2 = g2s / (1.0f + native_exp(-g2s));
  const float silu3 = g3s / (1.0f + native_exp(-g3s));

  swiglu_out[output_off + n + 0] = (half)(silu0 * u0s);
  swiglu_out[output_off + n + 1] = (half)(silu1 * u1s);
  swiglu_out[output_off + n + 2] = (half)(silu2 * u2s);
  swiglu_out[output_off + n + 3] = (half)(silu3 * u3s);
}
