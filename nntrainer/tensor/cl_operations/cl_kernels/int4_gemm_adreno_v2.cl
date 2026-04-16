#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable

// Phase 3c step 1: same kernel body as `gpu_int4_gemm_adreno` but the
// weight buffer is read through a TEXTURE path instead of generic
// `__global ushort *`. The wrapper wraps the existing
// channel-wise-packed ushort weights as an `image1d_buffer_t` with
// CL_RGBA / CL_UNSIGNED_INT16 format so each texel holds 4 consecutive
// channel ushorts (= 4 nibble-groups for one k-block), matching the
// `vload4(0, weights + (k/4)*N + n)` access the original kernel did.
//
// Motivation (see `tflite/delegates/gpu/common/tasks/fully_connected.cc`
// at origin/litert): LiteRT hits ~18.7 TFLOPS on Adreno 830 via fp16
// MACs, NOT via DP4A. The single biggest memory lever they pull is
// routing weight reads through the TEXTURE_2D path so the texture-cache
// + hardware UINT8->half conversion unit can run in parallel with ALU.
// Our existing fp16 kernel already reads INPUT from a texture
// (image1d_buffer_t) but was reading WEIGHTS from a regular __global
// buffer, so every weight fetch contended with activations/outputs for
// the same L1 port.
//
// Layout / divisibility: identical to v1.
//   weights  : [(K/4) * N] ushort, 4 nibbles per ushort
//   texture  : CL_RGBA/CL_UNSIGNED_INT16, width = (K/4 * N) / 4 texels
//   texel idx at (k-group kg, channel-group ng=n/4) = kg * (N/4) + ng
//
// Assumes N % 4 == 0, K % 4 == 0, scales_group_size == K (channel-wise),
// same as v1.

#define CEIL_DIV(a, b) (((a) + (b)-1) / (b))
#define ALIGN(a, b) (CEIL_DIV(a, b) * (b))

__attribute__((qcom_reqd_sub_group_size("full"))) kernel void
gpu_int4_gemm_adreno_v2(__read_only image1d_buffer_t input,
                        __global const half *scales, __global half *output,
                        __read_only image1d_buffer_t weights, const int K,
                        const int N, const int M,
                        const int quantization_group_size) {
  const int align_N = ALIGN(N, 32);

  const int m = get_global_id(0) * 2;
  const int n = get_global_id(1) * 4;
  const int M_4 = CEIL_DIV(M, 4);
  const int N_4 = N >> 2;

  half8 c0 = 0, c1 = 0, c2 = 0, c3 = 0;
  half8 input_reg;
  half4 dq_weights_reg;
  uint4 packed_w;   // read_imageui returns uint4 (zero-extended ushort lanes)
  half4 scale;

  for (int k = 0; k < K; k += 4) {
    if ((k & 0x1F) == 0) {
      scale = vload4(0, scales + (k / quantization_group_size) * align_N + n);
    }
    // Texel index matches `vload4(weights + (k/4)*N + n)` because the
    // channel stride inside a k-group is 1 ushort and we pack 4 ushorts
    // per texel (RGBA). n is a multiple of 4 (global_id(1) * 4) so
    // n/4 is exact and the 4 read lanes line up with channels
    // (n, n+1, n+2, n+3).
    packed_w = read_imageui(weights, (k >> 2) * N_4 + (n >> 2));

    input_reg.s0123 = read_imageh(input, k * M_4 + m);
    input_reg.s4567 = read_imageh(input, k * M_4 + m + 1);

    dq_weights_reg.s0 = (half)((int)(packed_w.s0 & 0x000Fu) - 8) * scale.s0;
    dq_weights_reg.s1 = (half)((int)(packed_w.s1 & 0x000Fu) - 8) * scale.s1;
    dq_weights_reg.s2 = (half)((int)(packed_w.s2 & 0x000Fu) - 8) * scale.s2;
    dq_weights_reg.s3 = (half)((int)(packed_w.s3 & 0x000Fu) - 8) * scale.s3;

    c0 += input_reg * dq_weights_reg.s0;
    c1 += input_reg * dq_weights_reg.s1;
    c2 += input_reg * dq_weights_reg.s2;
    c3 += input_reg * dq_weights_reg.s3;

    input_reg.s0123 = read_imageh(input, (k + 1) * M_4 + m);
    input_reg.s4567 = read_imageh(input, (k + 1) * M_4 + m + 1);

    dq_weights_reg.s0 = (half)((int)((packed_w.s0 >> 4) & 0x000Fu) - 8) * scale.s0;
    dq_weights_reg.s1 = (half)((int)((packed_w.s1 >> 4) & 0x000Fu) - 8) * scale.s1;
    dq_weights_reg.s2 = (half)((int)((packed_w.s2 >> 4) & 0x000Fu) - 8) * scale.s2;
    dq_weights_reg.s3 = (half)((int)((packed_w.s3 >> 4) & 0x000Fu) - 8) * scale.s3;

    c0 += input_reg * dq_weights_reg.s0;
    c1 += input_reg * dq_weights_reg.s1;
    c2 += input_reg * dq_weights_reg.s2;
    c3 += input_reg * dq_weights_reg.s3;

    input_reg.s0123 = read_imageh(input, (k + 2) * M_4 + m);
    input_reg.s4567 = read_imageh(input, (k + 2) * M_4 + m + 1);

    dq_weights_reg.s0 = (half)((int)((packed_w.s0 >> 8) & 0x000Fu) - 8) * scale.s0;
    dq_weights_reg.s1 = (half)((int)((packed_w.s1 >> 8) & 0x000Fu) - 8) * scale.s1;
    dq_weights_reg.s2 = (half)((int)((packed_w.s2 >> 8) & 0x000Fu) - 8) * scale.s2;
    dq_weights_reg.s3 = (half)((int)((packed_w.s3 >> 8) & 0x000Fu) - 8) * scale.s3;

    c0 += input_reg * dq_weights_reg.s0;
    c1 += input_reg * dq_weights_reg.s1;
    c2 += input_reg * dq_weights_reg.s2;
    c3 += input_reg * dq_weights_reg.s3;

    input_reg.s0123 = read_imageh(input, (k + 3) * M_4 + m);
    input_reg.s4567 = read_imageh(input, (k + 3) * M_4 + m + 1);

    dq_weights_reg.s0 = (half)((int)((packed_w.s0 >> 12) & 0x000Fu) - 8) * scale.s0;
    dq_weights_reg.s1 = (half)((int)((packed_w.s1 >> 12) & 0x000Fu) - 8) * scale.s1;
    dq_weights_reg.s2 = (half)((int)((packed_w.s2 >> 12) & 0x000Fu) - 8) * scale.s2;
    dq_weights_reg.s3 = (half)((int)((packed_w.s3 >> 12) & 0x000Fu) - 8) * scale.s3;

    c0 += input_reg * dq_weights_reg.s0;
    c1 += input_reg * dq_weights_reg.s1;
    c2 += input_reg * dq_weights_reg.s2;
    c3 += input_reg * dq_weights_reg.s3;
  }

  int idx = (m << 2) * N + n;

  if (idx + 3 < M * N) {
    vstore4((half4)(c0.s0, c1.s0, c2.s0, c3.s0), 0, output + idx);
    idx += N;
  }
  if (idx + 3 < M * N) {
    vstore4((half4)(c0.s1, c1.s1, c2.s1, c3.s1), 0, output + idx);
    idx += N;
  }
  if (idx + 3 < M * N) {
    vstore4((half4)(c0.s2, c1.s2, c2.s2, c3.s2), 0, output + idx);
    idx += N;
  }
  if (idx + 3 < M * N) {
    vstore4((half4)(c0.s3, c1.s3, c2.s3, c3.s3), 0, output + idx);
    idx += N;
  }
  if (idx + 3 < M * N) {
    vstore4((half4)(c0.s4, c1.s4, c2.s4, c3.s4), 0, output + idx);
    idx += N;
  }
  if (idx + 3 < M * N) {
    vstore4((half4)(c0.s5, c1.s5, c2.s5, c3.s5), 0, output + idx);
    idx += N;
  }
  if (idx + 3 < M * N) {
    vstore4((half4)(c0.s6, c1.s6, c2.s6, c3.s6), 0, output + idx);
    idx += N;
  }
  if (idx + 3 < M * N) {
    vstore4((half4)(c0.s7, c1.s7, c2.s7, c3.s7), 0, output + idx);
  }
}
