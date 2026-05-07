// Multi-output int4 GEMV for M=1 decode -- weights via image2d.
//
// Same algorithm as gpu_fused_gemv_int4 but reads each partition's
// weight matrix from a CL_RGBA16UI image2d view of the SVM weight
// bytes (image2d_from_buffer, zero-copy). Phase A2/A3 of the
// image2d migration: extends Phase B1's per-FC weight image2d
// success (+5.3% TPS bit-exact) to the QKV / Gate-Up batched
// dispatch.
//
// Image2D layout (per partition, identical to int4_gemv_weight_image2d):
//   width = N_part / 4, height = K / 4
//   pixel = ushort4 = 4 output channels x 4 K-position nibbles
//
// For the gate_up case (N_v == 0) the caller passes the q_image as
// the v_image dummy slot -- the kernel never enters the v branch
// when N_v == 0, so the image is never read.
//
// Activation kept SVM (matches v3's __constant binding for the
// per-FC path).

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__constant sampler_t fused_w_smp = CLK_NORMALIZED_COORDS_FALSE |
                                    CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

__attribute__((reqd_work_group_size(64, 1, 1)))
kernel void
gpu_fused_gemv_int4_image2d(__global const half *input,
                             __read_only image2d_t weights_q,
                             __global const half *scales_q,
                             __global half *q_out,
                             __read_only image2d_t weights_k,
                             __global const half *scales_k,
                             __global half *k_out,
                             __read_only image2d_t weights_v,
                             __global const half *scales_v,
                             __global half *v_out,
                             const int K,
                             const int N_q,
                             const int N_k,
                             const int N_v) {
  const int gid = get_global_id(0);
  const int n   = gid * 4;
  const int total_n = N_q + N_k + N_v;
  if (n >= total_n) return;

  __global const half *scales;
  __global half       *out;
  int local_n;
  int N_part;
  // Image2D handle selection: each partition uses its own image.
  // image2d_t can't go through ternary so we duplicate per branch.
  if (n < N_q) {
    scales = scales_q; out = q_out;
    local_n = n;        N_part = N_q;
  } else if (n < N_q + N_k) {
    scales = scales_k; out = k_out;
    local_n = n - N_q;  N_part = N_k;
  } else {
    scales = scales_v; out = v_out;
    local_n = n - N_q - N_k;
    N_part = N_v;
  }
  if (local_n >= N_part) return;

  const int n_pixel = local_n / 4;

  float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
  for (int k = 0; k < K; k += 4) {
    const half4 in_v = vload4(0, input + k);

    // Read weight pixel from the selected partition's image2d.
    uint4 packed_u;
    if (n < N_q) {
      packed_u = read_imageui(weights_q, fused_w_smp, (int2)(n_pixel, k / 4));
    } else if (n < N_q + N_k) {
      packed_u = read_imageui(weights_k, fused_w_smp, (int2)(n_pixel, k / 4));
    } else {
      packed_u = read_imageui(weights_v, fused_w_smp, (int2)(n_pixel, k / 4));
    }
    const ushort4 packed = (ushort4)((ushort)packed_u.x, (ushort)packed_u.y,
                                     (ushort)packed_u.z, (ushort)packed_u.w);

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

  const half4 scale = vload4(0, scales + local_n);
  out[local_n + 0] = (half)(acc0 * (float)scale.s0);
  out[local_n + 1] = (half)(acc1 * (float)scale.s1);
  out[local_n + 2] = (half)(acc2 * (float)scale.s2);
  out[local_n + 3] = (half)(acc3 * (float)scale.s3);
}
