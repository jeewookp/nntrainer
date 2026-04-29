// Fused post-attention RMSNorm + Gate/Up projection for M=1 decode.
//
// Replaces the 3-dispatch sequence per layer (post-attention path):
//   rms_norm (K_in elem)        -> dispatch 1
//   Gate proj fully_connected   -> dispatch 2 (output N_gate)
//   Up   proj fully_connected   -> dispatch 3 (output N_up)
// with a SINGLE dispatch that reuses the cooperative sum_sq +
// __local normalised input cache pattern from
// fused_rmsnorm_qkv.cl.
//
// For Qwen3-4B: K_in = 2560 (post-residual), N_gate = N_up = 9728,
// so total = 19456 output channels = 4864 work-items at 4 ch/WI =
// 76 work-groups of 64 lanes.
//
// All weights/scales are channel-wise int4 quantised, same layout
// as gpu_int4_gemv_adreno.

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define WG_SIZE 64

__attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
kernel void
gpu_fused_rmsnorm_gate_up(__global const half *input,
                          __global const float *gamma,
                          __global const ushort *weights_gate,
                          __global const half *scales_gate,
                          __global half *gate_out,
                          __global const ushort *weights_up,
                          __global const half *scales_up,
                          __global half *up_out,
                          const int K_in,
                          const int N_gate,
                          const int N_up,
                          const float epsilon) {
  const int lid = get_local_id(0);
  const int gid = get_global_id(0);
  const int n   = gid * 4;

  // -------- Pass 1: cooperative sum_sq over input --------
  __local float l_sum_sq[WG_SIZE];
  float partial = 0.0f;
  for (int k = lid; k < K_in; k += WG_SIZE) {
    const float v = (float)input[k];
    partial += v * v;
  }
  l_sum_sq[lid] = partial;
  barrier(CLK_LOCAL_MEM_FENCE);

  for (int step = WG_SIZE / 2; step > 0; step >>= 1) {
    if (lid < step) l_sum_sq[lid] += l_sum_sq[lid + step];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  const float inv_rms = 1.0f / sqrt(l_sum_sq[0] / (float)K_in + epsilon);

  // -------- Pass 2: cache normalised input in local memory --------
  __local half l_norm[2560];  // assumes K_in <= 2560 (Qwen3-4B)
  for (int k = lid; k < K_in; k += WG_SIZE) {
    l_norm[k] = (half)((float)input[k] * inv_rms * gamma[k]);
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  // -------- Pass 3: per-WI MAC over Gate/Up partitions --------
  // Gate: gid in [0, N_gate/4)
  // Up:   gid in [N_gate/4, (N_gate+N_up)/4)
  __global const ushort *weights;
  __global const half   *scales;
  __global half         *out;
  int local_n;
  int N_part;
  if (n < N_gate) {
    weights = weights_gate; scales = scales_gate; out = gate_out;
    local_n = n;            N_part = N_gate;
  } else {
    weights = weights_up;   scales = scales_up;   out = up_out;
    local_n = n - N_gate;   N_part = N_up;
  }
  if (local_n >= N_part) return;

  float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
  for (int k = 0; k < K_in; k += 4) {
    const float in0 = (float)l_norm[k + 0];
    const float in1 = (float)l_norm[k + 1];
    const float in2 = (float)l_norm[k + 2];
    const float in3 = (float)l_norm[k + 3];

    const ushort4 packed = vload4(0, weights + (k / 4) * N_part + local_n);

    acc0 += in0 * (float)((int)(packed.s0 & 0x000F) - 8);
    acc1 += in0 * (float)((int)(packed.s1 & 0x000F) - 8);
    acc2 += in0 * (float)((int)(packed.s2 & 0x000F) - 8);
    acc3 += in0 * (float)((int)(packed.s3 & 0x000F) - 8);

    acc0 += in1 * (float)((int)((packed.s0 & 0x00F0) >> 4) - 8);
    acc1 += in1 * (float)((int)((packed.s1 & 0x00F0) >> 4) - 8);
    acc2 += in1 * (float)((int)((packed.s2 & 0x00F0) >> 4) - 8);
    acc3 += in1 * (float)((int)((packed.s3 & 0x00F0) >> 4) - 8);

    acc0 += in2 * (float)((int)((packed.s0 & 0x0F00) >> 8) - 8);
    acc1 += in2 * (float)((int)((packed.s1 & 0x0F00) >> 8) - 8);
    acc2 += in2 * (float)((int)((packed.s2 & 0x0F00) >> 8) - 8);
    acc3 += in2 * (float)((int)((packed.s3 & 0x0F00) >> 8) - 8);

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
