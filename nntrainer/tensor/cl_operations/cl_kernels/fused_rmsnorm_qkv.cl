// Fused RMSNorm + Q/K/V projection for M=1 decode.
//
// Replaces the current 4-dispatch sequence per layer:
//   rms_norm (4096 elem)        -> dispatch 1
//   Q proj  fully_connected     -> dispatch 2 (output 4096)
//   K proj  fully_connected     -> dispatch 3 (output 1024)
//   V proj  fully_connected     -> dispatch 4 (output 1024)
// with a SINGLE dispatch that:
//   - reads input fp16[K_in] once
//   - computes sum_sq via cooperative workgroup reduction
//     (Adreno wavefront = 64, do tree reduction on shared mem)
//   - broadcasts inv_rms
//   - each work-item computes 4 output channels of Q, K, or V
//     depending on its global_id partition
//   - writes to one of three SVM out buffers
//
// Intermediate normalised input lives in __local memory throughout
// the dispatch -- saves three SVM memory roundtrips of the
// normalised hidden vector and removes three per-FC SVMMap
// drains from the host CPU side.
//
// Layout (all per-channel int4, channel-wise quant -- same as
// the existing gpu_int4_gemv_adreno):
//   weights_q : ushort[(K_in/4) * N_q]
//   scales_q  : half[N_q]    -- N_q = num_heads_q * head_dim
//   weights_k : ushort[(K_in/4) * N_k]
//   scales_k  : half[N_k]    -- N_k = num_heads_kv * head_dim
//   weights_v : ushort[(K_in/4) * N_v]    -- N_v = N_k
//   scales_v  : half[N_v]
//   gamma     : float[K_in]   -- rmsnorm scale, kept fp32 to match
//                               the canonical CPU rmsnorm path
//   input     : half[K_in]
//   q_out     : half[N_q]
//   k_out     : half[N_k]
//   v_out     : half[N_v]
//
// Dispatch:
//   global = ((N_q + N_k + N_v) / 4, 1, 1)
//   local  = (WG_SIZE = 64, 1, 1)
//
// Each WG handles 64 output channels (16 work-items * 4 ch/WI),
// all from the same partition (Q, K, or V) -- partitions are
// 256-aligned in our shapes so the WI's global_id maps cleanly.

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define WG_SIZE 64

__attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
kernel void
gpu_fused_rmsnorm_qkv(__global const half *input,
                      __global const float *gamma,
                      __global const ushort *weights_q,
                      __global const half *scales_q,
                      __global half *q_out,
                      __global const ushort *weights_k,
                      __global const half *scales_k,
                      __global half *k_out,
                      __global const ushort *weights_v,
                      __global const half *scales_v,
                      __global half *v_out,
                      const int K_in,
                      const int N_q,
                      const int N_k,
                      const int N_v,
                      const float epsilon) {
  const int lid = get_local_id(0);
  const int gid = get_global_id(0);
  const int n   = gid * 4;       // base output channel for this WI

  // -------- Pass 1: cooperative sum_sq over input --------
  // Each WI accumulates K_in / WG_SIZE elements (rounded up).
  // For Qwen3-4B K_in = 2560, WG_SIZE = 64 -> 40 elements/WI.
  __local float l_sum_sq[WG_SIZE];
  float partial = 0.0f;
  for (int k = lid; k < K_in; k += WG_SIZE) {
    const float v = (float)input[k];
    partial += v * v;
  }
  l_sum_sq[lid] = partial;
  barrier(CLK_LOCAL_MEM_FENCE);

  // Tree reduction.
  for (int step = WG_SIZE / 2; step > 0; step >>= 1) {
    if (lid < step) l_sum_sq[lid] += l_sum_sq[lid + step];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  const float inv_rms = 1.0f / sqrt(l_sum_sq[0] / (float)K_in + epsilon);

  // -------- Pass 2: cache normalised input in local memory --------
  // local[k] = input[k] * inv_rms * gamma[k]
  // Saves K_in memory reads in pass 3 (each WI of the WG would
  // otherwise re-read the same input + gamma when computing its
  // 4 output channels).
  __local half l_norm[2560];  // assumes K_in <= 2560 (Qwen3-4B)
  for (int k = lid; k < K_in; k += WG_SIZE) {
    l_norm[k] = (half)((float)input[k] * inv_rms * gamma[k]);
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  // -------- Pass 3: per-WI MAC over Q/K/V partitions --------
  // Identify which partition we're in based on gid's range.
  // Q: gid in [0, N_q/4)
  // K: gid in [N_q/4, (N_q+N_k)/4)
  // V: gid in [(N_q+N_k)/4, (N_q+N_k+N_v)/4)
  __global const ushort *weights;
  __global const half   *scales;
  __global half         *out;
  int local_n;       // 0-based channel within partition
  int N_part;
  if (n < N_q) {
    weights = weights_q; scales = scales_q; out = q_out;
    local_n = n;        N_part = N_q;
  } else if (n < N_q + N_k) {
    weights = weights_k; scales = scales_k; out = k_out;
    local_n = n - N_q;  N_part = N_k;
  } else {
    weights = weights_v; scales = scales_v; out = v_out;
    local_n = n - N_q - N_k;
    N_part = N_v;
  }
  if (local_n >= N_part) return;  // padding WIs do nothing

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
