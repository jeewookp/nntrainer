// fused_gemv_int4 -- v2.
// Identical to fused_gemv_int4.cl except `input` is bound through
// __constant memory.  Same trick that gpu_int4_gemv_adreno_v3 used to
// pick up +1.1% TPS on the per-FC dotQInteger M=1 path -- routes the
// activation vector through Adreno's per-CU constant cache (separate
// from L1) so all 64 lanes of every WG share one cached read instead
// of independent L1 traffic.
//
// fused_gemv_int4 handles the QKV (3 partitions) and Gate-Up (2
// partitions) batched M=1 calls, which together represent a much
// larger weight share per layer than the per-FC path. Same
// constant-cache mechanism, applied where it matters most.
//
// All other parameters and the inner-loop math are byte-identical to
// fused_gemv_int4.cl, so any timing delta is purely the input-side
// memory path.

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__attribute__((reqd_work_group_size(64, 1, 1)))
kernel void
gpu_fused_gemv_int4_v2(__constant half *input,
                       __global const ushort *weights_q,
                       __global const half *scales_q,
                       __global half *q_out,
                       __global const ushort *weights_k,
                       __global const half *scales_k,
                       __global half *k_out,
                       __global const ushort *weights_v,
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

  __global const ushort *weights;
  __global const half   *scales;
  __global half         *out;
  int local_n;
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
  if (local_n >= N_part) return;

  float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
  for (int k = 0; k < K; k += 4) {
    const half4 in_v = vload4(0, input + k);
    const ushort4 packed = vload4(0, weights + (k / 4) * N_part + local_n);

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
