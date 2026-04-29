// Multi-output int4 GEMV for M=1 decode -- fuses 2 or 3 partition
// projections that share the same activation into a single dispatch.
//
// Use cases:
//   - QKV projections: (N_q, N_k, N_v) all > 0
//   - Gate/Up MLP:    (N_q=N_gate, N_k=N_up, N_v=0)
//
// Compared to running the baseline gpu_int4_gemv_adreno once per
// partition, this fuses N drains into 1 (one SVMMap at the end of
// the dispatch instead of one per partition).  For Qwen3-4B that's
// the difference between (3 + 2) = 5 SVMMap drains/layer and 2 -- a
// ~750 us saving per layer at ~250 us per drain.
//
// Each work-item handles 4 output channels (matches baseline gemv).
// Dispatch:
//   global = (N_q + N_k + N_v) / 4   (round up to multiple of WG=64)
//   local  = (64, 1, 1)
//
// Weight/scale layout per partition is identical to the baseline
// gpu_int4_gemv_adreno (channel-wise int4 in ushort, packed 4 nibbles
// at a time along K, and one fp16 scale per output channel).

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__attribute__((reqd_work_group_size(64, 1, 1)))
kernel void
gpu_fused_gemv_int4(__global const half *input,
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
