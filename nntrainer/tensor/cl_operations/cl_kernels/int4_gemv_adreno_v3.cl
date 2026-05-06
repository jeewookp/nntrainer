
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// Channel-wise int4 GEMV kernel for Adreno (M = 1) -- v3.
// Algorithm and dispatch geometry byte-identical to v1; only the
// `input` parameter binding differs (`__constant` instead of
// `__global`) so any timing delta vs v1 isolates the input-side
// memory path.
//
// History:
//   * Phase A.1 -- __constant alone, no other QCOM hints: 4.43 TPS
//     vs 4.38 v1 baseline (+1.1%). Activation goes through
//     Adreno's per-CU constant cache (~64 KB advertised) instead
//     of L1. Worst-case Qwen3-4B input K=9728 fp16 = 19456 B,
//     well inside the budget; no max_constant_size hint set so
//     the compiler enforces the device max.
//   * Phase A.3/A.4 layered LiteRT-style hints from captured
//     program_002.cl (__attribute__((sub_group_uniform)) on input
//     +/- __attribute__((qcom_max_concurrent_subgroups(12))) on
//     the kernel + 3 cl_qcom_* pragmas). A.3 with both attrs
//     failed at compile (empty err log). A.4 dropped
//     qcom_max_concurrent_subgroups, kept sub_group_uniform +
//     pragmas; built but REGRESSED to 4.28 TPS (-3.4% vs A.1).
//     Reason: sub_group_uniform attribute alone, without the
//     matching DDR->xmem stream mechanism that LiteRT pairs it
//     with (qcom_sub_group_constant_load8), is net overhead --
//     driver sets up subgroup-broadcast paths that don't match
//     our 16-thread-WG GEMV access pattern.
//
// Reverted to A.1 state (just __constant). Replicating the real
// LiteRT speedup requires the explicit DDR->xmem stream call --
// Phase C, separate kernel.

kernel void
gpu_int4_gemv_adreno_v3(__constant half *input,
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
