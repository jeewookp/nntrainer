
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// Channel-wise int4 GEMV kernel for Adreno (M = 1) -- v3.
//
// Layered on top of int4_gemv_adreno_v2 (which puts `input` in
// __constant memory). v3 adds two LiteRT-style QCOM-specific hints
// observed in the captured program_002.cl:
//
//   1. cl_qcom_subgroup_uniform_load + __attribute__((sub_group_uniform))
//      on the input pointer. For M=1 GEMV every lane in a subgroup
//      reads the SAME input[k] (same k, different n) -- textbook
//      subgroup-uniform pattern. Marking it lets the driver issue one
//      load that broadcasts to all subgroup lanes instead of one load
//      per lane.
//
//   2. __attribute__((qcom_max_concurrent_subgroups(N))) on the kernel
//      tells the driver to schedule up to N subgroups concurrently
//      (LiteRT used 12 in program_002.cl). Helps occupancy for small
//      decode kernels.
//
// Both attributes are gated on the cl_qcom_subgroup_uniform_load
// extension define -- on non-Adreno devices the macros expand to
// nothing and the kernel reduces to v2.
//
// Algorithm itself is byte-for-byte identical to v1/v2 so any timing
// delta is purely the QCOM hints + constant-cache effect.

// Phase A bisection: the first build with both QCOM hints
// (sub_group_uniform + qcom_max_concurrent_subgroups) failed at
// runtime registerClKernel. We don't yet know which attribute the
// driver rejected. As a clean baseline, this revision REMOVES both
// QCOM hints so v3 reduces to "v2 with a different name" -- pure
// __constant input. Once we have the build log (now surfaced to
// stderr as [CL_BUILD_FAIL] / [CL_BUILD_LOG]), the next iteration
// re-adds the hints one at a time guided by the actual diagnostic.
//
// NOTE: keeping the v3 kernel name + dispatcher so the env-gate
// NNTRAINER_GEMV_ADRENO_V3=1 path still routes through this file --
// the toggle just measures __constant alone now.

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
