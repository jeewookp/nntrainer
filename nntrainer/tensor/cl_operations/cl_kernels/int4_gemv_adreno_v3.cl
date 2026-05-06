
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

// __constant input alone (Phase A.1 measurement) gave +1.1% TPS
// vs the __global v1 baseline. Layering the LiteRT-style QCOM hints
// from the captured program_002.cl on top:
//
//   - cl_qcom_subgroup_uniform_load extension + sub_group_uniform
//     attribute on the input pointer. For M=1 GEMV every lane in a
//     subgroup reads input[k] from the SAME address (loop k uniform
//     across lanes, only n differs); marking the pointer lets the
//     driver issue ONE subgroup-broadcast load instead of N
//     per-lane loads.
//   - qcom_max_concurrent_subgroups(12) attribute on the kernel
//     (LiteRT used this exact value in program_002.cl). Occupancy
//     hint for small decode dispatches.
//
// Both attributes are gated on the cl_qcom_subgroup_uniform_load
// extension #ifdef so non-Adreno builds reduce to v2-equivalent
// (__constant input, no hints).
//
// If the driver rejects either attribute the build log surfaces
// via [CL_BUILD_FAIL] / [CL_BUILD_LOG] in stderr and we adjust the
// syntax based on the actual diagnostic.

#ifdef cl_qcom_subgroup_uniform_load
#pragma OPENCL EXTENSION cl_qcom_subgroup_uniform_load : enable
#define ADRENO_SG_UNIFORM __attribute__((sub_group_uniform))
#define ADRENO_MAX_CONCURRENT __attribute__((qcom_max_concurrent_subgroups(12)))
#else
#define ADRENO_SG_UNIFORM
#define ADRENO_MAX_CONCURRENT
#endif

ADRENO_MAX_CONCURRENT
kernel void
gpu_int4_gemv_adreno_v3(__constant half *input ADRENO_SG_UNIFORM,
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
