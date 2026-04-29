
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// Channel-wise int4 GEMV kernel for Adreno (M = 1) -- v2.
//
// Identical to int4_gemv_adreno.cl except `input` is bound through
// __constant memory.  Inspiration is LiteRT's program_002.cl
// (matmul_micro_benchmark int8_per_tensor capture) which uses
//   __constant half8* xmem_buffer __attribute__((max_constant_size((6144))))
// to pull the activation vector through Adreno's per-CU constant cache
// (~32 KB) instead of the L1 data cache that __global goes through.
// For decode M = 1 the activation is the entire hot input read by every
// work-item, so promoting it to __constant lets a constant cache hit
// service all 64 lanes from a single broadcast instead of independent
// L1 traffic per lane.
//
// All other parameters (weights, scales, output) and the inner-loop
// math are byte-for-byte identical to int4_gemv_adreno.cl, so any
// timing delta is attributable solely to the input-side memory path.
//
// Constant-memory budget on Adreno is device-dependent (Adreno 830
// advertises CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE = 64 KB).  Worst-case
// Qwen3-4B input is the down_proj K = 9728 fp16 = 19456 bytes, well
// inside that budget.  No `max_constant_size` attribute is set so the
// compiler just enforces the device max.
kernel void
gpu_int4_gemv_adreno_v2(__constant half *input,
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
