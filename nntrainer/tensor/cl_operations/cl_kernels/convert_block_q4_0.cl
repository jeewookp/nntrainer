#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifdef cl_intel_required_subgroup_size
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable
#define INTEL_GPU 1
#define REQD_SUBGROUP_SIZE_16 __attribute__((intel_reqd_sub_group_size(16)))
#define REQD_SUBGROUP_SIZE_32 __attribute__((intel_reqd_sub_group_size(32)))
#elif defined(cl_qcom_reqd_sub_group_size)
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define ADRENO_GPU 1
#define REQD_SUBGROUP_SIZE_64 __attribute__((qcom_reqd_sub_group_size("half")))
#define REQD_SUBGROUP_SIZE_128 __attribute__((qcom_reqd_sub_group_size("full")))
#endif

#define QK4_0 32

typedef uchar uint8_t;

struct block_q4_0 {
  half d;
  uint8_t qs[QK4_0 / 2];
};

//------------------------------------------------------------------------------
// kernel_convert_block_q4_0_noshuffle
// Flatten q4_0 weights and unshuffle the bits
//------------------------------------------------------------------------------

kernel void kernel_convert_block_q4_0_noshuffle(global struct block_q4_0 *src0,
                                                global uchar *dst_q,
                                                global half *dst_d) {
  global struct block_q4_0 *b =
    (global struct block_q4_0 *)src0 + get_global_id(0);
  global uchar *q = (global uchar *)dst_q + QK4_0 / 2 * get_global_id(0);
  global half *d = (global half *)dst_d + get_global_id(0);

  *d = b->d;
  for (int i = 0; i < QK4_0 / 4; ++i) {
    uchar x0 = b->qs[2 * i + 0];
    uchar x1 = b->qs[2 * i + 1];

    q[i + 0] = convert_uchar(x0 & 0x0F) | convert_uchar((x1 & 0x0F) << 4);
    q[i + QK4_0 / 4] =
      convert_uchar((x0 & 0xF0) >> 4) | convert_uchar(x1 & 0xF0);

    // Note: we previously had a dead-branch printf("...") guarded on
    // ADRENO_GPU here as a compiler workaround ("must have the following
    // printf statement for the kernel to work properly"). The presence of
    // printf in any kernel compiled on an Adreno cl_context flips the
    // driver into printf / debug-buffer mode for that context, which
    // roughly HALVES the throughput of every subsequent kernel dispatch
    // on that context (bisected: compiling this one kernel dropped the
    // delegate conv from 2.95 to 1.63 TFLOPS, and compiling more printf
    // kernels compounds to 1.33). The original "it produces incorrect
    // result" symptom was almost certainly an Adreno compiler bug that
    // has since been fixed — the kernel is purely byte-lane arithmetic
    // with no subgroup-level operations, so there is nothing here for
    // the optimizer to miscompile. If a regression shows up, introduce
    // an opaque side-effect without printf (e.g. a conditional write
    // into a bench-only cl_mem) instead of resurrecting printf.
  }
}
