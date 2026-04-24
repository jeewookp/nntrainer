// Step 1 / incremental: weight-load-only probe for the M=1 image2d
// gemv rewrite.
//
// Motivation: baseline gpu_int4_gemv_adreno has cl_event-measured
// gpu_kernel = 3.4 s / decode = 0.43 ms/call = ~7-8x above its
// theoretical memory-bandwidth floor (3.3 MB weights / ~60 GB/s =
// 55 us/call).  Something in the kernel's access pattern is
// leaving most of the memory-bandwidth budget on the table.
//
// The incremental plan mirrors the attention-kernel approach from
// earlier in this branch:
//   Step 1 (here): weight load only, no input / no MAC / no scale,
//                  write zero to output.  Measures pure weight
//                  read throughput.
//   Step 2: + input load, still no MAC (sink to avoid DCE).
//           Measures weight + input + register overhead.
//   Step 3: + dequant + MAC, per-channel accumulator.
//           Measures with real arithmetic intensity.
//   Step 4: + scale + proper output write (correctness restored).
//
// Dispatch / sink strategy: each WI owns one output channel n.
// It scans the full K dimension loading ushort packed weights.
// To keep the compiler from dead-code-eliminating the loads we
// XOR them into a sink scalar and write (half)(sink & 0) to
// output -- the store is live but the value is deterministic zero.
//
// Memory layout matches gpu_int4_gemv_adreno so the SAME weight
// buffer can be reused across step 1..4 without re-packing:
//   weights : __global const ushort* of shape (K/4) * N
//   scales  : __global const half*   of shape align(N, 32) (unused step 1)
//   input   : __global const half*   of shape K          (unused step 1)
//   output  : __global       half*   of shape N

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__attribute__((reqd_work_group_size(64, 1, 1)))
kernel void
gpu_int4_gemv_image_v1(__global const half *input,
                       __global const half *scales,
                       __global half *output,
                       __global const ushort *weights,
                       const int K,
                       const int N) {
  (void)input;
  (void)scales;

  const int n = get_global_id(0);
  if (n >= N) return;

  // Weight-load-only sink.  Vector load (ushort4) would be faster
  // but step 1 uses scalar to stay aligned with the baseline
  // kernel's memory access pattern so the gpu_kernel delta
  // attributable to step 1 additions is apples-to-apples with the
  // baseline.
  ushort sink = 0;
  for (int k = 0; k < K; k += 4) {
    const ushort packed = weights[(k >> 2) * N + n];
    sink ^= packed;
  }

  // `sink & 0` is always zero; keeping the & expression means the
  // compiler must compute `sink` (and hence run the load loop) to
  // evaluate it.
  output[n] = (half)((float)(sink & 0));
}
