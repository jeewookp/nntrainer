// Step 4 / incremental: + scale + final output write (correctness).
//
// Step-by-step gpu_kernel measurements so far:
//   Step 1 (weight-only):          176 ms  (22 us/call)
//   Step 2 (+ input load):         177 ms
//   Step 3 (+ dequant + MAC):     2924 ms  (363 us/call)  <-- 85% of baseline
//   Baseline gemv_int4_adreno:    3436 ms  (430 us/call)
//
// Step 3 -> step 4 delta should be tiny (~60 ms): we only add a
// scalar scale read + one multiply + the real output store.  The
// point of step 4 is primarily to restore correctness so we can
// verify the kernel produces the right tokens before diving into
// MAC-loop vectorization in step 5+.
//
// Layout: same as baseline gemv_int4_adreno_cl except each WI
// produces 1 output channel instead of 4.  WG=64 = one Adreno 830
// wavefront.  All per-call state (memory layout, nibble packing)
// matches the existing kernel exactly.

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__attribute__((reqd_work_group_size(64, 1, 1)))
kernel void
gpu_int4_gemv_image_v4(__global const half *input,
                       __global const half *scales,
                       __global half *output,
                       __global const ushort *weights,
                       const int K,
                       const int N) {
  const int n = get_global_id(0);
  if (n >= N) return;

  float acc = 0.0f;
  for (int k = 0; k < K; k += 4) {
    const ushort packed = weights[(k >> 2) * N + n];
    acc += (float)input[k + 0]
           * (float)((int)((packed >>  0) & 0xF) - 8);
    acc += (float)input[k + 1]
           * (float)((int)((packed >>  4) & 0xF) - 8);
    acc += (float)input[k + 2]
           * (float)((int)((packed >>  8) & 0xF) - 8);
    acc += (float)input[k + 3]
           * (float)((int)((packed >> 12) & 0xF) - 8);
  }

  output[n] = (half)(acc * (float)scales[n]);
}
