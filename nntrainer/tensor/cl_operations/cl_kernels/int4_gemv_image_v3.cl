// Step 3 / incremental: + dequant + MAC, no scale yet.
//
// Step 1 (weight load):   gpu_kernel = 176 ms (22 us/call)
// Step 2 (+ input load):  gpu_kernel = 176 ms (unchanged, +0.3)
// Baseline full gemv:     gpu_kernel = 3436 ms (430 us/call)
//
// The ~408 us/call gap from memory-only to full is therefore in
// the dequant + MAC + scale + final output pipeline.  Step 3
// builds the real dequant + MAC: each (k, n) pair reads the
// packed nibble, subtracts the bias-8, multiplies by the input
// lane, and accumulates to a float.  No per-channel scale yet.
//
// Output sink: write (half)acc directly so the compiler MUST
// retain the MAC chain (acc feeds a global store).  Value is the
// un-scaled running sum; downstream decode tokens are garbage as
// before.  The compiler cannot DCE through a global store so
// gpu_kernel here reflects real compute.
//
// Dispatch / layout identical to step 1 + 2 (WG=64, 1 channel/WI).

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__attribute__((reqd_work_group_size(64, 1, 1)))
kernel void
gpu_int4_gemv_image_v3(__global const half *input,
                       __global const half *scales,
                       __global half *output,
                       __global const ushort *weights,
                       const int K,
                       const int N) {
  (void)scales;

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

  // Global store forces the compiler to retain the acc chain.
  // No scale applied -- step 4 adds it and restores correctness.
  output[n] = (half)acc;
}
