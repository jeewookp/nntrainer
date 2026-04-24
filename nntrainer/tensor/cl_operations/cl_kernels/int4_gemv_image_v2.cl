// Step 2 / incremental: weight + input load, no MAC.
//
// Step 1 (weight-only) measured:
//   gpu_kernel = 176 ms / 8064 calls = 22 us/call
// vs baseline gemv_int4_adreno's 430 us/call.  So the weight load
// itself is only a 5% slice of the baseline kernel's work -- the
// remaining 408 us/call lives in the dequant + MAC + scale pipeline
// we haven't built yet.
//
// Step 2 adds the INPUT load to sink, still no MAC.  Measures
// (weight load + input load) bandwidth.  If this stays close to
// step 1 (~25-30 us/call), input is cache-resident (K halves fit
// in L1/L2).  If it jumps, the input feed is also expensive and
// we'll need to cooperatively load into __local.
//
// Dispatch: WG=(64,1,1), global=align(N,64).  Same as step 1 so
// the delta vs step 1 is purely attributable to the input-load
// addition.

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__attribute__((reqd_work_group_size(64, 1, 1)))
kernel void
gpu_int4_gemv_image_v2(__global const half *input,
                       __global const half *scales,
                       __global half *output,
                       __global const ushort *weights,
                       const int K,
                       const int N) {
  (void)scales;

  const int n = get_global_id(0);
  if (n >= N) return;

  // Same scalar weight load as step 1 (isolates the additional cost
  // coming from the input-load loop that follows).
  ushort w_sink = 0;
  for (int k = 0; k < K; k += 4) {
    const ushort packed = weights[(k >> 2) * N + n];
    w_sink ^= packed;
  }

  // New in step 2: scalar input load across the K dim.  Every WI
  // does K scalar reads of input -- 64 WIs per WG hit the same
  // input[k] at the same cycle, which Adreno's L1 should coalesce
  // into a single cache-line fetch shared across the wavefront.
  // If it doesn't, this loop will dominate the kernel time.
  uint i_sink = 0;
  for (int k = 0; k < K; ++k) {
    // Read half, reinterpret to ushort, XOR to prevent DCE.
    const half in_h = input[k];
    i_sink ^= as_ushort(in_h);
  }

  // Combine sinks so the compiler must issue BOTH loops.  Mask to
  // 0 so we don't perturb decoding correctness of downstream
  // layers more than necessary.
  const ushort combined = w_sink ^ (ushort)i_sink;
  output[n] = (half)((float)(combined & 0));
}
