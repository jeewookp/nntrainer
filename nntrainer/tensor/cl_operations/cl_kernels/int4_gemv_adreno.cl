
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// Channel-wise int4 GEMV kernel for Adreno 830 (M = 1 path).
//
// Memory layout (must match Int4Utils::convertKaiToChannelwise -- the
// same layout consumed by gpu_int4_gemm_adreno):
//
//   weights : ushort[(K/4) * N]
//             Each ushort packs 4 unsigned bias-8 nibbles for one channel
//             at K positions [k, k+1, k+2, k+3]:
//               bits  [0..3]  -> k
//               bits  [4..7]  -> k+1
//               bits  [8..11] -> k+2
//               bits [12..15] -> k+3
//             dequantized weight = ((nibble) - 8) * scale.
//
//   scales  : half[align(N, 32)]
//             one fp16 scale per output channel (per-channel quantization).
//
//   input   : half[K]
//             single-token activation (M = 1).
//
//   output  : half[N]
//             y[n] = scale[n] * sum_k ((nibble[k,n] - 8) * input[k])
//
// Decomposition (V2 — Adreno 830 wavefront-matched):
//
//   * WG size = 64 = one native Adreno 830 wavefront (prior version
//     used WG = 16, one-quarter of a wavefront, wasting occupancy).
//   * Each work-item produces 1 output channel.  The prior 4-channels/WI
//     packing saved on the per-(k/4) ushort4 vload vs. four scalar ushorts,
//     but we now read weights as scalar ushort per WI.  Adreno coalesces
//     the 64 adjacent-n scalar reads across the wavefront into one or two
//     cache lines, so arithmetic intensity is unchanged.
//   * Cooperative input load: every WG loads the full K-length
//     activation row into __local once, and all 64 WIs MAC against the
//     shared copy.  Previously every WI fetched the whole K-vector
//     independently, paying N/(WG*channels_per_wi) redundant reads —
//     e.g. 40 redundant reads for N=2560, WG=16, channels=4.  The
//     cooperative load makes it 1 read per WG -> N/64 reads total.  For
//     Qwen3-4B FC widths (K <= 9728) the __local buffer fits in 19.5 KB
//     per WG, well under the per-WG local-mem limit.
//
// Dispatch from the host:
//   global = align(N, 64)       (each WI handles 1 channel)
//   local  = {64, 1, 1}
//
// All Qwen3-4B FC N values (1024, 2560, 4096, 9728) are already
// multiples of 64, so align_N == N in practice; the defensive `n < N`
// guard inside the kernel covers any future shape that isn't aligned.

#define WG    64
#define MAX_K 9728   // largest K across Qwen3-4B FC layers (down_proj)

__attribute__((reqd_work_group_size(WG, 1, 1)))
kernel void
gpu_int4_gemv_adreno(__global const half *input,
                     __global const half *scales,
                     __global half *output,
                     __global const ushort *weights,
                     const int K,
                     const int N) {
  const int n   = get_global_id(0);
  const int lid = get_local_id(0);
  const bool active = (n < N);

  // Cooperative input load.  Every WI in the WG participates
  // unconditionally (including the inactive lanes n >= N) so the
  // barrier below doesn't deadlock.  Stride by WG so adjacent lanes
  // fetch adjacent halves, giving the hardware a coalesced access.
  __local half input_shared[MAX_K];
  for (int k = lid; k < K; k += WG) {
    input_shared[k] = input[k];
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  // From here on, inactive lanes have nothing to do -- but OpenCL
  // doesn't allow returning before a barrier we already crossed, so
  // we simply skip the MAC loop and the store below.
  if (!active) return;

  float acc = 0.0f;

  // Each WI scans the full K dim for its single channel n.  Weight
  // at position (k/4, n) packs 4 nibbles for k..k+3 into one ushort.
  for (int k = 0; k < K; k += 4) {
    const ushort packed = weights[(k >> 2) * N + n];
    acc += (float)input_shared[k + 0]
           * (float)((int)((packed >> 0) & 0xF) - 8);
    acc += (float)input_shared[k + 1]
           * (float)((int)((packed >> 4) & 0xF) - 8);
    acc += (float)input_shared[k + 2]
           * (float)((int)((packed >> 8) & 0xF) - 8);
    acc += (float)input_shared[k + 3]
           * (float)((int)((packed >> 12) & 0xF) - 8);
  }

  output[n] = (half)(acc * (float)scales[n]);
}
