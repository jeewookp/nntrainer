// Fused SwiGLU + int4 GEMV (down projection) for M=1 decode -- v2.
//
// Difference from v1:
//   v1 had each WI compute silu(gate[k])*up[k] independently, leading
//   to redundant silu (64x for a 64-lane WG) AND inflated per-WI
//   register/code-size pressure that dropped Adreno occupancy.
//   Result was 2.0 ms per call vs ~0.7 ms for separate swiglu+gemv.
//
// v2 uses __local memory tiling:
//   - Each WG cooperatively computes silu(gate)*up for a TILE_SIZE
//     chunk of K, storing the result in __local tile[].
//   - Each WI in the WG handles TILE_SIZE/WG_SIZE elements of the
//     silu/up combine.
//   - After a barrier, all 64 lanes consume tile[] to MAC against
//     their N-partition's weight nibbles -- standard int4_gemv inner
//     loop unchanged, just reading from __local instead of __global.
//
// Layout matches int4_gemv_adreno.cl:
//   weights : ushort[(K/4) * N], 4 nibbles per ushort, channel-wise
//   scales  : half[align(N, 32)]
//   gate, up: half[K], single-token activations (M=1)
//   output  : half[N]
//
// Each WI produces 4 N output channels.
//
// Dispatch:
//   global = align(N, 256) / 4
//   local  = (64, 1, 1)

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define WG_SIZE 64
#define TILE_SIZE 64       // K elements processed per cooperative chunk
                            // Equals WG_SIZE so each lane does 1 silu/chunk.

__attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
kernel void
gpu_swiglu_down_int4_v2(__global const half *gate,
                         __global const half *up,
                         __global const half *scales,
                         __global half *output,
                         __global const ushort *weights,
                         const int K,
                         const int N) {
  const int n   = get_global_id(0) * 4;
  const int lid = get_local_id(0);

  __local half tile[TILE_SIZE];

  float acc0 = 0.0f;
  float acc1 = 0.0f;
  float acc2 = 0.0f;
  float acc3 = 0.0f;

  for (int k_chunk = 0; k_chunk < K; k_chunk += TILE_SIZE) {
    // Cooperative silu(gate)*up: each lane handles one element.
    const int k_idx = k_chunk + lid;
    if (k_idx < K) {
      const float g = (float)gate[k_idx];
      const float u = (float)up[k_idx];
      tile[lid] = (half)((g / (1.0f + native_exp(-g))) * u);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // All 64 lanes now read the chunk's silu*up from __local and
    // MAC against weight nibbles.  Standard int4_gemv inner loop,
    // chunked by 4 K elements (1 ushort packs 4 nibbles).
    if (n < N) {
      for (int t = 0; t < TILE_SIZE; t += 4) {
        const int k = k_chunk + t;
        if (k >= K) break;
        const half4 in_v = vload4(0, (const __local half *)&tile[t]);
        const ushort4 packed = vload4(0, weights + (k / 4) * N + n);

        // Lane k+0 (low 4 bits)
        const float in0 = (float)in_v.s0;
        acc0 += in0 * (float)((int)(packed.s0 & 0x000F) - 8);
        acc1 += in0 * (float)((int)(packed.s1 & 0x000F) - 8);
        acc2 += in0 * (float)((int)(packed.s2 & 0x000F) - 8);
        acc3 += in0 * (float)((int)(packed.s3 & 0x000F) - 8);

        const float in1 = (float)in_v.s1;
        acc0 += in1 * (float)((int)((packed.s0 & 0x00F0) >> 4) - 8);
        acc1 += in1 * (float)((int)((packed.s1 & 0x00F0) >> 4) - 8);
        acc2 += in1 * (float)((int)((packed.s2 & 0x00F0) >> 4) - 8);
        acc3 += in1 * (float)((int)((packed.s3 & 0x00F0) >> 4) - 8);

        const float in2 = (float)in_v.s2;
        acc0 += in2 * (float)((int)((packed.s0 & 0x0F00) >> 8) - 8);
        acc1 += in2 * (float)((int)((packed.s1 & 0x0F00) >> 8) - 8);
        acc2 += in2 * (float)((int)((packed.s2 & 0x0F00) >> 8) - 8);
        acc3 += in2 * (float)((int)((packed.s3 & 0x0F00) >> 8) - 8);

        const float in3 = (float)in_v.s3;
        acc0 += in3 * (float)((int)((packed.s0 & 0xF000) >> 12) - 8);
        acc1 += in3 * (float)((int)((packed.s1 & 0xF000) >> 12) - 8);
        acc2 += in3 * (float)((int)((packed.s2 & 0xF000) >> 12) - 8);
        acc3 += in3 * (float)((int)((packed.s3 & 0xF000) >> 12) - 8);
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (n >= N) return;
  const half4 scale = vload4(0, scales + n);
  output[n + 0] = (half)(acc0 * (float)scale.s0);
  output[n + 1] = (half)(acc1 * (float)scale.s1);
  output[n + 2] = (half)(acc2 * (float)scale.s2);
  output[n + 3] = (half)(acc3 * (float)scale.s3);
}
