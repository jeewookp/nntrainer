// GPU kernel: dequantize int4 channel-wise weights to fp16 in delegate
// layout.  2x variant of dequant_int4_to_delegate_fp16: each thread now
// owns (z, it, out_slice) covering BOTH src_slices (s = out_slice and
// s = out_slice + 8) of a (z, it) block.
//
// Per-thread work:
//   2 x ushort4 load       (16 + 16 nibbles  =  32 weights)
//   1 x half4   scale load (same scale feeds both src_slices)
//   32 x nibble dequant    (shift, mask, sub 8, mul scale)
//   8 x half4  store       (4 rows * 2 src_slices)
//
// Total threads : (N / 32) * (K / 8) * 8  =  N * K / 32  (half of v1).
// Output layout : same blocked form as v1,
//                 out[((z*iters + it) * 16 + s) * 16 + c * 4 + j].
// Requires      : N % 32 == 0, K % 8 == 0 (already guaranteed by the
//                 delegate conv call sites).

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void dequant_int4_to_delegate_fp16_x2(
    __global const ushort *restrict packed_weights, // [(K/4) * N] int4 packed
    __global const half *restrict scales,           // [N] per-channel
    __global half *restrict out,                    // delegate layout
    const int N, const int K) {

  const int iters = (K / 4) / 2; // K / 8

  const int tid = get_global_id(0);
  // Packing: tid = ((z * iters) + it) * 8 + out_slice
  const int out_slice = tid & 7;
  const int block = tid >> 3;
  const int it = block % iters;
  const int z = block / iters;

  const int base_out_ch = z * 32 + out_slice * 4;

  // Single scale load reused for both src_slices.
  const half4 sc = vload4(0, scales + base_out_ch);

  // Two weight loads, one per src_slice row in packed[(K/4) * N].
  const ushort4 w0 =
    vload4(0, packed_weights + (2 * it + 0) * N + base_out_ch);
  const ushort4 w1 =
    vload4(0, packed_weights + (2 * it + 1) * N + base_out_ch);

  // Output positions: s=out_slice (src_slice=0), s=out_slice+8 (src_slice=1).
  const int block_base = (z * iters + it) * 16;
  __global half4 *out4_0 =
    (__global half4 *)(out + (block_base + out_slice) * 16);
  __global half4 *out4_1 =
    (__global half4 *)(out + (block_base + out_slice + 8) * 16);

  #pragma unroll
  for (int c = 0; c < 4; ++c) {
    const int shift = c << 2;
    half4 qv0, qv1;
    qv0.x = (half)((int)((w0.x >> shift) & 0xF) - 8);
    qv0.y = (half)((int)((w0.y >> shift) & 0xF) - 8);
    qv0.z = (half)((int)((w0.z >> shift) & 0xF) - 8);
    qv0.w = (half)((int)((w0.w >> shift) & 0xF) - 8);
    out4_0[c] = qv0 * sc;

    qv1.x = (half)((int)((w1.x >> shift) & 0xF) - 8);
    qv1.y = (half)((int)((w1.y >> shift) & 0xF) - 8);
    qv1.z = (half)((int)((w1.z >> shift) & 0xF) - 8);
    qv1.w = (half)((int)((w1.w >> shift) & 0xF) - 8);
    out4_1[c] = qv1 * sc;
  }
}
