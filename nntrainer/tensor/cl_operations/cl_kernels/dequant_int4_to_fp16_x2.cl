// GPU kernel: dequantize int4 channel-wise weights to fp16 in delegate
// layout.  2x variant of dequant_int4_to_delegate_fp16: each thread now
// owns (z, it, out_slice) covering BOTH src_slices (s = out_slice and
// s = out_slice + 8) of a (z, it) block.
//
// Per-thread work:
//   2 x ushort4 load       (16 + 16 nibbles  =  32 weights)
//   1 x half4   scale load (same scale feeds both src_slices)
//   32 x nibble dequant    (shift, mask, sub 8, mul scale)
//   2 x vstore16 writes    (one half16 per src_slice, down from 4x half4)
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

  // Single scale load reused for both src_slices, broadcast to half16.
  const half4 sc = vload4(0, scales + base_out_ch);
  const half16 sc16 = (half16)(sc, sc, sc, sc);

  // Two weight loads, one per src_slice row in packed[(K/4) * N].
  const ushort4 w0 =
    vload4(0, packed_weights + (2 * it + 0) * N + base_out_ch);
  const ushort4 w1 =
    vload4(0, packed_weights + (2 * it + 1) * N + base_out_ch);

  // Nibble decode for each c = 0..3. Produces one half4 per c per src_slice.
  // Construct full half16 rows via 4-half4 vector literal and store once.
  #define DQ4(W, shift) \
    (half4)((half)((int)(((W).x >> (shift)) & 0xF) - 8), \
            (half)((int)(((W).y >> (shift)) & 0xF) - 8), \
            (half)((int)(((W).z >> (shift)) & 0xF) - 8), \
            (half)((int)(((W).w >> (shift)) & 0xF) - 8))

  const half4 r0_c0 = DQ4(w0,  0);
  const half4 r0_c1 = DQ4(w0,  4);
  const half4 r0_c2 = DQ4(w0,  8);
  const half4 r0_c3 = DQ4(w0, 12);
  const half4 r1_c0 = DQ4(w1,  0);
  const half4 r1_c1 = DQ4(w1,  4);
  const half4 r1_c2 = DQ4(w1,  8);
  const half4 r1_c3 = DQ4(w1, 12);
  #undef DQ4

  const half16 row0 = (half16)(r0_c0, r0_c1, r0_c2, r0_c3);
  const half16 row1 = (half16)(r1_c0, r1_c1, r1_c2, r1_c3);

  // Output positions: s=out_slice (src_slice=0), s=out_slice+8 (src_slice=1).
  const int block_base = (z * iters + it) * 16;
  vstore16(row0 * sc16, 0, out + (block_base + out_slice)     * 16);
  vstore16(row1 * sc16, 0, out + (block_base + out_slice + 8) * 16);
}
