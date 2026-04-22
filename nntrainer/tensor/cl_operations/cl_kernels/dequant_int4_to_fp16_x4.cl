// GPU kernel: dequantize int4 channel-wise weights to fp16 in delegate
// layout. 4x variant: each thread covers (z, it, out_slice_pair) where
// out_slice_pair packs TWO consecutive out_slices (base & base+1).
//
// Per-thread work:
//   2 x ushort8 load       (8 ushorts / row × 2 rows = 64 nibbles)
//   1 x half8   scale load (8 halves = 2 out_slices × 4 out_ch)
//   64 x nibble dequant
//   4 x vstore16 writes    (2 src_slices × 2 out_slices)
//
// Total threads : N * K / 64  (quarter of v1, half of x2).
// Output layout : identical to v1,
//                 out[((z*iters + it) * 16 + s) * 16 + c * 4 + j].
// Requires      : N % 32 == 0, K % 8 == 0 (already guaranteed).

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void dequant_int4_to_delegate_fp16_x4(
    __global const ushort *restrict packed_weights, // [(K/4) * N] int4 packed
    __global const half *restrict scales,           // [N] per-channel
    __global half *restrict out,                    // delegate layout
    const int N, const int K) {

  const int iters = (K / 4) / 2; // K / 8

  const int tid = get_global_id(0);
  // tid = ((z * iters + it) * 4) + out_slice_pair
  const int pair = tid & 3;
  const int block = tid >> 2;
  const int it = block % iters;
  const int z = block / iters;

  const int out_slice_base = pair << 1; // 0, 2, 4, 6
  const int base_out_ch = z * 32 + out_slice_base * 4;

  // 8 consecutive scales (2 out_slices × 4 out_ch).
  const half8 sc8 = vload8(0, scales + base_out_ch);
  const half4 sc_a = sc8.lo; // scales for out_slice_base
  const half4 sc_b = sc8.hi; // scales for out_slice_base + 1
  const half16 sc16_a = (half16)(sc_a, sc_a, sc_a, sc_a);
  const half16 sc16_b = (half16)(sc_b, sc_b, sc_b, sc_b);

  // 8 ushorts per row × 2 rows (src_slice 0 and 1).
  const ushort8 w0 =
    vload8(0, packed_weights + (2 * it + 0) * N + base_out_ch);
  const ushort8 w1 =
    vload8(0, packed_weights + (2 * it + 1) * N + base_out_ch);
  const ushort4 w0a = w0.lo, w0b = w0.hi;
  const ushort4 w1a = w1.lo, w1b = w1.hi;

  #define DQ4(W, shift) \
    (half4)((half)((int)(((W).x >> (shift)) & 0xF) - 8), \
            (half)((int)(((W).y >> (shift)) & 0xF) - 8), \
            (half)((int)(((W).z >> (shift)) & 0xF) - 8), \
            (half)((int)(((W).w >> (shift)) & 0xF) - 8))

  const half16 r0a =
    (half16)(DQ4(w0a, 0), DQ4(w0a, 4), DQ4(w0a, 8), DQ4(w0a, 12));
  const half16 r0b =
    (half16)(DQ4(w0b, 0), DQ4(w0b, 4), DQ4(w0b, 8), DQ4(w0b, 12));
  const half16 r1a =
    (half16)(DQ4(w1a, 0), DQ4(w1a, 4), DQ4(w1a, 8), DQ4(w1a, 12));
  const half16 r1b =
    (half16)(DQ4(w1b, 0), DQ4(w1b, 4), DQ4(w1b, 8), DQ4(w1b, 12));
  #undef DQ4

  const int block_base = (z * iters + it) * 16;
  // src_slice 0 rows (s = out_slice_base, out_slice_base + 1).
  vstore16(r0a * sc16_a, 0,
           out + (block_base + out_slice_base + 0) * 16);
  vstore16(r0b * sc16_b, 0,
           out + (block_base + out_slice_base + 1) * 16);
  // src_slice 1 rows (s = out_slice_base + 8, +9).
  vstore16(r1a * sc16_a, 0,
           out + (block_base + out_slice_base + 8) * 16);
  vstore16(r1b * sc16_b, 0,
           out + (block_base + out_slice_base + 9) * 16);
}
