// GPU kernel: dequantize int4 channel-wise weights to fp16 in delegate layout.
// Input:  packed int4 ushort at weights[(k/4)*N + n], 4 nibbles per ushort
//         scales[n] as fp16 (per-channel)
// Output: fp16 half in delegate's blocked layout:
//         out[((z*iters + it)*16 + s)*16 + c*4 + j]
//         where Z=out_ch/32, s=slice(0..15), c=input_ch_in_slice, j=out_ch%4
//
// Each thread processes one (z, it, s) triple = 16 consecutive output halves.
// Compared to the per-output baseline this does 16x fewer dispatches, one
// ushort4 (8 B) weight load, one half4 (8 B) scale load, and four half4
// stores. Requires N %  4 == 0 (always true for delegate shapes where
// N % 32 == 0 is required by the conv kernel itself).

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void dequant_int4_to_delegate_fp16(
    __global const ushort* restrict packed_weights,  // [(K/4) * N] int4 packed
    __global const half*   restrict scales,           // [N] per-channel
    __global half*         restrict out,              // delegate layout
    const int N,
    const int K) {

  const int iters = (K / 4) / 2;

  const int tid = get_global_id(0);
  const int blk = tid >> 4;     // (z * iters + it)
  const int s   = tid & 15;     // 0..15

  const int it = blk % iters;
  const int z  = blk / iters;

  const int src_slice_idx = (it << 1) + (s >> 3);   // s<8 ? 0 : 1
  const int out_slice     = s & 7;
  const int base_out_ch   = z * 32 + out_slice * 4;
  const int packed_row    = src_slice_idx;

  // Load 4 adjacent scales as half4.
  const half4 sc = vload4(0, scales + base_out_ch);

  // Load 4 adjacent ushort weight-columns (one per j=0..3) as ushort4.
  const ushort4 w = vload4(0, packed_weights + packed_row * N + base_out_ch);

  // Produce 4 half4 rows (c = 0..3), each row spans j = 0..3 of the same
  // (z, it, s, c). out[(..) + c*4 + j] = (nibble_c_of_w[j] - 8) * sc[j].
  __global half4 *out4 = (__global half4 *)(out + tid * 16);

  #pragma unroll
  for (int c = 0; c < 4; ++c) {
    const int shift = c << 2;
    half4 qv;
    qv.x = (half)((int)((w.x >> shift) & 0xF) - 8);
    qv.y = (half)((int)((w.y >> shift) & 0xF) - 8);
    qv.z = (half)((int)((w.z >> shift) & 0xF) - 8);
    qv.w = (half)((int)((w.w >> shift) & 0xF) - 8);
    out4[c] = qv * sc;
  }
}
