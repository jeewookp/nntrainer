// GPU kernel: dequantize int4 channel-wise weights to fp16 in delegate layout.
// Input:  packed int4 ushort at weights[(k/4)*N + n], 4 nibbles per ushort
//         scales[n] as fp16 (per-channel)
// Output: fp16 half in delegate's blocked layout:
//         out[(Z*iters + it)*256 + s*16 + c*4 + j]
//         where Z=out_ch/32, s=slice(0..15), c=input_ch_in_slice, j=out_ch%4

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void dequant_int4_to_delegate_fp16(
    __global const ushort* restrict packed_weights,  // [(K/4)*N] int4 packed
    __global const half* restrict scales,             // [N] per-channel
    __global half* restrict out,                      // delegate layout
    const int N,
    const int K) {

  const int src_slices = K / 4;
  const int dst_slices = N / 4;
  const int n_z = (dst_slices + 7) / 8;
  const int iters = src_slices / 2;

  // Each work item handles one (Z, iteration, s, c, j) = one output half
  int gid = get_global_id(0);
  int total = n_z * iters * 256;
  if (gid >= total) return;

  int blk_idx = gid / 256;
  int within = gid % 256;
  int z = blk_idx / iters;
  int it = blk_idx % iters;
  int s = within / 16;
  int rem = within % 16;
  int c = rem / 4;
  int j = rem % 4;

  int base_n = z * 32;
  int src_slice_idx = it * 2 + (s < 8 ? 0 : 1);
  int out_slice = s % 8;
  int in_ch = src_slice_idx * 4 + c;
  int out_ch = base_n + out_slice * 4 + j;

  if (out_ch >= N || in_ch >= K) {
    out[gid] = (half)0.0h;
    return;
  }

  int packed_row = in_ch / 4;
  int nibble_pos = in_ch % 4;
  ushort packed = packed_weights[packed_row * N + out_ch];
  int nibble = (packed >> (nibble_pos * 4)) & 0xF;
  int q = nibble - 8;

  half scale = scales[out_ch];
  out[gid] = (half)q * scale;
}
