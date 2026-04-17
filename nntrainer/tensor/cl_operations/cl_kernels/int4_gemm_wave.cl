#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// Int4 GEMM with __local weight tiling.
// Key insight: threads sharing the same Z (output channel group) read
// identical weights. With 128 M-threads per Z, tiling saves 128x reads.
//
// Workgroup: (WG_M, 1, WG_Z) where WG_M=32, WG_Z=4
// Each thread: 4 output channels per Z-slice, processes all K
// Total per thread: WG_Z not used - each thread does 1 Z slice × 4 ch

// Simpler approach: WG along M, each WG shares weights for one Z group.
// WG_M threads cooperatively load weights into __local, then all read locally.

#define WG_M 64

__constant sampler_t smp_zero = CLK_NORMALIZED_COORDS_FALSE |
                                 CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

__kernel __attribute__((reqd_work_group_size(WG_M, 1, 1)))
void gemm_int4_wave(
    __global const ushort* restrict weights,
    __global const ushort* restrict unused,
    __global const half* restrict scales,
    __write_only image2d_t dst_image,
    __read_only image2d_t src_image,
    const int M,
    const int N,
    const int K) {

  const int src_slices = K >> 2;
  const int dst_slices = N >> 2;

  const int lid = get_local_id(0);           // 0..WG_M-1
  const int X = get_group_id(0) * WG_M + lid;  // spatial position (M)
  const int Z = get_group_id(1);              // output slice group (8 slices = 32 ch)
  if (Z * 8 >= dst_slices) return;

  int base_n = Z * 32;

  // Accumulators: 8 output slices × 4 channels
  half4 r0 = (half4)(0.0h), r1 = (half4)(0.0h);
  half4 r2 = (half4)(0.0h), r3 = (half4)(0.0h);
  half4 r4 = (half4)(0.0h), r5 = (half4)(0.0h);
  half4 r6 = (half4)(0.0h), r7 = (half4)(0.0h);

  // Preload scales (same for all threads in this WG)
  half4 sc0 = vload4(0, scales + base_n);
  half4 sc1 = vload4(0, scales + base_n + 4);
  half4 sc2 = vload4(0, scales + base_n + 8);
  half4 sc3 = vload4(0, scales + base_n + 12);
  half4 sc4 = vload4(0, scales + base_n + 16);
  half4 sc5 = vload4(0, scales + base_n + 20);
  half4 sc6 = vload4(0, scales + base_n + 24);
  half4 sc7 = vload4(0, scales + base_n + 28);

  // __local tile for weights: 32 ushorts per packed row, 2 rows per iter = 64
  __local ushort w_tile[64];

#define DQ(V, S, SC) (((half)((int)((V >> S) & 0xFu) - 8)) * SC)

#define ACC4(R, SRC, OFF, SC) \
  R.x += SRC.x*DQ(w_tile[OFF],0,SC.x) + SRC.y*DQ(w_tile[OFF],4,SC.x) + SRC.z*DQ(w_tile[OFF],8,SC.x) + SRC.w*DQ(w_tile[OFF],12,SC.x); \
  R.y += SRC.x*DQ(w_tile[OFF+1],0,SC.y) + SRC.y*DQ(w_tile[OFF+1],4,SC.y) + SRC.z*DQ(w_tile[OFF+1],8,SC.y) + SRC.w*DQ(w_tile[OFF+1],12,SC.y); \
  R.z += SRC.x*DQ(w_tile[OFF+2],0,SC.z) + SRC.y*DQ(w_tile[OFF+2],4,SC.z) + SRC.z*DQ(w_tile[OFF+2],8,SC.z) + SRC.w*DQ(w_tile[OFF+2],12,SC.z); \
  R.w += SRC.x*DQ(w_tile[OFF+3],0,SC.w) + SRC.y*DQ(w_tile[OFF+3],4,SC.w) + SRC.z*DQ(w_tile[OFF+3],8,SC.w) + SRC.w*DQ(w_tile[OFF+3],12,SC.w);

  int coord_s = 0;
  do {
    // Cooperative weight load: WG_M threads load 64 ushorts
    // First 32 threads load row 0, next 32 load row 1
    if (lid < 32) {
      w_tile[lid] = weights[coord_s * N + base_n + lid];
    } else if (lid < 64) {
      w_tile[lid] = weights[(coord_s + 1) * N + base_n + (lid - 32)];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (X < M) {
      half4 src0 = read_imageh(src_image, smp_zero, (int2)(X, coord_s));
      half4 src1 = read_imageh(src_image, smp_zero, (int2)(X, coord_s + 1));

      // Row 0 (src0)
      ACC4(r0, src0, 0,  sc0)
      ACC4(r1, src0, 4,  sc1)
      ACC4(r2, src0, 8,  sc2)
      ACC4(r3, src0, 12, sc3)
      ACC4(r4, src0, 16, sc4)
      ACC4(r5, src0, 20, sc5)
      ACC4(r6, src0, 24, sc6)
      ACC4(r7, src0, 28, sc7)

      // Row 1 (src1)
      ACC4(r0, src1, 32, sc0)
      ACC4(r1, src1, 36, sc1)
      ACC4(r2, src1, 40, sc2)
      ACC4(r3, src1, 44, sc3)
      ACC4(r4, src1, 48, sc4)
      ACC4(r5, src1, 52, sc5)
      ACC4(r6, src1, 56, sc6)
      ACC4(r7, src1, 60, sc7)
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    coord_s += 2;
  } while (coord_s < src_slices);

#undef DQ
#undef ACC4

  if (X >= M) return;

  int out_s = Z * 8;
  if (out_s + 0 < dst_slices) write_imageh(dst_image, (int2)(X, out_s + 0), r0);
  if (out_s + 1 < dst_slices) write_imageh(dst_image, (int2)(X, out_s + 1), r1);
  if (out_s + 2 < dst_slices) write_imageh(dst_image, (int2)(X, out_s + 2), r2);
  if (out_s + 3 < dst_slices) write_imageh(dst_image, (int2)(X, out_s + 3), r3);
  if (out_s + 4 < dst_slices) write_imageh(dst_image, (int2)(X, out_s + 4), r4);
  if (out_s + 5 < dst_slices) write_imageh(dst_image, (int2)(X, out_s + 5), r5);
  if (out_s + 6 < dst_slices) write_imageh(dst_image, (int2)(X, out_s + 6), r6);
  if (out_s + 7 < dst_slices) write_imageh(dst_image, (int2)(X, out_s + 7), r7);
}
