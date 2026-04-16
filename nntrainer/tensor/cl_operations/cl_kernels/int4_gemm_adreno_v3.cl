#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable

// Phase 3c step 2: K-axis WG-level split + __local reduction.
//
// v1 / v2 have a single WI reduce the entire K dimension by itself
// (1024 MACs sequential for K=1024). v3 splits that reduction across
// K_SPLIT=4 WIs in the same work-group: each computes the partial sum
// for its quarter of K, writes the tile into __local, then the ly==0
// leader sums the 4 quarters and emits the output tile. This mirrors
// the pattern LiteRT's FullyConnected kernel uses for Adreno and it is
// the missing lever that texture-only (v2) could not provide -- it
// quadruples the number of concurrent memory streams for the same
// output element, exposing more latency hiding.
//
// Layout:
//   local  = (32, K_SPLIT=4, 1)  = 128 WIs/WG (2 Adreno 64-lane waves)
//     local_x (0..31)  : n-tile-within-WG (one 4-col tile per local_x)
//     local_y (0..3)   : K-split index (which quarter of K this WI owns)
//   global = (N/4, K_SPLIT, ceilDiv(M, 8))
//   Output tile per WI : 8 rows x 4 cols.
//
// Weight/scale/input layout and contract are identical to v2.
//
// Shape assumptions: K % (K_SPLIT * 4) == 0 so the inner loop has no
// residue, N % 4 == 0.

#define CEIL_DIV(a, b) (((a) + (b)-1) / (b))
#define ALIGN(a, b) (CEIL_DIV(a, b) * (b))

#define K_SPLIT 4
#define WG_X 32

__attribute__((qcom_reqd_sub_group_size("full"))) kernel void
gpu_int4_gemm_adreno_v3(__read_only image1d_buffer_t input,
                        __global const half *scales, __global half *output,
                        __read_only image1d_buffer_t weights, const int K,
                        const int N, const int M,
                        const int quantization_group_size) {
  const int align_N = ALIGN(N, 32);

  const int lx = get_local_id(0);
  const int ly = get_local_id(1);

  // v1-compatible m in units of 4-row-groups: a WI covers rows
  // [m*4, m*4 + 8). global[2] = ceilDiv(M, 8) -> m = global_id(2)*2.
  const int m = get_global_id(2) * 2;
  const int n = get_global_id(0) * 4;
  const int M_4 = CEIL_DIV(M, 4);
  const int N_4 = N >> 2;

  // K-range this WI owns. K_SPLIT must divide K (asserted in the
  // wrapper) so k_end - k_start is a positive multiple of 4.
  const int K_per_split = K / K_SPLIT;
  const int k_start = ly * K_per_split;
  const int k_end = k_start + K_per_split;

  half8 c0 = 0, c1 = 0, c2 = 0, c3 = 0;
  half8 input_reg;
  half4 dq_weights_reg;
  uint4 packed_w;
  half4 scale;

  for (int k = k_start; k < k_end; k += 4) {
    if ((k & 0x1F) == 0) {
      scale = vload4(0, scales + (k / quantization_group_size) * align_N + n);
    }
    packed_w = read_imageui(weights, (k >> 2) * N_4 + (n >> 2));

    input_reg.s0123 = read_imageh(input, k * M_4 + m);
    input_reg.s4567 = read_imageh(input, k * M_4 + m + 1);

    dq_weights_reg.s0 = (half)((int)(packed_w.s0 & 0x000Fu) - 8) * scale.s0;
    dq_weights_reg.s1 = (half)((int)(packed_w.s1 & 0x000Fu) - 8) * scale.s1;
    dq_weights_reg.s2 = (half)((int)(packed_w.s2 & 0x000Fu) - 8) * scale.s2;
    dq_weights_reg.s3 = (half)((int)(packed_w.s3 & 0x000Fu) - 8) * scale.s3;

    c0 += input_reg * dq_weights_reg.s0;
    c1 += input_reg * dq_weights_reg.s1;
    c2 += input_reg * dq_weights_reg.s2;
    c3 += input_reg * dq_weights_reg.s3;

    input_reg.s0123 = read_imageh(input, (k + 1) * M_4 + m);
    input_reg.s4567 = read_imageh(input, (k + 1) * M_4 + m + 1);

    dq_weights_reg.s0 = (half)((int)((packed_w.s0 >> 4) & 0x000Fu) - 8) * scale.s0;
    dq_weights_reg.s1 = (half)((int)((packed_w.s1 >> 4) & 0x000Fu) - 8) * scale.s1;
    dq_weights_reg.s2 = (half)((int)((packed_w.s2 >> 4) & 0x000Fu) - 8) * scale.s2;
    dq_weights_reg.s3 = (half)((int)((packed_w.s3 >> 4) & 0x000Fu) - 8) * scale.s3;

    c0 += input_reg * dq_weights_reg.s0;
    c1 += input_reg * dq_weights_reg.s1;
    c2 += input_reg * dq_weights_reg.s2;
    c3 += input_reg * dq_weights_reg.s3;

    input_reg.s0123 = read_imageh(input, (k + 2) * M_4 + m);
    input_reg.s4567 = read_imageh(input, (k + 2) * M_4 + m + 1);

    dq_weights_reg.s0 = (half)((int)((packed_w.s0 >> 8) & 0x000Fu) - 8) * scale.s0;
    dq_weights_reg.s1 = (half)((int)((packed_w.s1 >> 8) & 0x000Fu) - 8) * scale.s1;
    dq_weights_reg.s2 = (half)((int)((packed_w.s2 >> 8) & 0x000Fu) - 8) * scale.s2;
    dq_weights_reg.s3 = (half)((int)((packed_w.s3 >> 8) & 0x000Fu) - 8) * scale.s3;

    c0 += input_reg * dq_weights_reg.s0;
    c1 += input_reg * dq_weights_reg.s1;
    c2 += input_reg * dq_weights_reg.s2;
    c3 += input_reg * dq_weights_reg.s3;

    input_reg.s0123 = read_imageh(input, (k + 3) * M_4 + m);
    input_reg.s4567 = read_imageh(input, (k + 3) * M_4 + m + 1);

    dq_weights_reg.s0 = (half)((int)((packed_w.s0 >> 12) & 0x000Fu) - 8) * scale.s0;
    dq_weights_reg.s1 = (half)((int)((packed_w.s1 >> 12) & 0x000Fu) - 8) * scale.s1;
    dq_weights_reg.s2 = (half)((int)((packed_w.s2 >> 12) & 0x000Fu) - 8) * scale.s2;
    dq_weights_reg.s3 = (half)((int)((packed_w.s3 >> 12) & 0x000Fu) - 8) * scale.s3;

    c0 += input_reg * dq_weights_reg.s0;
    c1 += input_reg * dq_weights_reg.s1;
    c2 += input_reg * dq_weights_reg.s2;
    c3 += input_reg * dq_weights_reg.s3;
  }

  // K-split reduction via __local.
  //
  // Layout: tmp[n-tile-in-WG][K-split][col_of_4] = half8 (8 rows).
  // 32 * 4 * 4 * 16 B = 8 KB per WG -- well under Adreno 830's typical
  // 32 KB __local.
  __local half8 tmp[WG_X][K_SPLIT][4];

  tmp[lx][ly][0] = c0;
  tmp[lx][ly][1] = c1;
  tmp[lx][ly][2] = c2;
  tmp[lx][ly][3] = c3;
  barrier(CLK_LOCAL_MEM_FENCE);

  // Only the ly==0 lane sums + writes. The 3 other lanes exit. This is
  // the same pattern LiteRT's FullyConnected uses; the remaining 3x
  // lanes of a WG could be used for prefetch of the next m-tile in a
  // follow-up optimization, but keeping them idle here is simpler and
  // the big win is the 4x inner-loop concurrency, not using those
  // lanes in the epilogue.
  if (ly != 0) {
    return;
  }

  c0 = tmp[lx][0][0] + tmp[lx][1][0] + tmp[lx][2][0] + tmp[lx][3][0];
  c1 = tmp[lx][0][1] + tmp[lx][1][1] + tmp[lx][2][1] + tmp[lx][3][1];
  c2 = tmp[lx][0][2] + tmp[lx][1][2] + tmp[lx][2][2] + tmp[lx][3][2];
  c3 = tmp[lx][0][3] + tmp[lx][1][3] + tmp[lx][2][3] + tmp[lx][3][3];

  // Same output write pattern as v1/v2.
  int idx = (m << 2) * N + n;

  if (idx + 3 < M * N) {
    vstore4((half4)(c0.s0, c1.s0, c2.s0, c3.s0), 0, output + idx);
    idx += N;
  }
  if (idx + 3 < M * N) {
    vstore4((half4)(c0.s1, c1.s1, c2.s1, c3.s1), 0, output + idx);
    idx += N;
  }
  if (idx + 3 < M * N) {
    vstore4((half4)(c0.s2, c1.s2, c2.s2, c3.s2), 0, output + idx);
    idx += N;
  }
  if (idx + 3 < M * N) {
    vstore4((half4)(c0.s3, c1.s3, c2.s3, c3.s3), 0, output + idx);
    idx += N;
  }
  if (idx + 3 < M * N) {
    vstore4((half4)(c0.s4, c1.s4, c2.s4, c3.s4), 0, output + idx);
    idx += N;
  }
  if (idx + 3 < M * N) {
    vstore4((half4)(c0.s5, c1.s5, c2.s5, c3.s5), 0, output + idx);
    idx += N;
  }
  if (idx + 3 < M * N) {
    vstore4((half4)(c0.s6, c1.s6, c2.s6, c3.s6), 0, output + idx);
    idx += N;
  }
  if (idx + 3 < M * N) {
    vstore4((half4)(c0.s7, c1.s7, c2.s7, c3.s7), 0, output + idx);
  }
}
