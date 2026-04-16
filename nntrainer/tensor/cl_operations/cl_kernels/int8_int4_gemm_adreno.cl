#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable

// int8 activation x int4 weight GEMM for Adreno.
//
// Complements `gpu_int4_gemm_adreno` (fp16 activation path) -- the
// difference is the inner MAC: this kernel does signed int8 * int8 -> int32
// accumulation which Adreno 7xx+ maps to the DP4A (ssad) instruction,
// reaching roughly 3.6x the fp16 FMA throughput on hardware with the
// `cl_qcom_dot_product8` extension.
//
// Activation quant is done on the host (CPU for now; a GPU
// `gpu_activation_quantize_int8` pass can replace it later). The wrapper
// passes:
//   x_q      : [align(M, 8)][K] signed char, row-major. Rows [M, M_pad)
//              are zero-padded so vload4 is always in-bounds.
//   x_scale  : fp16 [M], per-row max|x|/127 rounded through fp16.
//   weights  : ushort[(K/4) * N], exactly the same channel-wise int4
//              layout that `gpu_int4_gemm_adreno` reads. Each ushort packs
//              4 nibbles (k, k+1, k+2, k+3) for one output channel with
//              bias-8 unsigned encoding.
//   w_scale  : fp16 [align(N, 4)], per-channel max|w[n,:]|/7 rounded
//              through fp16 (channel-wise only).
//   output   : fp16 [M][N]
//
// Tile: each WI computes out[m:m+8, n:n+4] = 32 outputs, matching the
// `gpu_int4_gemm_adreno` tile for apples-to-apples comparison.
//
// Dispatch: global = (align(M, 8) / 8, N / 4, 1), local = (1, 128, 1).
// Divisibility: K % 4 == 0 (4 nibbles per ushort), N % 4 == 0 (WI writes
// 4 channels).

#define CEIL_DIV(a, b) (((a) + (b)-1) / (b))
#define ALIGN(a, b) (CEIL_DIV(a, b) * (b))

// 4-way int8 dot-product + accumulate.
//
// Written as explicit char-to-int widening multiply-adds so the Adreno
// OpenCL compiler can fuse the pattern to the `ssad` (signed saturating
// accumulating dot-product) hardware op. Clang/LLVM pattern matches the
// `acc + mad(a, b, mad(a, b, ...))` chain for cl_qcom_dot_product8.
inline int dot4_i8(char4 a, char4 b, int acc) {
  return acc + (int)a.s0 * (int)b.s0 + (int)a.s1 * (int)b.s1 +
         (int)a.s2 * (int)b.s2 + (int)a.s3 * (int)b.s3;
}

// Decode a 16-bit packed-int4 value (4 nibbles with bias-8 encoding) into
// a signed char4 in the range [-8, 7].
inline char4 unpack_int4x4(ushort p) {
  return (char4)((char)(((int)(p & 0x000Fu)) - 8),
                 (char)(((int)((p >> 4) & 0x000Fu)) - 8),
                 (char)(((int)((p >> 8) & 0x000Fu)) - 8),
                 (char)(((int)((p >> 12) & 0x000Fu)) - 8));
}

__attribute__((qcom_reqd_sub_group_size("full"))) kernel void
gpu_int8_int4_gemm_adreno(__global const char *x_q,
                          __global const half *x_scale,
                          __global const half *w_scale,
                          __global const ushort *weights,
                          __global half *output, const int K, const int N,
                          const int M) {
  const int m = get_global_id(0) * 8;
  const int n = get_global_id(1) * 4;

  // 32 int32 accumulators, organized as [column n+j].[row m+i]:
  //   cj.si = acc for out[m+i, n+j]
  int8 c0 = (int8)(0);
  int8 c1 = (int8)(0);
  int8 c2 = (int8)(0);
  int8 c3 = (int8)(0);

  // K is a multiple of 4 so no residue-loop is needed.
  for (int k = 0; k < K; k += 4) {
    // Load 4 k-lane char values for each of the 8 rows.
    char4 x0 = vload4(0, x_q + ((long)(m + 0) * K + k));
    char4 x1 = vload4(0, x_q + ((long)(m + 1) * K + k));
    char4 x2 = vload4(0, x_q + ((long)(m + 2) * K + k));
    char4 x3 = vload4(0, x_q + ((long)(m + 3) * K + k));
    char4 x4 = vload4(0, x_q + ((long)(m + 4) * K + k));
    char4 x5 = vload4(0, x_q + ((long)(m + 5) * K + k));
    char4 x6 = vload4(0, x_q + ((long)(m + 6) * K + k));
    char4 x7 = vload4(0, x_q + ((long)(m + 7) * K + k));

    // Load packed int4 weights for 4 channels (k..k+3 per channel).
    ushort4 pw = vload4(0, weights + (k / 4) * N + n);

    // Unpack to char4 (4 k-lanes for each of the 4 channels).
    char4 w0 = unpack_int4x4(pw.s0);
    char4 w1 = unpack_int4x4(pw.s1);
    char4 w2 = unpack_int4x4(pw.s2);
    char4 w3 = unpack_int4x4(pw.s3);

    // 32 DP4As: one per (row, column) pair.
    c0.s0 = dot4_i8(x0, w0, c0.s0);
    c0.s1 = dot4_i8(x1, w0, c0.s1);
    c0.s2 = dot4_i8(x2, w0, c0.s2);
    c0.s3 = dot4_i8(x3, w0, c0.s3);
    c0.s4 = dot4_i8(x4, w0, c0.s4);
    c0.s5 = dot4_i8(x5, w0, c0.s5);
    c0.s6 = dot4_i8(x6, w0, c0.s6);
    c0.s7 = dot4_i8(x7, w0, c0.s7);

    c1.s0 = dot4_i8(x0, w1, c1.s0);
    c1.s1 = dot4_i8(x1, w1, c1.s1);
    c1.s2 = dot4_i8(x2, w1, c1.s2);
    c1.s3 = dot4_i8(x3, w1, c1.s3);
    c1.s4 = dot4_i8(x4, w1, c1.s4);
    c1.s5 = dot4_i8(x5, w1, c1.s5);
    c1.s6 = dot4_i8(x6, w1, c1.s6);
    c1.s7 = dot4_i8(x7, w1, c1.s7);

    c2.s0 = dot4_i8(x0, w2, c2.s0);
    c2.s1 = dot4_i8(x1, w2, c2.s1);
    c2.s2 = dot4_i8(x2, w2, c2.s2);
    c2.s3 = dot4_i8(x3, w2, c2.s3);
    c2.s4 = dot4_i8(x4, w2, c2.s4);
    c2.s5 = dot4_i8(x5, w2, c2.s5);
    c2.s6 = dot4_i8(x6, w2, c2.s6);
    c2.s7 = dot4_i8(x7, w2, c2.s7);

    c3.s0 = dot4_i8(x0, w3, c3.s0);
    c3.s1 = dot4_i8(x1, w3, c3.s1);
    c3.s2 = dot4_i8(x2, w3, c3.s2);
    c3.s3 = dot4_i8(x3, w3, c3.s3);
    c3.s4 = dot4_i8(x4, w3, c3.s4);
    c3.s5 = dot4_i8(x5, w3, c3.s5);
    c3.s6 = dot4_i8(x6, w3, c3.s6);
    c3.s7 = dot4_i8(x7, w3, c3.s7);
  }

  // Dequant: out[m+i, n+j] = (half)(float(c_j[i]) * x_scale[m+i] * w_scale[n+j]).
  //
  // Stay in float until the final (half) cast because int32 accumulators
  // can exceed half's 65504 max (e.g. K=1024 with |acc| up to ~1e6).
  // The 4 w_scales are shared across all 8 rows so we promote them once.
  float w_s0 = vload_half(n + 0, w_scale);
  float w_s1 = vload_half(n + 1, w_scale);
  float w_s2 = vload_half(n + 2, w_scale);
  float w_s3 = vload_half(n + 3, w_scale);

  // Write each row, bounds-checked in case M is not a multiple of 8.
  // The input is zero-padded so accumulators for m+r >= M are valid but
  // unused.
#define WRITE_ROW(R)                                                           \
  if ((m + (R)) < M) {                                                         \
    float xs = vload_half(m + (R), x_scale);                                   \
    half4 out4 = (half4)((half)((float)c0.s##R * xs * w_s0),                   \
                         (half)((float)c1.s##R * xs * w_s1),                   \
                         (half)((float)c2.s##R * xs * w_s2),                   \
                         (half)((float)c3.s##R * xs * w_s3));                  \
    vstore4(out4, 0, output + (m + (R)) * N + n);                              \
  }

  WRITE_ROW(0)
  WRITE_ROW(1)
  WRITE_ROW(2)
  WRITE_ROW(3)
  WRITE_ROW(4)
  WRITE_ROW(5)
  WRITE_ROW(6)
  WRITE_ROW(7)

#undef WRITE_ROW
}
