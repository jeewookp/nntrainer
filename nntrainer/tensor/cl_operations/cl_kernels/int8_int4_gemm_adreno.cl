#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable

// int8 activation x int4 weight GEMM for Adreno.
//
// Complements `gpu_int4_gemm_adreno` (fp16 activation path) -- the
// difference is the inner MAC: this kernel does signed int8 * int8 -> int32
// accumulation which Adreno 7xx+ maps to the DP4A (ssad) instruction via
// `cl_khr_integer_dot_product::dot_acc_sat`.
//
// Activation is pre-quantized to int8 on the host and exposed to the GPU
// as `image1d_buffer_t` with CL_RGBA / CL_SIGNED_INT8 format (1 texel =
// char4). Using the texture path matches the fp16 kernel's memory
// topology so we get the texture-cache + texture-fetch-unit parallelism
// that regular `__global const char *` loads do not.
//
// The wrapper passes:
//   x_q      : image1d_buffer_t, [align(M, 8)][K/4] char4 texels,
//              row-major (texel at (row=m, k-group=g) -> index
//              m*(K/4) + g). Rows [M, align(M, 8)) are zero-padded.
//   x_scale  : fp16 [M], per-row max|x|/127 rounded through fp16.
//   weights  : ushort[(K/4) * N], exactly the same channel-wise int4
//              layout that `gpu_int4_gemm_adreno` reads. Each ushort packs
//              4 nibbles (k, k+1, k+2, k+3) for one output channel with
//              bias-8 unsigned encoding.
//   w_scale  : fp16 [align(N, 4)], per-channel max|w[n,:]|/7 rounded
//              through fp16 (channel-wise only).
//   output   : fp16 [M][N]
//
// Tile: each WI computes out[m:m+8, n:n+8] = 64 outputs (Phase 3a: tile
// widened from 8x4 to 8x8 to halve the memory:compute ratio. The 8
// activation texture reads are now shared across 8 output channels
// instead of 4, so inner-loop throughput doubles without extra x_q
// bandwidth.)
//
// Dispatch: global = (align(M, 8) / 8, N / 8, 1), local = (1, 128, 1).
// Divisibility: K % 4 == 0 (4 nibbles per ushort), N % 8 == 0 (WI writes
// 8 channels). All test shapes already satisfy N % 32 == 0 (from
// Int4Utils::channelwise_layout_size) so N % 8 is automatic.

#define CEIL_DIV(a, b) (((a) + (b)-1) / (b))
#define ALIGN(a, b) (CEIL_DIV(a, b) * (b))

// 4-way int8 dot-product + accumulate.
//
// First attempt used explicit scalar `(int)a.s0*b.s0 + ...` -- the Adreno
// compiler did NOT fuse that to the ssad instruction; the unfused kernel
// ran 2.5x SLOWER than the fp16 path because it emitted 128 scalar
// imul/iadd ops per WI per iter vs the fp16 path's 4 half8 FMAs.
//
// We now use the standardized `cl_khr_integer_dot_product` extension's
// `dot_acc_sat(char4, char4, int)` intrinsic, which maps directly to the
// ssad hardware op on Adreno 7xx+ (and on ARM Mali via the same
// standardized entry point). If the device does not advertise the
// extension we fall back to a vector-widening form which sometimes
// pattern-matches on Adreno even when the extension is absent, but that
// is a correctness path, not a perf path -- treat a "no extension"
// device as "DP4A path should not be used here".
#if defined(cl_khr_integer_dot_product)
#pragma OPENCL EXTENSION cl_khr_integer_dot_product : enable
#define DP4A_HAVE_INTRINSIC 1
#else
#define DP4A_HAVE_INTRINSIC 0
#endif

inline int dot4_i8(char4 a, char4 b, int acc) {
#if DP4A_HAVE_INTRINSIC
  // Hardware ssad (signed saturating accumulating dot product).
  return dot_acc_sat(a, b, acc);
#else
  // Vector fallback. Relies on the compiler to recognize the pattern.
  int4 p = convert_int4(a) * convert_int4(b);
  return acc + p.s0 + p.s1 + p.s2 + p.s3;
#endif
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
gpu_int8_int4_gemm_adreno(__read_only image1d_buffer_t x_q,
                          __global const half *x_scale,
                          __global const half *w_scale,
                          __global const ushort *weights,
                          __global half *output, const int K, const int N,
                          const int M) {
  const int m = get_global_id(0) * 8;
  const int n = get_global_id(1) * 8;
  // Row stride in texels: K chars / 4 chars-per-texel.
  const int K_4 = K >> 2;

  // 64 int32 accumulators, organized as [column n+j].[row m+i]:
  //   cj.si = acc for out[m+i, n+j]
  int8 c0 = (int8)(0);
  int8 c1 = (int8)(0);
  int8 c2 = (int8)(0);
  int8 c3 = (int8)(0);
  int8 c4 = (int8)(0);
  int8 c5 = (int8)(0);
  int8 c6 = (int8)(0);
  int8 c7 = (int8)(0);

  // K is a multiple of 4 so no residue-loop is needed.
  for (int k = 0; k < K; k += 4) {
    // One texture fetch per row returns 4 sign-extended int8 lanes as
    // an int4; truncate to char4 for ssad.
    const int k_4 = k >> 2;
    char4 x0 = convert_char4(read_imagei(x_q, (m + 0) * K_4 + k_4));
    char4 x1 = convert_char4(read_imagei(x_q, (m + 1) * K_4 + k_4));
    char4 x2 = convert_char4(read_imagei(x_q, (m + 2) * K_4 + k_4));
    char4 x3 = convert_char4(read_imagei(x_q, (m + 3) * K_4 + k_4));
    char4 x4 = convert_char4(read_imagei(x_q, (m + 4) * K_4 + k_4));
    char4 x5 = convert_char4(read_imagei(x_q, (m + 5) * K_4 + k_4));
    char4 x6 = convert_char4(read_imagei(x_q, (m + 6) * K_4 + k_4));
    char4 x7 = convert_char4(read_imagei(x_q, (m + 7) * K_4 + k_4));

    // Load packed int4 weights for 8 channels (k..k+3 per channel). One
    // 16-byte vload8 replaces the 8x4-tile's vload4; alignment is
    // guaranteed because n is a multiple of 8 and (k/4)*N is a multiple
    // of 32 for every test shape.
    ushort8 pw = vload8(0, weights + (k / 4) * N + n);

    // Unpack to char4 (4 k-lanes for each of the 8 channels).
    char4 w0 = unpack_int4x4(pw.s0);
    char4 w1 = unpack_int4x4(pw.s1);
    char4 w2 = unpack_int4x4(pw.s2);
    char4 w3 = unpack_int4x4(pw.s3);
    char4 w4 = unpack_int4x4(pw.s4);
    char4 w5 = unpack_int4x4(pw.s5);
    char4 w6 = unpack_int4x4(pw.s6);
    char4 w7 = unpack_int4x4(pw.s7);

    // 64 DP4As: one per (row, column) pair.
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

    c4.s0 = dot4_i8(x0, w4, c4.s0);
    c4.s1 = dot4_i8(x1, w4, c4.s1);
    c4.s2 = dot4_i8(x2, w4, c4.s2);
    c4.s3 = dot4_i8(x3, w4, c4.s3);
    c4.s4 = dot4_i8(x4, w4, c4.s4);
    c4.s5 = dot4_i8(x5, w4, c4.s5);
    c4.s6 = dot4_i8(x6, w4, c4.s6);
    c4.s7 = dot4_i8(x7, w4, c4.s7);

    c5.s0 = dot4_i8(x0, w5, c5.s0);
    c5.s1 = dot4_i8(x1, w5, c5.s1);
    c5.s2 = dot4_i8(x2, w5, c5.s2);
    c5.s3 = dot4_i8(x3, w5, c5.s3);
    c5.s4 = dot4_i8(x4, w5, c5.s4);
    c5.s5 = dot4_i8(x5, w5, c5.s5);
    c5.s6 = dot4_i8(x6, w5, c5.s6);
    c5.s7 = dot4_i8(x7, w5, c5.s7);

    c6.s0 = dot4_i8(x0, w6, c6.s0);
    c6.s1 = dot4_i8(x1, w6, c6.s1);
    c6.s2 = dot4_i8(x2, w6, c6.s2);
    c6.s3 = dot4_i8(x3, w6, c6.s3);
    c6.s4 = dot4_i8(x4, w6, c6.s4);
    c6.s5 = dot4_i8(x5, w6, c6.s5);
    c6.s6 = dot4_i8(x6, w6, c6.s6);
    c6.s7 = dot4_i8(x7, w6, c6.s7);

    c7.s0 = dot4_i8(x0, w7, c7.s0);
    c7.s1 = dot4_i8(x1, w7, c7.s1);
    c7.s2 = dot4_i8(x2, w7, c7.s2);
    c7.s3 = dot4_i8(x3, w7, c7.s3);
    c7.s4 = dot4_i8(x4, w7, c7.s4);
    c7.s5 = dot4_i8(x5, w7, c7.s5);
    c7.s6 = dot4_i8(x6, w7, c7.s6);
    c7.s7 = dot4_i8(x7, w7, c7.s7);
  }

  // Dequant: out[m+i, n+j] = (half)(float(c_j[i]) * x_scale[m+i] * w_scale[n+j]).
  //
  // Stay in float until the final (half) cast because int32 accumulators
  // can exceed half's 65504 max (e.g. K=1024 with |acc| up to ~1e6).
  // The 8 w_scales are shared across all 8 rows so we promote them once.
  float w_s0 = vload_half(n + 0, w_scale);
  float w_s1 = vload_half(n + 1, w_scale);
  float w_s2 = vload_half(n + 2, w_scale);
  float w_s3 = vload_half(n + 3, w_scale);
  float w_s4 = vload_half(n + 4, w_scale);
  float w_s5 = vload_half(n + 5, w_scale);
  float w_s6 = vload_half(n + 6, w_scale);
  float w_s7 = vload_half(n + 7, w_scale);

  // Write each row, bounds-checked in case M is not a multiple of 8.
  // The input is zero-padded so accumulators for m+r >= M are valid but
  // unused.
#define WRITE_ROW(R)                                                           \
  if ((m + (R)) < M) {                                                         \
    float xs = vload_half(m + (R), x_scale);                                   \
    half8 out8 = (half8)((half)((float)c0.s##R * xs * w_s0),                   \
                         (half)((float)c1.s##R * xs * w_s1),                   \
                         (half)((float)c2.s##R * xs * w_s2),                   \
                         (half)((float)c3.s##R * xs * w_s3),                   \
                         (half)((float)c4.s##R * xs * w_s4),                   \
                         (half)((float)c5.s##R * xs * w_s5),                   \
                         (half)((float)c6.s##R * xs * w_s6),                   \
                         (half)((float)c7.s##R * xs * w_s7));                  \
    vstore8(out8, 0, output + (m + (R)) * N + n);                              \
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
