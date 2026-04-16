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
// Tile: each WI computes out[m:m+8, n:n+4] = 32 outputs. Phase 3a tried
// widening to 8x8 (64 outputs) but the ratio improvement did not convert
// to a speedup on Adreno 830 -- the DP4A kernel was NOT memory-bound at
// the x_q texture fetch, so giving weight/activation reads more reuse
// yielded ~0%. Phase 3b instead keeps the small 8x4 tile (keeps register
// pressure low, leaves headroom for higher wave occupancy) but unrolls
// the K loop 4x (k += 16) so four independent nibble-blocks are in
// flight per iteration. The expectation is better ssad issue rate via
// ILP, since the compiler can interleave four independent ssad chains
// and hide the single-cycle dependency between consecutive ssads on the
// same accumulator.
//
// Dispatch: global = (align(M, 8) / 8, N / 4, 1), local = (1, 128, 1).
// Divisibility: K % 16 == 0 (so the K+=16 unroll has no residue loop),
// N % 4 == 0 (WI writes 4 channels).

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
  const int n = get_global_id(1) * 4;
  // Row stride in texels: K chars / 4 chars-per-texel.
  const int K_4 = K >> 2;

  // 32 int32 accumulators, organized as [column n+j].[row m+i]:
  //   cj.si = acc for out[m+i, n+j]
  int8 c0 = (int8)(0);
  int8 c1 = (int8)(0);
  int8 c2 = (int8)(0);
  int8 c3 = (int8)(0);

  // K-unroll factor = 4 blocks of 4 k-values each => k += 16 per iter.
  // DO_K_BLOCK(B) runs block B in {0,1,2,3} of the current outer step:
  //   k-range processed = [k + 4*B, k + 4*B + 4)
  //   k-group offset    = k/4 + B
  // Declarations are scoped in a do-while(0) block so each block's
  // temporaries (x0..x7, pw, w0..w3) don't collide across blocks -- this
  // makes it easy on the register allocator: it can reuse the same phys
  // regs across blocks while the 4 accumulator chains (c0..c3) stay
  // live across the whole outer iter. ILP comes from the 32 ssads per
  // block being independent of the *next* block's ssads (different k
  // lanes, same accumulator) so the scheduler can interleave four
  // independent dependency chains.
#define DO_K_BLOCK(B)                                                          \
  do {                                                                         \
    const int k_off = (k >> 2) + (B);                                          \
    char4 x0 = convert_char4(read_imagei(x_q, (m + 0) * K_4 + k_off));         \
    char4 x1 = convert_char4(read_imagei(x_q, (m + 1) * K_4 + k_off));         \
    char4 x2 = convert_char4(read_imagei(x_q, (m + 2) * K_4 + k_off));         \
    char4 x3 = convert_char4(read_imagei(x_q, (m + 3) * K_4 + k_off));         \
    char4 x4 = convert_char4(read_imagei(x_q, (m + 4) * K_4 + k_off));         \
    char4 x5 = convert_char4(read_imagei(x_q, (m + 5) * K_4 + k_off));         \
    char4 x6 = convert_char4(read_imagei(x_q, (m + 6) * K_4 + k_off));         \
    char4 x7 = convert_char4(read_imagei(x_q, (m + 7) * K_4 + k_off));         \
    ushort4 pw = vload4(0, weights + k_off * N + n);                           \
    char4 w0 = unpack_int4x4(pw.s0);                                           \
    char4 w1 = unpack_int4x4(pw.s1);                                           \
    char4 w2 = unpack_int4x4(pw.s2);                                           \
    char4 w3 = unpack_int4x4(pw.s3);                                           \
    c0.s0 = dot4_i8(x0, w0, c0.s0);                                            \
    c0.s1 = dot4_i8(x1, w0, c0.s1);                                            \
    c0.s2 = dot4_i8(x2, w0, c0.s2);                                            \
    c0.s3 = dot4_i8(x3, w0, c0.s3);                                            \
    c0.s4 = dot4_i8(x4, w0, c0.s4);                                            \
    c0.s5 = dot4_i8(x5, w0, c0.s5);                                            \
    c0.s6 = dot4_i8(x6, w0, c0.s6);                                            \
    c0.s7 = dot4_i8(x7, w0, c0.s7);                                            \
    c1.s0 = dot4_i8(x0, w1, c1.s0);                                            \
    c1.s1 = dot4_i8(x1, w1, c1.s1);                                            \
    c1.s2 = dot4_i8(x2, w1, c1.s2);                                            \
    c1.s3 = dot4_i8(x3, w1, c1.s3);                                            \
    c1.s4 = dot4_i8(x4, w1, c1.s4);                                            \
    c1.s5 = dot4_i8(x5, w1, c1.s5);                                            \
    c1.s6 = dot4_i8(x6, w1, c1.s6);                                            \
    c1.s7 = dot4_i8(x7, w1, c1.s7);                                            \
    c2.s0 = dot4_i8(x0, w2, c2.s0);                                            \
    c2.s1 = dot4_i8(x1, w2, c2.s1);                                            \
    c2.s2 = dot4_i8(x2, w2, c2.s2);                                            \
    c2.s3 = dot4_i8(x3, w2, c2.s3);                                            \
    c2.s4 = dot4_i8(x4, w2, c2.s4);                                            \
    c2.s5 = dot4_i8(x5, w2, c2.s5);                                            \
    c2.s6 = dot4_i8(x6, w2, c2.s6);                                            \
    c2.s7 = dot4_i8(x7, w2, c2.s7);                                            \
    c3.s0 = dot4_i8(x0, w3, c3.s0);                                            \
    c3.s1 = dot4_i8(x1, w3, c3.s1);                                            \
    c3.s2 = dot4_i8(x2, w3, c3.s2);                                            \
    c3.s3 = dot4_i8(x3, w3, c3.s3);                                            \
    c3.s4 = dot4_i8(x4, w3, c3.s4);                                            \
    c3.s5 = dot4_i8(x5, w3, c3.s5);                                            \
    c3.s6 = dot4_i8(x6, w3, c3.s6);                                            \
    c3.s7 = dot4_i8(x7, w3, c3.s7);                                            \
  } while (0)

  // K must be a multiple of 16 to avoid a residue loop. All current test
  // shapes (K in {32,64,128,256,512,1024}) satisfy this.
  for (int k = 0; k < K; k += 16) {
    DO_K_BLOCK(0);
    DO_K_BLOCK(1);
    DO_K_BLOCK(2);
    DO_K_BLOCK(3);
  }

#undef DO_K_BLOCK

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
