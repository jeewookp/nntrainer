// SPDX-License-Identifier: Apache-2.0
// v8c GEMM kernel set: paper-aligned int8×int4 prefill compute (8/4/4).
//
// Weight format (in cl_mem buffer, viewed as image2d via image2d_from_buffer):
//   - Layout: row-major [N output channels][K/2 bytes per row]
//   - Encoding: int4 offset (value + 8) in low 4 bits of each byte
//   - Image2D view: CL_RGBA + CL_UNSIGNED_INT32, width=K/32, height=N,
//                    row_pitch=K/2 (16 bytes per texel = 32 int4 K-channels)
// Activation format:
//   - Layout: row-major [M rows][K bytes per row]
//   - Encoding: signed int8
//   - Image2D view: CL_RGBA + CL_UNSIGNED_INT32, width=K/16, height=M,
//                    row_pitch=K (16 bytes per texel = 16 int8 K-channels)
// Math: acc(i,j) = Σ_k act_ik × (enc_kj − 8)
//                = Σ_k dot_4x8packed_su_int(act_packed, enc_masked) − 8·row_sum_act_i
// row_sum_act_i = Σ_k act_ik, precomputed in the act-quant kernel.

#pragma OPENCL EXTENSION cl_khr_integer_dot_product : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__constant sampler_t SMP_v8c = CLK_NORMALIZED_COORDS_FALSE |
                               CLK_ADDRESS_CLAMP_TO_EDGE |
                               CLK_FILTER_NEAREST;

// ============================================================
// fp→int8 activation quantizer + row_sum, paper §3.7 separate kernel.
//   Input:  fp16 activations [M, K] (row-major in cl_mem buffer)
//   Outputs: int8 acts [M, K] (row-major), per-row fp32 scale, per-row int32 row_sum
// One work-item per row (M work-items total). K assumed multiple of 4.
// ============================================================
// Asymmetric int8 activation quantization, matching KAI qai8dxp_f32. Per-row
// scale = (rmax-rmin)/255 + zero_point. The symmetric (amax/127) variant
// rounded small values to 0 around outliers in skewed post-SwiGLU
// activations, which on a 28-layer Qwen3 stack pushed token logits far
// enough to flip selection. Asymmetric handles single-sided outliers
// gracefully.
//
// Outputs:
//   act_int8[M, K]            row-major signed int8 in [-128, 127]
//   scale_per_row[M]          recip-scale, i.e. (rmax-rmin)/255 — used at
//                             dequant: act_fp = (act_int - zp) * scale
//   zp_per_row[M]             nudged_zero_point (signed int)
//   row_sum_act[M]            Σ_k act_int8 (used for w-offset correction:
//                             - 8 * row_sum_act in the GEMM)
__kernel void v8c_act_quant_f16(
    __global const half *act_fp16,           // [M, K] fp16
    __global       char *act_int8,           // [M, K] int8
    __global       float *scale_per_row,     // [M] fp32 = (rmax-rmin)/255
    __global       int   *zp_per_row,        // [M] int   = nudged_zero_point
    __global       int   *row_sum_act,       // [M] int   = Σ_k act_int8
    const int M, const int K) {
  int i = get_global_id(0);
  if (i >= M) return;
  // pass 1: min/max
  float fmin = 0.0f, fmax = 0.0f;
  for (int k = 0; k < K; k++) {
    float v = (float)act_fp16[(long)i * K + k];
    if (v < fmin) fmin = v;
    if (v > fmax) fmax = v;
  }
  const float rmin = fmin < 0.0f ? fmin : 0.0f;
  const float rmax = fmax > 0.0f ? fmax : 0.0f;
  const float qmin = -128.0f, qmax = 127.0f;
  const float range = rmax - rmin;
  float scale_q = range > 0.0f ? 255.0f / range : 1.0f;       // act_fp → int
  float recip = range > 0.0f ? range / 255.0f : 1.0f;          // int → act_fp
  // Pick the zero point with the smaller of the two rounding errors at the
  // saturation bounds (matches kai_lhs_quant_pack_qai8dxp_f32 lines 137-147).
  float dmin = rmin * scale_q, dmax = rmax * scale_q;
  float zp_lo = qmin - dmin, zp_hi = qmax - dmax;
  float zp_f = (qmin + dmin) + (qmax + dmax) > 0.0f ? zp_lo : zp_hi;
  if (zp_f < qmin) zp_f = qmin;
  if (zp_f > qmax) zp_f = qmax;
  const int zp = (int)rint(zp_f);
  scale_per_row[i] = recip;
  zp_per_row[i] = zp;
  // pass 2: quantize + accumulate row_sum (int8 value, NOT (value - zp))
  int rs = 0;
  for (int k = 0; k < K; k++) {
    int q = (int)rint((float)act_fp16[(long)i * K + k] * scale_q) + zp;
    if (q < -128) q = -128;
    if (q > 127) q = 127;
    act_int8[(long)i * K + k] = (char)q;
    rs += q;
  }
  row_sum_act[i] = rs;
}

__kernel void v8c_act_quant_f32(
    __global const float *act_fp32,
    __global       char  *act_int8,
    __global       float *scale_per_row,
    __global       int   *zp_per_row,
    __global       int   *row_sum_act,
    const int M, const int K) {
  int i = get_global_id(0);
  if (i >= M) return;
  float fmin = 0.0f, fmax = 0.0f;
  for (int k = 0; k < K; k++) {
    float v = act_fp32[(long)i * K + k];
    if (v < fmin) fmin = v;
    if (v > fmax) fmax = v;
  }
  const float rmin = fmin < 0.0f ? fmin : 0.0f;
  const float rmax = fmax > 0.0f ? fmax : 0.0f;
  const float qmin = -128.0f, qmax = 127.0f;
  const float range = rmax - rmin;
  float scale_q = range > 0.0f ? 255.0f / range : 1.0f;
  float recip = range > 0.0f ? range / 255.0f : 1.0f;
  float dmin = rmin * scale_q, dmax = rmax * scale_q;
  float zp_lo = qmin - dmin, zp_hi = qmax - dmax;
  float zp_f = (qmin + dmin) + (qmax + dmax) > 0.0f ? zp_lo : zp_hi;
  if (zp_f < qmin) zp_f = qmin;
  if (zp_f > qmax) zp_f = qmax;
  const int zp = (int)rint(zp_f);
  scale_per_row[i] = recip;
  zp_per_row[i] = zp;
  int rs = 0;
  for (int k = 0; k < K; k++) {
    int q = (int)rint(act_fp32[(long)i * K + k] * scale_q) + zp;
    if (q < -128) q = -128;
    if (q > 127) q = 127;
    act_int8[(long)i * K + k] = (char)q;
    rs += q;
  }
  row_sum_act[i] = rs;
}

// ============================================================
// Parallel-reduction variant of v8c_act_quant_f32. LWS=64 work-items
// collaborate on a single row's K=1024 channels. Replaces the 1-WI-per-row
// scalar pass that dominated 71% of prefill cost (cf. NNTR_V8C_PROFILE
// breakdown on Adreno 830). Math is byte-identical to v8c_act_quant_f32:
// same KAI qai8dxp_f32 asymmetric formula, just split across WIs and
// reduced through local memory.
//
// Layout: one workgroup of LWS=64 WIs per row. K is assumed a multiple
// of LWS (true for Qwen3 hidden dims 1024/2048/3072/etc.).
// ============================================================
#define V8C_QUANT_LWS 64

__attribute__((reqd_work_group_size(V8C_QUANT_LWS, 1, 1)))
__kernel void v8c_act_quant_f32_par(
    __global const float *act_fp32,
    __global       char  *act_int8,
    __global       float *scale_per_row,
    __global       int   *zp_per_row,
    __global       int   *row_sum_act,
    const int M, const int K) {
  const int row = get_group_id(0);
  if (row >= M) return;
  const int tid = get_local_id(0);
  __local float lmin[V8C_QUANT_LWS];
  __local float lmax[V8C_QUANT_LWS];
  __local int   lsum[V8C_QUANT_LWS];
  __local float l_scale_q;
  __local int   l_zp;

  // pass 1: per-WI partial min/max
  float pmin = 0.0f, pmax = 0.0f;
  for (int k = tid; k < K; k += V8C_QUANT_LWS) {
    float v = act_fp32[(long)row * K + k];
    pmin = fmin(pmin, v);
    pmax = fmax(pmax, v);
  }
  lmin[tid] = pmin;
  lmax[tid] = pmax;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = V8C_QUANT_LWS / 2; s > 0; s >>= 1) {
    if (tid < s) {
      lmin[tid] = fmin(lmin[tid], lmin[tid + s]);
      lmax[tid] = fmax(lmax[tid], lmax[tid + s]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (tid == 0) {
    const float fmn = lmin[0], fmx = lmax[0];
    const float rmin = fmn < 0.0f ? fmn : 0.0f;
    const float rmax = fmx > 0.0f ? fmx : 0.0f;
    const float qmin = -128.0f, qmax = 127.0f;
    const float range = rmax - rmin;
    const float scale_q = range > 0.0f ? 255.0f / range : 1.0f;
    const float recip = range > 0.0f ? range / 255.0f : 1.0f;
    const float dmin = rmin * scale_q, dmax = rmax * scale_q;
    const float zp_lo = qmin - dmin, zp_hi = qmax - dmax;
    float zp_f =
      (qmin + dmin) + (qmax + dmax) > 0.0f ? zp_lo : zp_hi;
    if (zp_f < qmin) zp_f = qmin;
    if (zp_f > qmax) zp_f = qmax;
    l_scale_q = scale_q;
    l_zp = (int)rint(zp_f);
    scale_per_row[row] = recip;
    zp_per_row[row] = l_zp;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  // pass 2: per-WI quantize + partial row_sum
  const float scale_q = l_scale_q;
  const int zp = l_zp;
  int psum = 0;
  for (int k = tid; k < K; k += V8C_QUANT_LWS) {
    int q = (int)rint(act_fp32[(long)row * K + k] * scale_q) + zp;
    if (q < -128) q = -128;
    if (q > 127) q = 127;
    act_int8[(long)row * K + k] = (char)q;
    psum += q;
  }
  lsum[tid] = psum;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = V8C_QUANT_LWS / 2; s > 0; s >>= 1) {
    if (tid < s) lsum[tid] += lsum[tid + s];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (tid == 0) row_sum_act[row] = lsum[0];
}

__attribute__((reqd_work_group_size(V8C_QUANT_LWS, 1, 1)))
__kernel void v8c_act_quant_f16_par(
    __global const half  *act_fp16,
    __global       char  *act_int8,
    __global       float *scale_per_row,
    __global       int   *zp_per_row,
    __global       int   *row_sum_act,
    const int M, const int K) {
  const int row = get_group_id(0);
  if (row >= M) return;
  const int tid = get_local_id(0);
  __local float lmin[V8C_QUANT_LWS];
  __local float lmax[V8C_QUANT_LWS];
  __local int   lsum[V8C_QUANT_LWS];
  __local float l_scale_q;
  __local int   l_zp;

  float pmin = 0.0f, pmax = 0.0f;
  for (int k = tid; k < K; k += V8C_QUANT_LWS) {
    float v = (float)act_fp16[(long)row * K + k];
    pmin = fmin(pmin, v);
    pmax = fmax(pmax, v);
  }
  lmin[tid] = pmin;
  lmax[tid] = pmax;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = V8C_QUANT_LWS / 2; s > 0; s >>= 1) {
    if (tid < s) {
      lmin[tid] = fmin(lmin[tid], lmin[tid + s]);
      lmax[tid] = fmax(lmax[tid], lmax[tid + s]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (tid == 0) {
    const float fmn = lmin[0], fmx = lmax[0];
    const float rmin = fmn < 0.0f ? fmn : 0.0f;
    const float rmax = fmx > 0.0f ? fmx : 0.0f;
    const float qmin = -128.0f, qmax = 127.0f;
    const float range = rmax - rmin;
    const float scale_q = range > 0.0f ? 255.0f / range : 1.0f;
    const float recip = range > 0.0f ? range / 255.0f : 1.0f;
    const float dmin = rmin * scale_q, dmax = rmax * scale_q;
    const float zp_lo = qmin - dmin, zp_hi = qmax - dmax;
    float zp_f =
      (qmin + dmin) + (qmax + dmax) > 0.0f ? zp_lo : zp_hi;
    if (zp_f < qmin) zp_f = qmin;
    if (zp_f > qmax) zp_f = qmax;
    l_scale_q = scale_q;
    l_zp = (int)rint(zp_f);
    scale_per_row[row] = recip;
    zp_per_row[row] = l_zp;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  const float scale_q = l_scale_q;
  const int zp = l_zp;
  int psum = 0;
  for (int k = tid; k < K; k += V8C_QUANT_LWS) {
    int q = (int)rint((float)act_fp16[(long)row * K + k] * scale_q) + zp;
    if (q < -128) q = -128;
    if (q > 127) q = 127;
    act_int8[(long)row * K + k] = (char)q;
    psum += q;
  }
  lsum[tid] = psum;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = V8C_QUANT_LWS / 2; s > 0; s >>= 1) {
    if (tid < s) lsum[tid] += lsum[tid + s];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (tid == 0) row_sum_act[row] = lsum[0];
}

// ============================================================
// v8c int8 × int4(offset) GEMM (signed-unsigned packed dot product).
// Output: fp16 [M, N] (row-major)
// Canonical work-item tile: TM=4, TN=8; LWS 4×16; tile %TM/%TN must divide M/N.
// (Configurable via -DTM= -DTN= at build time if needed; defaults match the
//  best-measured config on Adreno 830: 87% of HW peak.)
// ============================================================
#ifndef V8C_TM
#define V8C_TM 4
#endif
#ifndef V8C_TN
#define V8C_TN 8
#endif

__kernel void v8c_gemm_int8_int4(
    __read_only image2d_t  Ximg,            // act image view (RGBA UINT32, K/16 × M)
    __read_only image2d_t  Wimg,            // weight image view (RGBA UINT32, K/32 × N)
    __global const float  *scale_act,       // [M] per-row act recip-scale
    __global const float  *scale_wgt,       // [N] per-channel weight recip-scale
    __global const int    *row_sum_act,     // [M] sum_k(int8 act_ik)
    __global const int    *zp_act,          // [M] per-row asymmetric zero point
    __global const int    *row_sum_w_int4,  // [N] sum_k(int4 w_nk), pre-decoded
    __global       half   *Y,               // [M, N] fp16 output
    const int M, const int N, const int K) {
  const int n0 = get_global_id(0) * V8C_TN;
  const int m0 = get_global_id(1) * V8C_TM;
  const int K32 = K >> 5;

  int acc[V8C_TM][V8C_TN];
  #pragma unroll
  for (int i = 0; i < V8C_TM; i++)
    #pragma unroll
    for (int j = 0; j < V8C_TN; j++) acc[i][j] = 0;

  for (int k32 = 0; k32 < K32; k32++) {
    // 2 activation texels (16 K each) cover 32 K of activation row
    uint4 a_lo[V8C_TM], a_hi[V8C_TM];
    #pragma unroll
    for (int i = 0; i < V8C_TM; i++) {
      a_lo[i] = read_imageui(Ximg, SMP_v8c, (int2)(2*k32  , m0 + i));
      a_hi[i] = read_imageui(Ximg, SMP_v8c, (int2)(2*k32+1, m0 + i));
    }
    #pragma unroll
    for (int j = 0; j < V8C_TN; j++) {
      // 1 weight texel = 32 int4 K-channels (offset-encoded)
      uint4 w = read_imageui(Wimg, SMP_v8c, (int2)(k32, n0 + j));
      // Mask-only unpack: each masked uint = 4 unsigned bytes in [0..15]
      // (= encoded values; real value = encoded - 8).
      const uint M4 = 0x0F0F0F0Fu;
      uint w0lo =  w.x        & M4;  // K = j_block*32 + [ 0.. 3]
      uint w0hi = (w.x >> 4)  & M4;  // K = j_block*32 + [ 4.. 7]
      uint w1lo =  w.y        & M4;  // K = j_block*32 + [ 8..11]
      uint w1hi = (w.y >> 4)  & M4;  // K = j_block*32 + [12..15]
      uint w2lo =  w.z        & M4;  // K = j_block*32 + [16..19]
      uint w2hi = (w.z >> 4)  & M4;  // K = j_block*32 + [20..23]
      uint w3lo =  w.w        & M4;  // K = j_block*32 + [24..27]
      uint w3hi = (w.w >> 4)  & M4;  // K = j_block*32 + [28..31]
      #pragma unroll
      for (int i = 0; i < V8C_TM; i++) {
        // dot_4x8packed_su_int: signed_int8 (act) × unsigned_uint8 (encoded)
        acc[i][j] += dot_4x8packed_su_int(a_lo[i].x, w0lo)
                   + dot_4x8packed_su_int(a_lo[i].y, w0hi)
                   + dot_4x8packed_su_int(a_lo[i].z, w1lo)
                   + dot_4x8packed_su_int(a_lo[i].w, w1hi)
                   + dot_4x8packed_su_int(a_hi[i].x, w2lo)
                   + dot_4x8packed_su_int(a_hi[i].y, w2hi)
                   + dot_4x8packed_su_int(a_hi[i].z, w3lo)
                   + dot_4x8packed_su_int(a_hi[i].w, w3hi);
      }
    }
  }
  // Bias correction for asymmetric int8 activation + offset-encoded int4 wgt:
  //   acc                          = Σ_k act_q[i,k] * w_enc[j,k]
  //                                 where w_enc = w_int4 + 8 ∈ [0..15]
  //   Σ_k act_q[i,k] * w_int4[j,k] = acc - 8 * row_sum_act[i]
  //   Σ_k (act_q[i,k] - zp_a[i]) * w_int4[j,k]
  //                                 = (acc - 8*row_sum_act[i])
  //                                   - zp_a[i] * row_sum_w_int4[j]
  //   v[i,j]                       = corrected * recip_scale_act[i]
  //                                   * recip_scale_wgt[j]
  #pragma unroll
  for (int i = 0; i < V8C_TM; i++) {
    int rs = row_sum_act[m0 + i];
    int zp = zp_act[m0 + i];
    float s_i = scale_act[m0 + i];
    #pragma unroll
    for (int j = 0; j < V8C_TN; j++) {
      int corrected = acc[i][j] - 8 * rs - zp * row_sum_w_int4[n0 + j];
      float v = (float)corrected * s_i * scale_wgt[n0 + j];
      Y[(long)(m0 + i) * N + (n0 + j)] = (half)v;
    }
  }
}

// ============================================================
// M=1 (decode-phase) specialization. The default TM=4 kernel pads M up to
// 4 for the M=1 decode case and burns 3× the activation reads + 3×
// wasted output writes per work-group. This variant collapses TM=1 so
// each WI only touches row 0, cutting decode-phase GEMM cost ~4×
// (measured: 2002 → ~500 ms cumulative across 6272 decode FC calls).
// gws = (N / V8C_TN_M1, 1, 1); LWS = NULL (driver picks).
// ============================================================
#ifndef V8C_TN_M1
#define V8C_TN_M1 8
#endif

__kernel void v8c_gemm_int8_int4_m1(
    __read_only image2d_t  Ximg,
    __read_only image2d_t  Wimg,
    __global const float  *scale_act,
    __global const float  *scale_wgt,
    __global const int    *row_sum_act,
    __global const int    *zp_act,
    __global const int    *row_sum_w_int4,
    __global       half   *Y,
    const int M, const int N, const int K) {
  const int n0 = get_global_id(0) * V8C_TN_M1;
  const int K32 = K >> 5;

  int acc[V8C_TN_M1];
  #pragma unroll
  for (int j = 0; j < V8C_TN_M1; j++) acc[j] = 0;

  for (int k32 = 0; k32 < K32; k32++) {
    uint4 a_lo = read_imageui(Ximg, SMP_v8c, (int2)(2*k32  , 0));
    uint4 a_hi = read_imageui(Ximg, SMP_v8c, (int2)(2*k32+1, 0));
    #pragma unroll
    for (int j = 0; j < V8C_TN_M1; j++) {
      uint4 w = read_imageui(Wimg, SMP_v8c, (int2)(k32, n0 + j));
      const uint M4 = 0x0F0F0F0Fu;
      uint w0lo =  w.x        & M4;
      uint w0hi = (w.x >> 4)  & M4;
      uint w1lo =  w.y        & M4;
      uint w1hi = (w.y >> 4)  & M4;
      uint w2lo =  w.z        & M4;
      uint w2hi = (w.z >> 4)  & M4;
      uint w3lo =  w.w        & M4;
      uint w3hi = (w.w >> 4)  & M4;
      acc[j] += dot_4x8packed_su_int(a_lo.x, w0lo)
              + dot_4x8packed_su_int(a_lo.y, w0hi)
              + dot_4x8packed_su_int(a_lo.z, w1lo)
              + dot_4x8packed_su_int(a_lo.w, w1hi)
              + dot_4x8packed_su_int(a_hi.x, w2lo)
              + dot_4x8packed_su_int(a_hi.y, w2hi)
              + dot_4x8packed_su_int(a_hi.z, w3lo)
              + dot_4x8packed_su_int(a_hi.w, w3hi);
    }
  }
  const int rs = row_sum_act[0];
  const int zp = zp_act[0];
  const float s_i = scale_act[0];
  #pragma unroll
  for (int j = 0; j < V8C_TN_M1; j++) {
    int corrected = acc[j] - 8 * rs - zp * row_sum_w_int4[n0 + j];
    float v = (float)corrected * s_i * scale_wgt[n0 + j];
    Y[n0 + j] = (half)v;
  }
}

// ============================================================
// #46l: V projection fused with OHWI-reversed scatter (paper §3.6
// "tensor reordering ops fused"). Same compute as v8c_gemm_int8_int4
// but writes directly into V cache [hKV, d, S_max] OHWI-reversed
// layout at offset position+m (over the t-axis). Eliminates the
// separate v_scatter_ohwi_t pass that was ~700 ms at M=1024.
//
// Caller constraints (enforced by checks in host wrapper):
//   * d * hKV == N   (otherwise n decomposition is wrong)
//   * V8C_TN must divide d (so an 8-wide n-tile stays in one head)
//   * Output layout: V[head, x, position+t] at byte offset
//     2 * (head*d*S_max + x*S_max + (position+t))
// ============================================================
__kernel void v8c_gemm_int8_int4_v_ohwi(
    __read_only image2d_t  Ximg,
    __read_only image2d_t  Wimg,
    __global const float  *scale_act,
    __global const float  *scale_wgt,
    __global const int    *row_sum_act,
    __global const int    *zp_act,
    __global const int    *row_sum_w_int4,
    __global       half   *V_ohwi,        // [hKV, d, S_max] fp16
    const int M, const int N, const int K,
    const int d, const int S_max, const int position,
    const int M_real) {                   // skip writes for rows >= M_real
  const int n0 = get_global_id(0) * V8C_TN;
  const int m0 = get_global_id(1) * V8C_TM;
  const int K32 = K >> 5;

  int acc[V8C_TM][V8C_TN];
  #pragma unroll
  for (int i = 0; i < V8C_TM; i++)
    #pragma unroll
    for (int j = 0; j < V8C_TN; j++) acc[i][j] = 0;

  for (int k32 = 0; k32 < K32; k32++) {
    uint4 a_lo[V8C_TM], a_hi[V8C_TM];
    #pragma unroll
    for (int i = 0; i < V8C_TM; i++) {
      a_lo[i] = read_imageui(Ximg, SMP_v8c, (int2)(2*k32  , m0 + i));
      a_hi[i] = read_imageui(Ximg, SMP_v8c, (int2)(2*k32+1, m0 + i));
    }
    #pragma unroll
    for (int j = 0; j < V8C_TN; j++) {
      uint4 w = read_imageui(Wimg, SMP_v8c, (int2)(k32, n0 + j));
      const uint M4 = 0x0F0F0F0Fu;
      uint w0lo =  w.x        & M4;
      uint w0hi = (w.x >> 4)  & M4;
      uint w1lo =  w.y        & M4;
      uint w1hi = (w.y >> 4)  & M4;
      uint w2lo =  w.z        & M4;
      uint w2hi = (w.z >> 4)  & M4;
      uint w3lo =  w.w        & M4;
      uint w3hi = (w.w >> 4)  & M4;
      #pragma unroll
      for (int i = 0; i < V8C_TM; i++) {
        acc[i][j] += dot_4x8packed_su_int(a_lo[i].x, w0lo)
                   + dot_4x8packed_su_int(a_lo[i].y, w0hi)
                   + dot_4x8packed_su_int(a_lo[i].z, w1lo)
                   + dot_4x8packed_su_int(a_lo[i].w, w1hi)
                   + dot_4x8packed_su_int(a_hi[i].x, w2lo)
                   + dot_4x8packed_su_int(a_hi[i].y, w2hi)
                   + dot_4x8packed_su_int(a_hi[i].z, w3lo)
                   + dot_4x8packed_su_int(a_hi[i].w, w3hi);
      }
    }
  }
  // n0 is V8C_TN-aligned and V8C_TN divides d → all 8 n's share one head.
  const int head = n0 / d;
  const int x_base = n0 - head * d;
  const long head_off = (long)head * (long)d * (long)S_max;
  #pragma unroll
  for (int i = 0; i < V8C_TM; i++) {
    const int m = m0 + i;
    if (m >= M_real) continue;
    int rs = row_sum_act[m];
    int zp = zp_act[m];
    float s_i = scale_act[m];
    const long t_off = (long)(position + m);
    #pragma unroll
    for (int j = 0; j < V8C_TN; j++) {
      int corrected = acc[i][j] - 8 * rs - zp * row_sum_w_int4[n0 + j];
      float v = (float)corrected * s_i * scale_wgt[n0 + j];
      V_ohwi[head_off + (long)(x_base + j) * (long)S_max + t_off] = (half)v;
    }
  }
}

// ============================================================
// 8/4/4 paper ATTENTION path: int8(act) × int8(weight) channel-wise GEMM.
// Weight format (cl_mem viewed as image2d): plain ROW-MAJOR signed int8 [N][K],
//   image RGBA + UNSIGNED_INT32, width=K/16, height=N, row_pitch=K (16 int8/texel).
// Activation: identical int8 image as the int4 path (width K/16, 16 int8/texel).
// Math: acc(i,j) = Σ_k act_q[i,k]·w_q[j,k]   (dot_4x8packed_ss_int, both signed).
//   corrected = acc − zp_act[i]·row_sum_w[j]   (NO −8 term: int8 weight is signed,
//                                               not offset-encoded like int4+8)
//   y[i,j]    = corrected · scale_act[i] · scale_wgt[j]
// Signature is byte-identical to v8c_gemm_int8_int4 (row_sum_act arg ignored) so
// the host wrapper is shared; only the kernel name + weight image width differ.
// ============================================================
__kernel void v8c_gemm_int8_int8(
    __read_only image2d_t  Ximg,            // act image (RGBA UINT32, K/16 × M)
    __read_only image2d_t  Wimg,            // weight image (RGBA UINT32, K/16 × N)
    __global const float  *scale_act,       // [M]
    __global const float  *scale_wgt,       // [N]
    __global const int    *row_sum_act,     // [M] UNUSED (kept for ABI parity)
    __global const int    *zp_act,          // [M]
    __global const int    *row_sum_w,       // [N] = Σ_k int8 w[n,k]
    __global       half   *Y,               // [M, N] fp16
    const int M, const int N, const int K) {
  const int n0 = get_global_id(0) * V8C_TN;
  const int m0 = get_global_id(1) * V8C_TM;
  const int K16 = K >> 4;

  int acc[V8C_TM][V8C_TN];
  #pragma unroll
  for (int i = 0; i < V8C_TM; i++)
    #pragma unroll
    for (int j = 0; j < V8C_TN; j++) acc[i][j] = 0;

  for (int k16 = 0; k16 < K16; k16++) {
    uint4 a[V8C_TM];
    #pragma unroll
    for (int i = 0; i < V8C_TM; i++)
      a[i] = read_imageui(Ximg, SMP_v8c, (int2)(k16, m0 + i));
    #pragma unroll
    for (int j = 0; j < V8C_TN; j++) {
      uint4 w = read_imageui(Wimg, SMP_v8c, (int2)(k16, n0 + j));
      #pragma unroll
      for (int i = 0; i < V8C_TM; i++) {
        acc[i][j] += dot_4x8packed_ss_int(a[i].x, w.x)
                   + dot_4x8packed_ss_int(a[i].y, w.y)
                   + dot_4x8packed_ss_int(a[i].z, w.z)
                   + dot_4x8packed_ss_int(a[i].w, w.w);
      }
    }
  }
  #pragma unroll
  for (int i = 0; i < V8C_TM; i++) {
    int zp = zp_act[m0 + i];
    float s_i = scale_act[m0 + i];
    #pragma unroll
    for (int j = 0; j < V8C_TN; j++) {
      int corrected = acc[i][j] - zp * row_sum_w[n0 + j];
      Y[(long)(m0 + i) * N + (n0 + j)] = (half)((float)corrected * s_i * scale_wgt[n0 + j]);
    }
  }
}

// M=1 (decode) specialization of v8c_gemm_int8_int8. gws=(N/V8C_TN_M1,1,1).
__kernel void v8c_gemm_int8_int8_m1(
    __read_only image2d_t  Ximg,
    __read_only image2d_t  Wimg,
    __global const float  *scale_act,
    __global const float  *scale_wgt,
    __global const int    *row_sum_act,     // UNUSED
    __global const int    *zp_act,
    __global const int    *row_sum_w,
    __global       half   *Y,
    const int M, const int N, const int K) {
  const int n0 = get_global_id(0) * V8C_TN_M1;
  const int K16 = K >> 4;
  int acc[V8C_TN_M1];
  #pragma unroll
  for (int j = 0; j < V8C_TN_M1; j++) acc[j] = 0;
  for (int k16 = 0; k16 < K16; k16++) {
    uint4 a = read_imageui(Ximg, SMP_v8c, (int2)(k16, 0));
    #pragma unroll
    for (int j = 0; j < V8C_TN_M1; j++) {
      uint4 w = read_imageui(Wimg, SMP_v8c, (int2)(k16, n0 + j));
      acc[j] += dot_4x8packed_ss_int(a.x, w.x)
              + dot_4x8packed_ss_int(a.y, w.y)
              + dot_4x8packed_ss_int(a.z, w.z)
              + dot_4x8packed_ss_int(a.w, w.w);
    }
  }
  const int zp = zp_act[0];
  const float s_i = scale_act[0];
  #pragma unroll
  for (int j = 0; j < V8C_TN_M1; j++) {
    int corrected = acc[j] - zp * row_sum_w[n0 + j];
    Y[n0 + j] = (half)((float)corrected * s_i * scale_wgt[n0 + j]);
  }
}

// int8×int8 variant of v8c_gemm_int8_int4_v_ohwi (fused OHWI-reversed V scatter
// for the wv projection). Same OHWI write math, int8 weight + ss dot.
__kernel void v8c_gemm_int8_int8_v_ohwi(
    __read_only image2d_t  Ximg,
    __read_only image2d_t  Wimg,
    __global const float  *scale_act,
    __global const float  *scale_wgt,
    __global const int    *row_sum_act,     // UNUSED
    __global const int    *zp_act,
    __global const int    *row_sum_w,
    __global       half   *V_ohwi,          // [hKV, d, S_max] fp16
    const int M, const int N, const int K,
    const int d, const int S_max, const int position,
    const int M_real) {
  const int n0 = get_global_id(0) * V8C_TN;
  const int m0 = get_global_id(1) * V8C_TM;
  const int K16 = K >> 4;
  int acc[V8C_TM][V8C_TN];
  #pragma unroll
  for (int i = 0; i < V8C_TM; i++)
    #pragma unroll
    for (int j = 0; j < V8C_TN; j++) acc[i][j] = 0;
  for (int k16 = 0; k16 < K16; k16++) {
    uint4 a[V8C_TM];
    #pragma unroll
    for (int i = 0; i < V8C_TM; i++)
      a[i] = read_imageui(Ximg, SMP_v8c, (int2)(k16, m0 + i));
    #pragma unroll
    for (int j = 0; j < V8C_TN; j++) {
      uint4 w = read_imageui(Wimg, SMP_v8c, (int2)(k16, n0 + j));
      #pragma unroll
      for (int i = 0; i < V8C_TM; i++) {
        acc[i][j] += dot_4x8packed_ss_int(a[i].x, w.x)
                   + dot_4x8packed_ss_int(a[i].y, w.y)
                   + dot_4x8packed_ss_int(a[i].z, w.z)
                   + dot_4x8packed_ss_int(a[i].w, w.w);
      }
    }
  }
  const int head = n0 / d;
  const int x_base = n0 - head * d;
  const long head_off = (long)head * (long)d * (long)S_max;
  #pragma unroll
  for (int i = 0; i < V8C_TM; i++) {
    const int m = m0 + i;
    if (m >= M_real) continue;
    int zp = zp_act[m];
    float s_i = scale_act[m];
    const long t_off = (long)(position + m);
    #pragma unroll
    for (int j = 0; j < V8C_TN; j++) {
      int corrected = acc[i][j] - zp * row_sum_w[n0 + j];
      float v = (float)corrected * s_i * scale_wgt[n0 + j];
      V_ohwi[head_off + (long)(x_base + j) * (long)S_max + t_off] = (half)v;
    }
  }
}
