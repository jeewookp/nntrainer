// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Samsung Electronics Co., Ltd.
 *
 * @file	unittest_opencl_kernels_int4_adreno.cpp
 * @date	December 2025
 * @brief	Unit tests for the Adreno-specific int4 GEMM OpenCL kernel
 *              (`gpu_int4_gemm_adreno`) reached via
 *              `nntrainer::gemm_int4_adreno_cl`.
 * @see		https://github.com/nnstreamer/nntrainer
 * @bug		No known bugs except for NYI items
 *
 * The test is modeled on the original version carried on the `claude`
 * branch (`test/unittest/unittest_opencl_kernels_int4_adreno.cpp` there)
 * but:
 *   - Uses the all_gpu wrapper name `gemm_int4_adreno_cl`.
 *   - Drops the experimental Q4_0 -> channel-wise int4 repack that was
 *     folded into the timing loop in the claude copy (it wasn't part of
 *     the correctness contract, and it made the test non-deterministic).
 *   - Adds proper MSE-against-reference tolerance plus a small number of
 *     element-wise spot checks so the test catches layout bugs that an
 *     MSE average would smooth out.
 *
 * The weight packing path uses `Int4Utils::quantizeAndRepackSimpleLayout`,
 * which lands the int4 nibbles directly in the channel-wise layout the
 * `gpu_int4_gemm_adreno` kernel reads (matches
 * `Int4Utils::convertKaiToChannelwise`). The CPU reference GEMM runs on
 * the same fp32 weights via `nntrainer::sgemm`; the tolerance has to
 * cover (a) per-channel int4 quantization error and (b) fp16 rounding
 * on the activation / output path.
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <gtest/gtest.h>
#include <vector>

#include "int4_utils.h"
#include "nntrainer_test_util.h"
#include "q4_0_utils.h"
#include <blas_kernels.h>
#include <cl_context.h>
#include <cpu_backend.h>
#include <fp16.h>

using namespace nntrainer;

namespace {

// True channel-wise int4 quantization helper.
//
// Why we need a test-local helper instead of
// `Int4Utils::quantizeAndRepackSimpleLayout`:
//
//   - The gemm_int4_adreno_cl wrapper forces the kernel's
//     `quantization_group_size` parameter to K (see blas_kernels.cpp:977),
//     meaning the kernel only ever reads one scale per output channel
//     (scales[n], the (k / K) factor is 0 for all k < K).
//   - `quantizeAndRepackSimpleLayout` requires `group_size` to be one of
//     {32, 64, 128}, so we can't pass K to it.
//   - Passing group_size=32 / 128 to that helper produces per-group scales
//     (computed as max|w| over a 32- or 128-element k sub-row), and the
//     kernel then reads only the FIRST group's scales for each channel.
//     That silently uses "max of w[n, 0..31]" as the scale for the entire
//     row, which is wrong and the claude branch's original test didn't
//     catch it because it only printed the MSE without asserting.
//
// We therefore do the quant in the test file:
//   - per-channel symmetric scale = max|w[n, :]| / 7  (int4 range [-8, 7])
//   - scale is round-tripped through fp16 so the dequantized float
//     weights match BYTE-FOR-BYTE what the kernel materializes from
//     `((nibble - 8) * fp16(scale))`.
//   - nibble layout: out_nibbles[(k/4) * N + n] packs
//       bits 0..3 = (k,   unsigned bias-8 nibble)
//       bits 4..7 = (k+1, ...)
//       bits 8..11= (k+2, ...)
//       bits 12..15 = (k+3, ...)
//     (matches `Int4Utils::convertKaiToChannelwise`).
//
// Output `dequant_weights` is the reference fp32 weight matrix used by
// the CPU sgemm pass, so the CPU reference materializes the exact
// post-quant values the GPU sees. This keeps the MSE tolerance purely
// a function of fp16 accumulation noise, not int4 quantization error.
void PackInt4ChannelwiseAdreno(const float *weights_fp32, unsigned int N,
                               unsigned int K,
                               std::vector<uint16_t> &out_nibbles,
                               std::vector<uint16_t> &out_scales_fp16,
                               std::vector<float> &out_dequant) {
  ASSERT_EQ(K % 4u, 0u)
    << "PackInt4ChannelwiseAdreno: K must be a multiple of 4";
  ASSERT_EQ(N % 32u, 0u)
    << "PackInt4ChannelwiseAdreno: N must be a multiple of 32 (kernel "
       "vload4 walks along align_N = N)";

  out_nibbles.assign(static_cast<size_t>(K / 4) * N, 0u);
  out_scales_fp16.assign(N, 0u);
  out_dequant.assign(static_cast<size_t>(N) * K, 0.0f);

  for (unsigned int n = 0; n < N; ++n) {
    // 1. Per-channel scale = max(|w[n, :]|) / 7 (symmetric int4 range
    //    [-8, 7]; 7 is used so that the largest magnitude round-trips
    //    to -7 or +7 without overflow into -8, which has asymmetric
    //    error for symmetric inputs).
    float max_abs = 0.0f;
    for (unsigned int k = 0; k < K; ++k) {
      const float v = std::fabs(weights_fp32[n * K + k]);
      if (v > max_abs)
        max_abs = v;
    }
    const float scale = (max_abs > 0.0f) ? (max_abs / 7.0f) : 1.0f;
    out_scales_fp16[n] = compute_fp32_to_fp16(scale);

    // Use the fp16-rounded scale for BOTH the pack step and the CPU
    // reference dequant so the two sides use numerically identical
    // constants.
    const float scale_rounded = compute_fp16_to_fp32(out_scales_fp16[n]);
    const float inv_scale =
      (scale_rounded > 0.0f) ? (1.0f / scale_rounded) : 0.0f;

    // 2. Per-element quant to signed int in [-8, 7], store as unsigned
    //    bias-8 nibble (0..15), packed 4 nibbles per ushort at
    //    out_nibbles[(k/4) * N + n] with bit position 4*(k%4).
    for (unsigned int k = 0; k < K; ++k) {
      const float w = weights_fp32[n * K + k];
      int q = static_cast<int>(std::nearbyint(w * inv_scale));
      if (q < -8)
        q = -8;
      if (q > 7)
        q = 7;
      const uint16_t nibble = static_cast<uint16_t>(q + 8) & 0xFu;
      const size_t dst_idx = static_cast<size_t>(k / 4) * N + n;
      out_nibbles[dst_idx] |= static_cast<uint16_t>(nibble << (4 * (k % 4)));

      // Dequantized reference weight -- exactly what the kernel
      // computes as ((nibble - 8) * scale).
      out_dequant[n * K + k] = static_cast<float>(q) * scale_rounded;
    }
  }
}

// Per-element MSE tolerance. Because the CPU reference uses the
// dequantized weights (same fp32 values the GPU materializes), the
// only error source is fp16 accumulation along the K dimension, which
// grows ~linearly in K. A 1e-2 / element bound (squared) with a 4x
// safety factor is enough for all shapes we test and still tight
// enough to catch a layout bug that blows up a whole row.
float GetMseTolerance(unsigned int M, unsigned int K, unsigned int N) {
  (void)M;
  (void)N;
  const float per_elem = static_cast<float>(K) * 1e-3f;
  return per_elem * per_elem * 4.0f;
}

// Per-row int8 activation quantization (symmetric, range [-127, 127]).
//
// Matches what `gpu_int8_int4_gemm_adreno` expects:
//   x_q[m * K_pad + k]   = round(x[m, k] * 127 / max|x[m, :]|), clamped
//                          to [-127, 127]
//   x_scale[m]           = max|x[m, :]| / 127                  (fp16)
//   out_dequant[m * K + k] = q * scale_rounded                 (fp32)
//
// `out_dequant` is the fp32 reference activation the CPU sgemm pass
// uses, so the A/B comparison measures kernel correctness (including
// DP4A path) against the same quantized math, not against the full
// fp32 reference. This keeps the MSE tolerance purely a function of
// fp16 output rounding + int8 quant error, not two stacked quant
// errors.
//
// K_pad lets the caller zero-pad the flat int8 buffer to a width >= K
// (needed for align(M,8) row-padding on the DP4A path -- we only pad
// rows, not columns, but keep the arg for symmetry).
void QuantizeActivationInt8(const float *x_fp32, unsigned int M,
                            unsigned int K, unsigned int M_padded,
                            unsigned int K_padded,
                            std::vector<int8_t> &out_q,
                            std::vector<uint16_t> &out_scale_fp16,
                            std::vector<float> &out_dequant) {
  out_q.assign(static_cast<size_t>(M_padded) * K_padded, 0);
  out_scale_fp16.assign(M, 0u);
  out_dequant.assign(static_cast<size_t>(M) * K, 0.0f);

  for (unsigned int m = 0; m < M; ++m) {
    float max_abs = 0.0f;
    for (unsigned int k = 0; k < K; ++k) {
      const float v = std::fabs(x_fp32[m * K + k]);
      if (v > max_abs)
        max_abs = v;
    }
    const float scale = (max_abs > 0.0f) ? (max_abs / 127.0f) : 1.0f;
    out_scale_fp16[m] = compute_fp32_to_fp16(scale);

    const float scale_rounded = compute_fp16_to_fp32(out_scale_fp16[m]);
    const float inv_scale =
      (scale_rounded > 0.0f) ? (1.0f / scale_rounded) : 0.0f;

    for (unsigned int k = 0; k < K; ++k) {
      int q = static_cast<int>(std::nearbyint(x_fp32[m * K + k] * inv_scale));
      if (q < -127)
        q = -127;
      if (q > 127)
        q = 127;
      out_q[static_cast<size_t>(m) * K_padded + k] = static_cast<int8_t>(q);
      out_dequant[m * K + k] = static_cast<float>(q) * scale_rounded;
    }
    // Rest of this row's K_padded tail stays zero (set by assign).
  }
  // Rows [M, M_padded) already zero-initialized by the assign call.
}

// MSE tolerance for the DP4A path. The dequantized weights and the
// dequantized activations are both used as the CPU reference, so the
// dominant error source is fp16 output rounding over a sum of K int32
// products. A per-element ~1e-3 * K bound (squared, 4x safety) covers
// shapes up to K=1024 without being loose enough to hide layout bugs.
float GetMseToleranceDp4a(unsigned int M, unsigned int K, unsigned int N) {
  (void)M;
  (void)N;
  const float per_elem = static_cast<float>(K) * 1.5e-3f;
  return per_elem * per_elem * 4.0f;
}

} // namespace

static void run_gemm_int4_adreno_test_(const unsigned int M,
                                        const unsigned int K,
                                        const unsigned int N) {
  auto *blas_cc = static_cast<nntrainer::ClContext *>(
    nntrainer::Engine::Global().getRegisteredContext("gpu"));
  ASSERT_NE(blas_cc, nullptr)
    << "GPU ClContext unavailable -- cannot run gemm_int4_adreno test";

  // The gemm_int4_adreno_cl wrapper forces channel-wise quantization by
  // overriding any caller-supplied group size:
  //     const int q_group_size = static_cast<int>(K); // per-channel
  // so the device kernel only ever loads `scales[n]` for output channel
  // `n` (the `(k / quantization_group_size) * align_N` factor is zero
  // for all k < K). The test must match: one fp16 scale per output
  // channel, and `quantizeAndRepackSimpleLayout` invoked with
  // group_size = K so `computeScales` yields per-row max_abs / 7, not
  // per-group max over a 32- / 128-element sub-row.
  //
  // A mismatched group size here silently ran with wrong scales: the
  // kernel read scales[n] which `computeScales(group_size=32)` had
  // computed using only w[n, 0..31] as the "group 0" max, so every
  // output channel ended up scaled by at most the magnitude of its
  // first 32 weights. MSE passed by luck on small shapes and failed
  // on large ones.
  const unsigned int scale_group_size = K;

  constexpr int INT4_BLOCK_N_SIZE = 32;
  const unsigned int alignN = align(N, INT4_BLOCK_N_SIZE);
  const unsigned int alignK = align(K, scale_group_size);

  // --- Random fp32 input [M][K] and weights [N][K] --------------------
  std::vector<float> input_orig =
    generate_random_vector<float, false>(M * K, -1.0f, 1.0f);
  std::vector<float> weight_fp32 =
    generate_random_vector<float, false>(N * K, -1.0f, 1.0f);

  // --- Channel-wise int4 pack (symmetric, per-row scale) --------------
  // dequant_weights is the float weight matrix the GPU actually
  // materializes from `((nibble - 8) * scale)` -- we use it for the
  // CPU reference so comparison measures kernel correctness, not
  // quantization fidelity.
  std::vector<uint16_t> packed_nibbles;
  std::vector<uint16_t> packed_scales_fp16;
  std::vector<float> dequant_weights;
  PackInt4ChannelwiseAdreno(weight_fp32.data(), N, K, packed_nibbles,
                             packed_scales_fp16, dequant_weights);

  // --- CPU reference: ref = input * W_deq^T ---------------------------
  // Row-major sgemm with TransB=true matches the kernel math:
  //   out[m, n] = sum_k input[m, k] * W_deq[n, k]
  std::vector<float> ref_dst(static_cast<size_t>(M) * N, 0.0f);
  nntrainer::sgemm(/*TStorageOrder=*/0, /*TransA=*/false, /*TransB=*/true,
                   /*M=*/M, /*N=*/N, /*K=*/K, /*alpha=*/1.0f,
                   input_orig.data(), /*lda=*/K, dequant_weights.data(),
                   /*ldb=*/K, /*beta=*/0.0f, ref_dst.data(), /*ldc=*/N);

  // --- Pad the input to [M][alignK]. For channel-wise quant alignK == K,
  //     so this is a no-op copy, but the wrapper still expects a buffer
  //     of exactly M * alignK fp16 elements.
  std::vector<float> input_padded(static_cast<size_t>(M) * alignK, 0.0f);
  for (unsigned int m = 0; m < M; ++m) {
    for (unsigned int k = 0; k < K; ++k) {
      input_padded[m * alignK + k] = input_orig[m * K + k];
    }
  }

  // --- Allocate the SVM buffers the Adreno kernel wrapper expects ------
  //   input            : fp16 [M * alignK]                      (run-time)
  //   input_transposed : fp16 [align(M, 4) * alignK]             (scratch)
  //   weights          : ushort [(K / 4) * N]                    (const)
  //   scales           : fp16 [alignN]                           (const,
  //                                                               channel-wise)
  //   output           : fp16 [M * N]                            (run-time)
  //
  // Channel-wise note: the kernel reads
  //   scales[(k / q_group_size) * align_N + n]
  // with q_group_size = K forced by the wrapper, so the index
  // collapses to `scales[n]`. Allocating just `alignN` fp16 is enough
  // -- we padded N up to align(N, 32) above so every `n in [0, N)` the
  // kernel touches is in-bounds.
  const size_t input_svm_bytes =
    static_cast<size_t>(M) * alignK * sizeof(uint16_t);
  const size_t input_t_svm_bytes =
    static_cast<size_t>(align(M, 4u)) * alignK * sizeof(uint16_t);
  const size_t weights_svm_bytes =
    static_cast<size_t>(alignK / 4u) * N * sizeof(uint16_t);
  const size_t scales_svm_bytes =
    static_cast<size_t>(alignN) * sizeof(uint16_t);
  const size_t output_svm_bytes =
    static_cast<size_t>(M) * N * sizeof(uint16_t);

  uint16_t *input_ptr = static_cast<uint16_t *>(allocateSVM(input_svm_bytes));
  uint16_t *input_t_ptr =
    static_cast<uint16_t *>(allocateSVM(input_t_svm_bytes));
  uint16_t *weight_ptr =
    static_cast<uint16_t *>(allocateSVM(weights_svm_bytes));
  uint16_t *scale_ptr =
    static_cast<uint16_t *>(allocateSVM(scales_svm_bytes));
  uint16_t *output_ptr = static_cast<uint16_t *>(allocateSVM(output_svm_bytes));

  // Map before host-side fill. The command queue is lazy about transfer
  // ordering, and the wrapper internally wraps these as CL_MEM_USE_HOST_PTR
  // so the runtime wants to know when host side has touched them.
  blas_cc->command_queue_inst_.enqueueSVMMap(input_ptr, input_svm_bytes, false);
  blas_cc->command_queue_inst_.enqueueSVMMap(input_t_ptr, input_t_svm_bytes,
                                              false);
  blas_cc->command_queue_inst_.enqueueSVMMap(weight_ptr, weights_svm_bytes,
                                              false);
  blas_cc->command_queue_inst_.enqueueSVMMap(scale_ptr, scales_svm_bytes,
                                              false);
  blas_cc->command_queue_inst_.enqueueSVMMap(output_ptr, output_svm_bytes,
                                              false);

  for (size_t i = 0; i < static_cast<size_t>(M) * alignK; ++i) {
    input_ptr[i] = compute_fp32_to_fp16(input_padded[i]);
  }
  std::memset(input_t_ptr, 0, input_t_svm_bytes);
  // packed_nibbles has (K / 4) * N entries (with K a multiple of 4 per
  // the assert in PackInt4ChannelwiseAdreno).
  for (size_t i = 0; i < packed_nibbles.size(); ++i) {
    weight_ptr[i] = packed_nibbles[i];
  }
  // Channel-wise: N scales for the real channels, then pad to alignN
  // with zeros. The kernel never reads beyond scales[N-1] because n =
  // global_id(1) * 4 < N is the loop invariant, but the SVM region is
  // alignN wide so we explicitly zero the tail.
  std::memset(scale_ptr, 0, scales_svm_bytes);
  for (size_t n = 0; n < N; ++n) {
    scale_ptr[n] = packed_scales_fp16[n];
  }
  std::memset(output_ptr, 0, output_svm_bytes);

  blas_cc->command_queue_inst_.enqueueSVMUnmap(input_ptr);
  blas_cc->command_queue_inst_.enqueueSVMUnmap(input_t_ptr);
  blas_cc->command_queue_inst_.enqueueSVMUnmap(weight_ptr);
  blas_cc->command_queue_inst_.enqueueSVMUnmap(scale_ptr);
  blas_cc->command_queue_inst_.enqueueSVMUnmap(output_ptr);

  // --- Run the kernel --------------------------------------------------
  // Warm up once (weights upload / kernel compile is not what we're
  // measuring; the wrapper handles its own final SVMMap on output).
  nntrainer::gemm_int4_adreno_cl(input_ptr, input_t_ptr, weight_ptr, scale_ptr,
                                   output_ptr, M, N, K);

  // Second, timed call — a single call avoids cross-iteration side
  // effects from the weight cache. For perf sweeps the Benchmark at the
  // end of the file loops this.
  const auto t0 = std::chrono::high_resolution_clock::now();
  nntrainer::gemm_int4_adreno_cl(input_ptr, input_t_ptr, weight_ptr, scale_ptr,
                                   output_ptr, M, N, K);
  const auto t1 = std::chrono::high_resolution_clock::now();
  const double gpu_ms =
    std::chrono::duration<double, std::milli>(t1 - t0).count();

  // --- Validate --------------------------------------------------------
  std::vector<float> gpu_out_fp32(static_cast<size_t>(M) * N, 0.0f);
  for (size_t i = 0; i < static_cast<size_t>(M) * N; ++i) {
    gpu_out_fp32[i] = compute_fp16_to_fp32(output_ptr[i]);
  }

  const float mse_err =
    mse<float>(ref_dst.data(), gpu_out_fp32.data(), M * N);
  const float mse_tol = GetMseTolerance(M, K, N);

  std::cout << "int4_gemm_adreno M=" << M << " K=" << K << " N=" << N
            << " (channel-wise) gpu=" << gpu_ms << " ms  MSE=" << mse_err
            << " (tol=" << mse_tol << ")" << std::endl;

  EXPECT_LT(mse_err, mse_tol)
    << "MSE exceeded tolerance for gemm_int4_adreno M=" << M << " K=" << K
    << " N=" << N << " (mse=" << mse_err << " tol=" << mse_tol << ")";

  // Spot-check a handful of output elements against the CPU reference
  // with a per-element relative tolerance. These catch layout bugs where
  // a few rows/cols get swapped but the MSE stays within the loose
  // tolerance.
  const size_t num_spot_checks = std::min<size_t>(16, static_cast<size_t>(M) * N);
  for (size_t i = 0; i < num_spot_checks; ++i) {
    const size_t idx = (i * 131u + 17u) % (static_cast<size_t>(M) * N);
    const float cpu = ref_dst[idx];
    const float gpu = gpu_out_fp32[idx];
    const float abs_err = std::fabs(cpu - gpu);
    // Relative tolerance of 20 % on top of an absolute floor of
    // (K * 0.02). The absolute floor covers the case where cpu is
    // close to 0 (so relative error explodes) while the accumulation
    // noise is still O(K).
    const float rel_scale = std::max(std::fabs(cpu), 1.0f);
    const float abs_floor = static_cast<float>(K) * 0.02f;
    EXPECT_LT(abs_err, std::max(0.20f * rel_scale, abs_floor))
      << "Spot check failed at idx=" << idx << " (m=" << (idx / N)
      << ", n=" << (idx % N) << ") cpu=" << cpu << " gpu=" << gpu;
  }

  freeSVM(output_ptr);
  freeSVM(scale_ptr);
  freeSVM(weight_ptr);
  freeSVM(input_t_ptr);
  freeSVM(input_ptr);
}

#define DECLARE_int4_gemm_adreno_test_M_K_N(M, K, N)                           \
  TEST(nntrainer_opencl_adreno_kernels_int4,                                   \
       int4_gemm_adreno_test_##M##_##K##_##N) {                                \
    run_gemm_int4_adreno_test_(M, K, N);                                       \
  }

// Small sanity shapes. N must be a multiple of 32 and K a multiple of 4
// per Int4Utils::channelwise_layout_size / the kernel's vload4 guard.
// These verify nibble packing + the input_transpose pre-pass + dispatch
// geometry (ceilDiv(M, 8) x N/4) without running for seconds.
DECLARE_int4_gemm_adreno_test_M_K_N(8, 32, 32);
DECLARE_int4_gemm_adreno_test_M_K_N(8, 64, 64);
DECLARE_int4_gemm_adreno_test_M_K_N(16, 128, 128);
DECLARE_int4_gemm_adreno_test_M_K_N(32, 256, 256);

// Medium prefill-like shapes that hit the M > 1 wrapper path via
// HalfTensor::dotQInteger. These approximate Qwen3 / Gemma4 matmul
// blocks (hidden 1536..2048 range).
DECLARE_int4_gemm_adreno_test_M_K_N(64, 512, 256);
DECLARE_int4_gemm_adreno_test_M_K_N(128, 1024, 512);
DECLARE_int4_gemm_adreno_test_M_K_N(512, 1024, 1024);

// =========================================================================
// INT8 activation x INT4 weight (DP4A) tests + A/B benchmark.
//
// These use the same weight layout as the fp16 path above; the only
// difference is per-row int8 activation quant and the
// `gpu_int8_int4_gemm_adreno` device kernel. Because the CPU reference
// uses dequantized weights AND dequantized activations, the MSE
// tolerance only has to cover fp16 rounding of the output (quant error
// is absorbed by matching the reference to the quantized inputs).
// =========================================================================
static void run_gemm_int8_int4_adreno_test_(const unsigned int M,
                                             const unsigned int K,
                                             const unsigned int N) {
  auto *blas_cc = static_cast<nntrainer::ClContext *>(
    nntrainer::Engine::Global().getRegisteredContext("gpu"));
  ASSERT_NE(blas_cc, nullptr)
    << "GPU ClContext unavailable -- cannot run gemm_int8_int4_adreno test";

  constexpr int INT4_BLOCK_N_SIZE = 32;
  const unsigned int alignN = align(N, INT4_BLOCK_N_SIZE);
  // The DP4A kernel reads 8 rows per WI via direct char4 loads, so rows
  // [M, align(M, 8)) have to be zero-padded by the caller.
  const unsigned int alignM = align(M, 8u);

  std::vector<float> input_orig =
    generate_random_vector<float, false>(M * K, -1.0f, 1.0f);
  std::vector<float> weight_fp32 =
    generate_random_vector<float, false>(N * K, -1.0f, 1.0f);

  std::vector<uint16_t> packed_nibbles;
  std::vector<uint16_t> packed_w_scales_fp16;
  std::vector<float> dequant_weights;
  PackInt4ChannelwiseAdreno(weight_fp32.data(), N, K, packed_nibbles,
                             packed_w_scales_fp16, dequant_weights);

  // Per-row int8 activation quant. The padded dims match the SVM layout
  // we'll hand to the kernel (alignM rows x K columns).
  std::vector<int8_t> x_q;
  std::vector<uint16_t> x_scale_fp16;
  std::vector<float> dequant_x;
  QuantizeActivationInt8(input_orig.data(), M, K, alignM, K, x_q,
                          x_scale_fp16, dequant_x);

  // CPU reference: ref = X_deq * W_deq^T (row-major, TransB=true)
  std::vector<float> ref_dst(static_cast<size_t>(M) * N, 0.0f);
  nntrainer::sgemm(/*TStorageOrder=*/0, /*TransA=*/false, /*TransB=*/true,
                   /*M=*/M, /*N=*/N, /*K=*/K, /*alpha=*/1.0f,
                   dequant_x.data(), /*lda=*/K, dequant_weights.data(),
                   /*ldb=*/K, /*beta=*/0.0f, ref_dst.data(), /*ldc=*/N);

  // --- SVM buffer allocation ------------------------------------------
  //   x_q_ptr      : int8 [alignM * K]
  //   x_scale_ptr  : fp16 [M]
  //   weight_ptr   : ushort[(K/4) * N]
  //   w_scale_ptr  : fp16 [alignN]
  //   output_ptr   : fp16 [M * N]
  const size_t x_q_bytes =
    static_cast<size_t>(alignM) * K * sizeof(int8_t);
  const size_t x_scale_bytes = static_cast<size_t>(M) * sizeof(uint16_t);
  const size_t weight_bytes =
    static_cast<size_t>(K / 4) * N * sizeof(uint16_t);
  const size_t w_scale_bytes =
    static_cast<size_t>(alignN) * sizeof(uint16_t);
  const size_t output_bytes =
    static_cast<size_t>(M) * N * sizeof(uint16_t);

  int8_t *x_q_ptr = static_cast<int8_t *>(allocateSVM(x_q_bytes));
  uint16_t *x_scale_ptr =
    static_cast<uint16_t *>(allocateSVM(x_scale_bytes));
  uint16_t *weight_ptr =
    static_cast<uint16_t *>(allocateSVM(weight_bytes));
  uint16_t *w_scale_ptr =
    static_cast<uint16_t *>(allocateSVM(w_scale_bytes));
  uint16_t *output_ptr =
    static_cast<uint16_t *>(allocateSVM(output_bytes));

  blas_cc->command_queue_inst_.enqueueSVMMap(x_q_ptr, x_q_bytes, false);
  blas_cc->command_queue_inst_.enqueueSVMMap(x_scale_ptr, x_scale_bytes,
                                              false);
  blas_cc->command_queue_inst_.enqueueSVMMap(weight_ptr, weight_bytes, false);
  blas_cc->command_queue_inst_.enqueueSVMMap(w_scale_ptr, w_scale_bytes,
                                              false);
  blas_cc->command_queue_inst_.enqueueSVMMap(output_ptr, output_bytes, false);

  std::memcpy(x_q_ptr, x_q.data(), x_q_bytes);
  for (unsigned int m = 0; m < M; ++m)
    x_scale_ptr[m] = x_scale_fp16[m];
  for (size_t i = 0; i < packed_nibbles.size(); ++i)
    weight_ptr[i] = packed_nibbles[i];
  std::memset(w_scale_ptr, 0, w_scale_bytes);
  for (unsigned int n = 0; n < N; ++n)
    w_scale_ptr[n] = packed_w_scales_fp16[n];
  std::memset(output_ptr, 0, output_bytes);

  blas_cc->command_queue_inst_.enqueueSVMUnmap(x_q_ptr);
  blas_cc->command_queue_inst_.enqueueSVMUnmap(x_scale_ptr);
  blas_cc->command_queue_inst_.enqueueSVMUnmap(weight_ptr);
  blas_cc->command_queue_inst_.enqueueSVMUnmap(w_scale_ptr);
  blas_cc->command_queue_inst_.enqueueSVMUnmap(output_ptr);

  // --- Run ------------------------------------------------------------
  // Warmup (kernel compile + first-touch SVM) then timed call.
  nntrainer::gemm_int8_int4_adreno_cl(x_q_ptr, x_scale_ptr, w_scale_ptr,
                                       weight_ptr, output_ptr, M, N, K);

  const auto t0 = std::chrono::high_resolution_clock::now();
  nntrainer::gemm_int8_int4_adreno_cl(x_q_ptr, x_scale_ptr, w_scale_ptr,
                                       weight_ptr, output_ptr, M, N, K);
  const auto t1 = std::chrono::high_resolution_clock::now();
  const double gpu_ms =
    std::chrono::duration<double, std::milli>(t1 - t0).count();

  std::vector<float> gpu_out_fp32(static_cast<size_t>(M) * N, 0.0f);
  for (size_t i = 0; i < static_cast<size_t>(M) * N; ++i)
    gpu_out_fp32[i] = compute_fp16_to_fp32(output_ptr[i]);

  const float mse_err = mse<float>(ref_dst.data(), gpu_out_fp32.data(), M * N);
  const float mse_tol = GetMseToleranceDp4a(M, K, N);

  std::cout << "int8_int4_gemm_adreno M=" << M << " K=" << K << " N=" << N
            << " (DP4A, channel-wise) gpu=" << gpu_ms
            << " ms  MSE=" << mse_err << " (tol=" << mse_tol << ")"
            << std::endl;

  EXPECT_LT(mse_err, mse_tol)
    << "MSE exceeded tolerance for gemm_int8_int4_adreno M=" << M
    << " K=" << K << " N=" << N << " (mse=" << mse_err
    << " tol=" << mse_tol << ")";

  const size_t num_spot_checks =
    std::min<size_t>(16, static_cast<size_t>(M) * N);
  for (size_t i = 0; i < num_spot_checks; ++i) {
    const size_t idx = (i * 131u + 17u) % (static_cast<size_t>(M) * N);
    const float cpu = ref_dst[idx];
    const float gpu = gpu_out_fp32[idx];
    const float abs_err = std::fabs(cpu - gpu);
    const float rel_scale = std::max(std::fabs(cpu), 1.0f);
    const float abs_floor = static_cast<float>(K) * 0.03f;
    EXPECT_LT(abs_err, std::max(0.25f * rel_scale, abs_floor))
      << "Spot check failed (DP4A) at idx=" << idx
      << " (m=" << (idx / N) << ", n=" << (idx % N) << ") cpu=" << cpu
      << " gpu=" << gpu;
  }

  freeSVM(output_ptr);
  freeSVM(w_scale_ptr);
  freeSVM(weight_ptr);
  freeSVM(x_scale_ptr);
  freeSVM(x_q_ptr);
}

#define DECLARE_int8_int4_gemm_adreno_test_M_K_N(M, K, N)                      \
  TEST(nntrainer_opencl_adreno_kernels_int8_int4,                              \
       int8_int4_gemm_adreno_test_##M##_##K##_##N) {                           \
    run_gemm_int8_int4_adreno_test_(M, K, N);                                  \
  }

DECLARE_int8_int4_gemm_adreno_test_M_K_N(8, 32, 32);
DECLARE_int8_int4_gemm_adreno_test_M_K_N(8, 64, 64);
DECLARE_int8_int4_gemm_adreno_test_M_K_N(16, 128, 128);
DECLARE_int8_int4_gemm_adreno_test_M_K_N(32, 256, 256);
DECLARE_int8_int4_gemm_adreno_test_M_K_N(64, 512, 256);
DECLARE_int8_int4_gemm_adreno_test_M_K_N(128, 1024, 512);
DECLARE_int8_int4_gemm_adreno_test_M_K_N(512, 1024, 1024);

// -------------------------------------------------------------------------
// A/B benchmark: run both fp16 and DP4A paths on the same input, compare
// wall-time + TFLOPS + MSE. Uses the largest prefill shape so the GPU
// compute dominates host overhead (fp16 path is GPU-bound there; the
// DP4A speedup should show up in the wall-time ratio).
// -------------------------------------------------------------------------
static void run_ab_bench_(const unsigned int M, const unsigned int K,
                           const unsigned int N) {
  auto *blas_cc = static_cast<nntrainer::ClContext *>(
    nntrainer::Engine::Global().getRegisteredContext("gpu"));
  ASSERT_NE(blas_cc, nullptr);

  constexpr int INT4_BLOCK_N_SIZE = 32;
  const unsigned int alignN = align(N, INT4_BLOCK_N_SIZE);
  const unsigned int alignK = K; // channel-wise
  const unsigned int alignM_fp16 = align(M, 4u);
  const unsigned int alignM_dp4a = align(M, 8u);

  std::vector<float> input_orig =
    generate_random_vector<float, false>(M * K, -1.0f, 1.0f);
  std::vector<float> weight_fp32 =
    generate_random_vector<float, false>(N * K, -1.0f, 1.0f);

  // Shared weight pack (same layout for both paths).
  std::vector<uint16_t> packed_nibbles;
  std::vector<uint16_t> packed_w_scales_fp16;
  std::vector<float> dequant_weights;
  PackInt4ChannelwiseAdreno(weight_fp32.data(), N, K, packed_nibbles,
                             packed_w_scales_fp16, dequant_weights);

  // Per-row int8 activation (DP4A path uses these; fp16 path uses raw
  // fp16(input_orig)).
  std::vector<int8_t> x_q;
  std::vector<uint16_t> x_scale_fp16;
  std::vector<float> dequant_x;
  QuantizeActivationInt8(input_orig.data(), M, K, alignM_dp4a, K, x_q,
                          x_scale_fp16, dequant_x);

  // fp16 reference: input_orig * W_deq^T
  std::vector<float> ref_fp16(static_cast<size_t>(M) * N, 0.0f);
  nntrainer::sgemm(0, false, true, M, N, K, 1.0f, input_orig.data(), K,
                   dequant_weights.data(), K, 0.0f, ref_fp16.data(), N);

  // DP4A reference: dequant_x * W_deq^T
  std::vector<float> ref_dp4a(static_cast<size_t>(M) * N, 0.0f);
  nntrainer::sgemm(0, false, true, M, N, K, 1.0f, dequant_x.data(), K,
                   dequant_weights.data(), K, 0.0f, ref_dp4a.data(), N);

  // --- Allocate FP16 path SVM buffers ---------------------------------
  const size_t fp16_in_bytes =
    static_cast<size_t>(M) * alignK * sizeof(uint16_t);
  const size_t fp16_in_t_bytes =
    static_cast<size_t>(alignM_fp16) * alignK * sizeof(uint16_t);
  const size_t weight_bytes =
    static_cast<size_t>(alignK / 4u) * N * sizeof(uint16_t);
  const size_t w_scale_bytes =
    static_cast<size_t>(alignN) * sizeof(uint16_t);
  const size_t output_bytes =
    static_cast<size_t>(M) * N * sizeof(uint16_t);

  uint16_t *fp16_in = static_cast<uint16_t *>(allocateSVM(fp16_in_bytes));
  uint16_t *fp16_in_t =
    static_cast<uint16_t *>(allocateSVM(fp16_in_t_bytes));
  uint16_t *weight_ptr =
    static_cast<uint16_t *>(allocateSVM(weight_bytes));
  uint16_t *w_scale_ptr =
    static_cast<uint16_t *>(allocateSVM(w_scale_bytes));
  uint16_t *output_fp16 =
    static_cast<uint16_t *>(allocateSVM(output_bytes));

  // --- Allocate DP4A path SVM buffers ---------------------------------
  const size_t x_q_bytes =
    static_cast<size_t>(alignM_dp4a) * K * sizeof(int8_t);
  const size_t x_scale_bytes = static_cast<size_t>(M) * sizeof(uint16_t);
  int8_t *x_q_ptr = static_cast<int8_t *>(allocateSVM(x_q_bytes));
  uint16_t *x_scale_ptr =
    static_cast<uint16_t *>(allocateSVM(x_scale_bytes));
  uint16_t *output_dp4a =
    static_cast<uint16_t *>(allocateSVM(output_bytes));

  // --- Fill (fp16 path) -----------------------------------------------
  blas_cc->command_queue_inst_.enqueueSVMMap(fp16_in, fp16_in_bytes, false);
  blas_cc->command_queue_inst_.enqueueSVMMap(fp16_in_t, fp16_in_t_bytes,
                                              false);
  blas_cc->command_queue_inst_.enqueueSVMMap(weight_ptr, weight_bytes, false);
  blas_cc->command_queue_inst_.enqueueSVMMap(w_scale_ptr, w_scale_bytes,
                                              false);
  blas_cc->command_queue_inst_.enqueueSVMMap(output_fp16, output_bytes,
                                              false);
  blas_cc->command_queue_inst_.enqueueSVMMap(x_q_ptr, x_q_bytes, false);
  blas_cc->command_queue_inst_.enqueueSVMMap(x_scale_ptr, x_scale_bytes,
                                              false);
  blas_cc->command_queue_inst_.enqueueSVMMap(output_dp4a, output_bytes,
                                              false);

  for (size_t i = 0; i < static_cast<size_t>(M) * alignK; ++i)
    fp16_in[i] = compute_fp32_to_fp16(input_orig[i]);
  std::memset(fp16_in_t, 0, fp16_in_t_bytes);
  for (size_t i = 0; i < packed_nibbles.size(); ++i)
    weight_ptr[i] = packed_nibbles[i];
  std::memset(w_scale_ptr, 0, w_scale_bytes);
  for (unsigned int n = 0; n < N; ++n)
    w_scale_ptr[n] = packed_w_scales_fp16[n];
  std::memset(output_fp16, 0, output_bytes);
  std::memcpy(x_q_ptr, x_q.data(), x_q_bytes);
  for (unsigned int m = 0; m < M; ++m)
    x_scale_ptr[m] = x_scale_fp16[m];
  std::memset(output_dp4a, 0, output_bytes);

  blas_cc->command_queue_inst_.enqueueSVMUnmap(fp16_in);
  blas_cc->command_queue_inst_.enqueueSVMUnmap(fp16_in_t);
  blas_cc->command_queue_inst_.enqueueSVMUnmap(weight_ptr);
  blas_cc->command_queue_inst_.enqueueSVMUnmap(w_scale_ptr);
  blas_cc->command_queue_inst_.enqueueSVMUnmap(output_fp16);
  blas_cc->command_queue_inst_.enqueueSVMUnmap(x_q_ptr);
  blas_cc->command_queue_inst_.enqueueSVMUnmap(x_scale_ptr);
  blas_cc->command_queue_inst_.enqueueSVMUnmap(output_dp4a);

  // --- Warmup both paths (compile + first-touch) ----------------------
  nntrainer::gemm_int4_adreno_cl(fp16_in, fp16_in_t, weight_ptr, w_scale_ptr,
                                   output_fp16, M, N, K);
  nntrainer::gemm_int8_int4_adreno_cl(x_q_ptr, x_scale_ptr, w_scale_ptr,
                                       weight_ptr, output_dp4a, M, N, K);

  // --- Timed: average of N_ITER calls per path ------------------------
  constexpr int N_ITER = 10;
  double fp16_ms_sum = 0.0;
  double dp4a_ms_sum = 0.0;

  for (int it = 0; it < N_ITER; ++it) {
    const auto t0 = std::chrono::high_resolution_clock::now();
    nntrainer::gemm_int4_adreno_cl(fp16_in, fp16_in_t, weight_ptr,
                                     w_scale_ptr, output_fp16, M, N, K);
    const auto t1 = std::chrono::high_resolution_clock::now();
    fp16_ms_sum +=
      std::chrono::duration<double, std::milli>(t1 - t0).count();
  }
  for (int it = 0; it < N_ITER; ++it) {
    const auto t0 = std::chrono::high_resolution_clock::now();
    nntrainer::gemm_int8_int4_adreno_cl(x_q_ptr, x_scale_ptr, w_scale_ptr,
                                         weight_ptr, output_dp4a, M, N, K);
    const auto t1 = std::chrono::high_resolution_clock::now();
    dp4a_ms_sum +=
      std::chrono::duration<double, std::milli>(t1 - t0).count();
  }

  const double fp16_ms = fp16_ms_sum / N_ITER;
  const double dp4a_ms = dp4a_ms_sum / N_ITER;
  const double gflops = 2.0 * static_cast<double>(M) * K * N / 1.0e9;
  const double fp16_tflops = gflops / (fp16_ms * 1e-3) / 1000.0;
  const double dp4a_tflops = gflops / (dp4a_ms * 1e-3) / 1000.0;
  const double speedup = fp16_ms / dp4a_ms;

  // --- Validate DP4A output against its own dequant reference ----------
  std::vector<float> dp4a_out_fp32(static_cast<size_t>(M) * N, 0.0f);
  for (size_t i = 0; i < static_cast<size_t>(M) * N; ++i)
    dp4a_out_fp32[i] = compute_fp16_to_fp32(output_dp4a[i]);
  const float mse_dp4a =
    mse<float>(ref_dp4a.data(), dp4a_out_fp32.data(), M * N);
  const float mse_tol = GetMseToleranceDp4a(M, K, N);

  std::cout << "\n======== A/B bench M=" << M << " K=" << K << " N=" << N
            << " ========" << std::endl;
  std::cout << "  fp16 path : " << fp16_ms << " ms / iter  ("
            << fp16_tflops << " TFLOPS)" << std::endl;
  std::cout << "  dp4a path : " << dp4a_ms << " ms / iter  ("
            << dp4a_tflops << " TFLOPS)" << std::endl;
  std::cout << "  speedup   : " << speedup << "x" << std::endl;
  std::cout << "  dp4a MSE  : " << mse_dp4a << " (tol=" << mse_tol << ")"
            << std::endl;

  EXPECT_LT(mse_dp4a, mse_tol)
    << "DP4A path A/B bench MSE exceeded tolerance";

  freeSVM(output_dp4a);
  freeSVM(x_scale_ptr);
  freeSVM(x_q_ptr);
  freeSVM(output_fp16);
  freeSVM(w_scale_ptr);
  freeSVM(weight_ptr);
  freeSVM(fp16_in_t);
  freeSVM(fp16_in);
}

TEST(nntrainer_opencl_adreno_kernels_ab, ab_bench_512_1024_1024) {
  run_ab_bench_(512, 1024, 1024);
}

TEST(nntrainer_opencl_adreno_kernels_ab, ab_bench_128_1024_512) {
  run_ab_bench_(128, 1024, 512);
}

// Dump the OpenCL device extension string once at process start so we can
// tell whether the DP4A kernel's `cl_khr_integer_dot_product` fast path
// is actually enabled on this hardware. Prints to stderr (test harness
// tees stderr into the run log).
static void DumpClDeviceInfo() {
  auto *blas_cc = static_cast<nntrainer::ClContext *>(
    nntrainer::Engine::Global().getRegisteredContext("gpu"));
  if (!blas_cc) {
    std::fprintf(stderr, "[CL device] ClContext unavailable\n");
    return;
  }
  // Already-initialized ClContext holds a reference to the context
  // manager; reuse it rather than directly pulling the singleton to keep
  // this test self-contained against any include-ordering changes.
  const cl_device_id dev = blas_cc->context_inst_.GetDeviceId();
  if (!dev) {
    std::fprintf(stderr, "[CL device] device_id null\n");
    return;
  }

  auto get_str = [&](cl_device_info param) -> std::string {
    size_t len = 0;
    if (clGetDeviceInfo(dev, param, 0, nullptr, &len) != CL_SUCCESS || len == 0)
      return "(unavailable)";
    std::string s(len, '\0');
    if (clGetDeviceInfo(dev, param, len, &s[0], nullptr) != CL_SUCCESS)
      return "(query failed)";
    if (!s.empty() && s.back() == '\0')
      s.pop_back();
    return s;
  };

  const std::string name = get_str(CL_DEVICE_NAME);
  const std::string ver = get_str(CL_DEVICE_VERSION);
  const std::string drv = get_str(CL_DRIVER_VERSION);
  const std::string ext = get_str(CL_DEVICE_EXTENSIONS);

  std::fprintf(stderr, "[CL device] name=%s\n", name.c_str());
  std::fprintf(stderr, "[CL device] version=%s driver=%s\n", ver.c_str(),
               drv.c_str());

  // Grep-friendly flags for the extensions we care about.
  auto has = [&](const char *ext_name) -> bool {
    return ext.find(ext_name) != std::string::npos;
  };
  std::fprintf(stderr,
               "[CL ext flags]"
               " khr_fp16=%d"
               " khr_integer_dot_product=%d"
               " qcom_dot_product8=%d"
               " qcom_reqd_sub_group_size=%d"
               " qcom_subgroup_shuffle=%d\n",
               (int)has("cl_khr_fp16"),
               (int)has("cl_khr_integer_dot_product"),
               (int)has("cl_qcom_dot_product8"),
               (int)has("cl_qcom_reqd_sub_group_size"),
               (int)has("cl_qcom_subgroup_shuffle"));

  // Full extension string, one per line, for offline analysis. Truncated
  // if huge (shouldn't be on Adreno).
  std::fprintf(stderr, "[CL device] extensions=\n");
  size_t start = 0;
  while (start < ext.size()) {
    size_t sp = ext.find(' ', start);
    if (sp == std::string::npos)
      sp = ext.size();
    if (sp > start)
      std::fprintf(stderr, "  %.*s\n", (int)(sp - start), ext.c_str() + start);
    start = sp + 1;
  }
}

GTEST_API_ int main(int argc, char **argv) {
  int result = -1;

  try {
    testing::InitGoogleTest(&argc, argv);
  } catch (...) {
    std::cerr << "Error during InitGoogleTest" << std::endl;
    return 0;
  }

  DumpClDeviceInfo();

  try {
    result = RUN_ALL_TESTS();
  } catch (...) {
    std::cerr << "Error during RUN_ALL_TESTS()" << std::endl;
  }

  return result;
}
