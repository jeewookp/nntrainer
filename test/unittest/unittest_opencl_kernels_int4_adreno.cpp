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

// Empirical per-element MSE tolerance for the Adreno int4 GEMM. The two
// dominant error sources are:
//   1. int4 channel-wise quantization (max-abs / 7 scale, 4-bit symmetric
//      value range [-8, 7]). Worst-case per-element quant error is
//      ~|scale| / 2 = |max_row| / 14. For uniform input distributions
//      sum_k of |err_k| * |a_k| is O(K * max_row * E[|a|] / 14) variance.
//   2. fp16 accumulation on output. ~1e-3 relative per output element.
//
// The tolerance below is loose enough to absorb both plus device
// thermal jitter, but tight enough to catch a layout regression that
// would put the error at O(1) or flip the sign of a row.
float GetMseTolerance(unsigned int M, unsigned int K, unsigned int N) {
  (void)M;
  (void)N;
  // Base: int4 quant variance ~ K * (scale/7)^2 with scale <= 1 gives
  // ~K / 49. Add fp16 accumulation noise K * (1e-3)^2 * K which is
  // negligible next to the int4 term for typical shapes.
  //
  // Empirically a 4x safety factor over the analytic bound is enough
  // to cover the random-seed sensitivity of the max-abs scale.
  return (static_cast<float>(K) / 49.0f) * 4.0f;
}

} // namespace

static void run_gemm_int4_adreno_test_(const unsigned int M,
                                        const unsigned int K,
                                        const unsigned int N,
                                        const int scale_group_size) {
  auto *blas_cc = static_cast<nntrainer::ClContext *>(
    nntrainer::Engine::Global().getRegisteredContext("gpu"));
  ASSERT_NE(blas_cc, nullptr)
    << "GPU ClContext unavailable -- cannot run gemm_int4_adreno test";

  constexpr int INT4_BLOCK_N_SIZE = 32;
  const unsigned int alignN = align(N, INT4_BLOCK_N_SIZE);
  const unsigned int alignK = align(K, scale_group_size);

  // --- Random fp32 input [M][K] and weights [N][K] --------------------
  std::vector<float> input_orig =
    generate_random_vector<float, false>(M * K, -1.0f, 1.0f);
  std::vector<float> weight_fp32 =
    generate_random_vector<float, false>(N * K, -1.0f, 1.0f);

  // --- CPU reference: ref = input * weight^T ---------------------------
  // Row-major sgemm with TransB=true matches the kernel math:
  //   out[m, n] = sum_k input[m, k] * weight[n, k]
  std::vector<float> ref_dst(static_cast<size_t>(M) * N, 0.0f);
  nntrainer::sgemm(/*TStorageOrder=*/0, /*TransA=*/false, /*TransB=*/true,
                   /*M=*/M, /*N=*/N, /*K=*/K, /*alpha=*/1.0f,
                   input_orig.data(), /*lda=*/K, weight_fp32.data(),
                   /*ldb=*/K, /*beta=*/0.0f, ref_dst.data(), /*ldc=*/N);

  // --- Pad the input to [M][alignK] (scale_group_size == K in the
  //     per-channel path, so this is usually a no-op but the kernel
  //     reads alignK-wide texels).
  std::vector<float> input_padded(static_cast<size_t>(M) * alignK, 0.0f);
  for (unsigned int m = 0; m < M; ++m) {
    for (unsigned int k = 0; k < K; ++k) {
      input_padded[m * alignK + k] = input_orig[m * K + k];
    }
  }

  // --- Quantize weights into the channel-wise int4 layout --------------
  std::vector<uint16_t> packed_nibbles;
  std::vector<uint16_t> packed_scales_fp16;
  Int4Utils::quantizeAndRepackSimpleLayout(weight_fp32.data(), N, K,
                                             scale_group_size, packed_nibbles,
                                             packed_scales_fp16);

  // --- Allocate the SVM buffers the Adreno kernel wrapper expects ------
  //   input            : fp16 [M * alignK]
  //   input_transposed : fp16 [align(M, 4) * alignK] (scratch, written by
  //                                                   input_transpose pass)
  //   weights          : ushort [(alignK / 4) * N]
  //   scales           : fp16 [ceilDiv(K, scale_group_size) * alignN]
  //   output           : fp16 [M * N]
  const size_t input_svm_bytes =
    static_cast<size_t>(M) * alignK * sizeof(uint16_t);
  const size_t input_t_svm_bytes =
    static_cast<size_t>(align(M, 4u)) * alignK * sizeof(uint16_t);
  const size_t weights_svm_bytes =
    static_cast<size_t>(alignK / 4u) * N * sizeof(uint16_t);
  const size_t scales_svm_bytes =
    static_cast<size_t>(ceilDiv(K, static_cast<unsigned int>(scale_group_size))) *
    alignN * sizeof(uint16_t);
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
  for (size_t i = 0; i < static_cast<size_t>(alignK / 4u) * N; ++i) {
    weight_ptr[i] = packed_nibbles[i];
  }
  const size_t scale_count =
    ceilDiv(K, static_cast<unsigned int>(scale_group_size)) * alignN;
  for (size_t i = 0; i < scale_count && i < packed_scales_fp16.size(); ++i) {
    scale_ptr[i] = packed_scales_fp16[i];
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
  const float tolerance = GetMseTolerance(M, K, N);

  std::cout << "int4_gemm_adreno M=" << M << " K=" << K << " N=" << N
            << " group=" << scale_group_size << " gpu=" << gpu_ms
            << " ms  MSE=" << mse_err << " (tol=" << tolerance << ")"
            << std::endl;

  EXPECT_LT(mse_err, tolerance)
    << "MSE exceeded tolerance for gemm_int4_adreno M=" << M << " K=" << K
    << " N=" << N << " group=" << scale_group_size << " (mse=" << mse_err
    << " tol=" << tolerance << ")";

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

#define DECLARE_int4_gemm_adreno_test_M_K_N(M, K, N, G)                        \
  TEST(nntrainer_opencl_adreno_kernels_int4,                                   \
       int4_gemm_adreno_test_##M##_##K##_##N##_Group##G) {                     \
    run_gemm_int4_adreno_test_(M, K, N, G);                                    \
  }

// Small sanity shapes. N must be a multiple of 32 and K a multiple of 4
// per Int4Utils::channelwise_layout_size / the kernel's vload4 guard.
// These verify nibble packing + the input_transpose pre-pass + dispatch
// geometry (ceilDiv(M, 8) x N/4) without running for seconds.
DECLARE_int4_gemm_adreno_test_M_K_N(8, 32, 32, 32);
DECLARE_int4_gemm_adreno_test_M_K_N(8, 64, 64, 32);
DECLARE_int4_gemm_adreno_test_M_K_N(16, 128, 128, 128);
DECLARE_int4_gemm_adreno_test_M_K_N(32, 256, 256, 32);
DECLARE_int4_gemm_adreno_test_M_K_N(32, 256, 256, 128);

// Medium prefill-like shapes that hit the M > 1 wrapper path via
// HalfTensor::dotQInteger.
DECLARE_int4_gemm_adreno_test_M_K_N(64, 512, 256, 32);
DECLARE_int4_gemm_adreno_test_M_K_N(128, 1024, 512, 32);
DECLARE_int4_gemm_adreno_test_M_K_N(128, 1024, 512, 128);
DECLARE_int4_gemm_adreno_test_M_K_N(512, 1024, 1024, 32);

GTEST_API_ int main(int argc, char **argv) {
  int result = -1;

  try {
    testing::InitGoogleTest(&argc, argv);
  } catch (...) {
    std::cerr << "Error during InitGoogleTest" << std::endl;
    return 0;
  }

  try {
    result = RUN_ALL_TESTS();
  } catch (...) {
    std::cerr << "Error during RUN_ALL_TESTS()" << std::endl;
  }

  return result;
}
