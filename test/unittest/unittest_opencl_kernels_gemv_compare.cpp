// SPDX-License-Identifier: Apache-2.0
/**
 * @file   unittest_opencl_kernels_gemv_compare.cpp
 * @brief  M=1 gemv comparison: GPU `gemv_int4_adreno_cl` vs CPU
 *         `gemm_q4_0` on Qwen3-4B FC shapes.  Goal is to isolate
 *         the per-call decode FC time on each path so we can see
 *         exactly where the ~5x CPU/GPU gap on decode comes from
 *         without the rest of the model pipeline noise (KV cache,
 *         norms, attention, sampling).
 *
 *         CPU side uses gemm_q4_0 with M=1 -- this is the same
 *         path HalfTensor::dotQInteger M=1 takes when the GPU
 *         path is disabled, and is what the 20 TPS CPU baseline
 *         actually exercises.
 *
 *         GPU side uses gemv_int4_adreno_cl directly, the kernel
 *         our decode hits per FC call.
 */
#include <chrono>
#include <cmath>
#include <cstring>
#include <gtest/gtest.h>
#include <iostream>
#include <vector>

#include "nntrainer_test_util.h"
#include "q4_0_utils.h"
#include <blas_kernels.h>
#include <cl_context.h>
#include <cpu_backend.h>
#include <fp16.h>

// Q4_0 block size constant.  Defined in ggml_impl_common.h (one of
// the cpu_backend internals) but that header isn't on the test
// include path.  unittest_nntrainer_cpu_backend solves this by
// re-defining it locally; do the same here.
#ifndef Q4_0
#define Q4_0 32
#endif

using namespace nntrainer;

namespace {

// True channel-wise int4 packing -- copied from
// unittest_opencl_kernels_int4_adreno.cpp because the only
// alternative (Int4Utils::quantizeAndRepackSimpleLayout) hard-
// codes group_size in {32,64,128} and we want one scale per
// output channel (group = K) to match the gpu_int4_gemv_adreno
// kernel's actual scale-load pattern.
void PackInt4ChannelwiseAdreno(const float *weights_fp32, unsigned int N,
                               unsigned int K,
                               std::vector<uint16_t> &out_nibbles,
                               std::vector<uint16_t> &out_scales_fp16) {
  ASSERT_EQ(K % 4u, 0u);
  ASSERT_EQ(N % 32u, 0u);

  out_nibbles.assign(static_cast<size_t>(K / 4) * N, 0u);
  out_scales_fp16.assign(N, 0u);

  for (unsigned int n = 0; n < N; ++n) {
    float max_abs = 0.0f;
    for (unsigned int k = 0; k < K; ++k) {
      const float v = std::fabs(weights_fp32[n * K + k]);
      if (v > max_abs) max_abs = v;
    }
    const float scale = (max_abs > 0.0f) ? (max_abs / 7.0f) : 1.0f;
    out_scales_fp16[n] = compute_fp32_to_fp16(scale);
    const float scale_rounded = compute_fp16_to_fp32(out_scales_fp16[n]);
    const float inv = (scale_rounded > 0.0f) ? (1.0f / scale_rounded) : 0.0f;
    for (unsigned int k = 0; k < K; ++k) {
      int q = static_cast<int>(std::nearbyint(weights_fp32[n * K + k] * inv));
      if (q < -8) q = -8;
      if (q > 7)  q = 7;
      const uint16_t nibble = static_cast<uint16_t>(q + 8) & 0xFu;
      const size_t dst_idx = static_cast<size_t>(k / 4) * N + n;
      out_nibbles[dst_idx] |= static_cast<uint16_t>(nibble << (4 * (k % 4)));
    }
  }
}

// Run gemv_int4_adreno_cl `iters` times on the same SVM buffers
// and return total wall ms.  Includes the helper's own blocking
// SVMMap (which is what the decode path actually pays per FC
// call).
double TimeGpuGemv(uint16_t *input_svm, uint16_t *weight_svm,
                  uint16_t *scale_svm, uint16_t *output_svm,
                  unsigned int K, unsigned int N, unsigned int iters) {
  const auto t0 = std::chrono::high_resolution_clock::now();
  for (unsigned int i = 0; i < iters; ++i) {
    nntrainer::gemv_int4_adreno_cl(input_svm, weight_svm, scale_svm, output_svm,
                                    K, N);
  }
  const auto t1 = std::chrono::high_resolution_clock::now();
  return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// Run gemm_q4_0 with M=1 `iters` times.  This is the CPU path
// HalfTensor::dotQInteger takes when the GPU short-circuit is
// disabled, and is the actual 20-TPS baseline reference.
double TimeCpuQ4_0Gemv(const float *input, const void *q4_weight, float *output,
                      unsigned int K, unsigned int N, unsigned int iters) {
  const auto t0 = std::chrono::high_resolution_clock::now();
  for (unsigned int i = 0; i < iters; ++i) {
    nntrainer::gemm_q4_0(/*M=*/1u, N, K, input, K, q4_weight, N, output, N);
  }
  const auto t1 = std::chrono::high_resolution_clock::now();
  return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// Run gemv_int4_image2d_cl `iters` times.  Pre-condition: caller has
// already published input_svm to GpuImagePool; otherwise every call
// pool-misses and the helper short-circuits (returns false) without
// dispatching, leaving a misleading timing.  We drain the queue at
// the end of the timing window with a blocking SVMMap on the SVM-out
// companion so the per-call wall reflects host-visible completion
// (matches what the production decode path's consumer fence does).
double TimeGpuGemvImage2d(uint16_t *input_svm, uint16_t *weight_svm,
                          uint16_t *scale_svm, uint16_t *output_svm,
                          unsigned int K, unsigned int N, unsigned int iters) {
  auto *blas_cc = static_cast<nntrainer::ClContext *>(
    nntrainer::Engine::Global().getRegisteredContext("gpu"));
  const auto t0 = std::chrono::high_resolution_clock::now();
  for (unsigned int i = 0; i < iters; ++i) {
    nntrainer::gemv_int4_image2d_cl(input_svm, weight_svm, scale_svm,
                                    output_svm, K, N);
  }
  if (blas_cc) {
    blas_cc->command_queue_inst_.enqueueSVMMap(
      output_svm, (size_t)N * sizeof(uint16_t), /*read_only=*/true);
  }
  const auto t1 = std::chrono::high_resolution_clock::now();
  return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

} // namespace

static void run_gemv_compare_(unsigned int K, unsigned int N) {
  auto *blas_cc = static_cast<ClContext *>(
    Engine::Global().getRegisteredContext("gpu"));
  ASSERT_NE(blas_cc, nullptr);
  ASSERT_EQ(K % 32u, 0u);
  ASSERT_EQ(N % 32u, 0u);

  // ---- 1. Reference fp32 weights and input ----
  std::vector<float> weight_fp32 =
    generate_random_vector<float>(static_cast<size_t>(N) * K, -1.0f, 1.0f);
  std::vector<float> input_fp32 = generate_random_vector<float>(K, -1.0f, 1.0f);

  // ---- 2. Pack for GPU (channel-wise int4) ----
  std::vector<uint16_t> nibble_packed;
  std::vector<uint16_t> scales_fp16;
  PackInt4ChannelwiseAdreno(weight_fp32.data(), N, K, nibble_packed,
                            scales_fp16);

  // ---- 3. Pack for CPU (q4_0 = 32-block group quant) ----
  ASSERT_EQ(K % Q4_0, 0u);
  ASSERT_EQ(N % 8u, 0u);
  const size_t q4_bytes =
    static_cast<size_t>(K) * N / Q4_0 * sizeof(block_q4_0);
  std::vector<uint8_t> q4_weight(q4_bytes);
  std::vector<uint8_t> q4_weight_repack(q4_bytes);
  nntrainer::quantize_q4_0(weight_fp32.data(), q4_weight.data(), N, K, nullptr);
  nntrainer::repack_q4_0(q4_weight_repack.data(), q4_weight.data(), q4_bytes, N,
                         K);

  // ---- 4. Allocate SVM for the GPU path ----
  uint16_t *input_svm  = (uint16_t *)allocateSVM(K * sizeof(uint16_t));
  uint16_t *weight_svm = (uint16_t *)allocateSVM(static_cast<size_t>(K / 4) * N
                                                 * sizeof(uint16_t));
  uint16_t *scale_svm  = (uint16_t *)allocateSVM(N * sizeof(uint16_t));
  uint16_t *output_svm = (uint16_t *)allocateSVM(N * sizeof(uint16_t));
  ASSERT_NE(input_svm, nullptr);
  ASSERT_NE(weight_svm, nullptr);
  ASSERT_NE(scale_svm, nullptr);
  ASSERT_NE(output_svm, nullptr);

  for (unsigned int k = 0; k < K; ++k)
    input_svm[k] = compute_fp32_to_fp16(input_fp32[k]);
  std::memcpy(weight_svm, nibble_packed.data(),
              nibble_packed.size() * sizeof(uint16_t));
  std::memcpy(scale_svm, scales_fp16.data(),
              scales_fp16.size() * sizeof(uint16_t));
  std::memset(output_svm, 0, N * sizeof(uint16_t));

  // ---- 5. CPU output buffer ----
  std::vector<float> cpu_output(N, 0.0f);

  // ---- 6. Warmup + auto-tune iter count so each path runs ~0.5 s ----
  TimeGpuGemv(input_svm, weight_svm, scale_svm, output_svm, K, N, /*iters=*/3);
  TimeCpuQ4_0Gemv(input_fp32.data(), q4_weight_repack.data(), cpu_output.data(),
                  K, N, /*iters=*/3);

  const double gpu_warm =
    TimeGpuGemv(input_svm, weight_svm, scale_svm, output_svm, K, N, 5) / 5.0;
  const double cpu_warm =
    TimeCpuQ4_0Gemv(input_fp32.data(), q4_weight_repack.data(),
                    cpu_output.data(), K, N, 5) / 5.0;
  const unsigned int gpu_iters =
    std::max(20u, static_cast<unsigned int>(500.0 / std::max(0.001, gpu_warm)));
  const unsigned int cpu_iters =
    std::max(20u, static_cast<unsigned int>(500.0 / std::max(0.001, cpu_warm)));

  // ---- 7. Timed runs ----
  const double gpu_total =
    TimeGpuGemv(input_svm, weight_svm, scale_svm, output_svm, K, N, gpu_iters);
  const double cpu_total =
    TimeCpuQ4_0Gemv(input_fp32.data(), q4_weight_repack.data(),
                    cpu_output.data(), K, N, cpu_iters);

  // ---- 7b. Image2d gemv timing ------------------------------------------
  // Publish input_svm so the helper finds it in GpuImagePool.  Use a
  // separate SVM out buffer; the kernel writes both image2d and SVM
  // (vstore4 inside the kernel) so we can drain via SVMMap.
  uint16_t *image2d_out_svm =
    (uint16_t *)allocateSVM(N * sizeof(uint16_t));
  ASSERT_NE(image2d_out_svm, nullptr);
  std::memset(image2d_out_svm, 0, N * sizeof(uint16_t));
  nntrainer::svm_to_image2d_publish(input_svm, /*M=*/1u, K);
  // Warmup
  TimeGpuGemvImage2d(input_svm, weight_svm, scale_svm, image2d_out_svm,
                     K, N, /*iters=*/3);
  const double img_warm =
    TimeGpuGemvImage2d(input_svm, weight_svm, scale_svm, image2d_out_svm,
                       K, N, 5) / 5.0;
  const unsigned int img_iters =
    std::max(20u,
              static_cast<unsigned int>(500.0 / std::max(0.001, img_warm)));
  const double img_total =
    TimeGpuGemvImage2d(input_svm, weight_svm, scale_svm, image2d_out_svm,
                       K, N, img_iters);

  const double gpu_avg = gpu_total / gpu_iters;
  const double cpu_avg = cpu_total / cpu_iters;
  const double img_avg = img_total / img_iters;
  const double ratio   = cpu_avg > 0.0 ? gpu_avg / cpu_avg : 0.0;
  const double img_vs_gpu =
    gpu_avg > 0.0 ? img_avg / gpu_avg : 0.0;

  std::cout << "[gemv_compare] K=" << K << " N=" << N
            << "  GPU=" << gpu_avg << " ms"
            << "  IMG2D=" << img_avg << " ms"
            << "  CPU=" << cpu_avg << " ms"
            << "  ratio(GPU/CPU)=" << ratio
            << "  ratio(IMG2D/GPU)=" << img_vs_gpu
            << std::endl;

  freeSVM(image2d_out_svm);
  freeSVM(input_svm);
  freeSVM(weight_svm);
  freeSVM(scale_svm);
  freeSVM(output_svm);
}

#define DECLARE_gemv_compare_K_N(K, N)                                        \
  TEST(nntrainer_gemv_compare, K_##K##_N_##N) { run_gemv_compare_(K, N); }

// Qwen3-4B FC shapes -- the actual decode path hits these once per
// transformer layer per token (252 calls total per token across 36
// layers).  The 20 TPS CPU baseline = ~50 ms/token / 252 calls
// = ~0.2 ms per FC call on average.  GPU 4 TPS = ~250 ms/token /
// 252 = ~1 ms per call.  Ratio per call ~5x, matches end-to-end.
// K & V projections share the shape (K=2560 N=1024) and so do gate &
// up (K=2560 N=9728).  Declare each unique shape once; gtest's TEST()
// macro generates a class name per (suite, test) pair so duplicate
// declarations would collide.
DECLARE_gemv_compare_K_N(2560, 4096);  // Q proj
DECLARE_gemv_compare_K_N(2560, 1024);  // K / V proj
DECLARE_gemv_compare_K_N(4096, 2560);  // O proj
DECLARE_gemv_compare_K_N(2560, 9728);  // gate / up proj
DECLARE_gemv_compare_K_N(9728, 2560);  // down proj

// Defining main() directly in this .o file is the only reliable way to
// keep gtest as the program entry point.  Without it the linker pulls
// `main` from one of the static archives (googletest_main *or*
// xgemm.o from libclblast.a, which has its own benchmark `main()`).
// Link order on ndk-build picks the CLBlast tuner main and the binary
// then runs the CLBlast m=n=k sweep instead of any gtest at all.
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
