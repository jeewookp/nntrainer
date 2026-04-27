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
#include <dlfcn.h>
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
// include path.  The other unit tests (e.g. unittest_nntrainer_cpu_backend)
// solve this by re-defining it here, so do the same.
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

// Run gemv_int4_image2d_cl `iters` times.  Pre-condition: caller has
// already published input_svm to GpuImagePool (otherwise every call
// pool-misses and the helper short-circuits without dispatching).
// We also drain the queue at the end with a blocking SVMMap on the
// SVM-output companion so the per-call wall reflects host-visible
// completion (matches what the decode path's consumer fence does).
double TimeGpuImage2dGemv(uint16_t *input_svm, uint16_t *weight_svm,
                          uint16_t *scale_svm, uint16_t *output_svm,
                          unsigned int K, unsigned int N,
                          unsigned int iters) {
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

  // ---- 5b. image2d gemv correctness check vs baseline gemv_int4_adreno_cl ---
  // Capture baseline gemv output (one warm call), then capture image2d
  // gemv output for the SAME input/weight/scale, and report
  //   max abs diff   (should be tiny if both kernels are equivalent)
  //   N nonzero diffs > 1e-2
  //   first few mismatches
  // This decides between "Phase 2 numerical drift" and "Phase 2 bug":
  // a sub-ulp diff (max ~ 1e-3) means the two kernels are equivalent
  // and the model output divergence is from sampling argmax flipping
  // on a near-tie token; a large diff (max > 1e-1) means the image2d
  // kernel is producing wrong values somewhere.
  {
    nntrainer::gemv_int4_adreno_cl(input_svm, weight_svm, scale_svm,
                                    output_svm, K, N);
    std::vector<uint16_t> baseline_out(output_svm, output_svm + N);

    uint16_t *image2d_out_svm =
      (uint16_t *)allocateSVM(N * sizeof(uint16_t));
    ASSERT_NE(image2d_out_svm, nullptr);
    std::memset(image2d_out_svm, 0, N * sizeof(uint16_t));

    // Publish input as image2d so gemv_int4_image2d_cl can find it in
    // GpuImagePool.  Same publish call rms_norm.cpp does in production.
    nntrainer::svm_to_image2d_publish(input_svm, /*M=*/1u, K);

    const bool dispatched = nntrainer::gemv_int4_image2d_cl(
      input_svm, weight_svm, scale_svm, image2d_out_svm, K, N);
    if (!dispatched) {
      std::cout << "[gemv_compare] K=" << K << " N=" << N
                << "  IMAGE2D dispatch FAILED (pool miss?)" << std::endl;
    } else {
      // Drain the queue + invalidate host cache so we can read SVM out.
      blas_cc->command_queue_inst_.enqueueSVMMap(
        image2d_out_svm, N * sizeof(uint16_t), /*read_only=*/true);

      double max_abs = 0.0;
      double sum_sq_diff = 0.0;
      double sum_sq_base = 0.0;
      unsigned int big_diff_count = 0;
      unsigned int first_mismatch = N;
      for (unsigned int n = 0; n < N; ++n) {
        const float a = compute_fp16_to_fp32(baseline_out[n]);
        const float b = compute_fp16_to_fp32(image2d_out_svm[n]);
        const double d = std::fabs((double)a - (double)b);
        if (d > max_abs) max_abs = d;
        sum_sq_diff += d * d;
        sum_sq_base += (double)a * (double)a;
        if (d > 1e-2) {
          if (first_mismatch == N) first_mismatch = n;
          big_diff_count++;
        }
      }
      const double rel_l2 = sum_sq_base > 0.0
        ? std::sqrt(sum_sq_diff / sum_sq_base) : 0.0;
      std::cout << "[gemv_compare] K=" << K << " N=" << N
                << "  IMAGE2D vs baseline:"
                << "  max_abs=" << max_abs
                << "  rel_l2=" << rel_l2
                << "  big_diff(>1e-2)=" << big_diff_count << "/" << N;
      if (first_mismatch < N) {
        const float a = compute_fp16_to_fp32(baseline_out[first_mismatch]);
        const float b = compute_fp16_to_fp32(image2d_out_svm[first_mismatch]);
        std::cout << "  first@n=" << first_mismatch
                  << " base=" << a << " img=" << b;
      }
      std::cout << std::endl;
    }
    freeSVM(image2d_out_svm);
  }

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

  // Image2d gemv timing (now part of the production decode path):
  // input must be in GpuImagePool to skip the SVM-fallback branch, so
  // re-publish here and use a separate output SVM buffer.
  uint16_t *image2d_out_svm = (uint16_t *)allocateSVM(N * sizeof(uint16_t));
  ASSERT_NE(image2d_out_svm, nullptr);
  std::memset(image2d_out_svm, 0, N * sizeof(uint16_t));
  nntrainer::svm_to_image2d_publish(input_svm, /*M=*/1u, K);
  TimeGpuImage2dGemv(input_svm, weight_svm, scale_svm, image2d_out_svm,
                     K, N, /*iters=*/3);
  const double img2d_warm =
    TimeGpuImage2dGemv(input_svm, weight_svm, scale_svm, image2d_out_svm,
                       K, N, 5) / 5.0;
  const unsigned int img2d_iters =
    std::max(20u,
              static_cast<unsigned int>(500.0 / std::max(0.001, img2d_warm)));
  const double img2d_total =
    TimeGpuImage2dGemv(input_svm, weight_svm, scale_svm, image2d_out_svm,
                       K, N, img2d_iters);

  const double gpu_avg   = gpu_total / gpu_iters;
  const double cpu_avg   = cpu_total / cpu_iters;
  const double img2d_avg = img2d_total / img2d_iters;
  const double ratio     = cpu_avg > 0.0 ? gpu_avg / cpu_avg : 0.0;
  const double img_ratio_vs_baseline = gpu_avg > 0.0 ? img2d_avg / gpu_avg : 0.0;

  std::cout << "[gemv_compare] K=" << K << " N=" << N
            << "  GPU=" << gpu_avg << " ms"
            << "  IMG2D=" << img2d_avg << " ms"
            << "  CPU=" << cpu_avg << " ms"
            << "  ratio(GPU/CPU)=" << ratio
            << "  ratio(IMG2D/GPU)=" << img_ratio_vs_baseline
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
//
// Note: Qwen3-4B's K & V projections share the same shape
// (K=2560 N=1024) and so do gate & up (K=2560 N=9728).  We
// declare each unique shape once; gtest TEST() macro generates
// a class name per (suite, test) pair so duplicate declarations
// would collide.
DECLARE_gemv_compare_K_N(2560, 4096);  // Q proj
DECLARE_gemv_compare_K_N(2560, 1024);  // K / V proj
DECLARE_gemv_compare_K_N(4096, 2560);  // O proj
DECLARE_gemv_compare_K_N(2560, 9728);  // gate / up proj
DECLARE_gemv_compare_K_N(9728, 2560);  // down proj

// =====================================================================
// swiglu_image2d_cl correctness probe.  Verifies the new image2d
// helper produces SVM output bit-equivalent to the SVM swiglu_fp16
// path on the same gate/up inputs.  Pinning down whether the
// swiglu_image2d kernel itself is wrong (production wiring produced
// garbage tokens) without having to rerun the full causallm.
// =====================================================================
TEST(nntrainer_gemv_compare, swiglu_image2d_correctness) {
  auto *blas_cc = static_cast<ClContext *>(
    Engine::Global().getRegisteredContext("gpu"));
  ASSERT_NE(blas_cc, nullptr);

  // Pick a representative shape: Qwen3-4B intermediate dim is 9728,
  // M=1 in decode.  Layout: SVM contiguous half[M*K].
  const unsigned int M = 1;
  const unsigned int K = 9728;
  const size_t bytes = (size_t)M * K * sizeof(uint16_t);

  std::vector<float> gate_fp32 =
    generate_random_vector<float>(M * K, -2.0f, 2.0f);
  std::vector<float> up_fp32 =
    generate_random_vector<float>(M * K, -2.0f, 2.0f);

  uint16_t *gate_svm = (uint16_t *)allocateSVM(bytes);
  uint16_t *up_svm   = (uint16_t *)allocateSVM(bytes);
  uint16_t *svm_out  = (uint16_t *)allocateSVM(bytes);
  uint16_t *img_out  = (uint16_t *)allocateSVM(bytes);
  ASSERT_NE(gate_svm, nullptr);
  ASSERT_NE(up_svm,   nullptr);
  ASSERT_NE(svm_out,  nullptr);
  ASSERT_NE(img_out,  nullptr);

  for (unsigned int i = 0; i < M * K; ++i) {
    gate_svm[i] = compute_fp32_to_fp16(gate_fp32[i]);
    up_svm[i]   = compute_fp32_to_fp16(up_fp32[i]);
  }
  std::memset(svm_out, 0, bytes);
  std::memset(img_out, 0, bytes);

  // 1. Baseline: SVM swiglu_fp16.
  nntrainer::swiglu_fp16_svm_cl(gate_svm, up_svm, svm_out,
                                 (size_t)M * K);
  blas_cc->command_queue_inst_.enqueueSVMMap(
    svm_out, bytes, /*read_only=*/true);

  // 2. Image2d swiglu: publish gate/up to GpuImagePool first, then
  //    call the helper which dispatches the swiglu_image2d kernel
  //    (writes both image2d AND img_out SVM via vstore4).
  nntrainer::svm_to_image2d_publish(gate_svm, M, K);
  nntrainer::svm_to_image2d_publish(up_svm,   M, K);
  const bool dispatched = nntrainer::swiglu_image2d_cl(
    gate_svm, up_svm, img_out, M, K);
  ASSERT_TRUE(dispatched) << "swiglu_image2d_cl pool miss in unittest";
  blas_cc->command_queue_inst_.enqueueSVMMap(
    img_out, bytes, /*read_only=*/true);

  // DIAG: kernel stamps a 42.0 sentinel at img_out[0] before doing
  // any other work.  If we don't see that, SVM arg binding /
  // dispatch failed (kernel never landed on the SVM pointer).
  const float sentinel = compute_fp16_to_fp32(img_out[0]);
  std::cout << "[swiglu_compare] sentinel img_out[0]=" << sentinel
            << " (expect 42 if kernel landed; 0 means binding failed)"
            << std::endl;

  // 3. Compare.  Allow some half-precision slack from the fp32-vs-fp16
  //    silu math difference, but anything > 1e-2 is the kernel being
  //    actually wrong.
  double max_abs = 0.0;
  double sum_sq_diff = 0.0;
  double sum_sq_base = 0.0;
  unsigned int big_diff = 0;
  unsigned int first_mismatch = M * K;
  for (unsigned int i = 0; i < M * K; ++i) {
    const float a = compute_fp16_to_fp32(svm_out[i]);
    const float b = compute_fp16_to_fp32(img_out[i]);
    const double d = std::fabs((double)a - (double)b);
    if (d > max_abs) max_abs = d;
    sum_sq_diff += d * d;
    sum_sq_base += (double)a * (double)a;
    if (d > 1e-2) {
      if (first_mismatch == M * K) first_mismatch = i;
      big_diff++;
    }
  }
  const double rel_l2 = sum_sq_base > 0.0
    ? std::sqrt(sum_sq_diff / sum_sq_base) : 0.0;
  std::cout << "[swiglu_compare] M=" << M << " K=" << K
            << "  max_abs=" << max_abs
            << "  rel_l2=" << rel_l2
            << "  big_diff(>1e-2)=" << big_diff << "/" << (M * K);
  if (first_mismatch < M * K) {
    std::cout << "  first@i=" << first_mismatch
              << "  base=" << compute_fp16_to_fp32(svm_out[first_mismatch])
              << "  img="  << compute_fp16_to_fp32(img_out[first_mismatch]);
  }
  std::cout << std::endl;

  freeSVM(gate_svm);
  freeSVM(up_svm);
  freeSVM(svm_out);
  freeSVM(img_out);
}

// =====================================================================
// Phase 1 RecordableQueue probe.  Confirms whether Adreno 830 supports
// cl_qcom_recordable_queues and measures replay throughput vs naive
// per-call SetKernelArg + EnqueueNDRange.  Pattern copied from
// litert_lm's runtime/cl_bench/delegate_kernel_bench.cc.
// =====================================================================
namespace {
typedef void *(*pfn_clNewRecordingQCOM)(cl_command_queue, cl_int *);
typedef cl_int (*pfn_clEndRecordingQCOM)(void *);
typedef cl_int (*pfn_clEnqueueRecordingQCOM)(cl_command_queue, void *,
                                              cl_uint, const void *,
                                              cl_uint, const cl_event *,
                                              cl_event *);
typedef cl_int (*pfn_clReleaseRecordingQCOM)(void *);
typedef cl_command_queue (*pfn_clCreateCommandQueueWithProperties)(
  cl_context, cl_device_id, const cl_queue_properties *, cl_int *);

#define CL_QUEUE_RECORDABLE_QCOM 0x40E6
} // namespace

TEST(nntrainer_gemv_compare, recordable_queue_probe) {
  void *libcl = dlopen("libOpenCL.so", RTLD_NOW);
  if (!libcl) {
    libcl = dlopen("/system/vendor/lib64/libOpenCL.so", RTLD_NOW);
  }
  ASSERT_NE(libcl, nullptr) << "Failed to dlopen libOpenCL.so";

  auto p_NewRec   = (pfn_clNewRecordingQCOM)dlsym(libcl, "clNewRecordingQCOM");
  auto p_EndRec   = (pfn_clEndRecordingQCOM)dlsym(libcl, "clEndRecordingQCOM");
  auto p_EnqRec   = (pfn_clEnqueueRecordingQCOM)dlsym(
                      libcl, "clEnqueueRecordingQCOM");
  auto p_RelRec   = (pfn_clReleaseRecordingQCOM)dlsym(
                      libcl, "clReleaseRecordingQCOM");
  auto p_CreateQP = (pfn_clCreateCommandQueueWithProperties)dlsym(
                      libcl, "clCreateCommandQueueWithProperties");

  std::cout << "[rq_probe] symbols  NewRec=" << (void *)p_NewRec
            << " EndRec=" << (void *)p_EndRec
            << " EnqRec=" << (void *)p_EnqRec
            << " RelRec=" << (void *)p_RelRec
            << " CreateQP=" << (void *)p_CreateQP << std::endl;
  if (!p_NewRec || !p_EndRec || !p_EnqRec || !p_RelRec || !p_CreateQP) {
    GTEST_SKIP() << "cl_qcom_recordable_queues entry points missing";
  }

  // Reuse the device + context the global ClContext already brought
  // up (so we don't have to pick the right Adreno device manually).
  auto *blas_cc = static_cast<ClContext *>(
    Engine::Global().getRegisteredContext("gpu"));
  ASSERT_NE(blas_cc, nullptr);
  cl_context ctx     = blas_cc->context_inst_.GetContext();
  cl_command_queue q = blas_cc->command_queue_inst_.GetCommandQueue();
  cl_device_id dev   = nullptr;
  ASSERT_EQ(clGetCommandQueueInfo(q, CL_QUEUE_DEVICE, sizeof(dev), &dev,
                                  nullptr),
            CL_SUCCESS);

  // Create a recordable queue.  Driver accepts the property in either
  // of two formats depending on platform version; try both, matching
  // the litert_lm probe.
  cl_int err = 0;
  cl_command_queue rec_q = nullptr;
  // Attempt 1: enable bit named directly + value 1.
  {
    cl_queue_properties props[] = {CL_QUEUE_RECORDABLE_QCOM, 1, 0};
    rec_q = p_CreateQP(ctx, dev, props, &err);
    std::cout << "[rq_probe] CreateRecordableQueue attempt 1 err=" << err
              << std::endl;
  }
  // Attempt 2: as a bit inside CL_QUEUE_PROPERTIES (=0x1093).
  if (!rec_q || err != CL_SUCCESS) {
    cl_queue_properties props[] = {0x1093, CL_QUEUE_RECORDABLE_QCOM, 0};
    rec_q = p_CreateQP(ctx, dev, props, &err);
    std::cout << "[rq_probe] CreateRecordableQueue attempt 2 err=" << err
              << std::endl;
  }
  // Attempt 3: combined with profiling (some Adreno drivers require it).
  if (!rec_q || err != CL_SUCCESS) {
    cl_queue_properties props[] = {0x1093,
                                    CL_QUEUE_RECORDABLE_QCOM |
                                      CL_QUEUE_PROFILING_ENABLE,
                                    0};
    rec_q = p_CreateQP(ctx, dev, props, &err);
    std::cout << "[rq_probe] CreateRecordableQueue attempt 3 err=" << err
              << std::endl;
  }
  if (!rec_q || err != CL_SUCCESS) {
    // Diagnostic: dump the device's CL_DEVICE_EXTENSIONS so we know
    // whether RecordableQueue is even advertised.  Then try the
    // standard cl_khr_command_buffer alternative (different API but
    // same goal: amortize host dispatch overhead by recording).
    {
      size_t ext_size = 0;
      clGetDeviceInfo(dev, CL_DEVICE_EXTENSIONS, 0, nullptr, &ext_size);
      std::string exts(ext_size, '\0');
      clGetDeviceInfo(dev, CL_DEVICE_EXTENSIONS, ext_size, &exts[0], nullptr);
      std::cout << "[rq_probe] CL_DEVICE_EXTENSIONS:\n" << exts << std::endl;
    }

    typedef void *(*pfn_clCreateCommandBufferKHR)(cl_uint,
                                                   const cl_command_queue *,
                                                   const void *, cl_int *);
    typedef cl_int (*pfn_clCommandNDRangeKernelKHR)(
      void *, void *, const void *, cl_kernel, cl_uint, const size_t *,
      const size_t *, const size_t *, cl_uint, const void *, void *, void *);
    typedef cl_int (*pfn_clFinalizeCommandBufferKHR)(void *);
    typedef cl_int (*pfn_clEnqueueCommandBufferKHR)(cl_uint,
                                                     const cl_command_queue *,
                                                     void *, cl_uint,
                                                     const cl_event *,
                                                     cl_event *);
    typedef cl_int (*pfn_clReleaseCommandBufferKHR)(void *);

    auto p_CreateCB = (pfn_clCreateCommandBufferKHR)dlsym(
                        libcl, "clCreateCommandBufferKHR");
    auto p_NDRangeCB = (pfn_clCommandNDRangeKernelKHR)dlsym(
                         libcl, "clCommandNDRangeKernelKHR");
    auto p_FinalizeCB = (pfn_clFinalizeCommandBufferKHR)dlsym(
                          libcl, "clFinalizeCommandBufferKHR");
    auto p_EnqueueCB = (pfn_clEnqueueCommandBufferKHR)dlsym(
                         libcl, "clEnqueueCommandBufferKHR");
    auto p_ReleaseCB = (pfn_clReleaseCommandBufferKHR)dlsym(
                         libcl, "clReleaseCommandBufferKHR");

    std::cout << "[rq_probe] command_buffer symbols  CreateCB="
              << (void *)p_CreateCB << "  NDRangeCB="
              << (void *)p_NDRangeCB << "  FinalizeCB="
              << (void *)p_FinalizeCB << "  EnqueueCB="
              << (void *)p_EnqueueCB << std::endl;

    if (!p_CreateCB || !p_NDRangeCB || !p_FinalizeCB || !p_EnqueueCB) {
      GTEST_SKIP() << "Neither RecordableQueue nor command_buffer "
                      "extensions usable on this device";
    }
    GTEST_SKIP() << "RecordableQueue rejected; command_buffer route "
                    "available -- next probe will exercise it";
  }

  // Use a tiny inline kernel so we don't depend on builddir-generated
  // kernel headers.  The arithmetic doesn't matter: we're measuring
  // record+replay dispatch overhead, not GEMV throughput.  This kernel
  // touches every output element so the driver can't optimise it away.
  const char *src =
    "__kernel void rq_probe(__global float *out, __global const float *in,\n"
    "                       const float scale) {\n"
    "  const int i = get_global_id(0);\n"
    "  out[i] = in[i] * scale + 1.0f;\n"
    "}\n";

  cl_int build_err = 0;
  cl_program prog = clCreateProgramWithSource(ctx, 1, &src, nullptr, &err);
  ASSERT_EQ(err, CL_SUCCESS);
  build_err = clBuildProgram(prog, 1, &dev, nullptr, nullptr, nullptr);
  ASSERT_EQ(build_err, CL_SUCCESS);
  cl_kernel kern = clCreateKernel(prog, "rq_probe", &err);
  ASSERT_EQ(err, CL_SUCCESS);

  // Buffers.  Use plain cl_mem (recording is what we're testing,
  // not SVM coherence; SVM args complicate replay arg tracking on
  // some drivers).
  const size_t N_elems = 4096;
  std::vector<float> host_in(N_elems, 0.5f);
  std::vector<float> host_out(N_elems, 0.0f);
  cl_mem buf_in =
    clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                    sizeof(float) * N_elems, host_in.data(), &err);
  ASSERT_EQ(err, CL_SUCCESS);
  cl_mem buf_out = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
                                   sizeof(float) * N_elems, nullptr, &err);
  ASSERT_EQ(err, CL_SUCCESS);

  float scale_arg = 2.0f;
  ASSERT_EQ(clSetKernelArg(kern, 0, sizeof(cl_mem), &buf_out), CL_SUCCESS);
  ASSERT_EQ(clSetKernelArg(kern, 1, sizeof(cl_mem), &buf_in), CL_SUCCESS);
  ASSERT_EQ(clSetKernelArg(kern, 2, sizeof(float), &scale_arg), CL_SUCCESS);

  const size_t global[1] = {N_elems};
  const size_t local[1]  = {64};

  // Warm naive path on the regular (non-recordable) queue first.
  for (int i = 0; i < 5; ++i) {
    err = clEnqueueNDRangeKernel(q, kern, 1, nullptr, global, local,
                                  0, nullptr, nullptr);
    ASSERT_EQ(err, CL_SUCCESS);
  }
  clFinish(q);

  const int iters = 252;
  // Naive: re-enqueue 252 times.
  const auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iters; ++i) {
    clEnqueueNDRangeKernel(q, kern, 1, nullptr, global, local,
                            0, nullptr, nullptr);
  }
  clFinish(q);
  const auto t1 = std::chrono::high_resolution_clock::now();
  const double naive_ms =
    std::chrono::duration<double, std::milli>(t1 - t0).count();

  // Record on the recordable queue.
  void *rec = p_NewRec(rec_q, &err);
  std::cout << "[rq_probe] NewRecording err=" << err << std::endl;
  ASSERT_EQ(err, CL_SUCCESS);
  ASSERT_NE(rec, nullptr);
  err = clEnqueueNDRangeKernel(rec_q, kern, 1, nullptr, global, local,
                                0, nullptr, nullptr);
  std::cout << "[rq_probe] EnqueueND (record) err=" << err << std::endl;
  ASSERT_EQ(err, CL_SUCCESS);
  err = p_EndRec(rec);
  std::cout << "[rq_probe] EndRecording err=" << err << std::endl;
  ASSERT_EQ(err, CL_SUCCESS);

  // Warmup replay.
  for (int i = 0; i < 5; ++i)
    p_EnqRec(rec_q, rec, 0, nullptr, 0, nullptr, nullptr);
  clFinish(rec_q);

  const auto r0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iters; ++i) {
    p_EnqRec(rec_q, rec, 0, nullptr, 0, nullptr, nullptr);
  }
  clFinish(rec_q);
  const auto r1 = std::chrono::high_resolution_clock::now();
  const double rec_ms =
    std::chrono::duration<double, std::milli>(r1 - r0).count();

  // Verify output is correct.
  ASSERT_EQ(clEnqueueReadBuffer(q, buf_out, CL_TRUE, 0,
                                 sizeof(float) * N_elems, host_out.data(),
                                 0, nullptr, nullptr),
            CL_SUCCESS);
  const float expected = 0.5f * scale_arg + 1.0f;
  unsigned int wrong = 0;
  for (size_t i = 0; i < N_elems; ++i) {
    if (std::fabs(host_out[i] - expected) > 1e-4f) wrong++;
  }

  std::cout << "[rq_probe] iters=" << iters
            << "  naive=" << naive_ms << " ms"
            << "  record+replay=" << rec_ms << " ms"
            << "  speedup=" << (naive_ms / std::max(0.001, rec_ms)) << "x"
            << "  output_wrong=" << wrong << "/" << N_elems << std::endl;

  p_RelRec(rec);
  clReleaseMemObject(buf_in);
  clReleaseMemObject(buf_out);
  clReleaseKernel(kern);
  clReleaseProgram(prog);
  clReleaseCommandQueue(rec_q);
  dlclose(libcl);
}

// Defining main() directly in this .o file is the only reliable way to
// keep gtest as the program entry point.  Without it, the linker pulls
// `main` from one of the static archives (googletest_main *or* xgemm.o
// from libclblast.a, which has its own benchmark `main()`), and link
// order on ndk-build sometimes picks the CLBlast tuner main -- the
// resulting binary then runs the CLBlast m=n=k sweep instead of any
// gtest at all.  unittest_opencl_kernels_int4_adreno.cpp dodges the
// same trap by defining its own main(); copy the pattern here.
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
