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

// Run gemv_int4_adreno_v2_cl `iters` times -- baseline gemv with `input`
// promoted to __constant memory.  Same SVMMap drain contract as v1 so
// per-call wall reflects host-visible completion.
double TimeGpuGemvV2(uint16_t *input_svm, uint16_t *weight_svm,
                     uint16_t *scale_svm, uint16_t *output_svm,
                     unsigned int K, unsigned int N, unsigned int iters) {
  const auto t0 = std::chrono::high_resolution_clock::now();
  for (unsigned int i = 0; i < iters; ++i) {
    nntrainer::gemv_int4_adreno_v2_cl(input_svm, weight_svm, scale_svm,
                                       output_svm, K, N);
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

  // ---- 7b. v2 (input via __constant) timing + correctness ---------------
  // v2 differs from baseline by exactly one byte: the kernel binds
  // `input` as __constant instead of __global.  Inspired by LiteRT's
  // program_002.cl (matmul_micro_benchmark int8 capture) which routes
  // its xmem activation buffer through __constant.  Any timing delta
  // is attributable to the input-side memory path alone.
  uint16_t *v2_out_svm = (uint16_t *)allocateSVM(N * sizeof(uint16_t));
  ASSERT_NE(v2_out_svm, nullptr);
  std::memset(v2_out_svm, 0, N * sizeof(uint16_t));

  // Correctness: bit-exact vs baseline expected (math is identical).
  {
    nntrainer::gemv_int4_adreno_cl(input_svm, weight_svm, scale_svm,
                                    output_svm, K, N);
    blas_cc->command_queue_inst_.enqueueSVMMap(
      output_svm, (size_t)N * sizeof(uint16_t), /*read_only=*/true);
    nntrainer::gemv_int4_adreno_v2_cl(input_svm, weight_svm, scale_svm,
                                       v2_out_svm, K, N);
    double max_abs = 0.0;
    unsigned int big_diff = 0;
    unsigned int first_mismatch = N;
    for (unsigned int n = 0; n < N; ++n) {
      const float a = compute_fp16_to_fp32(output_svm[n]);
      const float b = compute_fp16_to_fp32(v2_out_svm[n]);
      const double d = std::fabs((double)a - (double)b);
      if (d > max_abs) max_abs = d;
      if (d > 1e-3) {
        if (first_mismatch == N) first_mismatch = n;
        big_diff++;
      }
    }
    std::cout << "[gemv_compare] K=" << K << " N=" << N
              << "  V2 CORRECTNESS  max_abs=" << max_abs
              << "  big_diff(>1e-3)=" << big_diff << "/" << N;
    if (first_mismatch < N) {
      std::cout << "  first@n=" << first_mismatch
                << "  baseline=" << compute_fp16_to_fp32(output_svm[first_mismatch])
                << "  v2=" << compute_fp16_to_fp32(v2_out_svm[first_mismatch]);
    }
    std::cout << std::endl;
  }

  // Warmup + timing.
  TimeGpuGemvV2(input_svm, weight_svm, scale_svm, v2_out_svm, K, N,
                /*iters=*/3);
  const double v2_warm =
    TimeGpuGemvV2(input_svm, weight_svm, scale_svm, v2_out_svm, K, N, 5) / 5.0;
  const unsigned int v2_iters =
    std::max(20u, static_cast<unsigned int>(500.0 / std::max(0.001, v2_warm)));
  const double v2_total =
    TimeGpuGemvV2(input_svm, weight_svm, scale_svm, v2_out_svm, K, N, v2_iters);

  const double gpu_avg = gpu_total / gpu_iters;
  const double cpu_avg = cpu_total / cpu_iters;
  const double v2_avg  = v2_total / v2_iters;
  const double ratio_gc = cpu_avg > 0.0 ? gpu_avg / cpu_avg : 0.0;
  const double ratio_vg = gpu_avg > 0.0 ? v2_avg / gpu_avg  : 0.0;

  std::cout << "[gemv_compare] K=" << K << " N=" << N
            << "  GPU=" << gpu_avg << " ms"
            << "  V2=" << v2_avg << " ms"
            << "  CPU=" << cpu_avg << " ms"
            << "  ratio(GPU/CPU)=" << ratio_gc
            << "  ratio(V2/GPU)=" << ratio_vg
            << std::endl;

  freeSVM(v2_out_svm);
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

// =====================================================================
// Phase 1A: fused RMSNorm + Q/K/V projection.  Verify correctness
// against the canonical 4-dispatch sequence (rms_norm fp32 reference
// + 3 separate gemv calls) and time the single fused dispatch.
// Saving = 3 SVMMap drains (~1.5 ms) per layer; this is the per-layer
// micro that lets us project the production gain.
// =====================================================================
namespace {
// Reference rms_norm: out = in * gamma / sqrt(mean(in^2) + eps)
// Computed in fp32 for both reference baseline and fused kernel
// internally, so the result is the same precision as the kernel.
void cpu_rmsnorm_ref(const float *in, const float *gamma, float *out,
                     unsigned int K, float eps) {
  double sumsq = 0.0;
  for (unsigned int k = 0; k < K; ++k) sumsq += (double)in[k] * (double)in[k];
  const double inv_rms = 1.0 / std::sqrt(sumsq / (double)K + (double)eps);
  for (unsigned int k = 0; k < K; ++k)
    out[k] = (float)((double)in[k] * inv_rms * (double)gamma[k]);
}
} // namespace

static void run_fused_rmsnorm_qkv_(unsigned int K_in, unsigned int N_q,
                                    unsigned int N_k, unsigned int N_v) {
  auto *blas_cc = static_cast<ClContext *>(
    Engine::Global().getRegisteredContext("gpu"));
  ASSERT_NE(blas_cc, nullptr);
  ASSERT_LE(K_in, 2560u);  // local mem cache limit in kernel

  // ---- 1. fp32 reference: random input + gamma + 3 random weight matrices --
  std::vector<float> input_fp32 =
    generate_random_vector<float>(K_in, -1.0f, 1.0f);
  std::vector<float> gamma_fp32 =
    generate_random_vector<float>(K_in, 0.5f, 1.5f);
  std::vector<float> q_w_fp32 =
    generate_random_vector<float>((size_t)N_q * K_in, -1.0f, 1.0f);
  std::vector<float> k_w_fp32 =
    generate_random_vector<float>((size_t)N_k * K_in, -1.0f, 1.0f);
  std::vector<float> v_w_fp32 =
    generate_random_vector<float>((size_t)N_v * K_in, -1.0f, 1.0f);

  const float eps = 1e-6f;

  // ---- 2. Reference using SAME int4 weights via gemv_int4_adreno_cl --
  // The fused kernel uses int4 quantized weights, so the canonical
  // reference can't be a fp32 unquantized matvec (that would diff
  // by quantization error, max_abs ~8 for K=2560 ±1 inputs).
  // Compute reference by:
  //   a) host fp32 rmsnorm
  //   b) fp32 -> fp16 SVM input
  //   c) run baseline gemv_int4_adreno_cl 3 times on the same int4
  //      weights/scales the fused kernel will use
  // Then compare element-wise.  Should be bit-exact (max_abs == 0)
  // because the per-channel int4 dequant + MAC math is identical
  // between baseline kernel and fused kernel.
  std::vector<float> norm_in(K_in);
  cpu_rmsnorm_ref(input_fp32.data(), gamma_fp32.data(), norm_in.data(), K_in,
                  eps);
  // Pack the rmsnorm-applied input as fp16 SVM for baseline gemv.
  uint16_t *norm_in_svm =
    (uint16_t *)allocateSVM(K_in * sizeof(uint16_t));
  ASSERT_NE(norm_in_svm, nullptr);
  for (unsigned int k = 0; k < K_in; ++k)
    norm_in_svm[k] = compute_fp32_to_fp16(norm_in[k]);

  // ---- 3. Pack weights/scales for GPU (channel-wise int4) ---------------
  std::vector<uint16_t> q_nibbles, q_scales, k_nibbles, k_scales,
    v_nibbles, v_scales;
  PackInt4ChannelwiseAdreno(q_w_fp32.data(), N_q, K_in, q_nibbles, q_scales);
  PackInt4ChannelwiseAdreno(k_w_fp32.data(), N_k, K_in, k_nibbles, k_scales);
  PackInt4ChannelwiseAdreno(v_w_fp32.data(), N_v, K_in, v_nibbles, v_scales);

  // ---- 4. SVM allocations -----------------------------------------------
  uint16_t *input_svm = (uint16_t *)allocateSVM(K_in * sizeof(uint16_t));
  float    *gamma_svm = (float *)allocateSVM(K_in * sizeof(float));
  uint16_t *q_w_svm = (uint16_t *)allocateSVM(q_nibbles.size() * sizeof(uint16_t));
  uint16_t *q_s_svm = (uint16_t *)allocateSVM(q_scales.size() * sizeof(uint16_t));
  uint16_t *q_out_svm = (uint16_t *)allocateSVM(N_q * sizeof(uint16_t));
  uint16_t *k_w_svm = (uint16_t *)allocateSVM(k_nibbles.size() * sizeof(uint16_t));
  uint16_t *k_s_svm = (uint16_t *)allocateSVM(k_scales.size() * sizeof(uint16_t));
  uint16_t *k_out_svm = (uint16_t *)allocateSVM(N_k * sizeof(uint16_t));
  uint16_t *v_w_svm = (uint16_t *)allocateSVM(v_nibbles.size() * sizeof(uint16_t));
  uint16_t *v_s_svm = (uint16_t *)allocateSVM(v_scales.size() * sizeof(uint16_t));
  uint16_t *v_out_svm = (uint16_t *)allocateSVM(N_v * sizeof(uint16_t));
  ASSERT_NE(input_svm, nullptr);
  ASSERT_NE(gamma_svm, nullptr);

  for (unsigned int k = 0; k < K_in; ++k) {
    input_svm[k] = compute_fp32_to_fp16(input_fp32[k]);
    gamma_svm[k] = gamma_fp32[k];
  }
  std::memcpy(q_w_svm, q_nibbles.data(), q_nibbles.size() * sizeof(uint16_t));
  std::memcpy(q_s_svm, q_scales.data(), q_scales.size() * sizeof(uint16_t));
  std::memcpy(k_w_svm, k_nibbles.data(), k_nibbles.size() * sizeof(uint16_t));
  std::memcpy(k_s_svm, k_scales.data(), k_scales.size() * sizeof(uint16_t));
  std::memcpy(v_w_svm, v_nibbles.data(), v_nibbles.size() * sizeof(uint16_t));
  std::memcpy(v_s_svm, v_scales.data(), v_scales.size() * sizeof(uint16_t));
  std::memset(q_out_svm, 0, N_q * sizeof(uint16_t));
  std::memset(k_out_svm, 0, N_k * sizeof(uint16_t));
  std::memset(v_out_svm, 0, N_v * sizeof(uint16_t));

  // ---- 5. Reference: baseline gemv_int4_adreno_cl on rmsnorm-input ----
  uint16_t *q_ref_svm = (uint16_t *)allocateSVM(N_q * sizeof(uint16_t));
  uint16_t *k_ref_svm = (uint16_t *)allocateSVM(N_k * sizeof(uint16_t));
  uint16_t *v_ref_svm = (uint16_t *)allocateSVM(N_v * sizeof(uint16_t));
  ASSERT_NE(q_ref_svm, nullptr);
  std::memset(q_ref_svm, 0, N_q * sizeof(uint16_t));
  std::memset(k_ref_svm, 0, N_k * sizeof(uint16_t));
  std::memset(v_ref_svm, 0, N_v * sizeof(uint16_t));
  nntrainer::gemv_int4_adreno_cl(norm_in_svm, q_w_svm, q_s_svm, q_ref_svm,
                                  K_in, N_q);
  nntrainer::gemv_int4_adreno_cl(norm_in_svm, k_w_svm, k_s_svm, k_ref_svm,
                                  K_in, N_k);
  nntrainer::gemv_int4_adreno_cl(norm_in_svm, v_w_svm, v_s_svm, v_ref_svm,
                                  K_in, N_v);

  // ---- 6. Dispatch fused kernel + correctness compare ----
  ASSERT_TRUE(nntrainer::fused_rmsnorm_qkv_cl(
    input_svm, gamma_svm, q_w_svm, q_s_svm, q_out_svm,
    k_w_svm, k_s_svm, k_out_svm, v_w_svm, v_s_svm, v_out_svm,
    K_in, N_q, N_k, N_v, eps));

  auto check_partition = [&](uint16_t *ref, uint16_t *out, unsigned int N,
                              const char *name) {
    double max_abs = 0.0;
    double sum_sq_diff = 0.0;
    double sum_sq_ref = 0.0;
    unsigned int big_diff = 0;
    unsigned int first_mismatch = N;
    for (unsigned int n = 0; n < N; ++n) {
      const float a = compute_fp16_to_fp32(ref[n]);
      const float b = compute_fp16_to_fp32(out[n]);
      const double d = std::fabs((double)a - (double)b);
      if (d > max_abs) max_abs = d;
      sum_sq_diff += d * d;
      sum_sq_ref += (double)a * (double)a;
      // Both paths apply identical int4 dequant + MAC; the only
      // difference is fp32 vs fp16 rmsnorm precision.  Tolerance
      // 1e-2 catches real disagreement while allowing the small
      // rmsnorm precision drift.
      if (d > 1e-2) {
        if (first_mismatch == N) first_mismatch = n;
        big_diff++;
      }
    }
    const double rel = sum_sq_ref > 0.0
      ? std::sqrt(sum_sq_diff / sum_sq_ref) : 0.0;
    std::cout << "[fused_rmsnorm_qkv] K_in=" << K_in
              << " " << name << "=" << N
              << "  max_abs=" << max_abs
              << "  rel_l2=" << rel
              << "  big_diff(>1e-2)=" << big_diff << "/" << N;
    if (first_mismatch < N) {
      std::cout << "  first@n=" << first_mismatch
                << "  ref=" << compute_fp16_to_fp32(ref[first_mismatch])
                << "  fused=" << compute_fp16_to_fp32(out[first_mismatch]);
    }
    std::cout << std::endl;
  };
  check_partition(q_ref_svm, q_out_svm, N_q, "N_q");
  check_partition(k_ref_svm, k_out_svm, N_k, "N_k");
  check_partition(v_ref_svm, v_out_svm, N_v, "N_v");

  // ---- 6. Time the fused dispatch ---------------------------------------
  auto time_fused = [&](unsigned int iters) {
    const auto t0 = std::chrono::high_resolution_clock::now();
    for (unsigned int i = 0; i < iters; ++i) {
      nntrainer::fused_rmsnorm_qkv_cl(
        input_svm, gamma_svm, q_w_svm, q_s_svm, q_out_svm,
        k_w_svm, k_s_svm, k_out_svm, v_w_svm, v_s_svm, v_out_svm,
        K_in, N_q, N_k, N_v, eps);
    }
    const auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
  };
  time_fused(3);  // warmup
  const double warm = time_fused(5) / 5.0;
  const unsigned int iters =
    std::max(20u, static_cast<unsigned int>(500.0 / std::max(0.001, warm)));
  const double total = time_fused(iters);
  const double avg = total / iters;
  std::cout << "[fused_rmsnorm_qkv] K_in=" << K_in
            << " N_q=" << N_q << " N_k=" << N_k << " N_v=" << N_v
            << "  FUSED=" << avg << " ms (replaces 1 rms_norm + "
                                      "3 gemv = ~1.4 ms baseline)"
            << std::endl;

  freeSVM(input_svm);  freeSVM(gamma_svm);  freeSVM(norm_in_svm);
  freeSVM(q_w_svm);  freeSVM(q_s_svm);  freeSVM(q_out_svm);  freeSVM(q_ref_svm);
  freeSVM(k_w_svm);  freeSVM(k_s_svm);  freeSVM(k_out_svm);  freeSVM(k_ref_svm);
  freeSVM(v_w_svm);  freeSVM(v_s_svm);  freeSVM(v_out_svm);  freeSVM(v_ref_svm);
}

TEST(nntrainer_fused_rmsnorm_qkv, qwen3_4b_shapes) {
  // Qwen3-4B: hidden 2560, num_heads_q=32 head_dim=128 -> N_q=4096
  //           num_heads_kv=8 (GQA group size 4) -> N_k=N_v=1024
  run_fused_rmsnorm_qkv_(/*K_in=*/2560, /*N_q=*/4096,
                          /*N_k=*/1024, /*N_v=*/1024);
}

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
