// SPDX-License-Identifier: Apache-2.0
/**
 * @file   unittest_kai_q40_comparison_test.cpp
 * @date   14 January 2026
 * @brief  Unit tests comparing Q4_0 Tensor and KAI4 Tensor dot operations
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Hyeonggwon <hyeonggwon@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include "kleidiai_interface.h"

#include <gtest/gtest.h>
#include <tensor.h>
#include <kai4_tensor.h>
#include <q4_0_tensor.h>
#include <cpu_backend.h>
#include <fallback_internal.h>

#include <q4_0_utils.h>
#include <random>
#include <vector>
#include <cmath>
#include <cstring>
#include <limits>

#include <chrono>
#include <iostream>
#include <fstream>
#include <string>

#include <opencl_buffer_manager.h>
#include "nntrainer_test_util.h"
#include <blas_kernels.h>
#include <fp16.h>


using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::microseconds;
using std::chrono::milliseconds;
using std::chrono::nanoseconds;
using std::chrono::seconds;

// /**
//  * @brief Generate random FP32 vector
//  */
// template <typename T>
// static std::vector<T> generate_random_vector(size_t size, float min_val = -1.0f, float max_val = 1.0f) {
//   std::mt19937 gen(42);
//   std::uniform_real_distribution<float> dist(min_val, max_val);
//   std::vector<T> vec(size);
//   for (auto &val : vec) {
//     val = static_cast<T>(dist(gen));
//   }
//   return vec;
// }

/**
 * @brief Compute MSE between two vectors
 */
__attribute__((unused))
static float compute_mse(const std::vector<float>& a, const std::vector<float>& b) {
  if (a.size() != b.size()) return std::numeric_limits<float>::max();
  
  double sum = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    float diff = a[i] - b[i];
    sum += diff * diff;
  }
  return static_cast<float>(sum / a.size());
}




/**
 * @brief Test Kai through Tensor::dot() API (full integration)
 */
// #if defined(ENABLE_FP16) && defined(__aarch64__)
static void test_kai_tensor_dot_api(unsigned int M, unsigned int K, unsigned int N, unsigned int idx) {

  auto *blas_cc = static_cast<nntrainer::ClContext *>(
    nntrainer::Engine::Global().getRegisteredContext("gpu"));

  // 1. Generate random FP32 data
  std::vector<float> activation_fp32 = generate_random_vector<float>(M * K);
  std::vector<float> weight_fp32 = generate_random_vector<float>(N * K);
  
  #ifdef DEBUG
  for (int x = 0; x < K; x++) {
    for (int y = 0; y < N; y++) {
      printf("%7.3f ", weight_fp32[y * K + x]);
    }
    printf("\n");
  }
  printf("---------------------------\n");
  #endif

  
  // 2. Compute FP32 reference
  std::vector<float> reference_output(M * N);
  nntrainer::sgemm(0, false, true, M, N, K, 1.0f,
                   activation_fp32.data(), K,
                   weight_fp32.data(), K,
                   0.0f, reference_output.data(), N);
  
  // 3. Create FP32 activation tensor
  nntrainer::TensorDim activation_dim(1, 1, M, K, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32);
  nntrainer::Tensor activation_tensor(activation_dim);
  std::memcpy(activation_tensor.getData<float>(), activation_fp32.data(), M * K * sizeof(float));
  
  // 4. Create Kai weight tensor using QINT4 datatype (creates Kai4Tensor on ARM64)
  nntrainer::TensorDim kai_dim(1, 1, K, N, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::QINT4);
  nntrainer::Tensor kai_weight_tensor(kai_dim, false, nntrainer::Initializer::NONE, "", nntrainer::QScheme::PER_CHANNEL_AFFINE, idx);
  
  // Quantize using Kai's native channel-wise quantization
  const size_t rhs_native_size_qs4cx = static_cast<size_t>(N) * (((K + 2 - 1) / 2) * 2 / 2) * sizeof(uint8_t); //nxk
  const size_t rhs_scales_size_f32 = N * sizeof(float);

  std::vector<uint8_t> kai_quant_data(rhs_native_size_qs4cx);
  std::vector<uint8_t> kai_quant_scale(rhs_scales_size_f32);

  nntrainer::nntr_quant_qs4cx_f32(static_cast<size_t> (N), static_cast<size_t> (K), (void *)weight_fp32.data(), (void *)kai_quant_data.data(), kai_quant_scale.data(), true);


  #ifdef DEBUG
  int temp;
  float *kai_quant_scale_f32 = (float *) kai_quant_scale.data();
  for (int x = 0; x < K; x++) {
    for (int y = 0; y < N; y++) {
      if (x%2==0){
        temp = kai_quant_data[(x / 2) + y * ((K+1)/2)]%16;
      }
      else {
        temp = kai_quant_data[(x / 2) + y * ((K+1)/2)]/16;
      }
      // printf("%7.3f ", (float)(temp-8) * kai_quant_scale_f32[y]);
      printf("%3d ", temp);
    }
    printf("\n");
  }
  printf("---------------------------\n");

  for (int y = 0; y < N; y++) {
    printf("%10.5f ", kai_quant_scale_f32[y]);
  }
  printf("\n---------------------------\n");
  #endif


  
  // RHS Packing for offline-packed Kai API
  uint32_t idx_variant = idx;  // Using variant 4
  bool transB = true;

  size_t packed_size = nntr_kai_get_rhs_packed_size_qsi4cxp_qs4cxs1s0(static_cast<size_t> (N), static_cast<size_t> (K), idx_variant, transB);
  std::vector<uint8_t> kai_packed_data(packed_size);

  nntr_kai_qsi4cxp_qs4cxs1s0_rhs_pack(N, K,
                                       kai_packed_data.data(),
                                       kai_quant_data.data(),
                                       kai_quant_scale.data(),
                                       idx_variant, transB);


  // #ifdef DEBUG



  int temp2;
  int nr = 4;
  int kr = 16;
  int sr = 2;
  int block_length_in_bytes = kr / sr;
  int k_interleaved_v = 16;
  int rhs_packed_stride = nr * (((((K+31)/32)*32) / 2) + 12);





  for (int x = 0; x < K; x++) {
    for (int y = 0; y < N; y++) {
      int block_base = (y/nr) * rhs_packed_stride + (x/(2*k_interleaved_v))*(k_interleaved_v*nr);
      int block_index = ((x%k_interleaved_v)/block_length_in_bytes)*block_length_in_bytes*nr + (y%nr)*block_length_in_bytes+x%block_length_in_bytes;
      int temp = kai_packed_data[block_base+block_index];
      if ((x/k_interleaved_v)%2==0){
        temp = temp%16;
      }
      else{
        temp = temp/16;
      }
      temp = temp ^ 0x8;

      if (x%2==0){
        temp2 = kai_quant_data[(x / 2) + y * ((K+1)/2)]%16;
      }
      else {
        temp2 = kai_quant_data[(x / 2) + y * ((K+1)/2)]/16;
      }

      if (temp!=temp2){
        printf("%d %d %d %d\n",temp, temp2, x, y);
      }

    }
  }

  // #endif


  uint32_t alignK = ((K + 31) / 32) * 32;
  uint32_t alignN = ((N + 31) / 32) * 32;

  uint8_t *kai_packed_data_ptr = (uint8_t *)allocateSVM(packed_size * sizeof(uint8_t));
  uint16_t *weights = (uint16_t *)allocateSVM(K * N * sizeof(uint16_t) / 4);
  uint16_t *scales = (uint16_t *)allocateSVM((K/32) * alignN * sizeof(uint16_t));

  blas_cc->command_queue_inst_.enqueueSVMMap(kai_packed_data_ptr, packed_size * sizeof(uint8_t), false);

  for (unsigned int i = 0; i < packed_size; ++i) {
    kai_packed_data_ptr[i] = (kai_packed_data.data())[i];
  }

  blas_cc->command_queue_inst_.enqueueSVMUnmap(kai_packed_data_ptr);

  // Repack warmup
  nntrainer::repack_kai_to_adreno(kai_packed_data_ptr, weights, scales, N, K, rhs_packed_stride, 32);

  // Repack timing (50 iterations)
  const int T_REPACK = 50;
  auto t_repack0 = high_resolution_clock::now();
  for (int i = 0; i < T_REPACK; i++) {
    nntrainer::repack_kai_to_adreno(kai_packed_data_ptr, weights, scales, N, K, rhs_packed_stride, 32);
  }
  auto t_repack1 = high_resolution_clock::now();
  auto repack_time = std::chrono::duration_cast<std::chrono::milliseconds>(t_repack1 - t_repack0).count();
  std::cout << "Repack KAI->Adreno: " << repack_time / (T_REPACK * 1.0f) << " ms " << std::endl;

  // --- GPU Adreno INT4 GEMM ---

  // Allocate SVM buffers for GPU input/output
  uint16_t *input_ptr = (uint16_t *)allocateSVM(M * alignK * sizeof(uint16_t));
  uint16_t *input_transpose_ptr = (uint16_t *)allocateSVM(((M + 3) / 4 * 4) * alignK * sizeof(uint16_t));
  uint16_t *output_ptr = (uint16_t *)allocateSVM(M * N * sizeof(uint16_t));

  // Convert FP32 activation to FP16 and copy to SVM
  blas_cc->command_queue_inst_.enqueueSVMMap(input_ptr, M * alignK * sizeof(uint16_t), false);
  for (unsigned int i = 0; i < M * alignK; ++i) {
    if ((i % alignK) < K)
      input_ptr[i] = nntrainer::compute_fp32_to_fp16(activation_fp32[((i / alignK) * K) + (i % alignK)]);
    else
      input_ptr[i] = 0;
  }
  blas_cc->command_queue_inst_.enqueueSVMUnmap(input_ptr);

  // GPU warmup (repack + GEMM)
  for (unsigned int i = 0; i < 10; i++) {
    nntrainer::repack_kai_to_adreno(kai_packed_data_ptr, weights, scales, N, K, rhs_packed_stride, 32);
    nntrainer::gemm_int4_cl_adreno(input_ptr, input_transpose_ptr, weights, scales, output_ptr, M, N, K, 32);
  }

  // GPU timing: repack + GEMM together, and GEMM-only separately
  const int T_GPU = 50;
  auto t_gpu0 = high_resolution_clock::now();
  for (unsigned int i = 0; i < T_GPU; i++) {
    nntrainer::repack_kai_to_adreno(kai_packed_data_ptr, weights, scales, N, K, rhs_packed_stride, 32);
    nntrainer::gemm_int4_cl_adreno(input_ptr, input_transpose_ptr, weights, scales, output_ptr, M, N, K, 32);
  }
  auto t_gpu1 = high_resolution_clock::now();
  auto gpu_total_time = std::chrono::duration_cast<std::chrono::milliseconds>(t_gpu1 - t_gpu0).count();

  auto t_gemm0 = high_resolution_clock::now();
  for (unsigned int i = 0; i < T_GPU; i++) {
    nntrainer::gemm_int4_cl_adreno(input_ptr, input_transpose_ptr, weights, scales, output_ptr, M, N, K, 32);
  }
  auto t_gemm1 = high_resolution_clock::now();
  auto gpu_gemm_time = std::chrono::duration_cast<std::chrono::milliseconds>(t_gemm1 - t_gemm0).count();

  // Convert GPU FP16 output to FP32
  std::vector<float> gpu_output_fp32(M * N, 0.0f);
  for (unsigned int i = 0; i < M * N; ++i) {
    gpu_output_fp32[i] = nntrainer::compute_fp16_to_fp32(output_ptr[i]);
  }

  float gpu_mse = compute_mse(reference_output, gpu_output_fp32);

  std::cout << "GPU repack+GEMM  : " << gpu_total_time / (T_GPU * 1.0f) << " ms " << std::endl;
  std::cout << "GPU GEMM only    : " << gpu_gemm_time / (T_GPU * 1.0f) << " ms " << std::endl;
  std::cout << "GPU repack only  : " << repack_time / (T_REPACK * 1.0f) << " ms " << std::endl;
  std::cout << "GPU MSE vs FP32 ref: " << gpu_mse << std::endl;

  // --- CPU KAI GEMM ---
  // Allocate and set Kai tensor data with packed weights
  kai_weight_tensor.allocate();
  std::memcpy(kai_weight_tensor.getData(), kai_packed_data.data(), packed_size);

  // Run through Tensor::dot() API - goes through FloatTensor::dot() -> dotQInteger()
  nntrainer::TensorDim output_dim(1, 1, M, N, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32);
  nntrainer::Tensor kai_output_tensor(output_dim);

  for (unsigned int i = 0; i < 10; i ++){
    activation_tensor.dot(kai_weight_tensor, kai_output_tensor, false, false, 0.0f);
  }
  const int T = 50;
  auto t0 = high_resolution_clock::now();
  for (unsigned int i = 0; i < T; i ++){
    activation_tensor.dot(kai_weight_tensor, kai_output_tensor, false, false, 0.0f);
  }
  auto t1 = high_resolution_clock::now();

  auto cpu_time = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

  std::cout << "CPU KAI kernel   : " << cpu_time / (T * 1.0f) << " ms " << std::endl;

  // Compare
  std::vector<float> kai_vec(kai_output_tensor.getData<float>(),
                              kai_output_tensor.getData<float>() + M * N);
  float cpu_mse = compute_mse(reference_output, kai_vec);

  std::cout << "CPU MSE vs FP32 ref: " << cpu_mse << std::endl;
  std::cout << "Speedup (CPU/GPU): " << (cpu_time * 1.0f) / (gpu_time > 0 ? gpu_time : 1) << "x" << std::endl;
  std::cout << "-------------------------------------------" << std::endl;

  // Tolerance check
  constexpr float eps = 1e-5;
  const float tolerance_value = eps * M * K * N;

  EXPECT_LT(cpu_mse, tolerance_value)
    << "CPU Tensor::dot() API: MSE too high for M=" << M << ", K=" << K << ", N=" << N
    << ": MSE=" << cpu_mse << ", tolerance=" << tolerance_value;

  // Free SVM buffers
  freeSVM(input_ptr);
  freeSVM(input_transpose_ptr);
  freeSVM(output_ptr);
  freeSVM(kai_packed_data_ptr);
  freeSVM(weights);
  freeSVM(scales);
}
// #endif





#if defined(ENABLE_FP16) && defined(__aarch64__)


#ifdef DEBUG
TEST(KAI_acc3, GEMM_32x32x32) {
  test_kai_tensor_dot_api(32, 32, 32, 3);
}
#else


TEST(KAI_acc3, GEMM_1024x2560x1024) {
  test_kai_tensor_dot_api(1024, 2560, 1024, 3);
}


TEST(KAI_acc33, GEMM_1024x2560x4096) {
  test_kai_tensor_dot_api(1024, 2560, 4096, 3);
}


TEST(KAI_acc35, GEMM_1024x2560x9728) {
  test_kai_tensor_dot_api(1024, 2560, 9728, 3);
}


TEST(KAI_acc37, GEMM_1024x4096x2560) {
  test_kai_tensor_dot_api(1024, 4096, 2560, 3);
}


TEST(KAI_acc39, GEMM_1024x9728x2560) {
  test_kai_tensor_dot_api(1024, 9728, 2560, 3);
}

#endif







#endif




/**
 * @brief Main function for Google Test
 */
int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
