// SPDX-License-Identifier: Apache-2.0
/**
 * @file   unittest_kai_q40_comparison.cpp
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

using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::microseconds;
using std::chrono::milliseconds;
using std::chrono::nanoseconds;
using std::chrono::seconds;

#define QK4_0 32

#define N_K 8

/**
 * @brief FP16 to float conversion (using memcpy to avoid strict-aliasing warnings)
 */
static inline float fp16_to_fp32(uint16_t h) {
  uint32_t sign = (h & 0x8000) << 16;
  uint32_t exponent = (h & 0x7C00) >> 10;
  uint32_t mantissa = (h & 0x03FF);

  if (exponent == 0) {
    if (mantissa == 0) {
      uint32_t result = sign;
      float f;
      std::memcpy(&f, &result, sizeof(float));
      return f;
    } else {
      exponent = 1;
      while ((mantissa & 0x0400) == 0) {
        mantissa <<= 1;
        exponent--;
      }
      mantissa &= 0x03FF;
    }
  } else if (exponent == 0x1F) {
    uint32_t result = sign | 0x7F800000 | (mantissa << 13);
    float f;
    std::memcpy(&f, &result, sizeof(float));
    return f;
  }

  uint32_t result = sign | ((exponent + 112) << 23) | (mantissa << 13);
  float f;
  std::memcpy(&f, &result, sizeof(float));
  return f;
}

/**
 * @brief Generate random FP32 vector
 */
template <typename T>
static std::vector<T> generate_random_vector(size_t size, float min_val = -1.0f, float max_val = 1.0f) {
  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(min_val, max_val);
  std::vector<T> vec(size);
  for (auto &val : vec) {
    val = static_cast<T>(dist(gen));
  }
  return vec;
}

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




// ============================================================================
// Tensor::dot() API Integration Tests - Validate full API path, not just kernels
// ============================================================================

/**
 * @brief Test Q4_0 through Tensor::dot() API (full integration)
 * This validates FloatTensor::dot() -> dotQInteger() -> Kai/Q4_0 kernels
 */
static void test_q40_tensor_dot_api(unsigned int M, unsigned int K, unsigned int N) {
  // 1. Generate random FP32 data
  std::vector<float> activation_fp32 = generate_random_vector<float>(M * K);
  std::vector<float> weight_fp32 = generate_random_vector<float>(N * K);
  
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
  
  // 4. Create FP32 weight tensor
  nntrainer::TensorDim weight_dim(1, 1, K, N, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32);
  nntrainer::Tensor weight_tensor_fp32(weight_dim);
  std::memcpy(weight_tensor_fp32.getData<float>(), weight_fp32.data(), N * K * sizeof(float));
  
  // 5. Create Q4_0 weight tensor by converting from FP32
  // This tests the full Tensor API quantization path
  nntrainer::TensorDim q4_0_dim(1, 1, K, N, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::Q4_0);
  nntrainer::Tensor q4_0_weight_tensor(q4_0_dim);
  
  // Quantize using low-level API (since Tensor::copyData doesn't support QINT4)
  const size_t block_size = 32;
  size_t num_blocks = (N * K) / block_size;
  size_t q4_0_size = num_blocks * sizeof(block_q4_0);
  std::vector<uint8_t> q4_0_data(q4_0_size);
  nntrainer::quantize_q4_0(weight_fp32.data(), (void *)q4_0_data.data(), N, K, nullptr);
  std::vector<uint8_t> q4_0_repacked(q4_0_size);
  nntrainer::repack_q4_0(q4_0_data.data(), q4_0_repacked.data(), q4_0_size, N, K);
  
  // Allocate and set Q4_0 tensor data
  q4_0_weight_tensor.allocate();
  std::memcpy(q4_0_weight_tensor.getData(), q4_0_repacked.data(), q4_0_size);
  
  // 6. Run through Tensor::dot() API - this goes through FloatTensor::dot() -> dotQInteger()
  nntrainer::TensorDim output_dim(1, 1, M, N, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32);
  nntrainer::Tensor q4_0_output_tensor(output_dim);
  q4_0_output_tensor.allocate();
  activation_tensor.dot(q4_0_weight_tensor, q4_0_output_tensor, false, false, 0.0f);
 
  // 7. Compare Q4_0 Tensor::dot() output vs FP32 reference
  std::vector<float> q4_0_vec(q4_0_output_tensor.getData<float>(), 
                               q4_0_output_tensor.getData<float>() + M * N);
  float mse = compute_mse(reference_output, q4_0_vec);
  
  // Same tolerance as kernel tests
  const float base_eps = 1e-5;
  const float tolerance = base_eps * M * K * N;
  
  EXPECT_LT(mse, tolerance) 
    << "Tensor::dot() API: MSE too high for M=" << M << ", K=" << K << ", N=" << N
    << ": MSE=" << mse << ", tolerance=" << tolerance;
}

/**
 * @brief Test Kai through Tensor::dot() API (full integration)
 */
#if defined(ENABLE_FP16) && defined(__aarch64__)
static void test_kai_tensor_dot_api(unsigned int M, unsigned int K, unsigned int N, unsigned int idx) {
  // 1. Generate random FP32 data
  std::vector<float> activation_fp32 = generate_random_vector<float>(M * K);
  std::vector<float> weight_fp32 = generate_random_vector<float>(N * K);
  
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

  // Allocate and set Kai tensor data with packed weights
  kai_weight_tensor.allocate();
  std::memcpy(kai_weight_tensor.getData(), kai_packed_data.data(), packed_size);
  
  // 5. Run through Tensor::dot() API - goes through FloatTensor::dot() -> dotQInteger()
  nntrainer::TensorDim output_dim(1, 1, M, N, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32);
  nntrainer::Tensor kai_output_tensor(output_dim);
  activation_tensor.dot(kai_weight_tensor, kai_output_tensor, false, false, 0.0f);
  
  // 6. Compare Kai Tensor::dot() output vs FP32 reference
  std::vector<float> kai_vec(kai_output_tensor.getData<float>(), 
                              kai_output_tensor.getData<float>() + M * N);
  float mse = compute_mse(reference_output, kai_vec);
  
  // Same tolerance as kernel tests
  constexpr float eps = 1e-5;
  const float tolerance = eps * M * K * N;
  
  EXPECT_LT(mse, tolerance) 
    << "Tensor::dot() API: MSE too high for M=" << M << ", K=" << K << ", N=" << N
    << ": MSE=" << mse << ", tolerance=" << tolerance;
}
#endif

static void test_q40_vs_kai(unsigned int M, unsigned int K, unsigned int N) {

  const int T = 50; // T iterations
  
  std::string uname[8] = {"matmul_clamp_f32_qai8dxp1x8_qsi4cxp4x8_1x4x32_neon_dotprod", "matmul_clamp_f32_qai8dxp1x8_qsi4cxp8x8_1x8x32_neon_dotprod", "matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_4x4x32_neon_i8mm", "matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_8x4x32_neon_i8mm", "matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_4x8x32_neon_i8mm", "matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm", "matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x4_8x8x32_neon_dotprod", "matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x4_16x4x32_neon_dotprod"};
  // 1. Generate random FP32 data 
  std::vector<float> activation_fp32 = generate_random_vector<float>(M * K * T);
  std::vector<float> weight_fp32 = generate_random_vector<float>(N * K * T);

  // 2. Declare FP32 activation tensor, wegith tensor (both kai and q4_0)
  nntrainer::TensorDim activation_dim(1, 1, M, K, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32);
  nntrainer::Tensor activation_tensor(activation_dim);
  activation_tensor.allocate();
  
  nntrainer::TensorDim kai_dim(1, 1, K, N, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::QINT4);


  nntrainer::TensorDim q4_0_dim(1, 1, K, N, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::Q4_0);
  nntrainer::Tensor q4_0_weight_tensor(q4_0_dim);
  q4_0_weight_tensor.allocate();

  // 3. Declare quantization & packing related ones
  //const size_t num_blocks = K;
  //const size_t bytes_per_block = sizeof(uint16_t) + N / 2;

  
  size_t packed_size;
  uint32_t idx_variant;
  bool transB = true;

  const size_t block_size = 32;
  const size_t q4_0_num_blocks = (N * K) / block_size;
  size_t q4_0_size = q4_0_num_blocks * sizeof(block_q4_0);
  std::vector<uint8_t> q4_0_data(q4_0_size);
  std::vector<uint8_t> q4_0_repacked(q4_0_size);


  nntrainer::TensorDim output_dim(1, 1, M, N, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32);
  nntrainer::TensorDim output_dim2(1, 1, M, N, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32);
  nntrainer::Tensor kai_output_tensor(output_dim);
  nntrainer::Tensor q4_0_output_tensor(output_dim2);

  kai_output_tensor.allocate();
  q4_0_output_tensor.allocate();
  

  const size_t rhs_native_size_qs4cx = static_cast<size_t>(N) * (((K + 2 - 1) / 2) * 2 / 2) * sizeof(uint8_t); //nxk
  const size_t rhs_scales_size_f32 = N * sizeof(float);

  std::vector<uint8_t> kai_quant_data(rhs_native_size_qs4cx);
  std::vector<uint8_t> kai_quant_scale(rhs_scales_size_f32);

  microseconds execution_time{};

  
  for (unsigned int j = 0; j < N_K; j++){
    // j-th ukernel (KAI)
    idx_variant = j;
    nntrainer::Tensor kai_weight_tensor(kai_dim, false, nntrainer::Initializer::NONE, "", nntrainer::QScheme::PER_CHANNEL_AFFINE, idx_variant);
    kai_weight_tensor.allocate();

    packed_size = nntr_kai_get_rhs_packed_size_qsi4cxp_qs4cxs1s0(N, K, idx_variant, transB);

    std::vector<uint8_t> kai_packed_data(packed_size);

    // Copy i-th acvitvation chunk
    std::memcpy(activation_tensor.getData<float>(), activation_fp32.data(), M * K * sizeof(float));
    // 4. Create Kai weight tensor using QINT4 datatype (creates Kai4Tensor on ARM64)
    
    nntrainer::nntr_quant_qs4cx_f32(N, K, weight_fp32.data(), kai_quant_data.data(), kai_quant_scale.data());

    nntr_kai_qsi4cxp_qs4cxs1s0_rhs_pack(N, K,
                                        kai_packed_data.data(),
                                        kai_quant_data.data(),
                                        kai_quant_scale.data(),
                                        idx_variant, transB);                                        

    // Allocate and set Kai tensor data with packed weights

    
    
    std::memcpy(kai_weight_tensor.getData(), kai_packed_data.data(), packed_size);
    
   
      // i-th iteraiton
      
      
      
      // 4. Run through Tensor::dot() API - goes through FloatTensor::dot() -> dotQInteger()
    auto t0 = high_resolution_clock::now();
    for (unsigned int i = 0; i < T; i ++){
      activation_tensor.dot(kai_weight_tensor, kai_output_tensor, false, false, 0.0f);
    }
    auto t1 = high_resolution_clock::now(); 

    execution_time = duration_cast<microseconds>(t1 - t0);
      
    std::cout << "QINT4 kernel " << uname[j] << " : " << execution_time.count()/T << " ms " << std::endl;
      
    
  }

  // Quantize using low-level API (since Tensor::copyData doesn't support QINT4)
  std::memcpy(activation_tensor.getData<float>(), activation_fp32.data(), M * K * sizeof(float));

  nntrainer::quantize_q4_0(weight_fp32.data(), (void *)q4_0_data.data(), K, N, nullptr);
  nntrainer::repack_q4_0(q4_0_data.data(), q4_0_repacked.data(), q4_0_size, K, N);
  
  
  std::memcpy(q4_0_weight_tensor.getData(), q4_0_repacked.data(), q4_0_size);

 
    
    
    
    
    
  auto t0 = high_resolution_clock::now();
  for (unsigned int i = 0; i < T; i ++){
    activation_tensor.dot(q4_0_weight_tensor, q4_0_output_tensor, false, false, 0.0f);
  }
  auto t1 = high_resolution_clock::now(); 

  execution_time = duration_cast<microseconds>(t1 - t0);

  
  std::cout << "Q40 kernel : " << execution_time.count()/T << " ms " << std::endl;
    
  
 
}




#if defined(ENABLE_FP16) && defined(__aarch64__)
TEST(Q40_acc, GEMM_1024x2560x1024) {
  test_q40_tensor_dot_api(1024, 2560, 1024);
}

TEST(KAI_acc0, GEMM_1024x2560x1024) {
  test_kai_tensor_dot_api(1024, 2560, 1024, 0);
}

TEST(KAI_acc1, GEMM_1024x2560x1024) {
  test_kai_tensor_dot_api(1024, 2560, 1024, 1);
}

TEST(KAI_acc2, GEMM_1024x2560x1024) {
  test_kai_tensor_dot_api(1024, 2560, 1024, 2);
}

TEST(KAI_acc3, GEMM_1024x2560x1024) {
  test_kai_tensor_dot_api(1024, 2560, 1024, 3);
}

TEST(KAI_acc4, GEMM_1024x2560x1024) {
  test_kai_tensor_dot_api(1024, 2560, 1024, 4);
}

TEST(KAI_acc5, GEMM_1024x2560x1024) {
  test_kai_tensor_dot_api(1024, 2560, 1024, 5);
}

TEST(KAI_acc6, GEMM_1024x2560x1024) {
  test_kai_tensor_dot_api(1024, 2560, 1024, 6);
}

TEST(KAI_acc7, GEMM_1024x2560x1024) {
  test_kai_tensor_dot_api(1024, 2560, 1024, 7);
}





TEST(Q40_acc2, GEMV_1x2560x1024) {
  test_q40_tensor_dot_api(1, 2560, 1024);
}

TEST(KAI_acc02, GEMV_1x2560x1024) {
  test_kai_tensor_dot_api(1, 2560, 1024, 0);
}

TEST(KAI_acc12, GEMV_1x2560x1024) {
  test_kai_tensor_dot_api(1, 2560, 1024, 1);
}

TEST(KAI_acc22, GEMV_1x2560x1024) {
  test_kai_tensor_dot_api(1, 2560, 1024, 2);
}

TEST(KAI_acc32, GEMV_1x2560x1024) {
  test_kai_tensor_dot_api(1, 2560, 1024, 3);
}

TEST(KAI_acc42, GEMV_1x2560x1024) {
  test_kai_tensor_dot_api(1, 2560, 1024, 4);
}

TEST(KAI_acc52, GEMM_1x2560x1024) {
  test_kai_tensor_dot_api(1, 2560, 1024, 5);
}

TEST(KAI_acc62, GEMM_1x2560x1024) {
  test_kai_tensor_dot_api(1, 2560, 1024, 6);
}

TEST(KAI_acc72, GEMM_1x2560x1024) {
  test_kai_tensor_dot_api(1, 2560, 1024, 7);
}




TEST(Q40_acc3, GEMM_1024x2560x4096) {
  test_q40_tensor_dot_api(1024, 2560, 4096);
}

TEST(KAI_acc03, GEMM_1024x2560x4096) {
  test_kai_tensor_dot_api(1024, 2560, 4096, 0);
}

TEST(KAI_acc13, GEMM_1024x2560x4096) {
  test_kai_tensor_dot_api(1024, 2560, 4096, 1);
}

TEST(KAI_acc23, GEMM_1024x2560x4096) {
  test_kai_tensor_dot_api(1024, 2560, 4096, 2);
}

TEST(KAI_acc33, GEMM_1024x2560x4096) {
  test_kai_tensor_dot_api(1024, 2560, 4096, 3);
}

TEST(KAI_acc43, GEMM_1024x2560x4096) {
  test_kai_tensor_dot_api(1024, 2560, 4096, 4);
}

TEST(KAI_acc53, GEMM_1024x2560x4096) {
  test_kai_tensor_dot_api(1024, 2560, 4096, 5);
}

TEST(KAI_acc63, GEMM_1024x2560x4096) {
  test_kai_tensor_dot_api(1024, 2560, 4096, 6);
}

TEST(KAI_acc73, GEMM_1024x2560x4096) {
  test_kai_tensor_dot_api(1024, 2560, 4096, 7);
}



TEST(Q40_acc4, GEMV_1x2560x4096) {
  test_q40_tensor_dot_api(1, 2560, 4096);
}

TEST(KAI_acc04, GEMV_1x2560x4096) {
  test_kai_tensor_dot_api(1, 2560, 4096, 0);
}

TEST(KAI_acc14, GEMV_1x2560x4096) {
  test_kai_tensor_dot_api(1, 2560, 4096, 1);
}

TEST(KAI_acc24, GEMV_1x2560x4096) {
  test_kai_tensor_dot_api(1, 2560, 4096, 2);
}

TEST(KAI_acc34, GEMV_1x2560x4096) {
  test_kai_tensor_dot_api(1, 2560, 4096, 3);
}

TEST(KAI_acc44, GEMV_1x2560x4096) {
  test_kai_tensor_dot_api(1, 2560, 4096, 4);
}

TEST(KAI_acc54, GEMV_1x2560x4096) {
  test_kai_tensor_dot_api(1, 2560, 4096, 5);
}

TEST(KAI_acc64, GEMV_1x2560x4096) {
  test_kai_tensor_dot_api(1, 2560, 4096, 6);
}

TEST(KAI_acc74, GEMV_1x2560x4096) {
  test_kai_tensor_dot_api(1, 2560, 4096, 7);
}




TEST(Q40_acc5, GEMM_1024x2560x9728) {
  test_q40_tensor_dot_api(1024, 2560, 9728);
}

TEST(KAI_acc05, GEMM_1024x2560x9728) {
  test_kai_tensor_dot_api(1024, 2560, 4096, 0);
}

TEST(KAI_acc15, GEMM_1024x2560x9728) {
  test_kai_tensor_dot_api(1024, 2560, 9728, 1);
}

TEST(KAI_acc25, GEMM_1024x2560x9728) {
  test_kai_tensor_dot_api(1024, 2560, 9728, 2);
}

TEST(KAI_acc35, GEMM_1024x2560x9728) {
  test_kai_tensor_dot_api(1024, 2560, 9728, 3);
}

TEST(KAI_acc45, GEMM_1024x2560x9728) {
  test_kai_tensor_dot_api(1024, 2560, 9728, 4);
}

TEST(KAI_acc55, GEMM_1024x2560x9728) {
  test_kai_tensor_dot_api(1024, 2560, 9728, 5);
}

TEST(KAI_acc65, GEMM_1024x2560x9728) {
  test_kai_tensor_dot_api(1024, 2560, 9728, 6);
}

TEST(KAI_acc75, GEMM_1024x2560x9728) {
  test_kai_tensor_dot_api(1024, 2560, 9728, 7);
}




TEST(Q40_acc6, GEMV_1x2560x9728) {
  test_q40_tensor_dot_api(1, 2560, 9728);
}

TEST(KAI_acc06, GEMV_1x2560x9728) {
  test_kai_tensor_dot_api(1, 2560, 4096, 0);
}

TEST(KAI_acc16, GEMV_1x2560x9728) {
  test_kai_tensor_dot_api(1, 2560, 9728, 1);
}

TEST(KAI_acc26, GEMV_1x2560x9728) {
  test_kai_tensor_dot_api(1, 2560, 9728, 2);
}

TEST(KAI_acc36, GEMV_1x2560x9728) {
  test_kai_tensor_dot_api(1, 2560, 9728, 3);
}

TEST(KAI_acc46, GEMV_1x2560x9728) {
  test_kai_tensor_dot_api(1, 2560, 9728, 4);
}

TEST(KAI_acc56, GEMV_1x2560x9728) {
  test_kai_tensor_dot_api(1, 2560, 9728, 5);
}

TEST(KAI_acc66, GEMV_1x2560x9728) {
  test_kai_tensor_dot_api(1, 2560, 9728, 6);
}

TEST(KAI_acc76, GEMV_1x2560x9728) {
  test_kai_tensor_dot_api(1, 2560, 9728, 7);
}



TEST(Q40_acc7, GEMM_1024x4096x2560) {
  test_q40_tensor_dot_api(1024, 4096, 2560);
}

TEST(KAI_acc07, GEMM_1024x4096x2560) {
  test_kai_tensor_dot_api(1024, 4096, 2560, 0);
}

TEST(KAI_acc17, GEMM_1024x4096x2560) {
  test_kai_tensor_dot_api(1024, 4096, 2560, 1);
}

TEST(KAI_acc27, GEMM_1024x4096x2560) {
  test_kai_tensor_dot_api(1024, 4096, 2560, 2);
}

TEST(KAI_acc37, GEMM_1024x4096x2560) {
  test_kai_tensor_dot_api(1024, 4096, 2560, 3);
}

TEST(KAI_acc47, GEMM_1024x4096x2560) {
  test_kai_tensor_dot_api(1024, 4096, 2560, 4);
}

TEST(KAI_acc57, GEMM_1024x4096x2560) {
  test_kai_tensor_dot_api(1024, 4096, 2560, 5);
}

TEST(KAI_acc67, GEMM_1024x4096x2560) {
  test_kai_tensor_dot_api(1024, 4096, 2560, 6);
}

TEST(KAI_acc77, GEMM_1024x4096x2560) {
  test_kai_tensor_dot_api(1024, 4096, 2560, 7);
}




TEST(Q40_acc8, GEMV_1x4096x2560) {
  test_q40_tensor_dot_api(1, 4096, 2560);
}

TEST(KAI_acc08, GEMV_1x4096x2560) {
  test_kai_tensor_dot_api(1, 4096, 2560, 0);
}

TEST(KAI_acc18, GEMV_1x4096x2560) {
  test_kai_tensor_dot_api(1, 4096, 2560, 1);
}

TEST(KAI_acc28, GEMV_1x4096x2560) {
  test_kai_tensor_dot_api(1, 4096, 2560, 2);
}

TEST(KAI_acc38, GEMV_1x4096x2560) {
  test_kai_tensor_dot_api(1, 4096, 2560, 3);
}

TEST(KAI_acc48, GEMV_1x4096x2560) {
  test_kai_tensor_dot_api(1, 4096, 2560, 4);
}

TEST(KAI_acc58, GEMV_1x4096x2560) {
  test_kai_tensor_dot_api(1, 4096, 2560, 5);
}

TEST(KAI_acc68, GEMV_1x4096x2560) {
  test_kai_tensor_dot_api(1, 4096, 2560, 6);
}

TEST(KAI_acc78, GEMV_1x4096x2560) {
  test_kai_tensor_dot_api(1, 4096, 2560, 7);
}




TEST(Q40_acc9, GEMM_1024x9728x2560) {
  test_q40_tensor_dot_api(1024, 9728, 2560);
}

TEST(KAI_acc09, GEMM_1024x9728x2560) {
  test_kai_tensor_dot_api(1024, 9728, 2560, 0);
}

TEST(KAI_acc19, GEMM_1024x9728x2560) {
  test_kai_tensor_dot_api(1024, 9728, 2560, 1);
}

TEST(KAI_acc29, GEMM_1024x9728x2560) {
  test_kai_tensor_dot_api(1024, 9728, 2560, 2);
}

TEST(KAI_acc39, GEMM_1024x9728x2560) {
  test_kai_tensor_dot_api(1024, 9728, 2560, 3);
}

TEST(KAI_acc49, GEMM_1024x9728x2560) {
  test_kai_tensor_dot_api(1024, 9728, 2560, 4);
}

TEST(KAI_acc59, GEMM_1024x9728x2560) {
  test_kai_tensor_dot_api(1024, 9728, 2560, 5);
}

TEST(KAI_acc69, GEMM_1024x9728x2560) {
  test_kai_tensor_dot_api(1024, 9728, 2560, 6);
}

TEST(KAI_acc79, GEMM_1024x9728x2560) {
  test_kai_tensor_dot_api(1024, 9728, 2560, 7);
}





TEST(Q40_acc10, GEMV_1x9728x2560) {
  test_q40_tensor_dot_api(1, 9728, 2560);
}

TEST(KAI_acc10, GEMV_1x9728x2560) {
  test_kai_tensor_dot_api(1, 9728, 2560, 0);
}

TEST(KAI_acc110, GEMV_1x9728x2560) {
  test_kai_tensor_dot_api(1, 9728, 2560, 1);
}

TEST(KAI_acc210, GEMV_1x9728x2560) {
  test_kai_tensor_dot_api(1, 9728, 2560, 2);
}

TEST(KAI_acc310, GEMV_1x9728x2560) {
  test_kai_tensor_dot_api(1, 9728, 2560, 3);
}

TEST(KAI_acc410, GEMV_1x9728x2560) {
  test_kai_tensor_dot_api(1, 9728, 2560, 4);
}

TEST(KAI_acc510, GEMV_1x9728x2560) {
  test_kai_tensor_dot_api(1, 9728, 2560, 5);
}

TEST(KAI_acc610, GEMV_1x9728x2560) {
  test_kai_tensor_dot_api(1, 9728, 2560, 6);
}

TEST(KAI_acc710, GEMV_1x9728x2560) {
  test_kai_tensor_dot_api(1, 9728, 2560, 7);
}










TEST(Q40_vs_kai, GEMM_1024x2560x1024) {
  test_q40_vs_kai(1024, 2560, 1024);
}

TEST(Q40_vs_kai, GEMV_1x2560x1024) {
  test_q40_vs_kai(1, 2560, 1024);
}

TEST(Q40_vs_kai, GEMM_1024x2560x4096) {
  test_q40_vs_kai(1024, 2560, 4096);
}

TEST(Q40_vs_kai, GEMV_1x2560x4096) {
  test_q40_vs_kai(1, 2560, 4096);
}

TEST(Q40_vs_kai, GEMM_1024x2560x9728) {
  test_q40_vs_kai(1024, 2560, 9728);
}

TEST(Q40_vs_kai, GEMV_1x2560x9728) {
  test_q40_vs_kai(1, 2560, 9728);
}

TEST(Q40_vs_kai, GEMM_1024x4096x2560) {
  test_q40_vs_kai(1024, 4096, 2560);
}

TEST(Q40_vs_kai, GEMV_1x4096x2560) {
  test_q40_vs_kai(1, 4096, 2560);
}

TEST(Q40_vs_kai, GEMM_1024x9728x2560) {
  test_q40_vs_kai(1024, 9728, 2560);
}

TEST(Q40_vs_kai, GEMV_1x9728x2560) {
  test_q40_vs_kai(1, 9728, 2560);
}




#endif




/**
 * @brief Main function for Google Test
 */
int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
