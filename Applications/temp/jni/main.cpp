#include <layer.h>
#include <cl_context.h>
#include "q4_0_utils.h"
#include "nntrainer_test_util.h"
#include "int4_utils.h"
#include <fp16.h>

using namespace nntrainer;

#define Q4_0 32

static void run_dequantization_test_(const uint32_t K, const uint32_t N) {
  std::cout<<"\nrun_dequantization_test_"<< K <<  "_"<< N <<"\n";
  const float epsilon = 0.01f;

  auto *blas_cc = static_cast<nntrainer::ClContext *>(
    nntrainer::Engine::Global().getRegisteredContext("gpu"));
  std::vector<float> weight_fp32 =
    generate_random_vector<float>(N * K, -2.0f, 2.0f);

  // Dequantization Q4_0
  if (K % Q4_0 == 0 && N % 8 == 0) {
    size_t q4_data_size = K * N / Q4_0 * sizeof(block_q4_0);
    std::vector<uint8_t> q4_weight(q4_data_size);
    std::vector<uint8_t> q4_weight_repack(q4_data_size);
    nntrainer::quantize_q4_0(weight_fp32.data(), q4_weight.data(), N, K,
                             nullptr);
    nntrainer::repack_q4_0(q4_weight_repack.data(), q4_weight.data(),
                           q4_data_size, N, K);

    std::vector<float> dequantized_weights_q4(N * K);
    Q4_0Utils::dequantizeQ4_0x4(q4_weight_repack.data(), N, K,
                                dequantized_weights_q4.data());

    float mse_dequantized_q4 =
      mse<float>(weight_fp32.data(), dequantized_weights_q4.data(), N * K);

    std::cout << "MSE dequantized Q4_0: " << std::setprecision(10)
              << mse_dequantized_q4 << std::endl;

  }

  // Dequantization INT4
  int scale_group_size = 32;
  std::vector<uint8_t> quantized_weights;
  std::vector<uint16_t> quantized_scales;
  Int4Utils::quantizeAndRepack(weight_fp32.data(), N, K, scale_group_size,
                               quantized_weights, quantized_scales);

  std::vector<float> dequantized_weights_int4;
  Int4Utils::dequantizePacked(quantized_weights, quantized_scales, N, K,
                              scale_group_size, dequantized_weights_int4);

  // Dequantize QINT4 by row
  std::vector<float> dequantized_weights_int4_row(N * K, 0);
  for (int row_idx = 0; row_idx < N; ++row_idx) {
    Int4Utils::dequantizePackedRow(
      quantized_weights.data(), quantized_scales.data(), N, K, scale_group_size,
      row_idx, dequantized_weights_int4_row.data() + (K * row_idx));
  }

  float mse_dequantized_int4 =
    mse<float>(weight_fp32.data(), dequantized_weights_int4.data(), N * K);

  float mse_dequantized_int4_row =
    mse<float>(dequantized_weights_int4_row.data(), weight_fp32.data(), N * K);

  std::cout << "MSE dequantized INT4: " << std::setprecision(10)
            << mse_dequantized_int4 << std::endl;
  std::cout << "MSE dequantized INT4 by row: " << std::setprecision(10)
            << mse_dequantized_int4_row << std::endl;
  std::cout<<"\n";
}

static void run_int4_gemm_test_(const uint32_t M, const uint32_t K, 
                                const uint32_t N, int scale_group_size, bool debug = false) {
  auto *blas_cc = static_cast<nntrainer::ClContext *>(
    nntrainer::Engine::Global().getRegisteredContext("gpu"));

  const int INT4_BLOCK_N_SIZE = 32;
  uint32_t alignN = align(N, INT4_BLOCK_N_SIZE);
  uint32_t alignK = align(K, scale_group_size);

  uint32_t input_size = M * alignK;

  std::vector<float> input;
  std::vector<float> weight_fp32;
  if (debug){
    float ones_ratio = 0.1f;
    input = generate_01_vector(input_size, ones_ratio);
    weight_fp32 = generate_01_vector(N * K, ones_ratio);
  }
  else {
    input = generate_random_vector<float, false>(input_size, -1.0, 1.0);
    weight_fp32 = generate_random_vector<float, false>(N * K, -1.0, 1.0);
  }

  if (debug) {
    for (int x = 0; x < M; x++) {
      for (int y = 0; y < alignK; y++) {
        if (y % 10 == 0) {
          printf("| ");
        }
        if (input[y * M + x] > 0.1) {
          printf("1 ");
        } else {
          printf("0 ");
        }
      }
      printf("\n");
    }
    printf("---------------------------\n");
    for (int x = 0; x < K; x++) {
      for (int y = 0; y < N; y++) {
        if (y % 10 == 0) {
          printf("| ");
        }
        if (weight_fp32[y * K + x] > 0.1) {
          printf("1 ");
        } else {
          printf("0 ");
        }
      }
      printf("\n");
    }
  }

  std::vector<float> ref_dst(M * N, 0.0f);
  nntrainer::sgemm(0, false, true, M, N, K, 1.F, input.data(), K,
                   weight_fp32.data(), K, 0.F, ref_dst.data(), N);

  unsigned int run_count = 100;

  float mse_q4 = 0.0f;

  if (K % Q4_0 == 0 && N % 8 == 0) {
    size_t q4_data_size = K * N / Q4_0 * sizeof(block_q4_0);
    std::vector<float> q4_output_fp32(M * N);
    std::vector<uint8_t> q4_weight(q4_data_size);
    std::vector<uint8_t> q4_weight_repack(q4_data_size);
    nntrainer::quantize_q4_0(weight_fp32.data(), q4_weight.data(), N, K,
                             nullptr);
    nntrainer::repack_q4_0(q4_weight_repack.data(), q4_weight.data(),
                           q4_data_size, N, K);

    auto t1 = std::chrono::high_resolution_clock::now();
    for (unsigned int i = 0; i < run_count; ++i) {
      nntrainer::gemm_q4_0(M, N, K, input.data(), K, q4_weight_repack.data(), N,
                         q4_output_fp32.data(), N);
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    auto cpu_dt =
      std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

    std::cout << " - time : CPU = " << cpu_dt / (run_count * 1.0f) << " ms"
            << std::endl;

    mse_q4 = mse<float>(ref_dst.data(), q4_output_fp32.data(), M * N);
  }

  uint16_t *input_ptr = (uint16_t *)allocateSVM(input_size * sizeof(uint16_t));
  int8_t *weight_ptr = (int8_t *)allocateSVM(alignK * alignN / 2);
  uint16_t *scale_ptr = (uint16_t *)allocateSVM(ceilDiv(K, scale_group_size) *
                                                alignN * sizeof(uint16_t));
  uint16_t *output_ptr = (uint16_t *)allocateSVM(M * alignN * sizeof(uint16_t));

  blas_cc->command_queue_inst_.enqueueSVMMap(
    input_ptr, input_size * sizeof(uint16_t), false);
  blas_cc->command_queue_inst_.enqueueSVMMap(weight_ptr, alignK * alignN / 2,
                                             false);
  blas_cc->command_queue_inst_.enqueueSVMMap(
    scale_ptr, ceilDiv(K, scale_group_size) * alignN * sizeof(uint16_t), false);

  std::vector<uint8_t> quantized_weights;
  std::vector<uint16_t> quantized_scales;
  Int4Utils::quantizeAndRepack(weight_fp32.data(), N, K, scale_group_size,
                               quantized_weights, quantized_scales);

}



  int main(int argc, char **argv) {
  // run_dequantization_test_(8192, 3072);
  // run_dequantization_test_(3072, 8192);
  // run_dequantization_test_(8188, 3068);
  // run_dequantization_test_(3068, 8188);
  // run_dequantization_test_(144, 168);
  run_int4_gemm_test_(32, 32, 32, 32, true);
  run_int4_gemm_test_(1024, 1024, 1024, 32);
}