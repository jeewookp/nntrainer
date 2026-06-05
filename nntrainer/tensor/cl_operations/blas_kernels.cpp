// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Debadri Samaddar <s.debadri@samsung.com>
 *
 * @file	blas_kernels.cpp
 * @date	14 May 2024
 * @brief	Common blas OpenCL kernels
 * @see		https://github.com/nntrainer/nntrainer
 * @author	Debadri Samaddar <s.debadri@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#include "blas_kernels_templates.h"
#include <cl_kernels/cl_kernels.h>

#include "cl_tensor_view.h"
#include "util_func.h"
#include <array>
#include <cstdio>
#include <cstdlib>
#include <fp16.h>

namespace nntrainer {

void gemv_int4_async_cl(std::vector<void *> weights,
                        std::vector<uint16_t *> scales, uint16_t *input,
                        std::vector<uint16_t *> outputs, unsigned int K,
                        std::vector<unsigned int> Ns,
                        unsigned int quantization_group_size) {
  bool result = false;
  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  auto &clbuffInstance = ClBufferManager::Global();

  const bool scale_row_major = false;

  ClContext::SharedPtrClKernel kernel_ptr = blas_cc->registerClKernel(
    int4_gemv_kernel, "fully_connected_gpu_int4_gemv");
  if (!kernel_ptr) {
    throw std::runtime_error(
      "Failed to get kernel_ptr for fully_connected_gpu_int4_gemv");
    return;
  }

  const int work_group_size[3] = {16, 1, 16};

  for (unsigned int i = 0; i < Ns.size(); ++i) {
    int arg = 0;
    int N = Ns[i];
    const auto N_GROUP_SIZE = 32; // due to input data format
    const unsigned int alignN = align(N, N_GROUP_SIZE);
    void *weight = weights[i];
    uint16_t *scale = scales[i];
    uint16_t *output = outputs[i];
    result = kernel_ptr->SetKernelSVMArguments(arg++, input);
    if (!result)
      throw std::runtime_error(
        "Failed to set kernel argument 0 for fully_connected_gpu_int4_gemv");

    kernel_ptr->SetKernelSVMArguments(arg++, scale);
    if (!result)
      throw std::runtime_error(
        "Failed to set kernel argument 1 for fully_connected_gpu_int4_gemv");

    result = kernel_ptr->SetKernelSVMArguments(arg++, output);

    if (!result)
      throw std::runtime_error(
        "Failed to set kernel argument 2 for fully_connected_gpu_int4_gemv");

    result = kernel_ptr->SetKernelSVMArguments(arg++, weight);
    if (!result)
      throw std::runtime_error(
        "Failed to set kernel argument 3 for fully_connected_gpu_int4_gemv");

    result = kernel_ptr->SetKernelArguments(arg++, &K, sizeof(int));
    if (!result)
      throw std::runtime_error(
        "Failed to set kernel argument 4 for fully_connected_gpu_int4_gemv");

    result = kernel_ptr->SetKernelArguments(arg++, &N, sizeof(int));
    if (!result)
      throw std::runtime_error(
        "Failed to set kernel argument 5 for fully_connected_gpu_int4_gemv");

    int q_group_size = quantization_group_size;
    result = kernel_ptr->SetKernelArguments(arg++, &q_group_size, sizeof(int));
    if (!result)
      throw std::runtime_error(
        "Failed to set kernel argument 6 for fully_connected_gpu_int4_gemv");

    int row_major = scale_row_major;
    result = kernel_ptr->SetKernelArguments(arg++, &row_major, sizeof(int));
    if (!result)
      throw std::runtime_error(
        "Failed to set kernel argument 7 for fully_connected_gpu_int4_gemv");

    const int work_groups_count[3] = {(int)(alignN / 2), 1, 16};
    result = blas_cc->command_queue_inst_.DispatchCommand(
      kernel_ptr, work_groups_count, work_group_size);
    if (!result) {
      throw std::runtime_error(
        "Failed to dispatch kernel for fully_connected_gpu_int4_gemv");
      return;
    }
  }

  for (unsigned int i = 0; i < Ns.size(); ++i) {
    blas_cc->command_queue_inst_.enqueueSVMMap(outputs[i],
                                               Ns[i] * sizeof(uint16_t), true);
  }
  if (!result) {
    throw std::runtime_error(
      "Failed to read output data for fully_connected_gpu_int4_gemv");
    return;
  }
}

void gemv_int4_cl(char *weight, uint16_t *scale, uint16_t *input,
                  uint16_t *output, unsigned int K, unsigned int N,
                  unsigned int quantization_group_size) {
  const auto N_GROUP_SIZE = 32; // due to input data format
  const unsigned int alignN = align(N, N_GROUP_SIZE);

  bool result = false;
  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  auto &clbuffInstance = ClBufferManager::Global();

  const bool scale_row_major = false;

  ClContext::SharedPtrClKernel kernel_ptr = blas_cc->registerClKernel(
    int4_gemv_kernel, "fully_connected_gpu_int4_gemv");
  if (!kernel_ptr) {
    throw std::runtime_error(
      "Failed to get kernel_ptr for fully_connected_gpu_int4_gemv");
    return;
  }

  int arg = 0;

  result = kernel_ptr->SetKernelSVMArguments(arg++, input);
  if (!result)
    throw std::runtime_error(
      "Failed to set kernel argument 0 for fully_connected_gpu_int4_gemv");

  kernel_ptr->SetKernelSVMArguments(arg++, scale);
  if (!result)
    throw std::runtime_error(
      "Failed to set kernel argument 1 for fully_connected_gpu_int4_gemv");

  result = kernel_ptr->SetKernelSVMArguments(arg++, output);

  if (!result)
    throw std::runtime_error(
      "Failed to set kernel argument 2 for fully_connected_gpu_int4_gemv");

  result = kernel_ptr->SetKernelSVMArguments(arg++, weight);
  if (!result)
    throw std::runtime_error(
      "Failed to set kernel argument 3 for fully_connected_gpu_int4_gemv");

  result = kernel_ptr->SetKernelArguments(arg++, &K, sizeof(int));
  if (!result)
    throw std::runtime_error(
      "Failed to set kernel argument 4 for fully_connected_gpu_int4_gemv");

  result = kernel_ptr->SetKernelArguments(arg++, &N, sizeof(int));
  if (!result)
    throw std::runtime_error(
      "Failed to set kernel argument 5 for fully_connected_gpu_int4_gemv");

  int q_group_size = quantization_group_size;
  result = kernel_ptr->SetKernelArguments(arg++, &q_group_size, sizeof(int));
  if (!result)
    throw std::runtime_error(
      "Failed to set kernel argument 6 for fully_connected_gpu_int4_gemv");

  int row_major = scale_row_major;
  result = kernel_ptr->SetKernelArguments(arg++, &row_major, sizeof(int));
  if (!result)
    throw std::runtime_error(
      "Failed to set kernel argument 7 for fully_connected_gpu_int4_gemv");

  const int work_groups_count[3] = {(int)(alignN / 2), 1, 16};
  const int work_group_size[3] = {16, 1, 16};

  result = blas_cc->command_queue_inst_.DispatchCommand(
    kernel_ptr, work_groups_count, work_group_size);
  if (!result) {
    throw std::runtime_error(
      "Failed to dispatch kernel for fully_connected_gpu_int4_gemv");
    return;
  }

  /// @todo synchronize when only needed
  blas_cc->command_queue_inst_.enqueueSVMMap(output, N * sizeof(uint16_t),
                                             true);
  if (!result) {
    throw std::runtime_error(
      "Failed to read output data for fully_connected_gpu_int4_gemv");
    return;
  }
}

void gemv_int4_async_cl(std::vector<void *> weights,
                        std::vector<uint16_t *> scales, float *input,
                        std::vector<float *> outputs, unsigned int K,
                        std::vector<unsigned int> Ns,
                        unsigned int quantization_group_size) {
  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  auto &clbuffInstance = ClBufferManager::Global();

  // copy fp32 input to fp16
  copy_fp32_u16(K, input, (uint16_t *)clbuffInstance.getSVMInput());
  std::vector<uint16_t *> output_vec;

  for (int i = 0; i < Ns.size(); ++i) {
    output_vec.push_back((uint16_t *)clbuffInstance.getSVMOutput(i));
  }

  gemv_int4_async_cl(weights, scales, (uint16_t *)clbuffInstance.getSVMInput(),
                     output_vec, K, Ns, quantization_group_size);

  for (int i = 0; i < Ns.size(); ++i) {
    copy_u16_fp32(Ns[i], (uint16_t *)clbuffInstance.getSVMOutput(i),
                  outputs[i]);
  }
}

void gemv_int4_cl(char *weight, uint16_t *scale, float *input, float *output,
                  unsigned int K, unsigned int N,
                  unsigned int quantization_group_size) {
  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  auto &clbuffInstance = ClBufferManager::Global();

  // copy fp32 input to fp16
  copy_fp32_u16(K, input, (uint16_t *)clbuffInstance.getSVMInput());

  // perform int4 matmul
  gemv_int4_cl(weight, scale, (uint16_t *)clbuffInstance.getSVMInput(),
               (uint16_t *)clbuffInstance.getSVMOutput(), K, N,
               quantization_group_size);

  // copy fp16 output to fp32
  copy_u16_fp32(N, (uint16_t *)clbuffInstance.getSVMOutput(), output);
}

void gemm_q4_0_async_cl(std::vector<void *> matAdata, float *matBdata,
                        std::vector<float *> matCdata, unsigned int M,
                        std::vector<unsigned int> Ns, unsigned int K) {
  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  auto &clbuffInstance = ClBufferManager::Global();

  int padding = 0;
  if (M % 8 > 0) {
    padding = 8 - (M % 8);
  }

  int padded_M = M + padding;

  ClContext::SharedPtrClKernel kernel_ptr = blas_cc->registerClKernel(
    q4_0_ab_bi_8x4_kernel, "kernel_mul_mat_Ab_Bi_8x4");
  if (!kernel_ptr) {
    throw std::runtime_error(
      "Failed to get kernel_ptr for kernel_mul_mat_Ab_Bi_8x4");
    return;
  }

  bool result = false;

  /// @note Transpose fp32 input. This can only be done once
  transpose_32_16(matBdata, M, K);

  const int work_group_size[3] = {1, 128, 1};

  for (unsigned int i = 0; i < Ns.size(); ++i) {
    int N = Ns[i];
    void *mdata = matAdata[i];
    float *rdata = matCdata[i];

    unpack_q4_0x8_transpose16(mdata, (uint16_t *)clbuffInstance.getSVMScale(i),
                              (uint16_t *)clbuffInstance.getSVMQuant(i), N, K);

    int arg = 0;

    result =
      kernel_ptr->SetKernelSVMArguments(arg++, clbuffInstance.getSVMQuant(i));
    if (!result)
      throw std::runtime_error(
        "Failed to set kernel argument 0 for kernel_mul_mat_Ab_Bi_8x4");

    result =
      kernel_ptr->SetKernelSVMArguments(arg++, clbuffInstance.getSVMScale(i));
    if (!result)
      throw std::runtime_error(
        "Failed to set kernel argument 1 for kernel_mul_mat_Ab_Bi_8x4");

    result =
      kernel_ptr->SetKernelSVMArguments(arg++, clbuffInstance.getSVMInput());
    if (!result)
      throw std::runtime_error(
        "Failed to set kernel argument 2 for kernel_mul_mat_Ab_Bi_8x4");

    result = kernel_ptr->SetKernelSVMArguments(arg++, rdata);
    if (!result)
      throw std::runtime_error(
        "Failed to set kernel argument 3 for kernel_mul_mat_Ab_Bi_8x4");

    result = kernel_ptr->SetKernelArguments(arg++, &N, sizeof(int));
    if (!result)
      throw std::runtime_error(
        "Failed to set kernel argument 4 for kernel_mul_mat_Ab_Bi_8x4");

    result = kernel_ptr->SetKernelArguments(arg++, &padded_M, sizeof(int));
    if (!result)
      throw std::runtime_error(
        "Failed to set kernel argument 5 for kernel_mul_mat_Ab_Bi_8x4");

    result = kernel_ptr->SetKernelArguments(arg++, &K, sizeof(int));
    if (!result)
      throw std::runtime_error(
        "Failed to set kernel argument 6 for kernel_mul_mat_Ab_Bi_8x4");

    result = kernel_ptr->SetKernelArguments(arg++, &M, sizeof(int));
    if (!result)
      throw std::runtime_error(
        "Failed to set kernel argument 7 for kernel_mul_mat_Ab_Bi_8x4");
    const int work_groups_count[3] = {(int)ceil(M / 8.0f), (int)N / 4, 1};

    // Perform Matrix Multiplication
    result = blas_cc->command_queue_inst_.DispatchCommand(
      kernel_ptr, work_groups_count, work_group_size);
    if (!result) {
      throw std::runtime_error(
        "Failed to dispatch kernel for kernel_mul_mat_Ab_Bi_8x4");
    }
  }

  for (unsigned int i = 0; i < Ns.size(); ++i) {
    blas_cc->command_queue_inst_.enqueueSVMMap(matCdata[i],
                                               M * Ns[i] * sizeof(float), true);
  }
}

void gemm_q4_0_cl(void *matAdata, float *matBdata, float *matCdata,
                  unsigned int M, unsigned int N, unsigned int K) {
  bool result = false;
  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  auto &clbuffInstance = ClBufferManager::Global();

  size_t q_size_bytes = N * (K / 2);
  size_t d_size_bytes = N * (K / 32) * 2;

  // 1. Preprocess matrix A
  // 1.1 Unpack the Q4_0x8 matrix A to make a struct of array (src_q, src_d)
  // 1.2 Perform 2D 16-bit transpose src_q, src_d
  unpack_q4_0x8_transpose16(matAdata, (uint16_t *)clbuffInstance.getSVMScale(),
                            (uint16_t *)clbuffInstance.getSVMQuant(), N, K);

  // 2. Preprocess matrix B: Transpose the Matrix B and convert to FP16
  /// @note mat mul will compute 8 elements at once, padding
  // will be added if M is not multiple of 8.
  transpose_32_16(matBdata, M, K);

  int padding = 0;
  if (M % 8 > 0) {
    padding = 8 - (M % 8);
  }

  int padded_M = M + padding;

  // 3. Perform Matrix Multiplication
  ClContext::SharedPtrClKernel kernel_ptr = blas_cc->registerClKernel(
    q4_0_ab_bi_8x4_kernel, "kernel_mul_mat_Ab_Bi_8x4");
  if (!kernel_ptr) {
    throw std::runtime_error(
      "Failed to get kernel_ptr for kernel_mul_mat_Ab_Bi_8x4");
    return;
  }

  int arg = 0;

  result =
    kernel_ptr->SetKernelSVMArguments(arg++, clbuffInstance.getSVMQuant());
  if (!result)
    throw std::runtime_error(
      "Failed to set kernel argument 0 for kernel_mul_mat_Ab_Bi_8x4");

  kernel_ptr->SetKernelSVMArguments(arg++, clbuffInstance.getSVMScale());
  if (!result)
    throw std::runtime_error(
      "Failed to set kernel argument 1 for kernel_mul_mat_Ab_Bi_8x4");

  result =
    kernel_ptr->SetKernelSVMArguments(arg++, clbuffInstance.getSVMInput());

  if (!result)
    throw std::runtime_error(
      "Failed to set kernel argument 2 for kernel_mul_mat_Ab_Bi_8x4");

  result = kernel_ptr->SetKernelSVMArguments(arg++, matCdata);
  if (!result)
    throw std::runtime_error(
      "Failed to set kernel argument 3 for kernel_mul_mat_Ab_Bi_8x4");

  result = kernel_ptr->SetKernelArguments(arg++, &N, sizeof(int));
  if (!result)
    throw std::runtime_error(
      "Failed to set kernel argument 4 for kernel_mul_mat_Ab_Bi_8x4");

  result = kernel_ptr->SetKernelArguments(arg++, &padded_M, sizeof(int));
  if (!result)
    throw std::runtime_error(
      "Failed to set kernel argument 5 for kernel_mul_mat_Ab_Bi_8x4");

  result = kernel_ptr->SetKernelArguments(arg++, &K, sizeof(int));
  if (!result)
    throw std::runtime_error(
      "Failed to set kernel argument 6 for kernel_mul_mat_Ab_Bi_8x4");

  result = kernel_ptr->SetKernelArguments(arg++, &M, sizeof(int));
  if (!result)
    throw std::runtime_error(
      "Failed to set kernel argument 7 for kernel_mul_mat_Ab_Bi_8x4");

  const int work_groups_count[3] = {(int)ceil(M / 8.0f), (int)N / 4, 1};
  const int work_group_size[3] = {1, 128, 1};

  result = blas_cc->command_queue_inst_.DispatchCommand(
    kernel_ptr, work_groups_count, work_group_size);
  if (!result) {
    throw std::runtime_error(
      "Failed to dispatch kernel for kernel_mul_mat_Ab_Bi_8x4");
    return;
  }

  /// @todo synchronize when only needed
  blas_cc->command_queue_inst_.enqueueSVMMap(matCdata, M * N * sizeof(float),
                                             true);
  if (!result) {
    throw std::runtime_error(
      "Failed to read output data for kernel_mul_mat_Ab_Bi_8x4");
    return;
  }
}

void gemm_int4_async_cl(float *input, std::vector<void *> weights,
                        std::vector<uint16_t *> scales,
                        std::vector<float *> matCdata, unsigned int M,
                        std::vector<unsigned int> Ns, unsigned int K,
                        unsigned int quantization_group_size) {
  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  auto &clbuffInstance = ClBufferManager::Global();

  bool result = false;

  // copy fp32 input to fp16
  copy_fp32_u16(M * K, input, (uint16_t *)clbuffInstance.getSVMInput());

  std::vector<cl_event> quantize_event(1);
  {
    int alignK = align(K, quantization_group_size);

    ClContext::SharedPtrClKernel kernel_ptr = blas_cc->registerClKernel(
      int4_quantize_input_kernel, "quantize_input_int4_pad");
    if (!kernel_ptr) {
      throw std::runtime_error("Failed to get kernel_ptr for quantize_input");
      return;
    }

    int arg = 0;

    result =
      kernel_ptr->SetKernelSVMArguments(arg++, clbuffInstance.getSVMInput());
    if (!result)
      throw std::runtime_error("Failed to set kernel argument 0 for "
                               "quantize_input");

    result =
      kernel_ptr->SetKernelSVMArguments(arg++, clbuffInstance.getSVMQuant());
    if (!result)
      throw std::runtime_error("Failed to set kernel argument 1 for "
                               "quantize_input");

    result =
      kernel_ptr->SetKernelSVMArguments(arg++, clbuffInstance.getSVMScale());
    if (!result)
      throw std::runtime_error("Failed to set kernel argument 2 for "
                               "quantize_input");

    int size_n = Ns[0];
    int size_k = K;
    int q_group_size = quantization_group_size;

    result = kernel_ptr->SetKernelArguments(arg++, &size_n, sizeof(int));
    if (!result)
      throw std::runtime_error("Failed to set kernel argument 3 for "
                               "quantize_input");

    result = kernel_ptr->SetKernelArguments(arg++, &size_k, sizeof(int));
    if (!result)
      throw std::runtime_error("Failed to set kernel argument 4 for "
                               "quantize_input");

    result = kernel_ptr->SetKernelArguments(arg++, &q_group_size, sizeof(int));
    if (!result)
      throw std::runtime_error("Failed to set kernel argument 5 for "
                               "quantize_input");

    std::array<size_t, 3> global_work_size = {
      (M * alignK) / quantization_group_size, 1, 1};

    blas_cc->command_queue_inst_.enqueueKernel(
      kernel_ptr->GetKernel(), global_work_size.size(), global_work_size.data(),
      nullptr, 0, nullptr, &quantize_event.front());
  }

  for (unsigned int i = 0; i < Ns.size(); ++i) {
    int N = Ns[i];
    const auto N_GROUP_SIZE = 32; // due to input data format
    const unsigned int alignN = align(N, N_GROUP_SIZE);

    const bool scale_row_major = false;

    ClContext::SharedPtrClKernel kernel_ptr =
      blas_cc->registerClKernel(gemm_int4_kernel, "fc_bf_tiled_kernel_default");
    if (!kernel_ptr) {
      throw std::runtime_error(
        "Failed to get kernel_ptr for fc_bf_tiled_kernel_default");
      return;
    }

    int arg = 0;
    int size_n = N;
    int size_k = K;
    int q_group_size = quantization_group_size;
    int row_major = scale_row_major;

    result =
      kernel_ptr->SetKernelSVMArguments(arg++, clbuffInstance.getSVMInput());

    if (!result)
      throw std::runtime_error(
        "Failed to set kernel argument 0 for fc_bf_tiled_kernel_default");

    result = kernel_ptr->SetKernelSVMArguments(arg++, scales[i]);
    if (!result)
      throw std::runtime_error(
        "Failed to set kernel argument 1 for fc_bf_tiled_kernel_default");

    result =
      kernel_ptr->SetKernelSVMArguments(arg++, clbuffInstance.getSVMOutput(i));
    if (!result)
      throw std::runtime_error(
        "Failed to set kernel argument 2 for fc_bf_tiled_kernel_default");

    result = kernel_ptr->SetKernelSVMArguments(arg++, weights[i]);
    if (!result)
      throw std::runtime_error(
        "Failed to set kernel argument 3 for fc_bf_tiled_kernel_default");

    result =
      kernel_ptr->SetKernelSVMArguments(arg++, clbuffInstance.getSVMQuant());
    if (!result)
      throw std::runtime_error(
        "Failed to set kernel argument 4 for fc_bf_tiled_kernel_default");

    result =
      kernel_ptr->SetKernelSVMArguments(arg++, clbuffInstance.getSVMScale());
    if (!result)
      throw std::runtime_error(
        "Failed to set kernel argument 5 for fc_bf_tiled_kernel_default");

    result = kernel_ptr->SetKernelArguments(arg++, &M, sizeof(int));
    if (!result)
      throw std::runtime_error(
        "Failed to set kernel argument 6 for fc_bf_tiled_kernel_default");

    result = kernel_ptr->SetKernelArguments(arg++, &size_n, sizeof(int));
    if (!result)
      throw std::runtime_error(
        "Failed to set kernel argument 7 for fc_bf_tiled_kernel_default");

    result = kernel_ptr->SetKernelArguments(arg++, &size_k, sizeof(int));
    if (!result)
      throw std::runtime_error(
        "Failed to set kernel argument 8 for fc_bf_tiled_kernel_default");

    result = kernel_ptr->SetKernelArguments(arg++, &q_group_size, sizeof(int));
    if (!result)
      throw std::runtime_error(
        "Failed to set kernel argument 9 for fc_bf_tiled_kernel_default");

    result = kernel_ptr->SetKernelArguments(arg++, &row_major, sizeof(int));
    if (!result)
      throw std::runtime_error(
        "Failed to set kernel argument 10 for fc_bf_tiled_kernel_default");

    const int work_groups_count[3] = {(int)(alignN / 2),
                                      (int)(align(ceilDiv(M, 8), 8)), 1};
    const int work_group_size[3] = {16, 8, 1};

    result = blas_cc->command_queue_inst_.DispatchCommand(
      kernel_ptr, work_groups_count, work_group_size, nullptr, quantize_event);
    if (!result) {
      throw std::runtime_error(
        "Failed to dispatch kernel for fc_bf_tiled_kernel_default");
      return;
    }
  }

  for (unsigned int i = 0; i < Ns.size(); ++i) {
    blas_cc->command_queue_inst_.enqueueSVMMap(
      clbuffInstance.getSVMOutput(i), M * Ns[i] * sizeof(uint16_t), true);

    // copy fp16 output to fp32
    copy_u16_fp32(M * Ns[i], (uint16_t *)clbuffInstance.getSVMOutput(i),
                  matCdata[i]);
  }
}

///  @note remove this when fp16 is enabled on Windows
void sgemm_int4_cl(float *input, char *weight, uint16_t *scale, float *output,
                   unsigned int M, unsigned int N, unsigned int K,
                   unsigned int quantization_group_size) {
  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  auto &clbuffInstance = ClBufferManager::Global();

  // copy fp32 input to fp16
  copy_fp32_u16(M * K, input, (uint16_t *)clbuffInstance.getSVMInput());

  // perform int4 matmul
  gemm_int4_cl(clbuffInstance.getSVMInput(), weight, scale,
               clbuffInstance.getSVMOutput(), M, N, K, quantization_group_size);

  // copy fp16 output to fp32
  copy_u16_fp32(M * N, (uint16_t *)clbuffInstance.getSVMOutput(), output);
}

void gemm_int4_cl(void *input, void *weights, void *scales, void *output,
                  unsigned int M, unsigned int N, unsigned int K,
                  unsigned int quantization_group_size) {
  int alignK = align(K, quantization_group_size);
  const auto N_GROUP_SIZE = 32; // due to input data format
  int alignN = align(N, N_GROUP_SIZE);

  bool result = false;
  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  auto &clbuffInstance = ClBufferManager::Global();
  const bool scale_row_major = false;

  std::vector<cl_event> quantize_event(1);
  {
    ClContext::SharedPtrClKernel kernel_ptr = blas_cc->registerClKernel(
      int4_quantize_input_kernel, "quantize_input_int4_pad");
    if (!kernel_ptr) {
      throw std::runtime_error("Failed to get kernel_ptr for quantize_input");
      return;
    }

    int arg = 0;
    int size_n = N;
    int size_k = K;
    int q_group_size = quantization_group_size;

    result = kernel_ptr->SetKernelSVMArguments(arg++, input);

    if (!result)
      throw std::runtime_error("Failed to set kernel argument 0 for "
                               "quantize_input");

    result =
      kernel_ptr->SetKernelSVMArguments(arg++, clbuffInstance.getSVMQuant());
    if (!result)
      throw std::runtime_error("Failed to set kernel argument 1 for "
                               "quantize_input");

    result =
      kernel_ptr->SetKernelSVMArguments(arg++, clbuffInstance.getSVMScale());
    if (!result)
      throw std::runtime_error("Failed to set kernel argument 2 for "
                               "quantize_input");

    result = kernel_ptr->SetKernelArguments(arg++, &size_n, sizeof(int));
    if (!result)
      throw std::runtime_error("Failed to set kernel argument 3 for "
                               "quantize_input");

    result = kernel_ptr->SetKernelArguments(arg++, &size_k, sizeof(int));
    if (!result)
      throw std::runtime_error("Failed to set kernel argument 4 for "
                               "quantize_input");

    result = kernel_ptr->SetKernelArguments(arg++, &q_group_size, sizeof(int));
    if (!result)
      throw std::runtime_error("Failed to set kernel argument 5 for "
                               "quantize_input");

    std::array<size_t, 3> global_work_size = {
      (M * alignK) / quantization_group_size, 1, 1};

    blas_cc->command_queue_inst_.enqueueKernel(
      kernel_ptr->GetKernel(), global_work_size.size(), global_work_size.data(),
      nullptr, 0, nullptr, &quantize_event.front());
  }

  // 3. Perform Matrix Multiplication
  ClContext::SharedPtrClKernel kernel_ptr =
    blas_cc->registerClKernel(gemm_int4_kernel, "fc_bf_tiled_kernel_default");
  if (!kernel_ptr) {
    throw std::runtime_error(
      "Failed to get kernel_ptr for fc_bf_tiled_kernel_default");
    return;
  }

  int arg = 0;
  int size_n = N;
  int size_k = K;
  int q_group_size = quantization_group_size;
  int row_major = scale_row_major;

  result = kernel_ptr->SetKernelSVMArguments(arg++, input);

  if (!result)
    throw std::runtime_error(
      "Failed to set kernel argument 0 for fc_bf_tiled_kernel_default");

  result = kernel_ptr->SetKernelSVMArguments(arg++, scales);
  if (!result)
    throw std::runtime_error(
      "Failed to set kernel argument 1 for fc_bf_tiled_kernel_default");

  result = kernel_ptr->SetKernelSVMArguments(arg++, output);
  if (!result)
    throw std::runtime_error(
      "Failed to set kernel argument 2 for fc_bf_tiled_kernel_default");

  result = kernel_ptr->SetKernelSVMArguments(arg++, weights);
  if (!result)
    throw std::runtime_error(
      "Failed to set kernel argument 3 for fc_bf_tiled_kernel_default");

  result =
    kernel_ptr->SetKernelSVMArguments(arg++, clbuffInstance.getSVMQuant());
  if (!result)
    throw std::runtime_error(
      "Failed to set kernel argument 4 for fc_bf_tiled_kernel_default");

  result =
    kernel_ptr->SetKernelSVMArguments(arg++, clbuffInstance.getSVMScale());
  if (!result)
    throw std::runtime_error(
      "Failed to set kernel argument 5 for fc_bf_tiled_kernel_default");

  result = kernel_ptr->SetKernelArguments(arg++, &M, sizeof(int));
  if (!result)
    throw std::runtime_error(
      "Failed to set kernel argument 6 for fc_bf_tiled_kernel_default");

  result = kernel_ptr->SetKernelArguments(arg++, &size_n, sizeof(int));
  if (!result)
    throw std::runtime_error(
      "Failed to set kernel argument 7 for fc_bf_tiled_kernel_default");

  result = kernel_ptr->SetKernelArguments(arg++, &size_k, sizeof(int));
  if (!result)
    throw std::runtime_error(
      "Failed to set kernel argument 8 for fc_bf_tiled_kernel_default");

  result = kernel_ptr->SetKernelArguments(arg++, &q_group_size, sizeof(int));
  if (!result)
    throw std::runtime_error(
      "Failed to set kernel argument 9 for fc_bf_tiled_kernel_default");

  result = kernel_ptr->SetKernelArguments(arg++, &row_major, sizeof(int));
  if (!result)
    throw std::runtime_error(
      "Failed to set kernel argument 10 for fc_bf_tiled_kernel_default");

  const int work_groups_count[3] = {(int)(alignN / 2),
                                    (int)(align(ceilDiv(M, 8), 8)), 1};
  const int work_group_size[3] = {16, 8, 1};

  result = blas_cc->command_queue_inst_.DispatchCommand(
    kernel_ptr, work_groups_count, work_group_size, nullptr, quantize_event);
  if (!result) {
    throw std::runtime_error(
      "Failed to dispatch kernel for fc_bf_tiled_kernel_default");
    return;
  }

  /// @todo synchronize when only needed
  blas_cc->command_queue_inst_.enqueueSVMMap(output, M * N * sizeof(uint16_t),
                                             true);
  if (!result) {
    throw std::runtime_error(
      "Failed to read output data for fc_bf_tiled_kernel_default");
    return;
  }
}

void sgemv_q6_k_cl(void *matAdata, float *vecXdata, float *vecYdata,
                   unsigned int M, unsigned int N) {
  bool result = false;

  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));

  ClContext::SharedPtrClKernel kernel_q6_k_sgemv_ptr;

  kernel_q6_k_sgemv_ptr =
    blas_cc->registerClKernel(q6_k_sgemv_kernel, "kernel_mul_mv_q6_K_f32");

  if (!kernel_q6_k_sgemv_ptr) {
    ml_loge("Failed to register kernel_q6_k_sgemv_ptr");
    return;
  }

  const size_t q6k_bytes = 210 * M * N / 256;

  result = blas_cc->command_queue_inst_.enqueueSVMUnmap(matAdata);
  if (!result) {
    ml_loge("Failed to write data to input buffer A for kernel_q6_k_sgemv_ptr");
    return;
  }

  result = blas_cc->command_queue_inst_.enqueueSVMUnmap(vecXdata);
  if (!result) {
    ml_loge("Failed to write data to input buffer B for kernel_q6_k_sgemv_ptr");
    return;
  }

  int ne00 = M; // number of rows in matrix X
  int ne01 = N; // number of columns in matrix X
  int ne02 = 1; // number of channels in matrix X
  int ne10 = M; // number of rows in vector A
  int ne11 = 1; // number of columns in vector A
  int ne12 = 1; // number of channels in vector A
  int ne13 = 1; // number of channels in vector A (Need to check)
  int ne0 = N;  // number of rows in output vector Y
  int ne1 = 1;  // number of columns in output vector Y

  int r2 = 1; // number of batches in vector A
  int r3 = 1; // number of batches in matrix X

  int nth0 = 2;
  int nth1 = 16;

  cl_ulong offset0 = 0;
  cl_ulong offset1 = 0;
  cl_ulong offsetd = 0;

  result = kernel_q6_k_sgemv_ptr->SetKernelSVMArguments(0, matAdata);

  if (!result) {
    ml_loge("Failed to set kernel argument 0 for kernel_q6_k_sgemv_ptr");
    return;
  }

  result =
    kernel_q6_k_sgemv_ptr->SetKernelArguments(1, &offset0, sizeof(cl_ulong));

  if (!result) {
    ml_loge("Failed to set kernel argument 1 for kernel_q6_k_sgemv_ptr");
    return;
  }

  result = kernel_q6_k_sgemv_ptr->SetKernelSVMArguments(2, vecXdata);

  if (!result) {
    ml_loge("Failed to set kernel argument 2 for kernel_q6_k_sgemv_ptr");
    return;
  }

  result =
    kernel_q6_k_sgemv_ptr->SetKernelArguments(3, &offset1, sizeof(cl_ulong));

  if (!result) {
    ml_loge("Failed to set kernel argument 3 for kernel_q6_k_sgemv_ptr");
    return;
  }

  result = kernel_q6_k_sgemv_ptr->SetKernelSVMArguments(4, vecYdata);

  if (!result) {
    ml_loge("Failed to set kernel argument 4 for kernel_q6_k_sgemv_ptr");
    return;
  }

  result =
    kernel_q6_k_sgemv_ptr->SetKernelArguments(5, &offsetd, sizeof(cl_ulong));

  if (!result) {
    ml_loge("Failed to set kernel argument 5 for kernel_q6_k_sgemv_ptr");
    return;
  }

  result = kernel_q6_k_sgemv_ptr->SetKernelArguments(6, &ne00, sizeof(int));

  if (!result) {
    ml_loge("Failed to set kernel argument 6 for kernel_q6_k_sgemv_ptr");
    return;
  }

  result = kernel_q6_k_sgemv_ptr->SetKernelArguments(7, &ne01, sizeof(int));

  if (!result) {
    ml_loge("Failed to set kernel argument 7 for kernel_q6_k_sgemv_ptr");
    return;
  }

  result = kernel_q6_k_sgemv_ptr->SetKernelArguments(8, &ne02, sizeof(int));

  if (!result) {
    ml_loge("Failed to set kernel argument 8 for kernel_q6_k_sgemv_ptr");
    return;
  }

  result = kernel_q6_k_sgemv_ptr->SetKernelArguments(9, &ne10, sizeof(int));

  if (!result) {
    ml_loge("Failed to set kernel argument 9 for kernel_q6_k_sgemv_ptr");
    return;
  }

  result = kernel_q6_k_sgemv_ptr->SetKernelArguments(10, &ne12, sizeof(int));

  if (!result) {
    ml_loge("Failed to set kernel argument 10 for kernel_q6_k_sgemv_ptr");
    return;
  }

  result = kernel_q6_k_sgemv_ptr->SetKernelArguments(11, &ne0, sizeof(int));

  if (!result) {
    ml_loge("Failed to set kernel argument 11 for kernel_q6_k_sgemv_ptr");
    return;
  }

  result = kernel_q6_k_sgemv_ptr->SetKernelArguments(12, &ne1, sizeof(int));

  if (!result) {
    ml_loge("Failed to set kernel argument 12 for kernel_q6_k_sgemv_ptr");
    return;
  }

  result = kernel_q6_k_sgemv_ptr->SetKernelArguments(13, &r2, sizeof(int));

  if (!result) {
    ml_loge("Failed to set kernel argument 13 for kernel_q6_k_sgemv_ptr");
    return;
  }

  result = kernel_q6_k_sgemv_ptr->SetKernelArguments(14, &r3, sizeof(int));

  if (!result) {
    ml_loge("Failed to set kernel argument 14 for kernel_q6_k_sgemv_ptr");
    return;
  }

#define N_SIMDWIDTH 16
#define N_SIMDGROUP 2

  const int work_groups_count[3] = {((ne0 + N_SIMDGROUP - 1) / N_SIMDGROUP) *
                                      (N_SIMDGROUP * N_SIMDWIDTH),
                                    ne1, 1};
  /// @todo: create a group size by device & input
  const int work_group_size[3] = {32, 1, 1};

  result = opencl::CommandQueueManager::Global().DispatchCommand(
    kernel_q6_k_sgemv_ptr, work_groups_count, work_group_size);
  if (!result) {
    ml_loge("Failed to dispatch kernel q6_k_sgemv");
    return;
  }

  result = blas_cc->command_queue_inst_.enqueueSVMMap(vecYdata,
                                                      N * sizeof(float), true);

  if (!result) {
    ml_loge(
      "Failed to read data from the output buffer for kernel_q6_k_sgemv_ptr");

    return;
  }
}

void sgemv_cl(const float *matAdata, const float *vecXdata, float *vecYdata,
              bool TransA, unsigned int dim1, unsigned int dim2,
              unsigned int lda) {
  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));

  ClContext::SharedPtrClKernel kernel_sgemv_ptr;

  if (TransA) {
    kernel_sgemv_ptr = blas_cc->registerClKernel(sgemv_kernel, "sgemv_cl");
  } else {
    kernel_sgemv_ptr =
      blas_cc->registerClKernel(sgemv_no_trans_kernel, "sgemv_cl_noTrans");
  }

  if (!kernel_sgemv_ptr) {
    return;
  }

  sgemv_cl_internal<float>(kernel_sgemv_ptr, matAdata, vecXdata, vecYdata, dim1,
                           dim2, lda);
}

float dot_cl(const float *vecAdata, const float *vecXdata, unsigned int dim1) {
  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));

  ClContext::SharedPtrClKernel kernel_dot_ptr =
    blas_cc->registerClKernel(dot_kernel, "dot_cl");
  if (!kernel_dot_ptr) {
    return {};
  }

  return dot_cl_internal<float>(kernel_dot_ptr, vecAdata, vecXdata, dim1);
}

void sgemm_cl(bool TransA, bool TransB, const float *A, const float *B,
              float *C, unsigned int M, unsigned int N, unsigned int K,
              unsigned int lda, unsigned int ldb, unsigned int ldc) {
  std::string kernel_func_;
  std::string sgemm_cl_kernel_;

  if (!TransA && !TransB) {
    kernel_func_ = "sgemm_cl_noTrans";
    sgemm_cl_kernel_ = sgemm_no_trans_kernel;
  } else if (TransA && !TransB) {
    kernel_func_ = "sgemm_cl_transA";
    sgemm_cl_kernel_ = sgemm_trans_a_kernel;
  } else if (!TransA && TransB) {
    kernel_func_ = "sgemm_cl_transB";
    sgemm_cl_kernel_ = sgemm_trans_b_kernel;
  } else {
    kernel_func_ = "sgemm_cl_transAB";
    sgemm_cl_kernel_ = sgemm_trans_ab_kernel;
  }

  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));

  ClContext::SharedPtrClKernel kernel_sgemm_ptr =
    blas_cc->registerClKernel(sgemm_cl_kernel_, kernel_func_);
  if (!kernel_sgemm_ptr) {
    return;
  }

  sgemm_cl_internal<float>(kernel_sgemm_ptr, TransA, TransB, A, B, C, M, N, K,
                           lda, ldb, ldc);
}

void addition_cl(const float *input, float *res, unsigned int size_input,
                 unsigned int size_res) {
  bool result = false;
  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));

  ClContext::SharedPtrClKernel kernel_addition_ptr =
    blas_cc->registerClKernel(addition_kernel, "addition_cl");
  if (!kernel_addition_ptr) {
    return;
  }

  addition_cl_internal<float>(kernel_addition_ptr, input, res, size_input,
                              size_res);
}

void rmsnorm_cl(const float *input, const float *gamma, float *result,
                const float epsilon, unsigned int height, unsigned int width,
                bool use_svm) {
  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));

  ClContext::SharedPtrClKernel kernel_rmsnorm_ptr =
    blas_cc->registerClKernel(rmsnorm_kernel, "rmsnorm_cl");
  if (!kernel_rmsnorm_ptr) {
    return;
  }

  rmsnorm_cl_internal<float>(kernel_rmsnorm_ptr, input, gamma, result, epsilon,
                             height, width, use_svm);
}

void sscal_cl(float *X, const unsigned int N, const float alpha) {
  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));

  ClContext::SharedPtrClKernel kernel_ptr =
    blas_cc->registerClKernel(sscal_kernel, "sscal_cl");

  if (!kernel_ptr) {
    return;
  }

  sscal_cl_internal<float>(kernel_ptr, X, N, alpha);
}

void transpose_cl_axis(const float *in, float *res,
                       unsigned int input_batch_size,
                       unsigned int input_channels, unsigned int input_height,
                       unsigned int input_width, unsigned int axis) {
  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));

  ClContext::SharedPtrClKernel kernel_transpose_ptr;
  switch (axis) {
  case 0:
    kernel_transpose_ptr =
      blas_cc->registerClKernel(transpose_axis_0_kernel, "transpose_cl_axis0");
    break;
  case 1:
    kernel_transpose_ptr =
      blas_cc->registerClKernel(transpose_axis_1_kernel, "transpose_cl_axis1");
    break;
  case 2:
    kernel_transpose_ptr =
      blas_cc->registerClKernel(transpose_axis_2_kernel, "transpose_cl_axis2");
    break;
  default:
    throw std::invalid_argument("failed to register CL kernel");
    break;
  }
  if (!kernel_transpose_ptr) {
    return;
  }

  transpose_cl_axis_internal<float>(kernel_transpose_ptr, in, res,
                                    input_batch_size, input_channels,
                                    input_height, input_width, axis);
}

void flatten_block_q4_0_cl(const void *src, void *dst_q, void *dst_d,
                           unsigned int num_blocks) {
  bool result = false;

  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  auto &clbuffInstance = ClBufferManager::Global();

  ClContext::SharedPtrClKernel kernel_ptr = blas_cc->registerClKernel(
    convert_block_q4_0_kernel, "kernel_convert_block_q4_0_noshuffle");
  if (!kernel_ptr) {
    ml_loge("Failed to register kernel_ptr for flatten_block_q4_0_cl");
    return;
  }

  int argIdx = 0;

  result = kernel_ptr->SetKernelSVMArguments(argIdx++, src);
  if (!result) {
    ml_loge("Failed to set kernel argument 0 for flatten_block_q4_0_cl");
    return;
  }

  result =
    kernel_ptr->SetKernelSVMArguments(argIdx++, clbuffInstance.getSVMQuant());
  if (!result) {
    ml_loge("Failed to set kernel argument 1 for flatten_block_q4_0_cl");
    return;
  }

  result =
    kernel_ptr->SetKernelSVMArguments(argIdx++, clbuffInstance.getSVMScale());
  if (!result) {
    ml_loge("Failed to set kernel argument 2 for flatten_block_q4_0_cl");
    return;
  }

  const int work_groups_count[3] = {(int)num_blocks, 1, 1};
  const int work_group_size[3] = {64, 1, 1};

  result = blas_cc->command_queue_inst_.DispatchCommand(
    kernel_ptr, work_groups_count, work_group_size);
  if (!result) {
    ml_loge("Failed to dispatch kernel for flatten_block_q4_0_cl");
    return;
  }
}

void restore_block_q4_0_cl(const void *src_q, const void *src_d, void *dst,
                           unsigned int num_blocks) {
  bool result = false;

  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));

  ClContext::SharedPtrClKernel kernel_ptr = blas_cc->registerClKernel(
    convert_block_q4_0_kernel, "kernel_restore_block_q4_0");
  if (!kernel_ptr) {
    ml_loge("Failed to register kernel_ptr for restore_block_q4_0_cl");
    return;
  }

  int argIdx = 0;

  result = kernel_ptr->SetKernelSVMArguments(argIdx++, src_q);
  if (!result) {
    ml_loge("Failed to set kernel argument 0 for restore_block_q4_0_cl");
    return;
  }

  result = kernel_ptr->SetKernelSVMArguments(argIdx++, src_d);
  if (!result) {
    ml_loge("Failed to set kernel argument 1 for restore_block_q4_0_cl");
    return;
  }

  result = kernel_ptr->SetKernelSVMArguments(argIdx++, dst);
  if (!result) {
    ml_loge("Failed to set kernel argument 2 for restore_block_q4_0_cl");
    return;
  }

  const int work_groups_count[3] = {(int)num_blocks, 1, 1};
  const int work_group_size[3] = {1, 1, 1};

  result = blas_cc->command_queue_inst_.DispatchCommand(
    kernel_ptr, work_groups_count, work_group_size);
  if (!result) {
    ml_loge("Failed to dispatch kernel for restore_block_q4_0_cl");
    return;
  }
}

void transpose_32_16(float *data, int M, int K) {
  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  auto &clbuffInstance = ClBufferManager::Global();

  ClContext::SharedPtrClKernel kernel_ptr = blas_cc->registerClKernel(
    transpose_32bit_16bit_kernel, "kernel_transpose_32_16");
  if (!kernel_ptr) {
    throw std::runtime_error(
      "Failed to get kernel_ptr for kernel_transpose_32_16");
    return;
  }

  int extra_elements = M % 8;
  int padding = 0;
  if (extra_elements > 0) {
    padding = 8 - extra_elements;
  }

  int width = K / 4;
  int height = M / 4;
  if (height == 0) {
    height = 1;
  }
  int padded_height = (M + padding) / 4;

  int arg = 0;
  bool result = false;

  result = kernel_ptr->SetKernelSVMArguments(arg++, data);
  if (!result)
    throw std::runtime_error(
      "Failed to set kernel argument 0 for kernel_transpose_32_16");

  result =
    kernel_ptr->SetKernelSVMArguments(arg++, clbuffInstance.getSVMInput());

  if (!result)
    throw std::runtime_error(
      "Failed to set kernel argument 1 for kernel_transpose_32_16");

  result = kernel_ptr->SetKernelArguments(arg++, &height, sizeof(int));
  if (!result)
    throw std::runtime_error(
      "Failed to set kernel argument 2 for kernel_transpose_32_16");

  result = kernel_ptr->SetKernelArguments(arg++, &width, sizeof(int));
  if (!result)
    throw std::runtime_error(
      "Failed to set kernel argument 3 for kernel_transpose_32_16");

  result = kernel_ptr->SetKernelArguments(arg++, &padded_height, sizeof(int));
  if (!result)
    throw std::runtime_error(
      "Failed to set kernel argument 4 for kernel_transpose_32_16");

  const int work_groups_count[3] = {width, padded_height, 1};
  const int work_group_size[3] = {1, 16, 1};

  result = blas_cc->command_queue_inst_.DispatchCommand(
    kernel_ptr, work_groups_count, work_group_size);
  if (!result) {
    ml_loge("Failed to dispatch kernel for kernel_transpose_32_16");
    return;
  }
}

/** @todo Enable transpose_16 with proper fix.
void transpose_16(void *input, void *output, int width, int height,
                  int size_bytes, bool isQuant) {
  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  auto &clbuffInstance = ClBufferManager::Global();

  ClContext::SharedPtrClKernel kernel_ptr =
    blas_cc->registerClKernel(transpose_16bit_kernel,
    "kernel_transpose_16");
  if (!kernel_ptr) {
    throw std::runtime_error(
      "Failed to get kernel_ptr for kernel_transpose_16");
    return;
  }

  int arg = 0;
  bool result = false;

  if (isQuant) {
    kernel_ptr->SetKernelSVMArguments(arg++, clbuffInstance.getSVMQuant());
    kernel_ptr->SetKernelSVMArguments(arg++, clbuffInstance.getSVMQuantT());
  } else {
    kernel_ptr->SetKernelSVMArguments(arg++, clbuffInstance.getSVMScale());
    kernel_ptr->SetKernelSVMArguments(arg++, clbuffInstance.getSVMScaleT());
  }

  result = kernel_ptr->SetKernelArguments(arg++, &height, sizeof(int));
  if (!result)
    throw std::runtime_error(
      "Failed to set kernel argument 2 for kernel_transpose_16");

  result = kernel_ptr->SetKernelArguments(arg++, &width, sizeof(int));
  if (!result)
    throw std::runtime_error(
      "Failed to set kernel argument 3 for kernel_transpose_16");

  const int work_groups_count[3] = {width, height, 1};
  const int work_group_size[3] = {4, 16, 1};

  result = blas_cc->command_queue_inst_.DispatchCommand(
    kernel_ptr, work_groups_count, work_group_size);
  if (!result) {
    ml_loge("Failed to dispatch kernel for kernel_transpose_16");
    return;
  }
}
*/

// =============================================================================
// v8c (paper 8/4/4) host wrappers — channel-wise QINT4 weight + int8 activation
// dot_4x8packed_su_int with bias-subtraction trick. Validated 87% of Adreno 830
// dp4a peak (4499 GFLOP/s ffn_down, 4344 GFLOP/s ffn_gate at t4×8 LWS 4×16).
// All inputs are plain cl_mem (image2d_from_buffer or buffer), no SVM.
// =============================================================================

// Device caps for v8c dispatch (paper §3.4 device specialization), queried once.
// Used to cap the LWS work-group product to the device's max — so the tuned
// 4×16 sweet spot is honored on Adreno 830 (max 1024) yet auto-reduced on a
// device with a smaller max work-group instead of silently dropping to NULL.
static const tv::DeviceImageCaps &v8c_device_caps() {
  static const tv::DeviceImageCaps caps = []() {
    auto *cc =
      static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
    cl_device_id dev = cc ? cc->context_inst_.GetDeviceId() : nullptr;
    return tv::queryDeviceImageCaps(dev);
  }();
  return caps;
}

// Shared v8c LWS policy: preferred 4×16 (env NNTR_V8C_LWS overrides), capped to
// the device max work-group size and required to divide gws. Returns whether a
// valid LWS was chosen; out is filled when true.
// Runtime-settable v8c GEMM local work size override ({0,0} = use env default).
// Lets the in-process LWS sweep (NNTR_LWS_SWEEP in main.cpp) retune the local
// size between forwards without reloading the model.
static std::array<size_t, 2> g_v8c_lws_override = {0, 0};
void set_v8c_lws_override(int lx, int ly) {
  g_v8c_lws_override = {(size_t)(lx > 0 ? lx : 0), (size_t)(ly > 0 ? ly : 0)};
}

// Runtime weight-prefetch override for the in-process A/B (-1 = use env;
// 0 = none; 1 = 1-ahead; 2 = 2-ahead). The copt is appended per dispatch so
// switching this re-registers the matching compiled kernel without reloading.
static int g_v8c_prefetch_override = -1;
void set_v8c_prefetch_override(int mode) { g_v8c_prefetch_override = mode; }

// Runtime register-tile override (TM x TN) for the M>4 v8c GEMM, for the
// in-process A/B (e.g. 4x8 default vs 8x4 = 2x weight reuse). {0,0} = default
// 4,8. Both the copts (-DV8C_TM/-DV8C_TN) and the wrapper gws read this, so a
// matching kernel is compiled and dispatched. Requires M_pad % TM == 0.
static std::array<int, 2> g_v8c_tile_override = {0, 0};
void set_v8c_tile_override(int tm, int tn) {
  g_v8c_tile_override = {tm > 0 ? tm : 0, tn > 0 ? tn : 0};
}
static std::string g_v8c_tile_copt() {
  if (g_v8c_tile_override[0] > 0 && g_v8c_tile_override[1] > 0)
    return " -DV8C_TM=" + std::to_string(g_v8c_tile_override[0]) +
           " -DV8C_TN=" + std::to_string(g_v8c_tile_override[1]);
  return "";
}
static std::string g_v8c_prefetch_copt() {
  int m = g_v8c_prefetch_override;
  if (m < 0) { // use env (default path)
    const char *p2 = getenv("NNTR_V8C_PREFETCH2");
    if (p2 && p2[0] == '1') return " -DV8C_PREFETCH2";
    const char *p1 = getenv("NNTR_V8C_PREFETCH");
    if (p1 && p1[0] == '1') return " -DV8C_PREFETCH";
    return "";
  }
  if (m == 2) return " -DV8C_PREFETCH2";
  if (m == 1) return " -DV8C_PREFETCH";
  return "";
}

static bool v8c_pick_lws(size_t gws_x, size_t gws_y, std::array<size_t, 2> &out) {
  static const std::array<size_t, 2> env_pref = []() {
    const char *e = std::getenv("NNTR_V8C_LWS");
    size_t lx = 4, ly = 16; // swept sweet spot @ M=1024 (WG=64), 8.9× vs NULL.
    if (e) {
      long a = 0, b = 0;
      if (std::sscanf(e, "%ld,%ld", &a, &b) == 2 && a > 0 && b > 0) {
        lx = (size_t)a;
        ly = (size_t)b;
      }
    }
    return std::array<size_t, 2>{lx, ly};
  }();
  // Runtime override (set_v8c_lws_override) lets an in-process LWS sweep change
  // the local size without re-loading the model. {0,0} = use the env default.
  const std::array<size_t, 2> &pref =
    (g_v8c_lws_override[0] && g_v8c_lws_override[1]) ? g_v8c_lws_override
                                                     : env_pref;
  size_t ox = 0, oy = 0;
  bool ok = tv::select2dLws(gws_x, gws_y, pref[0], pref[1], v8c_device_caps(),
                            &ox, &oy);
  out = {ox, oy};
  return ok;
}

// Device-specialization gate (paper §3.4). NNTR_V8C_BUF=1 selects the
// BUFFER-LOAD v8c GEMM kernels (no sampled-image reads) for runtimes that
// advertise cl_khr_image2d_from_buffer but cannot compile integer-coordinate
// read_imageui (e.g. Intel NEO). Default 0 = image path (Adreno BIT-identical).
// Queried once per process.
static bool v8c_use_buffer_path() {
  static const bool use_buf = []() {
    const char *e = std::getenv("NNTR_V8C_BUF");
    return e && std::atoi(e) != 0;
  }();
  return use_buf;
}

// Compile options for the buffer-load v8c program. -DV8C_BUFFER_ONLY excludes
// the image-sampling kernel bodies; -cl-std=CL3.0 exposes the core OpenCL C
// 3.0 dot_4x8packed_* builtins (Intel NEO does not declare them under the
// default CL1.2 std, and the legacy cl_khr_integer_dot_product #pragma is
// ignored by its front-end). All v8c registration sites on the buffer path
// must pass the IDENTICAL string so the same cached program is reused.
static const char *kV8cBufCompileOpts = "-DV8C_BUFFER_ONLY -cl-std=CL3.0";

void gemm_int8_v8c_cl(cl_mem act_image, cl_mem weight_image, cl_mem scale_act,
                      cl_mem scale_wgt, cl_mem row_sum_act, cl_mem zp_act,
                      cl_mem row_sum_w_int4, cl_mem output_fp16, unsigned int M,
                      unsigned int N, unsigned int K) {
  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  // For the M=1 (M_pad=4) decode case the default TM=4 kernel burns ~4×
  // the work needed (3 zero-padded rows). Dispatch a TM=1 variant
  // instead. The caller passes M_pad here, not M, so M_pad <= 4 means
  // "the real input had 1 valid row" (no QINT4 model has padded prefill
  // with M_pad <= 4 except via the M=1 decode rounding).
  // Buffer path (NNTR_V8C_BUF=1): act_image/weight_image carry the raw cl_mem
  // buffers; widths derived from K (int4 weight texel = 32 K, act texel = 16 K).
  const bool use_buf = v8c_use_buffer_path();
  const char *kname =
    use_buf ? ((M <= 4) ? "v8c_gemm_int8_int4_m1_buf" : "v8c_gemm_int8_int4_buf")
            : ((M <= 4) ? "v8c_gemm_int8_int4_m1" : "v8c_gemm_int8_int4");
  // Buffer path compiles the program with -DV8C_BUFFER_ONLY so the
  // image-sampling kernel bodies are excluded (Intel NEO can't compile them).
  std::string copts = use_buf ? kV8cBufCompileOpts : "";
  // NNTR_V8C_KCLOCK: in-kernel cl_khr_kernel_clock measurement build (off by
  // default). "1" = time the full K-loop; "2" = also drop the dp4a (fetch-only
  // floor) to isolate fetch-stall from compute. Measurement-only; the printf is
  // emitted by one work-item per dispatch.
  // NNTR_V8C_KCLOCK loop-decomposition modes (in-kernel clock_read_device):
  //   1 full | 2 no-dp4a (fetch+unpack) | 3 no-weight-fetch | 4 no-act-fetch |
  //   5 compute-only (no fetch). Diff of modes isolates weight- vs act-fetch
  //   vs pure compute. Synthetic runtime values suppress a stream w/o DCE.
  // Env-gated diagnostic/experimental compile flags — process-constant, so read
  // ONCE (static), not per dispatch. NNTR_V8C_KCLOCK {1..5} (in-kernel clock
  // decomposition), NNTR_V8C_PREFETCH=1 (1-ahead weight prefetch), NNTR_V8C_MFAST
  // =1 (M-fast dispatch order). All bit-identical / default-off.
  static const std::string v8c_env_copts = []() {
    std::string s;
    if (const char *kc = getenv("NNTR_V8C_KCLOCK")) {
      const char m = kc[0];
      if (m >= '1' && m <= '5')
        s += " -DV8C_KCLOCK";
      if (m == '2')
        s += " -DV8C_KCLOCK_NOCOMPUTE";
      if (m == '3' || m == '5')
        s += " -DV8C_KCLOCK_NOWFETCH";
      if (m == '4' || m == '5')
        s += " -DV8C_KCLOCK_NOAFETCH";
    }
    // NOTE: prefetch (-DV8C_PREFETCH / -DV8C_PREFETCH2) is appended per-call
    // below (g_v8c_prefetch_copt) so the in-process A/B can switch it at
    // runtime without reloading; it is NOT baked into this static string.
    if (const char *mf = getenv("NNTR_V8C_MFAST"))
      if (mf[0] == '1')
        s += " -DV8C_MFAST";
    // Wall-time fetch/compute decomposition (no in-kernel clock, no printf):
    // suppress the weight / activation fetch or the dp4a compute on their own
    // so the GEMM stage wall time isolates each contribution. Numerics become
    // garbage (synthetic fetch values) — measurement-only, off by default.
    if (const char *e = getenv("NNTR_V8C_NOWFETCH"))
      if (e[0] == '1')
        s += " -DV8C_KCLOCK_NOWFETCH";
    if (const char *e = getenv("NNTR_V8C_NOAFETCH"))
      if (e[0] == '1')
        s += " -DV8C_KCLOCK_NOAFETCH";
    if (const char *e = getenv("NNTR_V8C_NOCOMPUTE"))
      if (e[0] == '1')
        s += " -DV8C_KCLOCK_NOCOMPUTE";
    return s;
  }();
  static const bool v8c_mfast = []() {
    const char *mf = getenv("NNTR_V8C_MFAST");
    return mf && mf[0] == '1';
  }();
  copts += v8c_env_copts;
  copts += g_v8c_prefetch_copt(); // runtime-switchable prefetch depth (A/B)
  copts += g_v8c_tile_copt();     // runtime-switchable TMxTN register tile (A/B)
  ClContext::SharedPtrClKernel kp =
    blas_cc->registerClKernel(int8_int4_gemm_v8c_kernel, kname, copts);
  if (!kp)
    throw std::runtime_error(std::string("v8c_gemm: registerClKernel failed: ") +
                             kname);

  int arg = 0;
  if (!kp->SetKernelArguments(arg++, &act_image, sizeof(cl_mem)))
    throw std::runtime_error("v8c gemm arg 0 (act_image)");
  if (!kp->SetKernelArguments(arg++, &weight_image, sizeof(cl_mem)))
    throw std::runtime_error("v8c gemm arg 1 (weight_image)");
  if (!kp->SetKernelArguments(arg++, &scale_act, sizeof(cl_mem)))
    throw std::runtime_error("v8c gemm arg 2 (scale_act)");
  if (!kp->SetKernelArguments(arg++, &scale_wgt, sizeof(cl_mem)))
    throw std::runtime_error("v8c gemm arg 3 (scale_wgt)");
  if (!kp->SetKernelArguments(arg++, &row_sum_act, sizeof(cl_mem)))
    throw std::runtime_error("v8c gemm arg 4 (row_sum_act)");
  if (!kp->SetKernelArguments(arg++, &zp_act, sizeof(cl_mem)))
    throw std::runtime_error("v8c gemm arg 5 (zp_act)");
  if (!kp->SetKernelArguments(arg++, &row_sum_w_int4, sizeof(cl_mem)))
    throw std::runtime_error("v8c gemm arg 6 (row_sum_w_int4)");
  if (!kp->SetKernelArguments(arg++, &output_fp16, sizeof(cl_mem)))
    throw std::runtime_error("v8c gemm arg 7 (output_fp16)");
  int Mi = (int)M, Ni = (int)N, Ki = (int)K;
  if (!kp->SetKernelArguments(arg++, &Mi, sizeof(int)))
    throw std::runtime_error("v8c gemm arg 8 (M)");
  if (!kp->SetKernelArguments(arg++, &Ni, sizeof(int)))
    throw std::runtime_error("v8c gemm arg 9 (N)");
  if (!kp->SetKernelArguments(arg++, &Ki, sizeof(int)))
    throw std::runtime_error("v8c gemm arg 10 (K)");
  if (use_buf) {
    int W_act = (int)(K / 16), W_wgt = (int)(K / 32);
    if (!kp->SetKernelArguments(arg++, &W_act, sizeof(int)))
      throw std::runtime_error("v8c gemm arg 11 (W_act)");
    if (!kp->SetKernelArguments(arg++, &W_wgt, sizeof(int)))
      throw std::runtime_error("v8c gemm arg 12 (W_wgt)");
  }

  // V8C_TM=4, V8C_TN=8 (defaults in the .cl). global = (N/TN, M/TM); kernel
  // requires M%4=0, N%8=0, K%32=0. The historic 4×16 LWS is only valid when
  // gws[1] = M/TM is a multiple of 16, i.e. M is a multiple of 64. The
  // QINT4 caller pads M up to V8C_TM (=4) for the M=1 decode / M=18 prefill
  // cases, so M/TM can be as small as 1 or 5 and never satisfies that
  // constraint. Pass NULL lws so the runtime picks a compatible
  // workgroup shape — the kernel has no in-flight cross-thread sync, so
  // any LWS that divides gws works. (OpenCL 3.0 on Adreno 830 allows
  // non-uniform groups, but staying uniform with a runtime-chosen LWS is
  // the conservative win.)
  // Tag this dispatch's profile entry with its shape (N,K uniquely identify
  // each transformer FC: q/k/v/o/gate/up/down) so NNTR_OPENCL_PROFILING splits
  // the aggregate v8c_gemm_int8_int4 number per GEMM shape. Host-only; consumed
  // by the next enqueueKernel; no effect on kernel output.
  {
    char lbl[48];
    snprintf(lbl, sizeof(lbl), ":N%u_K%u", N, K);
    blas_cc->command_queue_inst_.setNextProfileLabel(lbl);
  }
  if (M <= 4) {
    // M=1 (TM=1) GEMV-style dispatch: 1-D grid over output channels.
    constexpr size_t TN_M1 = 8;
    std::array<size_t, 3> gws = {(size_t)N / TN_M1, 1, 1};
    blas_cc->command_queue_inst_.enqueueKernel(
      kp->GetKernel(), 1, gws.data(), nullptr, 0, nullptr, nullptr);
  } else {
    // Tile defaults 4x8; the runtime override (set_v8c_tile_override) must
    // match the -DV8C_TM/-DV8C_TN copts appended above so kernel + grid agree.
    const size_t TM = g_v8c_tile_override[0] > 0 ? (size_t)g_v8c_tile_override[0]
                                                 : 4;
    const size_t TN = g_v8c_tile_override[1] > 0 ? (size_t)g_v8c_tile_override[1]
                                                 : 8;
    // NNTR_V8C_MFAST: swap so M/TM is the fast-varying (dim0) axis; the kernel
    // (-DV8C_MFAST) reads m0 from gid0, n0 from gid1 to match. Weight-reuse
    // dispatch order. Default keeps {N/TN, M/TM}.
    std::array<size_t, 3> gws = v8c_mfast
                                  ? std::array<size_t, 3>{(size_t)M / TM,
                                                          (size_t)N / TN, 1}
                                  : std::array<size_t, 3>{(size_t)N / TN,
                                                          (size_t)M / TM, 1};
    // CL-event profiling showed the historic NULL lws (driver-chosen
    // workgroup) lands the GEMM at ~14% of dp4a peak in-forward, vs 87%
    // in the standalone microbench which used a tuned LWS. Pick a device-
    // specialized LWS (4×16 sweet spot, capped to the device max work-group
    // size and required to divide gws); fall back to NULL when none fits
    // (small-M prefill). NNTR_V8C_LWS="lx,ly" overrides.
    std::array<size_t, 2> picked{};
    const bool lws_ok = v8c_pick_lws(gws[0], gws[1], picked);
    std::array<size_t, 3> lws = {picked[0], picked[1], 1};
    blas_cc->command_queue_inst_.enqueueKernel(
      kp->GetKernel(), 2, gws.data(), lws_ok ? lws.data() : nullptr, 0, nullptr,
      nullptr);
  }
}

void gemm_int8_v8c_v_ohwi_cl(cl_mem act_image, cl_mem weight_image,
                             cl_mem scale_act, cl_mem scale_wgt,
                             cl_mem row_sum_act, cl_mem zp_act,
                             cl_mem row_sum_w_int4, cl_mem v_ohwi,
                             unsigned int M_pad, unsigned int N, unsigned int K,
                             unsigned int head_dim, unsigned int S_max,
                             unsigned int position, unsigned int M_real) {
  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  if (M_pad == 0 || N == 0 || K == 0 || head_dim == 0 || S_max == 0)
    throw std::runtime_error("gemm_int8_v8c_v_ohwi: zero dim");
  constexpr unsigned int V8C_TM = 4, V8C_TN = 8;
  if (M_pad % V8C_TM != 0 || N % V8C_TN != 0 || K % 32 != 0)
    throw std::runtime_error("gemm_int8_v8c_v_ohwi: M/N/K not aligned");
  if (head_dim % V8C_TN != 0)
    throw std::runtime_error("gemm_int8_v8c_v_ohwi: head_dim % TN != 0");
  if (N % head_dim != 0)
    throw std::runtime_error("gemm_int8_v8c_v_ohwi: N % head_dim != 0");
  if (M_real > M_pad) M_real = M_pad;

  const bool use_buf = v8c_use_buffer_path();
  ClContext::SharedPtrClKernel kp = blas_cc->registerClKernel(
    int8_int4_gemm_v8c_kernel,
    use_buf ? "v8c_gemm_int8_int4_v_ohwi_buf" : "v8c_gemm_int8_int4_v_ohwi",
    use_buf ? kV8cBufCompileOpts : "");
  if (!kp)
    throw std::runtime_error("v8c_v_ohwi: registerClKernel failed");

  int arg = 0;
  if (!kp->SetKernelArguments(arg++, &act_image, sizeof(cl_mem)) ||
      !kp->SetKernelArguments(arg++, &weight_image, sizeof(cl_mem)) ||
      !kp->SetKernelArguments(arg++, &scale_act, sizeof(cl_mem)) ||
      !kp->SetKernelArguments(arg++, &scale_wgt, sizeof(cl_mem)) ||
      !kp->SetKernelArguments(arg++, &row_sum_act, sizeof(cl_mem)) ||
      !kp->SetKernelArguments(arg++, &zp_act, sizeof(cl_mem)) ||
      !kp->SetKernelArguments(arg++, &row_sum_w_int4, sizeof(cl_mem)) ||
      !kp->SetKernelArguments(arg++, &v_ohwi, sizeof(cl_mem)))
    throw std::runtime_error("v8c_v_ohwi: cl_mem arg failed");
  int Mi = (int)M_pad, Ni = (int)N, Ki = (int)K;
  int di = (int)head_dim, Si = (int)S_max, pi = (int)position;
  int Mr = (int)M_real;
  if (!kp->SetKernelArguments(arg++, &Mi, sizeof(int)) ||
      !kp->SetKernelArguments(arg++, &Ni, sizeof(int)) ||
      !kp->SetKernelArguments(arg++, &Ki, sizeof(int)) ||
      !kp->SetKernelArguments(arg++, &di, sizeof(int)) ||
      !kp->SetKernelArguments(arg++, &Si, sizeof(int)) ||
      !kp->SetKernelArguments(arg++, &pi, sizeof(int)) ||
      !kp->SetKernelArguments(arg++, &Mr, sizeof(int)))
    throw std::runtime_error("v8c_v_ohwi: int arg failed");
  if (use_buf) {
    int W_act = (int)(K / 16), W_wgt = (int)(K / 32);
    if (!kp->SetKernelArguments(arg++, &W_act, sizeof(int)) ||
        !kp->SetKernelArguments(arg++, &W_wgt, sizeof(int)))
      throw std::runtime_error("v8c_v_ohwi: width arg failed");
  }
  std::array<size_t, 3> gws = {(size_t)N / V8C_TN, (size_t)M_pad / V8C_TM, 1};
  blas_cc->command_queue_inst_.enqueueKernel(
    kp->GetKernel(), 2, gws.data(), nullptr, 0, nullptr, nullptr);
}

static void quantize_act_v8c_cl_impl(cl_mem act_in, cl_mem out_int8,
                                     cl_mem out_scale, cl_mem out_zp,
                                     cl_mem out_row_sum, unsigned int M,
                                     unsigned int K, const char *kernel_name,
                                     bool parallel) {
  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  // The act-quant kernels are image-free, but they live in the same program
  // string as the image GEMMs. On the buffer path we must build that program
  // with -DV8C_BUFFER_ONLY too, otherwise this (option-less) build compiles
  // the image kernels and fails on Intel NEO. Matches the GEMM wrappers so the
  // same cached program is reused.
  const std::string copts = v8c_use_buffer_path() ? kV8cBufCompileOpts : "";
  ClContext::SharedPtrClKernel kp =
    blas_cc->registerClKernel(int8_int4_gemm_v8c_kernel, kernel_name, copts);
  if (!kp)
    throw std::runtime_error(std::string("v8c quant: registerClKernel failed: ") +
                             kernel_name);
  int arg = 0;
  if (!kp->SetKernelArguments(arg++, &act_in, sizeof(cl_mem)))
    throw std::runtime_error("v8c quant arg 0");
  if (!kp->SetKernelArguments(arg++, &out_int8, sizeof(cl_mem)))
    throw std::runtime_error("v8c quant arg 1");
  if (!kp->SetKernelArguments(arg++, &out_scale, sizeof(cl_mem)))
    throw std::runtime_error("v8c quant arg 2");
  if (!kp->SetKernelArguments(arg++, &out_zp, sizeof(cl_mem)))
    throw std::runtime_error("v8c quant arg 3 (out_zp)");
  if (!kp->SetKernelArguments(arg++, &out_row_sum, sizeof(cl_mem)))
    throw std::runtime_error("v8c quant arg 4");
  int Mi = (int)M, Ki = (int)K;
  if (!kp->SetKernelArguments(arg++, &Mi, sizeof(int)))
    throw std::runtime_error("v8c quant arg 5");
  if (!kp->SetKernelArguments(arg++, &Ki, sizeof(int)))
    throw std::runtime_error("v8c quant arg 6");

  if (parallel) {
    // _par variant: one workgroup (LWS=64) per row. gws = M * 64.
    constexpr size_t LWS = 64;
    std::array<size_t, 3> gws = {(size_t)M * LWS, 1, 1};
    std::array<size_t, 3> lws = {LWS, 1, 1};
    blas_cc->command_queue_inst_.enqueueKernel(
      kp->GetKernel(), 1, gws.data(), lws.data(), 0, nullptr, nullptr);
  } else {
    std::array<size_t, 3> gws = {(size_t)M, 1, 1};
    blas_cc->command_queue_inst_.enqueueKernel(
      kp->GetKernel(), 1, gws.data(), nullptr, 0, nullptr, nullptr);
  }
}

void quantize_act_v8c_fp16_cl(cl_mem act_fp16, cl_mem out_int8, cl_mem out_scale,
                              cl_mem out_zp, cl_mem out_row_sum, unsigned int M,
                              unsigned int K) {
  // The _par variant requires K % LWS == 0 (LWS=64). Qwen3 hidden dims
  // (1024/2048/3072/etc.) all satisfy this; smaller K paths fall back.
  const bool can_par = (K % 64 == 0);
  quantize_act_v8c_cl_impl(act_fp16, out_int8, out_scale, out_zp, out_row_sum, M,
                           K, can_par ? "v8c_act_quant_f16_par"
                                      : "v8c_act_quant_f16",
                           can_par);
}

void quantize_act_v8c_fp32_cl(cl_mem act_fp32, cl_mem out_int8, cl_mem out_scale,
                              cl_mem out_zp, cl_mem out_row_sum, unsigned int M,
                              unsigned int K) {
  const bool can_par = (K % 64 == 0);
  quantize_act_v8c_cl_impl(act_fp32, out_int8, out_scale, out_zp, out_row_sum, M,
                           K, can_par ? "v8c_act_quant_f32_par"
                                      : "v8c_act_quant_f32",
                           can_par);
}

// fp16 (uint16 bit pattern) → fp32 host helper. Standard IEEE 754 half decode.
static inline float h2f_host(uint16_t h) {
  uint32_t s = (uint32_t)(h & 0x8000u) << 16;
  uint32_t e = (h >> 10) & 0x1fu;
  uint32_t m = h & 0x3ffu;
  uint32_t o;
  if (e == 0) {
    if (m == 0)
      o = s;
    else {
      e = 1;
      while ((m & 0x400u) == 0) {
        m <<= 1;
        e--;
      }
      m &= 0x3ffu;
      o = s | ((e + 112) << 23) | (m << 13);
    }
  } else if (e == 0x1f) {
    o = s | 0x7f800000u | (m << 13);
  } else {
    o = s | ((e + 112) << 23) | (m << 13);
  }
  float f;
  std::memcpy(&f, &o, 4);
  return f;
}

} // namespace nntrainer

// Use Int4Utils for osv32 row dequant; keep include inside the helper file
// (TU-local) to avoid leaking the dependency through the public header.
#include "../int4_utils.h"
#include <cmath>
#include <cstring>
#include <vector>

namespace nntrainer {

std::unique_ptr<tv::TensorBacking>
make_v8c_weight_backing(const uint8_t *osv32_packed,
                        const uint16_t *fp16_scales, unsigned int group_size,
                        unsigned int N, unsigned int K,
                        cl_mem *out_scale_buf) {
  if (K % 32 != 0)
    throw std::invalid_argument(
      "make_v8c_weight_backing: K must be a multiple of 32");

  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  cl_context ctx =
    blas_cc->context_inst_.GetContext(); // OpenCL context handle

  // 1) Walk osv32 row by row, dequantize → re-quantize per-channel (paper §4.2)
  //    Pack into row-major v8c layout: per row N, K/2 bytes; per K-block of 32
  //    one 16-byte texel; byte offset = (k_outer/32)*16 + c*4 + b within row.
  //    Byte content: low_nib = (q[k_outer+c*8+b] + 8) & 0xF,
  //                  high_nib = (q[k_outer+c*8+b+4] + 8) & 0xF.
  const size_t row_bytes = (size_t)K / 2;
  std::vector<uint8_t> packed((size_t)N * row_bytes);
  std::vector<float> per_channel_scale(N);
  std::vector<float> deq_row((size_t)K);

  // dequantizePackedRow takes non-const pointers; cast safely (it reads only).
  uint8_t *osv_nonconst = const_cast<uint8_t *>(osv32_packed);
  uint16_t *scales_nonconst = const_cast<uint16_t *>(fp16_scales);

  for (unsigned int n = 0; n < N; ++n) {
    // (a) Dequantize this row from osv32 + per-group scales → fp32 [K].
    Int4Utils::dequantizePackedRow(osv_nonconst, scales_nonconst,
                                   /*rows_count*/ N, /*cols_count*/ K,
                                   /*group_size*/ group_size,
                                   /*row_index*/ n, deq_row.data());
    // (b) Compute per-channel scale = max|.| / 7
    float amax = 0.0f;
    for (unsigned int k = 0; k < K; ++k) {
      float a = std::fabs(deq_row[k]);
      if (a > amax)
        amax = a;
    }
    float s = (amax > 0.0f) ? (amax / 7.0f) : 1.0f;
    per_channel_scale[n] = s;
    float inv_s = 1.0f / s;
    // (c) Re-quantize to int4 in [-7,7], offset-encode, pack to v8c layout
    uint8_t *out_row = packed.data() + (size_t)n * row_bytes;
    for (unsigned int k_outer = 0; k_outer < K; k_outer += 32) {
      for (int c = 0; c < 4; ++c) {
        for (int b = 0; b < 4; ++b) {
          unsigned int kLo = k_outer + (unsigned int)(c * 8 + b);
          unsigned int kHi = kLo + 4;
          int qL = (int)std::lrint(deq_row[kLo] * inv_s);
          int qH = (int)std::lrint(deq_row[kHi] * inv_s);
          qL = qL < -7 ? -7 : (qL > 7 ? 7 : qL);
          qH = qH < -7 ? -7 : (qH > 7 ? 7 : qH);
          uint8_t lo = (uint8_t)((qL + 8) & 0xF);
          uint8_t hi = (uint8_t)((qH + 8) & 0xF);
          size_t off = (size_t)(k_outer / 32) * 16 + (size_t)(c * 4 + b);
          out_row[off] = (uint8_t)((hi << 4) | lo);
        }
      }
    }
  }

  // 2) Upload packed weight to a new cl_mem buffer, wrap in TensorBacking.
  cl_int err = CL_SUCCESS;
  cl_mem w_buf =
    clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                   packed.size(), packed.data(), &err);
  if (err != CL_SUCCESS || !w_buf)
    throw std::runtime_error("make_v8c_weight_backing: clCreateBuffer (weight) "
                             "failed: " +
                             std::to_string(err));
  auto backing = std::make_unique<tv::TensorBacking>(
    ctx, w_buf, tv::Encoding::INT4_OFFSET, tv::Layout::ROW_MAJOR, packed.size(),
    /*owned=*/true);

  // 3) Upload per-channel scale buffer.
  cl_mem sb = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                             sizeof(float) * N, per_channel_scale.data(), &err);
  if (err != CL_SUCCESS || !sb)
    throw std::runtime_error("make_v8c_weight_backing: clCreateBuffer (scale) "
                             "failed: " +
                             std::to_string(err));
  *out_scale_buf = sb;
  return backing;
}

std::unique_ptr<tv::TensorBacking>
make_v8c_weight_backing_from_kai_section_a(const uint8_t *section_a,
                                           const uint16_t *fp16_scales,
                                           unsigned int N, unsigned int K,
                                           cl_mem *out_scale_buf,
                                           cl_mem *out_row_sum_w_int4_buf) {
  if (K % 32 != 0)
    throw std::invalid_argument(
      "make_v8c_weight_backing_from_kai_section_a: K must be a multiple of 32");
  if (N % 4 != 0)
    throw std::invalid_argument("make_v8c_weight_backing_from_kai_section_a: "
                                "N must be a multiple of KAI nr=4");

  // Constants from the KAI Section A producer (Int4Utils::quantizeAndPackKai).
  constexpr size_t KAI_NR = 4;             // output-channel super-row width
  constexpr size_t KAI_KR_BY_SR = 8;       // bytes per nr-block (= KR/SR = 16/2)
  constexpr size_t KAI_K_INTERLEAVE = 16;  // lo/hi nibble K stride within a byte

  const size_t k_blocks = K / 32;
  const size_t super_row_count = N / KAI_NR;
  const size_t nibble_bytes_per_super_row = KAI_NR * (K / 2);
  // Bytes consumed by one 32-K span across all 4 channels:
  //   4 nr * 8 bytes (super_block_a, K=[0..7] lo + K=[16..23] hi) +
  //   4 nr * 8 bytes (super_block_b, K=[8..15] lo + K=[24..31] hi) = 64
  constexpr size_t KAI_BYTES_PER_KBLK = 2 * KAI_NR * KAI_KR_BY_SR;

  // v8c per-channel row buffer (row-major, K/2 bytes per channel).
  const size_t v8c_row_bytes = (size_t)K / 2;
  std::vector<uint8_t> packed((size_t)N * v8c_row_bytes);

  for (size_t sr = 0; sr < super_row_count; ++sr) {
    const uint8_t *sr_base = section_a + sr * nibble_bytes_per_super_row;
    for (size_t nr = 0; nr < KAI_NR; ++nr) {
      const size_t n = sr * KAI_NR + nr;
      uint8_t *v8c_row = packed.data() + n * v8c_row_bytes;
      for (size_t kblk = 0; kblk < k_blocks; ++kblk) {
        // KAI: this channel's 16 bytes for K=[kblk*32 .. kblk*32+31] live in
        // two 8-byte nr-blocks at kblk*64 + nr*8 (super_block_a) and
        // kblk*64 + 32 + nr*8 (super_block_b).
        const uint8_t *blk_a =
          sr_base + kblk * KAI_BYTES_PER_KBLK + nr * KAI_KR_BY_SR;
        const uint8_t *blk_b = blk_a + KAI_NR * KAI_KR_BY_SR; // +32 bytes
        // Decode to per-K offset-encoded nibbles (0..15 = int4+8). Both KAI
        // and v8c use the same +8 offset, but KAI bytes are XOR'd with 0x88
        // — undoing the XOR yields the raw uint4 pair.
        uint8_t k_nibbles[32];
        for (size_t i = 0; i < KAI_KR_BY_SR; ++i) {
          const uint8_t b_a = blk_a[i] ^ 0x88;
          k_nibbles[i + 0] = (uint8_t)(b_a & 0x0F);                // K = kblk*32 + i
          k_nibbles[i + KAI_K_INTERLEAVE] =                        // K = kblk*32 + 16 + i
            (uint8_t)((b_a >> 4) & 0x0F);
          const uint8_t b_b = blk_b[i] ^ 0x88;
          k_nibbles[i + KAI_KR_BY_SR] = (uint8_t)(b_b & 0x0F);     // K = kblk*32 + 8 + i
          k_nibbles[i + KAI_KR_BY_SR + KAI_K_INTERLEAVE] =         // K = kblk*32 + 24 + i
            (uint8_t)((b_b >> 4) & 0x0F);
        }
        // Repack to v8c order: byte(c*4+b) at v8c K-block offset = (qH<<4)|qL
        // where qL = K(c*8+b), qH = K(c*8+b+4). Already offset-encoded.
        uint8_t *v8c_kblk = v8c_row + kblk * 16;
        for (size_t c = 0; c < 4; ++c) {
          for (size_t b = 0; b < 4; ++b) {
            const uint8_t qL = k_nibbles[c * 8 + b];
            const uint8_t qH = k_nibbles[c * 8 + b + 4];
            v8c_kblk[c * 4 + b] = (uint8_t)((qH << 4) | qL);
          }
        }
      }
    }
  }

  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  cl_context ctx = blas_cc->context_inst_.GetContext();

  // Upload packed weight buffer.
  cl_int err = CL_SUCCESS;
  cl_mem w_buf =
    clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, packed.size(),
                   packed.data(), &err);
  if (err != CL_SUCCESS || !w_buf)
    throw std::runtime_error(
      "make_v8c_weight_backing_from_kai_section_a: clCreateBuffer (weight) "
      "failed: " +
      std::to_string(err));
  auto backing = std::make_unique<tv::TensorBacking>(
    ctx, w_buf, tv::Encoding::INT4_OFFSET, tv::Layout::ROW_MAJOR, packed.size(),
    /*owned=*/true);

  // Per-channel scale: promote fp16 → fp32 (v8c gemm reads scale_wgt as fp32).
  std::vector<float> per_channel_scale(N);
  for (unsigned int n = 0; n < N; ++n)
    per_channel_scale[n] = compute_fp16_to_fp32(fp16_scales[n]);
  cl_mem sb =
    clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                   sizeof(float) * N, per_channel_scale.data(), &err);
  if (err != CL_SUCCESS || !sb)
    throw std::runtime_error(
      "make_v8c_weight_backing_from_kai_section_a: clCreateBuffer (scale) "
      "failed: " +
      std::to_string(err));
  *out_scale_buf = sb;

  // Per-channel int4 row sum: Σ_k int4_w[n,k]. The GEMM kernel needs this to
  // back out the asymmetric act zero-point contribution (see kernel comment
  // above the bias-correction loop). Decode from the freshly-packed v8c
  // bytes (cheaper than re-walking the KAI Section A) — same int4 value, just
  // already in v8c byte order. byte(c*4+b) within each 32-K block has:
  //   lo nibble = uint4(K = c*8+b)  → int4 = lo - 8
  //   hi nibble = uint4(K = c*8+b+4) → int4 = hi - 8
  std::vector<int32_t> row_sum_w_int4(N, 0);
  for (unsigned int n = 0; n < N; ++n) {
    const uint8_t *row = packed.data() + n * v8c_row_bytes;
    int32_t s = 0;
    for (size_t off = 0; off < v8c_row_bytes; ++off) {
      const uint8_t byte = row[off];
      const int lo = (int)(byte & 0x0Fu) - 8;
      const int hi = (int)((byte >> 4) & 0x0Fu) - 8;
      s += lo + hi;
    }
    row_sum_w_int4[n] = s;
  }
  cl_mem rsw_buf =
    clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                   sizeof(int32_t) * N, row_sum_w_int4.data(), &err);
  if (err != CL_SUCCESS || !rsw_buf)
    throw std::runtime_error(
      "make_v8c_weight_backing_from_kai_section_a: clCreateBuffer "
      "(row_sum_w_int4) failed: " +
      std::to_string(err));
  *out_row_sum_w_int4_buf = rsw_buf;

  return backing;
}

// ---------------------------------------------------------------------------
// 8/4/4 paper attention path: int8(act) × int8(weight) channel-wise GEMM.
// ---------------------------------------------------------------------------
void gemm_int8_int8_v8c_cl(cl_mem act_image, cl_mem weight_image,
                           cl_mem scale_act, cl_mem scale_wgt,
                           cl_mem row_sum_act, cl_mem zp_act, cl_mem row_sum_w,
                           cl_mem output_fp16, unsigned int M, unsigned int N,
                           unsigned int K) {
  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  // Buffer path (NNTR_V8C_BUF=1): int8 weight texel = 16 K, act texel = 16 K.
  const bool use_buf = v8c_use_buffer_path();
  const char *kname =
    use_buf ? ((M <= 4) ? "v8c_gemm_int8_int8_m1_buf" : "v8c_gemm_int8_int8_buf")
            : ((M <= 4) ? "v8c_gemm_int8_int8_m1" : "v8c_gemm_int8_int8");
  const std::string copts = use_buf ? kV8cBufCompileOpts : "";
  ClContext::SharedPtrClKernel kp =
    blas_cc->registerClKernel(int8_int4_gemm_v8c_kernel, kname, copts);
  if (!kp)
    throw std::runtime_error(
      std::string("v8c_gemm_int8_int8: registerClKernel failed: ") + kname);

  int arg = 0;
  if (!kp->SetKernelArguments(arg++, &act_image, sizeof(cl_mem)) ||
      !kp->SetKernelArguments(arg++, &weight_image, sizeof(cl_mem)) ||
      !kp->SetKernelArguments(arg++, &scale_act, sizeof(cl_mem)) ||
      !kp->SetKernelArguments(arg++, &scale_wgt, sizeof(cl_mem)) ||
      !kp->SetKernelArguments(arg++, &row_sum_act, sizeof(cl_mem)) ||
      !kp->SetKernelArguments(arg++, &zp_act, sizeof(cl_mem)) ||
      !kp->SetKernelArguments(arg++, &row_sum_w, sizeof(cl_mem)) ||
      !kp->SetKernelArguments(arg++, &output_fp16, sizeof(cl_mem)))
    throw std::runtime_error("v8c_gemm_int8_int8: cl_mem arg failed");
  int Mi = (int)M, Ni = (int)N, Ki = (int)K;
  if (!kp->SetKernelArguments(arg++, &Mi, sizeof(int)) ||
      !kp->SetKernelArguments(arg++, &Ni, sizeof(int)) ||
      !kp->SetKernelArguments(arg++, &Ki, sizeof(int)))
    throw std::runtime_error("v8c_gemm_int8_int8: int arg failed");
  if (use_buf) {
    int W_act = (int)(K / 16), W_wgt = (int)(K / 16);
    if (!kp->SetKernelArguments(arg++, &W_act, sizeof(int)) ||
        !kp->SetKernelArguments(arg++, &W_wgt, sizeof(int)))
      throw std::runtime_error("v8c_gemm_int8_int8: width arg failed");
  }

  if (M <= 4) {
    constexpr size_t TN_M1 = 8;
    std::array<size_t, 3> gws = {(size_t)N / TN_M1, 1, 1};
    blas_cc->command_queue_inst_.enqueueKernel(kp->GetKernel(), 1, gws.data(),
                                               nullptr, 0, nullptr, nullptr);
  } else {
    constexpr size_t TM = 4, TN = 8;
    std::array<size_t, 3> gws = {(size_t)N / TN, (size_t)M / TM, 1};
    // Device-specialized LWS (shared with the int4 path): 4×16 sweet spot
    // capped to device max work-group size + must divide gws, else NULL.
    std::array<size_t, 2> picked{};
    const bool lws_ok = v8c_pick_lws(gws[0], gws[1], picked);
    std::array<size_t, 3> lws = {picked[0], picked[1], 1};
    blas_cc->command_queue_inst_.enqueueKernel(
      kp->GetKernel(), 2, gws.data(), lws_ok ? lws.data() : nullptr, 0, nullptr,
      nullptr);
  }
}

void gemm_int8_int8_v8c_v_ohwi_cl(cl_mem act_image, cl_mem weight_image,
                                  cl_mem scale_act, cl_mem scale_wgt,
                                  cl_mem row_sum_act, cl_mem zp_act,
                                  cl_mem row_sum_w, cl_mem v_ohwi,
                                  unsigned int M_pad, unsigned int N,
                                  unsigned int K, unsigned int head_dim,
                                  unsigned int S_max, unsigned int position,
                                  unsigned int M_real) {
  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  if (M_pad == 0 || N == 0 || K == 0 || head_dim == 0 || S_max == 0)
    throw std::runtime_error("gemm_int8_int8_v8c_v_ohwi: zero dim");
  constexpr unsigned int V8C_TM = 4, V8C_TN = 8;
  if (M_pad % V8C_TM != 0 || N % V8C_TN != 0 || K % 32 != 0)
    throw std::runtime_error("gemm_int8_int8_v8c_v_ohwi: M/N/K not aligned");
  if (head_dim % V8C_TN != 0 || N % head_dim != 0)
    throw std::runtime_error("gemm_int8_int8_v8c_v_ohwi: head_dim constraint");
  if (M_real > M_pad) M_real = M_pad;

  const bool use_buf = v8c_use_buffer_path();
  ClContext::SharedPtrClKernel kp = blas_cc->registerClKernel(
    int8_int4_gemm_v8c_kernel,
    use_buf ? "v8c_gemm_int8_int8_v_ohwi_buf" : "v8c_gemm_int8_int8_v_ohwi",
    use_buf ? kV8cBufCompileOpts : "");
  if (!kp)
    throw std::runtime_error("v8c_int8_v_ohwi: registerClKernel failed");

  int arg = 0;
  if (!kp->SetKernelArguments(arg++, &act_image, sizeof(cl_mem)) ||
      !kp->SetKernelArguments(arg++, &weight_image, sizeof(cl_mem)) ||
      !kp->SetKernelArguments(arg++, &scale_act, sizeof(cl_mem)) ||
      !kp->SetKernelArguments(arg++, &scale_wgt, sizeof(cl_mem)) ||
      !kp->SetKernelArguments(arg++, &row_sum_act, sizeof(cl_mem)) ||
      !kp->SetKernelArguments(arg++, &zp_act, sizeof(cl_mem)) ||
      !kp->SetKernelArguments(arg++, &row_sum_w, sizeof(cl_mem)) ||
      !kp->SetKernelArguments(arg++, &v_ohwi, sizeof(cl_mem)))
    throw std::runtime_error("v8c_int8_v_ohwi: cl_mem arg failed");
  int Mi = (int)M_pad, Ni = (int)N, Ki = (int)K;
  int di = (int)head_dim, Si = (int)S_max, pi = (int)position, Mr = (int)M_real;
  if (!kp->SetKernelArguments(arg++, &Mi, sizeof(int)) ||
      !kp->SetKernelArguments(arg++, &Ni, sizeof(int)) ||
      !kp->SetKernelArguments(arg++, &Ki, sizeof(int)) ||
      !kp->SetKernelArguments(arg++, &di, sizeof(int)) ||
      !kp->SetKernelArguments(arg++, &Si, sizeof(int)) ||
      !kp->SetKernelArguments(arg++, &pi, sizeof(int)) ||
      !kp->SetKernelArguments(arg++, &Mr, sizeof(int)))
    throw std::runtime_error("v8c_int8_v_ohwi: int arg failed");
  if (use_buf) {
    int W_act = (int)(K / 16), W_wgt = (int)(K / 16);
    if (!kp->SetKernelArguments(arg++, &W_act, sizeof(int)) ||
        !kp->SetKernelArguments(arg++, &W_wgt, sizeof(int)))
      throw std::runtime_error("v8c_int8_v_ohwi: width arg failed");
  }
  std::array<size_t, 3> gws = {(size_t)N / V8C_TN, (size_t)M_pad / V8C_TM, 1};
  blas_cc->command_queue_inst_.enqueueKernel(kp->GetKernel(), 2, gws.data(),
                                             nullptr, 0, nullptr, nullptr);
}

std::unique_ptr<tv::TensorBacking>
make_v8c_int8_weight_backing(const int8_t *int8_weights,
                             const uint16_t *fp16_scales, unsigned int N,
                             unsigned int K, cl_mem *out_scale_buf,
                             cl_mem *out_row_sum_w_buf) {
  if (K % 16 != 0)
    throw std::invalid_argument("make_v8c_int8_weight_backing: K%16!=0");
  if (N % 4 != 0)
    throw std::invalid_argument("make_v8c_int8_weight_backing: N%4!=0");

  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  cl_context ctx = blas_cc->context_inst_.GetContext();
  const size_t nbytes = (size_t)N * K; // 1 byte/int8 weight, plain row-major

  cl_int err = CL_SUCCESS;
  cl_mem w_buf =
    clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, nbytes,
                   const_cast<int8_t *>(int8_weights), &err);
  if (err != CL_SUCCESS || !w_buf)
    throw std::runtime_error(
      "make_v8c_int8_weight_backing: clCreateBuffer (weight) failed: " +
      std::to_string(err));
  auto backing = std::make_unique<tv::TensorBacking>(
    ctx, w_buf, tv::Encoding::INT8, tv::Layout::ROW_MAJOR, nbytes,
    /*owned=*/true);

  std::vector<float> per_channel_scale(N);
  for (unsigned int n = 0; n < N; ++n)
    per_channel_scale[n] = compute_fp16_to_fp32(fp16_scales[n]);
  cl_mem sb = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                             sizeof(float) * N, per_channel_scale.data(), &err);
  if (err != CL_SUCCESS || !sb)
    throw std::runtime_error(
      "make_v8c_int8_weight_backing: clCreateBuffer (scale) failed");
  *out_scale_buf = sb;

  std::vector<int32_t> row_sum_w(N, 0);
  for (unsigned int n = 0; n < N; ++n) {
    const int8_t *row = int8_weights + (size_t)n * K;
    int32_t s = 0;
    for (unsigned int k = 0; k < K; ++k) s += (int)row[k];
    row_sum_w[n] = s;
  }
  cl_mem rb = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                             sizeof(int32_t) * N, row_sum_w.data(), &err);
  if (err != CL_SUCCESS || !rb)
    throw std::runtime_error(
      "make_v8c_int8_weight_backing: clCreateBuffer (row_sum_w) failed");
  *out_row_sum_w_buf = rb;

  return backing;
}

} // namespace nntrainer
