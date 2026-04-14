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

#include "util_func.h"
#include <fp16.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>

namespace nntrainer {

namespace {

// ---------------------------------------------------------------------------
// Phase 6: per-substage profiler for gemm_int4_adreno_cl (prefill M>1 path)
//
// Phase 5 showed that HalfTensor::dotQInteger spent ~90% of prefill time in
// `gpu_call` (i.e. everything inside this wrapper). This breaks that 90% into
// the individual OpenCL substages so we can tell whether the bottleneck is:
//   - per-call cl_mem create / cl_image create (6 objects/call, 252 calls)
//   - kernel arg setup
//   - the two DispatchCommand calls (input_transpose + gpu_int4_gemm_adreno)
//   - the blocking SVMMap at the end
//   - per-call clReleaseMemObject (4 objects/call)
//
// Only the M>1 path is instrumented; M=1 decode skips this entirely because
// substage overhead is tiny on single-row GEMV.
//
// Dumped at process exit via a static dtor (same pattern as Phase 5
// g_half_dotq_profile in half_tensor.cpp).
// ---------------------------------------------------------------------------
struct Int4AdrenoGemmProfile {
  std::atomic<uint64_t> calls{0};

  // --- host-side stage times (wall-clock, Phase 6) ----------------------
  std::atomic<uint64_t> ns_cl_mem_create{0};  // 2x clCreateBuffer + 2x clCreateImage
  std::atomic<uint64_t> ns_xt_setup{0};       // registerClKernel + 4x SetKernelArguments
  std::atomic<uint64_t> ns_xt_dispatch{0};    // DispatchCommand(input_transpose) - host side
  std::atomic<uint64_t> ns_gemm_setup{0};     // registerClKernel + 8x SetKernelArguments
  std::atomic<uint64_t> ns_gemm_dispatch{0};  // DispatchCommand(gpu_int4_gemm_adreno) - host side
  std::atomic<uint64_t> ns_svm_map_sync{0};   // enqueueSVMMap(blocking=true) -- drains pipeline
  std::atomic<uint64_t> ns_cl_mem_release{0}; // 4x clReleaseMemObject

  // --- device-side kernel times (Phase 7, clGetEventProfilingInfo) ------
  // Measured from cl_event's CL_PROFILING_COMMAND_START / _END which the
  // Adreno driver fills in ns. svm_map_sync stays as the host-wait time;
  // these two give the actual GPU execution cost of each kernel.
  //
  // With CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE these two kernels MAY
  // overlap (the runtime still enforces the implicit cl_mem write-read
  // dependency for Adreno in practice, but timings are per-event), so
  // (ns_xt_gpu + ns_gemm_gpu) can exceed ns_svm_map_sync if they overlap,
  // or fall short if there is GPU-idle gap between them.
  std::atomic<uint64_t> ns_xt_gpu{0};         // input_transpose device execution
  std::atomic<uint64_t> ns_gemm_gpu{0};       // gpu_int4_gemm_adreno device execution
  std::atomic<uint64_t> ns_prof_query{0};     // host time spent in clGetEventProfilingInfo

  ~Int4AdrenoGemmProfile() {
    const uint64_t c = calls.load();
    if (c == 0)
      return;

    const uint64_t mem_c = ns_cl_mem_create.load();
    const uint64_t xts = ns_xt_setup.load();
    const uint64_t xtd = ns_xt_dispatch.load();
    const uint64_t gms = ns_gemm_setup.load();
    const uint64_t gmd = ns_gemm_dispatch.load();
    const uint64_t svm = ns_svm_map_sync.load();
    const uint64_t mem_r = ns_cl_mem_release.load();
    const uint64_t prof = ns_prof_query.load();

    const uint64_t xt_gpu = ns_xt_gpu.load();
    const uint64_t gemm_gpu = ns_gemm_gpu.load();

    const uint64_t total = mem_c + xts + xtd + gms + gmd + svm + mem_r + prof;
    const double total_ms = total / 1.0e6;

    auto pct = [&](uint64_t v) -> double {
      return total == 0 ? 0.0 : (double)v / (double)total * 100.0;
    };
    auto ms = [](uint64_t v) -> double { return v / 1.0e6; };

    std::fprintf(stderr,
                 "\n[PROFILE gemm_int4_adreno_cl prefill (M>1)] "
                 "total=%.2f ms calls=%llu\n",
                 total_ms, (unsigned long long)c);
    std::fprintf(stderr,
                 "  cl_mem_create    : %8.2f ms (%5.1f%%)  "
                 "[2x clCreateBuffer + 2x clCreateImage]\n",
                 ms(mem_c), pct(mem_c));
    std::fprintf(stderr,
                 "  xt_setup         : %8.2f ms (%5.1f%%)  "
                 "[registerClKernel + 4x SetKernelArgs (input_transpose)]\n",
                 ms(xts), pct(xts));
    std::fprintf(stderr,
                 "  xt_dispatch      : %8.2f ms (%5.1f%%)  "
                 "[DispatchCommand(input_transpose) host-side]\n",
                 ms(xtd), pct(xtd));
    std::fprintf(stderr,
                 "  gemm_setup       : %8.2f ms (%5.1f%%)  "
                 "[registerClKernel + 8x SetKernelArgs (gpu_int4_gemm_adreno)]\n",
                 ms(gms), pct(gms));
    std::fprintf(stderr,
                 "  gemm_dispatch    : %8.2f ms (%5.1f%%)  "
                 "[DispatchCommand(gpu_int4_gemm_adreno) host-side]\n",
                 ms(gmd), pct(gmd));
    std::fprintf(stderr,
                 "  svm_map_sync     : %8.2f ms (%5.1f%%)  "
                 "[enqueueSVMMap(output, blocking=true) host-wait]\n",
                 ms(svm), pct(svm));
    std::fprintf(stderr,
                 "  cl_mem_release   : %8.2f ms (%5.1f%%)  "
                 "[4x clReleaseMemObject]\n",
                 ms(mem_r), pct(mem_r));
    std::fprintf(stderr,
                 "  prof_query       : %8.2f ms (%5.1f%%)  "
                 "[4x clGetEventProfilingInfo + 2x clReleaseEvent]\n",
                 ms(prof), pct(prof));
    std::fprintf(stderr,
                 "  --- device-side (event profiling, may overlap host "
                 "stages above) ---\n");
    std::fprintf(stderr,
                 "  xt_gpu           : %8.2f ms  "
                 "[input_transpose CL_PROFILING_COMMAND_START..END]\n",
                 ms(xt_gpu));
    std::fprintf(stderr,
                 "  gemm_gpu         : %8.2f ms  "
                 "[gpu_int4_gemm_adreno CL_PROFILING_COMMAND_START..END]\n",
                 ms(gemm_gpu));
    const uint64_t gpu_sum = xt_gpu + gemm_gpu;
    std::fprintf(stderr,
                 "  gpu_sum          : %8.2f ms  "
                 "[xt_gpu + gemm_gpu; compare to svm_map_sync above]\n",
                 ms(gpu_sum));
  }
};

Int4AdrenoGemmProfile g_int4_gemm_profile;

inline uint64_t now_ns_phase6() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
           std::chrono::steady_clock::now().time_since_epoch())
    .count();
}

} // namespace

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

void gemm_int4_adreno_cl(uint16_t *input, uint16_t *input_transposed,
                         uint16_t *weights, uint16_t *scales, uint16_t *output,
                         unsigned int M, unsigned int N, unsigned int K) {
  // Adopted from origin/main's `gemm_int4_cl_adreno`. The pipeline:
  //   1. Wrap `input` (host-prepared fp16 [M][alignK]) as a cl_mem
  //      backed by CL_MEM_USE_HOST_PTR, then expose it as an
  //      image1d_buffer_t (GPU image cache). Same for `input_transposed`
  //      which is the destination of the transpose pass.
  //   2. Dispatch the `input_transpose` kernel: GPU does the transpose
  //      that we used to do on CPU, eliminating the in_convert
  //      transpose loop bottleneck.
  //   3. Dispatch `gpu_int4_gemm_adreno`, reading the transposed image
  //      via read_imageh -- texture cache makes this much faster than
  //      the previous SVM half * load path.
  //   4. Release the per-call cl_mem / cl_image objects.
  //
  // Caller pre-fills `input[i]` (i in [0 .. M*alignK)) with fp16 of the
  // row-major activation, and pre-allocates `input_transposed` as an
  // SVM region of align(M, 4) * alignK fp16 elements.
  //
  // Layout / divisibility assumptions (Qwen3-4B FC widths satisfy all):
  //   K %  4 == 0 (channelwise int4 packs 4 nibbles per ushort)
  //   N %  4 == 0 (gemm kernel writes 4 channels per work-item)
  //   alignK = K (we use channel-wise quant where group_size = K)
  if (((N % 4) != 0) | ((K % 4) != 0)) {
    throw std::runtime_error(
      "gemm_int4_adreno_cl requires N and K to be multiples of 4");
  }

  const int q_group_size = static_cast<int>(K); // per-channel
  const int alignK = static_cast<int>(align(K, q_group_size));

  bool result = false;
  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));

  cl_int err;
  cl_image_format image_format;
  image_format.image_channel_order = CL_RGBA;
  image_format.image_channel_data_type = CL_HALF_FLOAT;

  cl_image_desc image_desc;

  // Phase 6 sub-stage profiler: per-call cumulative substage times accumulate
  // into g_int4_gemm_profile, which dumps a one-shot breakdown at process
  // exit. Every call to this wrapper is an M>1 prefill call, so no guard is
  // needed. See the profiler struct at the top of this file.
  //
  // Phase 7 adds per-kernel device-side timing via cl_event profiling; the
  // two events below are queried after the blocking SVMMap (which drains
  // the pipeline, guaranteeing the events have completed).
  cl_event xt_event = nullptr;
  cl_event gemm_event = nullptr;
  const uint64_t t_mem_c0 = now_ns_phase6();

  // ---- input image (host-prepared fp16 [M][alignK]) ----
  const size_t input_buf_bytes =
    static_cast<size_t>(M) * static_cast<size_t>(alignK) * sizeof(uint16_t);
  cl_mem input_buf =
    clCreateBuffer(blas_cc->context_inst_.GetContext(),
                   CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, input_buf_bytes,
                   input, &err);
  if (err != CL_SUCCESS) {
    throw std::runtime_error("Failed to create input cl_mem buffer");
  }

  std::memset(&image_desc, 0, sizeof(image_desc));
  image_desc.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
  image_desc.image_width =
    static_cast<size_t>(M) * static_cast<size_t>(alignK) / 4;
  image_desc.buffer = input_buf;

  cl_mem input_img =
    clCreateImage(blas_cc->context_inst_.GetContext(), CL_MEM_READ_ONLY,
                  &image_format, &image_desc, nullptr, &err);
  if (err != CL_SUCCESS) {
    clReleaseMemObject(input_buf);
    throw std::runtime_error("Failed to create image1d_buffer for input");
  }

  // ---- input_transposed image (GPU-written, then read by gemm) ----
  const size_t input_t_buf_bytes = static_cast<size_t>(align(M, 4)) *
                                   static_cast<size_t>(alignK) *
                                   sizeof(uint16_t);
  cl_mem input_t_buf =
    clCreateBuffer(blas_cc->context_inst_.GetContext(),
                   CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, input_t_buf_bytes,
                   input_transposed, &err);
  if (err != CL_SUCCESS) {
    clReleaseMemObject(input_img);
    clReleaseMemObject(input_buf);
    throw std::runtime_error("Failed to create input_transposed cl_mem");
  }

  std::memset(&image_desc, 0, sizeof(image_desc));
  image_desc.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
  image_desc.image_width =
    static_cast<size_t>(align(M, 4)) * static_cast<size_t>(alignK) / 4;
  image_desc.buffer = input_t_buf;

  cl_mem input_t_img =
    clCreateImage(blas_cc->context_inst_.GetContext(), CL_MEM_READ_WRITE,
                  &image_format, &image_desc, nullptr, &err);
  if (err != CL_SUCCESS) {
    clReleaseMemObject(input_t_buf);
    clReleaseMemObject(input_img);
    clReleaseMemObject(input_buf);
    throw std::runtime_error(
      "Failed to create image1d_buffer for input_transposed");
  }

  const uint64_t t_mem_c1 = now_ns_phase6();
  g_int4_gemm_profile.ns_cl_mem_create += t_mem_c1 - t_mem_c0;

  // ---- step 1: input_transpose kernel ----
  ClContext::SharedPtrClKernel xt_kernel =
    blas_cc->registerClKernel(input_transpose_kernel, "input_transpose");
  if (!xt_kernel) {
    clReleaseMemObject(input_t_img);
    clReleaseMemObject(input_t_buf);
    clReleaseMemObject(input_img);
    clReleaseMemObject(input_buf);
    throw std::runtime_error("Failed to register input_transpose kernel");
  }

  {
    int arg = 0;
    result = xt_kernel->SetKernelArguments(arg++, &input_img, sizeof(cl_mem));
    if (!result)
      throw std::runtime_error("input_transpose arg 0 (input image)");

    result =
      xt_kernel->SetKernelArguments(arg++, &input_t_img, sizeof(cl_mem));
    if (!result)
      throw std::runtime_error("input_transpose arg 1 (input_transposed)");

    int alignK_4 = alignK >> 2;
    result = xt_kernel->SetKernelArguments(arg++, &alignK_4, sizeof(int));
    if (!result)
      throw std::runtime_error("input_transpose arg 2 (alignK_4)");

    int M_4 = static_cast<int>(ceilDiv(M, 4u));
    result = xt_kernel->SetKernelArguments(arg++, &M_4, sizeof(int));
    if (!result)
      throw std::runtime_error("input_transpose arg 3 (M_4)");

    const int xt_global[3] = {alignK_4, M_4, 1};
    const int xt_local[3] = {1, 128, 1};

    const uint64_t t_xt_setup_end = now_ns_phase6();
    g_int4_gemm_profile.ns_xt_setup += t_xt_setup_end - t_mem_c1;

    result = blas_cc->command_queue_inst_.DispatchCommand(
      xt_kernel, xt_global, xt_local, &xt_event);
    if (!result)
      throw std::runtime_error("Failed to dispatch input_transpose");

    const uint64_t t_xt_dispatch_end = now_ns_phase6();
    g_int4_gemm_profile.ns_xt_dispatch += t_xt_dispatch_end - t_xt_setup_end;
  }

  const uint64_t t_gemm_setup_start = now_ns_phase6();

  // ---- step 2: gpu_int4_gemm_adreno kernel (texture input) ----
  ClContext::SharedPtrClKernel gemm_kernel = blas_cc->registerClKernel(
    int4_gemm_adreno_kernel, "gpu_int4_gemm_adreno");
  if (!gemm_kernel) {
    clReleaseMemObject(input_t_img);
    clReleaseMemObject(input_t_buf);
    clReleaseMemObject(input_img);
    clReleaseMemObject(input_buf);
    throw std::runtime_error(
      "Failed to register gpu_int4_gemm_adreno kernel");
  }

  {
    int arg = 0;
    result =
      gemm_kernel->SetKernelArguments(arg++, &input_t_img, sizeof(cl_mem));
    if (!result)
      throw std::runtime_error(
        "gpu_int4_gemm_adreno arg 0 (input_transposed image)");

    result = gemm_kernel->SetKernelSVMArguments(arg++, scales);
    if (!result)
      throw std::runtime_error("gpu_int4_gemm_adreno arg 1 (scales)");

    result = gemm_kernel->SetKernelSVMArguments(arg++, output);
    if (!result)
      throw std::runtime_error("gpu_int4_gemm_adreno arg 2 (output)");

    result = gemm_kernel->SetKernelSVMArguments(arg++, weights);
    if (!result)
      throw std::runtime_error("gpu_int4_gemm_adreno arg 3 (weights)");

    int size_k = static_cast<int>(K);
    int size_n = static_cast<int>(N);
    int size_m = static_cast<int>(M);
    int qg = q_group_size;

    result = gemm_kernel->SetKernelArguments(arg++, &size_k, sizeof(int));
    if (!result)
      throw std::runtime_error("gpu_int4_gemm_adreno arg 4 (K)");
    result = gemm_kernel->SetKernelArguments(arg++, &size_n, sizeof(int));
    if (!result)
      throw std::runtime_error("gpu_int4_gemm_adreno arg 5 (N)");
    result = gemm_kernel->SetKernelArguments(arg++, &size_m, sizeof(int));
    if (!result)
      throw std::runtime_error("gpu_int4_gemm_adreno arg 6 (M)");
    result = gemm_kernel->SetKernelArguments(arg++, &qg, sizeof(int));
    if (!result)
      throw std::runtime_error(
        "gpu_int4_gemm_adreno arg 7 (quantization_group_size)");

    // Dispatch:
    //   global = (ceilDiv(M, 8), N/4, 1)
    //     each WI handles 8 m tokens (m = global_id(0)*2, two half4 reads
    //     along M) and 4 n channels (n = global_id(1)*4)
    //   local  = {1, 128, 1}
    //     2 wavefronts per work-group on Adreno 830 (wave=64). Matches
    //     the main branch's tuned dispatch.
    const int gemm_global[3] = {static_cast<int>(ceilDiv(M, 8u)),
                                static_cast<int>(N) / 4, 1};
    const int gemm_local[3] = {1, 128, 1};

    const uint64_t t_gemm_setup_end = now_ns_phase6();
    g_int4_gemm_profile.ns_gemm_setup += t_gemm_setup_end - t_gemm_setup_start;

    result = blas_cc->command_queue_inst_.DispatchCommand(
      gemm_kernel, gemm_global, gemm_local, &gemm_event);
    if (!result)
      throw std::runtime_error("Failed to dispatch gpu_int4_gemm_adreno");

    const uint64_t t_gemm_dispatch_end = now_ns_phase6();
    g_int4_gemm_profile.ns_gemm_dispatch +=
      t_gemm_dispatch_end - t_gemm_setup_end;
  }

  const uint64_t t_svm_start = now_ns_phase6();

  // sync output back to host
  blas_cc->command_queue_inst_.enqueueSVMMap(
    output, static_cast<size_t>(M) * static_cast<size_t>(N) * sizeof(uint16_t),
    true);

  const uint64_t t_svm_end = now_ns_phase6();
  g_int4_gemm_profile.ns_svm_map_sync += t_svm_end - t_svm_start;

  // Phase 7: query per-kernel device-side execution time.
  //
  // The blocking SVMMap above drained the pipeline, so both events are
  // guaranteed to be CL_COMPLETE by now. Event timestamps are device-side
  // ns and do NOT depend on the host clock we use for the wall stages
  // above. Accumulate into g_int4_gemm_profile.ns_xt_gpu / ns_gemm_gpu.
  //
  // Failure to query (e.g. if the queue was created without
  // CL_QUEUE_PROFILING_ENABLE) is not fatal -- we just get zeros.
  {
    cl_ulong xt_start = 0, xt_end = 0, gm_start = 0, gm_end = 0;
    if (xt_event) {
      clGetEventProfilingInfo(xt_event, CL_PROFILING_COMMAND_START,
                              sizeof(cl_ulong), &xt_start, nullptr);
      clGetEventProfilingInfo(xt_event, CL_PROFILING_COMMAND_END,
                              sizeof(cl_ulong), &xt_end, nullptr);
    }
    if (gemm_event) {
      clGetEventProfilingInfo(gemm_event, CL_PROFILING_COMMAND_START,
                              sizeof(cl_ulong), &gm_start, nullptr);
      clGetEventProfilingInfo(gemm_event, CL_PROFILING_COMMAND_END,
                              sizeof(cl_ulong), &gm_end, nullptr);
    }
    if (xt_end >= xt_start)
      g_int4_gemm_profile.ns_xt_gpu += (xt_end - xt_start);
    if (gm_end >= gm_start)
      g_int4_gemm_profile.ns_gemm_gpu += (gm_end - gm_start);

    if (xt_event)
      clReleaseEvent(xt_event);
    if (gemm_event)
      clReleaseEvent(gemm_event);
  }

  const uint64_t t_prof_end = now_ns_phase6();
  g_int4_gemm_profile.ns_prof_query += t_prof_end - t_svm_end;

  // cleanup per-call cl_mem / cl_image objects
  clReleaseMemObject(input_t_img);
  clReleaseMemObject(input_t_buf);
  clReleaseMemObject(input_img);
  clReleaseMemObject(input_buf);

  const uint64_t t_mem_r_end = now_ns_phase6();
  g_int4_gemm_profile.ns_cl_mem_release += t_mem_r_end - t_prof_end;

  g_int4_gemm_profile.calls++;
}

void gemv_int4_adreno_cl(uint16_t *input, uint16_t *weights, uint16_t *scales,
                         uint16_t *output, unsigned int K, unsigned int N) {
  bool result = false;
  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));

  ClContext::SharedPtrClKernel kernel_ptr = blas_cc->registerClKernel(
    int4_gemv_adreno_kernel, "gpu_int4_gemv_adreno");
  if (!kernel_ptr) {
    throw std::runtime_error(
      "Failed to get kernel_ptr for gpu_int4_gemv_adreno");
    return;
  }

  int arg = 0;

  result = kernel_ptr->SetKernelSVMArguments(arg++, input);
  if (!result)
    throw std::runtime_error(
      "Failed to set kernel argument 0 (input) for gpu_int4_gemv_adreno");

  result = kernel_ptr->SetKernelSVMArguments(arg++, scales);
  if (!result)
    throw std::runtime_error(
      "Failed to set kernel argument 1 (scales) for gpu_int4_gemv_adreno");

  result = kernel_ptr->SetKernelSVMArguments(arg++, output);
  if (!result)
    throw std::runtime_error(
      "Failed to set kernel argument 2 (output) for gpu_int4_gemv_adreno");

  result = kernel_ptr->SetKernelSVMArguments(arg++, weights);
  if (!result)
    throw std::runtime_error(
      "Failed to set kernel argument 3 (weights) for gpu_int4_gemv_adreno");

  int size_k = static_cast<int>(K);
  int size_n = static_cast<int>(N);

  result = kernel_ptr->SetKernelArguments(arg++, &size_k, sizeof(int));
  if (!result)
    throw std::runtime_error(
      "Failed to set kernel argument 4 (K) for gpu_int4_gemv_adreno");

  result = kernel_ptr->SetKernelArguments(arg++, &size_n, sizeof(int));
  if (!result)
    throw std::runtime_error(
      "Failed to set kernel argument 5 (N) for gpu_int4_gemv_adreno");

  // Dispatch:
  //   global = align_N / 4 work-items along dim 0
  //            (each WI handles 4 output channels)
  //   local  = {16, 1, 1}
  //
  // dim_n divisibility: align_N is N rounded up to 32, so dim_n = align_N/4
  // is at least a multiple of 8. For Qwen3-4B FC widths the values are
  // dim_n in {256, 640, 1024, 2432}, all multiples of 16.
  const int align_N = static_cast<int>(align(N, 32));
  const int dim_n = align_N / 4;

  const int work_groups_count[3] = {dim_n, 1, 1};
  const int work_group_size[3] = {16, 1, 1};

  result = blas_cc->command_queue_inst_.DispatchCommand(
    kernel_ptr, work_groups_count, work_group_size);
  if (!result) {
    throw std::runtime_error(
      "Failed to dispatch kernel for gpu_int4_gemv_adreno");
    return;
  }

  /// @todo synchronize when only needed
  blas_cc->command_queue_inst_.enqueueSVMMap(
    output, static_cast<size_t>(N) * sizeof(uint16_t), true);
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
} // namespace nntrainer
