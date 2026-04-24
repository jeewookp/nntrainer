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
#include <gpu_image_pool.h>
#include <profile_gate.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <unordered_map>
#include <vector>

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
    if (nntrainer::prefill_profile_suppressed())
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

// Phase 2 diagnostic: split gemv_int4_adreno_cl per-call cost into
// arg setup, kernel dispatch, and blocking SVM map.  Decode path only
// (gemv_int4_adreno_cl is the M=1 kernel entry point), so no prefill
// gate.  Prints at process exit via static destructor.
//
// Phase 3 addition: cl_event-based kernel compute timing.  The
// wall-clock dispatch bucket measures the HOST time around
// clEnqueueNDRangeKernel (typically tiny since it returns async),
// and the svmmap bucket measures the BLOCKING host wait for the
// queue to drain.  Neither directly tells us how long the GPU
// actually spent inside the kernel.  Attaching a cl_event on
// DispatchCommand lets us read GPU-side PROFILING_START / END on
// the event and accumulate true compute time separately from host
// overhead.  Used for "is the kernel itself the hot path, or is it
// the sync?" questions without contaminating the measurement with
// the host-side wait.
struct GemvInt4AdrenoCallProfile {
  std::atomic<uint64_t> calls{0};
  std::atomic<uint64_t> ns_args{0};        // SetKernelSVMArguments/Arguments
  std::atomic<uint64_t> ns_dispatch{0};    // host wall around clEnqueueNDRangeKernel
  std::atomic<uint64_t> ns_svmmap{0};      // blocking clEnqueueSVMMap(read)
  std::atomic<uint64_t> ns_gpu_kernel{0};  // device PROFILING_START..END for kernel
  std::atomic<uint64_t> gpu_events{0};     // how many calls contributed a valid event

  ~GemvInt4AdrenoCallProfile() {
    const uint64_t c = calls.load();
    if (c == 0) return;
    const uint64_t args = ns_args.load();
    const uint64_t disp = ns_dispatch.load();
    const uint64_t smap = ns_svmmap.load();
    const uint64_t gpu  = ns_gpu_kernel.load();
    const uint64_t gpu_c = gpu_events.load();
    const uint64_t tot = args + disp + smap;
    auto ms = [](uint64_t v) { return v / 1.0e6; };
    auto pct = [&](uint64_t v) {
      return tot == 0 ? 0.0 : 100.0 * (double)v / (double)tot;
    };
    std::fprintf(stderr,
                 "\n[PROFILE gemv_int4_adreno_cl decode] calls=%llu "
                 "total=%.2f ms avg=%.3f ms\n",
                 (unsigned long long)c, ms(tot),
                 ms(tot) / (double)c);
    std::fprintf(stderr,
                 "  args_setup : %7.2f ms (%5.1f%%)  "
                 "[SetKernel{SVM,}Arguments]\n",
                 ms(args), pct(args));
    std::fprintf(stderr,
                 "  dispatch   : %7.2f ms (%5.1f%%)  "
                 "[host wall around clEnqueueNDRangeKernel]\n",
                 ms(disp), pct(disp));
    std::fprintf(stderr,
                 "  svmmap     : %7.2f ms (%5.1f%%)  "
                 "[blocking enqueueSVMMap(output) = queue flush + cache "
                 "invalidate]\n",
                 ms(smap), pct(smap));
    if (gpu_c > 0) {
      std::fprintf(stderr,
                   "  gpu_kernel : %7.2f ms             "
                   "[cl_event PROFILING_START..END, true GPU compute, "
                   "%llu events]\n",
                   ms(gpu), (unsigned long long)gpu_c);
    }
  }
};

static GemvInt4AdrenoCallProfile g_gemv_adreno_call_profile;

// Parallel profile for the incremental image2d gemv probe series
// (gpu_int4_gemv_image_v1 = step 1 = weight-load-only).  Only
// gpu_kernel matters for these builds -- the output is intentional
// garbage so "correct decode" numbers don't apply.
struct GemvInt4ImageCallProfile {
  std::atomic<uint64_t> calls{0};
  std::atomic<uint64_t> ns_dispatch{0};
  std::atomic<uint64_t> ns_svmmap{0};
  std::atomic<uint64_t> ns_gpu_kernel{0};
  std::atomic<uint64_t> gpu_events{0};

  ~GemvInt4ImageCallProfile() {
    const uint64_t c = calls.load();
    if (c == 0) return;
    const uint64_t disp = ns_dispatch.load();
    const uint64_t smap = ns_svmmap.load();
    const uint64_t gpu = ns_gpu_kernel.load();
    const uint64_t gpu_c = gpu_events.load();
    auto ms = [](uint64_t v) { return v / 1.0e6; };
    std::fprintf(stderr,
                 "\n[PROFILE gemv_int4_image_v1 decode] calls=%llu "
                 "avg_wall=%.3f ms avg_gpu=%.3f ms\n",
                 (unsigned long long)c,
                 c ? ms(disp + smap) / (double)c : 0.0,
                 gpu_c ? ms(gpu) / (double)gpu_c : 0.0);
    std::fprintf(stderr,
                 "  dispatch   : %7.2f ms   [host wall around clEnqueueNDRangeKernel]\n",
                 ms(disp));
    std::fprintf(stderr,
                 "  svmmap     : %7.2f ms   [blocking enqueueSVMMap(output)]\n",
                 ms(smap));
    std::fprintf(stderr,
                 "  gpu_kernel : %7.2f ms   [cl_event device time, %llu events]\n",
                 ms(gpu), (unsigned long long)gpu_c);
  }
};

static GemvInt4ImageCallProfile g_gemv_image_call_profile;

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

// =========================================================================
// Phase 2 (int8 DP4A) GEMM wrapper.
//
// Same weight / scale layout as gemm_int4_adreno_cl (channel-wise int4),
// but activations are pre-quantized to int8 by the caller and the device
// kernel uses `dot(char4, char4)` -> ssad for a ~3.6x MAC rate vs fp16.
//
// No input_transpose kernel here: the DP4A kernel reads x_q row-major
// (m-major, k-contiguous char4), which matches the natural activation
// layout and avoids the xt_gpu stage in the fp16 path.
// =========================================================================
void gemm_int8_int4_adreno_cl(int8_t *x_q, uint16_t *x_scale,
                              uint16_t *w_scale, uint16_t *weights,
                              uint16_t *output, unsigned int M, unsigned int N,
                              unsigned int K) {
  // Phase 3b reverts to 8x4 tile (N % 4 == 0 instead of N % 8) and
  // unrolls the K loop 4x, requiring K % 16 == 0. All current test
  // shapes (K in {32,64,128,256,512,1024}) satisfy this.
  if (((N % 4) != 0) || ((K % 16) != 0)) {
    throw std::runtime_error(
      "gemm_int8_int4_adreno_cl requires N % 4 == 0 and K % 16 == 0");
  }
  // The texture view below treats x_q as a char4 1D image, so the row
  // stride in texels is K/4. Caller must have zero-padded to align(M, 8).
  const unsigned int alignM = align(M, 8u);

  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));

  cl_int err = CL_SUCCESS;

  // --- Wrap x_q SVM as cl_mem + image1d_buffer (CL_SIGNED_INT8 / RGBA) ---
  //
  // Using the texture path (same memory topology as gpu_int4_gemm_adreno's
  // fp16 input) so reads go through the dedicated texture cache / fetch
  // unit rather than the generic L1 port that `__global char *` vloads
  // would use. First DP4A implementation used a plain global buffer and
  // came in ~1.6x slower than the fp16 path on M=512 K=1024 N=1024; the
  // switch to image1d_buffer mirrors the fp16 kernel's access pattern.
  const size_t x_q_bytes =
    static_cast<size_t>(alignM) * static_cast<size_t>(K) * sizeof(int8_t);
  cl_mem x_q_buf =
    clCreateBuffer(blas_cc->context_inst_.GetContext(),
                   CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, x_q_bytes, x_q,
                   &err);
  if (err != CL_SUCCESS)
    throw std::runtime_error("Failed to create cl_mem for x_q");

  cl_image_format x_q_fmt;
  x_q_fmt.image_channel_order = CL_RGBA;
  x_q_fmt.image_channel_data_type = CL_SIGNED_INT8;

  cl_image_desc x_q_desc;
  std::memset(&x_q_desc, 0, sizeof(x_q_desc));
  x_q_desc.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
  x_q_desc.image_width =
    static_cast<size_t>(alignM) * static_cast<size_t>(K) / 4u;
  x_q_desc.buffer = x_q_buf;

  cl_mem x_q_img =
    clCreateImage(blas_cc->context_inst_.GetContext(), CL_MEM_READ_ONLY,
                  &x_q_fmt, &x_q_desc, nullptr, &err);
  if (err != CL_SUCCESS) {
    clReleaseMemObject(x_q_buf);
    throw std::runtime_error(
      "Failed to create image1d_buffer for x_q "
      "(CL_RGBA/CL_SIGNED_INT8 unsupported?)");
  }

  ClContext::SharedPtrClKernel kernel_ptr = blas_cc->registerClKernel(
    int8_int4_gemm_adreno_kernel, "gpu_int8_int4_gemm_adreno");
  if (!kernel_ptr) {
    clReleaseMemObject(x_q_img);
    clReleaseMemObject(x_q_buf);
    throw std::runtime_error(
      "Failed to register gpu_int8_int4_gemm_adreno kernel");
  }

  bool result = false;
  int arg = 0;

  // arg 0 is now an image (cl_mem), not an SVM pointer.
  result = kernel_ptr->SetKernelArguments(arg++, &x_q_img, sizeof(cl_mem));
  if (!result)
    throw std::runtime_error("gpu_int8_int4_gemm_adreno arg 0 (x_q image)");

  result = kernel_ptr->SetKernelSVMArguments(arg++, x_scale);
  if (!result)
    throw std::runtime_error("gpu_int8_int4_gemm_adreno arg 1 (x_scale)");

  result = kernel_ptr->SetKernelSVMArguments(arg++, w_scale);
  if (!result)
    throw std::runtime_error("gpu_int8_int4_gemm_adreno arg 2 (w_scale)");

  result = kernel_ptr->SetKernelSVMArguments(arg++, weights);
  if (!result)
    throw std::runtime_error("gpu_int8_int4_gemm_adreno arg 3 (weights)");

  result = kernel_ptr->SetKernelSVMArguments(arg++, output);
  if (!result)
    throw std::runtime_error("gpu_int8_int4_gemm_adreno arg 4 (output)");

  int size_k = static_cast<int>(K);
  int size_n = static_cast<int>(N);
  int size_m = static_cast<int>(M);

  result = kernel_ptr->SetKernelArguments(arg++, &size_k, sizeof(int));
  if (!result)
    throw std::runtime_error("gpu_int8_int4_gemm_adreno arg 5 (K)");
  result = kernel_ptr->SetKernelArguments(arg++, &size_n, sizeof(int));
  if (!result)
    throw std::runtime_error("gpu_int8_int4_gemm_adreno arg 6 (N)");
  result = kernel_ptr->SetKernelArguments(arg++, &size_m, sizeof(int));
  if (!result)
    throw std::runtime_error("gpu_int8_int4_gemm_adreno arg 7 (M)");

  // Dispatch: 8x4 tile per WI, K-unroll is hidden inside the kernel.
  //   global = (ceilDiv(M, 8), N/4, 1)
  //   local  = {1, 128, 1}
  const int g_global[3] = {static_cast<int>(ceilDiv(M, 8u)),
                           static_cast<int>(N) / 4, 1};
  const int g_local[3] = {1, 128, 1};

  result = blas_cc->command_queue_inst_.DispatchCommand(kernel_ptr, g_global,
                                                        g_local);
  if (!result) {
    clReleaseMemObject(x_q_img);
    clReleaseMemObject(x_q_buf);
    throw std::runtime_error("Failed to dispatch gpu_int8_int4_gemm_adreno");
  }

  // Sync output back to host (blocking SVMMap, same as fp16 wrapper).
  blas_cc->command_queue_inst_.enqueueSVMMap(
    output, static_cast<size_t>(M) * static_cast<size_t>(N) * sizeof(uint16_t),
    true);

  clReleaseMemObject(x_q_img);
  clReleaseMemObject(x_q_buf);
}

// =========================================================================
// Phase 3c v2: fp16 GEMM with weights routed through a texture sampler.
//
// Identical code path to `gemm_int4_adreno_cl` except that the packed
// int4 weight buffer is additionally wrapped as an `image1d_buffer_t`
// with CL_RGBA / CL_UNSIGNED_INT16 format (4 ushorts per texel). This
// matches LiteRT's observation that on Adreno GPUs the texture cache +
// fetch unit runs in parallel with the generic L1 port, so reading
// weights + activations via independent texture handles has ~1.5x the
// effective memory bandwidth of reading one or the other from
// `__global`.
// =========================================================================
void gemm_int4_adreno_v2_cl(uint16_t *input, uint16_t *input_transposed,
                            uint16_t *weights, uint16_t *scales,
                            uint16_t *output, unsigned int M, unsigned int N,
                            unsigned int K) {
  if (((N % 4) != 0) || ((K % 4) != 0)) {
    throw std::runtime_error(
      "gemm_int4_adreno_v2_cl requires N and K to be multiples of 4");
  }

  const int q_group_size = static_cast<int>(K); // channel-wise
  const int alignK = static_cast<int>(align(K, q_group_size));

  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));

  cl_int err = CL_SUCCESS;

  // --- input image (fp16 [M][alignK]) ---
  const size_t input_buf_bytes =
    static_cast<size_t>(M) * static_cast<size_t>(alignK) * sizeof(uint16_t);
  cl_mem input_buf =
    clCreateBuffer(blas_cc->context_inst_.GetContext(),
                   CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, input_buf_bytes,
                   input, &err);
  if (err != CL_SUCCESS)
    throw std::runtime_error("v2: failed to create input cl_mem buffer");

  cl_image_format img_fmt_half;
  img_fmt_half.image_channel_order = CL_RGBA;
  img_fmt_half.image_channel_data_type = CL_HALF_FLOAT;

  cl_image_desc img_desc;
  std::memset(&img_desc, 0, sizeof(img_desc));
  img_desc.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
  img_desc.image_width =
    static_cast<size_t>(M) * static_cast<size_t>(alignK) / 4;
  img_desc.buffer = input_buf;

  cl_mem input_img =
    clCreateImage(blas_cc->context_inst_.GetContext(), CL_MEM_READ_ONLY,
                  &img_fmt_half, &img_desc, nullptr, &err);
  if (err != CL_SUCCESS) {
    clReleaseMemObject(input_buf);
    throw std::runtime_error("v2: failed to create input image1d_buffer");
  }

  // --- input_transposed image ---
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
    throw std::runtime_error("v2: failed to create input_transposed cl_mem");
  }

  std::memset(&img_desc, 0, sizeof(img_desc));
  img_desc.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
  img_desc.image_width =
    static_cast<size_t>(align(M, 4)) * static_cast<size_t>(alignK) / 4;
  img_desc.buffer = input_t_buf;

  cl_mem input_t_img =
    clCreateImage(blas_cc->context_inst_.GetContext(), CL_MEM_READ_WRITE,
                  &img_fmt_half, &img_desc, nullptr, &err);
  if (err != CL_SUCCESS) {
    clReleaseMemObject(input_t_buf);
    clReleaseMemObject(input_img);
    clReleaseMemObject(input_buf);
    throw std::runtime_error(
      "v2: failed to create image1d_buffer for input_transposed");
  }

  // --- weight image (THIS IS THE NEW PIECE) ---
  //
  // packed int4 weights: (K/4) * N ushorts; cast 4 ushorts per RGBA texel
  // so width = K/4 * N / 4 texels. N % 4 == 0 is already asserted.
  const size_t weight_buf_bytes = static_cast<size_t>(K / 4u) *
                                  static_cast<size_t>(N) * sizeof(uint16_t);
  cl_mem weight_buf =
    clCreateBuffer(blas_cc->context_inst_.GetContext(),
                   CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, weight_buf_bytes,
                   weights, &err);
  if (err != CL_SUCCESS) {
    clReleaseMemObject(input_t_img);
    clReleaseMemObject(input_t_buf);
    clReleaseMemObject(input_img);
    clReleaseMemObject(input_buf);
    throw std::runtime_error("v2: failed to create weight cl_mem buffer");
  }

  cl_image_format img_fmt_u16;
  img_fmt_u16.image_channel_order = CL_RGBA;
  img_fmt_u16.image_channel_data_type = CL_UNSIGNED_INT16;

  std::memset(&img_desc, 0, sizeof(img_desc));
  img_desc.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
  img_desc.image_width =
    static_cast<size_t>(K / 4u) * static_cast<size_t>(N) / 4u;
  img_desc.buffer = weight_buf;

  cl_mem weight_img =
    clCreateImage(blas_cc->context_inst_.GetContext(), CL_MEM_READ_ONLY,
                  &img_fmt_u16, &img_desc, nullptr, &err);
  if (err != CL_SUCCESS) {
    clReleaseMemObject(weight_buf);
    clReleaseMemObject(input_t_img);
    clReleaseMemObject(input_t_buf);
    clReleaseMemObject(input_img);
    clReleaseMemObject(input_buf);
    throw std::runtime_error(
      "v2: failed to create image1d_buffer for weights "
      "(CL_RGBA/CL_UNSIGNED_INT16 unsupported?)");
  }

  // --- step 1: input_transpose kernel (unchanged) ---
  ClContext::SharedPtrClKernel xt_kernel =
    blas_cc->registerClKernel(input_transpose_kernel, "input_transpose");
  if (!xt_kernel) {
    clReleaseMemObject(weight_img);
    clReleaseMemObject(weight_buf);
    clReleaseMemObject(input_t_img);
    clReleaseMemObject(input_t_buf);
    clReleaseMemObject(input_img);
    clReleaseMemObject(input_buf);
    throw std::runtime_error("v2: failed to register input_transpose kernel");
  }

  bool result = false;
  {
    int arg = 0;
    if (!xt_kernel->SetKernelArguments(arg++, &input_img, sizeof(cl_mem)))
      throw std::runtime_error("v2: input_transpose arg 0");
    if (!xt_kernel->SetKernelArguments(arg++, &input_t_img, sizeof(cl_mem)))
      throw std::runtime_error("v2: input_transpose arg 1");

    int alignK_4 = alignK >> 2;
    if (!xt_kernel->SetKernelArguments(arg++, &alignK_4, sizeof(int)))
      throw std::runtime_error("v2: input_transpose arg 2");
    int M_4 = static_cast<int>(ceilDiv(M, 4u));
    if (!xt_kernel->SetKernelArguments(arg++, &M_4, sizeof(int)))
      throw std::runtime_error("v2: input_transpose arg 3");

    const int xt_global[3] = {alignK_4, M_4, 1};
    const int xt_local[3] = {1, 128, 1};

    result = blas_cc->command_queue_inst_.DispatchCommand(xt_kernel, xt_global,
                                                          xt_local);
    if (!result)
      throw std::runtime_error("v2: failed to dispatch input_transpose");
  }

  // --- step 2: gpu_int4_gemm_adreno_v2 (weight texture) ---
  ClContext::SharedPtrClKernel gemm_kernel = blas_cc->registerClKernel(
    int4_gemm_adreno_v2_kernel, "gpu_int4_gemm_adreno_v2");
  if (!gemm_kernel) {
    clReleaseMemObject(weight_img);
    clReleaseMemObject(weight_buf);
    clReleaseMemObject(input_t_img);
    clReleaseMemObject(input_t_buf);
    clReleaseMemObject(input_img);
    clReleaseMemObject(input_buf);
    throw std::runtime_error(
      "v2: failed to register gpu_int4_gemm_adreno_v2 kernel");
  }

  {
    int arg = 0;
    if (!gemm_kernel->SetKernelArguments(arg++, &input_t_img, sizeof(cl_mem)))
      throw std::runtime_error("v2: gemm arg 0 (input_transposed image)");
    if (!gemm_kernel->SetKernelSVMArguments(arg++, scales))
      throw std::runtime_error("v2: gemm arg 1 (scales)");
    if (!gemm_kernel->SetKernelSVMArguments(arg++, output))
      throw std::runtime_error("v2: gemm arg 2 (output)");
    if (!gemm_kernel->SetKernelArguments(arg++, &weight_img, sizeof(cl_mem)))
      throw std::runtime_error("v2: gemm arg 3 (weights image)");

    int size_k = static_cast<int>(K);
    int size_n = static_cast<int>(N);
    int size_m = static_cast<int>(M);
    int qg = q_group_size;

    if (!gemm_kernel->SetKernelArguments(arg++, &size_k, sizeof(int)))
      throw std::runtime_error("v2: gemm arg 4 (K)");
    if (!gemm_kernel->SetKernelArguments(arg++, &size_n, sizeof(int)))
      throw std::runtime_error("v2: gemm arg 5 (N)");
    if (!gemm_kernel->SetKernelArguments(arg++, &size_m, sizeof(int)))
      throw std::runtime_error("v2: gemm arg 6 (M)");
    if (!gemm_kernel->SetKernelArguments(arg++, &qg, sizeof(int)))
      throw std::runtime_error("v2: gemm arg 7 (quantization_group_size)");

    const int g_global[3] = {static_cast<int>(ceilDiv(M, 8u)),
                             static_cast<int>(N) / 4, 1};
    const int g_local[3] = {1, 128, 1};

    result = blas_cc->command_queue_inst_.DispatchCommand(gemm_kernel, g_global,
                                                          g_local);
    if (!result)
      throw std::runtime_error(
        "v2: failed to dispatch gpu_int4_gemm_adreno_v2");
  }

  // Sync output back to host.
  blas_cc->command_queue_inst_.enqueueSVMMap(
    output, static_cast<size_t>(M) * static_cast<size_t>(N) * sizeof(uint16_t),
    true);

  clReleaseMemObject(weight_img);
  clReleaseMemObject(weight_buf);
  clReleaseMemObject(input_t_img);
  clReleaseMemObject(input_t_buf);
  clReleaseMemObject(input_img);
  clReleaseMemObject(input_buf);
}

// =========================================================================
// Phase 3c v3: v2 + K-axis split + __local reduction.
//
// Dispatch layout flips from v2's (ceilDiv(M,8), N/4, 1) to (N/4, 4,
// ceilDiv(M, 8)) so a single work-group holds all 4 K-split WIs that
// collaborate on the same 8x4 output tile. Everything else (weight
// texture, input texture, channel-wise scales) is inherited from v2.
// =========================================================================
void gemm_int4_adreno_v3_cl(uint16_t *input, uint16_t *input_transposed,
                            uint16_t *weights, uint16_t *scales,
                            uint16_t *output, unsigned int M, unsigned int N,
                            unsigned int K) {
  if (((N % 4) != 0) || ((K % 16) != 0)) {
    throw std::runtime_error(
      "gemm_int4_adreno_v3_cl requires N % 4 == 0 and K % 16 == 0 "
      "(K_SPLIT=4, each split processes multiples of 4 along k)");
  }

  const int q_group_size = static_cast<int>(K);
  const int alignK = static_cast<int>(align(K, q_group_size));

  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));

  cl_int err = CL_SUCCESS;

  // Same image wrapping as v2 (input + input_transposed + weights).
  const size_t input_buf_bytes =
    static_cast<size_t>(M) * static_cast<size_t>(alignK) * sizeof(uint16_t);
  cl_mem input_buf =
    clCreateBuffer(blas_cc->context_inst_.GetContext(),
                   CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, input_buf_bytes,
                   input, &err);
  if (err != CL_SUCCESS)
    throw std::runtime_error("v3: failed to create input cl_mem buffer");

  cl_image_format img_fmt_half;
  img_fmt_half.image_channel_order = CL_RGBA;
  img_fmt_half.image_channel_data_type = CL_HALF_FLOAT;

  cl_image_desc img_desc;
  std::memset(&img_desc, 0, sizeof(img_desc));
  img_desc.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
  img_desc.image_width =
    static_cast<size_t>(M) * static_cast<size_t>(alignK) / 4;
  img_desc.buffer = input_buf;
  cl_mem input_img =
    clCreateImage(blas_cc->context_inst_.GetContext(), CL_MEM_READ_ONLY,
                  &img_fmt_half, &img_desc, nullptr, &err);
  if (err != CL_SUCCESS) {
    clReleaseMemObject(input_buf);
    throw std::runtime_error("v3: failed to create input image1d_buffer");
  }

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
    throw std::runtime_error("v3: failed to create input_transposed cl_mem");
  }
  std::memset(&img_desc, 0, sizeof(img_desc));
  img_desc.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
  img_desc.image_width =
    static_cast<size_t>(align(M, 4)) * static_cast<size_t>(alignK) / 4;
  img_desc.buffer = input_t_buf;
  cl_mem input_t_img =
    clCreateImage(blas_cc->context_inst_.GetContext(), CL_MEM_READ_WRITE,
                  &img_fmt_half, &img_desc, nullptr, &err);
  if (err != CL_SUCCESS) {
    clReleaseMemObject(input_t_buf);
    clReleaseMemObject(input_img);
    clReleaseMemObject(input_buf);
    throw std::runtime_error(
      "v3: failed to create image1d_buffer for input_transposed");
  }

  const size_t weight_buf_bytes = static_cast<size_t>(K / 4u) *
                                  static_cast<size_t>(N) * sizeof(uint16_t);
  cl_mem weight_buf =
    clCreateBuffer(blas_cc->context_inst_.GetContext(),
                   CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, weight_buf_bytes,
                   weights, &err);
  if (err != CL_SUCCESS) {
    clReleaseMemObject(input_t_img);
    clReleaseMemObject(input_t_buf);
    clReleaseMemObject(input_img);
    clReleaseMemObject(input_buf);
    throw std::runtime_error("v3: failed to create weight cl_mem buffer");
  }

  cl_image_format img_fmt_u16;
  img_fmt_u16.image_channel_order = CL_RGBA;
  img_fmt_u16.image_channel_data_type = CL_UNSIGNED_INT16;

  std::memset(&img_desc, 0, sizeof(img_desc));
  img_desc.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
  img_desc.image_width =
    static_cast<size_t>(K / 4u) * static_cast<size_t>(N) / 4u;
  img_desc.buffer = weight_buf;
  cl_mem weight_img =
    clCreateImage(blas_cc->context_inst_.GetContext(), CL_MEM_READ_ONLY,
                  &img_fmt_u16, &img_desc, nullptr, &err);
  if (err != CL_SUCCESS) {
    clReleaseMemObject(weight_buf);
    clReleaseMemObject(input_t_img);
    clReleaseMemObject(input_t_buf);
    clReleaseMemObject(input_img);
    clReleaseMemObject(input_buf);
    throw std::runtime_error("v3: failed to create image1d_buffer for weights");
  }

  // step 1: input_transpose (identical to v1/v2)
  ClContext::SharedPtrClKernel xt_kernel =
    blas_cc->registerClKernel(input_transpose_kernel, "input_transpose");
  if (!xt_kernel) {
    clReleaseMemObject(weight_img);
    clReleaseMemObject(weight_buf);
    clReleaseMemObject(input_t_img);
    clReleaseMemObject(input_t_buf);
    clReleaseMemObject(input_img);
    clReleaseMemObject(input_buf);
    throw std::runtime_error("v3: failed to register input_transpose kernel");
  }

  bool result = false;
  {
    int arg = 0;
    if (!xt_kernel->SetKernelArguments(arg++, &input_img, sizeof(cl_mem)))
      throw std::runtime_error("v3: xt arg 0");
    if (!xt_kernel->SetKernelArguments(arg++, &input_t_img, sizeof(cl_mem)))
      throw std::runtime_error("v3: xt arg 1");
    int alignK_4 = alignK >> 2;
    if (!xt_kernel->SetKernelArguments(arg++, &alignK_4, sizeof(int)))
      throw std::runtime_error("v3: xt arg 2");
    int M_4 = static_cast<int>(ceilDiv(M, 4u));
    if (!xt_kernel->SetKernelArguments(arg++, &M_4, sizeof(int)))
      throw std::runtime_error("v3: xt arg 3");

    const int xt_global[3] = {alignK_4, M_4, 1};
    const int xt_local[3] = {1, 128, 1};
    result = blas_cc->command_queue_inst_.DispatchCommand(xt_kernel, xt_global,
                                                          xt_local);
    if (!result)
      throw std::runtime_error("v3: failed to dispatch input_transpose");
  }

  // step 2: gpu_int4_gemm_adreno_v3 (K-split + local reduction).
  ClContext::SharedPtrClKernel gemm_kernel = blas_cc->registerClKernel(
    int4_gemm_adreno_v3_kernel, "gpu_int4_gemm_adreno_v3");
  if (!gemm_kernel) {
    clReleaseMemObject(weight_img);
    clReleaseMemObject(weight_buf);
    clReleaseMemObject(input_t_img);
    clReleaseMemObject(input_t_buf);
    clReleaseMemObject(input_img);
    clReleaseMemObject(input_buf);
    throw std::runtime_error("v3: failed to register gpu_int4_gemm_adreno_v3");
  }

  {
    int arg = 0;
    if (!gemm_kernel->SetKernelArguments(arg++, &input_t_img, sizeof(cl_mem)))
      throw std::runtime_error("v3: gemm arg 0 (input_transposed image)");
    if (!gemm_kernel->SetKernelSVMArguments(arg++, scales))
      throw std::runtime_error("v3: gemm arg 1 (scales)");
    if (!gemm_kernel->SetKernelSVMArguments(arg++, output))
      throw std::runtime_error("v3: gemm arg 2 (output)");
    if (!gemm_kernel->SetKernelArguments(arg++, &weight_img, sizeof(cl_mem)))
      throw std::runtime_error("v3: gemm arg 3 (weights image)");

    int size_k = static_cast<int>(K);
    int size_n = static_cast<int>(N);
    int size_m = static_cast<int>(M);
    int qg = q_group_size;
    if (!gemm_kernel->SetKernelArguments(arg++, &size_k, sizeof(int)))
      throw std::runtime_error("v3: gemm arg 4 (K)");
    if (!gemm_kernel->SetKernelArguments(arg++, &size_n, sizeof(int)))
      throw std::runtime_error("v3: gemm arg 5 (N)");
    if (!gemm_kernel->SetKernelArguments(arg++, &size_m, sizeof(int)))
      throw std::runtime_error("v3: gemm arg 6 (M)");
    if (!gemm_kernel->SetKernelArguments(arg++, &qg, sizeof(int)))
      throw std::runtime_error("v3: gemm arg 7 (q_group_size)");

    // FLIPPED dispatch: v1/v2 = (ceilDiv(M,8), N/4, 1) with local
    // (1, 128, 1); v3 = (N/4, 4, ceilDiv(M,8)) with local (32, 4, 1).
    // The y=4 dimension is what lets all 4 K-split WIs land in the
    // same WG so they can share __local memory.
    //
    // N/4 is rounded up to a multiple of WG_X (=32) because the kernel
    // declares __attribute__((qcom_reqd_sub_group_size("full"))), which
    // on Adreno 8xx requires the WG to be a multiple of the wave size
    // (64 WIs). For shapes with N/4 < 32 (e.g. N=32 -> N/4=8, N=64 ->
    // N/4=16) the driver would otherwise either round up to a uniform
    // WG anyway or produce a non-uniform WG that violates the full-SG
    // requirement, and the extra OOB lanes along x would silently write
    // garbage to legitimate output cells (packed_w from an OOB image
    // read returns 0, and (0 & 0xF) - 8 = -8 still multiplies into a
    // nonzero dq value that gets vstore4'd to output + n, clobbering a
    // different lane's tile). Pad the global size to match the WG and
    // guard the writes via `n >= N` in the kernel so OOB lanes still
    // hit the barrier but skip the output store.
    constexpr int WG_X = 32;
    const int n_tiles_padded =
      static_cast<int>(ceilDiv(static_cast<unsigned int>(N) / 4u,
                               static_cast<unsigned int>(WG_X))) *
      WG_X;
    const int g_global[3] = {n_tiles_padded, 4,
                             static_cast<int>(ceilDiv(M, 8u))};
    const int g_local[3] = {WG_X, 4, 1};
    result = blas_cc->command_queue_inst_.DispatchCommand(gemm_kernel, g_global,
                                                          g_local);
    if (!result)
      throw std::runtime_error(
        "v3: failed to dispatch gpu_int4_gemm_adreno_v3");
  }

  blas_cc->command_queue_inst_.enqueueSVMMap(
    output, static_cast<size_t>(M) * static_cast<size_t>(N) * sizeof(uint16_t),
    true);

  clReleaseMemObject(weight_img);
  clReleaseMemObject(weight_buf);
  clReleaseMemObject(input_t_img);
  clReleaseMemObject(input_t_buf);
  clReleaseMemObject(input_img);
  clReleaseMemObject(input_buf);
}

void gemv_int4_adreno_cl(uint16_t *input, uint16_t *weights, uint16_t *scales,
                         uint16_t *output, unsigned int K, unsigned int N,
                         bool sync_output) {
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

  // Phase 2 diagnostic: split per-call cost into args / dispatch / svmmap.
  const uint64_t t_args_start = now_ns_phase6();

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

  const uint64_t t_dispatch_start = now_ns_phase6();
  cl_event kernel_ev = nullptr;
  result = blas_cc->command_queue_inst_.DispatchCommand(
    kernel_ptr, work_groups_count, work_group_size, &kernel_ev);
  if (!result) {
    throw std::runtime_error(
      "Failed to dispatch kernel for gpu_int4_gemv_adreno");
    return;
  }

  const uint64_t t_svmmap_start = now_ns_phase6();
  if (sync_output) {
    // Blocking SVMMap: queue flush + CPU cache invalidate.  The
    // non-blocking variant (matching prefill's delegate conv
    // pattern) made this call 113x faster but produced garbage
    // under LAYER_SYNC off -- Adreno 830's coarse-grained SVM
    // doesn't guarantee GPU->GPU cross-kernel visibility through
    // same-queue ordering alone when the prior kernel wrote the
    // input SVM buffer this kernel now reads.  Investigated
    // enqueueSVMUnmap(input) + svm_in staging as coherence hints;
    // neither worked.  Keep blocking until we either (a) move
    // decode onto the image2d pipeline prefill uses, (b) add
    // explicit clEnqueueMarkerWithWaitList between kernels, or
    // (c) understand the missing SVM barrier on Adreno.
    blas_cc->command_queue_inst_.enqueueSVMMap(
      output, static_cast<size_t>(N) * sizeof(uint16_t), /*read_only=*/true);
  }
  const uint64_t t_end = now_ns_phase6();

  g_gemv_adreno_call_profile.ns_args     += t_dispatch_start - t_args_start;
  g_gemv_adreno_call_profile.ns_dispatch += t_svmmap_start - t_dispatch_start;
  g_gemv_adreno_call_profile.ns_svmmap   += t_end - t_svmmap_start;
  g_gemv_adreno_call_profile.calls++;

  // Query the kernel event for true device-side compute time.  Safe
  // to query unconditionally AFTER the svmmap above drained the
  // queue (event is guaranteed complete).  When sync_output is
  // false the event may not be done yet; skip profiling in that
  // case to avoid blocking on clWaitForEvents here (defeats the
  // purpose of non-blocking).
  if (kernel_ev) {
    if (sync_output) {
      cl_ulong ev_start = 0, ev_end = 0;
      clGetEventProfilingInfo(kernel_ev, CL_PROFILING_COMMAND_START,
                              sizeof(ev_start), &ev_start, nullptr);
      clGetEventProfilingInfo(kernel_ev, CL_PROFILING_COMMAND_END,
                              sizeof(ev_end), &ev_end, nullptr);
      if (ev_end > ev_start) {
        g_gemv_adreno_call_profile.ns_gpu_kernel += ev_end - ev_start;
        g_gemv_adreno_call_profile.gpu_events++;
      }
    }
    clReleaseEvent(kernel_ev);
  }
}

// Step 1 of the incremental image2d gemv rewrite.  Signature matches
// gemv_int4_adreno_cl so the caller can env-swap them one at a time.
// Step 1 only LOADS weights and writes zero; output is garbage, but
// gpu_kernel timing isolates the weight-read cost from everything
// else we'll layer on in steps 2..4.
void gemv_int4_image_v1_cl(uint16_t *input, uint16_t *weights,
                           uint16_t *scales, uint16_t *output,
                           unsigned int K, unsigned int N) {
  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  if (!blas_cc) return;

  ClContext::SharedPtrClKernel kernel_ptr = blas_cc->registerClKernel(
    int4_gemv_image_v1_kernel, "gpu_int4_gemv_image_v1");
  if (!kernel_ptr) {
    throw std::runtime_error(
      "Failed to register gpu_int4_gemv_image_v1 kernel");
  }

  int arg = 0;
  kernel_ptr->SetKernelSVMArguments(arg++, input);
  kernel_ptr->SetKernelSVMArguments(arg++, scales);
  kernel_ptr->SetKernelSVMArguments(arg++, output);
  kernel_ptr->SetKernelSVMArguments(arg++, weights);
  int size_k = (int)K, size_n = (int)N;
  kernel_ptr->SetKernelArguments(arg++, &size_k, sizeof(int));
  kernel_ptr->SetKernelArguments(arg++, &size_n, sizeof(int));

  // Dispatch: global = align(N, 64), local = {64, 1, 1}.  Matches
  // the planned final shape so step-by-step gpu_kernel time
  // comparisons are like-for-like.
  const int align_N = static_cast<int>(align(N, 64));
  const int g[3] = {align_N, 1, 1};
  const int l[3] = {64, 1, 1};

  const uint64_t t_dispatch_start = now_ns_phase6();
  cl_event kernel_ev = nullptr;
  if (!blas_cc->command_queue_inst_.DispatchCommand(kernel_ptr, g, l,
                                                    &kernel_ev)) {
    throw std::runtime_error("Failed to dispatch gpu_int4_gemv_image_v1");
  }

  const uint64_t t_svmmap_start = now_ns_phase6();
  blas_cc->command_queue_inst_.enqueueSVMMap(
    output, (size_t)N * sizeof(uint16_t), /*read_only=*/true);
  const uint64_t t_end = now_ns_phase6();

  g_gemv_image_call_profile.ns_dispatch += t_svmmap_start - t_dispatch_start;
  g_gemv_image_call_profile.ns_svmmap   += t_end - t_svmmap_start;
  g_gemv_image_call_profile.calls++;

  if (kernel_ev) {
    cl_ulong ev_start = 0, ev_end = 0;
    clGetEventProfilingInfo(kernel_ev, CL_PROFILING_COMMAND_START,
                            sizeof(ev_start), &ev_start, nullptr);
    clGetEventProfilingInfo(kernel_ev, CL_PROFILING_COMMAND_END,
                            sizeof(ev_end), &ev_end, nullptr);
    if (ev_end > ev_start) {
      g_gemv_image_call_profile.ns_gpu_kernel += ev_end - ev_start;
      g_gemv_image_call_profile.gpu_events++;
    }
    clReleaseEvent(kernel_ev);
  }
}

// Step 2 of the incremental rewrite: adds scalar input-load loop
// alongside the step-1 weight load, still no MAC / dequant / scale.
// Delta vs step 1's gpu_kernel isolates the cost of the input feed.
// Same dispatch shape so the numbers are directly comparable.
void gemv_int4_image_v2_cl(uint16_t *input, uint16_t *weights,
                           uint16_t *scales, uint16_t *output,
                           unsigned int K, unsigned int N) {
  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  if (!blas_cc) return;

  ClContext::SharedPtrClKernel kernel_ptr = blas_cc->registerClKernel(
    int4_gemv_image_v2_kernel, "gpu_int4_gemv_image_v2");
  if (!kernel_ptr) {
    throw std::runtime_error(
      "Failed to register gpu_int4_gemv_image_v2 kernel");
  }

  int arg = 0;
  kernel_ptr->SetKernelSVMArguments(arg++, input);
  kernel_ptr->SetKernelSVMArguments(arg++, scales);
  kernel_ptr->SetKernelSVMArguments(arg++, output);
  kernel_ptr->SetKernelSVMArguments(arg++, weights);
  int size_k = (int)K, size_n = (int)N;
  kernel_ptr->SetKernelArguments(arg++, &size_k, sizeof(int));
  kernel_ptr->SetKernelArguments(arg++, &size_n, sizeof(int));

  const int align_N = static_cast<int>(align(N, 64));
  const int g[3] = {align_N, 1, 1};
  const int l[3] = {64, 1, 1};

  const uint64_t t_dispatch_start = now_ns_phase6();
  cl_event kernel_ev = nullptr;
  if (!blas_cc->command_queue_inst_.DispatchCommand(kernel_ptr, g, l,
                                                    &kernel_ev)) {
    throw std::runtime_error("Failed to dispatch gpu_int4_gemv_image_v2");
  }

  const uint64_t t_svmmap_start = now_ns_phase6();
  blas_cc->command_queue_inst_.enqueueSVMMap(
    output, (size_t)N * sizeof(uint16_t), /*read_only=*/true);
  const uint64_t t_end = now_ns_phase6();

  // Share the image profile bucket with step 1 -- they don't co-
  // dispatch (env gate picks one) so concatenating their
  // measurements just means a single unified number per run.
  g_gemv_image_call_profile.ns_dispatch += t_svmmap_start - t_dispatch_start;
  g_gemv_image_call_profile.ns_svmmap   += t_end - t_svmmap_start;
  g_gemv_image_call_profile.calls++;

  if (kernel_ev) {
    cl_ulong ev_start = 0, ev_end = 0;
    clGetEventProfilingInfo(kernel_ev, CL_PROFILING_COMMAND_START,
                            sizeof(ev_start), &ev_start, nullptr);
    clGetEventProfilingInfo(kernel_ev, CL_PROFILING_COMMAND_END,
                            sizeof(ev_end), &ev_end, nullptr);
    if (ev_end > ev_start) {
      g_gemv_image_call_profile.ns_gpu_kernel += ev_end - ev_start;
      g_gemv_image_call_profile.gpu_events++;
    }
    clReleaseEvent(kernel_ev);
  }
}

// Step 3: dequant + MAC added on top of step 2.  Scale not yet
// applied; output is the un-scaled running sum written as half so
// the MAC chain can't be DCE'd.  Delta vs step 2's gpu_kernel
// should isolate the compute cost of the inner nibble-dequant +
// float-MAC loop, which our hypothesis says is the bulk of the
// baseline kernel's 430 us/call.
void gemv_int4_image_v3_cl(uint16_t *input, uint16_t *weights,
                           uint16_t *scales, uint16_t *output,
                           unsigned int K, unsigned int N) {
  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  if (!blas_cc) return;

  ClContext::SharedPtrClKernel kernel_ptr = blas_cc->registerClKernel(
    int4_gemv_image_v3_kernel, "gpu_int4_gemv_image_v3");
  if (!kernel_ptr) {
    throw std::runtime_error(
      "Failed to register gpu_int4_gemv_image_v3 kernel");
  }

  int arg = 0;
  kernel_ptr->SetKernelSVMArguments(arg++, input);
  kernel_ptr->SetKernelSVMArguments(arg++, scales);
  kernel_ptr->SetKernelSVMArguments(arg++, output);
  kernel_ptr->SetKernelSVMArguments(arg++, weights);
  int size_k = (int)K, size_n = (int)N;
  kernel_ptr->SetKernelArguments(arg++, &size_k, sizeof(int));
  kernel_ptr->SetKernelArguments(arg++, &size_n, sizeof(int));

  const int align_N = static_cast<int>(align(N, 64));
  const int g[3] = {align_N, 1, 1};
  const int l[3] = {64, 1, 1};

  const uint64_t t_dispatch_start = now_ns_phase6();
  cl_event kernel_ev = nullptr;
  if (!blas_cc->command_queue_inst_.DispatchCommand(kernel_ptr, g, l,
                                                    &kernel_ev)) {
    throw std::runtime_error("Failed to dispatch gpu_int4_gemv_image_v3");
  }

  const uint64_t t_svmmap_start = now_ns_phase6();
  blas_cc->command_queue_inst_.enqueueSVMMap(
    output, (size_t)N * sizeof(uint16_t), /*read_only=*/true);
  const uint64_t t_end = now_ns_phase6();

  g_gemv_image_call_profile.ns_dispatch += t_svmmap_start - t_dispatch_start;
  g_gemv_image_call_profile.ns_svmmap   += t_end - t_svmmap_start;
  g_gemv_image_call_profile.calls++;

  if (kernel_ev) {
    cl_ulong ev_start = 0, ev_end = 0;
    clGetEventProfilingInfo(kernel_ev, CL_PROFILING_COMMAND_START,
                            sizeof(ev_start), &ev_start, nullptr);
    clGetEventProfilingInfo(kernel_ev, CL_PROFILING_COMMAND_END,
                            sizeof(ev_end), &ev_end, nullptr);
    if (ev_end > ev_start) {
      g_gemv_image_call_profile.ns_gpu_kernel += ev_end - ev_start;
      g_gemv_image_call_profile.gpu_events++;
    }
    clReleaseEvent(kernel_ev);
  }
}

// Step 4: scale + proper output -- correctness restored.  Same
// cl_event profiling path as steps 1-3; this is the last step of
// the "get the new kernel to match the baseline's output" part of
// the rewrite, after which step 5+ can optimize the MAC / dequant
// pipeline without worrying about correctness.
void gemv_int4_image_v4_cl(uint16_t *input, uint16_t *weights,
                           uint16_t *scales, uint16_t *output,
                           unsigned int K, unsigned int N) {
  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  if (!blas_cc) return;

  ClContext::SharedPtrClKernel kernel_ptr = blas_cc->registerClKernel(
    int4_gemv_image_v4_kernel, "gpu_int4_gemv_image_v4");
  if (!kernel_ptr) {
    throw std::runtime_error(
      "Failed to register gpu_int4_gemv_image_v4 kernel");
  }

  int arg = 0;
  kernel_ptr->SetKernelSVMArguments(arg++, input);
  kernel_ptr->SetKernelSVMArguments(arg++, scales);
  kernel_ptr->SetKernelSVMArguments(arg++, output);
  kernel_ptr->SetKernelSVMArguments(arg++, weights);
  int size_k = (int)K, size_n = (int)N;
  kernel_ptr->SetKernelArguments(arg++, &size_k, sizeof(int));
  kernel_ptr->SetKernelArguments(arg++, &size_n, sizeof(int));

  const int align_N = static_cast<int>(align(N, 64));
  const int g[3] = {align_N, 1, 1};
  const int l[3] = {64, 1, 1};

  const uint64_t t_dispatch_start = now_ns_phase6();
  cl_event kernel_ev = nullptr;
  if (!blas_cc->command_queue_inst_.DispatchCommand(kernel_ptr, g, l,
                                                    &kernel_ev)) {
    throw std::runtime_error("Failed to dispatch gpu_int4_gemv_image_v4");
  }

  const uint64_t t_svmmap_start = now_ns_phase6();
  blas_cc->command_queue_inst_.enqueueSVMMap(
    output, (size_t)N * sizeof(uint16_t), /*read_only=*/true);
  const uint64_t t_end = now_ns_phase6();

  g_gemv_image_call_profile.ns_dispatch += t_svmmap_start - t_dispatch_start;
  g_gemv_image_call_profile.ns_svmmap   += t_end - t_svmmap_start;
  g_gemv_image_call_profile.calls++;

  if (kernel_ev) {
    cl_ulong ev_start = 0, ev_end = 0;
    clGetEventProfilingInfo(kernel_ev, CL_PROFILING_COMMAND_START,
                            sizeof(ev_start), &ev_start, nullptr);
    clGetEventProfilingInfo(kernel_ev, CL_PROFILING_COMMAND_END,
                            sizeof(ev_end), &ev_end, nullptr);
    if (ev_end > ev_start) {
      g_gemv_image_call_profile.ns_gpu_kernel += ev_end - ev_start;
      g_gemv_image_call_profile.gpu_events++;
    }
    clReleaseEvent(kernel_ev);
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
// ============================================================================
// Delegate fp16 GEMM — optimized: pre-alloc + cached kernels + pipelined
// ============================================================================

void gemm_delegate_fp16_cl(uint16_t *input, uint16_t * /*input_transposed*/,
                           uint16_t *weights, uint16_t *scales,
                           uint16_t *output, unsigned int M, unsigned int N,
                           unsigned int K) {
  if (((N % 32) != 0) || ((K % 8) != 0))
    throw std::runtime_error("gemm_delegate_fp16_cl requires N%32==0 K%8==0");

  auto now_ns_outer = []() -> uint64_t {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
  };
  const uint64_t t_fn_entry = now_ns_outer();

  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  cl_context clctx = blas_cc->context_inst_.GetContext();
  cl_command_queue clq = blas_cc->command_queue_inst_.GetCommandQueue();

  const int src_slices = K / 4;
  const int dst_slices = N / 4;
  const int n_z = (dst_slices + 7) / 8;
  const int iters = src_slices / 2;
  cl_int err;

  // --- Profiling ---
  static std::atomic<uint64_t> t_dequant{0}, t_reformat{0}, t_conv{0},
    t_readback{0}, t_calls{0}, t_gflops_x1000{0}, t_conv_gpu_ns{0},
    t_reformat_hits{0}, t_resource_setup{0}, t_fn_total{0},
    t_dequant_gpu_ns{0}, t_reformat_gpu_ns{0}, t_readback_gpu_ns{0},
    t_svmmap_gpu_ns{0}, t_dequant_calls{0},
    t_dq_alloc{0}, t_dq_svmarg{0}, t_dq_arg{0}, t_dq_dispatch{0};
  static struct DelegateProfileDump2 {
    ~DelegateProfileDump2() {
      uint64_t c = t_calls.load();
      if (!c) return;
      if (nntrainer::prefill_profile_suppressed()) return;
      double g = t_gflops_x1000.load() / 1000.0;
      double tot = (t_dequant+t_reformat+t_conv+t_readback)/1e6;
      double conv_gpu_ms = t_conv_gpu_ns.load() / 1e6;
      double dq_gpu_ms = t_dequant_gpu_ns.load() / 1e6;
      double rf_gpu_ms = t_reformat_gpu_ns.load() / 1e6;
      double rb_gpu_ms = t_readback_gpu_ns.load() / 1e6;
      double sm_gpu_ms = t_svmmap_gpu_ns.load() / 1e6;
      double all_gpu_ms = conv_gpu_ms + dq_gpu_ms + rf_gpu_ms + rb_gpu_ms + sm_gpu_ms;
      double fn_total_ms = t_fn_total.load() / 1e6;
      double setup_ms = t_resource_setup.load() / 1e6;
      double accounted_ms = setup_ms + tot + all_gpu_ms;
      fprintf(stderr,
        "\n[PROFILE gemm_delegate_fp16_cl] calls=%lu gflops=%.0f\n"
        "  resource_setup: %7.1f ms\n"
        "  dequant(host):  %7.1f ms (%4.1f%%)  cache-miss=%lu/%lu\n"
        "    .alloc:       %7.1f ms  (clCreateBuffer + cache insert)\n"
        "    .svmarg:      %7.1f ms  (2x clSetKernelArgSVMPointer)\n"
        "    .arg:         %7.1f ms  (3x clSetKernelArg)\n"
        "    .dispatch:    %7.1f ms  (DispatchCommand = clEnqueueNDRangeKernel)\n"
        "  dequant(gpu):   %7.1f ms\n"
        "  reformat_in(host):%5.1f ms (%4.1f%%)  pool-hits=%lu/%lu\n"
        "  reformat_in(gpu):%6.1f ms\n"
        "  conv(wall):     %7.1f ms (%4.1f%%)\n"
        "  conv(gpu):      %7.1f ms            %.3f TFLOPS\n"
        "  readback(host): %7.1f ms (%4.1f%%)\n"
        "  readback(gpu):  %7.1f ms\n"
        "  svmmap(gpu):    %7.1f ms\n"
        "  sum(steps host):%7.1f ms            %.3f TFLOPS\n"
        "  sum(gpu events):%7.1f ms\n"
        "  fn_total(wall): %7.1f ms            "
        "(unaccounted = %.1f ms = host overhead + non-event sync gaps)\n",
        (unsigned long)c, g,
        setup_ms,
        t_dequant/1e6, t_dequant*100.0/((t_dequant+t_reformat+t_conv+t_readback) ?: 1),
        (unsigned long)t_dequant_calls.load(), (unsigned long)c,
        t_dq_alloc/1e6,
        t_dq_svmarg/1e6,
        t_dq_arg/1e6,
        t_dq_dispatch/1e6,
        dq_gpu_ms,
        t_reformat/1e6, t_reformat*100.0/((t_dequant+t_reformat+t_conv+t_readback) ?: 1),
        (unsigned long)t_reformat_hits.load(), (unsigned long)c,
        rf_gpu_ms,
        t_conv/1e6, t_conv*100.0/((t_dequant+t_reformat+t_conv+t_readback) ?: 1),
        conv_gpu_ms, g/(conv_gpu_ms/1e3)/1000.0,
        t_readback/1e6, t_readback*100.0/((t_dequant+t_reformat+t_conv+t_readback) ?: 1),
        rb_gpu_ms,
        sm_gpu_ms,
        tot, g/(tot/1e3)/1000.0,
        all_gpu_ms,
        fn_total_ms, fn_total_ms - accounted_ms);
    }
  } s_prof;

  auto now_ns = []() -> uint64_t {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
  };

  // --- Static resources (allocated once, reused across calls) ---
  // Dequant cache: weight_ptr → cl_mem (first call per layer does dequant,
  // subsequent calls reuse). With the first prefill touching every weight
  // exactly once, the cache is pure overhead — each of the ~5-8 entries
  // that fit under the cap makes its own clCreateBuffer (~5 ms on Adreno),
  // contributing 50 ms/prefill of pure alloc time. Disabled by default
  // (kMaxCacheBytes = 0), so every call falls through to the single
  // preallocated reusable buffer below. Set kMaxCacheBytes > 0 to
  // re-enable if multi-prefill process warm-up benefits outweigh the
  // first-prefill cost.
  static std::unordered_map<uintptr_t, cl_mem> s_dq_cache;
  static size_t s_dq_cache_bytes = 0;
  static constexpr size_t kMaxCacheBytes = 0;

  // Single reusable dequant buffer. Preallocated lazily on first call to
  // a size large enough for every delegate weight we will ever see in a
  // prefill (Qwen3-4B max: down_proj 2560 × 9728 fp16 = 47.5 MB), so the
  // "grow on demand" resize path never re-runs inside the prefill loop.
  static cl_mem s_w_cl = nullptr;
  static size_t s_w_size = 0;
  static constexpr size_t kPreallocWBytes = 64 * 1024 * 1024;
  static cl_mem s_xmem = nullptr;
  static cl_mem s_bias = nullptr;
  static int s_bias_slices = 0;
  static cl_mem s_src_img = nullptr;
  static int s_src_w = 0, s_src_h = 0;
  static cl_mem s_dst_img = nullptr;
  static int s_dst_w = 0, s_dst_h = 0;
  static ClContext::SharedPtrClKernel s_dq_kern, s_conv_kern;
  static ClContext::SharedPtrClKernel s_in_kern, s_out_kern;

  size_t w_halfs = (size_t)n_z * iters * 256;
  size_t w_bytes = w_halfs * 2;

  // xmem (6144 bytes, constant)
  if (!s_xmem)
    s_xmem = clCreateBuffer(clctx, CL_MEM_READ_WRITE, 6144, nullptr, &err);

  // Preallocate the reusable dequant-output buffer once, at the max size
  // any delegate weight could need. Done here so the cost lands in
  // resource_setup (one-time) rather than dequant.alloc (per-call).
  if (!s_w_cl) {
    s_w_cl = clCreateBuffer(clctx, CL_MEM_READ_WRITE, kPreallocWBytes,
                            nullptr, &err);
    s_w_size = kPreallocWBytes;
  }

  // Shape-keyed caches for bias / src_img / dst_img. Qwen3 prefill
  // cycles through ~4 distinct K values and ~4 distinct N values per
  // decoder layer; a single-slot cache was clReleaseMemObject +
  // clCreateImage-ing almost every call, which accounted for ~1 sec of
  // resource_setup time across 252 calls. Look up by (M, slices) into
  // a small map; the first ~7 calls populate it, everything after is
  // a hash hit.
  cl_image_format fmt_h = {CL_RGBA, CL_HALF_FLOAT};
  auto key2d = [](int M_, int H_) -> uint64_t {
    return ((uint64_t)(uint32_t)M_ << 32) | (uint64_t)(uint32_t)H_;
  };
  static std::unordered_map<uint64_t, cl_mem> s_src_img_cache;
  static std::unordered_map<uint64_t, cl_mem> s_dst_img_cache;
  static std::unordered_map<int, cl_mem> s_bias_cache;

  cl_mem bias_img = nullptr;
  {
    auto it = s_bias_cache.find(dst_slices);
    if (it != s_bias_cache.end()) {
      bias_img = it->second;
    } else {
      cl_image_format fmt = {CL_RGBA, CL_HALF_FLOAT};
      cl_image_desc bd = {};
      bd.image_type = CL_MEM_OBJECT_IMAGE2D;
      bd.image_width = dst_slices; bd.image_height = 1;
      bias_img = clCreateImage(clctx, CL_MEM_READ_ONLY, &fmt, &bd,
                                nullptr, &err);
      std::vector<uint16_t> z((size_t)dst_slices * 4, 0);
      size_t o[3]={0,0,0}, r[3]={(size_t)dst_slices,1,1};
      clEnqueueWriteImage(clq, bias_img, CL_TRUE, o, r, 0, 0, z.data(),
                           0, 0, 0);
      s_bias_cache[dst_slices] = bias_img;
    }
  }
  {
    const uint64_t k = key2d((int)M, src_slices);
    auto it = s_src_img_cache.find(k);
    if (it != s_src_img_cache.end()) {
      s_src_img = it->second;
    } else {
      cl_image_desc sd = {};
      sd.image_type = CL_MEM_OBJECT_IMAGE2D;
      sd.image_width = M; sd.image_height = src_slices;
      s_src_img = clCreateImage(clctx, CL_MEM_READ_ONLY, &fmt_h, &sd, 0, &err);
      s_src_img_cache[k] = s_src_img;
    }
    s_src_w = M; s_src_h = src_slices;
  }
  {
    const uint64_t k = key2d((int)M, dst_slices);
    auto it = s_dst_img_cache.find(k);
    if (it != s_dst_img_cache.end()) {
      s_dst_img = it->second;
    } else {
      cl_image_desc dd = {};
      dd.image_type = CL_MEM_OBJECT_IMAGE2D;
      dd.image_width = M; dd.image_height = dst_slices;
      s_dst_img = clCreateImage(clctx, CL_MEM_READ_WRITE, &fmt_h, &dd, 0, &err);
      s_dst_img_cache[k] = s_dst_img;
    }
    s_dst_w = M; s_dst_h = dst_slices;
  }

  // kernel objects (cached)
  if (!s_dq_kern)
    s_dq_kern = blas_cc->registerClKernel(
      dequant_int4_to_fp16_kernel, "dequant_int4_to_delegate_fp16");
  if (!s_conv_kern)
    // Match the unittest build flags for the delegate conv kernel.
    // Without -qcom-accelerate-16-bit=true the Qualcomm compiler does
    // not enable fp16 acceleration for the captured wave-memory
    // kernel and effective throughput drops from ~2.72 TFLOPS
    // (unittest_delegate_conv_wave ModelShapes_DelegateFp16) to
    // ~1.64 TFLOPS (measured via event-profiled conv(gpu) in the
    // production path).
    s_conv_kern = blas_cc->registerClKernel(
      delegate_conv_wave_kernel, "main_function",
      "@@OVERRIDE_DEFAULT@@-qcom-accelerate-16-bit=true -cl-std=CL2.0");
  if (!s_in_kern)
    s_in_kern = blas_cc->registerClKernel(
      image_reformat_kernel, "svm_to_image2d");
  if (!s_out_kern)
    s_out_kern = blas_cc->registerClKernel(
      image_reformat_kernel, "image2d_to_svm");

  // End of resource setup. Measure everything before this point.
  t_resource_setup += now_ns() - t_fn_entry;

  // GPU-side event handles for every enqueue we may do below. Left null
  // if the corresponding path is skipped (dequant cache hit, reformat
  // pool hit, etc.). Queried + released together at the end of the
  // function when NNTRAINER_PROFILE_LAYER_SYNC is set.
  cl_event dequant_ev = nullptr;
  cl_event reformat_ev = nullptr;
  cl_event readback_ev = nullptr;
  cl_event svmmap_ev = nullptr;

  // === Step 1: Dequant (cached per weight pointer, or fresh GPU dequant) ===
  const uint64_t t0 = now_ns();
  uintptr_t w_key = reinterpret_cast<uintptr_t>(weights);
  cl_mem w_cl_use = nullptr;
  auto cache_it = s_dq_cache.find(w_key);
  if (cache_it != s_dq_cache.end()) {
    w_cl_use = cache_it->second;  // cache hit — skip dequant entirely
  } else {
    // Cache miss: allocate + dequant
    const uint64_t tA = now_ns();
    // Default path: use the preallocated reusable buffer. Only take the
    // cache path if the static cap is enabled (see kMaxCacheBytes above).
    if (kMaxCacheBytes > 0 && s_dq_cache_bytes + w_bytes <= kMaxCacheBytes) {
      cl_mem new_w =
        clCreateBuffer(clctx, CL_MEM_READ_WRITE, w_bytes, nullptr, &err);
      if (!err) {
        s_dq_cache[w_key] = new_w;
        s_dq_cache_bytes += w_bytes;
        w_cl_use = new_w;
      }
    }
    if (!w_cl_use) {
      // s_w_cl was preallocated in resource_setup at kPreallocWBytes;
      // grow only if the current weight happens to exceed that bound.
      if (s_w_size < w_bytes) {
        clReleaseMemObject(s_w_cl);
        s_w_cl =
          clCreateBuffer(clctx, CL_MEM_READ_WRITE, w_bytes, nullptr, &err);
        s_w_size = w_bytes;
      }
      w_cl_use = s_w_cl;
    }
    const uint64_t tB = now_ns();
    int a = 0;
    s_dq_kern->SetKernelSVMArguments(a++, weights);
    s_dq_kern->SetKernelSVMArguments(a++, scales);
    const uint64_t tC = now_ns();
    s_dq_kern->SetKernelArguments(a++, &w_cl_use, sizeof(cl_mem));
    int sn = (int)N, sk = (int)K;
    s_dq_kern->SetKernelArguments(a++, &sn, sizeof(int));
    s_dq_kern->SetKernelArguments(a++, &sk, sizeof(int));
    const uint64_t tD = now_ns();
    // v1 kernel: 16 halves per thread.
    int tot = (int)(w_halfs / 16);
    const int dg[3] = {((tot+255)/256)*256, 1, 1};
    const int dl[3] = {256, 1, 1};
    blas_cc->command_queue_inst_.DispatchCommand(s_dq_kern, dg, dl, &dequant_ev);
    const uint64_t tE = now_ns();
    t_dq_alloc    += tB - tA;
    t_dq_svmarg   += tC - tB;
    t_dq_arg      += tD - tC;
    t_dq_dispatch += tE - tD;
    t_dequant_calls++;
  }
  const uint64_t t1 = now_ns();

  // === Step 2: GPU input reformat: SVM [M][K] → image2d ===
  //
  // Reformat cache: Qwen3 issues q_proj(x) / k_proj(x) / v_proj(x) and
  // gate_proj(x) / up_proj(x) as separate FC layers that share the same
  // activation `x`. We remember (ptr, M, K) of the last reformatted input
  // and skip svm_to_image2d when the next call consumes the same SVM
  // region unchanged. Cache invalidates automatically on any ptr / M / K
  // change, and conservatively on src_img shape recreation above.
  static uintptr_t s_last_in_ptr = 0;
  static unsigned int s_last_in_M = 0, s_last_in_K = 0;
  static cl_mem s_last_in_img = nullptr;
  const uintptr_t in_key = reinterpret_cast<uintptr_t>(input);
  // Reformat cache temporarily disabled while debugging deterministic
  // garbage output. The shared-input pattern (q/k/v + gate/up) made
  // 108/252 hits but the cache's "same ptr + same shape" key doesn't
  // account for tensor_pool reusing SVM regions across layer edges,
  // so a hit can read back a stale src_img. Re-enable once we
  // understand which invariant was wrong.
  // GpuImagePool lookup: if an upstream layer (RMSNorm image2d, SwiGLU
  // image2d, residual add image2d, ...) already published an image2d
  // for this input tensor pointer, use it as src directly and skip the
  // svm_to_image2d reformat. Miss → fall back to reformat-from-SVM.
  cl_mem src_img_for_conv = nullptr;
  int pool_src_w = 0, pool_src_h = 0;
  cl_mem pool_src =
    GpuImagePool::Global().get(input, &pool_src_w, &pool_src_h);
  // Pool image2d may have been allocated at the tensor's full height
  // (init_seq_len), while this prefill only processes the first M rows.
  // The conv kernel indexes (m,s) for m<M, so any pool_src_w >= M is
  // safe. Height (K/4 slices) must match exactly.
  const bool have_pool_src = pool_src && pool_src_w >= (int)M &&
                              pool_src_h == src_slices;
  if (have_pool_src) {
    src_img_for_conv = pool_src;
    t_reformat_hits++;
  } else {
    src_img_for_conv = s_src_img;
    if (s_last_in_ptr != in_key || s_last_in_M != M || s_last_in_K != K ||
        s_last_in_img != s_src_img) {
      int a = 0;
      s_in_kern->SetKernelSVMArguments(a++, input);
      s_in_kern->SetKernelArguments(a++, &s_src_img, sizeof(cl_mem));
      int sm = (int)M, sk = (int)K;
      s_in_kern->SetKernelArguments(a++, &sm, sizeof(int));
      s_in_kern->SetKernelArguments(a++, &sk, sizeof(int));
      const int ig[3] = {(int)((M+15)/16)*16,
                          (int)((src_slices+15)/16)*16, 1};
      const int il[3] = {16, 16, 1};
      blas_cc->command_queue_inst_.DispatchCommand(s_in_kern, ig, il,
                                                   &reformat_ev);
      s_last_in_ptr = in_key;
      s_last_in_M = M; s_last_in_K = K;
      s_last_in_img = s_src_img;
    }
  }
  // No clFinish — in-order queue guarantees dequant + reformat complete
  // before conv dispatch that follows.
  const uint64_t t2 = now_ns();

  // === Step 3: Conv dispatch ===
  cl_event conv_ev = nullptr;
  {
    struct { int x,y,z,w; } s0={1,dst_slices,(int)M,32};
    struct { int x,y,z,w; } s1={src_slices,0,0,src_slices};
    struct { int x,y,z,w; } s2={1,1,0,0};
    int a = 0;
    s_conv_kern->SetKernelArguments(a++, &w_cl_use, sizeof(cl_mem));
    s_conv_kern->SetKernelArguments(a++, &s_xmem, sizeof(cl_mem));
    s_conv_kern->SetKernelArguments(a++, &bias_img, sizeof(cl_mem));
    s_conv_kern->SetKernelArguments(a++, &s_dst_img, sizeof(cl_mem));
    s_conv_kern->SetKernelArguments(a++, &src_img_for_conv, sizeof(cl_mem));
    s_conv_kern->SetKernelArguments(a++, &s0, sizeof(s0));
    s_conv_kern->SetKernelArguments(a++, &s1, sizeof(s1));
    s_conv_kern->SetKernelArguments(a++, &s2, sizeof(s2));
    // Default local = (64, 1, 1). Tuned + production-verified against
    // the old (128,1,4) default:
    //
    //   Production verify (NNTR_DELEGATE_CONV_VERIFY=64,1,1) shows
    //   mism=0/1024 max|d|=0 rel_l2=0 for every layer call — the
    //   outputs are bit-exact identical to the (128,1,4) reference
    //   on real int4→fp16 delegate-layout weights + real prefill
    //   activations. Unittest tune measures ~9% faster (3.17 vs
    //   2.89 TFLOPS on 437×4096×2560).
    //
    // Validity condition for this kernel: local_size[0] *
    // local_size[1] must be a multiple of 64 (the Adreno subgroup
    // size). The kernel's qcom_sub_group_constant_load8 requires
    // local_id(2) to be uniform within a subgroup so f_offset =
    // Z * ... is the same on every lane participating in the load.
    // That holds iff lx*ly ∈ {64, 128, 192, 256, ...}. lx*ly = 32
    // (e.g. the bogus (32,1,2) "winner") spans two Z values inside
    // one subgroup and the load uses lane 0's f_offset for both —
    // half the lanes read the wrong weight rows and the FMA result
    // drifts into near-random noise (verified rel_l2 ≈ 1.0 on prod).
    //
    // NNTR_DELEGATE_CONV_LOCAL=X,Y,Z overrides for experiments.
    // Kernel mapping (from litert_lm's tune loop):
    //   X = group_id(1) * local[0] + local_id(0)   needs M values
    //   Y = group_id(2) * local[1] + local_id(1)   needs 1 value (h=1)
    //   Z = group_id(0) * local[2] + local_id(2)   needs (dst_slices+7)/8
    int lx = 64, ly = 1, lz = 1;
    if (const char *env = std::getenv("NNTR_DELEGATE_CONV_LOCAL")) {
      int a_, b_, c_;
      if (std::sscanf(env, "%d,%d,%d", &a_, &b_, &c_) == 3 &&
          a_ > 0 && b_ > 0 && c_ > 0) {
        lx = a_; ly = b_; lz = c_;
      }
    }
    const size_t need_z = (size_t)(dst_slices + 7) / 8;
    const size_t groups_z = (need_z + (size_t)lz - 1) / (size_t)lz;
    const size_t groups_x = ((size_t)M + (size_t)lx - 1) / (size_t)lx;
    const size_t groups_y = (1 + (size_t)ly - 1) / (size_t)ly;
    const int g[3] = {(int)(groups_z * (size_t)lx),
                      (int)(groups_x * (size_t)ly),
                      (int)(groups_y * (size_t)lz)};
    const int l[3] = {lx, ly, lz};
    // Dispatch with event for GPU-side timing. DO NOT redeclare
    // conv_ev here — the outer-scope variable (declared just above)
    // is what clGetEventProfilingInfo reads, and a shadowed inner
    // cl_event left it nullptr = inf TFLOPS in every earlier run.
    cl_command_queue raw_q = blas_cc->command_queue_inst_.GetCommandQueue();
    cl_kernel raw_k = s_conv_kern->GetKernel();
    size_t gs[3] = {(size_t)g[0], (size_t)g[1], (size_t)g[2]};
    size_t ls[3] = {(size_t)l[0], (size_t)l[1], (size_t)l[2]};

    // Per-call verification: run a second conv with candidate local into
    // a parallel dst image, read back samples of BOTH, and log where
    // they diverge. Only active when NNTR_DELEGATE_CONV_VERIFY=X,Y,Z is
    // set. The verify path uses the REAL production weights / input
    // for this call, so it catches bugs that synthetic unittest data
    // cannot (e.g. subgroup-level wave-memory errors that only show
    // up with the actual dequant-layout int4->fp16 weight pattern).
    // Adds ~one extra conv dispatch per call — enable only during
    // debug runs.
    static const char *s_verify_env =
      std::getenv("NNTR_DELEGATE_CONV_VERIFY");
    static int s_verify_x = 0, s_verify_y = 0, s_verify_z = 0;
    static bool s_verify_parsed = false;
    static cl_mem s_dst_img_cand = nullptr;
    static int s_dst_img_cand_w = 0, s_dst_img_cand_h = 0;
    static std::atomic<uint64_t> s_verify_call_idx{0};
    if (s_verify_env && !s_verify_parsed) {
      s_verify_parsed = true;
      if (std::sscanf(s_verify_env, "%d,%d,%d",
                      &s_verify_x, &s_verify_y, &s_verify_z) != 3) {
        s_verify_x = 0;
      }
    }
    if (s_verify_x > 0 && s_verify_y > 0 && s_verify_z > 0 &&
        !(s_verify_x == lx && s_verify_y == ly && s_verify_z == lz)) {
      // Ensure candidate dst image exists at the right shape.
      if (!s_dst_img_cand ||
          s_dst_img_cand_w != (int)M ||
          s_dst_img_cand_h != dst_slices) {
        if (s_dst_img_cand) clReleaseMemObject(s_dst_img_cand);
        cl_image_format fmt_c = {CL_RGBA, CL_HALF_FLOAT};
        cl_image_desc dcd = {};
        dcd.image_type = CL_MEM_OBJECT_IMAGE2D;
        dcd.image_width = M;
        dcd.image_height = dst_slices;
        cl_int cerr = 0;
        s_dst_img_cand = clCreateImage(clctx, CL_MEM_READ_WRITE, &fmt_c,
                                       &dcd, 0, &cerr);
        s_dst_img_cand_w = M;
        s_dst_img_cand_h = dst_slices;
      }
      // Dispatch candidate into s_dst_img_cand.
      s_conv_kern->SetKernelArguments(3, &s_dst_img_cand, sizeof(cl_mem));
      const size_t c_need_z = (size_t)(dst_slices + 7) / 8;
      const size_t c_gz = (c_need_z + (size_t)s_verify_z - 1) / (size_t)s_verify_z;
      const size_t c_gx = ((size_t)M + (size_t)s_verify_x - 1) / (size_t)s_verify_x;
      const size_t c_gy = (1 + (size_t)s_verify_y - 1) / (size_t)s_verify_y;
      size_t cgs[3] = {c_gz * (size_t)s_verify_x,
                       c_gx * (size_t)s_verify_y,
                       c_gy * (size_t)s_verify_z};
      size_t cls[3] = {(size_t)s_verify_x, (size_t)s_verify_y,
                       (size_t)s_verify_z};
      clEnqueueNDRangeKernel(raw_q, raw_k, 3, nullptr, cgs, cls, 0,
                              nullptr, nullptr);
      // Restore arg 3 to the production dst image.
      s_conv_kern->SetKernelArguments(3, &s_dst_img, sizeof(cl_mem));
    }

    clEnqueueNDRangeKernel(raw_q, raw_k, 3, nullptr, gs, ls, 0, nullptr, &conv_ev);

    // Sample-compare production dst vs candidate dst.
    if (s_verify_x > 0 && s_verify_y > 0 && s_verify_z > 0 &&
        !(s_verify_x == lx && s_verify_y == ly && s_verify_z == lz) &&
        s_dst_img_cand) {
      clFinish(raw_q);
      const size_t npix = (size_t)M * dst_slices;
      const size_t sample_halves = std::min<size_t>(npix * 4, 1024);
      std::vector<uint16_t> ref_buf(npix * 4), cand_buf(npix * 4);
      size_t o[3] = {0, 0, 0};
      size_t r[3] = {(size_t)M, (size_t)dst_slices, 1};
      clEnqueueReadImage(raw_q, s_dst_img, CL_TRUE, o, r, 0, 0,
                          ref_buf.data(), 0, nullptr, nullptr);
      clEnqueueReadImage(raw_q, s_dst_img_cand, CL_TRUE, o, r, 0, 0,
                          cand_buf.data(), 0, nullptr, nullptr);
      const size_t step =
        std::max<size_t>(1, (npix * 4) / sample_halves);
      int checked = 0, mism = 0;
      double max_abs = 0.0, sum_sq = 0.0, sum_ref_sq = 0.0;
      for (size_t i = 0; i < npix * 4; i += step) {
        auto h2f = [](uint16_t h) -> float {
          uint32_t sign = (uint32_t)(h & 0x8000) << 16;
          uint32_t exp = (h >> 10) & 0x1F;
          uint32_t mant = h & 0x3FF;
          uint32_t bits;
          if (exp == 0) bits = sign;
          else if (exp == 31) bits = sign | 0x7F800000 | (mant << 13);
          else bits = sign | ((exp - 15 + 127) << 23) | (mant << 13);
          float f; std::memcpy(&f, &bits, 4); return f;
        };
        float rv = h2f(ref_buf[i]), cv = h2f(cand_buf[i]);
        float d = rv - cv;
        if (std::abs(d) > std::abs(rv) * 0.05f + 1e-3f) mism++;
        if (std::abs(d) > max_abs) max_abs = std::abs(d);
        sum_sq += (double)d * d;
        sum_ref_sq += (double)rv * rv;
        checked++;
      }
      const double rel_l2 = sum_ref_sq > 0 ?
                            std::sqrt(sum_sq / sum_ref_sq) : 0.0;
      const uint64_t call_idx = s_verify_call_idx.fetch_add(1);
      fprintf(stderr,
        "[VERIFY call=%3lu M=%u N=%u K=%u] local(%d,%d,%d)→(%d,%d,%d) "
        "mism=%d/%d max|d|=%.4g rel_l2=%.4g\n",
        (unsigned long)call_idx, M, N, K, lx, ly, lz,
        s_verify_x, s_verify_y, s_verify_z,
        mism, checked, max_abs, rel_l2);
    }
  }
  const uint64_t t3 = now_ns();

  // === Step 4: GPU output reformat: image2d → SVM [M][N] ===
  {
    int a = 0;
    s_out_kern->SetKernelArguments(a++, &s_dst_img, sizeof(cl_mem));
    s_out_kern->SetKernelSVMArguments(a++, output);
    int sm = (int)M, sn = (int)N;
    s_out_kern->SetKernelArguments(a++, &sm, sizeof(int));
    s_out_kern->SetKernelArguments(a++, &sn, sizeof(int));
    const int og[3] = {(int)((M+15)/16)*16, (int)((dst_slices+15)/16)*16, 1};
    const int ol[3] = {16, 16, 1};
    blas_cc->command_queue_inst_.DispatchCommand(s_out_kern, og, ol,
                                                 &readback_ev);
  }
  // The unittest (DelegateConvWaveTest.ModelShapes_DelegateFp16)
  // clocks this kernel at 2.72 TFLOPS on Qwen3-4B shapes — 1165 ms
  // for the 252 prefill GEMMs. The old blocking SVMMap at the end
  // of every call drained the whole GPU pipeline host-side per
  // call and dropped the effective rate to 1.03 TFLOPS (readback
  // = 97% of the dotQInteger time).
  //
  // Use a NON-blocking SVMMap. It still enqueues the coherence
  // command, so any subsequent command on this in-order queue (the
  // next gemm, the next layer's clEnqueueWriteBuffer against the
  // same SVM host_ptr, etc.) is serialised after image2d_to_svm
  // above and sees coherent data — but the host thread never
  // waits here. All 252 conv dispatches can run back-to-back on
  // the GPU, approaching the raw 2.72 TFLOPS kernel rate.
  //
  // The wrapper forces blocking=CL_TRUE internally, so we call
  // clEnqueueSVMMap directly to pass CL_FALSE.
  //
  // Kill-switch: NNTRAINER_DELEGATE_SVMMAP=1 reverts to the old
  // blocking behaviour if we see correctness regressions.
  static const bool s_force_blocking_svmmap =
    std::getenv("NNTRAINER_DELEGATE_SVMMAP") != nullptr;
  {
    const cl_bool blk = s_force_blocking_svmmap ? CL_TRUE : CL_FALSE;
    clEnqueueSVMMap(clq, blk, CL_MAP_READ, output,
                    (size_t)M * N * sizeof(uint16_t), 0, nullptr, &svmmap_ev);
  }
  const uint64_t t4 = now_ns();

  // GPU event timing for EVERY sub-stage kernel. Only query profiling
  // when the user explicitly opts into it (NNTRAINER_PROFILE_LAYER_SYNC=1);
  // otherwise clWaitForEvents serializes the host with the GPU and
  // ~doubles wall time. When the env var IS set, wait on the last event
  // in the in-order queue (svmmap — or readback/conv if svmmap is null)
  // so every kernel is guaranteed done, then read each event's GPU
  // timestamp into its per-stage accumulator.
  static const bool s_event_profile =
    std::getenv("NNTRAINER_PROFILE_LAYER_SYNC") != nullptr;
  if (s_event_profile) {
    cl_event last_ev =
      svmmap_ev ? svmmap_ev :
      (readback_ev ? readback_ev : conv_ev);
    if (last_ev) clWaitForEvents(1, &last_ev);

    auto accum = [](cl_event ev, std::atomic<uint64_t> &counter) {
      if (!ev) return;
      cl_ulong s = 0, e = 0;
      clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(s),
                              &s, nullptr);
      clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(e),
                              &e, nullptr);
      if (e > s) counter += (e - s);
    };
    accum(dequant_ev, t_dequant_gpu_ns);
    accum(reformat_ev, t_reformat_gpu_ns);
    accum(conv_ev, t_conv_gpu_ns);
    accum(readback_ev, t_readback_gpu_ns);
    accum(svmmap_ev, t_svmmap_gpu_ns);
  }
  if (dequant_ev)  clReleaseEvent(dequant_ev);
  if (reformat_ev) clReleaseEvent(reformat_ev);
  if (conv_ev)     clReleaseEvent(conv_ev);
  if (readback_ev) clReleaseEvent(readback_ev);
  if (svmmap_ev)   clReleaseEvent(svmmap_ev);

  t_dequant += (t1 - t0);
  t_reformat += (t2 - t1);
  t_conv += (t3 - t2);
  t_readback += (t4 - t3);
  t_calls++;
  t_gflops_x1000 += (uint64_t)(2.0 * M * N * K / 1e6);
  t_fn_total += now_ns_outer() - t_fn_entry;
  // No per-call release — buffers are reused
}

// ============================================================================
// Batched delegate GEMM — one shared input, many weights.
//
// Designed for the Q/K/V (and gate/up) batched dispatch in
// HalfTensor::dot(std::vector<Tensor*>, ...) where every weight consumes
// the same activation. Cuts the per-call SVM->image2d reformat from N to
// 1, and replaces N blocking SVMMaps with a single end-of-batch SVMMap.
//
// Falls back on the caller (HalfTensor) if any Ns[i] % 32 != 0 or K % 8.
// ============================================================================
void gemm_delegate_fp16_cl_batched(uint16_t *input,
                                   const std::vector<uint16_t *> &weights,
                                   const std::vector<uint16_t *> &scales,
                                   const std::vector<uint16_t *> &outputs,
                                   const std::vector<unsigned int> &Ns,
                                   unsigned int M, unsigned int K) {
  const size_t count = weights.size();
  if (count == 0) return;
  if (scales.size() != count || outputs.size() != count || Ns.size() != count)
    throw std::runtime_error(
      "gemm_delegate_fp16_cl_batched: vector size mismatch");
  if ((K % 8) != 0)
    throw std::runtime_error(
      "gemm_delegate_fp16_cl_batched requires K%8==0");
  for (size_t i = 0; i < count; ++i) {
    if ((Ns[i] % 32) != 0)
      throw std::runtime_error(
        "gemm_delegate_fp16_cl_batched requires all Ns[i]%32==0");
  }

  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  cl_context clctx = blas_cc->context_inst_.GetContext();

  const int src_slices = K / 4;
  cl_int err;

  auto now_ns = []() -> uint64_t {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
  };

  // --- Profiling ---
  static std::atomic<uint64_t> t_dequant{0}, t_reformat{0}, t_conv{0},
    t_readback{0}, t_calls{0}, t_weights_total{0}, t_gflops_x1000{0};
  static struct BatchedProfileDump {
    ~BatchedProfileDump() {
      uint64_t c = t_calls.load();
      if (!c) return;
      if (nntrainer::prefill_profile_suppressed()) return;
      double g = t_gflops_x1000.load() / 1000.0;
      double tot = (t_dequant+t_reformat+t_conv+t_readback)/1e6;
      fprintf(stderr,
        "\n[PROFILE gemm_delegate_fp16_cl_batched] batches=%lu weights=%lu "
        "gflops=%.0f\n"
        "  dequant:     %7.1f ms\n"
        "  reformat_in: %7.1f ms  (once per batch — shared across weights)\n"
        "  conv:        %7.1f ms\n"
        "  readback:    %7.1f ms  (reformat_out N*, SVMMap once)\n"
        "  total:       %7.1f ms  %.3f TFLOPS\n",
        (unsigned long)c, (unsigned long)t_weights_total.load(), g,
        t_dequant/1e6, t_reformat/1e6, t_conv/1e6, t_readback/1e6,
        tot, g/(tot/1e3)/1000.0);
    }
  } s_prof;

  // --- Static resources (shared across batched calls) ---
  static std::unordered_map<uintptr_t, cl_mem> s_dq_cache;
  static size_t s_dq_cache_bytes = 0;
  static constexpr size_t kMaxCacheBytes = 200 * 1024 * 1024;

  static cl_mem s_xmem = nullptr;
  static cl_mem s_bias = nullptr;
  static int s_bias_slices = 0;
  static cl_mem s_src_img = nullptr;
  static int s_src_w = 0, s_src_h = 0;
  // dst_img is reused per-weight inside the batch; the in-order queue
  // serialises conv[i] -> image2d_to_svm[i] -> conv[i+1] so the buffer
  // never races with itself.
  static cl_mem s_dst_img = nullptr;
  static int s_dst_w = 0, s_dst_h = 0;
  static ClContext::SharedPtrClKernel s_dq_kern, s_conv_kern;
  static ClContext::SharedPtrClKernel s_in_kern, s_out_kern;

  // xmem (6144 bytes, constant scratch required by the wave kernel)
  if (!s_xmem)
    s_xmem = clCreateBuffer(clctx, CL_MEM_READ_WRITE, 6144, nullptr, &err);

  // Grow bias image to max dst_slices across the batch.
  unsigned int max_N = 0;
  for (size_t i = 0; i < count; ++i)
    if (Ns[i] > max_N) max_N = Ns[i];
  const int max_dst_slices = (int)(max_N / 4);
  if (!s_bias || s_bias_slices < max_dst_slices) {
    if (s_bias) clReleaseMemObject(s_bias);
    cl_image_format fmt = {CL_RGBA, CL_HALF_FLOAT};
    cl_image_desc bd = {};
    bd.image_type = CL_MEM_OBJECT_IMAGE2D;
    bd.image_width = max_dst_slices; bd.image_height = 1;
    s_bias = clCreateImage(clctx, CL_MEM_READ_ONLY, &fmt, &bd, nullptr, &err);
    std::vector<uint16_t> z((size_t)max_dst_slices * 4, 0);
    size_t o[3]={0,0,0}, r[3]={(size_t)max_dst_slices,1,1};
    clEnqueueWriteImage(blas_cc->command_queue_inst_.GetCommandQueue(),
                        s_bias, CL_TRUE, o, r, 0, 0, z.data(), 0, 0, 0);
    s_bias_slices = max_dst_slices;
  }

  // src image (shared across the batch)
  cl_image_format fmt_h = {CL_RGBA, CL_HALF_FLOAT};
  if (!s_src_img || s_src_w != (int)M || s_src_h != src_slices) {
    if (s_src_img) clReleaseMemObject(s_src_img);
    cl_image_desc sd = {};
    sd.image_type = CL_MEM_OBJECT_IMAGE2D;
    sd.image_width = M; sd.image_height = src_slices;
    s_src_img = clCreateImage(clctx, CL_MEM_READ_ONLY, &fmt_h, &sd, 0, &err);
    s_src_w = M; s_src_h = src_slices;
  }

  // dst image grown to the max slices we'll need.
  if (!s_dst_img || s_dst_w != (int)M || s_dst_h < max_dst_slices) {
    if (s_dst_img) clReleaseMemObject(s_dst_img);
    cl_image_desc dd = {};
    dd.image_type = CL_MEM_OBJECT_IMAGE2D;
    dd.image_width = M; dd.image_height = max_dst_slices;
    s_dst_img = clCreateImage(clctx, CL_MEM_READ_WRITE, &fmt_h, &dd, 0, &err);
    s_dst_w = M; s_dst_h = max_dst_slices;
  }

  if (!s_dq_kern)
    s_dq_kern = blas_cc->registerClKernel(
      dequant_int4_to_fp16_kernel, "dequant_int4_to_delegate_fp16");
  if (!s_conv_kern)
    // Match the unittest build flags for the delegate conv kernel.
    // Without -qcom-accelerate-16-bit=true the Qualcomm compiler does
    // not enable fp16 acceleration for the captured wave-memory
    // kernel and effective throughput drops from ~2.72 TFLOPS
    // (unittest_delegate_conv_wave ModelShapes_DelegateFp16) to
    // ~1.64 TFLOPS (measured via event-profiled conv(gpu) in the
    // production path).
    s_conv_kern = blas_cc->registerClKernel(
      delegate_conv_wave_kernel, "main_function",
      "@@OVERRIDE_DEFAULT@@-qcom-accelerate-16-bit=true -cl-std=CL2.0");
  if (!s_in_kern)
    s_in_kern = blas_cc->registerClKernel(
      image_reformat_kernel, "svm_to_image2d");
  if (!s_out_kern)
    s_out_kern = blas_cc->registerClKernel(
      image_reformat_kernel, "image2d_to_svm");

  // === Step 1: shared SVM -> image2d input reformat (once per batch) ===
  const uint64_t ti0 = now_ns();
  {
    int a = 0;
    s_in_kern->SetKernelSVMArguments(a++, input);
    s_in_kern->SetKernelArguments(a++, &s_src_img, sizeof(cl_mem));
    int sm = (int)M, sk = (int)K;
    s_in_kern->SetKernelArguments(a++, &sm, sizeof(int));
    s_in_kern->SetKernelArguments(a++, &sk, sizeof(int));
    const int ig[3] = {(int)((M+15)/16)*16,
                        (int)((src_slices+15)/16)*16, 1};
    const int il[3] = {16, 16, 1};
    blas_cc->command_queue_inst_.DispatchCommand(s_in_kern, ig, il);
  }
  const uint64_t ti1 = now_ns();
  t_reformat += (ti1 - ti0);

  // === Step 2..: per-weight dequant + conv + image2d_to_svm ===
  uint64_t dq_ns = 0, conv_ns = 0, rb_ns = 0;
  for (size_t i = 0; i < count; ++i) {
    const unsigned int Ni = Ns[i];
    const int dst_slices = Ni / 4;
    const int iters = src_slices / 2;
    const int n_z = (dst_slices + 7) / 8;
    const size_t w_halfs = (size_t)n_z * iters * 256;
    const size_t w_bytes = w_halfs * 2;

    // --- Dequant (cached per weight pointer) ---
    const uint64_t td0 = now_ns();
    uintptr_t w_key = reinterpret_cast<uintptr_t>(weights[i]);
    cl_mem w_cl_use = nullptr;
    auto cache_it = s_dq_cache.find(w_key);
    if (cache_it != s_dq_cache.end()) {
      w_cl_use = cache_it->second;
    } else {
      if (s_dq_cache_bytes + w_bytes <= kMaxCacheBytes) {
        cl_mem new_w =
          clCreateBuffer(clctx, CL_MEM_READ_WRITE, w_bytes, nullptr, &err);
        if (!err) {
          s_dq_cache[w_key] = new_w;
          s_dq_cache_bytes += w_bytes;
          w_cl_use = new_w;
        }
      }
      if (!w_cl_use) {
        // Cache full — dequant into a non-cached buffer for this call only.
        // Leaks until process exit; fine because this path only triggers
        // when model exceeds kMaxCacheBytes of delegate weights.
        w_cl_use =
          clCreateBuffer(clctx, CL_MEM_READ_WRITE, w_bytes, nullptr, &err);
      }
      int a = 0;
      s_dq_kern->SetKernelSVMArguments(a++, weights[i]);
      s_dq_kern->SetKernelSVMArguments(a++, scales[i]);
      s_dq_kern->SetKernelArguments(a++, &w_cl_use, sizeof(cl_mem));
      int sn = (int)Ni, sk = (int)K;
      s_dq_kern->SetKernelArguments(a++, &sn, sizeof(int));
      s_dq_kern->SetKernelArguments(a++, &sk, sizeof(int));
      // v1 kernel: 16 halves per thread.
      int tot = (int)(w_halfs / 16);
      const int dg[3] = {((tot+255)/256)*256, 1, 1};
      const int dl[3] = {256, 1, 1};
      blas_cc->command_queue_inst_.DispatchCommand(s_dq_kern, dg, dl);
    }
    const uint64_t td1 = now_ns();
    dq_ns += (td1 - td0);

    // --- Conv dispatch (shared src_img, per-weight w_cl_use, shared dst_img) ---
    {
      struct { int x,y,z,w; } s0={1,dst_slices,(int)M,32};
      struct { int x,y,z,w; } s1={src_slices,0,0,src_slices};
      struct { int x,y,z,w; } s2={1,1,0,0};
      int a = 0;
      s_conv_kern->SetKernelArguments(a++, &w_cl_use, sizeof(cl_mem));
      s_conv_kern->SetKernelArguments(a++, &s_xmem, sizeof(cl_mem));
      s_conv_kern->SetKernelArguments(a++, &s_bias, sizeof(cl_mem));
      s_conv_kern->SetKernelArguments(a++, &s_dst_img, sizeof(cl_mem));
      s_conv_kern->SetKernelArguments(a++, &s_src_img, sizeof(cl_mem));
      s_conv_kern->SetKernelArguments(a++, &s0, sizeof(s0));
      s_conv_kern->SetKernelArguments(a++, &s1, sizeof(s1));
      s_conv_kern->SetKernelArguments(a++, &s2, sizeof(s2));
      size_t gz = ((dst_slices+7)/8+3)/4;
      const int g[3] = {(int)(gz*128), (int)((M+127)/128), 4};
      const int l[3] = {128, 1, 4};
      // NOTE: the main gemm_delegate_fp16_cl uses local=(32,1,2) which
      // tuned ~9% faster on all Qwen3-4B prefill shapes. The batched
      // (MoE) variant has not been tuned yet — retune on representative
      // MoE shapes before migrating.
      blas_cc->command_queue_inst_.DispatchCommand(s_conv_kern, g, l);
    }
    const uint64_t td2 = now_ns();
    conv_ns += (td2 - td1);

    // --- image2d -> SVM output (no SVMMap yet) ---
    {
      int a = 0;
      s_out_kern->SetKernelArguments(a++, &s_dst_img, sizeof(cl_mem));
      s_out_kern->SetKernelSVMArguments(a++, outputs[i]);
      int sm = (int)M, sn = (int)Ni;
      s_out_kern->SetKernelArguments(a++, &sm, sizeof(int));
      s_out_kern->SetKernelArguments(a++, &sn, sizeof(int));
      const int og[3] = {(int)((M+15)/16)*16,
                          (int)((dst_slices+15)/16)*16, 1};
      const int ol[3] = {16, 16, 1};
      blas_cc->command_queue_inst_.DispatchCommand(s_out_kern, og, ol);
    }
    const uint64_t td3 = now_ns();
    rb_ns += (td3 - td2);

    t_gflops_x1000 += (uint64_t)(2.0 * M * Ni * K / 1e6);
  }

  // === Step N+1: single end-of-batch SVMMap for host coherence ===
  //
  // One SVMMap drains the in-order queue, so the host sees consistent
  // data across ALL outputs. This replaces N per-weight blocking maps.
  const uint64_t tm0 = now_ns();
  for (size_t i = 0; i < count; ++i) {
    blas_cc->command_queue_inst_.enqueueSVMMap(
      outputs[i], (size_t)M * Ns[i] * sizeof(uint16_t),
      /*blocking=*/(i + 1 == count));
  }
  const uint64_t tm1 = now_ns();
  rb_ns += (tm1 - tm0);

  t_dequant += dq_ns;
  t_conv += conv_ns;
  t_readback += rb_ns;
  t_calls++;
  t_weights_total += count;
}

// ============================================================================
// Phase A helper — svm_to_image2d + GpuImagePool publish.
// ============================================================================
void svm_to_image2d_publish(void *svm_ptr, unsigned int M, unsigned int K) {
  if (!svm_ptr || M == 0 || K == 0 || (K % 4) != 0) return;
  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  if (!blas_cc) return;
  cl_context clctx = blas_cc->context_inst_.GetContext();

  const int slices = (int)K / 4;
  cl_int err = 0;

  struct Cached { cl_mem img; int M, slices; };
  static std::unordered_map<uintptr_t, Cached> s_cache;
  const uintptr_t key = reinterpret_cast<uintptr_t>(svm_ptr);
  cl_mem img = nullptr;
  auto it = s_cache.find(key);
  if (it != s_cache.end() && it->second.M == (int)M &&
      it->second.slices == slices) {
    img = it->second.img;
  } else {
    if (it != s_cache.end() && it->second.img)
      clReleaseMemObject(it->second.img);
    cl_image_format fmt = {CL_RGBA, CL_HALF_FLOAT};
    cl_image_desc d = {};
    d.image_type = CL_MEM_OBJECT_IMAGE2D;
    d.image_width = M; d.image_height = slices;
    img = clCreateImage(clctx, CL_MEM_READ_WRITE, &fmt, &d, 0, &err);
    s_cache[key] = {img, (int)M, slices};
  }
  if (!img) return;

  static ClContext::SharedPtrClKernel s_in_kern;
  if (!s_in_kern) {
    s_in_kern = blas_cc->registerClKernel(image_reformat_kernel,
                                          "svm_to_image2d");
  }

  // The caller has just CPU-written to svm_ptr (e.g. RMSNorm NEON
  // compute). On Adreno coarse-grained SVM those writes live in the
  // CPU cache until an SVMUnmap commits them; without the unmap the
  // GPU kernel we dispatch below reads stale data.
  blas_cc->command_queue_inst_.enqueueSVMUnmap(svm_ptr);

  int a = 0;
  s_in_kern->SetKernelSVMArguments(a++, svm_ptr);
  s_in_kern->SetKernelArguments(a++, &img, sizeof(cl_mem));
  int sm = (int)M, sk = (int)K;
  s_in_kern->SetKernelArguments(a++, &sm, sizeof(int));
  s_in_kern->SetKernelArguments(a++, &sk, sizeof(int));
  const int ig[3] = {((int)M + 15) / 16 * 16,
                      (slices + 15) / 16 * 16, 1};
  const int il[3] = {16, 16, 1};
  blas_cc->command_queue_inst_.DispatchCommand(s_in_kern, ig, il);

  GpuImagePool::Global().set(svm_ptr, img, (int)M, slices);
}

// ============================================================================
// GPU RMSNorm via rmsnorm_image2d_v2. Env-gated opt-in through
// NNTRAINER_RMSNORM_GPU=1 (checked at the layer level). See the header
// documentation for semantics.
// ============================================================================
void rmsnorm_image2d_cl(void *in_svm, void *out_svm,
                        const float *gamma, unsigned int M, unsigned int K,
                        float epsilon) {
  if (!in_svm || !out_svm || !gamma || M == 0 || K == 0 || (K % 4) != 0) return;
  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  if (!blas_cc) return;
  cl_context clctx = blas_cc->context_inst_.GetContext();
  cl_command_queue clq = blas_cc->command_queue_inst_.GetCommandQueue();

  const int slices = (int)K / 4;
  cl_int err = 0;

  // --- gamma: cache a cl_mem buffer per gamma pointer. ---
  struct GammaCache { cl_mem buf; size_t bytes; };
  static std::unordered_map<uintptr_t, GammaCache> s_gamma_cache;
  cl_mem gamma_buf = nullptr;
  {
    const uintptr_t gk = reinterpret_cast<uintptr_t>(gamma);
    const size_t need = (size_t)K * sizeof(float);
    auto it = s_gamma_cache.find(gk);
    if (it != s_gamma_cache.end() && it->second.bytes >= need) {
      gamma_buf = it->second.buf;
    } else {
      if (it != s_gamma_cache.end() && it->second.buf)
        clReleaseMemObject(it->second.buf);
      gamma_buf = clCreateBuffer(clctx,
                                 CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 need, (void *)gamma, &err);
      if (err != CL_SUCCESS || !gamma_buf) return;
      s_gamma_cache[gk] = {gamma_buf, need};
    }
  }

  // --- input image2d: pool hit, or on-demand svm_to_image2d. ---
  int pool_M = 0, pool_slices = 0;
  cl_mem in_img = GpuImagePool::Global().get(in_svm, &pool_M, &pool_slices);
  bool in_img_is_pool = (in_img && pool_M == (int)M && pool_slices == slices);
  if (!in_img_is_pool) {
    // Allocate/reuse a staging image2d keyed by in_svm. Same cached-by-
    // ptr pattern as svm_to_image2d_publish so we avoid a per-call alloc.
    struct StagedIn { cl_mem img; int M, slices; };
    static std::unordered_map<uintptr_t, StagedIn> s_in_cache;
    const uintptr_t ik = reinterpret_cast<uintptr_t>(in_svm);
    auto it = s_in_cache.find(ik);
    if (it != s_in_cache.end() && it->second.M == (int)M &&
        it->second.slices == slices) {
      in_img = it->second.img;
    } else {
      if (it != s_in_cache.end() && it->second.img)
        clReleaseMemObject(it->second.img);
      cl_image_format fmt = {CL_RGBA, CL_HALF_FLOAT};
      cl_image_desc d = {};
      d.image_type = CL_MEM_OBJECT_IMAGE2D;
      d.image_width = M; d.image_height = slices;
      in_img = clCreateImage(clctx, CL_MEM_READ_WRITE, &fmt, &d, 0, &err);
      if (err != CL_SUCCESS || !in_img) return;
      s_in_cache[ik] = {in_img, (int)M, slices};
    }
    // Make the CPU-side writes visible before the GPU kernel reads them.
    blas_cc->command_queue_inst_.enqueueSVMUnmap(in_svm);
    // Reformat svm -> image2d.
    static ClContext::SharedPtrClKernel s_in_kern;
    if (!s_in_kern) {
      s_in_kern = blas_cc->registerClKernel(image_reformat_kernel,
                                            "svm_to_image2d");
    }
    if (s_in_kern) {
      int a = 0;
      s_in_kern->SetKernelSVMArguments(a++, in_svm);
      s_in_kern->SetKernelArguments(a++, &in_img, sizeof(cl_mem));
      int sm = (int)M, sk = (int)K;
      s_in_kern->SetKernelArguments(a++, &sm, sizeof(int));
      s_in_kern->SetKernelArguments(a++, &sk, sizeof(int));
      const int ig[3] = {((int)M + 15) / 16 * 16,
                          (slices + 15) / 16 * 16, 1};
      const int il[3] = {16, 16, 1};
      blas_cc->command_queue_inst_.DispatchCommand(s_in_kern, ig, il);
    }
  }

  // --- output image2d: cache keyed by out_svm. ---
  struct StagedOut { cl_mem img; int M, slices; };
  static std::unordered_map<uintptr_t, StagedOut> s_out_cache;
  cl_mem out_img = nullptr;
  {
    const uintptr_t ok = reinterpret_cast<uintptr_t>(out_svm);
    auto it = s_out_cache.find(ok);
    if (it != s_out_cache.end() && it->second.M == (int)M &&
        it->second.slices == slices) {
      out_img = it->second.img;
    } else {
      if (it != s_out_cache.end() && it->second.img)
        clReleaseMemObject(it->second.img);
      cl_image_format fmt = {CL_RGBA, CL_HALF_FLOAT};
      cl_image_desc d = {};
      d.image_type = CL_MEM_OBJECT_IMAGE2D;
      d.image_width = M; d.image_height = slices;
      out_img = clCreateImage(clctx, CL_MEM_READ_WRITE, &fmt, &d, 0, &err);
      if (err != CL_SUCCESS || !out_img) return;
      s_out_cache[ok] = {out_img, (int)M, slices};
    }
  }

  // --- rmsnorm kernel dispatch ---
  static ClContext::SharedPtrClKernel s_rms_kern;
  if (!s_rms_kern) {
    s_rms_kern = blas_cc->registerClKernel(rmsnorm_image2d_v2_kernel,
                                           "rmsnorm_image2d_v2");
  }
  if (s_rms_kern) {
    int a = 0;
    s_rms_kern->SetKernelArguments(a++, &in_img, sizeof(cl_mem));
    s_rms_kern->SetKernelArguments(a++, &out_img, sizeof(cl_mem));
    s_rms_kern->SetKernelArguments(a++, &gamma_buf, sizeof(cl_mem));
    const float eps = epsilon;
    s_rms_kern->SetKernelArguments(a++, &eps, sizeof(float));
    const int sm_i = (int)M, ss_i = slices;
    s_rms_kern->SetKernelArguments(a++, &sm_i, sizeof(int));
    s_rms_kern->SetKernelArguments(a++, &ss_i, sizeof(int));
    // WG_SIZE=64 in the kernel; one workgroup per position.
    const int g[3] = {(int)M * 64, 1, 1};
    const int l[3] = {64, 1, 1};
    blas_cc->command_queue_inst_.DispatchCommand(s_rms_kern, g, l);
  }

  // --- readback image2d -> SVM + publish to pool ---
  static ClContext::SharedPtrClKernel s_out_kern;
  if (!s_out_kern) {
    s_out_kern = blas_cc->registerClKernel(image_reformat_kernel,
                                           "image2d_to_svm");
  }
  if (s_out_kern) {
    int a = 0;
    s_out_kern->SetKernelArguments(a++, &out_img, sizeof(cl_mem));
    s_out_kern->SetKernelSVMArguments(a++, out_svm);
    int sm = (int)M, sk = (int)K;
    s_out_kern->SetKernelArguments(a++, &sm, sizeof(int));
    s_out_kern->SetKernelArguments(a++, &sk, sizeof(int));
    const int og[3] = {((int)M + 15) / 16 * 16,
                        (slices + 15) / 16 * 16, 1};
    const int ol[3] = {16, 16, 1};
    blas_cc->command_queue_inst_.DispatchCommand(s_out_kern, og, ol);
  }
  // Publish out_svm to GpuImagePool so the next gemm_delegate pool-hits
  // its svm_to_image2d reformat.
  //
  // Go through svm_to_image2d_publish (one extra kernel dispatch that
  // re-reads out_svm and writes a fresh image2d) instead of calling
  // GpuImagePool::set(out_svm, out_img, …) directly. Measured: the
  // direct-handoff variant produces garbage downstream output on this
  // device, most likely due to Adreno image2d tiling/flush state that
  // the SVM round-trip clears. The v2 kernel itself was verified
  // numerically correct via gpu_check (max_rel ~0.1%, fp16 rounding).
  svm_to_image2d_publish(out_svm, (unsigned int)M, (unsigned int)K);

  // Decode (M == 1) falls back to gemv_int4_adreno_cl downstream; that
  // path starts with a CPU scalar copy `svm_in[k] = in_u16[k]` (see
  // HalfTensor::dotQInteger M=1 branch). Without a blocking host fence
  // here, CPU reads of out_svm return pre-kernel contents and the gemv
  // consumes garbage — every decode token lands on the wrong row of
  // the int4 weight matmul and the generation goes off the rails
  // while prefill (which reads image2d via pool) stays coherent.
  //
  // For prefill (M > 1) the next gemm_delegate reads via pool image2d,
  // so the fence isn't needed and we'd be stalling the pipeline for
  // no reason.
  if (M == 1) {
    blas_cc->command_queue_inst_.enqueueSVMMap(
      out_svm, (size_t)M * K * sizeof(uint16_t), /*read_only=*/true);
  }
}

#ifdef ENABLE_FP16
// ============================================================================
// Fused FlashAttention (qk + softmax + av) via attention_fused_fp16.
// Env-gated opt-in through NNTRAINER_ATTN_GPU=1 in MHACoreLayer.  One
// dispatch per layer call replaces the three NEON stages; online-softmax
// running stats stay in __local across the K loop.
// ============================================================================
void attention_fused_fp16_cl(void *q_svm, void *k_cache_svm,
                              void *v_cache_svm,
                              void *out_svm,
                              unsigned int M, unsigned int T,
                              unsigned int from,
                              unsigned int num_heads_Q,
                              unsigned int gqa_size,
                              unsigned int head_dim, int is_causal) {
  if (!q_svm || !k_cache_svm || !v_cache_svm || !out_svm ||
      M == 0 || T == 0 || num_heads_Q == 0 || gqa_size == 0 ||
      (num_heads_Q % gqa_size) != 0)
    return;
  if (head_dim != 128) return;
  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  if (!blas_cc) return;

  static ClContext::SharedPtrClKernel s_kern;
  if (!s_kern) {
    // -cl-std=CL2.0 override matches the ClContext init registration so
    // ocl_kernel_map dedupes on (name + options).  Required for
    // work_group_reduce_add.
    s_kern = blas_cc->registerClKernel(
      attention_fused_fp16_kernel, "attention_fused_fp16",
      "@@OVERRIDE_DEFAULT@@-qcom-accelerate-16-bit=true -cl-std=CL2.0");
  }
  if (!s_kern) return;

  // Commit upstream CPU writes before the GPU reads.
  blas_cc->command_queue_inst_.enqueueSVMUnmap(q_svm);
  blas_cc->command_queue_inst_.enqueueSVMUnmap(k_cache_svm);
  blas_cc->command_queue_inst_.enqueueSVMUnmap(v_cache_svm);
  blas_cc->command_queue_inst_.enqueueSVMUnmap(out_svm);

  const float scale = 1.0f / std::sqrt((float)head_dim);
  const int M_i = (int)M, T_i = (int)T, from_i = (int)from;
  const int nhq = (int)num_heads_Q, gqa = (int)gqa_size;

  int a = 0;
  s_kern->SetKernelSVMArguments(a++, q_svm);
  s_kern->SetKernelSVMArguments(a++, k_cache_svm);
  s_kern->SetKernelSVMArguments(a++, v_cache_svm);
  s_kern->SetKernelSVMArguments(a++, out_svm);
  s_kern->SetKernelArguments(a++, &M_i, sizeof(int));
  s_kern->SetKernelArguments(a++, &T_i, sizeof(int));
  s_kern->SetKernelArguments(a++, &from_i, sizeof(int));
  s_kern->SetKernelArguments(a++, &nhq, sizeof(int));
  s_kern->SetKernelArguments(a++, &gqa, sizeof(int));
  s_kern->SetKernelArguments(a++, &is_causal, sizeof(int));
  s_kern->SetKernelArguments(a++, &scale, sizeof(float));

  // Step 6: WG still (64, 1, 1) = one real wavefront, but the WG now
  // processes TQ consecutive Q rows via per-thread register tiling.
  // One K / V fetch per kk feeds TQ rows -> K/V traffic drops by TQ.
  // TQ must match the attention_fused_fp16.cl #define (=2 here).
  constexpr int TQ = 4;
  const int m_tiles = (M_i + TQ - 1) / TQ;
  const int g[3] = {64, nhq, m_tiles};
  const int l[3] = {64, 1, 1};
  blas_cc->command_queue_inst_.DispatchCommand(s_kern, g, l);

  blas_cc->command_queue_inst_.enqueueSVMMap(
    out_svm,
    (size_t)M * (size_t)num_heads_Q * (size_t)head_dim * sizeof(uint16_t),
    /*read_only=*/true);
}

#endif

// ============================================================================
// Phase B helper — fp16 SwiGLU on SVM buffers via swiglu_cl_fp16 kernel.
// ============================================================================
void swiglu_fp16_svm_cl(const void *in1, const void *in2, void *out,
                        size_t total) {
  if (!in1 || !in2 || !out || total == 0) return;
  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  if (!blas_cc) return;

  static ClContext::SharedPtrClKernel s_kern;
  if (!s_kern) {
    s_kern = blas_cc->registerClKernel(swiglu_fp16_kernel, "swiglu_cl_fp16");
  }
  if (!s_kern) return;

  s_kern->SetKernelSVMArguments(0, in1);
  s_kern->SetKernelSVMArguments(1, in2);
  s_kern->SetKernelSVMArguments(2, out);

  const int total_i = (int)total;
  const int32_t desired_local = 64;
  const int32_t chosen_local = total_i >= desired_local ? desired_local
                                                        : total_i;
  const int g[3] = {total_i, 1, 1};
  const int l[3] = {chosen_local, 1, 1};
  blas_cc->command_queue_inst_.DispatchCommand(s_kern, g, l);
}

// ============================================================================
// Phase B helper — fp16 residual add on SVM.
// ============================================================================
void add2_fp16_svm_cl(const void *a, const void *b, void *out, size_t total) {
  if (!a || !b || !out || total == 0) return;
  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  if (!blas_cc) return;

  static ClContext::SharedPtrClKernel s_kern;
  if (!s_kern) {
    s_kern = blas_cc->registerClKernel(add2_fp16_svm_kernel, "add2_fp16_svm");
  }
  if (!s_kern) return;

  s_kern->SetKernelSVMArguments(0, a);
  s_kern->SetKernelSVMArguments(1, b);
  s_kern->SetKernelSVMArguments(2, out);
  const int total_i = (int)total;
  s_kern->SetKernelArguments(3, &total_i, sizeof(int));

  const int32_t desired_local = 64;
  const int32_t chosen_local = total_i >= desired_local ? desired_local
                                                        : total_i;
  // Round up to local size so no work-items fall outside workgroups.
  const int gcount = ((total_i + chosen_local - 1) / chosen_local) *
                     chosen_local;
  const int g[3] = {gcount, 1, 1};
  const int l[3] = {chosen_local, 1, 1};
  blas_cc->command_queue_inst_.DispatchCommand(s_kern, g, l);
}

} // namespace nntrainer
#if 0 // OLD — duplicate removed
  if (((N % 32) != 0) || ((K % 8) != 0))
    throw std::runtime_error(
      "gemm_delegate_fp16_cl requires N%32==0 and K%8==0");

  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  cl_context clctx = blas_cc->context_inst_.GetContext();
  cl_command_queue clq = blas_cc->command_queue_inst_.GetCommandQueue();

  const int src_slices = K / 4;
  const int dst_slices = N / 4;
  const int n_z = (dst_slices + 7) / 8;
  const int iters = src_slices / 2;
  cl_int err;

  static std::atomic<uint64_t> t_dequant{0}, t_in_reformat{0}, t_conv{0},
    t_out_reformat{0}, t_calls{0};
  static std::atomic<uint64_t> t_total_gflops_x1000{0};
  static struct DelegateProfileDump {
    ~DelegateProfileDump() {
      uint64_t c = t_calls.load();
      if (c == 0) return;
      if (nntrainer::prefill_profile_suppressed()) return;
      double total_ms = (t_dequant+t_in_reformat+t_conv+t_out_reformat)/1e6;
      double gflops = t_total_gflops_x1000.load() / 1000.0;
      fprintf(stderr,
        "\n[PROFILE gemm_delegate_fp16_cl] calls=%lu  total_gflops=%.1f\n"
        "  dequant:       %7.1f ms (%5.1f%%)  %.3f TFLOPS\n"
        "  in_reformat:   %7.1f ms (%5.1f%%)\n"
        "  conv:          %7.1f ms (%5.1f%%)  %.3f TFLOPS\n"
        "  out_reformat:  %7.1f ms (%5.1f%%)\n"
        "  total:         %7.1f ms           %.3f TFLOPS (effective)\n",
        (unsigned long)c, gflops,
        t_dequant/1e6, t_dequant*100.0/(t_dequant+t_in_reformat+t_conv+t_out_reformat),
          gflops / (t_dequant/1e9) / 1000.0,
        t_in_reformat/1e6, t_in_reformat*100.0/(t_dequant+t_in_reformat+t_conv+t_out_reformat),
        t_conv/1e6, t_conv*100.0/(t_dequant+t_in_reformat+t_conv+t_out_reformat),
          gflops / (t_conv/1e9) / 1000.0,
        t_out_reformat/1e6, t_out_reformat*100.0/(t_dequant+t_in_reformat+t_conv+t_out_reformat),
        total_ms, gflops / (total_ms/1e3) / 1000.0);
    }
  } s_delegate_profile_dump;

  auto now_ns = []() -> uint64_t {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
  };

  const uint64_t t0 = now_ns();

  // --- 1. Weight dequant on GPU (per-call, no cache) ---
  size_t out_halfs = (size_t)n_z * iters * 256;
  size_t out_bytes = out_halfs * sizeof(uint16_t);
  cl_mem w_cl = clCreateBuffer(clctx, CL_MEM_READ_WRITE, out_bytes, nullptr, &err);
  if (err) throw std::runtime_error("delegate: fp16 weight buf failed");

  {
    // Use SVM pointers directly — no host→device copy needed
    ClContext::SharedPtrClKernel dq_kern = blas_cc->registerClKernel(
      dequant_int4_to_fp16_kernel, "dequant_int4_to_delegate_fp16");
    if (!dq_kern) throw std::runtime_error("delegate: dequant kernel failed");

    int dq_arg = 0;
    dq_kern->SetKernelSVMArguments(dq_arg++, weights);
    dq_kern->SetKernelSVMArguments(dq_arg++, scales);
    dq_kern->SetKernelArguments(dq_arg++, &w_cl, sizeof(cl_mem));
    int size_n = (int)N, size_k = (int)K;
    dq_kern->SetKernelArguments(dq_arg++, &size_n, sizeof(int));
    dq_kern->SetKernelArguments(dq_arg++, &size_k, sizeof(int));

    int total_items = (int)out_halfs;
    const int dq_global[3] = {((total_items + 255) / 256) * 256, 1, 1};
    const int dq_local[3] = {256, 1, 1};
    blas_cc->command_queue_inst_.DispatchCommand(dq_kern, dq_global, dq_local);
    clFinish(clq);  // need sync before reading w_cl in conv
  }
  const uint64_t t1 = now_ns();

  // --- 2. Static resources ---
  static cl_mem s_xmem = nullptr;
  if (!s_xmem) {
    s_xmem = clCreateBuffer(clctx, CL_MEM_READ_WRITE, 6144, nullptr, &err);
  }

  static cl_mem s_bias = nullptr;
  static int s_bias_slices = 0;
  if (!s_bias || s_bias_slices < dst_slices) {
    if (s_bias) clReleaseMemObject(s_bias);
    cl_image_format fmt = {CL_RGBA, CL_HALF_FLOAT};
    cl_image_desc bd = {};
    bd.image_type = CL_MEM_OBJECT_IMAGE2D;
    bd.image_width = dst_slices;
    bd.image_height = 1;
    s_bias = clCreateImage(clctx, CL_MEM_READ_ONLY, &fmt, &bd, nullptr, &err);
    std::vector<uint16_t> zeros(dst_slices * 4, 0);
    size_t o[3] = {0, 0, 0}, r[3] = {(size_t)dst_slices, 1, 1};
    clEnqueueWriteImage(clq, s_bias, CL_TRUE, o, r, 0, 0, zeros.data(),
                        0, nullptr, nullptr);
    s_bias_slices = dst_slices;
  }

  // --- 3. Src image2d ---
  // Input SVM layout: input[m * K + k] (row-major [M][K])
  // Image layout: pixel(x=m, y=s).c = data[s * M * 4 + m * 4 + c]
  // Must reformat [M][K] → [src_slices][M][4]
  cl_image_format fmt_h = {CL_RGBA, CL_HALF_FLOAT};
  cl_image_desc sd = {};
  sd.image_type = CL_MEM_OBJECT_IMAGE2D;
  sd.image_width = M;
  sd.image_height = src_slices;
  cl_mem src_img = clCreateImage(clctx, CL_MEM_READ_ONLY, &fmt_h, &sd,
                                  nullptr, &err);
  if (err) throw std::runtime_error("delegate: src image failed");
  {
    // Reformat input to image layout
    std::vector<uint16_t> img_buf((size_t)src_slices * M * 4);
    for (int m = 0; m < (int)M; ++m) {
      for (int s = 0; s < src_slices; ++s) {
        for (int c = 0; c < 4; ++c) {
          int k = s * 4 + c;
          img_buf[((size_t)s * M + m) * 4 + c] =
            (k < (int)K) ? input[m * K + k] : 0;
        }
      }
    }
    size_t o[3] = {0, 0, 0}, r[3] = {(size_t)M, (size_t)src_slices, 1};
    clEnqueueWriteImage(clq, src_img, CL_TRUE, o, r, 0, 0, img_buf.data(),
                        0, nullptr, nullptr);
  }

  // --- 4. Dst image2d ---
  cl_image_desc dd = {};
  dd.image_type = CL_MEM_OBJECT_IMAGE2D;
  dd.image_width = M;
  dd.image_height = dst_slices;
  cl_mem dst_img = clCreateImage(clctx, CL_MEM_READ_WRITE, &fmt_h, &dd,
                                  nullptr, &err);
  if (err) throw std::runtime_error("delegate: dst image failed");

  const uint64_t t2 = now_ns();

  // --- 5. Dispatch delegate conv kernel ---
  ClContext::SharedPtrClKernel kern = blas_cc->registerClKernel(
    delegate_conv_wave_kernel, "main_function",
    "-qcom-accelerate-16-bit=true");
  if (!kern) throw std::runtime_error("delegate: conv kernel failed");

  struct { int x, y, z, w; } s0 = {1, dst_slices, (int)M, 32};
  struct { int x, y, z, w; } s1 = {src_slices, 0, 0, src_slices};
  struct { int x, y, z, w; } s2 = {1, 1, 0, 0};

  int arg = 0;
  kern->SetKernelArguments(arg++, &w_cl, sizeof(cl_mem));
  kern->SetKernelArguments(arg++, &s_xmem, sizeof(cl_mem));
  kern->SetKernelArguments(arg++, &s_bias, sizeof(cl_mem));
  kern->SetKernelArguments(arg++, &dst_img, sizeof(cl_mem));
  kern->SetKernelArguments(arg++, &src_img, sizeof(cl_mem));
  kern->SetKernelArguments(arg++, &s0, sizeof(s0));
  kern->SetKernelArguments(arg++, &s1, sizeof(s1));
  kern->SetKernelArguments(arg++, &s2, sizeof(s2));

  size_t gz = ((dst_slices + 7) / 8 + 3) / 4;
  const int global[3] = {(int)(gz * 128), (int)((M + 127) / 128), 4};
  const int local[3] = {128, 1, 4};

  blas_cc->command_queue_inst_.DispatchCommand(kern, global, local);
  clFinish(clq);
  const uint64_t t3 = now_ns();

  // --- 6. Read output and reformat from image [slice][M][4] to [M][N] ---
  {
    std::vector<uint16_t> img_out((size_t)dst_slices * M * 4);
    size_t o[3] = {0, 0, 0}, r[3] = {(size_t)M, (size_t)dst_slices, 1};
    clEnqueueReadImage(clq, dst_img, CL_TRUE, o, r, 0, 0, img_out.data(),
                       0, nullptr, nullptr);
    for (int m = 0; m < (int)M; ++m) {
      for (int s = 0; s < dst_slices; ++s) {
        for (int c = 0; c < 4; ++c) {
          int n = s * 4 + c;
          if (n < (int)N)
            output[m * N + n] = img_out[((size_t)s * M + m) * 4 + c];
        }
      }
    }
  }

  const uint64_t t4 = now_ns();

  t_dequant += (t1 - t0);
  t_in_reformat += (t2 - t1);
  t_conv += (t3 - t2);
  t_out_reformat += (t4 - t3);
  t_calls++;
  t_total_gflops_x1000 += (uint64_t)(2.0 * M * N * K / 1e6);  // GFLOPS * 1000

  clReleaseMemObject(w_cl);
  clReleaseMemObject(dst_img);
  clReleaseMemObject(src_img);
}
#endif
#if 0 // OLD CPU dequant version — replaced by GPU dequant above
// Weight layout for delegate kernel (reverse-engineered from program_002.cl):
//   For Z (output slice group 0..dst_slices/8-1), iteration i (0..src_slices/2-1):
//     32 half8 (256 halfs) at offset (Z * src_slices/2 + i) * 256
//     halfs[s*16 + c*4 + j] = weight for output_ch=Z*32+s*4+j, input_ch=i*8+(s<8?c:c+4)
//   where s=0..7 for src0, s=8..15 for src1
static void DequantAndRepack(const uint16_t *int4_weights,
                             const uint16_t *scales_fp16, unsigned int N,
                             unsigned int K, std::vector<uint16_t> &out) {
  const int src_slices = K / 4;
  const int dst_slices = N / 4;
  const int n_z = (dst_slices + 7) / 8;
  const int iters = src_slices / 2;

  out.resize(static_cast<size_t>(n_z) * iters * 256, 0);

  for (int z = 0; z < n_z; ++z) {
    int base_n = z * 32;
    for (int it = 0; it < iters; ++it) {
      size_t blk = (static_cast<size_t>(z) * iters + it) * 256;
      // Delegate layout: weights_cache[s].s(c*4+j) maps to
      //   s=0..7: output slice s, channels j=0..3, input from src0 channel c
      //   s=8..15: same but from src1
      for (int s = 0; s < 16; ++s) {
        int src_slice_idx = it * 2 + (s < 8 ? 0 : 1);
        int out_slice = s % 8;
        for (int c = 0; c < 4; ++c) {
          int in_ch = src_slice_idx * 4 + c;
          for (int j = 0; j < 4; ++j) {
            int out_ch = base_n + out_slice * 4 + j;
            if (out_ch >= (int)N || in_ch >= (int)K)
              continue;

            // Read int4 nibble
            int packed_row = in_ch / 4;
            int nibble_pos = in_ch % 4;
            uint16_t packed = int4_weights[packed_row * N + out_ch];
            int nibble = (packed >> (nibble_pos * 4)) & 0xF;
            int q = nibble - 8;

            // Dequant: half = q * scale
            // Read scale as fp16, convert to fp32 for multiply
            uint16_t sc_h = scales_fp16[out_ch];
            // Simple fp16 multiply via fp32 round-trip
            union {
              uint32_t u;
              float f;
            } su;
            uint32_t sc_sign = (sc_h & 0x8000) << 16;
            uint32_t sc_exp = (sc_h >> 10) & 0x1F;
            uint32_t sc_mant = sc_h & 0x3FF;
            if (sc_exp == 0)
              su.u = sc_sign;
            else
              su.u = sc_sign | ((sc_exp - 15 + 127) << 23) | (sc_mant << 13);

            float dq = (float)q * su.f;

            // Convert to fp16
            uint32_t fu;
            memcpy(&fu, &dq, 4);
            uint32_t sign = (fu >> 16) & 0x8000;
            int exp = ((fu >> 23) & 0xFF) - 127 + 15;
            uint32_t mant = (fu >> 13) & 0x3FF;
            uint16_t h;
            if (exp <= 0)
              h = sign;
            else if (exp >= 31)
              h = sign | 0x7C00;
            else
              h = sign | (exp << 10) | mant;

            out[blk + s * 16 + c * 4 + j] = h;
          }
        }
      }
    }
  }
}

void gemm_delegate_fp16_cl(uint16_t *input, uint16_t * /*input_transposed*/,
                           uint16_t *weights, uint16_t *scales,
                           uint16_t *output, unsigned int M, unsigned int N,
                           unsigned int K) {
  if (((N % 32) != 0) || ((K % 8) != 0))
    throw std::runtime_error(
      "gemm_delegate_fp16_cl requires N%32==0 and K%8==0");

  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  cl_context clctx = blas_cc->context_inst_.GetContext();
  cl_command_queue clq =
    blas_cc->command_queue_inst_.GetCommandQueue();

  const int src_slices = K / 4;
  const int dst_slices = N / 4;
  cl_int err;

  // --- 1. Weight conversion (cached) ---
  uintptr_t w_key = reinterpret_cast<uintptr_t>(weights);
  cl_mem w_cl = nullptr;
  auto it = g_delegate_weight_cache.find(w_key);
  if (it != g_delegate_weight_cache.end()) {
    w_cl = it->second;
  } else {
    std::vector<uint16_t> fp16_repacked;
    DequantAndRepack(weights, scales, N, K, fp16_repacked);
    size_t w_bytes = fp16_repacked.size() * sizeof(uint16_t);
    w_cl = clCreateBuffer(clctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                          w_bytes, fp16_repacked.data(), &err);
    if (err != CL_SUCCESS)
      throw std::runtime_error("delegate: weight buffer creation failed");
    g_delegate_weight_cache[w_key] = w_cl;
    fprintf(stderr, "[delegate_fp16] Cached weight for N=%u K=%u (%zu KB)\n",
            N, K, w_bytes / 1024);
  }

  // --- 2. xmem buffer (reuse static) ---
  static cl_mem s_xmem = nullptr;
  if (!s_xmem) {
    s_xmem = clCreateBuffer(clctx, CL_MEM_READ_WRITE, 6144, nullptr, &err);
    if (err) throw std::runtime_error("delegate: xmem buffer failed");
  }

  // --- 3. Bias (zeros, static) ---
  static cl_mem s_bias = nullptr;
  static int s_bias_slices = 0;
  if (!s_bias || s_bias_slices < dst_slices) {
    if (s_bias) clReleaseMemObject(s_bias);
    cl_image_format fmt = {CL_RGBA, CL_HALF_FLOAT};
    cl_image_desc bd = {};
    bd.image_type = CL_MEM_OBJECT_IMAGE2D;
    bd.image_width = dst_slices;
    bd.image_height = 1;
    s_bias = clCreateImage(clctx, CL_MEM_READ_ONLY, &fmt, &bd, nullptr, &err);
    if (err) throw std::runtime_error("delegate: bias image failed");
    std::vector<uint16_t> zeros(dst_slices * 4, 0);
    size_t o[3] = {0, 0, 0}, r[3] = {(size_t)dst_slices, 1, 1};
    clEnqueueWriteImage(clq, s_bias, CL_TRUE, o, r, 0, 0, zeros.data(),
                        0, nullptr, nullptr);
    s_bias_slices = dst_slices;
  }

  // --- 4. Src image2d (from input fp16 buffer) ---
  cl_image_format fmt_h = {CL_RGBA, CL_HALF_FLOAT};
  cl_image_desc sd = {};
  sd.image_type = CL_MEM_OBJECT_IMAGE2D;
  sd.image_width = M;
  sd.image_height = src_slices;
  // Create buffer from SVM ptr, then image
  size_t in_bytes = (size_t)M * K * sizeof(uint16_t);
  cl_mem in_buf = clCreateBuffer(clctx, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                                  in_bytes, input, &err);
  if (err) throw std::runtime_error("delegate: input buffer failed");

  // Write to image2d
  cl_mem src_img = clCreateImage(clctx, CL_MEM_READ_ONLY, &fmt_h, &sd,
                                  nullptr, &err);
  if (err) throw std::runtime_error("delegate: src image failed");
  {
    // Copy input into image2d row by row
    // Input layout: input[m * K + k], image pixel at (x=m, y=s), channel c=k%4
    size_t o[3] = {0, 0, 0}, r[3] = {(size_t)M, (size_t)src_slices, 1};
    clEnqueueWriteImage(clq, src_img, CL_TRUE, o, r, 0, 0, input,
                        0, nullptr, nullptr);
  }

  // --- 5. Dst image2d ---
  cl_image_desc dd = {};
  dd.image_type = CL_MEM_OBJECT_IMAGE2D;
  dd.image_width = M;
  dd.image_height = dst_slices;
  cl_mem dst_img = clCreateImage(clctx, CL_MEM_READ_WRITE, &fmt_h, &dd,
                                  nullptr, &err);
  if (err) throw std::runtime_error("delegate: dst image failed");

  // --- 6. Dispatch delegate kernel ---
  ClContext::SharedPtrClKernel kern = blas_cc->registerClKernel(
    delegate_conv_wave_kernel, "main_function",
    "-qcom-accelerate-16-bit=true");
  if (!kern) throw std::runtime_error("delegate: kernel registration failed");

  struct { int x, y, z, w; } s0 = {1, dst_slices, (int)M, 32};
  struct { int x, y, z, w; } s1 = {src_slices, 0, 0, src_slices};
  struct { int x, y, z, w; } s2 = {1, 1, 0, 0};

  int arg = 0;
  kern->SetKernelArguments(arg++, &w_cl, sizeof(cl_mem));
  kern->SetKernelArguments(arg++, &s_xmem, sizeof(cl_mem));
  kern->SetKernelArguments(arg++, &s_bias, sizeof(cl_mem));
  kern->SetKernelArguments(arg++, &dst_img, sizeof(cl_mem));
  kern->SetKernelArguments(arg++, &src_img, sizeof(cl_mem));
  kern->SetKernelArguments(arg++, &s0, sizeof(s0));
  kern->SetKernelArguments(arg++, &s1, sizeof(s1));
  kern->SetKernelArguments(arg++, &s2, sizeof(s2));

  size_t gz = ((dst_slices + 7) / 8 + 3) / 4;
  const int global[3] = {(int)(gz * 128), (int)((M + 127) / 128), 4};
  const int local[3] = {128, 1, 4};

  bool ok = blas_cc->command_queue_inst_.DispatchCommand(kern, global, local);
  if (!ok) throw std::runtime_error("delegate: kernel dispatch failed");

  // --- 7. Read output from image2d ---
  {
    size_t o[3] = {0, 0, 0}, r[3] = {(size_t)M, (size_t)dst_slices, 1};
    clEnqueueReadImage(clq, dst_img, CL_TRUE, o, r, 0, 0, output,
                       0, nullptr, nullptr);
  }

  // Cleanup per-call objects (cached objects stay)
  clReleaseMemObject(dst_img);
  clReleaseMemObject(src_img);
  clReleaseMemObject(in_buf);
}
#endif
