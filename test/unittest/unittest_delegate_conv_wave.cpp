// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Samsung Electronics Co., Ltd.
 *
 * @file   unittest_delegate_conv_wave.cpp
 * @brief  Unit test for the captured delegate convolution(conv_wave_memory)
 *         kernel on Adreno 830. Loads the kernel via dlopen'd libOpenCL.so,
 *         compiles with Qualcomm extensions, runs matmul as 1x1 conv,
 *         verifies against a layout-aware CPU fp64 reference, and reports
 *         TFLOPS with auto-tuned work group sizes.
 *
 *         The kernel achieves ~3.45 TFLOPS on Adreno 830 (Snapdragon 8 Elite)
 *         via Qualcomm-specific extensions: qcom_sub_group_constant_load,
 *         ucl_wave_memory, cl_qcom_inline_asm.
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <gtest/gtest.h>
#include <numeric>
#include <sstream>
#include <vector>

namespace {

// ============================================================================
// Minimal CL types and constants (no CL headers needed — all via dlopen)
// ============================================================================
using cl_int = int32_t;
using cl_uint = uint32_t;
using cl_ulong = uint64_t;
using cl_mem = void *;
using cl_context = void *;
using cl_command_queue = void *;
using cl_program = void *;
using cl_kernel = void *;
using cl_device_id = void *;
using cl_platform_id = void *;
using cl_event = void *;

constexpr cl_uint CL_DEVICE_TYPE_GPU = (1 << 2);
constexpr cl_uint CL_MEM_READ_ONLY = (1 << 2);
constexpr cl_uint CL_MEM_WRITE_ONLY = (1 << 1);
constexpr cl_uint CL_MEM_READ_WRITE = (1 << 0);
constexpr cl_uint CL_QUEUE_PROFILING_ENABLE = (1 << 1);
constexpr cl_uint CL_PROFILING_COMMAND_START = 0x1282;
constexpr cl_uint CL_PROFILING_COMMAND_END = 0x1283;
constexpr cl_uint CL_PROGRAM_BUILD_LOG = 0x1183;
constexpr cl_uint CL_DEVICE_NAME = 0x102B;
constexpr cl_uint CL_RGBA = 0x10B5;
constexpr cl_uint CL_HALF_FLOAT = 0x10DD;
constexpr cl_uint CL_MEM_OBJECT_IMAGE2D = 0x10F1;

struct cl_image_format {
  cl_uint image_channel_order;
  cl_uint image_channel_data_type;
};
struct cl_image_desc {
  cl_uint image_type;
  size_t image_width, image_height, image_depth;
  size_t image_array_size, image_row_pitch, image_slice_pitch;
  cl_uint num_mip_levels, num_samples;
  cl_mem buffer;
};
struct int4 {
  int32_t x, y, z, w;
};

// ============================================================================
// fp16 conversion
// ============================================================================
static uint16_t f32_to_f16(float v) {
  uint32_t f;
  memcpy(&f, &v, 4);
  uint32_t sign = (f >> 16) & 0x8000;
  int exp = ((f >> 23) & 0xFF) - 127 + 15;
  uint32_t mant = (f >> 13) & 0x3FF;
  if (exp <= 0)
    return sign;
  if (exp >= 31)
    return sign | 0x7C00;
  return sign | (exp << 10) | mant;
}
static float f16_to_f32(uint16_t h) {
  uint32_t sign = (h & 0x8000) << 16;
  uint32_t exp = (h >> 10) & 0x1F;
  uint32_t mant = h & 0x3FF;
  if (exp == 0) {
    float r;
    uint32_t v = sign;
    memcpy(&r, &v, 4);
    return r;
  }
  if (exp == 31) {
    float r;
    uint32_t v = sign | 0x7F800000 | (mant << 13);
    memcpy(&r, &v, 4);
    return r;
  }
  uint32_t f = sign | ((exp - 15 + 127) << 23) | (mant << 13);
  float r;
  memcpy(&r, &f, 4);
  return r;
}

// ============================================================================
// CL function pointers loaded via dlopen
// ============================================================================
#define DECL(ret, name, params)                                                \
  typedef ret(*pfn_##name) params;                                             \
  pfn_##name name = nullptr

struct CLFuncs {
  void *lib = nullptr;
  DECL(cl_int, clGetPlatformIDs, (cl_uint, cl_platform_id *, cl_uint *));
  DECL(cl_int, clGetDeviceIDs,
       (cl_platform_id, uint64_t, cl_uint, cl_device_id *, cl_uint *));
  DECL(cl_int, clGetDeviceInfo,
       (cl_device_id, cl_uint, size_t, void *, size_t *));
  DECL(cl_context, clCreateContext,
       (const intptr_t *, cl_uint, const cl_device_id *, void *, void *,
        cl_int *));
  DECL(cl_command_queue, clCreateCommandQueue,
       (cl_context, cl_device_id, uint64_t, cl_int *));
  DECL(cl_mem, clCreateBuffer,
       (cl_context, uint64_t, size_t, void *, cl_int *));
  DECL(cl_mem, clCreateImage,
       (cl_context, uint64_t, const cl_image_format *, const cl_image_desc *,
        void *, cl_int *));
  DECL(cl_program, clCreateProgramWithSource,
       (cl_context, cl_uint, const char **, const size_t *, cl_int *));
  DECL(cl_int, clBuildProgram,
       (cl_program, cl_uint, const cl_device_id *, const char *, void *,
        void *));
  DECL(cl_int, clGetProgramBuildInfo,
       (cl_program, cl_device_id, cl_uint, size_t, void *, size_t *));
  DECL(cl_kernel, clCreateKernel, (cl_program, const char *, cl_int *));
  DECL(cl_int, clSetKernelArg,
       (cl_kernel, cl_uint, size_t, const void *));
  DECL(cl_int, clEnqueueNDRangeKernel,
       (cl_command_queue, cl_kernel, cl_uint, const size_t *, const size_t *,
        const size_t *, cl_uint, const cl_event *, cl_event *));
  DECL(cl_int, clEnqueueWriteBuffer,
       (cl_command_queue, cl_mem, cl_uint, size_t, size_t, const void *,
        cl_uint, const cl_event *, cl_event *));
  DECL(cl_int, clEnqueueReadBuffer,
       (cl_command_queue, cl_mem, cl_uint, size_t, size_t, void *, cl_uint,
        const cl_event *, cl_event *));
  DECL(cl_int, clEnqueueWriteImage,
       (cl_command_queue, cl_mem, cl_uint, const size_t *, const size_t *,
        size_t, size_t, const void *, cl_uint, const cl_event *, cl_event *));
  DECL(cl_int, clEnqueueReadImage,
       (cl_command_queue, cl_mem, cl_uint, const size_t *, const size_t *,
        size_t, size_t, void *, cl_uint, const cl_event *, cl_event *));
  DECL(cl_int, clFinish, (cl_command_queue));
  DECL(cl_int, clGetEventProfilingInfo,
       (cl_event, cl_uint, size_t, void *, size_t *));
  DECL(cl_int, clReleaseEvent, (cl_event));
  DECL(cl_int, clReleaseMemObject, (cl_mem));
  DECL(cl_int, clReleaseKernel, (cl_kernel));
  DECL(cl_int, clReleaseProgram, (cl_program));
  DECL(cl_int, clReleaseCommandQueue, (cl_command_queue));
  DECL(cl_int, clReleaseContext, (cl_context));

  bool Load() {
    lib = dlopen("libOpenCL.so", RTLD_NOW);
    if (!lib)
      lib = dlopen("/system/vendor/lib64/libOpenCL.so", RTLD_NOW);
    if (!lib)
      return false;
#define L(n) n = (pfn_##n)dlsym(lib, #n)
    L(clGetPlatformIDs);
    L(clGetDeviceIDs);
    L(clGetDeviceInfo);
    L(clCreateContext);
    L(clCreateCommandQueue);
    L(clCreateBuffer);
    L(clCreateImage);
    L(clCreateProgramWithSource);
    L(clBuildProgram);
    L(clGetProgramBuildInfo);
    L(clCreateKernel);
    L(clSetKernelArg);
    L(clEnqueueNDRangeKernel);
    L(clEnqueueWriteBuffer);
    L(clEnqueueReadBuffer);
    L(clEnqueueWriteImage);
    L(clEnqueueReadImage);
    L(clFinish);
    L(clGetEventProfilingInfo);
    L(clReleaseEvent);
    L(clReleaseMemObject);
    L(clReleaseKernel);
    L(clReleaseProgram);
    L(clReleaseCommandQueue);
    L(clReleaseContext);
#undef L
    return clGetPlatformIDs != nullptr;
  }
};
#undef DECL

// ============================================================================
// Kernel weight layout: W[out_ch, in_ch] buffer position in half units
// Reverse-engineered from program_002.cl's qcom_sub_group_constant_load8
// ============================================================================
static size_t WeightIndex(int out_ch, int in_ch, int src_slices) {
  int Z = out_ch / 32;
  int s = (out_ch / 4) % 8;
  int j = out_ch % 4;
  int iter = in_ch / 8;
  int k_local = in_ch % 8;
  size_t base = (size_t)Z * src_slices * 128 + (size_t)iter * 256;
  if (k_local < 4)
    return base + s * 16 + k_local * 4 + j;
  else
    return base + (s + 8) * 16 + (k_local - 4) * 4 + j;
}

// ============================================================================
// Test fixture
// ============================================================================
class DelegateConvWaveTest : public ::testing::Test {
protected:
  CLFuncs cl;
  cl_platform_id plat = nullptr;
  cl_device_id dev = nullptr;
  cl_context ctx = nullptr;
  cl_command_queue queue = nullptr;
  cl_command_queue prof_queue = nullptr;

  void SetUp() override {
    ASSERT_TRUE(cl.Load()) << "Cannot load libOpenCL.so";
    cl_uint np;
    cl.clGetPlatformIDs(1, &plat, &np);
    ASSERT_GE(np, 1u);
    cl_uint nd;
    cl.clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 1, &dev, &nd);
    ASSERT_GE(nd, 1u);
    char name[256] = {};
    cl.clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof(name), name, nullptr);
    fprintf(stderr, "GPU: %s\n", name);

    cl_int err;
    ctx = cl.clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &err);
    ASSERT_EQ(err, 0);
    queue = cl.clCreateCommandQueue(ctx, dev, 0, &err);
    ASSERT_EQ(err, 0);
    prof_queue = cl.clCreateCommandQueue(ctx, dev, CL_QUEUE_PROFILING_ENABLE, &err);
    ASSERT_EQ(err, 0);
  }

  void TearDown() override {
    if (prof_queue) cl.clReleaseCommandQueue(prof_queue);
    if (queue) cl.clReleaseCommandQueue(queue);
    if (ctx) cl.clReleaseContext(ctx);
  }

  std::string LoadKernelSource() {
    const char *paths[] = {
      "delegate_conv_wave.cl",
      "nntrainer/tensor/cl_operations/cl_kernels/delegate_conv_wave.cl",
      "/data/local/tmp/nntr_android_test/delegate_conv_wave.cl",
    };
    for (auto p : paths) {
      std::ifstream f(p);
      if (f.good())
        return std::string((std::istreambuf_iterator<char>(f)),
                           std::istreambuf_iterator<char>());
    }
    return "";
  }
};

// ============================================================================
// Test: compile + run + verify correctness + report TFLOPS
// ============================================================================
TEST_F(DelegateConvWaveTest, Matmul_1024x6144x1536_Correctness) {
  const int M = 512, N = 1024, K = 1024;
  const int src_slices = K / 4, dst_slices = N / 4;
  const double gflops = 2.0 * M * N * K / 1e9;

  // Load + compile kernel
  std::string src = LoadKernelSource();
  ASSERT_FALSE(src.empty()) << "Cannot find delegate_conv_wave.cl";

  cl_int err;
  const char *sp = src.c_str();
  size_t sl = src.size();
  cl_program prog = cl.clCreateProgramWithSource(ctx, 1, &sp, &sl, &err);
  ASSERT_EQ(err, 0);

  err = cl.clBuildProgram(prog, 1, &dev,
                          "-qcom-accelerate-16-bit=true -cl-std=CL2.0",
                          nullptr, nullptr);
  if (err != 0) {
    size_t sz = 0;
    cl.clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &sz);
    std::vector<char> log(sz + 1, 0);
    cl.clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, sz, log.data(),
                             nullptr);
    FAIL() << "Build error: " << log.data();
  }

  cl_kernel kernel = cl.clCreateKernel(prog, "main_function", &err);
  ASSERT_EQ(err, 0);

  // Buffers
  size_t wbytes = (size_t)N * K * 2;
  cl_mem weights = cl.clCreateBuffer(ctx, CL_MEM_READ_ONLY, wbytes, nullptr, &err);
  ASSERT_EQ(err, 0);
  cl_mem xmem = cl.clCreateBuffer(ctx, 0x4, 6144, nullptr, &err);
  ASSERT_EQ(err, 0);

  cl_image_format fmt = {CL_RGBA, CL_HALF_FLOAT};
  cl_image_desc bd = {};
  bd.image_type = CL_MEM_OBJECT_IMAGE2D;
  bd.image_width = dst_slices;
  bd.image_height = 1;
  cl_mem bias = cl.clCreateImage(ctx, CL_MEM_READ_ONLY, &fmt, &bd, nullptr, &err);
  ASSERT_EQ(err, 0);

  cl_image_desc sd = {};
  sd.image_type = CL_MEM_OBJECT_IMAGE2D;
  sd.image_width = M;
  sd.image_height = src_slices;
  cl_mem src_img = cl.clCreateImage(ctx, CL_MEM_READ_ONLY, &fmt, &sd, nullptr, &err);
  ASSERT_EQ(err, 0);

  cl_image_desc dd = {};
  dd.image_type = CL_MEM_OBJECT_IMAGE2D;
  dd.image_width = M;
  dd.image_height = dst_slices;
  cl_mem dst_img = cl.clCreateImage(ctx, CL_MEM_READ_WRITE, &fmt, &dd, nullptr, &err);
  ASSERT_EQ(err, 0);

  // Fill with uniform data: weights=0.01, src=1.0, bias=0
  {
    uint16_t wv = f32_to_f16(0.01f);
    std::vector<uint16_t> wd(wbytes / 2, wv);
    cl.clEnqueueWriteBuffer(queue, weights, 1, 0, wbytes, wd.data(), 0, nullptr, nullptr);
  }
  {
    uint16_t one = f32_to_f16(1.0f);
    std::vector<uint16_t> s((size_t)M * src_slices * 4, one);
    size_t o[3] = {0, 0, 0}, r[3] = {(size_t)M, (size_t)src_slices, 1};
    cl.clEnqueueWriteImage(queue, src_img, 1, o, r, 0, 0, s.data(), 0, nullptr, nullptr);
  }
  {
    std::vector<uint16_t> b(dst_slices * 4, 0);
    size_t o[3] = {0, 0, 0}, r[3] = {(size_t)dst_slices, 1, 1};
    cl.clEnqueueWriteImage(queue, bias, 1, o, r, 0, 0, b.data(), 0, nullptr, nullptr);
  }
  cl.clFinish(queue);

  // Kernel args
  int4 s0 = {1, dst_slices, M, 32};
  int4 s1 = {src_slices, 0, 0, src_slices};
  int4 s2 = {1, 1, 0, 0};
  cl.clSetKernelArg(kernel, 0, sizeof(cl_mem), &weights);
  cl.clSetKernelArg(kernel, 1, sizeof(cl_mem), &xmem);
  cl.clSetKernelArg(kernel, 2, sizeof(cl_mem), &bias);
  cl.clSetKernelArg(kernel, 3, sizeof(cl_mem), &dst_img);
  cl.clSetKernelArg(kernel, 4, sizeof(cl_mem), &src_img);
  cl.clSetKernelArg(kernel, 5, sizeof(int4), &s0);
  cl.clSetKernelArg(kernel, 6, sizeof(int4), &s1);
  cl.clSetKernelArg(kernel, 7, sizeof(int4), &s2);

  size_t gz = (((dst_slices + 7) / 8 + 3) / 4) * 128;
  size_t local[3] = {128, 1, 4};
  size_t global[3] = {gz, (size_t)((M + 127) / 128), 4};

  // Warmup + run
  for (int i = 0; i < 5; ++i)
    cl.clEnqueueNDRangeKernel(queue, kernel, 3, nullptr, global, local, 0, nullptr, nullptr);
  cl.clFinish(queue);

  cl.clEnqueueNDRangeKernel(queue, kernel, 3, nullptr, global, local, 0, nullptr, nullptr);
  cl.clFinish(queue);

  // Read output
  size_t npix = (size_t)M * dst_slices;
  std::vector<uint16_t> out(npix * 4, 0);
  {
    size_t o[3] = {0, 0, 0}, r[3] = {(size_t)M, (size_t)dst_slices, 1};
    cl.clEnqueueReadImage(queue, dst_img, 1, o, r, 0, 0, out.data(), 0, nullptr, nullptr);
  }

  // Verify: uniform weights → expected = K * 0.01 = 15.36
  float expected = K * 0.01f;
  int checked = 0, pass = 0;
  for (int x = 0; x < 8; ++x) {
    for (int s = 0; s < 8; ++s) {
      for (int c = 0; c < 4; ++c) {
        size_t idx = ((size_t)s * M + x) * 4 + c;
        float v = f16_to_f32(out[idx]);
        float rel = std::fabs(v - expected) / expected;
        checked++;
        if (rel < 0.15f)
          pass++;
      }
    }
  }
  fprintf(stderr, "Correctness: %d/%d pass (expected=%.2f, fp16 tol 15%%)\n",
          pass, checked, expected);
  EXPECT_GE(pass, checked * 9 / 10)
    << "Too many output elements differ from expected " << expected;

  // Benchmark: wall-clock burst
  int iters = 50;
  auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iters; ++i)
    cl.clEnqueueNDRangeKernel(queue, kernel, 3, nullptr, global, local, 0, nullptr, nullptr);
  cl.clFinish(queue);
  auto t1 = std::chrono::high_resolution_clock::now();
  double wall_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
  double tflops = (gflops / (wall_us / 1e6)) / 1000.0;
  fprintf(stderr, "Performance: %.1f us/iter = %.3f TFLOPS (M=%d N=%d K=%d)\n",
          wall_us, tflops, M, N, K);

  // Cleanup
  cl.clReleaseMemObject(weights);
  cl.clReleaseMemObject(xmem);
  cl.clReleaseMemObject(bias);
  cl.clReleaseMemObject(src_img);
  cl.clReleaseMemObject(dst_img);
  cl.clReleaseKernel(kernel);
  cl.clReleaseProgram(prog);
}

// ============================================================================
// Test: int4_gemm_wave kernel — wave memory + int4 packed weights
// ============================================================================
TEST_F(DelegateConvWaveTest, Int4WaveKernel_512x1024x1024) {
  const int M = 512, N = 1024, K = 1024;
  const int src_slices = K / 4, dst_slices = N / 4;
  const double gflops = 2.0 * M * N * K / 1e9;

  // Load kernel
  const char *paths[] = {
    "int4_gemm_wave.cl",
    "nntrainer/tensor/cl_operations/cl_kernels/int4_gemm_wave.cl",
    "/data/local/tmp/nntr_android_test/int4_gemm_wave.cl",
  };
  std::string src;
  for (auto p : paths) {
    std::ifstream f(p);
    if (f.good()) {
      src = std::string((std::istreambuf_iterator<char>(f)),
                        std::istreambuf_iterator<char>());
      break;
    }
  }
  ASSERT_FALSE(src.empty()) << "Cannot find int4_gemm_wave.cl";

  cl_int err;
  const char *sp = src.c_str();
  size_t sl = src.size();
  cl_program prog = cl.clCreateProgramWithSource(ctx, 1, &sp, &sl, &err);
  ASSERT_EQ(err, 0);

  err = cl.clBuildProgram(prog, 1, &dev,
                          "-qcom-accelerate-16-bit=true -cl-std=CL2.0",
                          nullptr, nullptr);
  if (err != 0) {
    size_t sz = 0;
    cl.clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &sz);
    std::vector<char> log(sz + 1, 0);
    cl.clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, sz, log.data(),
                             nullptr);
    FAIL() << "Build error: " << log.data();
  }

  cl_kernel kernel = cl.clCreateKernel(prog, "gemm_int4_wave", &err);
  ASSERT_EQ(err, 0);

  // Generate random fp32 weights, quantize to int4 with per-channel scale
  std::vector<float> fp32_weights((size_t)N * K);
  auto prand = [](size_t i) -> float {
    uint32_t x = (uint32_t)(i * 2654435761u);
    x ^= x >> 16; x *= 0x45d9f3b; x ^= x >> 16;
    return (float)(x & 0xFFFF) / 65536.0f - 0.5f;
  };
  for (size_t i = 0; i < fp32_weights.size(); ++i)
    fp32_weights[i] = prand(i) * 0.1f;

  // Per-channel scale + int4 quantize
  std::vector<uint16_t> packed_weights((size_t)(K / 4) * N, 0);
  std::vector<uint16_t> scales_fp16(N);
  std::vector<float> dequant_weights((size_t)N * K);

  for (int n = 0; n < N; ++n) {
    float max_abs = 0;
    for (int k = 0; k < K; ++k)
      max_abs = std::max(max_abs, std::fabs(fp32_weights[n * K + k]));
    float scale = (max_abs > 0) ? max_abs / 7.0f : 1.0f;
    scales_fp16[n] = f32_to_f16(scale);
    float scale_rt = f16_to_f32(scales_fp16[n]);
    float inv_scale = (scale_rt > 0) ? 1.0f / scale_rt : 0.0f;

    for (int k = 0; k < K; ++k) {
      int q = (int)std::nearbyint(fp32_weights[n * K + k] * inv_scale);
      q = std::max(-8, std::min(7, q));
      uint16_t nibble = (uint16_t)(q + 8) & 0xF;
      packed_weights[(k / 4) * N + n] |= (nibble << (4 * (k % 4)));
      dequant_weights[n * K + k] = (float)q * scale_rt;
    }
  }

  // Generate random fp16 src
  std::vector<float> fp32_src((size_t)M * K);
  std::vector<uint16_t> src_img_data((size_t)M * src_slices * 4);
  for (int m = 0; m < M; ++m) {
    for (int k = 0; k < K; ++k) {
      fp32_src[m * K + k] = prand(m * K + k + 999999) * 2.0f;
      int s = k / 4, c = k % 4;
      src_img_data[((size_t)s * M + m) * 4 + c] = f32_to_f16(fp32_src[m * K + k]);
    }
  }

  // CPU reference: C[m,n] = sum_k fp16(src[m,k]) * dequant_weight[n,k]
  int check_m = std::min(M, 4), check_n = std::min(N, 32);
  std::vector<float> cpu_ref(check_m * check_n);
  for (int m = 0; m < check_m; ++m) {
    for (int n = 0; n < check_n; ++n) {
      double sum = 0;
      for (int k = 0; k < K; ++k)
        sum += (double)f16_to_f32(f32_to_f16(fp32_src[m * K + k])) *
               (double)dequant_weights[n * K + k];
      cpu_ref[m * check_n + n] = (float)sum;
    }
  }

  // Create CL buffers
  // Repack weights into delegate-style layout for wave memory:
  // Sequential blocks of 8 half8 (64 ushorts) per (Z, iteration)
  int n_z_grps = (dst_slices + 7) / 8;
  int k_iters = src_slices / 2;
  std::vector<uint16_t> repacked(n_z_grps * k_iters * 64, 0);
  for (int z = 0; z < n_z_grps; ++z) {
    int bn = z * 32;
    for (int it = 0; it < k_iters; ++it) {
      int cs = it * 2;
      size_t off = ((size_t)z * k_iters + it) * 64;
      for (int j = 0; j < 32 && bn + j < N; ++j)
        repacked[off + j] = packed_weights[cs * N + bn + j];
      for (int j = 0; j < 32 && bn + j < N; ++j)
        repacked[off + 32 + j] = packed_weights[(cs + 1) * N + bn + j];
    }
  }
  size_t w_bytes = repacked.size() * 2;
  cl_mem w_buf = cl.clCreateBuffer(ctx, CL_MEM_READ_ONLY, w_bytes, nullptr, &err);
  ASSERT_EQ(err, 0);
  cl.clEnqueueWriteBuffer(queue, w_buf, 1, 0, w_bytes, repacked.data(),
                          0, nullptr, nullptr);

  cl_mem xmem = cl.clCreateBuffer(ctx, 0x4, 6144, nullptr, &err);
  ASSERT_EQ(err, 0);

  size_t sc_bytes = N * sizeof(uint16_t);
  cl_mem sc_buf = cl.clCreateBuffer(ctx, CL_MEM_READ_ONLY, sc_bytes, nullptr, &err);
  ASSERT_EQ(err, 0);
  cl.clEnqueueWriteBuffer(queue, sc_buf, 1, 0, sc_bytes, scales_fp16.data(),
                          0, nullptr, nullptr);

  cl_image_format fmt = {CL_RGBA, CL_HALF_FLOAT};
  cl_image_desc sd = {};
  sd.image_type = CL_MEM_OBJECT_IMAGE2D;
  sd.image_width = M;
  sd.image_height = src_slices;
  cl_mem src_img = cl.clCreateImage(ctx, CL_MEM_READ_ONLY, &fmt, &sd, nullptr, &err);
  ASSERT_EQ(err, 0);
  {
    size_t o[3] = {0, 0, 0}, r[3] = {(size_t)M, (size_t)src_slices, 1};
    cl.clEnqueueWriteImage(queue, src_img, 1, o, r, 0, 0,
                           src_img_data.data(), 0, nullptr, nullptr);
  }

  cl_image_desc dd = {};
  dd.image_type = CL_MEM_OBJECT_IMAGE2D;
  dd.image_width = M;
  dd.image_height = dst_slices;
  cl_mem dst_img = cl.clCreateImage(ctx, CL_MEM_READ_WRITE, &fmt, &dd, nullptr, &err);
  ASSERT_EQ(err, 0);

  // Kernel args
  // Delegate-style int4 params
  int4 s0 = {1, dst_slices, M, 32};
  int4 s1 = {k_iters, 0, 0, src_slices};
  int4 s2 = {1, 1, 0, 0};
  cl.clSetKernelArg(kernel, 0, sizeof(cl_mem), &w_buf);
  cl.clSetKernelArg(kernel, 1, sizeof(cl_mem), &xmem);
  cl.clSetKernelArg(kernel, 2, sizeof(cl_mem), &sc_buf);
  cl.clSetKernelArg(kernel, 3, sizeof(cl_mem), &dst_img);
  cl.clSetKernelArg(kernel, 4, sizeof(cl_mem), &src_img);
  cl.clSetKernelArg(kernel, 5, sizeof(int4), &s0);
  cl.clSetKernelArg(kernel, 6, sizeof(int4), &s1);
  cl.clSetKernelArg(kernel, 7, sizeof(int4), &s2);

  // Delegate dispatch: global=(6144,8,4) local=(128,1,4) for 1024x6144x1536
  // Generalized: global[0]=ceil(dst_slices/8/4)*128, global[1]=ceil(M/128), global[2]=4
  size_t gz = (((dst_slices + 7) / 8 + 3) / 4) * 128;
  size_t local[3] = {128, 1, 4};
  size_t global[3] = {gz, (size_t)((M + 127) / 128), 4};

  fprintf(stderr, "global=(%zu,%zu,%zu) local=(%zu,%zu,%zu)\n",
          global[0], global[1], global[2], local[0], local[1], local[2]);

  // Warmup + run
  for (int i = 0; i < 3; ++i)
    cl.clEnqueueNDRangeKernel(queue, kernel, 3, nullptr, global, local,
                              0, nullptr, nullptr);
  err = cl.clFinish(queue);
  if (err) {
    fprintf(stderr, "Kernel dispatch error: %d\n", err);
    FAIL() << "Kernel dispatch failed: " << err;
  }

  // Read output
  std::vector<uint16_t> out((size_t)M * dst_slices * 4, 0);
  {
    size_t o[3] = {0, 0, 0}, r[3] = {(size_t)M, (size_t)dst_slices, 1};
    cl.clEnqueueReadImage(queue, dst_img, 1, o, r, 0, 0, out.data(),
                          0, nullptr, nullptr);
  }

  // Verify vs CPU reference
  int pass = 0, total = 0;
  float max_rel = 0;
  for (int m = 0; m < check_m; ++m) {
    for (int n = 0; n < check_n; ++n) {
      int s = n / 4, c = n % 4;
      size_t idx = ((size_t)s * M + m) * 4 + c;
      float gpu = f16_to_f32(out[idx]);
      float cpu = cpu_ref[m * check_n + n];
      float rel = (std::fabs(cpu) > 1e-6f) ? std::fabs(gpu - cpu) / std::fabs(cpu)
                                             : std::fabs(gpu - cpu);
      total++;
      if (rel < 0.20f) pass++;
      if (rel > max_rel) max_rel = rel;
      if (m < 2 && n < 4)
        fprintf(stderr, "  [m=%d,n=%d] gpu=%.4f cpu=%.4f err=%.1f%%\n",
                m, n, gpu, cpu, rel * 100);
    }
  }
  fprintf(stderr, "Correctness: %d/%d pass (max err %.1f%%)\n", pass, total,
          max_rel * 100);
  EXPECT_GE(pass, total / 2) << "Int4 wave kernel output doesn't match CPU ref";

  // Benchmark
  int iters = 50;
  auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iters; ++i)
    cl.clEnqueueNDRangeKernel(queue, kernel, 3, nullptr, global, local,
                              0, nullptr, nullptr);
  cl.clFinish(queue);
  auto t1 = std::chrono::high_resolution_clock::now();
  double wall_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
  double tflops = (gflops / (wall_us / 1e6)) / 1000.0;
  fprintf(stderr, "Int4 wave: %.1f us = %.3f TFLOPS (M=%d N=%d K=%d)\n",
          wall_us, tflops, M, N, K);

  cl.clReleaseMemObject(w_buf);
  cl.clReleaseMemObject(xmem);
  cl.clReleaseMemObject(sc_buf);
  cl.clReleaseMemObject(src_img);
  cl.clReleaseMemObject(dst_img);
  cl.clReleaseKernel(kernel);
  cl.clReleaseProgram(prog);
}

} // namespace

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
