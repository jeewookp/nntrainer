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

  std::string LoadFusedKernelSource() {
    const char *paths[] = {
      "fused_conv_int4_fp16.cl",
      "nntrainer/tensor/cl_operations/cl_kernels/fused_conv_int4_fp16.cl",
      "/data/local/tmp/nntr_android_test/fused_conv_int4_fp16.cl",
    };
    for (auto p : paths) {
      std::ifstream f(p);
      if (f.good())
        return std::string((std::istreambuf_iterator<char>(f)),
                           std::istreambuf_iterator<char>());
    }
    return "";
  }

  std::string LoadDequantKernelSource() {
    const char *paths[] = {
      "dequant_int4_to_fp16.cl",
      "nntrainer/tensor/cl_operations/cl_kernels/dequant_int4_to_fp16.cl",
      "/data/local/tmp/nntr_android_test/dequant_int4_to_fp16.cl",
    };
    for (auto p : paths) {
      std::ifstream f(p);
      if (f.good())
        return std::string((std::istreambuf_iterator<char>(f)),
                           std::istreambuf_iterator<char>());
    }
    return "";
  }

  std::string LoadProbeLoadKernelSource() {
    const char *paths[] = {
      "probe_delegate_load.cl",
      "nntrainer/tensor/cl_operations/cl_kernels/probe_delegate_load.cl",
      "/data/local/tmp/nntr_android_test/probe_delegate_load.cl",
    };
    for (auto p : paths) {
      std::ifstream f(p);
      if (f.good())
        return std::string((std::istreambuf_iterator<char>(f)),
                           std::istreambuf_iterator<char>());
    }
    return "";
  }

  std::string LoadDelegateInt4DenseKernelSource() {
    const char *paths[] = {
      "delegate_conv_int4_dense.cl",
      "nntrainer/tensor/cl_operations/cl_kernels/delegate_conv_int4_dense.cl",
      "/data/local/tmp/nntr_android_test/delegate_conv_int4_dense.cl",
    };
    for (auto p : paths) {
      std::ifstream f(p);
      if (f.good())
        return std::string((std::istreambuf_iterator<char>(f)),
                           std::istreambuf_iterator<char>());
    }
    return "";
  }

  std::string LoadProbeInt4R0KernelSource() {
    const char *paths[] = {
      "probe_delegate_int4_r0.cl",
      "nntrainer/tensor/cl_operations/cl_kernels/probe_delegate_int4_r0.cl",
      "/data/local/tmp/nntr_android_test/probe_delegate_int4_r0.cl",
    };
    for (auto p : paths) {
      std::ifstream f(p);
      if (f.good())
        return std::string((std::istreambuf_iterator<char>(f)),
                           std::istreambuf_iterator<char>());
    }
    return "";
  }

  std::string LoadProbeInt4R0R3KernelSource() {
    const char *paths[] = {
      "probe_delegate_int4_r0_r3.cl",
      "nntrainer/tensor/cl_operations/cl_kernels/probe_delegate_int4_r0_r3.cl",
      "/data/local/tmp/nntr_android_test/probe_delegate_int4_r0_r3.cl",
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
// Benchmark: model shapes — delegate fp16 vs int4 (existing kernel)
// Shapes from Qwen3-4B prefill (M=437)
// ============================================================================
struct ShapeConfig {
  int M, N, K;
  int count; // how many times per prefill
};

static const ShapeConfig kModelShapes[] = {
  {437, 4096, 2560, 36},  // q_proj/k_proj/v_proj/o_proj
  {437, 1024, 2560, 36},  // smaller projections
  {437, 1024, 2560, 36},
  {437, 2560, 4096, 36},
  {437, 9728, 2560, 36},  // gate/up proj
  {437, 9728, 2560, 36},
  {437, 2560, 9728, 36},  // down proj
};



// ============================================================================
// Load-only microbench for the delegate wave-memory path.
// Runs first so probe data is recorded even if a later, heavier test
// trips the Adreno watchdog. Strips compute to a trivial OR-reduction,
// sweeps two candidate int4 layouts:
//   count=32 — fp16 delegate's native stride (512 B/iter, 4x padded for int4)
//   count=8  — dense int4 stride (128 B/iter, no padding)
// Reports us/iter and cooperative "bytes requested" bandwidth so we can
// tell if (a) count=8 still hits peak (→ dense int4 layout is viable), or
// (b) the wave loader only reaches peak at count=32 (→ padded layout, int4
// wins nothing on load side, only on model size).
// ============================================================================
TEST_F(DelegateConvWaveTest, DelegateLoadProbe) {
  fprintf(stderr,
          "\n=== Delegate wave-load microbench (count=32 vs count=8) ===\n");

  std::string src_probe = LoadProbeLoadKernelSource();
  if (src_probe.empty()) {
    fprintf(stderr, "SKIP: probe kernel source not found\n");
    return;
  }

  auto build = [&](const std::string &src, const char *flags,
                   const char *kname) -> cl_kernel {
    cl_int e;
    const char *sp = src.c_str(); size_t sl = src.size();
    cl_program prog = cl.clCreateProgramWithSource(ctx, 1, &sp, &sl, &e);
    e = cl.clBuildProgram(prog, 1, &dev, flags, nullptr, nullptr);
    if (e) {
      size_t sz = 0;
      cl.clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &sz);
      std::vector<char> log(sz + 1, 0);
      cl.clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, sz,
                               log.data(), nullptr);
      fprintf(stderr, "BUILD FAIL (%s): %s\n", kname, log.data());
      return nullptr;
    }
    return cl.clCreateKernel(prog, kname, &e);
  };

  cl_kernel k32 = build(src_probe,
    "-qcom-accelerate-16-bit=true -cl-std=CL2.0", "probe_load_count32");
  cl_kernel k8  = build(src_probe,
    "-qcom-accelerate-16-bit=true -cl-std=CL2.0", "probe_load_count8");
  if (!k32 || !k8) return;

  for (const auto &s : kModelShapes) {
    const int M = s.M, N = s.N, K = s.K;
    const int src_slices = K / 4, dst_slices = N / 4;
    const int n_z = (dst_slices + 7) / 8;
    const int iters = src_slices / 2;  // = K/8

    fprintf(stderr, "\n  M=%d N=%d K=%d  (iters/thread=%d)\n", M, N, K, iters);

    const size_t weights_halves = (size_t)n_z * iters * 256 + 256;
    const size_t weights_bytes = weights_halves * 2;

    cl_int e;
    cl_mem weights_dev = cl.clCreateBuffer(ctx, CL_MEM_READ_ONLY,
      weights_bytes, nullptr, &e);
    cl_mem xm = cl.clCreateBuffer(ctx, 0x4, 6144, nullptr, &e);

    cl_image_format fmt = {CL_RGBA, CL_HALF_FLOAT};
    cl_image_desc dd = {}; dd.image_type = CL_MEM_OBJECT_IMAGE2D;
    dd.image_width = M; dd.image_height = dst_slices;
    cl_mem dst = cl.clCreateImage(ctx, CL_MEM_READ_WRITE, &fmt, &dd, 0, &e);

    int4 s0 = {1, dst_slices, M, 32};
    int4 s1 = {src_slices, 0, 0, src_slices};
    int4 s2 = {1, 1, 0, 0};
    size_t gz = (((dst_slices + 7) / 8 + 3) / 4) * 128;
    size_t gl[3] = {gz, (size_t)((M + 127) / 128), 4};
    size_t ll[3] = {128, 1, 4};

    const size_t n_wg_z = gl[0] / ll[0];
    const size_t n_wg_x = gl[1] / ll[1];
    const size_t n_wg_y = gl[2] / ll[2];
    const size_t n_wgs = n_wg_z * n_wg_x * n_wg_y;

    auto run = [&](cl_kernel k, int count_half8, const char *label) {
      cl.clSetKernelArg(k, 0, sizeof(cl_mem), &weights_dev);
      cl.clSetKernelArg(k, 1, sizeof(cl_mem), &xm);
      cl.clSetKernelArg(k, 2, sizeof(cl_mem), &dst);
      cl.clSetKernelArg(k, 3, sizeof(int4), &s0);
      cl.clSetKernelArg(k, 4, sizeof(int4), &s1);
      cl.clSetKernelArg(k, 5, sizeof(int4), &s2);

      for (int i = 0; i < 50; ++i)
        cl.clEnqueueNDRangeKernel(queue, k, 3, 0, gl, ll, 0, 0, 0);
      cl.clFinish(queue);

      auto t0 = std::chrono::high_resolution_clock::now();
      for (int i = 0; i < 50; ++i)
        cl.clEnqueueNDRangeKernel(queue, k, 3, 0, gl, ll, 0, 0, 0);
      cl.clFinish(queue);
      auto t1 = std::chrono::high_resolution_clock::now();
      double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / 50;

      const double bytes_per_wg = 12.0 * iters * count_half8 * 16.0;
      const double bytes_per_dispatch = bytes_per_wg * n_wgs;
      const double gbps = (bytes_per_dispatch / (us / 1e6)) / 1e9;

      fprintf(stderr,
              "    %-20s  %7.1f us   requested=%7.2f MB   %7.1f GB/s\n",
              label, us, bytes_per_dispatch / (1024.0 * 1024.0), gbps);
    };

    run(k32, 32, "count=32 (padded)");
    run(k8,   8, "count=8  (dense)");

    cl.clReleaseMemObject(weights_dev);
    cl.clReleaseMemObject(xm);
    cl.clReleaseMemObject(dst);
  }

  cl.clReleaseKernel(k32);
  cl.clReleaseKernel(k8);
}


// ============================================================================
// Fused int4 conv test: runs fused_conv_int4_fp16 against the reference
// (dequant_int4_to_delegate_fp16 + captured delegate_conv_wave) pipeline,
// reports per-shape rel_l2 correctness + TFLOPS so we can decide if the
// fused path is a production win. Same 7 Qwen3-4B prefill shapes, random
// int4 weights + random fp16 src in [-0.1, 0.1).
// ============================================================================
TEST_F(DelegateConvWaveTest, ModelShapes_Int4Fused) {
  fprintf(stderr, "\n=== Fused int4 conv vs dequant+delegate reference ===\n");

  std::string src_fused = LoadFusedKernelSource();
  std::string src_dequant = LoadDequantKernelSource();
  std::string src_delegate = LoadKernelSource();
  if (src_fused.empty() || src_dequant.empty() || src_delegate.empty()) {
    fprintf(stderr, "SKIP: kernel source not found\n");
    return;
  }

  auto build = [&](const std::string &src, const char *flags,
                   const char *kname) -> cl_kernel {
    cl_int e;
    const char *sp = src.c_str(); size_t sl = src.size();
    cl_program prog = cl.clCreateProgramWithSource(ctx, 1, &sp, &sl, &e);
    e = cl.clBuildProgram(prog, 1, &dev, flags, nullptr, nullptr);
    if (e) {
      size_t sz = 0;
      cl.clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &sz);
      std::vector<char> log(sz + 1, 0);
      cl.clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, sz,
                               log.data(), nullptr);
      fprintf(stderr, "BUILD FAIL (%s): %s\n", kname, log.data());
      return nullptr;
    }
    return cl.clCreateKernel(prog, kname, &e);
  };

  cl_kernel k_fused = build(src_fused,
    "-cl-std=CL2.0", "fused_conv_int4_fp16");
  cl_kernel k_dequant = build(src_dequant,
    "-cl-std=CL2.0", "dequant_int4_to_delegate_fp16");
  cl_kernel k_delegate = build(src_delegate,
    "-qcom-accelerate-16-bit=true -cl-std=CL2.0", "main_function");
  if (!k_fused || !k_dequant || !k_delegate) return;

  for (const auto &s : kModelShapes) {
    const int M = s.M, N = s.N, K = s.K;
    const int src_slices = K / 4, dst_slices = N / 4;
    const double gflops = 2.0 * M * N * K / 1e9;

    fprintf(stderr, "\n  M=%d N=%d K=%d\n", M, N, K);

    // Pseudo-random int4 weights + fp16 scales + fp16 src.
    auto lcg = [](uint32_t &s) { s = s * 1664525u + 1013904223u; return s; };
    auto rand_h = [&](uint32_t &s) {
      float f = ((int32_t)(lcg(s) >> 15) % 2048) / 10240.0f;
      return f32_to_f16(f);
    };

    // packed_weights: [(K/4) * N] ushort, each holding 4 nibbles.
    const size_t packed_count = (size_t)src_slices * N;
    std::vector<uint16_t> packed_host(packed_count);
    {
      uint32_t rs = 0xC0FFEEu ^ (uint32_t)(N * 31u + K);
      for (auto &u : packed_host) u = (uint16_t)(lcg(rs) & 0xFFFFu);
    }

    // scales: [N] fp16 (per-output-channel).
    std::vector<uint16_t> scales_host(N);
    {
      uint32_t rs = 0xDECADEu ^ (uint32_t)(N + K * 7u);
      for (auto &h : scales_host) h = rand_h(rs);
    }

    // src: random fp16 in [-0.1, 0.1).
    const size_t src_halves = (size_t)M * src_slices * 4;
    std::vector<uint16_t> src_host(src_halves);
    {
      uint32_t rs = 0xBADCAFEu ^ (uint32_t)(M * 31u + K);
      for (auto &h : src_host) h = rand_h(rs);
    }

    cl_int e;
    // Device buffers.
    cl_mem packed_dev = cl.clCreateBuffer(ctx, CL_MEM_READ_ONLY,
      packed_count * 2, nullptr, &e);
    cl.clEnqueueWriteBuffer(queue, packed_dev, 1, 0, packed_count * 2,
      packed_host.data(), 0, 0, 0);
    cl_mem scales_dev = cl.clCreateBuffer(ctx, CL_MEM_READ_ONLY,
      (size_t)N * 2, nullptr, &e);
    cl.clEnqueueWriteBuffer(queue, scales_dev, 1, 0, (size_t)N * 2,
      scales_host.data(), 0, 0, 0);
    // fp16 weight buffer for the reference pipeline (dequant writes here).
    cl_mem fp16_weights_dev = cl.clCreateBuffer(ctx, CL_MEM_READ_WRITE,
      (size_t)N * K * 2, nullptr, &e);
    cl_mem xm = cl.clCreateBuffer(ctx, 0x4, 6144, nullptr, &e);

    cl_image_format fmt = {CL_RGBA, CL_HALF_FLOAT};
    cl_image_desc sd = {}; sd.image_type = CL_MEM_OBJECT_IMAGE2D;
    sd.image_width = M; sd.image_height = src_slices;
    cl_mem si = cl.clCreateImage(ctx, CL_MEM_READ_ONLY, &fmt, &sd, 0, &e);
    {
      size_t o[3]={0,0,0}, r[3]={(size_t)M,(size_t)src_slices,1};
      cl.clEnqueueWriteImage(queue, si, 1, o, r, 0, 0, src_host.data(), 0, 0, 0);
    }
    cl_image_desc bd = {}; bd.image_type = CL_MEM_OBJECT_IMAGE2D;
    bd.image_width = dst_slices; bd.image_height = 1;
    cl_mem bias = cl.clCreateImage(ctx, CL_MEM_READ_ONLY, &fmt, &bd, 0, &e);
    {
      std::vector<uint16_t> z(dst_slices * 4, 0);
      size_t o[3]={0,0,0}, r[3]={(size_t)dst_slices,1,1};
      cl.clEnqueueWriteImage(queue, bias, 1, o, r, 0, 0, z.data(), 0, 0, 0);
    }
    cl_image_desc dd = {}; dd.image_type = CL_MEM_OBJECT_IMAGE2D;
    dd.image_width = M; dd.image_height = dst_slices;
    cl_mem dst_ref = cl.clCreateImage(ctx, CL_MEM_READ_WRITE, &fmt, &dd, 0, &e);
    cl_mem dst_fused = cl.clCreateImage(ctx, CL_MEM_READ_WRITE, &fmt, &dd, 0, &e);

    // ======= Reference: dequant + delegate conv =======
    // dequant
    {
      int sn = N, sk = K;
      cl.clSetKernelArg(k_dequant, 0, sizeof(cl_mem), &packed_dev);
      cl.clSetKernelArg(k_dequant, 1, sizeof(cl_mem), &scales_dev);
      cl.clSetKernelArg(k_dequant, 2, sizeof(cl_mem), &fp16_weights_dev);
      cl.clSetKernelArg(k_dequant, 3, sizeof(int), &sn);
      cl.clSetKernelArg(k_dequant, 4, sizeof(int), &sk);
      const int n_z = (dst_slices + 7) / 8;
      const int iters = src_slices / 2;
      const size_t w_halfs = (size_t)n_z * iters * 256;
      const int tot = (int)(w_halfs / 16);
      size_t dg[3] = {(size_t)(((tot+255)/256)*256), 1, 1};
      size_t dl[3] = {256, 1, 1};
      cl.clEnqueueNDRangeKernel(queue, k_dequant, 3, 0, dg, dl, 0, 0, 0);
    }
    // delegate conv
    {
      int4 s0 = {1, dst_slices, M, 32};
      int4 s1 = {src_slices, 0, 0, src_slices};
      int4 s2 = {1, 1, 0, 0};
      cl.clSetKernelArg(k_delegate, 0, sizeof(cl_mem), &fp16_weights_dev);
      cl.clSetKernelArg(k_delegate, 1, sizeof(cl_mem), &xm);
      cl.clSetKernelArg(k_delegate, 2, sizeof(cl_mem), &bias);
      cl.clSetKernelArg(k_delegate, 3, sizeof(cl_mem), &dst_ref);
      cl.clSetKernelArg(k_delegate, 4, sizeof(cl_mem), &si);
      cl.clSetKernelArg(k_delegate, 5, sizeof(int4), &s0);
      cl.clSetKernelArg(k_delegate, 6, sizeof(int4), &s1);
      cl.clSetKernelArg(k_delegate, 7, sizeof(int4), &s2);
      size_t gz = (((dst_slices + 7) / 8 + 3) / 4) * 128;
      size_t gl[3] = {gz, (size_t)((M + 127) / 128), 4};
      size_t ll[3] = {128, 1, 4};
      cl.clEnqueueNDRangeKernel(queue, k_delegate, 3, 0, gl, ll, 0, 0, 0);
    }
    cl.clFinish(queue);

    // ======= Candidate: fused kernel =======
    int M_i = M, N_i = N, K_i = K;
    cl.clSetKernelArg(k_fused, 0, sizeof(cl_mem), &packed_dev);
    cl.clSetKernelArg(k_fused, 1, sizeof(cl_mem), &scales_dev);
    cl.clSetKernelArg(k_fused, 2, sizeof(cl_mem), &si);
    cl.clSetKernelArg(k_fused, 3, sizeof(cl_mem), &bias);
    cl.clSetKernelArg(k_fused, 4, sizeof(cl_mem), &dst_fused);
    cl.clSetKernelArg(k_fused, 5, sizeof(int), &M_i);
    cl.clSetKernelArg(k_fused, 6, sizeof(int), &N_i);
    cl.clSetKernelArg(k_fused, 7, sizeof(int), &K_i);
    // 1 thread per (m, n_slice). Try local=(32, 8, 1) first.
    size_t flocal[3] = {32, 8, 1};
    size_t fglobal[3] = {
      ((size_t)M + 31) / 32 * 32,
      ((size_t)dst_slices + 7) / 8 * 8,
      1
    };

    // Warmup (50) + timed (50).
    for (int i = 0; i < 50; ++i)
      cl.clEnqueueNDRangeKernel(queue, k_fused, 3, 0, fglobal, flocal, 0, 0, 0);
    cl.clFinish(queue);

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 50; ++i)
      cl.clEnqueueNDRangeKernel(queue, k_fused, 3, 0, fglobal, flocal, 0, 0, 0);
    cl.clFinish(queue);
    auto t1 = std::chrono::high_resolution_clock::now();
    double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / 50;
    double tflops = (gflops / (us / 1e6)) / 1000.0;

    // Correctness: read both outputs, compute rel_l2.
    const size_t npix = (size_t)M * dst_slices;
    std::vector<uint16_t> ref_out(npix * 4), cand_out(npix * 4);
    size_t o3[3] = {0,0,0}, r3[3] = {(size_t)M, (size_t)dst_slices, 1};
    cl.clEnqueueReadImage(queue, dst_ref, 1, o3, r3, 0, 0,
                          ref_out.data(), 0, 0, 0);
    cl.clEnqueueReadImage(queue, dst_fused, 1, o3, r3, 0, 0,
                          cand_out.data(), 0, 0, 0);

    double sum_sq = 0.0, sum_ref_sq = 0.0, max_abs = 0.0;
    for (size_t i = 0; i < npix * 4; ++i) {
      float rv = f16_to_f32(ref_out[i]);
      float cv = f16_to_f32(cand_out[i]);
      float d = rv - cv;
      sum_sq += (double)d * d;
      sum_ref_sq += (double)rv * rv;
      if (std::abs(d) > max_abs) max_abs = std::abs(d);
    }
    double rel_l2 = sum_ref_sq > 0 ? std::sqrt(sum_sq / sum_ref_sq) : 0.0;

    fprintf(stderr, "    fused:  %7.1f us  %.3f TFLOPS  "
            "rel_l2=%.4g max|d|=%.4g  %s\n",
            us, tflops, rel_l2, max_abs,
            rel_l2 < 0.05 ? "✓" : "✗ DIVERGES");

    cl.clReleaseMemObject(packed_dev);
    cl.clReleaseMemObject(scales_dev);
    cl.clReleaseMemObject(fp16_weights_dev);
    cl.clReleaseMemObject(xm);
    cl.clReleaseMemObject(si);
    cl.clReleaseMemObject(bias);
    cl.clReleaseMemObject(dst_ref);
    cl.clReleaseMemObject(dst_fused);
  }

  cl.clReleaseKernel(k_fused);
  cl.clReleaseKernel(k_dequant);
  cl.clReleaseKernel(k_delegate);
}


// ============================================================================
// CPU-side dense repack for delegate_conv_int4_dense.cl.
// Source layout (original):   packed[(K/4) * N]  ushort, 4 nibbles each
// Dense delegate layout:      [(N/32) * (K/8) * 64] ushort, no padding
//   block_idx = z * (K/8) + it, each block = 64 ushorts:
//     [ 0..31] = packed[(2*it+0) * N + z*32 + 0..31]
//     [32..63] = packed[(2*it+1) * N + z*32 + 0..31]
// Same byte count as the input (2*N*K nibbles), just re-tiled per Z×it.
// ============================================================================
static void RepackInt4ForDelegateDense(const uint16_t *packed, int N, int K,
                                        std::vector<uint16_t> &out) {
  const int n_z = N / 32;
  const int iters = K / 8;
  out.assign((size_t)n_z * iters * 64, 0);
  for (int z = 0; z < n_z; ++z) {
    for (int it = 0; it < iters; ++it) {
      uint16_t *blk = out.data() + ((size_t)z * iters + it) * 64;
      const uint16_t *row0 = packed + (size_t)(2 * it + 0) * N + z * 32;
      const uint16_t *row1 = packed + (size_t)(2 * it + 1) * N + z * 32;
      memcpy(blk +  0, row0, 32 * sizeof(uint16_t));
      memcpy(blk + 32, row1, 32 * sizeof(uint16_t));
    }
  }
}


// ============================================================================
// Dense int4 delegate test — smallest shape only, 1-dispatch correctness
// gate, 3+3 warmup/timed budget. Purposely conservative to stay under the
// Adreno watchdog while we confirm the count=8 dense layout runs at all
// and matches the fp16 delegate reference.
// ============================================================================
TEST_F(DelegateConvWaveTest, SmallShape_DelegateInt4Dense) {
  fprintf(stderr, "\n=== Delegate int4 dense (count=8) vs fp16 ref ===\n");

  std::string src_int4 = LoadDelegateInt4DenseKernelSource();
  std::string src_dequant = LoadDequantKernelSource();
  std::string src_delegate = LoadKernelSource();
  if (src_int4.empty() || src_dequant.empty() || src_delegate.empty()) {
    fprintf(stderr, "SKIP: kernel source not found\n");
    return;
  }

  auto build = [&](const std::string &src, const char *flags,
                   const char *kname) -> cl_kernel {
    cl_int e;
    const char *sp = src.c_str(); size_t sl = src.size();
    cl_program prog = cl.clCreateProgramWithSource(ctx, 1, &sp, &sl, &e);
    e = cl.clBuildProgram(prog, 1, &dev, flags, nullptr, nullptr);
    if (e) {
      size_t sz = 0;
      cl.clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &sz);
      std::vector<char> log(sz + 1, 0);
      cl.clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, sz,
                               log.data(), nullptr);
      fprintf(stderr, "BUILD FAIL (%s): %s\n", kname, log.data());
      return nullptr;
    }
    return cl.clCreateKernel(prog, kname, &e);
  };

  cl_kernel k_int4 = build(src_int4,
    "-qcom-accelerate-16-bit=true -cl-std=CL2.0", "delegate_conv_int4_dense");
  cl_kernel k_dequant = build(src_dequant,
    "-cl-std=CL2.0", "dequant_int4_to_delegate_fp16");
  cl_kernel k_delegate = build(src_delegate,
    "-qcom-accelerate-16-bit=true -cl-std=CL2.0", "main_function");
  if (!k_int4 || !k_dequant || !k_delegate) return;

  // Smallest model shape — keep watchdog budget conservative.
  const int M = 437, N = 1024, K = 2560;
  const int src_slices = K / 4, dst_slices = N / 4;
  const double gflops = 2.0 * M * N * K / 1e9;
  fprintf(stderr, "\n  M=%d N=%d K=%d\n", M, N, K);

  auto lcg = [](uint32_t &s) { s = s * 1664525u + 1013904223u; return s; };
  auto rand_h = [&](uint32_t &s) {
    float f = ((int32_t)(lcg(s) >> 15) % 2048) / 10240.0f;
    return f32_to_f16(f);
  };

  const size_t packed_count = (size_t)src_slices * N;
  std::vector<uint16_t> packed_host(packed_count);
  {
    uint32_t rs = 0xC0FFEEu ^ (uint32_t)(N * 31u + K);
    for (auto &u : packed_host) u = (uint16_t)(lcg(rs) & 0xFFFFu);
  }
  std::vector<uint16_t> scales_host(N);
  {
    uint32_t rs = 0xDECADEu ^ (uint32_t)(N + K * 7u);
    for (auto &h : scales_host) h = rand_h(rs);
  }
  const size_t src_halves = (size_t)M * src_slices * 4;
  std::vector<uint16_t> src_host(src_halves);
  {
    uint32_t rs = 0xBADCAFEu ^ (uint32_t)(M * 31u + K);
    for (auto &h : src_host) h = rand_h(rs);
  }

  std::vector<uint16_t> dense_host;
  RepackInt4ForDelegateDense(packed_host.data(), N, K, dense_host);

  cl_int e;
  cl_mem packed_dev = cl.clCreateBuffer(ctx, CL_MEM_READ_ONLY,
    packed_count * 2, nullptr, &e);
  cl.clEnqueueWriteBuffer(queue, packed_dev, 1, 0, packed_count * 2,
    packed_host.data(), 0, 0, 0);
  cl_mem scales_dev = cl.clCreateBuffer(ctx, CL_MEM_READ_ONLY,
    (size_t)N * 2, nullptr, &e);
  cl.clEnqueueWriteBuffer(queue, scales_dev, 1, 0, (size_t)N * 2,
    scales_host.data(), 0, 0, 0);
  cl_mem fp16_weights_dev = cl.clCreateBuffer(ctx, CL_MEM_READ_WRITE,
    (size_t)N * K * 2, nullptr, &e);
  cl_mem dense_dev = cl.clCreateBuffer(ctx, CL_MEM_READ_ONLY,
    dense_host.size() * 2, nullptr, &e);
  cl.clEnqueueWriteBuffer(queue, dense_dev, 1, 0, dense_host.size() * 2,
    dense_host.data(), 0, 0, 0);
  cl_mem xm = cl.clCreateBuffer(ctx, 0x4, 6144, nullptr, &e);

  cl_image_format fmt = {CL_RGBA, CL_HALF_FLOAT};
  cl_image_desc sd = {}; sd.image_type = CL_MEM_OBJECT_IMAGE2D;
  sd.image_width = M; sd.image_height = src_slices;
  cl_mem si = cl.clCreateImage(ctx, CL_MEM_READ_ONLY, &fmt, &sd, 0, &e);
  {
    size_t o[3]={0,0,0}, r[3]={(size_t)M,(size_t)src_slices,1};
    cl.clEnqueueWriteImage(queue, si, 1, o, r, 0, 0, src_host.data(), 0, 0, 0);
  }
  cl_image_desc bd = {}; bd.image_type = CL_MEM_OBJECT_IMAGE2D;
  bd.image_width = dst_slices; bd.image_height = 1;
  cl_mem bias = cl.clCreateImage(ctx, CL_MEM_READ_ONLY, &fmt, &bd, 0, &e);
  {
    std::vector<uint16_t> z(dst_slices * 4, 0);
    size_t o[3]={0,0,0}, r[3]={(size_t)dst_slices,1,1};
    cl.clEnqueueWriteImage(queue, bias, 1, o, r, 0, 0, z.data(), 0, 0, 0);
  }
  cl_image_desc dd = {}; dd.image_type = CL_MEM_OBJECT_IMAGE2D;
  dd.image_width = M; dd.image_height = dst_slices;
  cl_mem dst_ref = cl.clCreateImage(ctx, CL_MEM_READ_WRITE, &fmt, &dd, 0, &e);
  cl_mem dst_int4 = cl.clCreateImage(ctx, CL_MEM_READ_WRITE, &fmt, &dd, 0, &e);

  int4 s0 = {1, dst_slices, M, 32};
  int4 s1 = {src_slices, 0, 0, src_slices};
  int4 s2 = {1, 1, 0, 0};
  size_t gz = (((dst_slices + 7) / 8 + 3) / 4) * 128;
  size_t gl[3] = {gz, (size_t)((M + 127) / 128), 4};
  size_t ll[3] = {128, 1, 4};

  // Reference: dequant → fp16 delegate.
  {
    int sn = N, sk = K;
    cl.clSetKernelArg(k_dequant, 0, sizeof(cl_mem), &packed_dev);
    cl.clSetKernelArg(k_dequant, 1, sizeof(cl_mem), &scales_dev);
    cl.clSetKernelArg(k_dequant, 2, sizeof(cl_mem), &fp16_weights_dev);
    cl.clSetKernelArg(k_dequant, 3, sizeof(int), &sn);
    cl.clSetKernelArg(k_dequant, 4, sizeof(int), &sk);
    const int n_z = (dst_slices + 7) / 8;
    const int d_iters = src_slices / 2;
    const size_t w_halfs = (size_t)n_z * d_iters * 256;
    const int tot = (int)(w_halfs / 16);
    size_t dg[3] = {(size_t)(((tot+255)/256)*256), 1, 1};
    size_t dl[3] = {256, 1, 1};
    cl.clEnqueueNDRangeKernel(queue, k_dequant, 3, 0, dg, dl, 0, 0, 0);
  }
  {
    cl.clSetKernelArg(k_delegate, 0, sizeof(cl_mem), &fp16_weights_dev);
    cl.clSetKernelArg(k_delegate, 1, sizeof(cl_mem), &xm);
    cl.clSetKernelArg(k_delegate, 2, sizeof(cl_mem), &bias);
    cl.clSetKernelArg(k_delegate, 3, sizeof(cl_mem), &dst_ref);
    cl.clSetKernelArg(k_delegate, 4, sizeof(cl_mem), &si);
    cl.clSetKernelArg(k_delegate, 5, sizeof(int4), &s0);
    cl.clSetKernelArg(k_delegate, 6, sizeof(int4), &s1);
    cl.clSetKernelArg(k_delegate, 7, sizeof(int4), &s2);
    for (int i = 0; i < 5; ++i)
      cl.clEnqueueNDRangeKernel(queue, k_delegate, 3, 0, gl, ll, 0, 0, 0);
    cl.clFinish(queue);
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 5; ++i)
      cl.clEnqueueNDRangeKernel(queue, k_delegate, 3, 0, gl, ll, 0, 0, 0);
    cl.clFinish(queue);
    auto t1 = std::chrono::high_resolution_clock::now();
    double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / 5;
    double tflops = (gflops / (us / 1e6)) / 1000.0;
    fprintf(stderr, "    fp16 ref (delegate):           %7.1f us  %.3f TFLOPS\n",
            us, tflops);
  }

  // Candidate: delegate_conv_int4_dense.
  // 1 dispatch → rel_l2 gate → small 3+3 bench only if correct.
  {
    cl.clSetKernelArg(k_int4, 0, sizeof(cl_mem), &dense_dev);
    cl.clSetKernelArg(k_int4, 1, sizeof(cl_mem), &xm);
    cl.clSetKernelArg(k_int4, 2, sizeof(cl_mem), &scales_dev);
    cl.clSetKernelArg(k_int4, 3, sizeof(cl_mem), &bias);
    cl.clSetKernelArg(k_int4, 4, sizeof(cl_mem), &dst_int4);
    cl.clSetKernelArg(k_int4, 5, sizeof(cl_mem), &si);
    cl.clSetKernelArg(k_int4, 6, sizeof(int4), &s0);
    cl.clSetKernelArg(k_int4, 7, sizeof(int4), &s1);
    cl.clSetKernelArg(k_int4, 8, sizeof(int4), &s2);

    cl.clEnqueueNDRangeKernel(queue, k_int4, 3, 0, gl, ll, 0, 0, 0);
    cl.clFinish(queue);

    const size_t npix = (size_t)M * dst_slices;
    std::vector<uint16_t> ref_out(npix * 4), cand_out(npix * 4);
    size_t o3[3] = {0,0,0}, r3[3] = {(size_t)M, (size_t)dst_slices, 1};
    cl.clEnqueueReadImage(queue, dst_ref, 1, o3, r3, 0, 0,
                          ref_out.data(), 0, 0, 0);
    cl.clEnqueueReadImage(queue, dst_int4, 1, o3, r3, 0, 0,
                          cand_out.data(), 0, 0, 0);
    double sum_sq = 0.0, sum_ref_sq = 0.0, max_abs = 0.0;
    for (size_t i = 0; i < npix * 4; ++i) {
      float rv = f16_to_f32(ref_out[i]);
      float cv = f16_to_f32(cand_out[i]);
      float d = rv - cv;
      sum_sq += (double)d * d;
      sum_ref_sq += (double)rv * rv;
      if (std::abs(d) > max_abs) max_abs = std::abs(d);
    }
    double rel_l2 = sum_ref_sq > 0 ? std::sqrt(sum_sq / sum_ref_sq) : 0.0;
    const bool ok = rel_l2 < 0.05;

    double us = 0.0, tflops = 0.0;
    if (ok) {
      for (int i = 0; i < 3; ++i)
        cl.clEnqueueNDRangeKernel(queue, k_int4, 3, 0, gl, ll, 0, 0, 0);
      cl.clFinish(queue);
      auto t0 = std::chrono::high_resolution_clock::now();
      for (int i = 0; i < 3; ++i)
        cl.clEnqueueNDRangeKernel(queue, k_int4, 3, 0, gl, ll, 0, 0, 0);
      cl.clFinish(queue);
      auto t1 = std::chrono::high_resolution_clock::now();
      us = std::chrono::duration<double, std::micro>(t1 - t0).count() / 3;
      tflops = (gflops / (us / 1e6)) / 1000.0;
    }
    fprintf(stderr, "    int4 dense (count=8 load):     "
                    "%7.1f us  %.3f TFLOPS  rel_l2=%.4g max|d|=%.4g  %s\n",
            us, tflops, rel_l2, max_abs,
            ok ? "OK" : "XX DIVERGES (benchmark skipped)");
    EXPECT_LT(rel_l2, 0.05) << "delegate_conv_int4_dense diverges from ref";
  }

  cl.clReleaseMemObject(packed_dev);
  cl.clReleaseMemObject(scales_dev);
  cl.clReleaseMemObject(fp16_weights_dev);
  cl.clReleaseMemObject(dense_dev);
  cl.clReleaseMemObject(xm);
  cl.clReleaseMemObject(si);
  cl.clReleaseMemObject(bias);
  cl.clReleaseMemObject(dst_ref);
  cl.clReleaseMemObject(dst_int4);
  cl.clReleaseKernel(k_int4);
  cl.clReleaseKernel(k_dequant);
  cl.clReleaseKernel(k_delegate);
}


// ============================================================================
// Probe: r0-only variant of delegate_conv_int4_dense. Same load (count=8),
// same inline int4 dequant, but only r0 is accumulated (1/8 of the FMA
// compute). Diagnostic question: is the full kernel slow because the
// basic block is too big (→ register spill, us_r0 << full_us/8) or because
// it genuinely costs 8x more FMAs (→ us_r0 ≈ full_us/8)?
//
// Correctness is checked only on output channels that are multiples of 8
// — the other 7 of every 8 stay zero in this kernel.
// ============================================================================
TEST_F(DelegateConvWaveTest, Probe_DelegateInt4_R0Only) {
  fprintf(stderr, "\n=== Probe: int4 dense r0-only (1/8 compute) ===\n");

  std::string src_r0 = LoadProbeInt4R0KernelSource();
  std::string src_dequant = LoadDequantKernelSource();
  std::string src_delegate = LoadKernelSource();
  if (src_r0.empty() || src_dequant.empty() || src_delegate.empty()) {
    fprintf(stderr, "SKIP: kernel source not found\n");
    return;
  }

  auto build = [&](const std::string &src, const char *flags,
                   const char *kname) -> cl_kernel {
    cl_int e;
    const char *sp = src.c_str(); size_t sl = src.size();
    cl_program prog = cl.clCreateProgramWithSource(ctx, 1, &sp, &sl, &e);
    e = cl.clBuildProgram(prog, 1, &dev, flags, nullptr, nullptr);
    if (e) {
      size_t sz = 0;
      cl.clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &sz);
      std::vector<char> log(sz + 1, 0);
      cl.clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, sz,
                               log.data(), nullptr);
      fprintf(stderr, "BUILD FAIL (%s): %s\n", kname, log.data());
      return nullptr;
    }
    return cl.clCreateKernel(prog, kname, &e);
  };

  cl_kernel k_r0 = build(src_r0,
    "-qcom-accelerate-16-bit=true -cl-std=CL2.0", "delegate_conv_int4_dense_r0");
  cl_kernel k_dequant = build(src_dequant,
    "-cl-std=CL2.0", "dequant_int4_to_delegate_fp16");
  cl_kernel k_delegate = build(src_delegate,
    "-qcom-accelerate-16-bit=true -cl-std=CL2.0", "main_function");
  if (!k_r0 || !k_dequant || !k_delegate) return;

  const int M = 437, N = 1024, K = 2560;
  const int src_slices = K / 4, dst_slices = N / 4;
  const double gflops_r0 = 2.0 * M * N * K / 1e9 / 8.0;  // 1/8 of full
  const double gflops_full = 2.0 * M * N * K / 1e9;
  fprintf(stderr, "\n  M=%d N=%d K=%d  (r0 does 1/8 of full FMAs)\n", M, N, K);

  auto lcg = [](uint32_t &s) { s = s * 1664525u + 1013904223u; return s; };
  auto rand_h = [&](uint32_t &s) {
    float f = ((int32_t)(lcg(s) >> 15) % 2048) / 10240.0f;
    return f32_to_f16(f);
  };

  const size_t packed_count = (size_t)src_slices * N;
  std::vector<uint16_t> packed_host(packed_count);
  {
    uint32_t rs = 0xC0FFEEu ^ (uint32_t)(N * 31u + K);
    for (auto &u : packed_host) u = (uint16_t)(lcg(rs) & 0xFFFFu);
  }
  std::vector<uint16_t> scales_host(N);
  {
    uint32_t rs = 0xDECADEu ^ (uint32_t)(N + K * 7u);
    for (auto &h : scales_host) h = rand_h(rs);
  }
  const size_t src_halves = (size_t)M * src_slices * 4;
  std::vector<uint16_t> src_host(src_halves);
  {
    uint32_t rs = 0xBADCAFEu ^ (uint32_t)(M * 31u + K);
    for (auto &h : src_host) h = rand_h(rs);
  }

  std::vector<uint16_t> dense_host;
  RepackInt4ForDelegateDense(packed_host.data(), N, K, dense_host);

  cl_int e;
  cl_mem packed_dev = cl.clCreateBuffer(ctx, CL_MEM_READ_ONLY,
    packed_count * 2, nullptr, &e);
  cl.clEnqueueWriteBuffer(queue, packed_dev, 1, 0, packed_count * 2,
    packed_host.data(), 0, 0, 0);
  cl_mem scales_dev = cl.clCreateBuffer(ctx, CL_MEM_READ_ONLY,
    (size_t)N * 2, nullptr, &e);
  cl.clEnqueueWriteBuffer(queue, scales_dev, 1, 0, (size_t)N * 2,
    scales_host.data(), 0, 0, 0);
  cl_mem fp16_weights_dev = cl.clCreateBuffer(ctx, CL_MEM_READ_WRITE,
    (size_t)N * K * 2, nullptr, &e);
  cl_mem dense_dev = cl.clCreateBuffer(ctx, CL_MEM_READ_ONLY,
    dense_host.size() * 2, nullptr, &e);
  cl.clEnqueueWriteBuffer(queue, dense_dev, 1, 0, dense_host.size() * 2,
    dense_host.data(), 0, 0, 0);
  cl_mem xm = cl.clCreateBuffer(ctx, 0x4, 6144, nullptr, &e);

  cl_image_format fmt = {CL_RGBA, CL_HALF_FLOAT};
  cl_image_desc sd = {}; sd.image_type = CL_MEM_OBJECT_IMAGE2D;
  sd.image_width = M; sd.image_height = src_slices;
  cl_mem si = cl.clCreateImage(ctx, CL_MEM_READ_ONLY, &fmt, &sd, 0, &e);
  {
    size_t o[3]={0,0,0}, r[3]={(size_t)M,(size_t)src_slices,1};
    cl.clEnqueueWriteImage(queue, si, 1, o, r, 0, 0, src_host.data(), 0, 0, 0);
  }
  cl_image_desc bd = {}; bd.image_type = CL_MEM_OBJECT_IMAGE2D;
  bd.image_width = dst_slices; bd.image_height = 1;
  cl_mem bias = cl.clCreateImage(ctx, CL_MEM_READ_ONLY, &fmt, &bd, 0, &e);
  {
    std::vector<uint16_t> z(dst_slices * 4, 0);
    size_t o[3]={0,0,0}, r[3]={(size_t)dst_slices,1,1};
    cl.clEnqueueWriteImage(queue, bias, 1, o, r, 0, 0, z.data(), 0, 0, 0);
  }
  cl_image_desc dd = {}; dd.image_type = CL_MEM_OBJECT_IMAGE2D;
  dd.image_width = M; dd.image_height = dst_slices;
  cl_mem dst_ref = cl.clCreateImage(ctx, CL_MEM_READ_WRITE, &fmt, &dd, 0, &e);
  cl_mem dst_r0 = cl.clCreateImage(ctx, CL_MEM_READ_WRITE, &fmt, &dd, 0, &e);

  int4 s0 = {1, dst_slices, M, 32};
  int4 s1 = {src_slices, 0, 0, src_slices};
  int4 s2 = {1, 1, 0, 0};
  size_t gz = (((dst_slices + 7) / 8 + 3) / 4) * 128;
  size_t gl[3] = {gz, (size_t)((M + 127) / 128), 4};
  size_t ll[3] = {128, 1, 4};

  // fp16 reference (same as SmallShape_DelegateInt4Dense, for speed baseline).
  {
    int sn = N, sk = K;
    cl.clSetKernelArg(k_dequant, 0, sizeof(cl_mem), &packed_dev);
    cl.clSetKernelArg(k_dequant, 1, sizeof(cl_mem), &scales_dev);
    cl.clSetKernelArg(k_dequant, 2, sizeof(cl_mem), &fp16_weights_dev);
    cl.clSetKernelArg(k_dequant, 3, sizeof(int), &sn);
    cl.clSetKernelArg(k_dequant, 4, sizeof(int), &sk);
    const int n_z = (dst_slices + 7) / 8;
    const int d_iters = src_slices / 2;
    const size_t w_halfs = (size_t)n_z * d_iters * 256;
    const int tot = (int)(w_halfs / 16);
    size_t dg[3] = {(size_t)(((tot+255)/256)*256), 1, 1};
    size_t dl[3] = {256, 1, 1};
    cl.clEnqueueNDRangeKernel(queue, k_dequant, 3, 0, dg, dl, 0, 0, 0);
  }
  {
    cl.clSetKernelArg(k_delegate, 0, sizeof(cl_mem), &fp16_weights_dev);
    cl.clSetKernelArg(k_delegate, 1, sizeof(cl_mem), &xm);
    cl.clSetKernelArg(k_delegate, 2, sizeof(cl_mem), &bias);
    cl.clSetKernelArg(k_delegate, 3, sizeof(cl_mem), &dst_ref);
    cl.clSetKernelArg(k_delegate, 4, sizeof(cl_mem), &si);
    cl.clSetKernelArg(k_delegate, 5, sizeof(int4), &s0);
    cl.clSetKernelArg(k_delegate, 6, sizeof(int4), &s1);
    cl.clSetKernelArg(k_delegate, 7, sizeof(int4), &s2);
    for (int i = 0; i < 5; ++i)
      cl.clEnqueueNDRangeKernel(queue, k_delegate, 3, 0, gl, ll, 0, 0, 0);
    cl.clFinish(queue);
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 5; ++i)
      cl.clEnqueueNDRangeKernel(queue, k_delegate, 3, 0, gl, ll, 0, 0, 0);
    cl.clFinish(queue);
    auto t1 = std::chrono::high_resolution_clock::now();
    double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / 5;
    double tflops = (gflops_full / (us / 1e6)) / 1000.0;
    fprintf(stderr, "    fp16 ref (delegate, full):      %7.1f us  %.3f TFLOPS (full)\n",
            us, tflops);
  }

  // r0-only int4 dense — measure speed, check correctness on every 8th out_ch.
  {
    cl.clSetKernelArg(k_r0, 0, sizeof(cl_mem), &dense_dev);
    cl.clSetKernelArg(k_r0, 1, sizeof(cl_mem), &xm);
    cl.clSetKernelArg(k_r0, 2, sizeof(cl_mem), &scales_dev);
    cl.clSetKernelArg(k_r0, 3, sizeof(cl_mem), &bias);
    cl.clSetKernelArg(k_r0, 4, sizeof(cl_mem), &dst_r0);
    cl.clSetKernelArg(k_r0, 5, sizeof(cl_mem), &si);
    cl.clSetKernelArg(k_r0, 6, sizeof(int4), &s0);
    cl.clSetKernelArg(k_r0, 7, sizeof(int4), &s1);
    cl.clSetKernelArg(k_r0, 8, sizeof(int4), &s2);

    cl.clEnqueueNDRangeKernel(queue, k_r0, 3, 0, gl, ll, 0, 0, 0);
    cl.clFinish(queue);

    const size_t npix = (size_t)M * dst_slices;
    std::vector<uint16_t> ref_out(npix * 4), r0_out(npix * 4);
    size_t o3[3] = {0,0,0}, r3[3] = {(size_t)M, (size_t)dst_slices, 1};
    cl.clEnqueueReadImage(queue, dst_ref, 1, o3, r3, 0, 0,
                          ref_out.data(), 0, 0, 0);
    cl.clEnqueueReadImage(queue, dst_r0, 1, o3, r3, 0, 0,
                          r0_out.data(), 0, 0, 0);
    // Correctness check: r0 writes only dst_slice rows where slice_idx % 8 == 0
    // — i.e. output RGBA pixels at (m, dst_slice=8*z). Compare only those.
    double sum_sq = 0.0, sum_ref_sq = 0.0, max_abs = 0.0;
    size_t cmp = 0;
    for (int m = 0; m < M; ++m) {
      for (int ds = 0; ds < dst_slices; ds += 8) {
        for (int j = 0; j < 4; ++j) {
          size_t idx = ((size_t)ds * M + m) * 4 + j;
          float rv = f16_to_f32(ref_out[idx]);
          float cv = f16_to_f32(r0_out[idx]);
          float d = rv - cv;
          sum_sq += (double)d * d;
          sum_ref_sq += (double)rv * rv;
          if (std::abs(d) > max_abs) max_abs = std::abs(d);
          cmp++;
        }
      }
    }
    double rel_l2 = sum_ref_sq > 0 ? std::sqrt(sum_sq / sum_ref_sq) : 0.0;
    const bool ok = rel_l2 < 0.05;

    double us = 0.0, tflops = 0.0;
    if (ok) {
      for (int i = 0; i < 3; ++i)
        cl.clEnqueueNDRangeKernel(queue, k_r0, 3, 0, gl, ll, 0, 0, 0);
      cl.clFinish(queue);
      auto t0 = std::chrono::high_resolution_clock::now();
      for (int i = 0; i < 3; ++i)
        cl.clEnqueueNDRangeKernel(queue, k_r0, 3, 0, gl, ll, 0, 0, 0);
      cl.clFinish(queue);
      auto t1 = std::chrono::high_resolution_clock::now();
      us = std::chrono::duration<double, std::micro>(t1 - t0).count() / 3;
      tflops = (gflops_r0 / (us / 1e6)) / 1000.0;
    }
    fprintf(stderr, "    int4 r0-only (count=8 + 1/8 FMA):"
                    " %7.1f us  %.3f TFLOPS (r0-scaled)\n"
                    "      rel_l2=%.4g max|d|=%.4g  cmp=%zu  %s\n",
            us, tflops, rel_l2, max_abs, cmp,
            ok ? "OK" : "XX DIVERGES (benchmark skipped)");
    EXPECT_LT(rel_l2, 0.05) << "r0-only probe diverges from ref";
  }

  cl.clReleaseMemObject(packed_dev);
  cl.clReleaseMemObject(scales_dev);
  cl.clReleaseMemObject(fp16_weights_dev);
  cl.clReleaseMemObject(dense_dev);
  cl.clReleaseMemObject(xm);
  cl.clReleaseMemObject(si);
  cl.clReleaseMemObject(bias);
  cl.clReleaseMemObject(dst_ref);
  cl.clReleaseMemObject(dst_r0);
  cl.clReleaseKernel(k_r0);
  cl.clReleaseKernel(k_dequant);
  cl.clReleaseKernel(k_delegate);
}


// ============================================================================
// Probe: r0..r3 variant (1/2 of full compute, 4 accumulators).
// Together with Probe_DelegateInt4_R0Only (1/8) and SmallShape_DelegateInt4Dense
// (8/8) this triangulates where the register-pressure cliff sits.
// Correctness checked only on the first 4 out_ch of every 8 (Z*8..Z*8+3).
// ============================================================================
TEST_F(DelegateConvWaveTest, Probe_DelegateInt4_R0_R3) {
  fprintf(stderr, "\n=== Probe: int4 dense r0..r3 (1/2 compute, 4 acc) ===\n");

  std::string src_r0r3 = LoadProbeInt4R0R3KernelSource();
  std::string src_dequant = LoadDequantKernelSource();
  std::string src_delegate = LoadKernelSource();
  if (src_r0r3.empty() || src_dequant.empty() || src_delegate.empty()) {
    fprintf(stderr, "SKIP: kernel source not found\n");
    return;
  }

  auto build = [&](const std::string &src, const char *flags,
                   const char *kname) -> cl_kernel {
    cl_int e;
    const char *sp = src.c_str(); size_t sl = src.size();
    cl_program prog = cl.clCreateProgramWithSource(ctx, 1, &sp, &sl, &e);
    e = cl.clBuildProgram(prog, 1, &dev, flags, nullptr, nullptr);
    if (e) {
      size_t sz = 0;
      cl.clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &sz);
      std::vector<char> log(sz + 1, 0);
      cl.clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, sz,
                               log.data(), nullptr);
      fprintf(stderr, "BUILD FAIL (%s): %s\n", kname, log.data());
      return nullptr;
    }
    return cl.clCreateKernel(prog, kname, &e);
  };

  cl_kernel k_r0r3 = build(src_r0r3,
    "-qcom-accelerate-16-bit=true -cl-std=CL2.0", "delegate_conv_int4_dense_r0_r3");
  cl_kernel k_dequant = build(src_dequant,
    "-cl-std=CL2.0", "dequant_int4_to_delegate_fp16");
  cl_kernel k_delegate = build(src_delegate,
    "-qcom-accelerate-16-bit=true -cl-std=CL2.0", "main_function");
  if (!k_r0r3 || !k_dequant || !k_delegate) return;

  const int M = 437, N = 1024, K = 2560;
  const int src_slices = K / 4, dst_slices = N / 4;
  const double gflops_half = 2.0 * M * N * K / 1e9 / 2.0;  // 1/2 of full
  const double gflops_full = 2.0 * M * N * K / 1e9;
  fprintf(stderr, "\n  M=%d N=%d K=%d  (r0..r3 does 1/2 of full FMAs)\n", M, N, K);

  auto lcg = [](uint32_t &s) { s = s * 1664525u + 1013904223u; return s; };
  auto rand_h = [&](uint32_t &s) {
    float f = ((int32_t)(lcg(s) >> 15) % 2048) / 10240.0f;
    return f32_to_f16(f);
  };

  const size_t packed_count = (size_t)src_slices * N;
  std::vector<uint16_t> packed_host(packed_count);
  {
    uint32_t rs = 0xC0FFEEu ^ (uint32_t)(N * 31u + K);
    for (auto &u : packed_host) u = (uint16_t)(lcg(rs) & 0xFFFFu);
  }
  std::vector<uint16_t> scales_host(N);
  {
    uint32_t rs = 0xDECADEu ^ (uint32_t)(N + K * 7u);
    for (auto &h : scales_host) h = rand_h(rs);
  }
  const size_t src_halves = (size_t)M * src_slices * 4;
  std::vector<uint16_t> src_host(src_halves);
  {
    uint32_t rs = 0xBADCAFEu ^ (uint32_t)(M * 31u + K);
    for (auto &h : src_host) h = rand_h(rs);
  }

  std::vector<uint16_t> dense_host;
  RepackInt4ForDelegateDense(packed_host.data(), N, K, dense_host);

  cl_int e;
  cl_mem packed_dev = cl.clCreateBuffer(ctx, CL_MEM_READ_ONLY,
    packed_count * 2, nullptr, &e);
  cl.clEnqueueWriteBuffer(queue, packed_dev, 1, 0, packed_count * 2,
    packed_host.data(), 0, 0, 0);
  cl_mem scales_dev = cl.clCreateBuffer(ctx, CL_MEM_READ_ONLY,
    (size_t)N * 2, nullptr, &e);
  cl.clEnqueueWriteBuffer(queue, scales_dev, 1, 0, (size_t)N * 2,
    scales_host.data(), 0, 0, 0);
  cl_mem fp16_weights_dev = cl.clCreateBuffer(ctx, CL_MEM_READ_WRITE,
    (size_t)N * K * 2, nullptr, &e);
  cl_mem dense_dev = cl.clCreateBuffer(ctx, CL_MEM_READ_ONLY,
    dense_host.size() * 2, nullptr, &e);
  cl.clEnqueueWriteBuffer(queue, dense_dev, 1, 0, dense_host.size() * 2,
    dense_host.data(), 0, 0, 0);
  cl_mem xm = cl.clCreateBuffer(ctx, 0x4, 6144, nullptr, &e);

  cl_image_format fmt = {CL_RGBA, CL_HALF_FLOAT};
  cl_image_desc sd = {}; sd.image_type = CL_MEM_OBJECT_IMAGE2D;
  sd.image_width = M; sd.image_height = src_slices;
  cl_mem si = cl.clCreateImage(ctx, CL_MEM_READ_ONLY, &fmt, &sd, 0, &e);
  {
    size_t o[3]={0,0,0}, r[3]={(size_t)M,(size_t)src_slices,1};
    cl.clEnqueueWriteImage(queue, si, 1, o, r, 0, 0, src_host.data(), 0, 0, 0);
  }
  cl_image_desc bd = {}; bd.image_type = CL_MEM_OBJECT_IMAGE2D;
  bd.image_width = dst_slices; bd.image_height = 1;
  cl_mem bias = cl.clCreateImage(ctx, CL_MEM_READ_ONLY, &fmt, &bd, 0, &e);
  {
    std::vector<uint16_t> z(dst_slices * 4, 0);
    size_t o[3]={0,0,0}, r[3]={(size_t)dst_slices,1,1};
    cl.clEnqueueWriteImage(queue, bias, 1, o, r, 0, 0, z.data(), 0, 0, 0);
  }
  cl_image_desc dd = {}; dd.image_type = CL_MEM_OBJECT_IMAGE2D;
  dd.image_width = M; dd.image_height = dst_slices;
  cl_mem dst_ref = cl.clCreateImage(ctx, CL_MEM_READ_WRITE, &fmt, &dd, 0, &e);
  cl_mem dst_r0r3 = cl.clCreateImage(ctx, CL_MEM_READ_WRITE, &fmt, &dd, 0, &e);

  int4 s0 = {1, dst_slices, M, 32};
  int4 s1 = {src_slices, 0, 0, src_slices};
  int4 s2 = {1, 1, 0, 0};
  size_t gz = (((dst_slices + 7) / 8 + 3) / 4) * 128;
  size_t gl[3] = {gz, (size_t)((M + 127) / 128), 4};
  size_t ll[3] = {128, 1, 4};

  // fp16 reference.
  {
    int sn = N, sk = K;
    cl.clSetKernelArg(k_dequant, 0, sizeof(cl_mem), &packed_dev);
    cl.clSetKernelArg(k_dequant, 1, sizeof(cl_mem), &scales_dev);
    cl.clSetKernelArg(k_dequant, 2, sizeof(cl_mem), &fp16_weights_dev);
    cl.clSetKernelArg(k_dequant, 3, sizeof(int), &sn);
    cl.clSetKernelArg(k_dequant, 4, sizeof(int), &sk);
    const int n_z = (dst_slices + 7) / 8;
    const int d_iters = src_slices / 2;
    const size_t w_halfs = (size_t)n_z * d_iters * 256;
    const int tot = (int)(w_halfs / 16);
    size_t dg[3] = {(size_t)(((tot+255)/256)*256), 1, 1};
    size_t dl[3] = {256, 1, 1};
    cl.clEnqueueNDRangeKernel(queue, k_dequant, 3, 0, dg, dl, 0, 0, 0);
  }
  {
    cl.clSetKernelArg(k_delegate, 0, sizeof(cl_mem), &fp16_weights_dev);
    cl.clSetKernelArg(k_delegate, 1, sizeof(cl_mem), &xm);
    cl.clSetKernelArg(k_delegate, 2, sizeof(cl_mem), &bias);
    cl.clSetKernelArg(k_delegate, 3, sizeof(cl_mem), &dst_ref);
    cl.clSetKernelArg(k_delegate, 4, sizeof(cl_mem), &si);
    cl.clSetKernelArg(k_delegate, 5, sizeof(int4), &s0);
    cl.clSetKernelArg(k_delegate, 6, sizeof(int4), &s1);
    cl.clSetKernelArg(k_delegate, 7, sizeof(int4), &s2);
    for (int i = 0; i < 5; ++i)
      cl.clEnqueueNDRangeKernel(queue, k_delegate, 3, 0, gl, ll, 0, 0, 0);
    cl.clFinish(queue);
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 5; ++i)
      cl.clEnqueueNDRangeKernel(queue, k_delegate, 3, 0, gl, ll, 0, 0, 0);
    cl.clFinish(queue);
    auto t1 = std::chrono::high_resolution_clock::now();
    double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / 5;
    double tflops = (gflops_full / (us / 1e6)) / 1000.0;
    fprintf(stderr, "    fp16 ref (delegate, full):      %7.1f us  %.3f TFLOPS (full)\n",
            us, tflops);
  }

  // r0..r3 int4 dense.
  {
    cl.clSetKernelArg(k_r0r3, 0, sizeof(cl_mem), &dense_dev);
    cl.clSetKernelArg(k_r0r3, 1, sizeof(cl_mem), &xm);
    cl.clSetKernelArg(k_r0r3, 2, sizeof(cl_mem), &scales_dev);
    cl.clSetKernelArg(k_r0r3, 3, sizeof(cl_mem), &bias);
    cl.clSetKernelArg(k_r0r3, 4, sizeof(cl_mem), &dst_r0r3);
    cl.clSetKernelArg(k_r0r3, 5, sizeof(cl_mem), &si);
    cl.clSetKernelArg(k_r0r3, 6, sizeof(int4), &s0);
    cl.clSetKernelArg(k_r0r3, 7, sizeof(int4), &s1);
    cl.clSetKernelArg(k_r0r3, 8, sizeof(int4), &s2);

    cl.clEnqueueNDRangeKernel(queue, k_r0r3, 3, 0, gl, ll, 0, 0, 0);
    cl.clFinish(queue);

    const size_t npix = (size_t)M * dst_slices;
    std::vector<uint16_t> ref_out(npix * 4), r0r3_out(npix * 4);
    size_t o3[3] = {0,0,0}, r3[3] = {(size_t)M, (size_t)dst_slices, 1};
    cl.clEnqueueReadImage(queue, dst_ref, 1, o3, r3, 0, 0,
                          ref_out.data(), 0, 0, 0);
    cl.clEnqueueReadImage(queue, dst_r0r3, 1, o3, r3, 0, 0,
                          r0r3_out.data(), 0, 0, 0);
    // Compare first 4 of every 8 dst slices: z*8 .. z*8+3 for z in 0..n_z-1.
    double sum_sq = 0.0, sum_ref_sq = 0.0, max_abs = 0.0;
    size_t cmp = 0;
    for (int m = 0; m < M; ++m) {
      for (int z = 0; z * 8 < dst_slices; ++z) {
        for (int k = 0; k < 4 && (z * 8 + k) < dst_slices; ++k) {
          int ds = z * 8 + k;
          for (int j = 0; j < 4; ++j) {
            size_t idx = ((size_t)ds * M + m) * 4 + j;
            float rv = f16_to_f32(ref_out[idx]);
            float cv = f16_to_f32(r0r3_out[idx]);
            float d = rv - cv;
            sum_sq += (double)d * d;
            sum_ref_sq += (double)rv * rv;
            if (std::abs(d) > max_abs) max_abs = std::abs(d);
            cmp++;
          }
        }
      }
    }
    double rel_l2 = sum_ref_sq > 0 ? std::sqrt(sum_sq / sum_ref_sq) : 0.0;
    const bool ok = rel_l2 < 0.05;

    double us = 0.0, tflops = 0.0;
    if (ok) {
      for (int i = 0; i < 3; ++i)
        cl.clEnqueueNDRangeKernel(queue, k_r0r3, 3, 0, gl, ll, 0, 0, 0);
      cl.clFinish(queue);
      auto t0 = std::chrono::high_resolution_clock::now();
      for (int i = 0; i < 3; ++i)
        cl.clEnqueueNDRangeKernel(queue, k_r0r3, 3, 0, gl, ll, 0, 0, 0);
      cl.clFinish(queue);
      auto t1 = std::chrono::high_resolution_clock::now();
      us = std::chrono::duration<double, std::micro>(t1 - t0).count() / 3;
      tflops = (gflops_half / (us / 1e6)) / 1000.0;
    }
    fprintf(stderr, "    int4 r0..r3 (count=8 + 1/2 FMA):"
                    " %7.1f us  %.3f TFLOPS (r0..r3-scaled)\n"
                    "      rel_l2=%.4g max|d|=%.4g  cmp=%zu  %s\n",
            us, tflops, rel_l2, max_abs, cmp,
            ok ? "OK" : "XX DIVERGES (benchmark skipped)");
    EXPECT_LT(rel_l2, 0.05) << "r0..r3 probe diverges from ref";
  }

  cl.clReleaseMemObject(packed_dev);
  cl.clReleaseMemObject(scales_dev);
  cl.clReleaseMemObject(fp16_weights_dev);
  cl.clReleaseMemObject(dense_dev);
  cl.clReleaseMemObject(xm);
  cl.clReleaseMemObject(si);
  cl.clReleaseMemObject(bias);
  cl.clReleaseMemObject(dst_ref);
  cl.clReleaseMemObject(dst_r0r3);
  cl.clReleaseKernel(k_r0r3);
  cl.clReleaseKernel(k_dequant);
  cl.clReleaseKernel(k_delegate);
}


} // namespace

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
