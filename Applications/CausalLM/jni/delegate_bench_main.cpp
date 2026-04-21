// SPDX-License-Identifier: Apache-2.0
//
// Standalone delegate conv_wave fp16 bench, mirroring
// unittest_delegate_conv_wave.cpp's ModelShapes_DelegateFp16 test byte-for-byte
// (dlopen libOpenCL, flags=0 queue, same kernel / shape / build flags /
// priming). Unlike the unittest, this binary is LINKED against libnntrainer,
// libccapi-nntrainer, and libcausallm_core (i.e. every shared library that
// nntrainer_causallm loads). The NNTR_BENCH_STAGE env var controls how much
// nntrainer code actually runs before the bench:
//
//   NNTR_BENCH_STAGE=0
//     Libraries are loaded via DT_NEEDED and their static initializers run,
//     but no nntrainer symbol is referenced from main().
//     Previous run: 2.87 TFLOPS on M=437 N=4096 K=2560 — matches unittest.
//
//   NNTR_BENCH_STAGE=1
//     Stage 0 + opencl::ContextManager::Global().GetContext(). Creates
//     nntrainer's cl_context with the Qualcomm HIGH perf / HIGH priority
//     hints. Nothing else (no queue, no kernel compile).
//
//   NNTR_BENCH_STAGE=2
//     Stage 1 + opencl::CommandQueueManager::Global().CreateCommandQueue().
//     Adds nntrainer's CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE |
//     CL_QUEUE_PROFILING_ENABLE queue on that context.
//
//   NNTR_BENCH_STAGE=3
//     Stage 2 + nntrainer::Engine::Global() — full ClContext::initialize():
//     compiles all blas / attention CL kernels, registers layer factories,
//     etc. Previous stage 1 measurement: 1.62 TFLOPS. This is what
//     nntrainer_causallm does.
//
//   NNTR_BENCH_STAGE=4
//     Stage 3 + Factory::registerModel call.
//
// Bisection: whichever stage first drops from 2.87 to 1.62 TFLOPS is the
// culprit.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#if NNTR_BENCH_STAGE >= 1
#include <opencl_context_manager.h>
#endif

#if NNTR_BENCH_STAGE >= 2
#include <opencl_command_queue_manager.h>
#endif

#if NNTR_BENCH_STAGE >= 3
#include <engine.h>
#endif

#if NNTR_BENCH_STAGE >= 4
#include <factory.h>
#include "causal_lm.h"
#include "qwen3_causallm.h"
#endif

namespace {

// ---------------------------------------------------------------------------
// Minimal CL types / constants (no CL headers — all via dlopen, same as the
// unittest so we don't share anything with nntrainer's opencl_loader).
// ---------------------------------------------------------------------------
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
constexpr cl_uint CL_MEM_READ_WRITE = (1 << 0);
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

static uint16_t f32_to_f16(float v) {
  uint32_t f;
  std::memcpy(&f, &v, 4);
  uint32_t sign = (f >> 16) & 0x8000;
  int exp = ((f >> 23) & 0xFF) - 127 + 15;
  uint32_t mant = (f >> 13) & 0x3FF;
  if (exp <= 0) return sign;
  if (exp >= 31) return sign | 0x7C00;
  return sign | (exp << 10) | mant;
}

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
  DECL(cl_int, clSetKernelArg, (cl_kernel, cl_uint, size_t, const void *));
  DECL(cl_int, clEnqueueNDRangeKernel,
       (cl_command_queue, cl_kernel, cl_uint, const size_t *, const size_t *,
        const size_t *, cl_uint, const cl_event *, cl_event *));
  DECL(cl_int, clEnqueueWriteBuffer,
       (cl_command_queue, cl_mem, cl_uint, size_t, size_t, const void *,
        cl_uint, const cl_event *, cl_event *));
  DECL(cl_int, clEnqueueWriteImage,
       (cl_command_queue, cl_mem, cl_uint, const size_t *, const size_t *,
        size_t, size_t, const void *, cl_uint, const cl_event *, cl_event *));
  DECL(cl_int, clFinish, (cl_command_queue));
  DECL(cl_int, clReleaseMemObject, (cl_mem));
  DECL(cl_int, clReleaseKernel, (cl_kernel));
  DECL(cl_int, clReleaseProgram, (cl_program));
  DECL(cl_int, clReleaseCommandQueue, (cl_command_queue));
  DECL(cl_int, clReleaseContext, (cl_context));

  bool Load() {
    lib = dlopen("libOpenCL.so", RTLD_NOW);
    if (!lib) lib = dlopen("/system/vendor/lib64/libOpenCL.so", RTLD_NOW);
    if (!lib) return false;
#define L(n) n = (pfn_##n)dlsym(lib, #n)
    L(clGetPlatformIDs); L(clGetDeviceIDs); L(clGetDeviceInfo);
    L(clCreateContext); L(clCreateCommandQueue);
    L(clCreateBuffer); L(clCreateImage);
    L(clCreateProgramWithSource); L(clBuildProgram); L(clGetProgramBuildInfo);
    L(clCreateKernel); L(clSetKernelArg); L(clEnqueueNDRangeKernel);
    L(clEnqueueWriteBuffer); L(clEnqueueWriteImage); L(clFinish);
    L(clReleaseMemObject); L(clReleaseKernel); L(clReleaseProgram);
    L(clReleaseCommandQueue); L(clReleaseContext);
#undef L
    return clGetPlatformIDs != nullptr;
  }
};
#undef DECL

std::string LoadKernelSource() {
  const char *paths[] = {
    "delegate_conv_wave.cl",
    "/data/local/tmp/nntrainer/test/delegate_conv_wave.cl",
    "nntrainer/tensor/cl_operations/cl_kernels/delegate_conv_wave.cl",
  };
  for (auto p : paths) {
    std::ifstream f(p);
    if (f.good())
      return std::string((std::istreambuf_iterator<char>(f)),
                         std::istreambuf_iterator<char>());
  }
  return "";
}

void dump_maps(const char *phase) {
  std::ifstream maps("/proc/self/maps");
  std::string line;
  std::vector<std::string> seen;
  while (std::getline(maps, line)) {
    bool hit =
      line.find("libOpenCL") != std::string::npos ||
      line.find("libGLES") != std::string::npos ||
      line.find("libadreno") != std::string::npos ||
      line.find("libCB") != std::string::npos ||
      line.find("libgsl") != std::string::npos ||
      line.find("libandroid") != std::string::npos ||
      line.find("libEGL") != std::string::npos ||
      line.find("libnntrainer") != std::string::npos ||
      line.find("libcausallm") != std::string::npos ||
      line.find("libccapi") != std::string::npos;
    if (!hit) continue;
    auto sp = line.find_last_of(' ');
    std::string path = sp == std::string::npos ? line : line.substr(sp + 1);
    if (!path.empty() && path[0] == '/' &&
        std::find(seen.begin(), seen.end(), path) == seen.end()) {
      seen.push_back(path);
      fprintf(stderr, "[nntr_delegate_bench maps %s] %s\n", phase, path.c_str());
    }
  }
}

} // namespace

int main(int, char **) {
  fprintf(stderr, "[nntr_delegate_bench] STAGE=%d\n", (int)NNTR_BENCH_STAGE);

  dump_maps("entry");

#if NNTR_BENCH_STAGE >= 1
  // Stage 1+: create nntrainer's cl_context via opencl::ContextManager.
  // This applies Qualcomm HIGH perf / HIGH priority context hints. No
  // command queue, no kernel compile beyond that.
  {
    auto cl_ctx = nntrainer::opencl::ContextManager::Global().GetContext();
    fprintf(stderr, "[nntr_delegate_bench] ContextManager.GetContext() = %p\n",
            (void *)cl_ctx);
  }
#endif

#if NNTR_BENCH_STAGE >= 2
  // Stage 2+: additionally create nntrainer's OOO+PROFILING command queue
  // on top of that context.
  {
    bool ok =
      nntrainer::opencl::CommandQueueManager::Global().CreateCommandQueue();
    fprintf(stderr, "[nntr_delegate_bench] CreateCommandQueue() = %d\n",
            (int)ok);
  }
#endif

#if NNTR_BENCH_STAGE >= 3
  // Stage 3+: full Engine::Global() — triggers ClContext::initialize()
  // which compiles all blas/attention CL kernels, registers layer
  // factories, and sets the MemAllocator.
  (void)nntrainer::Engine::Global();
  fprintf(stderr, "[nntr_delegate_bench] Engine::Global() done\n");
#endif

#if NNTR_BENCH_STAGE >= 4
  // Stage 4+: register a CausalLM model (mirrors main.cpp).
  causallm::Factory::Instance().registerModel(
    "Qwen3ForCausalLM",
    [](nlohmann::json cfg, nlohmann::json gen, nlohmann::json nntr) {
      return std::make_unique<causallm::Qwen3CausalLM>(cfg, gen, nntr);
    });
  fprintf(stderr, "[nntr_delegate_bench] factory.registerModel done\n");
#endif

  CLFuncs cl;
  if (!cl.Load()) {
    fprintf(stderr, "[nntr_delegate_bench] Cannot load libOpenCL.so\n");
    return 1;
  }

  cl_uint np;
  cl_platform_id plat;
  cl.clGetPlatformIDs(1, &plat, &np);
  cl_uint nd;
  cl_device_id dev;
  cl.clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 1, &dev, &nd);
  char name[256] = {};
  cl.clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof(name), name, nullptr);
  fprintf(stderr, "[nntr_delegate_bench] GPU: %s\n", name);

  cl_int err;
  cl_context ctx = cl.clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &err);
  if (!ctx) {
    fprintf(stderr, "[nntr_delegate_bench] clCreateContext failed: %d\n", err);
    return 1;
  }
  cl_command_queue queue = cl.clCreateCommandQueue(ctx, dev, 0, &err);

  std::string ksrc = LoadKernelSource();
  if (ksrc.empty()) {
    fprintf(stderr, "[nntr_delegate_bench] delegate_conv_wave.cl not found\n");
    return 1;
  }

  const char *sp_ = ksrc.c_str();
  size_t sl_ = ksrc.size();
  cl_program prog =
    cl.clCreateProgramWithSource(ctx, 1, &sp_, &sl_, &err);
  err = cl.clBuildProgram(prog, 1, &dev,
                          "-qcom-accelerate-16-bit=true -cl-std=CL2.0",
                          nullptr, nullptr);
  if (err != 0) {
    size_t sz = 0;
    cl.clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &sz);
    std::vector<char> log(sz + 1, 0);
    cl.clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, sz, log.data(),
                             nullptr);
    fprintf(stderr, "[nntr_delegate_bench] build failed: %s\n", log.data());
    return 1;
  }
  cl_kernel kern = cl.clCreateKernel(prog, "main_function", &err);

  // ModelShapes_DelegateFp16 shape: Qwen3-4B q_proj (also k/v/o, gate/up/down
  // at different N/K). Bench them all the same way as the unittest so we get
  // directly comparable numbers.
  struct ShapeConfig { int M, N, K; };
  static const ShapeConfig kShapes[] = {
    {437, 4096, 2560},
    {437, 1024, 2560},
    {437, 2560, 4096},
    {437, 9728, 2560},
    {437, 2560, 9728},
  };

  for (const auto &s : kShapes) {
    const int M = s.M, N = s.N, K = s.K;
    const int src_slices = K / 4, dst_slices = N / 4;
    const double gflops = 2.0 * M * N * K / 1e9;

    size_t wbytes = (size_t)N * K * 2;
    cl_mem w = cl.clCreateBuffer(ctx, CL_MEM_READ_ONLY, wbytes, nullptr, &err);
    {
      uint16_t v = f32_to_f16(0.01f);
      std::vector<uint16_t> d(wbytes / 2, v);
      cl.clEnqueueWriteBuffer(queue, w, 1, 0, wbytes, d.data(), 0, nullptr,
                              nullptr);
    }
    cl_mem xm = cl.clCreateBuffer(ctx, CL_MEM_READ_WRITE, 6144, nullptr, &err);

    cl_image_format fmt = {CL_RGBA, CL_HALF_FLOAT};
    cl_image_desc bd = {};
    bd.image_type = CL_MEM_OBJECT_IMAGE2D;
    bd.image_width = dst_slices;
    bd.image_height = 1;
    cl_mem bias = cl.clCreateImage(ctx, CL_MEM_READ_ONLY, &fmt, &bd, 0, &err);
    {
      std::vector<uint16_t> z(dst_slices * 4, 0);
      size_t o[3] = {0, 0, 0}, r[3] = {(size_t)dst_slices, 1, 1};
      cl.clEnqueueWriteImage(queue, bias, 1, o, r, 0, 0, z.data(), 0, nullptr,
                             nullptr);
    }

    cl_image_desc sd = {};
    sd.image_type = CL_MEM_OBJECT_IMAGE2D;
    sd.image_width = M;
    sd.image_height = src_slices;
    cl_mem si = cl.clCreateImage(ctx, CL_MEM_READ_ONLY, &fmt, &sd, 0, &err);
    {
      uint16_t one = f32_to_f16(1.0f);
      std::vector<uint16_t> d((size_t)M * src_slices * 4, one);
      size_t o[3] = {0, 0, 0}, r[3] = {(size_t)M, (size_t)src_slices, 1};
      cl.clEnqueueWriteImage(queue, si, 1, o, r, 0, 0, d.data(), 0, nullptr,
                             nullptr);
    }

    cl_image_desc dd = {};
    dd.image_type = CL_MEM_OBJECT_IMAGE2D;
    dd.image_width = M;
    dd.image_height = dst_slices;
    cl_mem di = cl.clCreateImage(ctx, CL_MEM_READ_WRITE, &fmt, &dd, 0, &err);

    int4 s0 = {1, dst_slices, M, 32};
    int4 s1 = {src_slices, 0, 0, src_slices};
    int4 s2 = {1, 1, 0, 0};
    cl.clSetKernelArg(kern, 0, sizeof(cl_mem), &w);
    cl.clSetKernelArg(kern, 1, sizeof(cl_mem), &xm);
    cl.clSetKernelArg(kern, 2, sizeof(cl_mem), &bias);
    cl.clSetKernelArg(kern, 3, sizeof(cl_mem), &di);
    cl.clSetKernelArg(kern, 4, sizeof(cl_mem), &si);
    cl.clSetKernelArg(kern, 5, sizeof(int4), &s0);
    cl.clSetKernelArg(kern, 6, sizeof(int4), &s1);
    cl.clSetKernelArg(kern, 7, sizeof(int4), &s2);

    size_t gz = (((dst_slices + 7) / 8 + 3) / 4) * 128;
    size_t local[3] = {128, 1, 4};
    size_t global[3] = {gz, (size_t)((M + 127) / 128), 4};

    for (int i = 0; i < 50; ++i)
      cl.clEnqueueNDRangeKernel(queue, kern, 3, nullptr, global, local, 0,
                                nullptr, nullptr);
    cl.clFinish(queue);
    const int N_ITER = 32;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N_ITER; ++i)
      cl.clEnqueueNDRangeKernel(queue, kern, 3, nullptr, global, local, 0,
                                nullptr, nullptr);
    cl.clFinish(queue);
    auto t1 = std::chrono::high_resolution_clock::now();
    double us =
      std::chrono::duration<double, std::micro>(t1 - t0).count() / N_ITER;
    double tflops = (gflops / (us / 1e6)) / 1000.0;
    fprintf(stderr,
            "[nntr_delegate_bench STAGE=%d] M=%d N=%d K=%d: %.1f us = %.3f "
            "TFLOPS\n",
            (int)NNTR_BENCH_STAGE, M, N, K, us, tflops);

    cl.clReleaseMemObject(w);
    cl.clReleaseMemObject(xm);
    cl.clReleaseMemObject(bias);
    cl.clReleaseMemObject(si);
    cl.clReleaseMemObject(di);
  }

  dump_maps("after-kernel");

  cl.clReleaseKernel(kern);
  cl.clReleaseProgram(prog);
  cl.clReleaseCommandQueue(queue);
  cl.clReleaseContext(ctx);
  return 0;
}
