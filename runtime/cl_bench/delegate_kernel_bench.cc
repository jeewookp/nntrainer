// Benchmark the captured delegate CL kernel (program_002.cl) directly.
//
// Loads the intercepted kernel source, sets up matching buffers/images,
// and dispatches with the correct arguments. Uses the device's own
// libOpenCL.so which supports the Qualcomm extensions (qcom_subgroup_*,
// inline asm, xmem_buffer).
//
// The kernel signature:
//   __kernel void main_function(
//     __constant half8* weights_buffer,    // packed weights
//     __constant half8* xmem_buffer,       // wave memory scratch (6144 bytes)
//     __read_only image2d_t biases_image2d,
//     __write_only image2d_t dst_tensor_image2d,
//     __read_only image2d_t src_tensor_image2d,
//     int4 shared_int4_0,
//     int4 shared_int4_1,
//     int4 shared_int4_2)
//
// For matmul C[M,N] = A[M,K] * B[K,N], mapped as 1x1 conv:
//   src: image2d, width=M, height=K/4 (slices)
//   dst: image2d, width=M, height=N/4 (slices)
//   weights: constant buffer, N*K*sizeof(half) bytes

#include <EGL/egl.h>
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

// Minimal CL types
typedef void* cl_context;
typedef void* cl_command_queue;
typedef void* cl_device_id;
typedef void* cl_platform_id;
typedef void* cl_program;
typedef void* cl_kernel;
typedef void* cl_mem;
typedef void* cl_event;
typedef int32_t cl_int;
typedef uint32_t cl_uint;
typedef uint64_t cl_ulong;
typedef uint64_t cl_bitfield;

// CL constants
#define CL_SUCCESS 0
#define CL_DEVICE_TYPE_GPU (1 << 2)
#define CL_MEM_READ_ONLY (1 << 2)
#define CL_MEM_WRITE_ONLY (1 << 1)
#define CL_MEM_READ_WRITE (1 << 0)
#define CL_MEM_COPY_HOST_PTR (1 << 5)
#define CL_DEVICE_NAME 0x102B
#define CL_DEVICE_MAX_COMPUTE_UNITS 0x1002
#define CL_PLATFORM_NAME 0x0902
#define CL_QUEUE_PROFILING_ENABLE (1 << 1)
#define CL_PROFILING_COMMAND_QUEUED 0x1280
#define CL_PROFILING_COMMAND_SUBMIT 0x1281
#define CL_PROFILING_COMMAND_START 0x1282
#define CL_PROFILING_COMMAND_END 0x1283
#define CL_PROGRAM_BUILD_LOG 0x1183
#define CL_RGBA 0x10B5
#define CL_HALF_FLOAT 0x10DD
#define CL_FLOAT 0x10DE

typedef struct { cl_uint image_channel_order; cl_uint image_channel_data_type; } cl_image_format;
typedef struct {
  cl_uint image_type; size_t image_width; size_t image_height; size_t image_depth;
  size_t image_array_size; size_t image_row_pitch; size_t image_slice_pitch;
  cl_uint num_mip_levels; cl_uint num_samples; cl_mem buffer;
} cl_image_desc;

#define CL_MEM_OBJECT_IMAGE2D 0x10F1

// CL function pointers
#define DECL_CL(ret, name, params) typedef ret (*pfn_##name) params; static pfn_##name p_##name = nullptr;
DECL_CL(cl_int, clGetPlatformIDs, (cl_uint, cl_platform_id*, cl_uint*))
DECL_CL(cl_int, clGetPlatformInfo, (cl_platform_id, cl_uint, size_t, void*, size_t*))
DECL_CL(cl_int, clGetDeviceIDs, (cl_platform_id, cl_bitfield, cl_uint, cl_device_id*, cl_uint*))
DECL_CL(cl_int, clGetDeviceInfo, (cl_device_id, cl_uint, size_t, void*, size_t*))
DECL_CL(cl_context, clCreateContext, (const intptr_t*, cl_uint, const cl_device_id*, void*, void*, cl_int*))
DECL_CL(cl_command_queue, clCreateCommandQueue, (cl_context, cl_device_id, cl_bitfield, cl_int*))
DECL_CL(cl_command_queue, clCreateCommandQueueWithProperties, (cl_context, cl_device_id, const uint64_t*, cl_int*))
DECL_CL(cl_mem, clCreateBuffer, (cl_context, cl_bitfield, size_t, void*, cl_int*))
DECL_CL(cl_mem, clCreateImage, (cl_context, cl_bitfield, const cl_image_format*, const cl_image_desc*, void*, cl_int*))
DECL_CL(cl_program, clCreateProgramWithSource, (cl_context, cl_uint, const char**, const size_t*, cl_int*))
DECL_CL(cl_int, clBuildProgram, (cl_program, cl_uint, const cl_device_id*, const char*, void*, void*))
DECL_CL(cl_int, clGetProgramBuildInfo, (cl_program, cl_device_id, cl_uint, size_t, void*, size_t*))
DECL_CL(cl_int, clGetProgramInfo, (cl_program, cl_uint, size_t, void*, size_t*))
DECL_CL(cl_kernel, clCreateKernel, (cl_program, const char*, cl_int*))
DECL_CL(cl_int, clSetKernelArg, (cl_kernel, cl_uint, size_t, const void*))
DECL_CL(cl_int, clEnqueueNDRangeKernel, (cl_command_queue, cl_kernel, cl_uint, const size_t*, const size_t*, const size_t*, cl_uint, const cl_event*, cl_event*))
DECL_CL(cl_int, clFinish, (cl_command_queue))
DECL_CL(cl_int, clReleaseMemObject, (cl_mem))
DECL_CL(cl_int, clReleaseKernel, (cl_kernel))
DECL_CL(cl_int, clReleaseProgram, (cl_program))
DECL_CL(cl_int, clReleaseCommandQueue, (cl_command_queue))
DECL_CL(cl_int, clReleaseContext, (cl_context))
DECL_CL(cl_int, clReleaseEvent, (cl_event))
DECL_CL(cl_int, clGetEventProfilingInfo, (cl_event, cl_uint, size_t, void*, size_t*))
DECL_CL(cl_int, clWaitForEvents, (cl_uint, const cl_event*))
DECL_CL(cl_int, clEnqueueWriteBuffer, (cl_command_queue, cl_mem, cl_uint, size_t, size_t, const void*, cl_uint, const cl_event*, cl_event*))
DECL_CL(cl_int, clEnqueueReadBuffer, (cl_command_queue, cl_mem, cl_uint, size_t, size_t, void*, cl_uint, const cl_event*, cl_event*))
DECL_CL(cl_int, clEnqueueWriteImage, (cl_command_queue, cl_mem, cl_uint, const size_t*, const size_t*, size_t, size_t, const void*, cl_uint, const cl_event*, cl_event*))
DECL_CL(cl_int, clEnqueueReadImage, (cl_command_queue, cl_mem, cl_uint, const size_t*, const size_t*, size_t, size_t, void*, cl_uint, const cl_event*, cl_event*))
// Qualcomm performance hint
DECL_CL(cl_int, clSetPerfHintQCOM, (cl_context, cl_uint))
// CL-GL interop device query
DECL_CL(cl_int, clGetGLContextInfoKHR, (const intptr_t*, cl_uint, size_t, void*, size_t*))
// Command buffer KHR
DECL_CL(void*, clCreateCommandBufferKHR, (cl_uint, const cl_command_queue*, const void*, cl_int*))
DECL_CL(cl_int, clCommandNDRangeKernelKHR, (void*, void*, const void*, cl_kernel, cl_uint, const size_t*, const size_t*, const size_t*, cl_uint, const void*, void*, void*))
DECL_CL(cl_int, clFinalizeCommandBufferKHR, (void*))
DECL_CL(cl_int, clEnqueueCommandBufferKHR, (cl_uint, const cl_command_queue*, void*, cl_uint, const cl_event*, cl_event*))
DECL_CL(cl_int, clReleaseCommandBufferKHR, (void*))
// Qualcomm RecordableQueue extension (cl_qcom_recordable_queues)
DECL_CL(void*, clNewRecordingQCOM, (cl_command_queue, cl_int*))
DECL_CL(cl_int, clEndRecordingQCOM, (void*))
DECL_CL(cl_int, clEnqueueRecordingQCOM, (cl_command_queue, void*, cl_uint, const void*, cl_uint, const cl_event*, cl_event*))
DECL_CL(cl_int, clReleaseRecordingQCOM, (void*))

#define LOAD(h, name) p_##name = (pfn_##name)dlsym(h, #name); \
  if (!p_##name) { fprintf(stderr, "WARN: %s not found\n", #name); }

static bool LoadCL() {
  void* h = dlopen("libOpenCL.so", RTLD_NOW);
  if (!h) h = dlopen("/system/vendor/lib64/libOpenCL.so", RTLD_NOW);
  if (!h) { fprintf(stderr, "Cannot load libOpenCL.so\n"); return false; }
  LOAD(h, clGetPlatformIDs); LOAD(h, clGetPlatformInfo);
  LOAD(h, clGetDeviceIDs); LOAD(h, clGetDeviceInfo);
  LOAD(h, clCreateContext); LOAD(h, clCreateCommandQueue);
  LOAD(h, clCreateCommandQueueWithProperties);
  LOAD(h, clCreateBuffer); LOAD(h, clCreateImage);
  LOAD(h, clCreateProgramWithSource); LOAD(h, clBuildProgram);
  LOAD(h, clGetProgramBuildInfo); LOAD(h, clGetProgramInfo);
  LOAD(h, clCreateKernel);
  LOAD(h, clSetKernelArg); LOAD(h, clEnqueueNDRangeKernel);
  LOAD(h, clFinish); LOAD(h, clReleaseMemObject);
  LOAD(h, clReleaseKernel); LOAD(h, clReleaseProgram);
  LOAD(h, clReleaseCommandQueue); LOAD(h, clReleaseContext);
  LOAD(h, clReleaseEvent); LOAD(h, clGetEventProfilingInfo);
  LOAD(h, clWaitForEvents); LOAD(h, clEnqueueWriteBuffer);
  LOAD(h, clEnqueueReadBuffer);
  LOAD(h, clEnqueueWriteImage); LOAD(h, clEnqueueReadImage);
  LOAD(h, clSetPerfHintQCOM);
  LOAD(h, clGetGLContextInfoKHR);
  LOAD(h, clCreateCommandBufferKHR);
  LOAD(h, clCommandNDRangeKernelKHR);
  LOAD(h, clFinalizeCommandBufferKHR);
  LOAD(h, clEnqueueCommandBufferKHR);
  LOAD(h, clReleaseCommandBufferKHR);
  LOAD(h, clNewRecordingQCOM); LOAD(h, clEndRecordingQCOM);
  LOAD(h, clEnqueueRecordingQCOM); LOAD(h, clReleaseRecordingQCOM);
  return true;
}

static std::string ReadFile(const char* path) {
  std::ifstream f(path);
  if (!f) return "";
  return std::string((std::istreambuf_iterator<char>(f)),
                     std::istreambuf_iterator<char>());
}

// int4 type matching OpenCL's int4
struct int4 { int32_t x, y, z, w; };

// fp16 ↔ fp32 conversion (IEEE 754)
static uint16_t f32_to_f16(float v) {
  uint32_t f; memcpy(&f, &v, 4);
  uint32_t sign = (f >> 16) & 0x8000;
  int exp = ((f >> 23) & 0xFF) - 127 + 15;
  uint32_t mant = (f >> 13) & 0x3FF;
  if (exp <= 0) return sign;
  if (exp >= 31) return sign | 0x7C00;
  return sign | (exp << 10) | mant;
}
static float f16_to_f32(uint16_t h) {
  uint32_t sign = (h & 0x8000) << 16;
  uint32_t exp = (h >> 10) & 0x1F;
  uint32_t mant = h & 0x3FF;
  if (exp == 0) { if (mant == 0) { float r; uint32_t v = sign; memcpy(&r, &v, 4); return r; }
    exp = 1; while (!(mant & 0x400)) { mant <<= 1; exp--; } mant &= 0x3FF; }
  else if (exp == 31) { float r; uint32_t v = sign | 0x7F800000 | (mant << 13); memcpy(&r, &v, 4); return r; }
  uint32_t f = sign | ((exp - 15 + 127) << 23) | (mant << 13);
  float r; memcpy(&r, &f, 4); return r;
}

int main(int argc, char** argv) {
  // Parse args
  int M = 1024, N = 6144, K = 1536;
  int warmup = 50, iters = 100;
  const char* cl_file = "cl_intercept/program_002.cl";

  for (int i = 1; i < argc; ++i) {
    if (sscanf(argv[i], "--shape=%dx%dx%d", &M, &N, &K) == 3) continue;
    if (sscanf(argv[i], "--warmup=%d", &warmup) == 1) continue;
    if (sscanf(argv[i], "--iters=%d", &iters) == 1) continue;
    if (strncmp(argv[i], "--cl_file=", 10) == 0) { cl_file = argv[i] + 10; continue; }
  }

  int src_slices = (K + 3) / 4;
  int dst_slices = (N + 3) / 4;
  int dst_slice_groups = (dst_slices + 7) / 8;
  double gflops = 2.0 * M * N * K / 1e9;

  fprintf(stderr, "[dk_bench] Delegate kernel benchmark\n");
  fprintf(stderr, "[dk_bench] M=%d N=%d K=%d (%.3f GFLOPS)\n", M, N, K, gflops);
  fprintf(stderr, "[dk_bench] src_slices=%d dst_slices=%d dst_groups=%d\n",
          src_slices, dst_slices, dst_slice_groups);
  fprintf(stderr, "[dk_bench] cl_file=%s\n", cl_file);

  // Load CL source
  std::string src = ReadFile(cl_file);
  if (src.empty()) { fprintf(stderr, "ERROR: cannot read %s\n", cl_file); return 1; }
  fprintf(stderr, "[dk_bench] Loaded %zu bytes CL source\n", src.size());

  if (!LoadCL()) return 1;

  // Init CL
  cl_platform_id plat; cl_uint nplat;
  p_clGetPlatformIDs(1, &plat, &nplat);
  cl_device_id dev; cl_uint ndev;
  p_clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 1, &dev, &ndev);
  char dname[256] = {};
  p_clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof(dname), dname, nullptr);
  fprintf(stderr, "[dk_bench] GPU: %s\n", dname);

  // ========================================================================
  // Initialize EGL — triggers GPU frequency boost on Adreno.
  // The delegate reuses an EGL environment ("Reusing provided EGL
  // environment" in logs). Without EGL, GPU stays at low freq (~1.2 GHz).
  // With EGL active, driver boosts to max freq (~2.0 GHz).
  // ========================================================================
  EGLDisplay egl_dpy = eglGetDisplay(EGL_DEFAULT_DISPLAY);
  EGLContext egl_ctx = EGL_NO_CONTEXT;
  if (egl_dpy != EGL_NO_DISPLAY) {
    EGLint major, minor;
    if (eglInitialize(egl_dpy, &major, &minor)) {
      fprintf(stderr, "[dk_bench] EGL %d.%d initialized\n", major, minor);
      EGLint cfg_attr[] = { EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT, EGL_NONE };
      EGLConfig cfg;
      EGLint ncfg;
      eglChooseConfig(egl_dpy, cfg_attr, &cfg, 1, &ncfg);
      if (ncfg > 0) {
        EGLint ctx_attr[] = { EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE };
        egl_ctx = eglCreateContext(egl_dpy, cfg, EGL_NO_CONTEXT, ctx_attr);
        if (egl_ctx != EGL_NO_CONTEXT) {
          EGLint pbuf_attr[] = { EGL_WIDTH, 1, EGL_HEIGHT, 1, EGL_NONE };
          EGLSurface pbuf = eglCreatePbufferSurface(egl_dpy, cfg, pbuf_attr);
          if (pbuf != EGL_NO_SURFACE) {
            eglMakeCurrent(egl_dpy, pbuf, pbuf, egl_ctx);
            fprintf(stderr, "[dk_bench] EGL context active\n");
          }
        }
      }
    }
  }

  cl_int err;
  // Create CL context from EGL context (CL-GL interop).
  // The delegate uses this ("Reusing provided EGL environment").
  // On Qualcomm, CL-GL shared context may enable faster memory paths.
  // CL_GL_CONTEXT_KHR = 0x2008, CL_EGL_DISPLAY_KHR = 0x3038
  // Delegate's exact context properties (captured via interception):
  //   0x2008 = CL_GL_CONTEXT_KHR
  //   0x2009 = CL_EGL_DISPLAY_KHR
  //   0x1084 = CL_CONTEXT_PLATFORM
  //   0x40C2 = CL_CONTEXT_PERF_HINT_QCOM → 0x40C3 (HIGH)
  //   0x40C9 → 0x40CB (unknown Qualcomm property)
  // Try to get CL device from EGL context (delegate does this)
  // CL_CURRENT_DEVICE_FOR_GL_CONTEXT_KHR = 0x2006
  cl_device_id egl_dev = nullptr;
  if (egl_dpy != EGL_NO_DISPLAY && egl_ctx != EGL_NO_CONTEXT && p_clGetGLContextInfoKHR) {
    intptr_t q_props[] = {
      (intptr_t)0x2008, (intptr_t)egl_ctx,
      (intptr_t)0x2009, (intptr_t)egl_dpy,
      (intptr_t)0x1084, (intptr_t)plat,
      0
    };
    err = p_clGetGLContextInfoKHR(q_props, 0x2006 /*CL_CURRENT_DEVICE_FOR_GL_CONTEXT_KHR*/,
                                   sizeof(cl_device_id), &egl_dev, nullptr);
    if (err == CL_SUCCESS && egl_dev) {
      fprintf(stderr, "[dk_bench] EGL CL device: %p (clGetDeviceIDs device: %p) %s\n",
              egl_dev, dev, egl_dev == dev ? "SAME" : "DIFFERENT!");
      dev = egl_dev;  // Use EGL device
    } else {
      fprintf(stderr, "[dk_bench] clGetGLContextInfoKHR failed: %d\n", err);
    }
  }

  cl_context ctx = nullptr;
  if (egl_dpy != EGL_NO_DISPLAY && egl_ctx != EGL_NO_CONTEXT) {
    intptr_t ctx_props[] = {
      (intptr_t)0x2008, (intptr_t)egl_ctx,    // CL_GL_CONTEXT_KHR
      (intptr_t)0x2009, (intptr_t)egl_dpy,    // CL_EGL_DISPLAY_KHR (0x2009, not 0x3038!)
      (intptr_t)0x1084, (intptr_t)plat,        // CL_CONTEXT_PLATFORM
      (intptr_t)0x40C2, (intptr_t)0x40C3,     // CL_CONTEXT_PERF_HINT_QCOM(HIGH)
      (intptr_t)0x40C9, (intptr_t)0x40CB,     // Qualcomm unknown (from delegate)
      0
    };
    ctx = p_clCreateContext(ctx_props, 1, &dev, nullptr, nullptr, &err);
    if (ctx && err == CL_SUCCESS) {
      fprintf(stderr, "[dk_bench] CL-EGL shared context created (with QCOM props)\n");
    } else {
      fprintf(stderr, "[dk_bench] CL-EGL context failed (%d), trying without EGL\n", err);
      ctx = nullptr;
    }
  }
  if (!ctx) {
    intptr_t ctx_props[] = {
      (intptr_t)0x1084, (intptr_t)plat,
      (intptr_t)0x40C2, (intptr_t)0x40C3,
      (intptr_t)0x40C9, (intptr_t)0x40CB,
      0
    };
    ctx = p_clCreateContext(ctx_props, 1, &dev, nullptr, nullptr, &err);
    if (err) {
      fprintf(stderr, "[dk_bench] QCOM context failed (%d), basic context\n", err);
      ctx = p_clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &err);
    }
  }
  // Also try runtime perf hint
  if (p_clSetPerfHintQCOM) {
    cl_int ph = p_clSetPerfHintQCOM(ctx, 0x40C3);
    fprintf(stderr, "[dk_bench] clSetPerfHintQCOM(HIGH): %s\n",
            ph == 0 ? "OK" : "failed");
  }
  // CL_QUEUE_PRIORITY_HIGH_KHR = 0x40C7 for high-priority execution
  cl_command_queue queue = p_clCreateCommandQueue(ctx, dev, 0, &err);
  cl_command_queue prof_queue = p_clCreateCommandQueue(ctx, dev, CL_QUEUE_PROFILING_ENABLE, &err);

  // Try to also create a high-priority command queue (Qualcomm extension)
  if (p_clCreateCommandQueueWithProperties) {
    uint64_t qprops[] = { 0x1093 /*CL_QUEUE_PROPERTIES*/, 0,
                           0x40C7 /*CL_QUEUE_PRIORITY_KHR*/, 0x40C3 /*HIGH*/, 0 };
    cl_command_queue hiq = p_clCreateCommandQueueWithProperties(ctx, dev, qprops, &err);
    if (hiq && err == CL_SUCCESS) {
      fprintf(stderr, "[dk_bench] High-priority queue created\n");
      // Use this for profiling too, but keep original for burst
    }
  }

  // Compile kernel
  const char* src_ptr = src.c_str();
  size_t src_len = src.size();
  cl_program prog = p_clCreateProgramWithSource(ctx, 1, &src_ptr, &src_len, &err);
  if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: CreateProgram: %d\n", err); return 1; }

  // Build options will be determined by intercepting delegate's clBuildProgram.
  const char* build_opts = getenv("DK_BUILD_OPTS");
  if (!build_opts) build_opts = "-qcom-accelerate-16-bit=true -cl-std=CL2.0";
  fprintf(stderr, "[dk_bench] Build opts: '%s'\n", build_opts);
  err = p_clBuildProgram(prog, 1, &dev, build_opts, nullptr, nullptr);
  if (err != CL_SUCCESS) {
    size_t log_sz = 0;
    p_clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_sz);
    std::vector<char> log(log_sz + 1, 0);
    p_clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, log_sz, log.data(), nullptr);
    fprintf(stderr, "BUILD ERROR:\n%s\n", log.data());
    return 1;
  }
  fprintf(stderr, "[dk_bench] Kernel compiled OK\n");

  // Check compiled binary size (delegate might produce different binary)
  {
    size_t bin_size = 0;
    p_clGetProgramInfo(prog, 0x1165 /*CL_PROGRAM_BINARY_SIZES*/, sizeof(size_t), &bin_size, nullptr);
    fprintf(stderr, "[dk_bench] Compiled binary size: %zu bytes\n", bin_size);
  }

  cl_kernel kernel = p_clCreateKernel(prog, "main_function", &err);
  if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: CreateKernel: %d\n", err); return 1; }

  // Create buffers/images
  // weights_buffer: __constant half8*, size = N * K * sizeof(half)
  size_t weight_bytes = (size_t)N * K * 2;  // half = 2 bytes
  // Delegate uses 0x14 = CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR.
  // On Adreno, ALLOC_HOST_PTR uses zero-copy memory with different caching.
  cl_mem weights_buf = p_clCreateBuffer(ctx, 0x14, weight_bytes, nullptr, &err);
  if (err) { fprintf(stderr, "ERROR: weights_buf: %d\n", err); return 1; }

  // xmem_buffer: wave memory scratch, 6144 bytes.
  // Delegate uses flags=0x4 (CL_MEM_WRITE_ONLY) — driver may place in special memory.
  cl_mem xmem_buf = p_clCreateBuffer(ctx, 0x4 /*CL_MEM_WRITE_ONLY*/, 6144, nullptr, &err);
  if (err) { fprintf(stderr, "ERROR: xmem_buf: %d\n", err); return 1; }

  // biases: image2d, (dst_slices, 1), RGBA half
  cl_image_format fmt_half = { CL_RGBA, CL_HALF_FLOAT };
  cl_image_desc bias_desc = {};
  bias_desc.image_type = CL_MEM_OBJECT_IMAGE2D;
  bias_desc.image_width = dst_slices;
  bias_desc.image_height = 1;
  cl_mem biases_img = p_clCreateImage(ctx, CL_MEM_READ_ONLY, &fmt_half, &bias_desc, nullptr, &err);
  if (err) { fprintf(stderr, "ERROR: biases_img: %d\n", err); return 1; }

  // src_tensor: image2d, (width=M, height=1*src_slices), RGBA half
  cl_image_desc src_desc = {};
  src_desc.image_type = CL_MEM_OBJECT_IMAGE2D;
  src_desc.image_width = M;
  src_desc.image_height = 1 * src_slices;  // Y * src_slices
  cl_mem src_img = p_clCreateImage(ctx, CL_MEM_READ_ONLY, &fmt_half, &src_desc, nullptr, &err);
  if (err) { fprintf(stderr, "ERROR: src_img: %d (w=%d h=%d)\n", err, M, src_slices); return 1; }

  // dst_tensor: image2d, (width=M, height=1*dst_slices), RGBA half
  cl_image_desc dst_desc = {};
  dst_desc.image_type = CL_MEM_OBJECT_IMAGE2D;
  dst_desc.image_width = M;
  dst_desc.image_height = 1 * dst_slices;
  cl_mem dst_img = p_clCreateImage(ctx, CL_MEM_WRITE_ONLY, &fmt_half, &dst_desc, nullptr, &err);
  if (err) { fprintf(stderr, "ERROR: dst_img: %d (w=%d h=%d)\n", err, M, dst_slices); return 1; }

  // Compute shared_int4 values from kernel analysis:
  // X >= shared_int4_0.z → X is width = M
  // Y >= shared_int4_0.x → Y is height = 1 (for matmul mapped as h=1,w=M)
  // Z*8 >= shared_int4_0.y → output slices
  // shared_int4_0.w = xmem entries per subgroup
  // shared_int4_1.x = weight stride factor
  // shared_int4_1.w = src_slices (K/4)
  // shared_int4_1.y = y_offset (0)
  // shared_int4_1.z = z_offset (0)
  // shared_int4_2.x = x_stride (1)
  // shared_int4_2.y = y_stride (1)

  // ========================================================================
  // Initialize data for correctness verification.
  // Weight layout in the kernel: for each inner loop iteration,
  //   qcom_sub_group_constant_load8 loads 32 half8 (=256 halfs) from
  //   weights_buffer at offset f_offset/2.  weights_cache is then
  //   indexed as half16[0..15], where each half16 holds 4 output channels
  //   × 4 components (one input channel each).
  //
  // For simplicity, fill weights with a constant (0.01h) and src with
  // 1.0h, bias with 0.  Then each output element ≈ K * 0.01 = 15.36
  // (modulo weight layout / accumulation order).
  // ========================================================================
  // ========================================================================
  // Full weight pipeline: program_000 → program_001 → program_002
  // ========================================================================
  fprintf(stderr, "[dk_bench] Running weight pipeline ...\n");

  std::string dir = std::string(cl_file);
  dir = dir.substr(0, dir.rfind('/'));
  std::string src_000 = ReadFile((dir + "/program_000.cl").c_str());
  std::string src_001 = ReadFile((dir + "/program_001.cl").c_str());

  auto compile_k = [&](const std::string& code, const char* nm,
                        const char* opts) -> cl_kernel {
    const char* p = code.c_str(); size_t l = code.size();
    cl_program pg = p_clCreateProgramWithSource(ctx, 1, &p, &l, &err);
    if (err) { fprintf(stderr, "  %s create: %d\n", nm, err); return nullptr; }
    err = p_clBuildProgram(pg, 1, &dev, opts, nullptr, nullptr);
    if (err) {
      size_t sz = 0;
      p_clGetProgramBuildInfo(pg, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &sz);
      std::vector<char> lg(sz + 1, 0);
      p_clGetProgramBuildInfo(pg, dev, CL_PROGRAM_BUILD_LOG, sz, lg.data(), nullptr);
      fprintf(stderr, "  %s build: %s\n", nm, lg.data());
      return nullptr;
    }
    return p_clCreateKernel(pg, "main_function", &err);
  };

  cl_kernel k000 = src_000.empty() ? nullptr : compile_k(src_000, "p000", "");
  cl_kernel k001 = src_001.empty() ? nullptr : compile_k(src_001, "p001", "");

  if (k000 && k001) {
    // Stage 0: int8 → packed uint4
    size_t int8_bytes = (size_t)N * K;
    cl_mem int8_buf = p_clCreateBuffer(ctx, 0x14, int8_bytes, nullptr, &err);
    {
      std::vector<int8_t> d(int8_bytes);
      for (size_t i = 0; i < int8_bytes; ++i)
        d[i] = (int8_t)((i * 7 + 3) % 11 - 5);
      p_clEnqueueWriteBuffer(queue, int8_buf, 1, 0, int8_bytes,
                             d.data(), 0, nullptr, nullptr);
    }
    // Captured: s0=(1536,589824,384,1536), s1=(1,6144,1,1)
    int pack_cnt = (N / 4) * (K / 4);
    int4 p0s0 = { dst_slices, pack_cnt, src_slices, dst_slices };
    int4 p0s1 = { 1, N, 1, 1 };
    p_clSetKernelArg(k000, 0, sizeof(cl_mem), &weights_buf);
    p_clSetKernelArg(k000, 1, sizeof(cl_mem), &int8_buf);
    p_clSetKernelArg(k000, 2, sizeof(int4), &p0s0);
    p_clSetKernelArg(k000, 3, sizeof(int4), &p0s1);
    size_t g0[] = { (size_t)pack_cnt, 1, 1 };
    size_t l0[] = { std::min((size_t)1024, (size_t)pack_cnt), 1, 1 };
    err = p_clEnqueueNDRangeKernel(queue, k000, 3, nullptr, g0, l0, 0, nullptr, nullptr);
    p_clFinish(queue);
    fprintf(stderr, "  prog_000: %s\n", err ? "FAIL" : "OK");

    // Stage 1: packed → dequant half
    // Need separate packed buffer (prog_000 output) and half buffer (prog_001 output)
    // Delegate: packed=34603008, half=18874368
    // For simplicity, use weights_buf as both (prog_000 writes uint4, prog_001 reads uint4 + writes half4)
    // But that overwrites! Need separate buffers.
    size_t packed_bytes = (size_t)pack_cnt * 16;
    cl_mem packed_buf = p_clCreateBuffer(ctx, CL_MEM_READ_ONLY, packed_bytes, nullptr, &err);
    // Copy pack result to packed_buf
    // Actually, prog_000 wrote to weights_buf as uint4. Let's re-run prog_000 into packed_buf.
    p_clSetKernelArg(k000, 0, sizeof(cl_mem), &packed_buf);
    err = p_clEnqueueNDRangeKernel(queue, k000, 3, nullptr, g0, l0, 0, nullptr, nullptr);
    p_clFinish(queue);

    // scale/zero_point images
    cl_image_desc sc_desc = {}; sc_desc.image_type = CL_MEM_OBJECT_IMAGE2D;
    sc_desc.image_width = dst_slices; sc_desc.image_height = 1;
    cl_mem sc_img = p_clCreateImage(ctx, CL_MEM_READ_ONLY, &fmt_half, &sc_desc, nullptr, &err);
    cl_mem zp_img = p_clCreateImage(ctx, CL_MEM_READ_ONLY, &fmt_half, &sc_desc, nullptr, &err);
    {
      uint16_t sc = f32_to_f16(0.01f), zp = f32_to_f16(0.0f);
      std::vector<uint16_t> sd(dst_slices * 4, sc), zd(dst_slices * 4, zp);
      size_t o[3] = {0,0,0}, r[3] = {(size_t)dst_slices, 1, 1};
      p_clEnqueueWriteImage(queue, sc_img, 1, o, r, 0, 0, sd.data(), 0, nullptr, nullptr);
      p_clEnqueueWriteImage(queue, zp_img, 1, o, r, 0, 0, zd.data(), 0, nullptr, nullptr);
    }

    // Captured: s0=(8,384,1536,1536), s1=(1,1,0,0)
    // NDRange: global=(3072,192,1) local=(64,8,1)
    int4 p1s0 = { 8, src_slices, dst_slices, dst_slices };
    int4 p1s1 = { 1, 1, 0, 0 };
    p_clSetKernelArg(k001, 0, sizeof(cl_mem), &weights_buf);
    p_clSetKernelArg(k001, 1, sizeof(cl_mem), &packed_buf);
    p_clSetKernelArg(k001, 2, sizeof(cl_mem), &sc_img);
    p_clSetKernelArg(k001, 3, sizeof(cl_mem), &zp_img);
    p_clSetKernelArg(k001, 4, sizeof(int4), &p1s0);
    p_clSetKernelArg(k001, 5, sizeof(int4), &p1s1);
    size_t g1[] = { 3072, 192, 1 };
    size_t l1[] = { 64, 8, 1 };
    err = p_clEnqueueNDRangeKernel(queue, k001, 3, nullptr, g1, l1, 0, nullptr, nullptr);
    p_clFinish(queue);
    fprintf(stderr, "  prog_001: %s\n", err ? "FAIL" : "OK");

    p_clReleaseMemObject(int8_buf);
    p_clReleaseMemObject(packed_buf);
    p_clReleaseMemObject(sc_img);
    p_clReleaseMemObject(zp_img);
    fprintf(stderr, "  Weight pipeline complete\n");
  } else {
    fprintf(stderr, "  Pipeline kernels not available, direct init\n");
    std::vector<uint16_t> wdata(weight_bytes / 2);
    for (size_t i = 0; i < wdata.size(); ++i)
      wdata[i] = f32_to_f16(0.005f + 0.01f * ((i * 7 + 13) % 100) / 100.0f);
    p_clEnqueueWriteBuffer(queue, weights_buf, 1, 0, weight_bytes,
                           wdata.data(), 0, nullptr, nullptr);
  }

  // Fill src image with 1.0h (RGBA half per pixel, all channels = 1.0)
  {
    uint16_t one = f32_to_f16(1.0f);
    size_t npixels = (size_t)M * src_slices;
    std::vector<uint16_t> sdata(npixels * 4, one);
    size_t origin[3] = {0, 0, 0};
    size_t region[3] = {(size_t)M, (size_t)src_slices, 1};
    p_clEnqueueWriteImage(queue, src_img, 1, origin, region, 0, 0,
                          sdata.data(), 0, nullptr, nullptr);
  }

  // Fill bias image with 0
  {
    std::vector<uint16_t> bdata(dst_slices * 4, 0);
    size_t origin[3] = {0, 0, 0};
    size_t region[3] = {(size_t)dst_slices, 1, 1};
    p_clEnqueueWriteImage(queue, biases_img, 1, origin, region, 0, 0,
                          bdata.data(), 0, nullptr, nullptr);
  }

  // Make dst readable for verification
  p_clReleaseMemObject(dst_img);
  dst_img = p_clCreateImage(ctx, CL_MEM_READ_WRITE, &fmt_half, &dst_desc, nullptr, &err);
  if (err) { fprintf(stderr, "ERROR: dst_img RW: %d\n", err); return 1; }

  p_clFinish(queue);

  // Captured from delegate via clSetKernelArg interception (1024x6144x1536):
  //   s0 = (1, 1536, 1024, 32)
  //   s1 = (384, 0, 0, 384)
  //   s2 = (1, 1, 0, 0)
  // Generalized:
  //   s0 = (height=1, dst_slices, width=M, xmem_per_sg=32)
  //   s1 = (src_slices, 0, 0, src_slices)
  //   s2 = (1, 1, 0, 0)
  int4 s0 = { 1, dst_slices, M, 32 };
  int4 s1 = { src_slices, 0, 0, src_slices };
  int4 s2 = { 1, 1, 0, 0 };

  // Set kernel args
  p_clSetKernelArg(kernel, 0, sizeof(cl_mem), &weights_buf);
  p_clSetKernelArg(kernel, 1, sizeof(cl_mem), &xmem_buf);
  p_clSetKernelArg(kernel, 2, sizeof(cl_mem), &biases_img);
  p_clSetKernelArg(kernel, 3, sizeof(cl_mem), &dst_img);
  p_clSetKernelArg(kernel, 4, sizeof(cl_mem), &src_img);
  p_clSetKernelArg(kernel, 5, sizeof(int4), &s0);
  p_clSetKernelArg(kernel, 6, sizeof(int4), &s1);
  p_clSetKernelArg(kernel, 7, sizeof(int4), &s2);

  // Captured from delegate: global=(6144,8,4) local=(128,1,4)
  // Kernel mapping: X=gid(1)*ls(0)+lid(0), Y=gid(2)*ls(1)+lid(1), Z=gid(0)*ls(2)+lid(2)
  //   dim0 → Z (output slice groups), dim1 → X (spatial), dim2 → Y (height)
  // For 1024x6144x1536: global=(6144,8,4) local=(128,1,4)
  //   dim0: 6144 = dst_slices*4 = 1536*4, local=128 → 48 groups
  //   dim1: 8, local=1 → 8 groups (X = gid(1)*128 + lid(0), so 8*128=1024=M)
  //   dim2: 4, local=4 → 1 group (Y = gid(2)*1 + lid(1))
  // Generalized:
  //   global[0] = dst_slices * 4
  //   global[1] = (M + 127) / 128
  //   global[2] = 4
  //   local = (128, 1, 4)
  // Correct dispatch for program_002 (conv kernel):
  //   global=(6144,8,4) local=(128,1,4) for 1024x6144x1536
  // Kernel mapping: X=gid(1)*ls(0)+lid(0), Y=gid(2)*ls(1)+lid(1), Z=gid(0)*ls(2)+lid(2)
  //   Z = gid(0)*4+lid(2), covers 0..191 → Z*8 covers all 1536 dst_slices
  //   X = gid(1)*128+lid(0), covers 0..1023 = M
  //   Y = 0 (height=1 for matmul)
  // Generalized:
  //   local = (128, 1, 4)
  //   global[0] = ceil(dst_slices/8 / 4) * 128  (48 groups * 128 = 6144)
  //   global[1] = ceil(M / 128)                  (8 groups)
  //   global[2] = 4
  size_t local[3] = { 128, 1, 4 };
  size_t groups_z = ((dst_slices + 7) / 8 + 3) / 4;  // ceil(dst_slices/8/4)
  size_t global[3] = {
    groups_z * 128,
    (size_t)((M + 127) / 128),
    4
  };
  const char* local_env = getenv("DK_LOCAL");
  if (local_env) {
    int l0, l1, l2;
    if (sscanf(local_env, "%d,%d,%d", &l0, &l1, &l2) == 3) {
      local[0] = l0; local[1] = l1; local[2] = l2;
    }
  }
  const char* global_env = getenv("DK_GLOBAL");
  if (global_env) {
    int g0, g1, g2;
    if (sscanf(global_env, "%d,%d,%d", &g0, &g1, &g2) == 3) {
      global[0] = g0; global[1] = g1; global[2] = g2;
    }
  }

  fprintf(stderr, "[dk_bench] global=(%zu,%zu,%zu) local=(%zu,%zu,%zu)\n",
          global[0], global[1], global[2], local[0], local[1], local[2]);
  fprintf(stderr, "[dk_bench] s0=(%d,%d,%d,%d) s1=(%d,%d,%d,%d) s2=(%d,%d,%d,%d)\n",
          s0.x, s0.y, s0.z, s0.w, s1.x, s1.y, s1.z, s1.w, s2.x, s2.y, s2.z, s2.w);

  // ========================================================================
  // Verification data: realistic random weights + src, CPU fp32 reference.
  // ========================================================================
  fprintf(stderr, "[dk_bench] Preparing verification data ...\n");

  // Generate random-ish data (deterministic seed for reproducibility)
  auto pseudo_rand = [](size_t i) -> float {
    uint32_t x = (uint32_t)(i * 2654435761u);
    x ^= x >> 16; x *= 0x45d9f3b; x ^= x >> 16;
    return (float)(x & 0xFFFF) / 65536.0f;  // [0, 1)
  };

  // Weights: range [-0.05, 0.05] (typical dequantized int8)
  std::vector<float> cpu_weights((size_t)N * K);
  {
    std::vector<uint16_t> wdata(weight_bytes / 2);
    for (size_t i = 0; i < cpu_weights.size(); ++i) {
      cpu_weights[i] = (pseudo_rand(i) - 0.5f) * 0.1f;
      wdata[i] = f32_to_f16(cpu_weights[i]);
    }
    p_clEnqueueWriteBuffer(queue, weights_buf, 1, 0, weight_bytes,
                           wdata.data(), 0, nullptr, nullptr);
  }

  // Src: range [0, 1] (typical activation)
  std::vector<float> cpu_src((size_t)M * K);
  {
    size_t npix = (size_t)M * src_slices;
    std::vector<uint16_t> sd(npix * 4);
    for (int x = 0; x < M; ++x) {
      for (int s = 0; s < src_slices; ++s) {
        for (int c = 0; c < 4; ++c) {
          int k = s * 4 + c;
          float v = (k < K) ? pseudo_rand(x * K + k + 999999) : 0.0f;
          cpu_src[x * K + std::min(k, K - 1)] = v;
          sd[((size_t)s * M + x) * 4 + c] = f32_to_f16(v);
        }
      }
    }
    size_t o[3] = {0,0,0}, r[3] = {(size_t)M, (size_t)src_slices, 1};
    p_clEnqueueWriteImage(queue, src_img, 1, o, r, 0, 0, sd.data(), 0, nullptr, nullptr);
  }

  // Bias: 0
  {
    std::vector<uint16_t> bd(dst_slices * 4, 0);
    size_t o[3] = {0,0,0}, r[3] = {(size_t)dst_slices, 1, 1};
    p_clEnqueueWriteImage(queue, biases_img, 1, o, r, 0, 0, bd.data(), 0, nullptr, nullptr);
  }
  p_clFinish(queue);

  // CPU fp32 reference using the ACTUAL weight buffer layout.
  //
  // From kernel analysis (program_002.cl):
  //   f_offset = Z * src_slices * 32 + iter * 64
  //   qcom_sub_group_constant_load8(xmem, weights_buf, c_off, f_offset>>1, 32)
  //   → loads 256 half values as weights_cache[0..15] (half16)
  //
  //   r_s[j] += src[k] * weights_cache[...][c*4+j]
  //
  // Weight W[out_ch, in_ch] is at buffer position (in half units):
  //   Z = out_ch / 32
  //   s = (out_ch / 4) % 8
  //   j = out_ch % 4
  //   iter = in_ch / 8
  //   k_local = in_ch % 8
  //   base = Z * src_slices * 128 + iter * 256
  //   if k_local < 4: idx = base + s*16 + k_local*4 + j
  //   if k_local >= 4: idx = base + (s+8)*16 + (k_local-4)*4 + j
  //
  // We read the weight buffer as half values and use this layout.
  fprintf(stderr, "  Computing CPU reference (kernel weight layout) ...\n");

  // Read back weight buffer as half values
  std::vector<uint16_t> wbuf_half(weight_bytes / 2);
  p_clEnqueueReadBuffer(queue, weights_buf, 1, 0, weight_bytes,
                        wbuf_half.data(), 0, nullptr, nullptr);

  int n_check_m = std::min(M, 4);
  int n_check_n = std::min(N, 32);
  std::vector<float> cpu_ref(n_check_m * n_check_n);

  for (int m = 0; m < n_check_m; ++m) {
    for (int out_ch = 0; out_ch < n_check_n; ++out_ch) {
      int Z = out_ch / 32;
      int s = (out_ch / 4) % 8;
      int j = out_ch % 4;
      double sum = 0;
      for (int k = 0; k < K; ++k) {
        int iter_idx = k / 8;
        int k_local = k % 8;
        size_t base = (size_t)Z * src_slices * 128 + (size_t)iter_idx * 256;
        size_t w_idx;
        if (k_local < 4)
          w_idx = base + s * 16 + k_local * 4 + j;
        else
          w_idx = base + (s + 8) * 16 + (k_local - 4) * 4 + j;
        float w = f16_to_f32(wbuf_half[w_idx]);
        sum += (double)cpu_src[m * K + k] * (double)w;
      }
      cpu_ref[m * n_check_n + out_ch] = (float)sum;
    }
  }
  fprintf(stderr, "  CPU ref[0,0]=%.4f ref[0,1]=%.4f ref[1,0]=%.4f\n",
          cpu_ref[0], cpu_ref[1], cpu_ref[n_check_n]);

  // GPU reference: run with (128,1,4) which we know is correct
  {
    size_t ref_local[3] = { 128, 1, 4 };
    size_t ref_groups_z = ((dst_slices + 7) / 8 + 3) / 4;
    size_t ref_global[3] = { ref_groups_z * 128, (size_t)((M + 127) / 128), 4 };
    p_clEnqueueNDRangeKernel(queue, kernel, 3, nullptr, ref_global, ref_local, 0, nullptr, nullptr);
    p_clFinish(queue);
  }
  // Read reference output
  size_t dst_npixels = (size_t)M * dst_slices;
  std::vector<uint16_t> gpu_ref_data(dst_npixels * 4, 0);
  {
    size_t o[3] = {0,0,0}, r[3] = {(size_t)M, (size_t)dst_slices, 1};
    p_clEnqueueReadImage(queue, dst_img, 1, o, r, 0, 0,
                         gpu_ref_data.data(), 0, nullptr, nullptr);
  }

  // Compare GPU reference vs CPU reference at sampled positions
  fprintf(stderr, "  GPU ref vs CPU ref (first few elements):\n");
  int gpu_cpu_match = 0, gpu_cpu_total = 0;
  for (int m = 0; m < n_check_m; ++m) {
    for (int n = 0; n < n_check_n; ++n) {
      int s = n / 4, c = n % 4;
      size_t idx = ((size_t)s * M + m) * 4 + c;
      float gpu_v = f16_to_f32(gpu_ref_data[idx]);
      float cpu_v = cpu_ref[m * n_check_n + n];
      float rel_err = (fabsf(cpu_v) > 1e-6f) ? fabsf(gpu_v - cpu_v) / fabsf(cpu_v) : fabsf(gpu_v - cpu_v);
      gpu_cpu_total++;
      if (rel_err < 0.15f) gpu_cpu_match++;
      if (m < 2 && n < 4)
        fprintf(stderr, "    [%d,%d] gpu=%.4f cpu=%.4f err=%.1f%%\n",
                m, n, gpu_v, cpu_v, rel_err * 100);
    }
  }
  fprintf(stderr, "  GPU-CPU match: %d/%d (within 15%%)\n", gpu_cpu_match, gpu_cpu_total);
  if (gpu_cpu_match < gpu_cpu_total / 2) {
    fprintf(stderr, "  WARNING: GPU ref doesn't match CPU. Weight layout may differ.\n");
    fprintf(stderr, "  Using GPU ref (128,1,4) as ground truth for tuning.\n");
  }

  // ========================================================================
  // Auto-tuning with correctness check: try local sizes, verify each,
  // only accept configurations that match GPU reference output.
  // ========================================================================
  // ========================================================================
  if (!getenv("DK_LOCAL")) {
    fprintf(stderr, "[dk_bench] Auto-tuning local sizes ...\n");
    struct TuneConfig { size_t l[3]; };
    // Candidate local sizes (must divide into global evenly)
    TuneConfig candidates[] = {
      {{128, 1, 4}},   // default
      {{64, 1, 4}},
      {{32, 1, 4}},
      {{64, 2, 4}},
      {{32, 2, 4}},
      {{128, 1, 2}},
      {{64, 1, 2}},
      {{32, 1, 2}},
      {{128, 1, 1}},
      {{64, 1, 1}},
      {{32, 1, 1}},
      {{256, 1, 2}},
      {{256, 1, 1}},
    };
    double best_us = 1e18;
    size_t best_local[3] = {128, 1, 4};
    size_t best_global[3] = {global[0], global[1], global[2]};
    int tune_iters = 10;

    for (auto& c : candidates) {
      // Recalculate global for this local.
      // Kernel mapping:
      //   X = get_group_id(1) * local_size(0) + get_local_id(0)  → need M values
      //   Y = get_group_id(2) * local_size(1) + get_local_id(1)  → need 1 value
      //   Z = get_group_id(0) * local_size(2) + get_local_id(2)  → need dst_slices/8 values
      size_t need_z = (dst_slices + 7) / 8;  // 192
      size_t groups_z = (need_z + c.l[2] - 1) / c.l[2];
      size_t groups_x = ((size_t)M + c.l[0] - 1) / c.l[0];
      size_t groups_y = (1 + c.l[1] - 1) / c.l[1];  // height=1
      size_t tg[3] = {
        groups_z * c.l[0],   // dim0: Z groups * local[0]
        groups_x * c.l[1],   // dim1: X groups * local[1]
        groups_y * c.l[2],   // dim2: Y groups * local[2]
      };

      // Quick warmup
      for (int i = 0; i < 3; ++i)
        p_clEnqueueNDRangeKernel(queue, kernel, 3, nullptr, tg, c.l, 0, nullptr, nullptr);
      p_clFinish(queue);

      // Time
      auto t0 = std::chrono::high_resolution_clock::now();
      for (int i = 0; i < tune_iters; ++i)
        p_clEnqueueNDRangeKernel(queue, kernel, 3, nullptr, tg, c.l, 0, nullptr, nullptr);
      p_clFinish(queue);
      auto t1 = std::chrono::high_resolution_clock::now();
      double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / tune_iters;

      // Quick correctness check: compare a few output pixels against reference
      std::vector<uint16_t> cand_data(dst_npixels * 4, 0);
      {
        size_t o[3] = {0,0,0}, r[3] = {(size_t)M, (size_t)dst_slices, 1};
        p_clEnqueueReadImage(queue, dst_img, 1, o, r, 0, 0,
                             cand_data.data(), 0, nullptr, nullptr);
      }
      int match = 0, check_total = 0;
      for (int i = 0; i < 64 && i < (int)(dst_npixels * 4); ++i) {
        float gv = f16_to_f32(cand_data[i]);
        float rv = f16_to_f32(gpu_ref_data[i]);
        float diff = fabsf(gv - rv);
        float tol = fabsf(rv) * 0.01f + 1e-4f;
        if (diff < tol) match++;
        check_total++;
      }
      bool correct = (match >= check_total * 9 / 10);  // 90% must match

      fprintf(stderr, "  local=(%zu,%zu,%zu) global=(%zu,%zu,%zu) → %.1f us (%.3f TFLOPS) %s\n",
              c.l[0], c.l[1], c.l[2], tg[0], tg[1], tg[2],
              us, (gflops / (us / 1e6)) / 1000.0,
              correct ? "✓" : "✗ WRONG OUTPUT");

      if (correct && us < best_us) {
        best_us = us;
        memcpy(best_local, c.l, sizeof(best_local));
        memcpy(best_global, tg, sizeof(best_global));
      }
    }
    memcpy(local, best_local, sizeof(local));
    memcpy(global, best_global, sizeof(global));
    fprintf(stderr, "[dk_bench] Best: local=(%zu,%zu,%zu) → %.1f us (%.3f TFLOPS)\n",
            local[0], local[1], local[2], best_us,
            (gflops / (best_us / 1e6)) / 1000.0);
  }

  fprintf(stderr, "[dk_bench] global=(%zu,%zu,%zu) local=(%zu,%zu,%zu)\n",
          global[0], global[1], global[2], local[0], local[1], local[2]);

  // Warmup
  fprintf(stderr, "[dk_bench] Warmup %d ...\n", warmup);
  for (int i = 0; i < warmup; ++i) {
    err = p_clEnqueueNDRangeKernel(queue, kernel, 3, nullptr, global, local, 0, nullptr, nullptr);
    if (err) { fprintf(stderr, "ERROR: dispatch: %d\n", err); return 1; }
  }
  p_clFinish(queue);

  // Wall-clock: burst-submit
  fprintf(stderr, "[dk_bench] Timed %d iters (burst) ...\n", iters);
  auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iters; ++i)
    p_clEnqueueNDRangeKernel(queue, kernel, 3, nullptr, global, local, 0, nullptr, nullptr);
  p_clFinish(queue);
  auto t1 = std::chrono::high_resolution_clock::now();
  double wall_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;

  // GPU profiling with all 4 timestamps
  // AND delegate-style batch profiling (multiple enqueue, single wait)
  double gpu_us = 0;
  double avg_queued_to_submit = 0, avg_submit_to_start = 0, avg_start_to_end = 0;
  double batch_avg_us = 0;
  double pipeline_avg_us = 0;
  {
    // === Single-dispatch profiling (our current method) ===
    std::vector<double> t_q2s, t_s2st, t_st2e;
    int prof_iters = std::min(iters, 30);
    for (int i = 0; i < prof_iters; ++i) {
      cl_event ev = nullptr;
      p_clEnqueueNDRangeKernel(prof_queue, kernel, 3, nullptr, global, local, 0, nullptr, &ev);
      p_clFinish(prof_queue);
      cl_ulong tq = 0, tsub = 0, tst = 0, te = 0;
      p_clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_QUEUED, sizeof(tq), &tq, nullptr);
      p_clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_SUBMIT, sizeof(tsub), &tsub, nullptr);
      p_clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(tst), &tst, nullptr);
      p_clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(te), &te, nullptr);
      p_clReleaseEvent(ev);
      t_q2s.push_back((tsub - tq) / 1000.0);
      t_s2st.push_back((tst - tsub) / 1000.0);
      t_st2e.push_back((te - tst) / 1000.0);
    }
    avg_queued_to_submit = std::accumulate(t_q2s.begin(), t_q2s.end(), 0.0) / t_q2s.size();
    avg_submit_to_start = std::accumulate(t_s2st.begin(), t_s2st.end(), 0.0) / t_s2st.size();
    avg_start_to_end = std::accumulate(t_st2e.begin(), t_st2e.end(), 0.0) / t_st2e.size();
    gpu_us = avg_start_to_end;

    // === Delegate-style batch profiling ===
    // Enqueue N kernels at once, attach event only to LAST one.
    // Total time / N = per-kernel average (cache-warm).
    // This matches delegate's ClarifyTimeMultipleEnqueue pattern.
    int batch_n = 5;
    std::vector<double> batch_times;
    for (int trial = 0; trial < prof_iters; ++trial) {
      // Dispatch batch_n kernels back-to-back, event on last only
      for (int b = 0; b < batch_n - 1; ++b)
        p_clEnqueueNDRangeKernel(prof_queue, kernel, 3, nullptr, global, local, 0, nullptr, nullptr);
      cl_event ev = nullptr;
      p_clEnqueueNDRangeKernel(prof_queue, kernel, 3, nullptr, global, local, 0, nullptr, &ev);
      p_clFinish(prof_queue);
      cl_ulong tst = 0, te = 0;
      p_clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(tst), &tst, nullptr);
      p_clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(te), &te, nullptr);
      p_clReleaseEvent(ev);
      batch_times.push_back((te - tst) / 1000.0);  // last kernel's time only
    }
    std::sort(batch_times.begin(), batch_times.end());
    batch_avg_us = batch_times[batch_times.size() / 2];  // median

    // === Pipeline-style profiling (delegate runs prog_001 before prog_002) ===
    // Simulate delegate's Run(): dispatch weight dequant kernel (k001) right
    // before the conv kernel (kernel), then measure conv kernel only.
    if (k001) {
      std::vector<double> pipe_times;
      size_t g1[] = { 3072, 192, 1 };
      size_t l1[] = { 64, 8, 1 };
      for (int i = 0; i < prof_iters; ++i) {
        // Dispatch weight dequant (no event, just warm GPU)
        p_clEnqueueNDRangeKernel(prof_queue, k001, 3, nullptr, g1, l1, 0, nullptr, nullptr);
        // Dispatch conv with event
        cl_event ev = nullptr;
        p_clEnqueueNDRangeKernel(prof_queue, kernel, 3, nullptr, global, local, 0, nullptr, &ev);
        p_clFinish(prof_queue);
        cl_ulong tst = 0, te = 0;
        p_clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(tst), &tst, nullptr);
        p_clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(te), &te, nullptr);
        p_clReleaseEvent(ev);
        pipe_times.push_back((te - tst) / 1000.0);
      }
      std::sort(pipe_times.begin(), pipe_times.end());
      pipeline_avg_us = pipe_times[pipe_times.size() / 2];
    }
  }

  double tflops_wall = (gflops / (wall_us / 1e6)) / 1000.0;
  double tflops_gpu = (gflops / (gpu_us / 1e6)) / 1000.0;
  double tflops_batch = (gflops / (batch_avg_us / 1e6)) / 1000.0;

  fprintf(stderr, "\n[dk_bench] Results: M=%d N=%d K=%d\n", M, N, K);
  fprintf(stderr, "  wall (burst):    %7.1f us  TFLOPS=%.3f\n", wall_us, tflops_wall);
  fprintf(stderr, "\n  Single-dispatch profiling:\n");
  fprintf(stderr, "    QUEUED→SUBMIT: %7.1f us\n", avg_queued_to_submit);
  fprintf(stderr, "    SUBMIT→START:  %7.1f us\n", avg_submit_to_start);
  fprintf(stderr, "    START→END:     %7.1f us  TFLOPS=%.3f\n", avg_start_to_end, tflops_gpu);
  fprintf(stderr, "\n  Batch profiling (delegate-style, last of %d):\n", 5);
  fprintf(stderr, "    START→END:     %7.1f us  TFLOPS=%.3f\n", batch_avg_us, tflops_batch);
  if (pipeline_avg_us > 0) {
    double tflops_pipe = (gflops / (pipeline_avg_us / 1e6)) / 1000.0;
    fprintf(stderr, "\n  Pipeline (prog_001 before conv):\n");
    fprintf(stderr, "    START→END:     %7.1f us  TFLOPS=%.3f\n", pipeline_avg_us, tflops_pipe);
  }
  fprintf(stderr, "\n  delegate ref:    %7.1f us  TFLOPS=5.2\n", 3691.0);
  fprintf(stderr, "  gpu:  %.1f us  TFLOPS=%.3f\n", gpu_us, tflops_gpu);
  fprintf(stderr, "  (delegate kernel_avg_us for reference: ~3720 us = 5.2 TFLOPS)\n");

  // ========================================================================
  // Command buffer timing: matches delegate's clCommandBufferKHR usage.
  // ========================================================================
  if (p_clCreateCommandBufferKHR && p_clCommandNDRangeKernelKHR &&
      p_clFinalizeCommandBufferKHR && p_clEnqueueCommandBufferKHR) {
    fprintf(stderr, "\n[dk_bench] Command buffer timing ...\n");

    void* cb = p_clCreateCommandBufferKHR(1, &queue, nullptr, &err);
    if (cb && err == CL_SUCCESS) {
      err = p_clCommandNDRangeKernelKHR(cb, nullptr, nullptr, kernel, 3,
                                         nullptr, global, local,
                                         0, nullptr, nullptr, nullptr);
      if (err == CL_SUCCESS) {
        err = p_clFinalizeCommandBufferKHR(cb);
      }
      if (err == CL_SUCCESS) {
        // Warmup
        for (int i = 0; i < warmup; ++i) {
          p_clEnqueueCommandBufferKHR(0, nullptr, cb, 0, nullptr, nullptr);
        }
        p_clFinish(queue);

        // Burst wall-clock
        auto cb_t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; ++i) {
          p_clEnqueueCommandBufferKHR(0, nullptr, cb, 0, nullptr, nullptr);
        }
        p_clFinish(queue);
        auto cb_t1 = std::chrono::high_resolution_clock::now();
        double cb_wall = std::chrono::duration<double, std::micro>(cb_t1 - cb_t0).count() / iters;

        // Per-dispatch GPU profiling
        double cb_gpu = 0;
        {
          std::vector<double> times;
          for (int i = 0; i < iters; ++i) {
            cl_event ev = nullptr;
            p_clEnqueueCommandBufferKHR(0, nullptr, cb, 0, nullptr, &ev);
            p_clFinish(queue);
            cl_ulong ts0 = 0, ts1 = 0;
            p_clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(ts0), &ts0, nullptr);
            p_clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(ts1), &ts1, nullptr);
            p_clReleaseEvent(ev);
            times.push_back((ts1 - ts0) / 1000.0);
          }
          cb_gpu = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
        }

        double cb_tflops_wall = (gflops / (cb_wall / 1e6)) / 1000.0;
        double cb_tflops_gpu = (gflops / (cb_gpu / 1e6)) / 1000.0;
        fprintf(stderr, "  CB wall: %.1f us  TFLOPS=%.3f\n", cb_wall, cb_tflops_wall);
        fprintf(stderr, "  CB gpu:  %.1f us  TFLOPS=%.3f\n", cb_gpu, cb_tflops_gpu);
      } else {
        fprintf(stderr, "  Command buffer setup failed: %d\n", err);
      }
      p_clReleaseCommandBufferKHR(cb);
    } else {
      fprintf(stderr, "  clCreateCommandBufferKHR failed: %d\n", err);
    }
  } else {
    fprintf(stderr, "\n[dk_bench] Command buffer not available\n");
  }

  // ========================================================================
  // Test 1: Qualcomm RecordableQueue (cl_qcom_recordable_queues)
  // ========================================================================
  if (p_clNewRecordingQCOM && p_clEndRecordingQCOM && p_clEnqueueRecordingQCOM) {
    fprintf(stderr, "\n[dk_bench] Qualcomm RecordableQueue test ...\n");
    // Create a recordable queue: CL_QUEUE_RECORDABLE_QCOM = 0x40E6 (from cl_ext_qcom.h)
    // Try multiple property combinations since the exact format is unknown
    cl_command_queue rec_queue = nullptr;
    // Attempt 1: recordable only (no profiling)
    uint64_t rq1[] = { 0x40E6, 1, 0 };
    if (p_clCreateCommandQueueWithProperties)
      rec_queue = p_clCreateCommandQueueWithProperties(ctx, dev, rq1, &err);
    if (!rec_queue || err) {
      fprintf(stderr, "  Attempt 1 (0x40E6 only): err=%d\n", err);
      // Attempt 2: as CL_QUEUE_PROPERTIES bit
      uint64_t rq2[] = { 0x1093, 0x40E6, 0 };
      rec_queue = p_clCreateCommandQueueWithProperties(ctx, dev, rq2, &err);
    }
    if (!rec_queue || err) {
      fprintf(stderr, "  Attempt 2 (as PROPERTIES bit): err=%d\n", err);
      // Attempt 3: plain queue then try recording on it
      rec_queue = p_clCreateCommandQueue(ctx, dev, 0, &err);
      fprintf(stderr, "  Attempt 3 (plain queue): err=%d\n", err);
    }
    if (rec_queue && err == CL_SUCCESS) {
      fprintf(stderr, "  Recordable queue created\n");
      // Record the kernel
      void* recording = p_clNewRecordingQCOM(rec_queue, &err);
      if (recording && err == CL_SUCCESS) {
        p_clEnqueueNDRangeKernel(rec_queue, kernel, 3, nullptr, global, local, 0, nullptr, nullptr);
        err = p_clEndRecordingQCOM(recording);
        if (err == CL_SUCCESS) {
          fprintf(stderr, "  Recording created\n");
          // Warmup replay
          for (int i = 0; i < warmup; ++i) {
            p_clEnqueueRecordingQCOM(prof_queue, recording, 0, nullptr, 0, nullptr, nullptr);
          }
          p_clFinish(prof_queue);
          // Timed replay (burst wall-clock)
          auto rt0 = std::chrono::high_resolution_clock::now();
          for (int i = 0; i < iters; ++i) {
            p_clEnqueueRecordingQCOM(prof_queue, recording, 0, nullptr, 0, nullptr, nullptr);
          }
          p_clFinish(prof_queue);
          auto rt1 = std::chrono::high_resolution_clock::now();
          double rq_wall = std::chrono::duration<double, std::micro>(rt1 - rt0).count() / iters;
          // Per-dispatch GPU profiling
          double rq_gpu = 0;
          {
            std::vector<double> times;
            for (int i = 0; i < 30; ++i) {
              cl_event ev = nullptr;
              p_clEnqueueRecordingQCOM(prof_queue, recording, 0, nullptr, 0, nullptr, &ev);
              p_clFinish(prof_queue);
              cl_ulong ts0 = 0, ts1 = 0;
              p_clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(ts0), &ts0, nullptr);
              p_clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(ts1), &ts1, nullptr);
              p_clReleaseEvent(ev);
              times.push_back((ts1 - ts0) / 1000.0);
            }
            std::sort(times.begin(), times.end());
            rq_gpu = times[times.size() / 2];
          }
          fprintf(stderr, "  RQ wall: %.1f us  TFLOPS=%.3f\n", rq_wall, (gflops / (rq_wall / 1e6)) / 1000.0);
          fprintf(stderr, "  RQ gpu:  %.1f us  TFLOPS=%.3f\n", rq_gpu, (gflops / (rq_gpu / 1e6)) / 1000.0);
        } else {
          fprintf(stderr, "  EndRecording failed: %d\n", err);
        }
        p_clReleaseRecordingQCOM(recording);
      } else {
        fprintf(stderr, "  NewRecording failed: %d\n", err);
      }
    } else {
      fprintf(stderr, "  Recordable queue creation failed: %d\n", err ? err : -999);
    }
  } else {
    fprintf(stderr, "\n[dk_bench] RecordableQueue not available\n");
  }

  // ========================================================================
  // Test 2: Multi-kernel pipeline (simulate delegate's full Run())
  // Dispatch prog_001 + prog_002 in sequence N times, measure amortized
  // per-conv time as total_wall / N.
  // ========================================================================
  if (k001) {
    fprintf(stderr, "\n[dk_bench] Multi-kernel pipeline test ...\n");
    size_t g1[] = { 3072, 192, 1 };
    size_t l1[] = { 64, 8, 1 };
    // Warmup
    for (int i = 0; i < 10; ++i) {
      p_clEnqueueNDRangeKernel(queue, k001, 3, nullptr, g1, l1, 0, nullptr, nullptr);
      p_clEnqueueNDRangeKernel(queue, kernel, 3, nullptr, global, local, 0, nullptr, nullptr);
    }
    p_clFinish(queue);
    // Timed: dispatch pairs, measure total wall / N
    int pipe_n = 50;
    auto pt0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < pipe_n; ++i) {
      p_clEnqueueNDRangeKernel(queue, k001, 3, nullptr, g1, l1, 0, nullptr, nullptr);
      p_clEnqueueNDRangeKernel(queue, kernel, 3, nullptr, global, local, 0, nullptr, nullptr);
    }
    p_clFinish(queue);
    auto pt1 = std::chrono::high_resolution_clock::now();
    double pipe_total = std::chrono::duration<double, std::micro>(pt1 - pt0).count();
    double pipe_per_conv = pipe_total / pipe_n;  // amortized per-conv (includes 001 time)
    // Also measure just conv alone in a burst for comparison
    auto pt2 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < pipe_n; ++i)
      p_clEnqueueNDRangeKernel(queue, kernel, 3, nullptr, global, local, 0, nullptr, nullptr);
    p_clFinish(queue);
    auto pt3 = std::chrono::high_resolution_clock::now();
    double conv_only = std::chrono::duration<double, std::micro>(pt3 - pt2).count() / pipe_n;

    fprintf(stderr, "  Pipeline (001+002) total/N: %.1f us per pair\n", pipe_per_conv);
    fprintf(stderr, "  Conv-only burst wall/N:     %.1f us  TFLOPS=%.3f\n",
            conv_only, (gflops / (conv_only / 1e6)) / 1000.0);
    fprintf(stderr, "  delegate ref:               %.1f us  TFLOPS=5.2\n", 3691.0);
  }

  // ========================================================================
  // Correctness verification: read back dst image and check values.
  // ========================================================================
  // Final verification: run best config, compare vs GPU ref and CPU ref.
  // ========================================================================
  fprintf(stderr, "\n[dk_bench] Final verification with best config ...\n");
  {
    p_clEnqueueNDRangeKernel(queue, kernel, 3, nullptr, global, local, 0, nullptr, nullptr);
    p_clFinish(queue);

    std::vector<uint16_t> final_data(dst_npixels * 4, 0);
    {
      size_t o[3] = {0,0,0}, r[3] = {(size_t)M, (size_t)dst_slices, 1};
      p_clEnqueueReadImage(queue, dst_img, 1, o, r, 0, 0,
                           final_data.data(), 0, nullptr, nullptr);
    }

    // Compare vs GPU reference (128,1,4)
    int match = 0, total = 0;
    float max_rel_err = 0;
    for (size_t i = 0; i < dst_npixels * 4 && i < 1000; ++i) {
      float gv = f16_to_f32(final_data[i]);
      float rv = f16_to_f32(gpu_ref_data[i]);
      float diff = fabsf(gv - rv);
      float tol = fabsf(rv) * 0.01f + 1e-4f;
      float rel = (fabsf(rv) > 1e-6f) ? diff / fabsf(rv) : diff;
      if (rel > max_rel_err) max_rel_err = rel;
      if (diff < tol) match++;
      total++;
    }
    fprintf(stderr, "  vs GPU ref (128,1,4): %d/%d match (max rel err %.2f%%)\n",
            match, total, max_rel_err * 100);

    // Compare vs CPU fp32 reference (layout-aware)
    fprintf(stderr, "  vs CPU fp32 ref (layout-aware):\n");
    int cpu_match = 0, cpu_total = 0;
    float cpu_max_err = 0;
    for (int m = 0; m < n_check_m; ++m) {
      for (int out_ch = 0; out_ch < n_check_n; ++out_ch) {
        int s = out_ch / 4, c = out_ch % 4;
        size_t idx = ((size_t)s * M + m) * 4 + c;
        float gv = f16_to_f32(final_data[idx]);
        float cv = cpu_ref[m * n_check_n + out_ch];
        float rel = (fabsf(cv) > 1e-6f) ? fabsf(gv - cv) / fabsf(cv) : fabsf(gv - cv);
        cpu_total++;
        if (rel < 0.15f) cpu_match++;
        if (rel > cpu_max_err) cpu_max_err = rel;
        if (m < 2 && out_ch < 8)
          fprintf(stderr, "    [m=%d,n=%d] gpu=%.6f cpu=%.6f err=%.1f%%\n",
                  m, out_ch, gv, cv, rel * 100);
      }
    }
    fprintf(stderr, "  CPU match: %d/%d (max err %.1f%%)\n",
            cpu_match, cpu_total, cpu_max_err * 100);

    if (match >= total * 9 / 10)
      fprintf(stderr, "  ✓ PASS: output matches reference\n");
    else
      fprintf(stderr, "  ✗ FAIL: output does NOT match reference\n");
  }
  fprintf(stderr, "\n");

  // Cleanup
  p_clReleaseMemObject(weights_buf);
  p_clReleaseMemObject(xmem_buf);
  p_clReleaseMemObject(biases_img);
  p_clReleaseMemObject(src_img);
  p_clReleaseMemObject(dst_img);
  p_clReleaseKernel(kernel);
  p_clReleaseProgram(prog);
  p_clReleaseCommandQueue(queue);
  p_clReleaseCommandQueue(prof_queue);
  p_clReleaseContext(ctx);
  return 0;
}
