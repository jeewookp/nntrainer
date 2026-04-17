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
DECL_CL(cl_mem, clCreateBuffer, (cl_context, cl_bitfield, size_t, void*, cl_int*))
DECL_CL(cl_mem, clCreateImage, (cl_context, cl_bitfield, const cl_image_format*, const cl_image_desc*, void*, cl_int*))
DECL_CL(cl_program, clCreateProgramWithSource, (cl_context, cl_uint, const char**, const size_t*, cl_int*))
DECL_CL(cl_int, clBuildProgram, (cl_program, cl_uint, const cl_device_id*, const char*, void*, void*))
DECL_CL(cl_int, clGetProgramBuildInfo, (cl_program, cl_device_id, cl_uint, size_t, void*, size_t*))
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

#define LOAD(h, name) p_##name = (pfn_##name)dlsym(h, #name); \
  if (!p_##name) { fprintf(stderr, "WARN: %s not found\n", #name); }

static bool LoadCL() {
  void* h = dlopen("libOpenCL.so", RTLD_NOW);
  if (!h) h = dlopen("/system/vendor/lib64/libOpenCL.so", RTLD_NOW);
  if (!h) { fprintf(stderr, "Cannot load libOpenCL.so\n"); return false; }
  LOAD(h, clGetPlatformIDs); LOAD(h, clGetPlatformInfo);
  LOAD(h, clGetDeviceIDs); LOAD(h, clGetDeviceInfo);
  LOAD(h, clCreateContext); LOAD(h, clCreateCommandQueue);
  LOAD(h, clCreateBuffer); LOAD(h, clCreateImage);
  LOAD(h, clCreateProgramWithSource); LOAD(h, clBuildProgram);
  LOAD(h, clGetProgramBuildInfo); LOAD(h, clCreateKernel);
  LOAD(h, clSetKernelArg); LOAD(h, clEnqueueNDRangeKernel);
  LOAD(h, clFinish); LOAD(h, clReleaseMemObject);
  LOAD(h, clReleaseKernel); LOAD(h, clReleaseProgram);
  LOAD(h, clReleaseCommandQueue); LOAD(h, clReleaseContext);
  LOAD(h, clReleaseEvent); LOAD(h, clGetEventProfilingInfo);
  LOAD(h, clWaitForEvents); LOAD(h, clEnqueueWriteBuffer);
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

int main(int argc, char** argv) {
  // Parse args
  int M = 1024, N = 6144, K = 1536;
  int warmup = 5, iters = 50;
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

  cl_int err;
  cl_context ctx = p_clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &err);
  cl_command_queue queue = p_clCreateCommandQueue(ctx, dev, 0, &err);
  cl_command_queue prof_queue = p_clCreateCommandQueue(ctx, dev, CL_QUEUE_PROFILING_ENABLE, &err);

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

  cl_kernel kernel = p_clCreateKernel(prog, "main_function", &err);
  if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: CreateKernel: %d\n", err); return 1; }

  // Create buffers/images
  // weights_buffer: __constant half8*, size = N * K * sizeof(half)
  size_t weight_bytes = (size_t)N * K * 2;  // half = 2 bytes
  cl_mem weights_buf = p_clCreateBuffer(ctx, CL_MEM_READ_ONLY, weight_bytes, nullptr, &err);
  if (err) { fprintf(stderr, "ERROR: weights_buf: %d\n", err); return 1; }

  // xmem_buffer: __constant half8*, max_constant_size 6144
  cl_mem xmem_buf = p_clCreateBuffer(ctx, CL_MEM_READ_ONLY, 6144, nullptr, &err);
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

  // GPU profiling
  double gpu_us = 0;
  {
    std::vector<double> times;
    for (int i = 0; i < iters; ++i) {
      cl_event ev = nullptr;
      p_clEnqueueNDRangeKernel(prof_queue, kernel, 3, nullptr, global, local, 0, nullptr, &ev);
      p_clFinish(prof_queue);
      cl_ulong ts0 = 0, ts1 = 0;
      p_clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(ts0), &ts0, nullptr);
      p_clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(ts1), &ts1, nullptr);
      p_clReleaseEvent(ev);
      times.push_back((ts1 - ts0) / 1000.0);
    }
    gpu_us = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
  }

  double tflops_wall = (gflops / (wall_us / 1e6)) / 1000.0;
  double tflops_gpu = (gflops / (gpu_us / 1e6)) / 1000.0;

  fprintf(stderr, "\n[dk_bench] Results: M=%d N=%d K=%d\n", M, N, K);
  fprintf(stderr, "  wall: %.1f us  TFLOPS=%.3f\n", wall_us, tflops_wall);
  fprintf(stderr, "  gpu:  %.1f us  TFLOPS=%.3f\n", gpu_us, tflops_gpu);
  fprintf(stderr, "  (delegate kernel_avg_us for reference: ~3720 us = 5.2 TFLOPS)\n\n");

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
