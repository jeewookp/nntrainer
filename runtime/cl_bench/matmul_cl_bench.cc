// Direct OpenCL matmul micro benchmark.
//
// Bypasses LiteRT entirely: dlopen's libOpenCL.so, builds a tiled GEMM
// kernel, and measures raw GPU throughput. This gives a clean wall-clock
// TFLOPS number without LiteRT's per-Run() overhead (~8.5ms framework +
// 2.1ms upload + 0.4ms weights_convert).

#include <dlfcn.h>
#include <stdint.h>
#include <string.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

// ============================================================================
// OpenCL type definitions (avoid dependency on CL headers)
// ============================================================================
typedef int32_t cl_int;
typedef uint32_t cl_uint;
typedef uint64_t cl_ulong;
typedef uint64_t cl_bitfield;
typedef cl_bitfield cl_device_type;
typedef cl_bitfield cl_mem_flags;
typedef cl_bitfield cl_command_queue_properties;
typedef cl_uint cl_device_info;
typedef cl_uint cl_platform_info;
typedef cl_uint cl_program_build_info;
typedef cl_uint cl_event_info;
typedef cl_uint cl_profiling_info;

typedef void* cl_platform_id;
typedef void* cl_device_id;
typedef void* cl_context;
typedef void* cl_command_queue;
typedef void* cl_mem;
typedef void* cl_program;
typedef void* cl_kernel;
typedef void* cl_event;

// CL constants
#define CL_SUCCESS 0
#define CL_DEVICE_TYPE_GPU (1 << 2)
#define CL_PLATFORM_NAME 0x0902
#define CL_DEVICE_NAME 0x102B
#define CL_DEVICE_MAX_COMPUTE_UNITS 0x1002
#define CL_DEVICE_MAX_CLOCK_FREQUENCY 0x100C
#define CL_DEVICE_GLOBAL_MEM_SIZE 0x101F
#define CL_MEM_READ_WRITE (1 << 0)
#define CL_MEM_READ_ONLY (1 << 2)
#define CL_MEM_WRITE_ONLY (1 << 1)
#define CL_MEM_COPY_HOST_PTR (1 << 5)
#define CL_PROGRAM_BUILD_LOG 0x1183
#define CL_QUEUE_PROFILING_ENABLE (1 << 1)
#define CL_PROFILING_COMMAND_START 0x1282
#define CL_PROFILING_COMMAND_END 0x1283
#define CL_EVENT_COMMAND_EXECUTION_STATUS 0x11D3
#define CL_COMPLETE 0x0

// ============================================================================
// CL function pointer types
// ============================================================================
#define CL_FUNC(ret, name, args) typedef ret(*PFN_##name) args;

CL_FUNC(cl_int, clGetPlatformIDs, (cl_uint, cl_platform_id*, cl_uint*))
CL_FUNC(cl_int, clGetPlatformInfo, (cl_platform_id, cl_platform_info, size_t, void*, size_t*))
CL_FUNC(cl_int, clGetDeviceIDs, (cl_platform_id, cl_device_type, cl_uint, cl_device_id*, cl_uint*))
CL_FUNC(cl_int, clGetDeviceInfo, (cl_device_id, cl_device_info, size_t, void*, size_t*))
CL_FUNC(cl_context, clCreateContext, (const void*, cl_uint, const cl_device_id*, void(*)(const char*, const void*, size_t, void*), void*, cl_int*))
CL_FUNC(cl_command_queue, clCreateCommandQueue, (cl_context, cl_device_id, cl_command_queue_properties, cl_int*))
CL_FUNC(cl_mem, clCreateBuffer, (cl_context, cl_mem_flags, size_t, void*, cl_int*))
CL_FUNC(cl_program, clCreateProgramWithSource, (cl_context, cl_uint, const char**, const size_t*, cl_int*))
CL_FUNC(cl_int, clBuildProgram, (cl_program, cl_uint, const cl_device_id*, const char*, void(*)(cl_program, void*), void*))
CL_FUNC(cl_int, clGetProgramBuildInfo, (cl_program, cl_device_id, cl_program_build_info, size_t, void*, size_t*))
CL_FUNC(cl_kernel, clCreateKernel, (cl_program, const char*, cl_int*))
CL_FUNC(cl_int, clSetKernelArg, (cl_kernel, cl_uint, size_t, const void*))
CL_FUNC(cl_int, clEnqueueNDRangeKernel, (cl_command_queue, cl_kernel, cl_uint, const size_t*, const size_t*, const size_t*, cl_uint, const cl_event*, cl_event*))
CL_FUNC(cl_int, clEnqueueWriteBuffer, (cl_command_queue, cl_mem, cl_uint, size_t, size_t, const void*, cl_uint, const cl_event*, cl_event*))
CL_FUNC(cl_int, clEnqueueReadBuffer, (cl_command_queue, cl_mem, cl_uint, size_t, size_t, void*, cl_uint, const cl_event*, cl_event*))
CL_FUNC(cl_int, clFinish, (cl_command_queue))
CL_FUNC(cl_int, clFlush, (cl_command_queue))
CL_FUNC(cl_int, clReleaseMemObject, (cl_mem))
CL_FUNC(cl_int, clReleaseKernel, (cl_kernel))
CL_FUNC(cl_int, clReleaseProgram, (cl_program))
CL_FUNC(cl_int, clReleaseCommandQueue, (cl_command_queue))
CL_FUNC(cl_int, clReleaseContext, (cl_context))
CL_FUNC(cl_int, clReleaseEvent, (cl_event))
CL_FUNC(cl_int, clWaitForEvents, (cl_uint, const cl_event*))
CL_FUNC(cl_int, clGetEventProfilingInfo, (cl_event, cl_profiling_info, size_t, void*, size_t*))

// ============================================================================
// Global CL function pointers
// ============================================================================
struct CLFunctions {
  PFN_clGetPlatformIDs GetPlatformIDs;
  PFN_clGetPlatformInfo GetPlatformInfo;
  PFN_clGetDeviceIDs GetDeviceIDs;
  PFN_clGetDeviceInfo GetDeviceInfo;
  PFN_clCreateContext CreateContext;
  PFN_clCreateCommandQueue CreateCommandQueue;
  PFN_clCreateBuffer CreateBuffer;
  PFN_clCreateProgramWithSource CreateProgramWithSource;
  PFN_clBuildProgram BuildProgram;
  PFN_clGetProgramBuildInfo GetProgramBuildInfo;
  PFN_clCreateKernel CreateKernel;
  PFN_clSetKernelArg SetKernelArg;
  PFN_clEnqueueNDRangeKernel EnqueueNDRangeKernel;
  PFN_clEnqueueWriteBuffer EnqueueWriteBuffer;
  PFN_clEnqueueReadBuffer EnqueueReadBuffer;
  PFN_clFinish Finish;
  PFN_clFlush Flush;
  PFN_clReleaseMemObject ReleaseMemObject;
  PFN_clReleaseKernel ReleaseKernel;
  PFN_clReleaseProgram ReleaseProgram;
  PFN_clReleaseCommandQueue ReleaseCommandQueue;
  PFN_clReleaseContext ReleaseContext;
  PFN_clReleaseEvent ReleaseEvent;
  PFN_clWaitForEvents WaitForEvents;
  PFN_clGetEventProfilingInfo GetEventProfilingInfo;
};

static CLFunctions cl;

#define LOAD_CL(handle, name)                                        \
  cl.name = reinterpret_cast<PFN_cl##name>(dlsym(handle, "cl" #name)); \
  if (!cl.name) {                                                    \
    fprintf(stderr, "ERROR: dlsym(cl%s) failed: %s\n", #name,       \
            dlerror());                                              \
    return false;                                                    \
  }

static bool LoadOpenCL() {
  const char* paths[] = {
      "libOpenCL.so",
      "/system/vendor/lib64/libOpenCL.so",
      "/system/lib64/libOpenCL.so",
      "/vendor/lib64/libOpenCL.so",
      nullptr,
  };
  void* handle = nullptr;
  for (int i = 0; paths[i]; ++i) {
    handle = dlopen(paths[i], RTLD_NOW | RTLD_LOCAL);
    if (handle) {
      fprintf(stderr, "[cl_bench] Loaded %s\n", paths[i]);
      break;
    }
  }
  if (!handle) {
    fprintf(stderr, "ERROR: Could not dlopen libOpenCL.so\n");
    return false;
  }

  LOAD_CL(handle, GetPlatformIDs);
  LOAD_CL(handle, GetPlatformInfo);
  LOAD_CL(handle, GetDeviceIDs);
  LOAD_CL(handle, GetDeviceInfo);
  LOAD_CL(handle, CreateContext);
  LOAD_CL(handle, CreateCommandQueue);
  LOAD_CL(handle, CreateBuffer);
  LOAD_CL(handle, CreateProgramWithSource);
  LOAD_CL(handle, BuildProgram);
  LOAD_CL(handle, GetProgramBuildInfo);
  LOAD_CL(handle, CreateKernel);
  LOAD_CL(handle, SetKernelArg);
  LOAD_CL(handle, EnqueueNDRangeKernel);
  LOAD_CL(handle, EnqueueWriteBuffer);
  LOAD_CL(handle, EnqueueReadBuffer);
  LOAD_CL(handle, Finish);
  LOAD_CL(handle, Flush);
  LOAD_CL(handle, ReleaseMemObject);
  LOAD_CL(handle, ReleaseKernel);
  LOAD_CL(handle, ReleaseProgram);
  LOAD_CL(handle, ReleaseCommandQueue);
  LOAD_CL(handle, ReleaseContext);
  LOAD_CL(handle, ReleaseEvent);
  LOAD_CL(handle, WaitForEvents);
  LOAD_CL(handle, GetEventProfilingInfo);

  return true;
}

// ============================================================================
// Tiled GEMM kernel: C[M,N] = A[M,K] * B[K,N]
// Row-major, fp16 compute with fp16 I/O (matches Adreno preference).
// TILE_M x TILE_N workgroup, each thread computes one output element.
// ============================================================================
static const char* kGemmKernelFp16 = R"(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void gemm_fp16(
    __global const half* restrict A,
    __global const half* restrict B,
    __global half* restrict C,
    const int M, const int N, const int K) {
  const int row = get_global_id(0);
  const int col = get_global_id(1);
  if (row >= M || col >= N) return;

  half acc = 0.0h;
  for (int k = 0; k < K; ++k) {
    acc += A[row * K + k] * B[k * N + col];
  }
  C[row * N + col] = acc;
}
)";

// Tiled version with local memory for better cache behavior on Adreno.
static const char* kGemmKernelTiledFp16 = R"(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define TS 16
__kernel void gemm_tiled_fp16(
    __global const half* restrict A,
    __global const half* restrict B,
    __global half* restrict C,
    const int M, const int N, const int K) {
  const int row = get_local_id(0);
  const int col = get_local_id(1);
  const int globalRow = TS * get_group_id(0) + row;
  const int globalCol = TS * get_group_id(1) + col;

  __local half Asub[TS][TS];
  __local half Bsub[TS][TS];

  half acc = 0.0h;
  const int numTiles = (K + TS - 1) / TS;
  for (int t = 0; t < numTiles; ++t) {
    const int tiledRow = TS * t + col;
    const int tiledCol = TS * t + row;
    Asub[row][col] = (globalRow < M && tiledRow < K) ?
                     A[globalRow * K + tiledRow] : 0.0h;
    Bsub[row][col] = (tiledCol < K && globalCol < N) ?
                     B[tiledCol * N + globalCol] : 0.0h;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int k = 0; k < TS; ++k) {
      acc += Asub[row][k] * Bsub[k][col];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (globalRow < M && globalCol < N) {
    C[globalRow * N + globalCol] = acc;
  }
}
)";

// Wave-memory GEMM: uses cl_khr_subgroups to broadcast weights across the
// wave (subgroup). On Adreno 830, wave_size=64 for fp16.
//
// Each wave computes a WG_M × WG_N tile of C.
// - Threads in the wave are mapped as (WG_M/TM) × (WG_N/TN)
// - Each thread computes a TM×TN sub-tile of C
// - Inner K loop: each thread loads TM A values + TN B values
//   and broadcasts to all via sub_group_broadcast
// - Accumulate in fp32 for precision
//
// Work group: 1D with wave_size threads (Adreno requires this for subgroups)
// Global: (ceil(N/WG_N), ceil(M/WG_M)) * wave_size
static const char* kGemmKernelWaveFp16 = R"(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable

#define WAVE 64
#define TM 4
#define TN 4
#define ROWS_PER_WAVE 8
#define COLS_PER_WAVE 8

__attribute__((reqd_work_group_size(WAVE, 1, 1)))
__kernel void gemm_wave_fp16(
    __global const half* restrict A,
    __global const half* restrict B,
    __global half* restrict C,
    const int M, const int N, const int K) {

  const int wg_row = get_group_id(1);
  const int wg_col = get_group_id(0);
  const int lane = get_sub_group_local_id();

  const int lane_row = lane / COLS_PER_WAVE;
  const int lane_col = lane % COLS_PER_WAVE;

  const int m_base = wg_row * (ROWS_PER_WAVE * TM) + lane_row * TM;
  const int n_base = wg_col * (COLS_PER_WAVE * TN) + lane_col * TN;

  float acc[TM][TN];
  for (int i = 0; i < TM; ++i)
    for (int j = 0; j < TN; ++j)
      acc[i][j] = 0.0f;

  for (int k = 0; k < K; ++k) {
    half a_priv[TM];
    for (int i = 0; i < TM; ++i) {
      int row = m_base + i;
      a_priv[i] = (row < M) ? A[row * K + k] : 0.0h;
    }

    half b_priv[TN];
    for (int j = 0; j < TN; ++j) {
      int col = n_base + j;
      b_priv[j] = (col < N) ? B[k * N + col] : 0.0h;
    }

    for (int i = 0; i < TM; ++i) {
      float a_val = (float)a_priv[i];
      for (int j = 0; j < TN; ++j) {
        acc[i][j] += a_val * (float)b_priv[j];
      }
    }
  }

  for (int i = 0; i < TM; ++i) {
    int row = m_base + i;
    if (row >= M) continue;
    for (int j = 0; j < TN; ++j) {
      int col = n_base + j;
      if (col < N)
        C[row * N + col] = (half)acc[i][j];
    }
  }
}
)";

// fp32 versions as fallback if fp16 is not supported.
static const char* kGemmKernelTiledFp32 = R"(
#define TS 16
__kernel void gemm_tiled_fp32(
    __global const float* restrict A,
    __global const float* restrict B,
    __global float* restrict C,
    const int M, const int N, const int K) {
  const int row = get_local_id(0);
  const int col = get_local_id(1);
  const int globalRow = TS * get_group_id(0) + row;
  const int globalCol = TS * get_group_id(1) + col;

  __local float Asub[TS][TS];
  __local float Bsub[TS][TS];

  float acc = 0.0f;
  const int numTiles = (K + TS - 1) / TS;
  for (int t = 0; t < numTiles; ++t) {
    const int tiledRow = TS * t + col;
    const int tiledCol = TS * t + row;
    Asub[row][col] = (globalRow < M && tiledRow < K) ?
                     A[globalRow * K + tiledRow] : 0.0f;
    Bsub[row][col] = (tiledCol < K && globalCol < N) ?
                     B[tiledCol * N + globalCol] : 0.0f;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int k = 0; k < TS; ++k) {
      acc += Asub[row][k] * Bsub[k][col];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (globalRow < M && globalCol < N) {
    C[globalRow * N + globalCol] = acc;
  }
}
)";

// ============================================================================
// Shape parsing
// ============================================================================
struct Shape {
  int m, n, k;
  double gflops() const { return 2.0 * m * n * k / 1e9; }
};

static std::vector<Shape> ParseShapes(const std::string& shapes_str) {
  std::vector<Shape> result;
  std::stringstream ss(shapes_str);
  std::string token;
  while (std::getline(ss, token, ',')) {
    int m = 0, n = 0, k = 0;
    if (sscanf(token.c_str(), "%dx%dx%d", &m, &n, &k) == 3 &&
        m > 0 && n > 0 && k > 0) {
      result.push_back({m, n, k});
    }
  }
  return result;
}

static std::vector<Shape> ParseShapesFromCsv(const std::string& csv_path) {
  std::vector<Shape> result;
  std::ifstream f(csv_path);
  if (!f.is_open()) {
    fprintf(stderr, "ERROR: Cannot open %s\n", csv_path.c_str());
    return result;
  }
  std::string line;
  bool header = true;
  while (std::getline(f, line)) {
    if (header) { header = false; continue; }
    if (line.empty()) continue;
    // matmul_roster.csv columns:
    //   label, subgraph_idx, subgraph_name, op_idx, op_type, m, n, k, ...
    // m/n/k are at indices 5/6/7. Split by comma and extract.
    std::vector<std::string> cols;
    std::stringstream ss(line);
    std::string col;
    while (std::getline(ss, col, ',')) cols.push_back(col);
    int m = 0, n = 0, k = 0;
    if (cols.size() >= 8) {
      m = atoi(cols[5].c_str());
      n = atoi(cols[6].c_str());
      k = atoi(cols[7].c_str());
    } else if (cols.size() >= 3) {
      m = atoi(cols[0].c_str());
      n = atoi(cols[1].c_str());
      k = atoi(cols[2].c_str());
    }
    if (m > 0 && n > 0 && k > 0) {
      result.push_back({m, n, k});
    }
  }
  // Dedup.
  std::sort(result.begin(), result.end(), [](const Shape& a, const Shape& b) {
    if (a.m != b.m) return a.m < b.m;
    if (a.n != b.n) return a.n < b.n;
    return a.k < b.k;
  });
  result.erase(std::unique(result.begin(), result.end(),
                           [](const Shape& a, const Shape& b) {
                             return a.m == b.m && a.n == b.n && a.k == b.k;
                           }),
               result.end());
  return result;
}

// ============================================================================
// CLI argument parsing (minimal, no absl dependency)
// ============================================================================
struct Args {
  std::string shapes;
  std::string shapes_csv;
  std::string csv_out;
  std::string dtype = "fp16";  // fp16 or fp32
  int warmup = 5;
  int iters = 50;
  int max_tensor_mb = 512;
};

static std::string GetFlag(int argc, char** argv, const std::string& name,
                           const std::string& default_val) {
  std::string prefix = "--" + name + "=";
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg.substr(0, prefix.size()) == prefix) {
      return arg.substr(prefix.size());
    }
  }
  return default_val;
}

static int GetIntFlag(int argc, char** argv, const std::string& name,
                      int default_val) {
  std::string val = GetFlag(argc, argv, name, "");
  if (val.empty()) return default_val;
  return atoi(val.c_str());
}

static Args ParseArgs(int argc, char** argv) {
  Args args;
  args.shapes = GetFlag(argc, argv, "shapes", "");
  args.shapes_csv = GetFlag(argc, argv, "shapes_csv", "");
  args.csv_out = GetFlag(argc, argv, "csv_out", "");
  args.dtype = GetFlag(argc, argv, "dtype", "fp16");
  args.warmup = GetIntFlag(argc, argv, "warmup", 5);
  args.iters = GetIntFlag(argc, argv, "iters", 50);
  args.max_tensor_mb = GetIntFlag(argc, argv, "max_tensor_mb", 512);
  return args;
}

// ============================================================================
// Benchmark result
// ============================================================================
struct BenchResult {
  Shape shape;
  int count;
  double min_us, avg_us, p50_us, p95_us, max_us;
  double gpu_min_us, gpu_avg_us;
  double tflops_wall, tflops_gpu;
};

// ============================================================================
// Main benchmark logic
// ============================================================================
static bool BenchmarkShape(
    const CLFunctions& cl, cl_context ctx, cl_command_queue queue,
    cl_command_queue prof_queue, cl_kernel kernel,
    const Shape& shape, bool is_fp16, bool is_wave,
    const Args& args, BenchResult* out) {
  int M = shape.m, N = shape.n, K = shape.k;
  size_t elem_size = is_fp16 ? 2 : 4;
  size_t a_bytes = (size_t)M * K * elem_size;
  size_t b_bytes = (size_t)K * N * elem_size;
  size_t c_bytes = (size_t)M * N * elem_size;
  size_t total_mb = (a_bytes + b_bytes + c_bytes) / (1024 * 1024);

  if ((int)total_mb > args.max_tensor_mb) {
    fprintf(stderr, "[cl_bench] SKIP %dx%dx%d: %zuMB > %dMB limit\n",
            M, N, K, total_mb, args.max_tensor_mb);
    return false;
  }

  cl_int err;
  cl_mem buf_a = cl.CreateBuffer(ctx, CL_MEM_READ_ONLY, a_bytes, nullptr, &err);
  if (err != CL_SUCCESS) {
    fprintf(stderr, "[cl_bench] clCreateBuffer(A) failed: %d\n", err);
    return false;
  }
  cl_mem buf_b = cl.CreateBuffer(ctx, CL_MEM_READ_ONLY, b_bytes, nullptr, &err);
  if (err != CL_SUCCESS) {
    cl.ReleaseMemObject(buf_a);
    fprintf(stderr, "[cl_bench] clCreateBuffer(B) failed: %d\n", err);
    return false;
  }
  cl_mem buf_c = cl.CreateBuffer(ctx, CL_MEM_READ_WRITE, c_bytes, nullptr, &err);
  if (err != CL_SUCCESS) {
    cl.ReleaseMemObject(buf_a);
    cl.ReleaseMemObject(buf_b);
    fprintf(stderr, "[cl_bench] clCreateBuffer(C) failed: %d\n", err);
    return false;
  }

  // Initialize A and B with small values to avoid overflow in fp16.
  std::vector<uint8_t> host_a(a_bytes, 0);
  std::vector<uint8_t> host_b(b_bytes, 0);
  if (is_fp16) {
    // fp16: 0x3C00 = 1.0h, use 0x2E66 ~= 0.1h to keep products small
    uint16_t val = 0x2E66;
    auto* pa = reinterpret_cast<uint16_t*>(host_a.data());
    auto* pb = reinterpret_cast<uint16_t*>(host_b.data());
    for (size_t i = 0; i < (size_t)M * K; ++i) pa[i] = val;
    for (size_t i = 0; i < (size_t)K * N; ++i) pb[i] = val;
  } else {
    auto* pa = reinterpret_cast<float*>(host_a.data());
    auto* pb = reinterpret_cast<float*>(host_b.data());
    for (size_t i = 0; i < (size_t)M * K; ++i) pa[i] = 0.01f;
    for (size_t i = 0; i < (size_t)K * N; ++i) pb[i] = 0.01f;
  }

  cl.EnqueueWriteBuffer(queue, buf_a, 1, 0, a_bytes, host_a.data(), 0, nullptr, nullptr);
  cl.EnqueueWriteBuffer(queue, buf_b, 1, 0, b_bytes, host_b.data(), 0, nullptr, nullptr);

  cl.SetKernelArg(kernel, 0, sizeof(cl_mem), &buf_a);
  cl.SetKernelArg(kernel, 1, sizeof(cl_mem), &buf_b);
  cl.SetKernelArg(kernel, 2, sizeof(cl_mem), &buf_c);
  cl.SetKernelArg(kernel, 3, sizeof(int), &M);
  cl.SetKernelArg(kernel, 4, sizeof(int), &N);
  cl.SetKernelArg(kernel, 5, sizeof(int), &K);

  cl_uint ndim;
  size_t local_work_2d[2] = {16, 16};
  size_t global_work_2d[2];
  size_t local_work_wave[2] = {64, 1};
  size_t global_work_wave[2];
  size_t *local_work, *global_work;

  if (is_wave) {
    // Wave kernel: WG_N=COLS_PER_WAVE*TN=32, WG_M=ROWS_PER_WAVE*TM=32
    global_work_wave[0] = static_cast<size_t>(((N + 31) / 32)) * 64;
    global_work_wave[1] = static_cast<size_t>((M + 31) / 32);
    local_work = local_work_wave;
    global_work = global_work_wave;
    ndim = 2;
  } else {
    global_work_2d[0] = static_cast<size_t>(((M + 15) / 16) * 16);
    global_work_2d[1] = static_cast<size_t>(((N + 15) / 16) * 16);
    local_work = local_work_2d;
    global_work = global_work_2d;
    ndim = 2;
  }

  // Warmup
  for (int i = 0; i < args.warmup; ++i) {
    cl.EnqueueNDRangeKernel(queue, kernel, ndim, nullptr, global_work,
                            local_work, 0, nullptr, nullptr);
  }
  cl.Finish(queue);

  // Timed: burst-submit all iters, then clFinish once (wall-clock).
  auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < args.iters; ++i) {
    cl.EnqueueNDRangeKernel(queue, kernel, ndim, nullptr, global_work,
                            local_work, 0, nullptr, nullptr);
  }
  cl.Finish(queue);
  auto t1 = std::chrono::high_resolution_clock::now();
  double total_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
  double avg_us = total_us / args.iters;

  // Per-iter wall-clock for statistics.
  std::vector<double> wall_times;
  wall_times.reserve(args.iters);
  for (int i = 0; i < args.iters; ++i) {
    auto s = std::chrono::high_resolution_clock::now();
    cl.EnqueueNDRangeKernel(queue, kernel, ndim, nullptr, global_work,
                            local_work, 0, nullptr, nullptr);
    cl.Finish(queue);
    auto e = std::chrono::high_resolution_clock::now();
    wall_times.push_back(std::chrono::duration<double, std::micro>(e - s).count());
  }

  std::sort(wall_times.begin(), wall_times.end());
  double min_us = wall_times.front();
  double max_us = wall_times.back();
  double p50_us = wall_times[wall_times.size() / 2];
  int p95_idx = static_cast<int>(wall_times.size() * 0.95);
  double p95_us = wall_times[std::min(p95_idx, (int)wall_times.size() - 1)];

  // GPU-only time via profiling queue (single dispatch with event).
  double gpu_min_us = 0, gpu_avg_us = 0;
  if (prof_queue) {
    std::vector<double> gpu_times;
    gpu_times.reserve(args.iters);
    for (int i = 0; i < args.iters; ++i) {
      cl_event ev = nullptr;
      cl.EnqueueNDRangeKernel(prof_queue, kernel, ndim, nullptr, global_work,
                              local_work, 0, nullptr, &ev);
      cl.Finish(prof_queue);
      cl_ulong ts_start = 0, ts_end = 0;
      cl.GetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(ts_start), &ts_start, nullptr);
      cl.GetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(ts_end), &ts_end, nullptr);
      cl.ReleaseEvent(ev);
      gpu_times.push_back(static_cast<double>(ts_end - ts_start) / 1000.0);
    }
    std::sort(gpu_times.begin(), gpu_times.end());
    gpu_min_us = gpu_times.front();
    gpu_avg_us = std::accumulate(gpu_times.begin(), gpu_times.end(), 0.0) / gpu_times.size();
  }

  double gflops = shape.gflops();
  out->shape = shape;
  out->count = args.iters;
  out->min_us = min_us;
  out->avg_us = avg_us;
  out->p50_us = p50_us;
  out->p95_us = p95_us;
  out->max_us = max_us;
  out->gpu_min_us = gpu_min_us;
  out->gpu_avg_us = gpu_avg_us;
  out->tflops_wall = (gflops / (avg_us / 1e6)) / 1000.0;
  out->tflops_gpu = gpu_avg_us > 0 ? (gflops / (gpu_avg_us / 1e6)) / 1000.0 : 0;

  cl.ReleaseMemObject(buf_a);
  cl.ReleaseMemObject(buf_b);
  cl.ReleaseMemObject(buf_c);
  return true;
}

// ============================================================================
// Main
// ============================================================================
int main(int argc, char** argv) {
  Args args = ParseArgs(argc, argv);

  fprintf(stderr, "[cl_bench] Direct OpenCL matmul benchmark\n");
  fprintf(stderr, "[cl_bench] dtype=%s warmup=%d iters=%d max_tensor_mb=%d\n",
          args.dtype.c_str(), args.warmup, args.iters, args.max_tensor_mb);

  // Resolve shapes.
  std::vector<Shape> shapes;
  if (!args.shapes.empty()) {
    shapes = ParseShapes(args.shapes);
  } else if (!args.shapes_csv.empty()) {
    shapes = ParseShapesFromCsv(args.shapes_csv);
  } else {
    fprintf(stderr, "ERROR: specify --shapes=MxNxK,... or --shapes_csv=path\n");
    return 1;
  }
  if (shapes.empty()) {
    fprintf(stderr, "ERROR: no valid shapes\n");
    return 1;
  }
  fprintf(stderr, "[cl_bench] %zu shapes to benchmark\n", shapes.size());

  // Load OpenCL.
  if (!LoadOpenCL()) return 1;

  // Init CL context.
  cl_platform_id platform;
  cl_uint num_plat = 0;
  cl.GetPlatformIDs(1, &platform, &num_plat);
  if (num_plat == 0) {
    fprintf(stderr, "ERROR: No OpenCL platforms\n");
    return 1;
  }

  char plat_name[256] = {};
  cl.GetPlatformInfo(platform, CL_PLATFORM_NAME, sizeof(plat_name), plat_name, nullptr);
  fprintf(stderr, "[cl_bench] Platform: %s\n", plat_name);

  cl_device_id device;
  cl_uint num_dev = 0;
  cl.GetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, &num_dev);
  if (num_dev == 0) {
    fprintf(stderr, "ERROR: No GPU devices\n");
    return 1;
  }

  char dev_name[256] = {};
  cl.GetDeviceInfo(device, CL_DEVICE_NAME, sizeof(dev_name), dev_name, nullptr);
  cl_uint compute_units = 0, max_freq = 0;
  cl_ulong gmem_size = 0;
  cl.GetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(compute_units), &compute_units, nullptr);
  cl.GetDeviceInfo(device, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(max_freq), &max_freq, nullptr);
  cl.GetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(gmem_size), &gmem_size, nullptr);
  fprintf(stderr, "[cl_bench] Device: %s (%u CUs, %u MHz, %lu MB GMEM)\n",
          dev_name, compute_units, max_freq,
          (unsigned long)(gmem_size / (1024 * 1024)));

  cl_int err;
  cl_context ctx = cl.CreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
  if (err != CL_SUCCESS) {
    fprintf(stderr, "ERROR: clCreateContext: %d\n", err);
    return 1;
  }

  cl_command_queue queue = cl.CreateCommandQueue(ctx, device, 0, &err);
  if (err != CL_SUCCESS) {
    fprintf(stderr, "ERROR: clCreateCommandQueue: %d\n", err);
    return 1;
  }

  cl_command_queue prof_queue = cl.CreateCommandQueue(
      ctx, device, CL_QUEUE_PROFILING_ENABLE, &err);
  if (err != CL_SUCCESS) {
    fprintf(stderr, "[cl_bench] WARNING: profiling queue failed (%d), GPU times unavailable\n", err);
    prof_queue = nullptr;
  }

  // Build kernel.
  bool is_fp16 = (args.dtype == "fp16" || args.dtype == "fp16_wave");
  bool is_wave = (args.dtype == "fp16_wave");
  const char* kernel_src;
  const char* kernel_name;
  if (is_wave) {
    kernel_src = kGemmKernelWaveFp16;
    kernel_name = "gemm_wave_fp16";
  } else if (is_fp16) {
    kernel_src = kGemmKernelTiledFp16;
    kernel_name = "gemm_tiled_fp16";
  } else {
    kernel_src = kGemmKernelTiledFp32;
    kernel_name = "gemm_tiled_fp32";
  }

  size_t src_len = strlen(kernel_src);
  cl_program program = cl.CreateProgramWithSource(ctx, 1, &kernel_src, &src_len, &err);
  if (err != CL_SUCCESS) {
    fprintf(stderr, "ERROR: clCreateProgramWithSource: %d\n", err);
    return 1;
  }

  err = cl.BuildProgram(program, 1, &device, is_fp16 ? "-cl-fast-relaxed-math" : "-cl-fast-relaxed-math", nullptr, nullptr);
  if (err != CL_SUCCESS) {
    size_t log_size = 0;
    cl.GetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
    std::vector<char> build_log(log_size + 1, 0);
    cl.GetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, build_log.data(), nullptr);
    fprintf(stderr, "ERROR: clBuildProgram: %d\n%s\n", err, build_log.data());
    return 1;
  }
  fprintf(stderr, "[cl_bench] Kernel '%s' compiled OK\n", kernel_name);

  cl_kernel kernel = cl.CreateKernel(program, kernel_name, &err);
  if (err != CL_SUCCESS) {
    fprintf(stderr, "ERROR: clCreateKernel: %d\n", err);
    return 1;
  }

  // Run benchmarks.
  std::vector<BenchResult> results;
  for (size_t i = 0; i < shapes.size(); ++i) {
    const auto& s = shapes[i];
    fprintf(stderr, "\n[cl_bench] [%zu/%zu] M=%d N=%d K=%d (%.3f GFLOPS)\n",
            i + 1, shapes.size(), s.m, s.n, s.k, s.gflops());
    BenchResult res;
    if (BenchmarkShape(cl, ctx, queue, prof_queue, kernel, s, is_fp16, is_wave, args, &res)) {
      fprintf(stderr,
              "  wall: avg=%.1f us  min=%.1f  p50=%.1f  p95=%.1f  max=%.1f\n"
              "  gpu:  avg=%.1f us  min=%.1f\n"
              "  TFLOPS: wall=%.3f  gpu=%.3f\n",
              res.avg_us, res.min_us, res.p50_us, res.p95_us, res.max_us,
              res.gpu_avg_us, res.gpu_min_us,
              res.tflops_wall, res.tflops_gpu);
      results.push_back(res);
    }
  }

  // Write CSV.
  FILE* csv_fp = stdout;
  bool close_csv = false;
  if (!args.csv_out.empty()) {
    csv_fp = fopen(args.csv_out.c_str(), "w");
    if (!csv_fp) {
      fprintf(stderr, "ERROR: cannot open %s for writing\n", args.csv_out.c_str());
      csv_fp = stdout;
    } else {
      close_csv = true;
    }
  }

  fprintf(csv_fp, "m,n,k,count,min_us,avg_us,p50_us,p95_us,max_us,"
                  "gpu_min_us,gpu_avg_us,gflops,tflops_wall,tflops_gpu,dtype\n");
  for (const auto& r : results) {
    fprintf(csv_fp, "%d,%d,%d,%d,%.3f,%.3f,%.3f,%.3f,%.3f,"
                    "%.3f,%.3f,%.6f,%.6f,%.6f,%s\n",
            r.shape.m, r.shape.n, r.shape.k, r.count,
            r.min_us, r.avg_us, r.p50_us, r.p95_us, r.max_us,
            r.gpu_min_us, r.gpu_avg_us, r.shape.gflops(),
            r.tflops_wall, r.tflops_gpu, args.dtype.c_str());
  }

  if (close_csv) {
    fclose(csv_fp);
    fprintf(stderr, "\n[cl_bench] CSV written to %s\n", args.csv_out.c_str());
  }

  // Cleanup.
  cl.ReleaseKernel(kernel);
  cl.ReleaseProgram(program);
  if (prof_queue) cl.ReleaseCommandQueue(prof_queue);
  cl.ReleaseCommandQueue(queue);
  cl.ReleaseContext(ctx);

  fprintf(stderr, "\n[cl_bench] Done. %zu shapes benchmarked.\n", results.size());
  return 0;
}
