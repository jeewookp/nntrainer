// OpenCL interceptor: wraps libOpenCL.so to capture CL kernel sources.
//
// Built as a shared library, pushed to device as "libOpenCL.so".
// When the TFLite GPU delegate dlopen's "libOpenCL.so", it finds this
// wrapper first (via LD_LIBRARY_PATH=.). We forward all calls to the
// real /system/vendor/lib64/libOpenCL.so, but for clCreateProgramWithSource
// we also save the CL source text to files.
//
// Output: /data/local/tmp/litert_lm/cl_intercept/program_NNN.cl

#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

// Minimal OpenCL typedefs (avoid pulling in CL headers).
typedef void* cl_context;
typedef void* cl_command_queue;
typedef void* cl_device_id;
typedef void* cl_platform_id;
typedef void* cl_program;
typedef void* cl_kernel;
typedef void* cl_mem;
typedef void* cl_event;
typedef void* cl_sampler;
typedef int32_t cl_int;
typedef uint32_t cl_uint;
typedef uint64_t cl_ulong;
typedef uint64_t cl_bitfield;
typedef cl_bitfield cl_mem_flags;
typedef cl_bitfield cl_command_queue_properties;
typedef cl_uint cl_mem_object_type;
typedef cl_uint cl_image_info;
typedef cl_uint cl_program_info;
typedef cl_uint cl_program_build_info;
typedef cl_uint cl_device_info;
typedef cl_uint cl_platform_info;
typedef cl_uint cl_context_info;
typedef cl_uint cl_kernel_info;
typedef cl_uint cl_kernel_work_group_info;
typedef cl_uint cl_mem_info;
typedef cl_uint cl_profiling_info;
typedef cl_uint cl_buffer_create_type;

typedef struct { cl_uint image_type; cl_uint num_mip_levels; cl_uint num_samples;
  cl_mem buffer; size_t image_width; size_t image_height; size_t image_depth;
  size_t image_array_size; size_t image_row_pitch; size_t image_slice_pitch;
} cl_image_desc;
typedef struct { cl_uint image_channel_order; cl_uint image_channel_data_type;
} cl_image_format;

static void* real_handle = NULL;
static int program_counter = 0;
static const char* DUMP_DIR = "/data/local/tmp/litert_lm/cl_intercept";
static const char* REAL_LIB = "/system/vendor/lib64/libOpenCL.so";

static void ensure_real() {
  if (real_handle) return;
  real_handle = dlopen(REAL_LIB, RTLD_NOW);
  if (!real_handle) {
    fprintf(stderr, "[cl_intercept] FATAL: cannot dlopen %s: %s\n",
            REAL_LIB, dlerror());
    abort();
  }
  fprintf(stderr, "[cl_intercept] Loaded real libOpenCL from %s\n", REAL_LIB);
  char cmd[512];
  snprintf(cmd, sizeof(cmd), "mkdir -p %s", DUMP_DIR);
  system(cmd);
}

static void* get_real(const char* name) {
  ensure_real();
  void* sym = dlsym(real_handle, name);
  if (!sym)
    fprintf(stderr, "[cl_intercept] WARNING: %s not found\n", name);
  return sym;
}

// Macro to define a forwarding wrapper.
#define FORWARD_0(ret, name) \
  ret name(void) { \
    typedef ret (*fn_t)(void); \
    static fn_t fn = NULL; if (!fn) fn = (fn_t)get_real(#name); \
    return fn(); }

#define FORWARD(ret, name, args, call) \
  ret name args { \
    typedef ret (*fn_t) args; \
    static fn_t fn = NULL; if (!fn) fn = (fn_t)get_real(#name); \
    return fn call; }

// ============================================================================
// Intercepted function: clCreateProgramWithSource
// ============================================================================
typedef cl_program (*pfn_CreateProgramWithSource)(
    cl_context, cl_uint, const char**, const size_t*, cl_int*);

cl_program clCreateProgramWithSource(
    cl_context context, cl_uint count,
    const char** strings, const size_t* lengths, cl_int* errcode_ret) {
  static pfn_CreateProgramWithSource real_fn = NULL;
  if (!real_fn) real_fn = (pfn_CreateProgramWithSource)get_real(
      "clCreateProgramWithSource");

  int idx = __sync_fetch_and_add(&program_counter, 1);
  char fname[512];
  snprintf(fname, sizeof(fname), "%s/program_%03d.cl", DUMP_DIR, idx);

  FILE* f = fopen(fname, "w");
  if (f) {
    for (cl_uint i = 0; i < count; ++i) {
      if (strings[i]) {
        size_t len = (lengths && lengths[i]) ? lengths[i] : strlen(strings[i]);
        fwrite(strings[i], 1, len, f);
        fputc('\n', f);
      }
    }
    fclose(f);
    fprintf(stderr, "[cl_intercept] Saved %s (%u source(s))\n", fname, count);
  }

  return real_fn(context, count, strings, lengths, errcode_ret);
}

// ============================================================================
// Forwarded OpenCL functions (all used by TFLite opencl_wrapper)
// ============================================================================

// Platform
FORWARD(cl_int, clGetPlatformIDs,
  (cl_uint n, cl_platform_id* p, cl_uint* num),
  (n, p, num))
FORWARD(cl_int, clGetPlatformInfo,
  (cl_platform_id p, cl_platform_info i, size_t s, void* v, size_t* r),
  (p, i, s, v, r))

// Device
FORWARD(cl_int, clGetDeviceIDs,
  (cl_platform_id p, cl_bitfield t, cl_uint n, cl_device_id* d, cl_uint* num),
  (p, t, n, d, num))
FORWARD(cl_int, clGetDeviceInfo,
  (cl_device_id d, cl_device_info i, size_t s, void* v, size_t* r),
  (d, i, s, v, r))

// Context
typedef void (*cl_context_cb)(const char*, const void*, size_t, void*);
FORWARD(cl_context, clCreateContext,
  (const intptr_t* props, cl_uint n, const cl_device_id* devs,
   cl_context_cb cb, void* ud, cl_int* err),
  (props, n, devs, cb, ud, err))
FORWARD(cl_int, clReleaseContext, (cl_context c), (c))
FORWARD(cl_int, clGetContextInfo,
  (cl_context c, cl_context_info i, size_t s, void* v, size_t* r),
  (c, i, s, v, r))
FORWARD(cl_int, clRetainContext, (cl_context c), (c))

// Command Queue
FORWARD(cl_command_queue, clCreateCommandQueue,
  (cl_context c, cl_device_id d, cl_command_queue_properties p, cl_int* e),
  (c, d, p, e))
FORWARD(cl_int, clReleaseCommandQueue, (cl_command_queue q), (q))
FORWARD(cl_int, clFlush, (cl_command_queue q), (q))
FORWARD(cl_int, clFinish, (cl_command_queue q), (q))

// Memory
FORWARD(cl_mem, clCreateBuffer,
  (cl_context c, cl_mem_flags f, size_t s, void* p, cl_int* e),
  (c, f, s, p, e))
FORWARD(cl_mem, clCreateImage,
  (cl_context c, cl_mem_flags f, const cl_image_format* fmt,
   const cl_image_desc* desc, void* p, cl_int* e),
  (c, f, fmt, desc, p, e))
FORWARD(cl_mem, clCreateImage2D,
  (cl_context c, cl_mem_flags f, const cl_image_format* fmt,
   size_t w, size_t h, size_t rp, void* p, cl_int* e),
  (c, f, fmt, w, h, rp, p, e))
FORWARD(cl_mem, clCreateSubBuffer,
  (cl_mem m, cl_mem_flags f, cl_buffer_create_type t, const void* i, cl_int* e),
  (m, f, t, i, e))
FORWARD(cl_int, clReleaseMemObject, (cl_mem m), (m))
FORWARD(cl_int, clGetMemObjectInfo,
  (cl_mem m, cl_mem_info i, size_t s, void* v, size_t* r),
  (m, i, s, v, r))
FORWARD(cl_int, clGetImageInfo,
  (cl_mem m, cl_image_info i, size_t s, void* v, size_t* r),
  (m, i, s, v, r))
FORWARD(cl_int, clRetainMemObject, (cl_mem m), (m))

// Map/Unmap
FORWARD(void*, clEnqueueMapBuffer,
  (cl_command_queue q, cl_mem m, cl_uint b, cl_mem_flags f,
   size_t off, size_t sz, cl_uint nw, const cl_event* wl, cl_event* ev, cl_int* e),
  (q, m, b, f, off, sz, nw, wl, ev, e))
FORWARD(cl_int, clEnqueueUnmapMemObject,
  (cl_command_queue q, cl_mem m, void* p, cl_uint nw, const cl_event* wl, cl_event* ev),
  (q, m, p, nw, wl, ev))

// Program
FORWARD(cl_int, clBuildProgram,
  (cl_program p, cl_uint n, const cl_device_id* d, const char* opts,
   void (*cb)(cl_program, void*), void* ud),
  (p, n, d, opts, cb, ud))
FORWARD(cl_int, clGetProgramInfo,
  (cl_program p, cl_program_info i, size_t s, void* v, size_t* r),
  (p, i, s, v, r))
FORWARD(cl_int, clGetProgramBuildInfo,
  (cl_program p, cl_device_id d, cl_program_build_info i, size_t s, void* v, size_t* r),
  (p, d, i, s, v, r))
FORWARD(cl_int, clReleaseProgram, (cl_program p), (p))
FORWARD(cl_int, clRetainProgram, (cl_program p), (p))

// Kernel
FORWARD(cl_kernel, clCreateKernel,
  (cl_program p, const char* n, cl_int* e),
  (p, n, e))
FORWARD(cl_int, clSetKernelArg,
  (cl_kernel k, cl_uint i, size_t s, const void* v),
  (k, i, s, v))
FORWARD(cl_int, clReleaseKernel, (cl_kernel k), (k))
FORWARD(cl_int, clGetKernelInfo,
  (cl_kernel k, cl_kernel_info i, size_t s, void* v, size_t* r),
  (k, i, s, v, r))
FORWARD(cl_int, clGetKernelWorkGroupInfo,
  (cl_kernel k, cl_device_id d, cl_kernel_work_group_info i, size_t s, void* v, size_t* r),
  (k, d, i, s, v, r))

// Enqueue
FORWARD(cl_int, clEnqueueNDRangeKernel,
  (cl_command_queue q, cl_kernel k, cl_uint dim, const size_t* go,
   const size_t* gw, const size_t* lw, cl_uint nw, const cl_event* wl, cl_event* ev),
  (q, k, dim, go, gw, lw, nw, wl, ev))
FORWARD(cl_int, clEnqueueWriteBuffer,
  (cl_command_queue q, cl_mem m, cl_uint b, size_t o, size_t s,
   const void* p, cl_uint nw, const cl_event* wl, cl_event* ev),
  (q, m, b, o, s, p, nw, wl, ev))
FORWARD(cl_int, clEnqueueReadBuffer,
  (cl_command_queue q, cl_mem m, cl_uint b, size_t o, size_t s,
   void* p, cl_uint nw, const cl_event* wl, cl_event* ev),
  (q, m, b, o, s, p, nw, wl, ev))
FORWARD(cl_int, clEnqueueCopyBuffer,
  (cl_command_queue q, cl_mem s, cl_mem d, size_t so, size_t do_, size_t sz,
   cl_uint nw, const cl_event* wl, cl_event* ev),
  (q, s, d, so, do_, sz, nw, wl, ev))
FORWARD(cl_int, clEnqueueReadImage,
  (cl_command_queue q, cl_mem m, cl_uint b, const size_t* o, const size_t* r,
   size_t rp, size_t sp, void* p, cl_uint nw, const cl_event* wl, cl_event* ev),
  (q, m, b, o, r, rp, sp, p, nw, wl, ev))
FORWARD(cl_int, clEnqueueWriteImage,
  (cl_command_queue q, cl_mem m, cl_uint b, const size_t* o, const size_t* r,
   size_t rp, size_t sp, const void* p, cl_uint nw, const cl_event* wl, cl_event* ev),
  (q, m, b, o, r, rp, sp, p, nw, wl, ev))

// Events
FORWARD(cl_int, clWaitForEvents, (cl_uint n, const cl_event* e), (n, e))
FORWARD(cl_int, clGetEventProfilingInfo,
  (cl_event e, cl_profiling_info i, size_t s, void* v, size_t* r),
  (e, i, s, v, r))
FORWARD(cl_int, clReleaseEvent, (cl_event e), (e))
FORWARD(cl_int, clRetainEvent, (cl_event e), (e))

// Image format query
FORWARD(cl_int, clGetSupportedImageFormats,
  (cl_context c, cl_mem_flags f, cl_mem_object_type t, cl_uint n,
   cl_image_format* fmt, cl_uint* num),
  (c, f, t, n, fmt, num))

// clCreateProgramWithBinary — also intercept to log
typedef cl_program (*pfn_CreateProgramWithBinary)(
    cl_context, cl_uint, const cl_device_id*, const size_t*,
    const unsigned char**, cl_int*, cl_int*);

cl_program clCreateProgramWithBinary(
    cl_context context, cl_uint num_devices, const cl_device_id* device_list,
    const size_t* lengths, const unsigned char** binaries,
    cl_int* binary_status, cl_int* errcode_ret) {
  static pfn_CreateProgramWithBinary real_fn = NULL;
  if (!real_fn) real_fn = (pfn_CreateProgramWithBinary)get_real(
      "clCreateProgramWithBinary");

  int idx = __sync_fetch_and_add(&program_counter, 1);
  fprintf(stderr, "[cl_intercept] clCreateProgramWithBinary #%d (binary, %u device(s))\n",
          idx, num_devices);

  return real_fn(context, num_devices, device_list, lengths, binaries,
                 binary_status, errcode_ret);
}
