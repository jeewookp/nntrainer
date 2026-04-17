// OpenCL kernel source interceptor.
//
// Wraps libOpenCL.so by forwarding ALL CL calls to the real vendor library
// at /system/vendor/lib64/libOpenCL.so. For clCreateProgramWithSource, also
// saves the CL source text to files.
//
// Push as "libOpenCL.so" to the device folder so the delegate loads it first.
// Output: /data/local/tmp/litert_lm/cl_intercept/program_NNN.cl

#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static void* real_lib = nullptr;
static int prog_cnt = 0;
static const char* DUMP_DIR = "/data/local/tmp/litert_lm/cl_intercept";
static const char* REAL_PATH = "/system/vendor/lib64/libOpenCL.so";

static void init() {
  if (real_lib) return;
  real_lib = dlopen(REAL_PATH, RTLD_NOW | RTLD_LOCAL);
  if (!real_lib) {
    fprintf(stderr, "[cl_intercept] FATAL: dlopen(%s): %s\n", REAL_PATH, dlerror());
    _exit(1);
  }
  fprintf(stderr, "[cl_intercept] Real OpenCL loaded from %s\n", REAL_PATH);
  char cmd[256];
  snprintf(cmd, sizeof(cmd), "mkdir -p %s", DUMP_DIR);
  system(cmd);
}

static void* sym(const char* n) {
  init();
  void* p = dlsym(real_lib, n);
  if (!p) fprintf(stderr, "[cl_intercept] WARNING: %s not found in real lib\n", n);
  return p;
}

// Generic forwarding macro. Uses a static function pointer per symbol.
#define FWD(RET, NAME, PARAMS, ARGS) \
  extern "C" RET NAME PARAMS { \
    typedef RET (*_fwd_fn_t) PARAMS; \
    static _fwd_fn_t _fwd_fn = nullptr; \
    if (!_fwd_fn) _fwd_fn = (_fwd_fn_t)sym(#NAME); \
    return _fwd_fn ARGS; \
  }

// ============================================================================
// INTERCEPTED: clCreateProgramWithSource — saves CL source to file.
// ============================================================================
typedef void* cl_program;
typedef void* cl_context;
typedef int32_t cl_int;
typedef uint32_t cl_uint;

typedef cl_program (*real_cpws_t)(cl_context, cl_uint, const char**, const size_t*, cl_int*);

extern "C" cl_program clCreateProgramWithSource(
    cl_context ctx, cl_uint count, const char** strings,
    const size_t* lengths, cl_int* err) {
  static real_cpws_t real_fn = nullptr;
  if (!real_fn) real_fn = (real_cpws_t)sym("clCreateProgramWithSource");

  int idx = __sync_fetch_and_add(&prog_cnt, 1);
  char fname[256];
  snprintf(fname, sizeof(fname), "%s/program_%03d.cl", DUMP_DIR, idx);
  FILE* f = fopen(fname, "w");
  if (f) {
    size_t total = 0;
    for (cl_uint i = 0; i < count; ++i) {
      if (!strings[i]) continue;
      size_t len = (lengths && lengths[i]) ? lengths[i] : strlen(strings[i]);
      fwrite(strings[i], 1, len, f);
      total += len;
    }
    fclose(f);
    fprintf(stderr, "[cl_intercept] #%d: %s (%zu bytes)\n", idx, fname, total);
  }
  return real_fn(ctx, count, strings, lengths, err);
}

// ============================================================================
// Also intercept clCreateProgramWithBinary for completeness.
// ============================================================================
typedef cl_program (*real_cpwb_t)(cl_context, cl_uint, const void*,
                                  const size_t*, const unsigned char**,
                                  cl_int*, cl_int*);
extern "C" cl_program clCreateProgramWithBinary(
    cl_context ctx, cl_uint nd, const void* dl,
    const size_t* lengths, const unsigned char** bins,
    cl_int* bs, cl_int* err) {
  static real_cpwb_t real_fn = nullptr;
  if (!real_fn) real_fn = (real_cpwb_t)sym("clCreateProgramWithBinary");
  int idx = __sync_fetch_and_add(&prog_cnt, 1);
  fprintf(stderr, "[cl_intercept] #%d: clCreateProgramWithBinary (binary)\n", idx);
  return real_fn(ctx, nd, dl, lengths, bins, bs, err);
}

// ============================================================================
// Intercept clCreateProgramWithIL (SPIR-V path, OpenCL 2.1+).
// The delegate may use SPIR-V instead of CL source text.
// ============================================================================
typedef cl_program (*real_cpwil_t)(cl_context, const void*, size_t, cl_int*);
extern "C" cl_program clCreateProgramWithIL(
    cl_context ctx, const void* il, size_t length, cl_int* err) {
  static real_cpwil_t real_fn = nullptr;
  if (!real_fn) real_fn = (real_cpwil_t)sym("clCreateProgramWithIL");

  int idx = __sync_fetch_and_add(&prog_cnt, 1);
  char fname[256];
  snprintf(fname, sizeof(fname), "%s/program_%03d.spv", DUMP_DIR, idx);
  FILE* fp = fopen(fname, "wb");
  if (fp) {
    fwrite(il, 1, length, fp);
    fclose(fp);
    fprintf(stderr, "[cl_intercept] #%d: clCreateProgramWithIL → %s (%zu bytes SPIR-V)\n",
            idx, fname, length);
  }
  if (!real_fn) {
    fprintf(stderr, "[cl_intercept] WARNING: clCreateProgramWithIL not available\n");
    if (err) *err = -1;
    return nullptr;
  }
  return real_fn(ctx, il, length, err);
}

// ============================================================================
// Intercept clCreateProgramWithBuiltInKernels.
// ============================================================================
typedef cl_program (*real_cpwbik_t)(cl_context, cl_uint, const void*,
                                    const char*, cl_int*);
extern "C" cl_program clCreateProgramWithBuiltInKernels(
    cl_context ctx, cl_uint nd, const void* dl,
    const char* names, cl_int* err) {
  static real_cpwbik_t real_fn = nullptr;
  if (!real_fn) real_fn = (real_cpwbik_t)sym("clCreateProgramWithBuiltInKernels");
  int idx = __sync_fetch_and_add(&prog_cnt, 1);
  fprintf(stderr, "[cl_intercept] #%d: clCreateProgramWithBuiltInKernels names='%s'\n",
          idx, names ? names : "(null)");
  return real_fn(ctx, nd, dl, names, err);
}

// ============================================================================
// Logging macro: wraps key CL functions with a log line.
// If clBuildProgram/clCreateKernel show up → delegate uses our wrapper.
// If NOT → delegate bypasses us entirely (linker namespace isolation).
// ============================================================================
#define FWD_LOG(RET, NAME, PARAMS, ARGS) \
  extern "C" RET NAME PARAMS { \
    typedef RET (*_fwd_fn_t) PARAMS; \
    static _fwd_fn_t _fwd_fn = nullptr; \
    if (!_fwd_fn) _fwd_fn = (_fwd_fn_t)sym(#NAME); \
    fprintf(stderr, "[cl_intercept] CALL: %s\n", #NAME); \
    return _fwd_fn ARGS; \
  }

// ============================================================================
// ALL forwarded CL functions. Covers everything TFLite opencl_wrapper loads.
// Using void* for opaque CL types, size_t for sizes.
// ============================================================================

// Platform
FWD_LOG(cl_int, clGetPlatformIDs, (cl_uint n, void* p, cl_uint* num), (n, p, num))
FWD(cl_int, clGetPlatformInfo, (void* p, cl_uint i, size_t s, void* v, size_t* r), (p, i, s, v, r))

// Device
FWD(cl_int, clGetDeviceIDs, (void* p, uint64_t t, cl_uint n, void* d, cl_uint* num), (p, t, n, d, num))
FWD(cl_int, clGetDeviceInfo, (void* d, cl_uint i, size_t s, void* v, size_t* r), (d, i, s, v, r))

// Context
extern "C" void* clCreateContext(const intptr_t* pr, cl_uint n, const void* d, void* cb, void* ud, cl_int* e) {
  typedef void* (*F)(const intptr_t*, cl_uint, const void*, void*, void*, cl_int*);
  static F _fwd_fn = nullptr;
  if (!_fwd_fn) _fwd_fn = (F)sym("clCreateContext");
  fprintf(stderr, "[cl_intercept] CALL: clCreateContext props=%p\n", (void*)pr);
  if (pr) {
    for (int i = 0; pr[i] != 0; i += 2) {
      fprintf(stderr, "[cl_intercept]   prop 0x%lx = 0x%lx\n",
              (unsigned long)pr[i], (unsigned long)pr[i+1]);
    }
  }
  return _fwd_fn(pr, n, d, cb, ud, e);
}
FWD(cl_int, clReleaseContext, (void* c), (c))
FWD(cl_int, clRetainContext, (void* c), (c))
FWD(cl_int, clGetContextInfo, (void* c, cl_uint i, size_t s, void* v, size_t* r), (c, i, s, v, r))

// Command Queue
FWD(void*, clCreateCommandQueue, (void* c, void* d, uint64_t p, cl_int* e), (c, d, p, e))
FWD(void*, clCreateCommandQueueWithProperties, (void* c, void* d, const uint64_t* p, cl_int* e), (c, d, p, e))
FWD(cl_int, clReleaseCommandQueue, (void* q), (q))
FWD(cl_int, clFlush, (void* q), (q))
FWD(cl_int, clFinish, (void* q), (q))

// Memory
extern "C" void* clCreateBuffer(void* c, uint64_t f, size_t s, void* p, cl_int* e) {
  typedef void* (*F)(void*, uint64_t, size_t, void*, cl_int*);
  static F _fwd_fn = nullptr;
  if (!_fwd_fn) _fwd_fn = (F)sym("clCreateBuffer");
  void* r = _fwd_fn(c, f, s, p, e);
  fprintf(stderr, "[cl_intercept] CreateBuffer flags=0x%llx size=%zu host_ptr=%p → %p\n",
          (unsigned long long)f, s, p, r);
  return r;
}
FWD(void*, clCreateSubBuffer, (void* m, uint64_t f, cl_uint t, const void* i, cl_int* e), (m, f, t, i, e))
extern "C" void* clCreateImage(void* c, uint64_t f, const void* fmt_v, const void* desc_v, void* p, cl_int* e) {
  typedef void* (*F)(void*, uint64_t, const void*, const void*, void*, cl_int*);
  static F _fwd_fn = nullptr;
  if (!_fwd_fn) _fwd_fn = (F)sym("clCreateImage");
  void* r = _fwd_fn(c, f, fmt_v, desc_v, p, e);
  if (desc_v) {
    const size_t* d = (const size_t*)desc_v;
    // cl_image_desc: type, width, height, depth, array_size, ...
    fprintf(stderr, "[cl_intercept] CreateImage flags=0x%llx type=%zu w=%zu h=%zu → %p\n",
            (unsigned long long)f, d[0], d[1], d[2], r);
  }
  return r;
}
FWD(void*, clCreateImage2D, (void* c, uint64_t f, const void* fmt, size_t w, size_t h, size_t rp, void* p, cl_int* e), (c, f, fmt, w, h, rp, p, e))
FWD(cl_int, clReleaseMemObject, (void* m), (m))
FWD(cl_int, clRetainMemObject, (void* m), (m))
FWD(cl_int, clGetMemObjectInfo, (void* m, cl_uint i, size_t s, void* v, size_t* r), (m, i, s, v, r))
FWD(cl_int, clGetImageInfo, (void* m, cl_uint i, size_t s, void* v, size_t* r), (m, i, s, v, r))

// Program (except clCreateProgramWithSource/Binary which are intercepted)
// clBuildProgram: log the build options string — this reveals what flags
// the delegate uses to enable Qualcomm extensions.
extern "C" cl_int clBuildProgram(void* p, cl_uint n, const void* d,
                                  const char* o, void* cb, void* ud) {
  typedef cl_int (*F)(void*, cl_uint, const void*, const char*, void*, void*);
  static F _fwd_fn = nullptr;
  if (!_fwd_fn) _fwd_fn = (F)sym("clBuildProgram");
  fprintf(stderr, "[cl_intercept] CALL: clBuildProgram opts='%s'\n",
          o ? o : "(null)");
  return _fwd_fn(p, n, d, o, cb, ud);
}
FWD(cl_int, clGetProgramInfo, (void* p, cl_uint i, size_t s, void* v, size_t* r), (p, i, s, v, r))
FWD(cl_int, clGetProgramBuildInfo, (void* p, void* d, cl_uint i, size_t s, void* v, size_t* r), (p, d, i, s, v, r))
FWD(cl_int, clReleaseProgram, (void* p), (p))
FWD(cl_int, clRetainProgram, (void* p), (p))

// Kernel
FWD_LOG(void*, clCreateKernel, (void* p, const char* n, cl_int* e), (p, n, e))
// clSetKernelArg: log arg index, size, and value (for int4 args, size=16)
extern "C" cl_int clSetKernelArg(void* k, cl_uint i, size_t s, const void* v) {
  typedef cl_int (*F)(void*, cl_uint, size_t, const void*);
  static F _fwd_fn = nullptr;
  if (!_fwd_fn) _fwd_fn = (F)sym("clSetKernelArg");
  if (s == 16 && v) {
    const int32_t* iv = (const int32_t*)v;
    fprintf(stderr, "[cl_intercept] SetKernelArg[%u] size=%zu int4=(%d,%d,%d,%d)\n",
            i, s, iv[0], iv[1], iv[2], iv[3]);
  } else if (s == sizeof(void*) && v) {
    fprintf(stderr, "[cl_intercept] SetKernelArg[%u] size=%zu mem=%p\n",
            i, s, *(void**)v);
  } else {
    fprintf(stderr, "[cl_intercept] SetKernelArg[%u] size=%zu\n", i, s);
  }
  return _fwd_fn(k, i, s, v);
}
FWD(cl_int, clReleaseKernel, (void* k), (k))
FWD(cl_int, clGetKernelInfo, (void* k, cl_uint i, size_t s, void* v, size_t* r), (k, i, s, v, r))
FWD(cl_int, clGetKernelWorkGroupInfo, (void* k, void* d, cl_uint i, size_t s, void* v, size_t* r), (k, d, i, s, v, r))

// Enqueue
// clEnqueueNDRangeKernel: log dispatch dimensions
extern "C" cl_int clEnqueueNDRangeKernel(void* q, void* k, cl_uint dim,
    const size_t* go, const size_t* gw, const size_t* lw,
    cl_uint nw, const void* wl, void* ev) {
  typedef cl_int (*F)(void*, void*, cl_uint, const size_t*, const size_t*,
                      const size_t*, cl_uint, const void*, void*);
  static F _fwd_fn = nullptr;
  if (!_fwd_fn) _fwd_fn = (F)sym("clEnqueueNDRangeKernel");
  if (dim == 3 && gw && lw) {
    fprintf(stderr, "[cl_intercept] NDRange dim=%u global=(%zu,%zu,%zu) local=(%zu,%zu,%zu)\n",
            dim, gw[0], gw[1], gw[2], lw[0], lw[1], lw[2]);
  } else if (dim > 0 && gw) {
    fprintf(stderr, "[cl_intercept] NDRange dim=%u global=(%zu,...) local=%s\n",
            dim, gw[0], lw ? "set" : "NULL");
  }
  return _fwd_fn(q, k, dim, go, gw, lw, nw, wl, ev);
}
FWD(cl_int, clEnqueueWriteBuffer, (void* q, void* m, cl_uint b, size_t o, size_t s, const void* p, cl_uint nw, const void* wl, void* ev), (q, m, b, o, s, p, nw, wl, ev))
FWD(cl_int, clEnqueueReadBuffer, (void* q, void* m, cl_uint b, size_t o, size_t s, void* p, cl_uint nw, const void* wl, void* ev), (q, m, b, o, s, p, nw, wl, ev))
FWD(cl_int, clEnqueueCopyBuffer, (void* q, void* s, void* d, size_t so, size_t do_, size_t sz, cl_uint nw, const void* wl, void* ev), (q, s, d, so, do_, sz, nw, wl, ev))
FWD(cl_int, clEnqueueFillBuffer, (void* q, void* m, const void* pat, size_t ps, size_t o, size_t s, cl_uint nw, const void* wl, void* ev), (q, m, pat, ps, o, s, nw, wl, ev))
FWD(cl_int, clEnqueueReadImage, (void* q, void* m, cl_uint b, const size_t* o, const size_t* r, size_t rp, size_t sp, void* p, cl_uint nw, const void* wl, void* ev), (q, m, b, o, r, rp, sp, p, nw, wl, ev))
FWD(cl_int, clEnqueueWriteImage, (void* q, void* m, cl_uint b, const size_t* o, const size_t* r, size_t rp, size_t sp, const void* p, cl_uint nw, const void* wl, void* ev), (q, m, b, o, r, rp, sp, p, nw, wl, ev))
FWD(cl_int, clEnqueueCopyBufferToImage, (void* q, void* s, void* d, size_t so, const size_t* do_, const size_t* r, cl_uint nw, const void* wl, void* ev), (q, s, d, so, do_, r, nw, wl, ev))
FWD(cl_int, clEnqueueCopyImageToBuffer, (void* q, void* s, void* d, const size_t* so, const size_t* r, size_t do_, cl_uint nw, const void* wl, void* ev), (q, s, d, so, r, do_, nw, wl, ev))

// Map/Unmap
FWD(void*, clEnqueueMapBuffer, (void* q, void* m, cl_uint b, uint64_t f, size_t o, size_t s, cl_uint nw, const void* wl, void* ev, cl_int* e), (q, m, b, f, o, s, nw, wl, ev, e))
FWD(void*, clEnqueueMapImage, (void* q, void* m, cl_uint b, uint64_t f, const size_t* o, const size_t* r, size_t* rp, size_t* sp, cl_uint nw, const void* wl, void* ev, cl_int* e), (q, m, b, f, o, r, rp, sp, nw, wl, ev, e))
FWD(cl_int, clEnqueueUnmapMemObject, (void* q, void* m, void* p, cl_uint nw, const void* wl, void* ev), (q, m, p, nw, wl, ev))

// Events
FWD(cl_int, clWaitForEvents, (cl_uint n, const void* e), (n, e))
FWD(cl_int, clGetEventProfilingInfo, (void* e, cl_uint i, size_t s, void* v, size_t* r), (e, i, s, v, r))
FWD(cl_int, clReleaseEvent, (void* e), (e))
FWD(cl_int, clRetainEvent, (void* e), (e))
FWD(cl_int, clEnqueueMarkerWithWaitList, (void* q, cl_uint n, const void* wl, void* ev), (q, n, wl, ev))
FWD(cl_int, clEnqueueBarrierWithWaitList, (void* q, cl_uint n, const void* wl, void* ev), (q, n, wl, ev))

// Image format
FWD(cl_int, clGetSupportedImageFormats, (void* c, uint64_t f, cl_uint t, cl_uint n, void* fmt, cl_uint* num), (c, f, t, n, fmt, num))

// Extensions
FWD(void*, clGetExtensionFunctionAddressForPlatform, (void* p, const char* n), (p, n))
FWD(void*, clGetExtensionFunctionAddress, (const char* n), (n))

// SVM (OpenCL 2.0)
FWD(void*, clSVMAlloc, (void* c, uint64_t f, size_t s, cl_uint a), (c, f, s, a))
FWD(void, clSVMFree, (void* c, void* p), (c, p))
FWD(cl_int, clEnqueueSVMMap, (void* q, cl_uint b, uint64_t f, void* p, size_t s, cl_uint nw, const void* wl, void* ev), (q, b, f, p, s, nw, wl, ev))
FWD(cl_int, clEnqueueSVMUnmap, (void* q, void* p, cl_uint nw, const void* wl, void* ev), (q, p, nw, wl, ev))
FWD(cl_int, clSetKernelArgSVMPointer, (void* k, cl_uint i, const void* v), (k, i, v))

// GL interop
FWD(void*, clCreateFromGLTexture, (void* c, uint64_t f, cl_uint t, cl_int ml, cl_uint tex, cl_int* e), (c, f, t, ml, tex, e))
FWD(void*, clCreateFromGLBuffer, (void* c, uint64_t f, cl_uint buf, cl_int* e), (c, f, buf, e))

// EGL interop
FWD(void*, clCreateFromEGLImageKHR, (void* c, void* d, void* img, uint64_t f, const intptr_t* al, cl_int* e), (c, d, img, f, al, e))
FWD(cl_int, clEnqueueAcquireEGLObjectsKHR, (void* q, cl_uint n, const void* m, cl_uint nw, const void* wl, void* ev), (q, n, m, nw, wl, ev))
FWD(cl_int, clEnqueueReleaseEGLObjectsKHR, (void* q, cl_uint n, const void* m, cl_uint nw, const void* wl, void* ev), (q, n, m, nw, wl, ev))

// Command buffer KHR extensions
FWD(void*, clCreateCommandBufferKHR, (cl_uint n, const void* q, const void* p, cl_int* e), (n, q, p, e))
FWD(cl_int, clFinalizeCommandBufferKHR, (void* cb), (cb))
FWD(cl_int, clEnqueueCommandBufferKHR, (cl_uint n, const void* q, void* cb, cl_uint nw, const void* wl, void* ev), (n, q, cb, nw, wl, ev))
FWD(cl_int, clReleaseCommandBufferKHR, (void* cb), (cb))
FWD(cl_int, clCommandNDRangeKernelKHR, (void* cb, void* q, const void* p, void* k, cl_uint dim, const size_t* go, const size_t* gw, const size_t* lw, cl_uint n, const void* sl, void* sh, void* h), (cb, q, p, k, dim, go, gw, lw, n, sl, sh, h))

// Qualcomm perf hint
FWD(cl_int, clSetPerfHintQCOM, (void* c, cl_uint h), (c, h))

// ARM import memory
FWD(void*, clImportMemoryARM, (void* c, uint64_t f, const void* p, void* m, size_t s, cl_int* e), (c, f, p, m, s, e))

// Semaphore KHR
FWD(void*, clCreateSemaphoreWithPropertiesKHR, (void* c, const void* p, cl_int* e), (c, p, e))
FWD(cl_int, clEnqueueWaitSemaphoresKHR, (void* q, cl_uint n, const void* s, const void* v, cl_uint nw, const void* wl, void* ev), (q, n, s, v, nw, wl, ev))
FWD(cl_int, clEnqueueSignalSemaphoresKHR, (void* q, cl_uint n, const void* s, const void* v, cl_uint nw, const void* wl, void* ev), (q, n, s, v, nw, wl, ev))
FWD(cl_int, clReleaseSemaphoreKHR, (void* s), (s))
