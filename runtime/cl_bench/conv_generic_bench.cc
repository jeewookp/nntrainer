// Matmul benchmark using TFLite GPU delegate's ConvGeneric kernel.
//
// Uses the open-source TFLite GPU CL runtime directly (NOT the closed-source
// libLiteRtGpuAccelerator.so). ConvGeneric generates the same optimized
// "convolution(conv_wave_memory)" kernel that achieves 5+ TFLOPS on Adreno
// in production prefill.
//
// Flow:
//   1. tflite::gpu::cl::CreateEnvironment() — auto-detect GPU, create CL ctx
//   2. ConvGeneric(op_def, attr) — build Adreno-tuned conv kernel
//   3. ClOperation::Compile() — JIT compile the CL source
//   4. Burst-submit N dispatches + clFinish — wall-clock timing
//   5. ProfilingCommandQueue — GPU-only timing

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/status/status.h"
#include "tflite/delegates/gpu/cl/cl_command_queue.h"
#include "tflite/delegates/gpu/cl/cl_event.h"
#include "tflite/delegates/gpu/cl/cl_operation.h"
#include "tflite/delegates/gpu/cl/environment.h"
#include "tflite/delegates/gpu/cl/opencl_wrapper.h"
#include "tflite/delegates/gpu/cl/tensor.h"
#include "tflite/delegates/gpu/common/data_type.h"
#include "tflite/delegates/gpu/common/operations.h"
#include "tflite/delegates/gpu/common/shape.h"
#include "tflite/delegates/gpu/common/status.h"
#include "tflite/delegates/gpu/common/task/gpu_operation.h"
#include "tflite/delegates/gpu/common/task/tensor_desc.h"
#include "tflite/delegates/gpu/common/tasks/conv_generic.h"
#include "tflite/delegates/gpu/common/tensor.h"

ABSL_FLAG(std::string, shapes, "",
          "Inline shapes MxNxK,... (e.g. '1024x6144x1536')");
ABSL_FLAG(std::string, shapes_csv, "",
          "Path to matmul_roster.csv (m/n/k at cols 5/6/7)");
ABSL_FLAG(std::string, csv_out, "", "Output CSV path");
ABSL_FLAG(std::string, precision, "fp16",
          "fp16 (F16), fp32 (F32), or fp16acc32 (F32_F16)");
ABSL_FLAG(int32_t, warmup, 5, "Warmup iterations");
ABSL_FLAG(int32_t, iters, 50, "Timed iterations");
ABSL_FLAG(int32_t, max_tensor_mb, 512, "Max tensor memory per shape (MB)");

namespace {

using namespace tflite::gpu;

struct Shape {
  int m, n, k;
  double gflops() const { return 2.0 * m * n * k / 1e9; }
};

struct BenchResult {
  Shape shape;
  int count;
  double wall_avg_us;
  double gpu_avg_us;
  double tflops_wall;
  double tflops_gpu;
};

std::vector<Shape> ParseShapes(const std::string& s) {
  std::vector<Shape> result;
  std::stringstream ss(s);
  std::string tok;
  while (std::getline(ss, tok, ',')) {
    int m = 0, n = 0, k = 0;
    if (sscanf(tok.c_str(), "%dx%dx%d", &m, &n, &k) == 3 &&
        m > 0 && n > 0 && k > 0) {
      result.push_back({m, n, k});
    }
  }
  return result;
}

std::vector<Shape> ParseShapesFromCsv(const std::string& path) {
  std::vector<Shape> result;
  std::ifstream f(path);
  if (!f.is_open()) {
    fprintf(stderr, "ERROR: Cannot open %s\n", path.c_str());
    return result;
  }
  std::string line;
  bool header = true;
  while (std::getline(f, line)) {
    if (header) { header = false; continue; }
    if (line.empty()) continue;
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
    if (m > 0 && n > 0 && k > 0) result.push_back({m, n, k});
  }
  std::sort(result.begin(), result.end(), [](const Shape& a, const Shape& b) {
    return std::tie(a.m, a.n, a.k) < std::tie(b.m, b.n, b.k);
  });
  result.erase(std::unique(result.begin(), result.end(),
                           [](const Shape& a, const Shape& b) {
                             return a.m == b.m && a.n == b.n && a.k == b.k;
                           }),
               result.end());
  return result;
}

CalculationsPrecision GetPrecision(const std::string& s) {
  if (s == "fp32") return CalculationsPrecision::F32;
  if (s == "fp16acc32") return CalculationsPrecision::F32_F16;
  return CalculationsPrecision::F16;
}

// Map matmul C[M,N] = A[M,K] * B[K,N] to 1x1 convolution:
//   input:  BHWC(1, M, 1, K)   — batch=1, height=M, width=1, channels=K
//   filter: OHWI(N, 1, 1, K)   — N output channels, 1x1 kernel, K input ch
//   output: BHWC(1, M, 1, N)
absl::Status BenchmarkShape(cl::Environment* env, const Shape& shape,
                            CalculationsPrecision precision,
                            int warmup, int iters, BenchResult* out) {
  const int M = shape.m;
  const int N = shape.n;
  const int K = shape.k;

  // Determine data type from precision.
  DataType data_type = precision == CalculationsPrecision::F32
                           ? DataType::FLOAT32
                           : DataType::FLOAT16;

  // Build OperationDef.
  OperationDef op_def;
  op_def.precision = precision;

  // Use the fastest storage type for this GPU (TEXTURE_ARRAY or TEXTURE_2D
  // on Adreno, not BUFFER which is ~100x slower).
  TensorStorageType storage = cl::GetFastestStorageType(
      env->GetDevicePtr()->info_);
  TensorDescriptor src_desc =
      TensorDescriptor(data_type, storage, Layout::HWC);
  src_desc.SetBHWCShape(BHWC(1, M, 1, K));
  TensorDescriptor dst_desc =
      TensorDescriptor(data_type, storage, Layout::HWC);
  dst_desc.SetBHWCShape(BHWC(1, M, 1, N));

  op_def.src_tensors.push_back(src_desc);
  op_def.dst_tensors.push_back(dst_desc);

  // Set up Convolution2DAttributes for 1x1 conv (= matmul).
  Convolution2DAttributes attr;
  attr.padding.prepended = HW(0, 0);
  attr.padding.appended = HW(0, 0);
  attr.strides = HW(1, 1);
  attr.dilations = HW(1, 1);

  // Weights: OHWI(N, 1, 1, K) — type is fixed to FLOAT32 by template.
  attr.weights.shape = OHWI(N, 1, 1, K);
  attr.weights.data.resize(N * K);
  for (int i = 0; i < N * K; ++i) attr.weights.data[i] = 0.01f;

  // Bias: N — type is fixed to FLOAT32 by template.
  attr.bias.shape = Linear(N);
  attr.bias.data.resize(N, 0.0f);

  // Create ConvGeneric — this selects Adreno-optimized kernel params.
  BHWC dst_shape(1, M, 1, N);
  ConvGeneric conv = CreateConvGeneric(env->GetDevicePtr()->info_, op_def, attr,
                                       &dst_shape);

  // AssembleCode finalizes the CL source and registers weight texture
  // argument slots (e.g. weights0_slice_stride). Must be called before
  // Compile so cl_args_.Init() can find all required arguments.
  RETURN_IF_ERROR(conv.AssembleCode(env->GetDevicePtr()->info_));

  // Create GPU tensors.
  cl::Tensor src_tensor, dst_tensor;
  TensorDescriptor dst_desc_with_shape = dst_desc;
  dst_desc_with_shape.SetBHWCShape(dst_shape);
  RETURN_IF_ERROR(
      src_tensor.CreateFromDescriptor(src_desc, &env->context()));
  RETURN_IF_ERROR(
      cl::CreateTensor(env->context(), dst_desc_with_shape, &dst_tensor));

  // Bind tensors to the operation BEFORE wrapping in ClOperation.
  conv.SetSrc(&src_tensor, 0);
  conv.SetDst(&dst_tensor, 0);

  // Set up ClOperation: Init → Compile → UpdateParams.
  cl::ClOperation cl_op;
  cl_op.Init(std::make_unique<ConvGeneric>(std::move(conv)));
  {
    cl::CreationContext creation_context;
    creation_context.device = env->GetDevicePtr();
    creation_context.context = &env->context();
    creation_context.queue = env->queue();
    creation_context.cache = env->program_cache();
    RETURN_IF_ERROR(cl_op.Compile(creation_context));
  }
  RETURN_IF_ERROR(cl_op.UpdateParams());

  // Warmup.
  for (int i = 0; i < warmup; ++i) {
    RETURN_IF_ERROR(cl_op.AddToQueue(env->queue()));
  }
  RETURN_IF_ERROR(env->queue()->WaitForCompletion());

  // Wall-clock: burst-submit + single sync.
  auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iters; ++i) {
    RETURN_IF_ERROR(cl_op.AddToQueue(env->queue()));
  }
  RETURN_IF_ERROR(env->queue()->WaitForCompletion());
  auto t1 = std::chrono::high_resolution_clock::now();
  double wall_total_us =
      std::chrono::duration<double, std::micro>(t1 - t0).count();
  double wall_avg_us = wall_total_us / iters;

  // GPU profiling: dispatch one at a time with events.
  double gpu_avg_us = 0;
  if (env->profiling_queue()) {
    std::vector<double> gpu_times;
    gpu_times.reserve(iters);
    for (int i = 0; i < iters; ++i) {
      cl::CLEvent event;
      RETURN_IF_ERROR(
          cl_op.AddToQueue(env->profiling_queue(), &event));
      RETURN_IF_ERROR(env->profiling_queue()->WaitForCompletion());
      gpu_times.push_back(event.GetEventTimeMs() * 1000.0);
    }
    gpu_avg_us = std::accumulate(gpu_times.begin(), gpu_times.end(), 0.0) /
                 gpu_times.size();
  }

  double gflops = shape.gflops();
  out->shape = shape;
  out->count = iters;
  out->wall_avg_us = wall_avg_us;
  out->gpu_avg_us = gpu_avg_us;
  out->tflops_wall = (gflops / (wall_avg_us / 1e6)) / 1000.0;
  out->tflops_gpu =
      gpu_avg_us > 0 ? (gflops / (gpu_avg_us / 1e6)) / 1000.0 : 0;
  return absl::OkStatus();
}

absl::Status Run() {
  std::string shapes_str = absl::GetFlag(FLAGS_shapes);
  std::string shapes_csv = absl::GetFlag(FLAGS_shapes_csv);
  std::string csv_out = absl::GetFlag(FLAGS_csv_out);
  std::string precision_str = absl::GetFlag(FLAGS_precision);
  int warmup = absl::GetFlag(FLAGS_warmup);
  int iters = absl::GetFlag(FLAGS_iters);
  int max_tensor_mb = absl::GetFlag(FLAGS_max_tensor_mb);

  std::vector<Shape> shapes;
  if (!shapes_str.empty()) {
    shapes = ParseShapes(shapes_str);
  } else if (!shapes_csv.empty()) {
    shapes = ParseShapesFromCsv(shapes_csv);
  } else {
    return absl::InvalidArgumentError(
        "Specify --shapes=MxNxK,... or --shapes_csv=path");
  }
  if (shapes.empty()) {
    return absl::InvalidArgumentError("No valid shapes");
  }

  CalculationsPrecision precision = GetPrecision(precision_str);
  fprintf(stderr,
          "[conv_bench] TFLite GPU ConvGeneric matmul benchmark\n"
          "[conv_bench] precision=%s warmup=%d iters=%d shapes=%zu\n",
          precision_str.c_str(), warmup, iters, shapes.size());

  // Load OpenCL and create environment.
  RETURN_IF_ERROR(cl::LoadOpenCL());
  cl::Environment env;
  RETURN_IF_ERROR(cl::CreateEnvironment(&env));
  env.SetHighPerformance();

  const auto& gpu_info = env.GetDevicePtr()->info_;
  fprintf(stderr, "[conv_bench] GPU: %s (%d CUs)\n",
          gpu_info.opencl_info.device_name.c_str(),
          gpu_info.GetComputeUnitsCount());

  std::vector<BenchResult> results;
  for (size_t i = 0; i < shapes.size(); ++i) {
    const auto& s = shapes[i];
    size_t elem_bytes = precision == CalculationsPrecision::F32 ? 4 : 2;
    size_t total_mb =
        ((size_t)s.m * s.k + (size_t)s.k * s.n + (size_t)s.m * s.n) *
        elem_bytes / (1024 * 1024);
    if ((int)total_mb > max_tensor_mb) {
      fprintf(stderr, "\n[conv_bench] [%zu/%zu] SKIP %dx%dx%d: %zuMB > %dMB\n",
              i + 1, shapes.size(), s.m, s.n, s.k, total_mb, max_tensor_mb);
      continue;
    }

    fprintf(stderr, "\n[conv_bench] [%zu/%zu] M=%d N=%d K=%d (%.3f GFLOPS)\n",
            i + 1, shapes.size(), s.m, s.n, s.k, s.gflops());

    BenchResult res;
    auto status = BenchmarkShape(&env, s, precision, warmup, iters, &res);
    if (!status.ok()) {
      fprintf(stderr, "  ERROR: %s\n", std::string(status.message()).c_str());
      continue;
    }

    fprintf(stderr,
            "  wall: avg=%.1f us  TFLOPS=%.3f\n"
            "  gpu:  avg=%.1f us  TFLOPS=%.3f\n",
            res.wall_avg_us, res.tflops_wall, res.gpu_avg_us, res.tflops_gpu);
    results.push_back(res);
  }

  // Write CSV.
  FILE* fp = stdout;
  bool close_fp = false;
  if (!csv_out.empty()) {
    fp = fopen(csv_out.c_str(), "w");
    if (!fp) {
      fprintf(stderr, "ERROR: cannot open %s\n", csv_out.c_str());
      fp = stdout;
    } else {
      close_fp = true;
    }
  }

  fprintf(fp,
          "m,n,k,count,wall_avg_us,gpu_avg_us,gflops,tflops_wall,tflops_gpu,"
          "precision\n");
  for (const auto& r : results) {
    fprintf(fp, "%d,%d,%d,%d,%.3f,%.3f,%.6f,%.6f,%.6f,%s\n", r.shape.m,
            r.shape.n, r.shape.k, r.count, r.wall_avg_us, r.gpu_avg_us,
            r.shape.gflops(), r.tflops_wall, r.tflops_gpu,
            precision_str.c_str());
  }
  if (close_fp) {
    fclose(fp);
    fprintf(stderr, "\n[conv_bench] CSV: %s\n", csv_out.c_str());
  }

  fprintf(stderr, "\n[conv_bench] Done. %zu shapes benchmarked.\n",
          results.size());
  return absl::OkStatus();
}

}  // namespace

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  auto status = Run();
  if (!status.ok()) {
    fprintf(stderr, "FATAL: %s\n", std::string(status.message()).c_str());
    return 1;
  }
  return 0;
}
