// Dump the CL kernel source that ConvGeneric generates for given shapes.
//
// Runs on device (needs OpenCL environment for GPU-specific kernel tuning).
// Outputs CL source files to --out_dir, one per shape:
//   conv_kernel_MxNxK.cl  — the CL source
//   conv_kernel_MxNxK.meta — kernel name, work group, grid, arg info

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/status/status.h"
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

ABSL_FLAG(std::string, shapes, "",
          "Shapes MxNxK,... (e.g. '1024x6144x1536')");
ABSL_FLAG(std::string, precision, "fp16", "fp16, fp32, or fp16acc32");
ABSL_FLAG(std::string, out_dir, ".", "Output directory for .cl files");

namespace {

using namespace tflite::gpu;

struct Shape {
  int m, n, k;
};

std::vector<Shape> ParseShapes(const std::string& s) {
  std::vector<Shape> result;
  std::stringstream ss(s);
  std::string tok;
  while (std::getline(ss, tok, ',')) {
    int m = 0, n = 0, k = 0;
    if (sscanf(tok.c_str(), "%dx%dx%d", &m, &n, &k) == 3 &&
        m > 0 && n > 0 && k > 0)
      result.push_back({m, n, k});
  }
  return result;
}

CalculationsPrecision GetPrecision(const std::string& s) {
  if (s == "fp32") return CalculationsPrecision::F32;
  if (s == "fp16acc32") return CalculationsPrecision::F32_F16;
  return CalculationsPrecision::F16;
}

absl::Status DumpKernel(cl::Environment* env, const Shape& shape,
                        CalculationsPrecision precision,
                        const std::string& out_dir) {
  const int M = shape.m, N = shape.n, K = shape.k;
  DataType data_type = precision == CalculationsPrecision::F32
                           ? DataType::FLOAT32
                           : DataType::FLOAT16;

  OperationDef op_def;
  op_def.precision = precision;
  TensorStorageType storage = TensorStorageType::IMAGE_BUFFER;
  TensorDescriptor src_desc(data_type, storage, Layout::HWC);
  src_desc.SetBHWCShape(BHWC(1, 1, M, K));
  TensorDescriptor dst_desc(data_type, storage, Layout::HWC);
  dst_desc.SetBHWCShape(BHWC(1, 1, M, N));
  op_def.src_tensors.push_back(src_desc);
  op_def.dst_tensors.push_back(dst_desc);

  Convolution2DAttributes attr;
  attr.padding.prepended = HW(0, 0);
  attr.padding.appended = HW(0, 0);
  attr.strides = HW(1, 1);
  attr.dilations = HW(1, 1);
  attr.weights.shape = OHWI(N, 1, 1, K);
  attr.weights.data.resize(N * K, 0.01f);
  attr.bias.shape = Linear(N);
  attr.bias.data.resize(N, 0.0f);

  BHWC dst_shape(1, 1, M, N);
  ConvGeneric conv = CreateConvGeneric(env->GetDevicePtr()->info_, op_def, attr,
                                       &dst_shape);

  // AssembleCode generates the final CL source with tensor accessors.
  RETURN_IF_ERROR(conv.AssembleCode(env->GetDevicePtr()->info_));

  // code_ contains the CL source after AssembleCode.
  const std::string& cl_source = conv.code_;

  // Get grid size info.
  cl::Tensor src_tensor, dst_tensor;
  TensorDescriptor dst_desc_ws = dst_desc;
  dst_desc_ws.SetBHWCShape(dst_shape);
  RETURN_IF_ERROR(src_tensor.CreateFromDescriptor(src_desc, &env->context()));
  RETURN_IF_ERROR(cl::CreateTensor(env->context(), dst_desc_ws, &dst_tensor));
  conv.SetSrc(&src_tensor, 0);
  conv.SetDst(&dst_tensor, 0);
  auto grid = conv.GetGridSize();
  auto wg = conv.work_group_size_;

  // Save CL source.
  char fname[256];
  snprintf(fname, sizeof(fname), "%s/conv_kernel_%dx%dx%d.cl",
           out_dir.c_str(), M, N, K);
  {
    std::ofstream f(fname);
    if (!f.is_open())
      return absl::NotFoundError(std::string("Cannot open ") + fname);
    f << cl_source;
  }
  fprintf(stderr, "[dump] Saved %s (%zu bytes)\n", fname, cl_source.size());

  // Save metadata.
  char meta_fname[256];
  snprintf(meta_fname, sizeof(meta_fname), "%s/conv_kernel_%dx%dx%d.meta",
           out_dir.c_str(), M, N, K);
  {
    std::ofstream f(meta_fname);
    if (!f.is_open())
      return absl::NotFoundError(std::string("Cannot open ") + meta_fname);
    f << "shape=" << M << "x" << N << "x" << K << "\n";
    f << "precision=" << (precision == CalculationsPrecision::F16 ? "fp16" :
                          precision == CalculationsPrecision::F32 ? "fp32" :
                          "fp16acc32") << "\n";
    f << "storage=IMAGE_BUFFER\n";
    f << "work_group=" << wg.x << "," << wg.y << "," << wg.z << "\n";
    f << "grid=" << grid.x << "," << grid.y << "," << grid.z << "\n";
    f << "cl_source_bytes=" << cl_source.size() << "\n";
  }
  fprintf(stderr, "[dump] Saved %s\n", meta_fname);
  return absl::OkStatus();
}

absl::Status Run() {
  std::string shapes_str = absl::GetFlag(FLAGS_shapes);
  std::string precision_str = absl::GetFlag(FLAGS_precision);
  std::string out_dir = absl::GetFlag(FLAGS_out_dir);

  if (shapes_str.empty())
    return absl::InvalidArgumentError("--shapes required");

  auto shapes = ParseShapes(shapes_str);
  if (shapes.empty())
    return absl::InvalidArgumentError("No valid shapes");

  auto precision = GetPrecision(precision_str);

  RETURN_IF_ERROR(cl::LoadOpenCL());
  cl::Environment env;
  RETURN_IF_ERROR(cl::CreateEnvironment(&env));

  const auto& gpu = env.GetDevicePtr()->info_;
  fprintf(stderr, "[dump] GPU: %s\n", gpu.opencl_info.device_name.c_str());
  fprintf(stderr, "[dump] Precision: %s, Shapes: %zu\n",
          precision_str.c_str(), shapes.size());

  for (const auto& s : shapes) {
    fprintf(stderr, "\n[dump] Generating CL for %dx%dx%d ...\n", s.m, s.n, s.k);
    auto status = DumpKernel(&env, s, precision, out_dir);
    if (!status.ok()) {
      fprintf(stderr, "  ERROR: %s\n", std::string(status.message()).c_str());
    }
  }
  fprintf(stderr, "\n[dump] Done.\n");
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
