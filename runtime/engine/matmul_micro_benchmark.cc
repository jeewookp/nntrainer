// Copyright 2026 The ODML Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Standalone matmul micro benchmark for LiteRT GPU (and CPU) backends.
//
// Purpose: the LiteRT GPU delegate's per-op profiler aggregates all
// dispatches of the same fused kernel name into a single row (e.g.
// `Delegate/convolution(conv_wave_memory) 28 907.68 ...`), which makes
// it impossible to attribute per-(M,N,K) cost when multiple matmul
// shapes share the same kernel. This binary builds a minimal
// FULLY_CONNECTED-only tflite model per unique (M,N,K), compiles it
// with litert::CompiledModel on the requested backend, runs warmup +
// timed iterations, and prints a CSV row per shape. The results can
// be placed next to the prefill delegate-stats table in
// temp_litert_run.log to attribute the bulk of prefill to specific
// matmul shapes.
//
// Shape sources (pick one or combine):
//   --shapes="1024x1536x6144,1024x6144x1536,..."        inline list
//   --shapes_csv=/path/to/matmul_roster.csv             CSV from the
//                                                        shape dumper
//                                                        (set LITERT_LM_MATMUL_ROSTER_CSV
//                                                        at prefill time
//                                                        to produce it)
//
// Example:
//   LITERT_LM_MATMUL_ROSTER_CSV=/data/local/tmp/roster.csv \
//     ./litert_lm_main ... --enable_op_profiling=true   # produces the roster
//   ./matmul_micro_benchmark --backend=gpu \
//     --shapes_csv=/data/local/tmp/roster.csv \
//     --iters=50 --warmup=5 \
//     --csv_out=/data/local/tmp/matmul_micro.csv

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/base/log_severity.h"  // from @com_google_absl
#include "absl/flags/flag.h"  // from @com_google_absl
#include "absl/flags/parse.h"  // from @com_google_absl
#include "absl/log/globals.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/ascii.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/numbers.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_split.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "absl/time/clock.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "litert/c/litert_compiled_model.h"  // from @litert
#include "litert/c/litert_profiler.h"  // from @litert
#include "litert/cc/litert_buffer_ref.h"  // from @litert
#include "litert/cc/litert_common.h"  // from @litert
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_model.h"  // from @litert
#include "litert/cc/litert_options.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "litert/cc/options/litert_gpu_options.h"  // from @litert
#include "litert/cc/options/litert_runtime_options.h"  // from @litert
#include "runtime/util/single_op_tflite_builder.h"

ABSL_FLAG(std::string, backend, "gpu",
          "Backend to benchmark on: 'gpu' or 'cpu'. GPU uses the same "
          "LiteRT CL path as the full Gemma4 prefill run.");
ABSL_FLAG(std::string, shapes, "",
          "Comma-separated MxNxK matmul shapes to benchmark, e.g. "
          "'1024x1536x6144,1024x6144x1536'. Empty = use --shapes_csv.");
ABSL_FLAG(std::string, shapes_csv, "",
          "Path to a matmul roster CSV emitted by "
          "runtime/util/matmul_shape_dump when prefill was run with "
          "LITERT_LM_MATMUL_ROSTER_CSV set. Each row with a positive "
          "m/n/k is picked up as a benchmark point. Duplicate "
          "(m,n,k) tuples are deduplicated.");
ABSL_FLAG(int, warmup, 5,
          "Warmup iterations before timing starts. These are not "
          "included in the reported stats but still contribute to "
          "on-device kernel compilation / weight upload.");
ABSL_FLAG(int, iters, 50,
          "Number of timed iterations per shape. Reported stats are "
          "avg / min / p50 / p95 / max in microseconds.");
ABSL_FLAG(std::string, csv_out, "",
          "Optional: write a machine-readable CSV of the results to "
          "this path. Columns: "
          "m,n,k,count,min_us,avg_us,p50_us,p95_us,max_us,flops,"
          "tflops_min,tflops_avg.");
ABSL_FLAG(int, max_shapes, 0,
          "If > 0, only benchmark the first N unique shapes from the "
          "input. Useful for smoke-testing on large rosters.");
ABSL_FLAG(bool, profile, false,
          "Enable LiteRT op profiling for each shape and dump the "
          "Delegate Statistics summary at the end. This wires up "
          "RuntimeOptions::SetEnableProfiling(true) on the GPU "
          "compilation options, retrieves the profiler handle via "
          "LiteRtCompiledModelGetProfiler, and brackets each timed "
          "iteration with LiteRtStartProfiler / LiteRtStopProfiler. "
          "After the timed loop the LiteRtGetProfileSummary string "
          "is printed to stderr -- this is what tells you which "
          "OpenCL kernel the GPU CL delegate actually picked for the "
          "shape (e.g. `convolution_int8(conv_wave_memory)` vs the "
          "fp32 fallback). Adds noticeable per-Run overhead so use "
          "for diagnostics, not steady-state numbers.");
ABSL_FLAG(std::string, dtype, "int8_per_tensor",
          "Tensor element type / op layout for the synthesized model. "
          "Valid values:\n"
          "  'int8_per_tensor' (default): single FC op with INT8 "
          "weights using PER-TENSOR symmetric quant (single scale + "
          "single zero_point), FLOAT32 bias, FLOAT32 signature I/O. "
          "Matches the open-source TFLite GPU CL delegate's int8 FC "
          "trigger (lstm_parser.cc:62: `weights_tensor->type == "
          "kTfLiteInt8 && quant_params->scale->size == 1`). All "
          "previous int8 attempts in this builder used per-CHANNEL "
          "weight quant (n scales) which the trigger explicitly "
          "rejects. Expected to compile as "
          "`convolution_int8(conv_wave_memory)` matching prefill.\n"
          "  'int8_chain': 4-op subgraph QUANTIZE -> FC1(int8) -> "
          "FC2(int8) -> DEQUANTIZE. Failed experiment: chains the "
          "live int8 intermediate but the delegate still picks "
          "fp16 conv1x1.\n"
          "  'fp32_conv2d': single CONV_2D op with 1x1 filter, NHWC "
          "layout (input [1,M,1,K], filter [N,1,1,K]), FLOAT32 "
          "throughout. Profile-confirmed to land on the same "
          "`convolution1x1(conv_wave_memory)` entry point as the FC "
          "modes -- the `1x1` suffix is the filter size, not an FC "
          "rewrite marker.\n"
          "  'int8': single-op wrapped int8 FC. The delegate fuses "
          "QUANTIZE+FC+DEQUANTIZE but compiles the conv as fp16 "
          "(no live int8 tensor). Kept as a comparison point.\n"
          "  'int8_hybrid': single FC op, int8 weights + fp32 "
          "input/output. Hybrid path, accepted but NOT lowered to "
          "int8.\n"
          "  'fp32': FLOAT32 single FC op. Baseline.\n"
          "  'fp16': FLOAT16 FC, currently broken (CPU reference "
          "kernel asserts fp32 input during prepare).");
ABSL_FLAG(std::string, cache_dir, "",
          "If non-empty, use this directory as the LiteRT GPU "
          "OpenCL program cache (via GpuOptions::SetSerializationDir) "
          "and serialize compiled weights "
          "(SetSerializeExternalTensors=true). A per-(M,N,K,dtype) "
          "cache key is derived and passed to SetModelCacheKey so "
          "entries do not collide across shapes.\n\n"
          "First run populates the cache: each CompiledModel::Create "
          "still pays the full OpenCL JIT + weight upload cost, and "
          "the compiled program + external tensors are written under "
          "cache_dir. Subsequent runs (same shape, same dtype, same "
          "GPU) skip the JIT and stream the cached program back in "
          "-- the 'per-shape CompiledModel::Create' wall-clock "
          "overhead drops by roughly the JIT cost (commonly tens to "
          "hundreds of ms per shape on Adreno). Steady-state "
          "per-iteration latency (min/avg/p50) is unaffected either "
          "way -- this is purely a warm-start optimization for "
          "people re-running the same roster over multiple "
          "iterations.\n\n"
          "Leave empty to disable (default). Use a persistent path "
          "like /data/local/tmp/litert_lm/cache so the cache survives "
          "benchmark-script reruns; a tmpfs path defeats the "
          "purpose.");

// In recent LiteRT revisions `litert::CompiledModel::Create` is a
// protected static factory: only subclasses can call it. The same
// pattern that runtime/executor/litert/kv_cache_test.cc uses -- derive
// a trivial subclass and re-export Create via a using-declaration --
// lets us call it from free functions here without patching LiteRT.
// The resulting object is still a litert::CompiledModel so it can be
// moved back into the base type after construction.
class CompiledModelAccess : public litert::CompiledModel {
 public:
  using litert::CompiledModel::CompiledModel;
  using litert::CompiledModel::Create;
};

struct Shape {
  int64_t m = 0;
  int64_t n = 0;
  int64_t k = 0;

  bool operator<(const Shape& other) const {
    return std::tie(m, n, k) < std::tie(other.m, other.n, other.k);
  }
};

absl::StatusOr<Shape> ParseShape(absl::string_view s) {
  // Accepts "MxNxK" or "M,N,K". Whitespace tolerant.
  std::vector<absl::string_view> parts;
  if (absl::StrContains(s, 'x') || absl::StrContains(s, 'X')) {
    parts = absl::StrSplit(s, absl::ByAnyChar("xX"));
  } else {
    parts = absl::StrSplit(s, ',');
  }
  if (parts.size() != 3) {
    return absl::InvalidArgumentError(
        absl::StrCat("Shape must have 3 components (MxNxK), got: ", s));
  }
  Shape shape;
  for (auto& p : parts) {
    p = absl::StripAsciiWhitespace(p);
  }
  if (!absl::SimpleAtoi(parts[0], &shape.m) ||
      !absl::SimpleAtoi(parts[1], &shape.n) ||
      !absl::SimpleAtoi(parts[2], &shape.k)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Shape has non-integer components: ", s));
  }
  if (shape.m <= 0 || shape.n <= 0 || shape.k <= 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("Shape has non-positive components: ", s));
  }
  return shape;
}

absl::StatusOr<std::vector<Shape>> LoadShapesFromCsv(
    absl::string_view csv_path) {
  std::ifstream ifs((std::string(csv_path)));
  if (!ifs) {
    return absl::NotFoundError(
        absl::StrCat("LoadShapesFromCsv: cannot open ", csv_path));
  }
  std::vector<Shape> shapes;
  std::string line;
  bool first = true;
  // Expected columns (matches runtime/util/matmul_shape_dump.cc):
  //   label, subgraph_idx, subgraph_name, op_idx, op_type,
  //   m, n, k, input_shapes, output_shapes
  // Indices 5/6/7 are m/n/k.
  while (std::getline(ifs, line)) {
    if (first) {
      first = false;
      continue;  // Skip CSV header
    }
    if (line.empty()) continue;
    std::vector<absl::string_view> cols = absl::StrSplit(line, ',');
    if (cols.size() < 8) continue;
    Shape s;
    if (!absl::SimpleAtoi(cols[5], &s.m)) continue;
    if (!absl::SimpleAtoi(cols[6], &s.n)) continue;
    if (!absl::SimpleAtoi(cols[7], &s.k)) continue;
    if (s.m <= 0 || s.n <= 0 || s.k <= 0) continue;
    shapes.push_back(s);
  }
  return shapes;
}

std::vector<Shape> Dedup(const std::vector<Shape>& in) {
  std::set<Shape> seen;
  std::vector<Shape> out;
  out.reserve(in.size());
  for (const auto& s : in) {
    if (seen.insert(s).second) out.push_back(s);
  }
  return out;
}

struct Stats {
  int64_t count = 0;
  double min_us = 0;
  double avg_us = 0;
  double p50_us = 0;
  double p95_us = 0;
  double max_us = 0;
};

Stats ComputeStats(std::vector<double> samples_us) {
  Stats s;
  if (samples_us.empty()) return s;
  std::sort(samples_us.begin(), samples_us.end());
  s.count = static_cast<int64_t>(samples_us.size());
  s.min_us = samples_us.front();
  s.max_us = samples_us.back();
  double sum = 0;
  for (double v : samples_us) sum += v;
  s.avg_us = sum / s.count;
  const size_t p50_idx = static_cast<size_t>(0.50 * (s.count - 1));
  const size_t p95_idx = static_cast<size_t>(0.95 * (s.count - 1));
  s.p50_us = samples_us[p50_idx];
  s.p95_us = samples_us[p95_idx];
  return s;
}

// Helper: convert a litert::Expected<T> error into an absl::InternalError
// with a contextual prefix. `ctx` is interpolated into the message.
template <typename T>
absl::Status ExpectedError(const T& exp, absl::string_view ctx) {
  return absl::InternalError(
      absl::StrCat(ctx, ": ", exp.Error().Message()));
}

// Builds, compiles, and times one matmul shape. Returns the per-iteration
// latency samples so the caller can compute aggregate stats. When
// `enable_profile` is true, also enables LiteRT op profiling on the GPU
// compilation options and prints the Delegate Statistics summary to
// stderr after the timed loop -- this is the canonical way to see
// which CL kernel the GPU delegate actually picked for the shape.
absl::StatusOr<std::vector<double>> BenchmarkOneShape(
    litert::Environment& env, const Shape& shape,
    absl::string_view backend_str, litert::lm::MatmulDtype dtype,
    int warmup_iters, int timed_iters, bool enable_profile,
    absl::string_view cache_dir, absl::string_view dtype_str) {
  auto build_or = litert::lm::BuildSingleFullyConnectedTfliteModel(
      shape.m, shape.n, shape.k, dtype);
  if (!build_or.ok()) return build_or.status();
  const litert::lm::SingleFcBuildResult& built = *build_or;

  // Wrap the raw flatbuffer bytes as a litert::BufferRef (non-owning).
  // The bytes stay alive for the duration of the local `built` object,
  // which outlives the compiled model in this function.
  litert::BufferRef<uint8_t> buffer_ref(
      reinterpret_cast<const uint8_t*>(built.flatbuffer.data()),
      built.flatbuffer.size());

  auto model_exp = litert::Model::CreateFromBuffer(buffer_ref);
  if (!model_exp.HasValue()) {
    return ExpectedError(
        model_exp,
        absl::StrCat("Model::CreateFromBuffer m=", shape.m, " n=", shape.n,
                     " k=", shape.k));
  }
  litert::Model model = std::move(model_exp.Value());

  // Compilation options: GPU path matches the Gemma4 prefill run
  // (fp16 precision, texture weights on non-Apple platforms).
  auto options_exp = litert::Options::Create();
  if (!options_exp.HasValue()) {
    return ExpectedError(options_exp, "Options::Create");
  }
  litert::Options options = std::move(options_exp.Value());

  if (backend_str == "gpu") {
    auto gpu_opts_exp = options.GetGpuOptions();
    if (!gpu_opts_exp.HasValue()) {
      return ExpectedError(gpu_opts_exp, "Options::GetGpuOptions");
    }
    auto& gpu_opts = gpu_opts_exp.Value();
    gpu_opts.EnableInfiniteFloatCapping(true);
    gpu_opts.SetPrecision(litert::GpuOptions::Precision::kFp16);
#if !defined(__APPLE__)
    gpu_opts.SetPreferTextureWeights(true);
#endif
    // OpenCL program cache wiring. When --cache_dir is set, tell the
    // GPU compilation options where to serialize compiled programs +
    // external tensor data, and attach a per-(M,N,K,dtype) cache key
    // so entries from different shapes don't collide. Matches the
    // production LLM path in
    // runtime/executor/llm_executor_settings_utils.cc:127-146.
    //
    // First run: JIT compiles the CL kernel, writes program + tensors
    // under cache_dir, steady-state numbers are unchanged.
    // Second run (same shape/dtype/GPU): reads the cached program,
    // skips JIT, CompiledModel::Create returns in a fraction of the
    // first-run time. Per-iteration latency in the timed loop is
    // unaffected either way -- the cache only fixes wall-clock of
    // CompiledModel::Create, not per-dispatch.
    if (!cache_dir.empty()) {
      const std::string cache_dir_str(cache_dir);
      gpu_opts.SetSerializationDir(cache_dir_str.c_str());
      gpu_opts.SetSerializeExternalTensors(true);
      const std::string model_cache_key =
          absl::StrCat("matmul_bench_m", shape.m, "_n", shape.n, "_k",
                       shape.k, "_", dtype_str);
      gpu_opts.SetModelCacheKey(model_cache_key.c_str());
    }
    options.SetHardwareAccelerators(litert::HwAccelerators::kGpu);
  } else if (backend_str == "cpu") {
    options.SetHardwareAccelerators(litert::HwAccelerators::kCpu);
  } else {
    return absl::InvalidArgumentError(
        absl::StrCat("Unknown --backend: ", backend_str));
  }

  // Op profiling: arms the tflite::Profiler hooks inside the compiled
  // model. We retrieve the LiteRtProfiler handle below, after
  // CompiledModel::Create, and Start/Stop it around each timed
  // iteration. The hooks survive across multiple Run() calls so the
  // final GetProfileSummary covers all `timed_iters` invocations.
  if (enable_profile) {
    auto runtime_opts_exp = options.GetRuntimeOptions();
    if (!runtime_opts_exp.HasValue()) {
      return ExpectedError(runtime_opts_exp, "Options::GetRuntimeOptions");
    }
    auto& runtime_opts = runtime_opts_exp.Value();
    auto set_exp = runtime_opts.SetEnableProfiling(true);
    if (!set_exp.HasValue()) {
      return ExpectedError(set_exp, "RuntimeOptions::SetEnableProfiling");
    }
  }

  auto compiled_model_exp =
      CompiledModelAccess::Create(env, model.Get(), options);
  if (!compiled_model_exp.HasValue()) {
    return ExpectedError(
        compiled_model_exp,
        absl::StrCat("CompiledModel::Create m=", shape.m, " n=", shape.n,
                     " k=", shape.k));
  }
  litert::CompiledModel compiled_model = std::move(compiled_model_exp.Value());

  // Grab the LiteRtProfiler handle if op profiling was enabled. The
  // handle is non-owning -- the compiled model owns the underlying
  // tflite::Profiler. A nullptr return means profiling wasn't actually
  // armed (e.g. on a non-GPU backend that ignores the runtime option),
  // in which case we just skip Start/Stop and the summary dump.
  LiteRtProfiler profiler_handle = nullptr;
  if (enable_profile) {
    (void)LiteRtCompiledModelGetProfiler(compiled_model.Get(),
                                          &profiler_handle);
  }

  auto input_buffers_exp =
      compiled_model.CreateInputBuffers(built.signature_key);
  if (!input_buffers_exp.HasValue()) {
    return ExpectedError(input_buffers_exp, "CreateInputBuffers");
  }
  auto input_buffers = std::move(input_buffers_exp.Value());

  auto output_buffers_exp =
      compiled_model.CreateOutputBuffers(built.signature_key);
  if (!output_buffers_exp.HasValue()) {
    return ExpectedError(output_buffers_exp, "CreateOutputBuffers");
  }
  auto output_buffers = std::move(output_buffers_exp.Value());

  // Fill the input activation buffer with all-zeros. We don't care
  // about exact numerical values for a latency-only benchmark: GPU
  // CL kernel dispatch time depends on shape / precision / weight
  // layout, not on activation content. Zeros also avoid NaN/Inf
  // propagation in the accumulator.
  //
  // The signature input tensor type per dtype mode:
  //   int8        : FLOAT32 input -> wrapped 3-op subgraph
  //                 (QUANTIZE -> int8 FC -> DEQUANTIZE) inside the
  //                 model. Host writes float zeros.
  //   int8_hybrid : FLOAT32 input -> dynamic int8 quant inside FC
  //   fp32        : FLOAT32 input
  //   fp16        : FLOAT16 input
  {
    const size_t num_elements = static_cast<size_t>(shape.m * shape.k);
    if (dtype == litert::lm::MatmulDtype::kFp16) {
      std::vector<uint16_t> zeros(num_elements, 0);
      auto write_exp =
          input_buffers[0].Write(absl::MakeConstSpan(zeros));
      if (!write_exp.HasValue()) {
        return ExpectedError(write_exp, "input_buffers[0].Write fp16");
      }
    } else {
      // fp32 input shared by kFp32, kInt8WeightFp32Act, and kInt8
      // (the wrapped int8 path keeps its signature input fp32 and
      // does the int8 quant inside the subgraph).
      std::vector<float> zeros(num_elements, 0.0f);
      auto write_exp =
          input_buffers[0].Write(absl::MakeConstSpan(zeros));
      if (!write_exp.HasValue()) {
        return ExpectedError(write_exp, "input_buffers[0].Write fp32");
      }
    }
  }

  // Warmup. The first Run() on a fresh compiled model triggers program
  // / weight upload on GPU, which is not the latency we want to report.
  for (int i = 0; i < warmup_iters; ++i) {
    auto run_exp = compiled_model.Run(built.signature_key, input_buffers,
                                       output_buffers);
    if (!run_exp.HasValue()) {
      return ExpectedError(
          run_exp,
          absl::StrCat("Run() warmup m=", shape.m, " n=", shape.n,
                       " k=", shape.k));
    }
  }

  // Timed runs. Wall-clock covers Run() + a read-mode scoped lock on
  // the output tensor. The scoped lock serves two purposes:
  //   1. On GPU, creating a kRead lock forces the LiteRT CL delegate to
  //      wait for the dispatched kernel to finish and to make the
  //      output host-accessible. That guarantees the t1 reading is
  //      taken _after_ the GPU work completes, not just after the
  //      submit call returns.
  //   2. It prevents the GPU from overlapping consecutive Run() calls
  //      across benchmark iterations, which would otherwise inflate
  //      throughput above the true per-dispatch cost.
  // The lock itself doesn't copy any data host-side -- it just yields
  // a const pointer to the mapped buffer -- so the overhead is
  // negligible compared to the matmul kernel time (hundreds of
  // microseconds and up for Gemma4 prefill shapes).
  //
  // If op profiling is enabled, Start/Stop the LiteRtProfiler around
  // the entire timed loop so the per-op event buffer covers all
  // `timed_iters` invocations. Doing it once around the whole loop
  // (instead of per iteration) keeps the start/stop overhead out of
  // the wall-clock samples below.
  if (enable_profile && profiler_handle != nullptr) {
    (void)LiteRtStartProfiler(profiler_handle);
  }
  std::vector<double> samples_us;
  samples_us.reserve(timed_iters);
  for (int i = 0; i < timed_iters; ++i) {
    const absl::Time t0 = absl::Now();
    auto run_exp = compiled_model.Run(built.signature_key, input_buffers,
                                       output_buffers);
    if (!run_exp.HasValue()) {
      return ExpectedError(
          run_exp,
          absl::StrCat("Run() timed m=", shape.m, " n=", shape.n,
                       " k=", shape.k));
    }
    {
      auto lock_exp = litert::TensorBufferScopedLock::Create(
          output_buffers[0], litert::TensorBuffer::LockMode::kRead);
      if (!lock_exp.HasValue()) {
        return ExpectedError(
            lock_exp,
            absl::StrCat("TensorBufferScopedLock m=", shape.m,
                         " n=", shape.n, " k=", shape.k));
      }
      // `lock_exp.Value()` is a (ScopedLock, void*) pair. We move it
      // into a named local so the lock stays alive until the end of
      // this block, then touch the first byte through a volatile
      // pointer to keep the compiler from optimizing the mapping
      // (and therefore the GPU -> host sync) away. Cost: one volatile
      // load.
      auto lock_and_addr = std::move(lock_exp.Value());
      volatile uint8_t* mapped =
          static_cast<volatile uint8_t*>(lock_and_addr.second);
      if (mapped != nullptr) {
        (void)mapped[0];
      }
    }
    const absl::Time t1 = absl::Now();
    samples_us.push_back(absl::ToDoubleMicroseconds(t1 - t0));
  }

  // Stop the profiler and dump the per-op summary so we can see which
  // CL kernel the GPU delegate actually picked. The summary string is
  // owned by the caller and must be std::free'd after use.
  if (enable_profile && profiler_handle != nullptr) {
    (void)LiteRtStopProfiler(profiler_handle);
    const char* summary = nullptr;
    LiteRtStatus status = LiteRtGetProfileSummary(
        profiler_handle, compiled_model.Get(), &summary);
    if (status == kLiteRtStatusOk && summary != nullptr) {
      std::fprintf(stderr,
                   "\n[MATMUL MICRO PROFILE] m=%lld n=%lld k=%lld\n%s\n",
                   static_cast<long long>(shape.m),
                   static_cast<long long>(shape.n),
                   static_cast<long long>(shape.k), summary);
      std::free(const_cast<char*>(summary));
    } else {
      std::fprintf(stderr,
                   "\n[MATMUL MICRO PROFILE] m=%lld n=%lld k=%lld: "
                   "LiteRtGetProfileSummary failed (status=%d)\n",
                   static_cast<long long>(shape.m),
                   static_cast<long long>(shape.n),
                   static_cast<long long>(shape.k), static_cast<int>(status));
    }
  }
  return samples_us;
}

absl::Status MainBody() {
  const std::string backend = absl::GetFlag(FLAGS_backend);
  const std::string shapes_flag = absl::GetFlag(FLAGS_shapes);
  const std::string shapes_csv = absl::GetFlag(FLAGS_shapes_csv);
  const int warmup = absl::GetFlag(FLAGS_warmup);
  const int iters = absl::GetFlag(FLAGS_iters);
  const std::string csv_out = absl::GetFlag(FLAGS_csv_out);
  const int max_shapes = absl::GetFlag(FLAGS_max_shapes);
  const std::string dtype_str = absl::GetFlag(FLAGS_dtype);
  const bool enable_profile = absl::GetFlag(FLAGS_profile);
  const std::string cache_dir = absl::GetFlag(FLAGS_cache_dir);

  if (warmup < 0 || iters <= 0) {
    return absl::InvalidArgumentError(
        "--warmup must be >= 0 and --iters must be > 0");
  }
  litert::lm::MatmulDtype dtype;
  if (dtype_str == "int8_per_tensor") {
    dtype = litert::lm::MatmulDtype::kInt8PerTensor;
  } else if (dtype_str == "int8") {
    dtype = litert::lm::MatmulDtype::kInt8;
  } else if (dtype_str == "int8_chain") {
    dtype = litert::lm::MatmulDtype::kInt8Chain;
  } else if (dtype_str == "int8_hybrid") {
    dtype = litert::lm::MatmulDtype::kInt8WeightFp32Act;
  } else if (dtype_str == "fp32") {
    dtype = litert::lm::MatmulDtype::kFp32;
  } else if (dtype_str == "fp32_conv2d") {
    dtype = litert::lm::MatmulDtype::kFp32Conv2d;
  } else if (dtype_str == "fp16") {
    dtype = litert::lm::MatmulDtype::kFp16;
  } else {
    return absl::InvalidArgumentError(absl::StrCat(
        "--dtype must be int8_per_tensor, int8, int8_chain, int8_hybrid, "
        "fp32, fp32_conv2d, or fp16, got: ",
        dtype_str));
  }
  // int8_chain mode runs two FCs per Run, so divide the per-Run
  // wall-clock samples by this number to get per-FC latency that's
  // comparable to single-op modes and to prefill numbers. Other
  // modes use 1 (raw per-Run timing).
  const int chain_divisor =
      (dtype == litert::lm::MatmulDtype::kInt8Chain) ? 2 : 1;

  // Gather shapes.
  std::vector<Shape> shapes;
  if (!shapes_flag.empty()) {
    for (absl::string_view raw : absl::StrSplit(shapes_flag, ',')) {
      raw = absl::StripAsciiWhitespace(raw);
      if (raw.empty()) continue;
      auto s_or = ParseShape(raw);
      if (!s_or.ok()) return s_or.status();
      shapes.push_back(*s_or);
    }
  }
  if (!shapes_csv.empty()) {
    auto csv_shapes_or = LoadShapesFromCsv(shapes_csv);
    if (!csv_shapes_or.ok()) return csv_shapes_or.status();
    for (const auto& s : *csv_shapes_or) shapes.push_back(s);
  }
  if (shapes.empty()) {
    return absl::InvalidArgumentError(
        "No shapes to benchmark. Pass --shapes= or --shapes_csv=.");
  }
  shapes = Dedup(shapes);
  // Sort so the output CSV is deterministic (M ascending, then N, then K).
  std::sort(shapes.begin(), shapes.end());
  if (max_shapes > 0 && static_cast<int>(shapes.size()) > max_shapes) {
    shapes.resize(max_shapes);
  }

  std::fprintf(stderr,
               "\n[MATMUL MICRO] backend=%s dtype=%s unique_shapes=%zu "
               "warmup=%d iters=%d profile=%d cache_dir=%s\n",
               backend.c_str(), dtype_str.c_str(), shapes.size(), warmup,
               iters, enable_profile ? 1 : 0,
               cache_dir.empty() ? "(disabled)" : cache_dir.c_str());

  // Create one LiteRT environment for the whole run; CompiledModels are
  // built / destroyed per shape so the GPU delegate's internal caches
  // reset and we don't get cross-shape contamination.
  auto env_exp = litert::Environment::Create({});
  if (!env_exp.HasValue()) {
    return absl::InternalError(absl::StrCat(
        "litert::Environment::Create: ", env_exp.Error().Message()));
  }
  litert::Environment env = std::move(env_exp.Value());

  // Human-readable table header.
  std::fprintf(
      stdout,
      "%-6s %-6s %-6s  %8s %8s %8s %8s %8s  %10s %8s\n",
      "m", "n", "k", "min_us", "avg_us", "p50_us", "p95_us", "max_us",
      "gflops", "tfl_avg");

  // Optional CSV output.
  std::unique_ptr<std::ofstream> csv_ofs;
  if (!csv_out.empty()) {
    csv_ofs = std::make_unique<std::ofstream>(csv_out);
    if (!csv_ofs->good()) {
      return absl::UnavailableError(
          absl::StrCat("Cannot open --csv_out=", csv_out));
    }
    (*csv_ofs) << "m,n,k,count,min_us,avg_us,p50_us,p95_us,max_us,"
                  "gflops,tflops_min,tflops_avg\n";
  }

  int failed = 0;
  for (const auto& s : shapes) {
    auto samples_or = BenchmarkOneShape(env, s, backend, dtype, warmup,
                                          iters, enable_profile, cache_dir,
                                          dtype_str);
    if (!samples_or.ok()) {
      const std::string err_msg(samples_or.status().message());
      std::fprintf(stderr,
                   "[MATMUL MICRO] SKIP m=%lld n=%lld k=%lld: %s\n",
                   static_cast<long long>(s.m), static_cast<long long>(s.n),
                   static_cast<long long>(s.k), err_msg.c_str());
      ++failed;
      continue;
    }
    // For chain modes (e.g. int8_chain runs 2 FCs per Run) divide
    // each per-Run sample so the reported numbers are per-conv,
    // comparable to single-op modes and to prefill rows.
    if (chain_divisor > 1) {
      for (auto& v : *samples_or) v /= static_cast<double>(chain_divisor);
    }
    const Stats st = ComputeStats(*samples_or);

    // Compute matmul flop count (2*M*N*K) and derived throughput.
    const double flops = 2.0 * static_cast<double>(s.m) *
                          static_cast<double>(s.n) *
                          static_cast<double>(s.k);
    const double gflops = flops / 1e9;
    const double tflops_avg =
        st.avg_us > 0 ? (flops / (st.avg_us * 1e-6)) / 1e12 : 0.0;
    const double tflops_min =
        st.min_us > 0 ? (flops / (st.min_us * 1e-6)) / 1e12 : 0.0;

    std::fprintf(stdout,
                 "%-6lld %-6lld %-6lld  %8.1f %8.1f %8.1f %8.1f %8.1f  %10.2f %8.3f\n",
                 static_cast<long long>(s.m), static_cast<long long>(s.n),
                 static_cast<long long>(s.k), st.min_us, st.avg_us, st.p50_us,
                 st.p95_us, st.max_us, gflops, tflops_avg);
    std::fflush(stdout);

    if (csv_ofs) {
      (*csv_ofs) << s.m << ',' << s.n << ',' << s.k << ',' << st.count << ','
                 << st.min_us << ',' << st.avg_us << ',' << st.p50_us << ','
                 << st.p95_us << ',' << st.max_us << ',' << gflops << ','
                 << tflops_min << ',' << tflops_avg << '\n';
    }
  }

  std::fprintf(stderr, "[MATMUL MICRO] done. %d / %zu shapes failed.\n",
               failed, shapes.size());
  if (failed == static_cast<int>(shapes.size())) {
    return absl::InternalError("All shapes failed to benchmark.");
  }
  return absl::OkStatus();
}

}  // namespace

int main(int argc, char** argv) {
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);
  absl::ParseCommandLine(argc, argv);
  auto status = MainBody();
  if (!status.ok()) {
    const std::string err_msg(status.message());
    std::fprintf(stderr, "[MATMUL MICRO] FATAL: %s\n", err_msg.c_str());
    return 1;
  }
  return 0;
}
