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
#include <optional>
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
ABSL_FLAG(int, max_tensor_mb, 512,
          "If > 0, skip shapes where any of weight / input / output "
          "tensors exceeds this many MiB (bytes counted at FLOAT32 "
          "size -- INT8 paths allocate less but CL texture conversion "
          "may still upconvert to fp16/fp32). Default 512 MiB is "
          "conservative enough to keep the Adreno 830 CL device from "
          "hitting `Failed to allocate 2049280128 bytes (clCreateBuffer)` "
          "errors that used to SIGKILL the whole process on the "
          "32003x32003 vocab-projection shapes.");
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
          "kernel asserts fp32 input during prepare).\n"
          "  'int8_conv2d_per_tensor': single CONV_2D 1x1 op with INT8 "
          "per-tensor weights, FLOAT32 bias, FLOAT32 signature I/O. "
          "Variant of int8_per_tensor that uses BuiltinOperator_CONV_2D "
          "instead of FULLY_CONNECTED. Native CONV_2D + per-tensor int8 "
          "weights is the schema that triggers "
          "`convolution_int8(conv_wave_memory)` on the LiteRT CL "
          "delegate -- the same kernel name Gemma4 prefill's Delegate "
          "Statistics shows for its int8 matmul rows. The FC path used "
          "by int8_per_tensor falls back to `convolution1x1(fp16)` for "
          "large M because the FC->conv1x1 rewrite drops the int8 "
          "attribute during lowering. Use this mode when you want "
          "per-shape numbers that are directly comparable to prefill's "
          "int8 rows.");
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

namespace {

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

// Per-shape kernel profile extracted from LiteRtGetProfileSummary. Holds
// the name of the matmul GPU kernel the CL delegate actually dispatched
// (e.g. "convolution_int8(conv_wave_memory)", "fully_connected_int8",
// "convolution1x1(conv_wave_memory)") and its average per-dispatch
// latency in microseconds. Populated from the "Delegate Statistics"
// table printed by LiteRtGetProfileSummary when --profile=true.
//
// Used to answer "is the micro bench picking the same kernel prefill
// picks?" and "does per-kernel latency match prefill's per-op row?"
// without the host-side wall-clock overhead that inflates the CSV
// avg_us column.
struct KernelProfile {
  std::string name;
  double avg_us = 0;
  int count = 0;
};

// Parse the LiteRtGetProfileSummary output to find the matmul-class
// Delegate Statistics row and return its name + avg_us.
//
// The Delegate Statistics block looks like:
//
//   Delegate Statistics:
//   Op Name                                     Count  Avg(us)  Min(us) ...
//   Delegate/DownloadGpuMemoryToTensorBufferGpuMemory   50   14.56 ...
//   Delegate/UploadOrBindTensorBuffer                   50  112.14 ...
//   Delegate/fully_connected_int8                       50    8.10 ...
//
// The matmul kernel is whichever row contains "convolution" or
// "fully_connected" (case-insensitive). Non-matmul rows (Upload,
// Download, weights_convert, quantize/dequantize helpers) are filtered
// out. If multiple matmul rows exist (shouldn't happen with a single-op
// model, but the CL delegate sometimes emits a wrapped convert around
// the real conv), the slowest one wins since that's the compute-bound
// one.
//
// Returns nullopt if the summary has no "Delegate Statistics:" block or
// no matmul-class row in it. The caller writes empty CSV fields in that
// case.
std::optional<KernelProfile> ParseDelegateStatsForMatmulKernel(
    absl::string_view summary) {
  const size_t stats_pos = summary.find("Delegate Statistics:");
  if (stats_pos == absl::string_view::npos) return std::nullopt;
  absl::string_view tail = summary.substr(stats_pos);

  std::optional<KernelProfile> best;
  for (absl::string_view line : absl::StrSplit(tail, '\n')) {
    absl::string_view trimmed = absl::StripAsciiWhitespace(line);
    if (!absl::StartsWith(trimmed, "Delegate/")) continue;

    // Tokenize by whitespace. The last 5 tokens are numeric columns:
    // Count, Avg(us), Min(us), Max(us), Total(us). Everything before
    // those is the op name (which may itself contain spaces for fused
    // kernels like "conv -> dequantize_to_float16").
    std::vector<absl::string_view> toks =
        absl::StrSplit(trimmed, ' ', absl::SkipEmpty());
    if (toks.size() < 6) continue;

    int count_i = 0;
    double avg_d = 0;
    if (!absl::SimpleAtoi(toks[toks.size() - 5], &count_i)) continue;
    if (!absl::SimpleAtod(toks[toks.size() - 4], &avg_d)) continue;

    // Reconstruct the op name from the leading tokens and strip the
    // "Delegate/" prefix.
    std::string name;
    for (size_t i = 0; i + 5 < toks.size(); ++i) {
      if (i) name.push_back(' ');
      name.append(std::string(toks[i]));
    }
    constexpr absl::string_view kPrefix = "Delegate/";
    if (absl::StartsWith(name, kPrefix)) {
      name = name.substr(kPrefix.size());
    }

    // Skip the well-known non-matmul rows.
    const std::string lower = absl::AsciiStrToLower(name);
    if (absl::StrContains(lower, "upload") ||
        absl::StrContains(lower, "download") ||
        absl::StrContains(lower, "weights_convert") ||
        absl::StrContains(lower, "synchronize") ||
        absl::StrContains(lower, "copy") ||
        absl::StrContains(lower, "quantize_and_dequantize") ||
        absl::StrContains(lower, "dequantize_to_float")) {
      continue;
    }
    // Keep only rows whose name looks like a matmul kernel. The CL
    // delegate uses "convolution" (incl. "convolution_int8" /
    // "convolution1x1") for FC->conv1x1 rewrites and native CONV_2D,
    // and "fully_connected" for the non-rewritten int8 FC path.
    if (!absl::StrContains(lower, "convolution") &&
        !absl::StrContains(lower, "fully_connected") &&
        !absl::StrContains(lower, "matmul")) {
      continue;
    }

    KernelProfile kp;
    kp.name = std::move(name);
    kp.avg_us = avg_d;
    kp.count = count_i;
    if (!best || kp.avg_us > best->avg_us) best = std::move(kp);
  }
  return best;
}

// Per-shape benchmark output. Samples drive the wall-clock stats CSV
// columns; kernel_profile (when set) drives the kernel_name /
// kernel_avg_us columns that are directly comparable to prefill's
// Delegate Statistics rows.
struct ShapeRunResult {
  std::vector<double> samples_us;
  std::optional<KernelProfile> kernel_profile;
};

// Builds, compiles, and times one matmul shape. Returns the per-iteration
// latency samples so the caller can compute aggregate stats. When
// `enable_profile` is true, also enables LiteRT op profiling on the GPU
// compilation options and prints the Delegate Statistics summary to
// stderr after the timed loop -- this is the canonical way to see
// which CL kernel the GPU delegate actually picked for the shape.
absl::StatusOr<ShapeRunResult> BenchmarkOneShape(
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
    // Mirror a subset of Gemma4 prefill's GPU compilation options
    // (runtime/executor/llm_executor_settings_utils.cc:193-206) to
    // minimize differences in delegate behavior between the micro
    // bench and the real model run.
    //
    // NOT enabled: EnableAllowSrcQuantizedFcConvOps(true). That flag
    // tells the CL delegate to accept the model's source quantized
    // FC/Conv schema end-to-end. Prefill's converter emits FULLY
    // quantized ops (int8 input + int8 weight + int32 bias + int8
    // output, with matching quant params on every tensor), which is
    // what the delegate's source-quantized kernel path expects. Our
    // synthetic builder produces WEIGHT-ONLY quant (fp32 input/bias/
    // output + int8 weight), which doesn't satisfy that path's
    // requirements -- enabling the flag causes the CL kernel code
    // generator to emit references to `q0`/`q1` weight quant
    // constants it never defines, and every CompiledModel::Create
    // fails with a "BC-src-code:25: use of undeclared identifier
    // 'q0'" build error. To use that flag we would have to emit a
    // fully-quantized schema (which the existing kInt8 wrapped path
    // tried and failed at for unrelated reasons -- the delegate
    // collapses the wrap back to fp16). Left off until someone does
    // that work.
    //
    // The remaining option block below mirrors the prefill defaults
    // from runtime/engine/shared_flags.cc and the code that consumes
    // them in runtime/executor/llm_executor_settings_utils.cc. Every
    // option has been observed (in prefill's Delegate Statistics) to
    // either move work out of Run() into CompiledModel::Create
    // (weights convert, command-buffer preparation) or to remove
    // host-side wait latency (active sync). For the benchmark this
    // matters a lot: without these options, the previous run showed
    // `weights_convert_uint8_to_float16` firing 50 times (419us per
    // call) even though weights are constants, and
    // `UploadOrBindTensorBuffer` at 2136us per iter even though the
    // input is written once and never changed. Both come from the
    // delegate defaulting to "on-demand" behavior, which prefill
    // turns off via the flags below.
    gpu_opts.SetHintFullyDelegatedToSingleDelegate(true);
    gpu_opts.EnableConstantTensorSharing(true);
    // Convert int8 weights to fp16 on the GPU once at Create time
    // (not per Run). Paired with WaitForWeightsConversionComplete
    // so Create blocks until conversion finishes -- otherwise the
    // conversion gets deferred to the first Run() calls and shows
    // up in the benchmark timings.
    gpu_opts.SetConvertWeightsOnGpu(true);
    gpu_opts.WaitForWeightsConversionComplete(true);
    // madvise SEQUENTIAL/WILLNEED on original shared (mmap'd) weight
    // tensors so the kernel reading them doesn't pay page-fault cost
    // on the hot path. Harmless for our in-memory synthetic weights,
    // matches prefill.
    gpu_opts.SetMadviseOriginalSharedTensors(true);
    // Active wait (spin) instead of passive wait for GPU completion.
    // Removes ~100us of host-thread wake-up latency after
    // clEnqueueNDRangeKernel completes -- important when the burst
    // submit pattern has 50 enqueues back-to-back. Prefill turns
    // this on when advanced_settings.is_benchmark is true (which it
    // is for the production benchmark path); the micro bench is
    // also a benchmark, so match.
    gpu_opts.SetSyncExecutionModeWaitType(
        litert::GpuOptions::SyncExecutionModeWaitType::kActive);
    // Keep shader optimizations (default in prefill). Passing false
    // means "don't disable", i.e. optimizations stay on.
    gpu_opts.DisableShaderOptimization(false);
    // Pre-prepare command buffers 2 steps ahead so the GPU queue
    // doesn't stall while LiteRT builds the next dispatch. Prefill
    // uses 2 because its KV cache swaps every 2 steps; for our
    // straight burst loop, 2 is also fine -- anything >=1 is an
    // improvement over the default of 0.
    gpu_opts.SetNumStepsOfCommandBufferPreparations(2);
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
      // v2 prefix: previous runs populated this cache dir with
      // kernel entries from a schema that the delegate's CL code
      // generator couldn't compile (weight-only int8 + the
      // EnableAllowSrcQuantizedFcConvOps flag -> undeclared
      // identifiers q0/q1 in the generated source). Those broken
      // entries would be reloaded by the old keys on every
      // subsequent run, even with the flag off, because LiteRT's
      // cache stores the compiled program bytes and doesn't
      // re-validate them against the current schema. Bumping the
      // prefix forces a fresh compile; old v1 entries become
      // orphaned and can be deleted manually with
      // `adb shell rm -rf /data/local/tmp/litert_lm/cache/*`.
      const std::string model_cache_key =
          absl::StrCat("matmul_bench_v2_m", shape.m, "_n", shape.n, "_k",
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

  // Timed runs.
  //
  // IMPORTANT: the measurement pattern here is deliberately different
  // from a per-iteration latency benchmark. Gemma4 prefill -- the
  // thing this micro bench is supposed to be comparable to -- submits
  // 1107 ops back-to-back without any host sync between them. The GPU
  // queue stays full, dispatches overlap with weight reads, and each
  // op's reported cost in Delegate Statistics is the per-kernel time
  // under a warm pipeline.
  //
  // The previous pattern (Run + TensorBufferScopedLock(kRead) per
  // iteration) drained the GPU queue every iteration. Wall-clock
  // samples ended up dominated by sync stall + LiteRT dispatch
  // overhead rather than kernel execution. For the shapes in
  // matmul_roster.csv that meant micro-bench numbers 3-15x larger
  // than prefill's Delegate Statistics for the same kernel.
  //
  // New pattern: submit all `timed_iters` Run() calls in a burst,
  // then issue a single ScopedLock(kRead) at the very end to flush
  // the GPU queue. Total elapsed time / timed_iters gives the
  // amortized per-iteration cost that matches prefill-style
  // pipelining. Individual per-iter variance is NOT measurable in
  // this mode (all samples are the mean by construction), so the
  // min/p50/p95 columns in the CSV will all equal avg; that's a
  // feature, not a bug -- the CSV is meant for comparison against
  // the Delegate Statistics avg column, not for latency distribution
  // analysis.
  //
  // If op profiling is enabled, Start/Stop the LiteRtProfiler around
  // the entire timed loop so the per-op event buffer covers all
  // `timed_iters` invocations. Same accounting as before.
  if (enable_profile && profiler_handle != nullptr) {
    (void)LiteRtStartProfiler(profiler_handle);
  }
  const absl::Time t_burst_start = absl::Now();
  for (int i = 0; i < timed_iters; ++i) {
    auto run_exp = compiled_model.Run(built.signature_key, input_buffers,
                                       output_buffers);
    if (!run_exp.HasValue()) {
      return ExpectedError(
          run_exp,
          absl::StrCat("Run() timed m=", shape.m, " n=", shape.n,
                       " k=", shape.k, " iter=", i));
    }
  }
  // Single GPU->host sync at the end. ScopedLock(kRead) forces the CL
  // delegate to wait for ALL submitted Run()s to complete. The
  // volatile read below prevents the compiler / LiteRT from
  // short-circuiting the mapping.
  {
    auto lock_exp = litert::TensorBufferScopedLock::Create(
        output_buffers[0], litert::TensorBuffer::LockMode::kRead);
    if (!lock_exp.HasValue()) {
      return ExpectedError(
          lock_exp,
          absl::StrCat("TensorBufferScopedLock final m=", shape.m,
                       " n=", shape.n, " k=", shape.k));
    }
    auto lock_and_addr = std::move(lock_exp.Value());
    volatile uint8_t* mapped =
        static_cast<volatile uint8_t*>(lock_and_addr.second);
    if (mapped != nullptr) {
      (void)mapped[0];
    }
  }
  const absl::Time t_burst_end = absl::Now();
  const double total_us =
      absl::ToDoubleMicroseconds(t_burst_end - t_burst_start);
  const double per_iter_us =
      timed_iters > 0 ? (total_us / static_cast<double>(timed_iters)) : 0;
  // Fill the samples vector with the amortized per-iter value so
  // downstream ComputeStats still emits sensible (if degenerate)
  // min/p50/p95 columns. The kernel_avg_us CSV column is the more
  // meaningful per-dispatch number; this is the wall-clock mean.
  std::vector<double> samples_us(timed_iters, per_iter_us);

  // Stop the profiler and dump the per-op summary so we can see which
  // CL kernel the GPU delegate actually picked. The summary string is
  // owned by the caller and must be std::free'd after use.
  // We also parse the summary to extract the matmul kernel's name +
  // avg_us for the CSV kernel_name / kernel_avg_us columns.
  std::optional<KernelProfile> kernel_profile;
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
      kernel_profile = ParseDelegateStatsForMatmulKernel(
          absl::string_view(summary));
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
  ShapeRunResult result;
  result.samples_us = std::move(samples_us);
  result.kernel_profile = std::move(kernel_profile);
  return result;
}

absl::Status MainBody() {
  const std::string backend = absl::GetFlag(FLAGS_backend);
  const std::string shapes_flag = absl::GetFlag(FLAGS_shapes);
  const std::string shapes_csv = absl::GetFlag(FLAGS_shapes_csv);
  const int warmup = absl::GetFlag(FLAGS_warmup);
  const int iters = absl::GetFlag(FLAGS_iters);
  const std::string csv_out = absl::GetFlag(FLAGS_csv_out);
  const int max_shapes = absl::GetFlag(FLAGS_max_shapes);
  const int max_tensor_mb = absl::GetFlag(FLAGS_max_tensor_mb);
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
  } else if (dtype_str == "int8_conv2d_per_tensor") {
    dtype = litert::lm::MatmulDtype::kInt8Conv2dPerTensor;
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
        "--dtype must be int8_per_tensor, int8_conv2d_per_tensor, int8, "
        "int8_chain, int8_hybrid, fp32, fp32_conv2d, or fp16, got: ",
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
  // Guard against shapes that will trigger the CL delegate's 2 GB
  // buffer allocation failure (and sometimes SIGKILL the process) --
  // e.g. 32003x32003 vocab-projection shapes. We count bytes at
  // FLOAT32 size (4 B/element) as a conservative upper bound: int8
  // paths allocate less than this, but the delegate may still
  // upconvert to fp16/fp32 for texture layout on certain precisions.
  // Any tensor (weight, input, or output) exceeding max_tensor_mb
  // causes the shape to be skipped with a stderr note; the CSV
  // simply won't have a row for it.
  if (max_tensor_mb > 0) {
    const int64_t max_bytes =
        static_cast<int64_t>(max_tensor_mb) * 1024 * 1024;
    std::vector<Shape> kept;
    kept.reserve(shapes.size());
    for (const auto& s : shapes) {
      const int64_t w_bytes = s.n * s.k * 4;          // filter fp32 cap
      const int64_t in_bytes = s.m * s.k * 4;         // input fp32
      const int64_t out_bytes = s.m * s.n * 4;        // output fp32
      const int64_t worst =
          std::max({w_bytes, in_bytes, out_bytes});
      if (worst > max_bytes) {
        std::fprintf(stderr,
                     "[MATMUL MICRO] SKIP (shape too large for "
                     "--max_tensor_mb=%d): m=%lld n=%lld k=%lld "
                     "(worst tensor %.1f MiB)\n",
                     max_tensor_mb, static_cast<long long>(s.m),
                     static_cast<long long>(s.n),
                     static_cast<long long>(s.k),
                     static_cast<double>(worst) / (1024.0 * 1024.0));
        continue;
      }
      kept.push_back(s);
    }
    shapes = std::move(kept);
  }
  if (shapes.empty()) {
    return absl::InvalidArgumentError(
        "No shapes left after --max_tensor_mb filter.");
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
    // kernel_name / kernel_avg_us are the matmul GPU kernel's name and
    // avg per-dispatch time as reported by the LiteRT Delegate
    // Statistics block (parsed out of LiteRtGetProfileSummary). They
    // are blank unless --profile=true. The kernel_avg_us column is
    // what's directly comparable to prefill's Delegate Statistics
    // rows; the wall-clock avg_us column also counts upload/download
    // and host sync and will always be larger.
    (*csv_ofs) << "m,n,k,count,min_us,avg_us,p50_us,p95_us,max_us,"
                  "gflops,tflops_min,tflops_avg,"
                  "kernel_name,kernel_avg_us\n";
  }

  int failed = 0;
  for (const auto& s : shapes) {
    auto run_or = BenchmarkOneShape(env, s, backend, dtype, warmup,
                                     iters, enable_profile, cache_dir,
                                     dtype_str);
    if (!run_or.ok()) {
      const std::string err_msg(run_or.status().message());
      std::fprintf(stderr,
                   "[MATMUL MICRO] SKIP m=%lld n=%lld k=%lld: %s\n",
                   static_cast<long long>(s.m), static_cast<long long>(s.n),
                   static_cast<long long>(s.k), err_msg.c_str());
      ++failed;
      continue;
    }
    ShapeRunResult& run = *run_or;
    // For chain modes (e.g. int8_chain runs 2 FCs per Run) divide
    // each per-Run sample so the reported numbers are per-conv,
    // comparable to single-op modes and to prefill rows.
    if (chain_divisor > 1) {
      for (auto& v : run.samples_us) v /= static_cast<double>(chain_divisor);
    }
    const Stats st = ComputeStats(run.samples_us);

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
      // kernel_name is the raw string from Delegate Statistics. It can
      // contain parentheses, arrows, and spaces (e.g. "convolution_int8
      // (conv_wave_memory) -> dequantize_to_float16"). We wrap it in
      // double-quotes and escape any embedded double-quote so the CSV
      // parses cleanly in spreadsheets / pandas.
      std::string kernel_name;
      double kernel_avg = 0;
      if (run.kernel_profile) {
        kernel_name = run.kernel_profile->name;
        kernel_avg = run.kernel_profile->avg_us;
        // Chain modes share the same comment as the wall-clock: each
        // kernel dispatch is one of `chain_divisor` FC ops, so divide.
        if (chain_divisor > 1) {
          kernel_avg /= static_cast<double>(chain_divisor);
        }
      }
      std::string escaped;
      escaped.reserve(kernel_name.size() + 2);
      escaped.push_back('"');
      for (char c : kernel_name) {
        if (c == '"') escaped.push_back('"');  // CSV-quote doubling
        escaped.push_back(c);
      }
      escaped.push_back('"');
      (*csv_ofs) << s.m << ',' << s.n << ',' << s.k << ',' << st.count << ','
                 << st.min_us << ',' << st.avg_us << ',' << st.p50_us << ','
                 << st.p95_us << ',' << st.max_us << ',' << gflops << ','
                 << tflops_min << ',' << tflops_avg << ','
                 << escaped << ',' << kernel_avg << '\n';
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
