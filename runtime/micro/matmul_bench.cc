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

// Standalone matmul microbenchmark for the LiteRT GPU / CPU backends.
// See runtime/micro/README.md for the why and the phase roadmap.
//
// Phase 2 scope (this commit): adds shape + dtype parsing and proves
// the BuildSingleFullyConnectedTfliteModel wiring end-to-end by
// building (not running) one flatbuffer per requested shape and
// reporting its byte count. The benchmark loop itself still lives
// in phase 3.

#include <cstdio>
#include <cstdlib>
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
#include "litert/cc/litert_common.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "runtime/util/single_op_tflite_builder.h"

// ---------------------------------------------------------------------------
// Flags
// ---------------------------------------------------------------------------
ABSL_FLAG(std::string, backend, "gpu",
          "[P1] Backend to target: 'gpu' (LiteRT CL delegate on Adreno) "
          "or 'cpu' (reference path). Only affects the per-shape "
          "CompiledModel options set in phase 3.");
ABSL_FLAG(std::string, dtype, "int8_per_tensor",
          "[P2] Matmul dtype variant. One of: fp32, fp16, "
          "int8_per_tensor, int8_chain, fp32_conv2d, int8, "
          "int8_weight_fp32_act.");
ABSL_FLAG(std::string, shapes, "",
          "[P2] Inline comma-separated MxNxK list, e.g. "
          "'1024x1536x6144,1x256x1536'. Empty means use --shapes_csv.");
ABSL_FLAG(std::string, shapes_csv, "",
          "[P3] Path to a CSV with at least m,n,k columns. Header row "
          "is assumed. Compatible with the roster CSV emitted by "
          "runtime/util/matmul_shape_dump.");
ABSL_FLAG(int, warmup, 5, "[P3] Warmup iterations before timing starts.");
ABSL_FLAG(int, iters, 50, "[P3] Timed iterations per shape.");
ABSL_FLAG(bool, profile, false,
          "[P3] Enable the LiteRT per-op profiler. When true the CSV "
          "gains a kernel_name column extracted from the summary so "
          "each row proves which CL kernel actually ran.");
ABSL_FLAG(std::string, csv_out, "",
          "[P3] If non-empty, append one row per shape to this CSV.");
ABSL_FLAG(std::string, runtime_lib_dir, ".",
          "[P1-echo] Directory that will contain libLiteRt.so and the "
          "accelerator shlibs. Currently echoed but not consumed -- "
          "the runner script is expected to set LD_LIBRARY_PATH. "
          "Phase 4 will wire this through the C env option.");
ABSL_FLAG(std::string, cache_dir, "",
          "[P4] Optional directory for the LiteRT OpenCL program "
          "cache. Wired in phase 4.");
ABSL_FLAG(uint64_t, seed, 0,
          "[P4] Deterministic PRNG seed. Currently unused; the "
          "underlying builder already uses shape-derived deterministic "
          "bytes. Will be plumbed through in phase 4 once we need "
          "multiple independent weight streams per shape.");

namespace {

// ---------------------------------------------------------------------------
// Shape parsing
// ---------------------------------------------------------------------------
//
// Mirrors runtime/engine/matmul_micro_benchmark.cc::Shape/ParseShape so
// CSV roster files produced for the old binary are drop-in compatible
// with this one. Only the inline --shapes path is wired in phase 2;
// --shapes_csv support lands with the benchmark loop in phase 3.
struct Shape {
  int64_t m = 0;
  int64_t n = 0;
  int64_t k = 0;

  bool operator<(const Shape& other) const {
    return std::tie(m, n, k) < std::tie(other.m, other.n, other.k);
  }
  bool operator==(const Shape& other) const {
    return m == other.m && n == other.n && k == other.k;
  }
};

absl::StatusOr<Shape> ParseShape(absl::string_view s) {
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
  for (auto& p : parts) p = absl::StripAsciiWhitespace(p);
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

absl::StatusOr<std::vector<Shape>> ParseInlineShapes(absl::string_view list) {
  std::vector<Shape> out;
  if (list.empty()) return out;
  for (absl::string_view tok : absl::StrSplit(list, ',', absl::SkipEmpty())) {
    auto s_or = ParseShape(absl::StripAsciiWhitespace(tok));
    if (!s_or.ok()) return s_or.status();
    out.push_back(*s_or);
  }
  return out;
}

// ---------------------------------------------------------------------------
// Dtype parsing
// ---------------------------------------------------------------------------
//
// String <-> enum mapping is the only dtype wiring the binary owns.
// The actual op schema emission lives in the builder; this function
// exists so an unrecognized --dtype fails fast with a clear message
// rather than defaulting to kInt8 silently.
absl::StatusOr<litert::lm::MatmulDtype> ParseDtype(absl::string_view s) {
  const std::string lower = absl::AsciiStrToLower(std::string(s));
  if (lower == "fp32") return litert::lm::MatmulDtype::kFp32;
  if (lower == "fp16") return litert::lm::MatmulDtype::kFp16;
  if (lower == "fp32_conv2d") return litert::lm::MatmulDtype::kFp32Conv2d;
  if (lower == "int8") return litert::lm::MatmulDtype::kInt8;
  if (lower == "int8_chain") return litert::lm::MatmulDtype::kInt8Chain;
  if (lower == "int8_per_tensor")
    return litert::lm::MatmulDtype::kInt8PerTensor;
  if (lower == "int8_weight_fp32_act")
    return litert::lm::MatmulDtype::kInt8WeightFp32Act;
  return absl::InvalidArgumentError(
      absl::StrCat("Unknown --dtype: ", s,
                   " (want one of: fp32, fp16, fp32_conv2d, int8, "
                   "int8_chain, int8_per_tensor, int8_weight_fp32_act)"));
}

const char* DtypeName(litert::lm::MatmulDtype d) {
  switch (d) {
    case litert::lm::MatmulDtype::kFp32: return "fp32";
    case litert::lm::MatmulDtype::kFp16: return "fp16";
    case litert::lm::MatmulDtype::kFp32Conv2d: return "fp32_conv2d";
    case litert::lm::MatmulDtype::kInt8: return "int8";
    case litert::lm::MatmulDtype::kInt8Chain: return "int8_chain";
    case litert::lm::MatmulDtype::kInt8PerTensor: return "int8_per_tensor";
    case litert::lm::MatmulDtype::kInt8WeightFp32Act:
      return "int8_weight_fp32_act";
  }
  return "unknown";
}

}  // namespace

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);

  const std::string backend = absl::GetFlag(FLAGS_backend);
  const std::string dtype_str = absl::GetFlag(FLAGS_dtype);
  const std::string runtime_lib_dir = absl::GetFlag(FLAGS_runtime_lib_dir);
  const std::string cache_dir = absl::GetFlag(FLAGS_cache_dir);
  const std::string shapes_inline = absl::GetFlag(FLAGS_shapes);
  const std::string shapes_csv = absl::GetFlag(FLAGS_shapes_csv);
  const int warmup = absl::GetFlag(FLAGS_warmup);
  const int iters = absl::GetFlag(FLAGS_iters);
  const bool enable_profile = absl::GetFlag(FLAGS_profile);
  const std::string csv_out = absl::GetFlag(FLAGS_csv_out);
  const uint64_t seed = absl::GetFlag(FLAGS_seed);

  auto dtype_or = ParseDtype(dtype_str);
  if (!dtype_or.ok()) {
    std::fprintf(stderr, "[matmul_bench] ERR %s\n",
                 std::string(dtype_or.status().message()).c_str());
    return 1;
  }
  const litert::lm::MatmulDtype dtype = *dtype_or;

  auto shapes_or = ParseInlineShapes(shapes_inline);
  if (!shapes_or.ok()) {
    std::fprintf(stderr, "[matmul_bench] ERR %s\n",
                 std::string(shapes_or.status().message()).c_str());
    return 1;
  }
  std::vector<Shape> shapes = *std::move(shapes_or);

  std::fprintf(stderr,
               "[matmul_bench] phase=2-builder backend=%s dtype=%s "
               "runtime_lib_dir=%s cache_dir=%s shapes_inline=%zu "
               "shapes_csv=%s warmup=%d iters=%d profile=%d csv_out=%s "
               "seed=%llu\n",
               backend.c_str(), DtypeName(dtype), runtime_lib_dir.c_str(),
               cache_dir.empty() ? "(none)" : cache_dir.c_str(), shapes.size(),
               shapes_csv.empty() ? "(none)" : shapes_csv.c_str(), warmup,
               iters, enable_profile ? 1 : 0,
               csv_out.empty() ? "(none)" : csv_out.c_str(),
               static_cast<unsigned long long>(seed));

  // ---------------------------------------------------------------------
  // Environment creation
  // ---------------------------------------------------------------------
  auto env_exp =
      litert::Environment::Create(std::vector<litert::Environment::Option>());
  if (!env_exp.HasValue()) {
    std::fprintf(stderr,
                 "[matmul_bench] ERR litert::Environment::Create: %s\n",
                 env_exp.Error().Message().data());
    return 2;
  }
  litert::Environment env = std::move(env_exp.Value());

  // ---------------------------------------------------------------------
  // Phase 2 smoke test: build one flatbuffer per shape and report size.
  // ---------------------------------------------------------------------
  //
  // The purpose is twofold:
  //   1. Prove the //runtime/util:single_op_tflite_builder dep links
  //      cleanly from runtime/micro -- the rewrite's "modularity"
  //      claim stands or falls on this.
  //   2. Catch obviously-bad shapes early (zero/negative dims, weight
  //      buffer overflow) before the benchmark loop in phase 3 tries
  //      to push them through litert::Model::CreateFromBuffer.
  //
  // Phase 3 will swap the "print size" body for the actual warmup +
  // timed-iter loop and move the builder call under BenchmarkOneShape.
  if (shapes.empty()) {
    std::fprintf(stderr,
                 "[matmul_bench] no --shapes provided; phase 2 is a "
                 "build-only smoke test so there is nothing to do. "
                 "Exiting 0.\n");
    return 0;
  }

  int ok = 0;
  int failed = 0;
  for (const Shape& s : shapes) {
    auto build_or = litert::lm::BuildSingleFullyConnectedTfliteModel(
        s.m, s.n, s.k, dtype);
    if (!build_or.ok()) {
      std::fprintf(stderr,
                   "[matmul_bench] build FAIL m=%lld n=%lld k=%lld dtype=%s: "
                   "%s\n",
                   static_cast<long long>(s.m), static_cast<long long>(s.n),
                   static_cast<long long>(s.k), DtypeName(dtype),
                   std::string(build_or.status().message()).c_str());
      ++failed;
      continue;
    }
    const litert::lm::SingleFcBuildResult& built = *build_or;
    std::fprintf(stderr,
                 "[matmul_bench] build OK   m=%lld n=%lld k=%lld dtype=%s "
                 "flatbuffer_bytes=%zu signature=%s input=%s output=%s\n",
                 static_cast<long long>(s.m), static_cast<long long>(s.n),
                 static_cast<long long>(s.k), DtypeName(dtype),
                 built.flatbuffer.size(), built.signature_key.c_str(),
                 built.input_name.c_str(), built.output_name.c_str());
    ++ok;
  }

  std::fprintf(stderr,
               "[matmul_bench] phase 2 done: %d ok / %d failed. Benchmark "
               "loop lands in phase 3.\n",
               ok, failed);
  return failed == 0 ? 0 : 3;
}
