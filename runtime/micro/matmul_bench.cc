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
//
// This binary is the from-scratch successor to
// runtime/engine/matmul_micro_benchmark.cc. Design goals that justify
// the rewrite:
//
//   1. Observability. The existing benchmark prints per-shape latency
//      but not which GPU kernel the delegate actually picked. When the
//      LiteRT CL delegate silently falls back (e.g. because the build
//      was linked without --export-dynamic-symbol=LiteRt*, or the
//      accelerator shlibs were not pushed to the device) the numbers
//      look plausible but come from the CPU path. The new binary will
//      parse the per-op profiler summary and emit the convolution
//      kernel name as a CSV column so every row is self-verifying.
//
//   2. Reproducibility across reruns. Will add an optional OpenCL
//      program-cache directory so the second invocation skips on-device
//      JIT; this cuts first-iteration latency by ~100x on Adreno.
//
//   3. Modularity. The .tflite builder, the benchmark loop, and the
//      shape roster extractor live under runtime/micro/ rather than
//      being spread across runtime/engine/ and runtime/util/.
//
// Phase 1 scope (this commit): scaffolding only. The full CLI surface
// is declared so downstream scripts can lock in flag names today, but
// only backend / dtype / runtime_lib_dir are echoed back. The
// environment is created with empty options to match the existing
// benchmark's current behaviour (relying on LD_LIBRARY_PATH); later
// phases will add kLiteRtEnvOptionTagRuntimeLibraryDir / ModelCacheDir
// once the C++ wrapper's OptionTag enum is confirmed.

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "absl/base/log_severity.h"  // from @com_google_absl
#include "absl/flags/flag.h"  // from @com_google_absl
#include "absl/flags/parse.h"  // from @com_google_absl
#include "absl/log/globals.h"  // from @com_google_absl
#include "litert/cc/litert_common.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert

// ---------------------------------------------------------------------------
// Flags
// ---------------------------------------------------------------------------
//
// The full CLI surface is declared up-front so the runner script and
// any downstream automation can lock in the flag names today. Flags
// consumed in phase 1 are marked [P1]; the rest are echoed back but
// otherwise unused until their owning phase lands.
ABSL_FLAG(std::string, backend, "gpu",
          "[P1] Backend to target: 'gpu' (LiteRT CL delegate on Adreno) "
          "or 'cpu' (reference path). The environment itself always "
          "loads all available accelerators; this only affects the "
          "per-shape CompiledModel options set in phase 3.");
ABSL_FLAG(std::string, dtype, "int8_per_tensor",
          "[P1] Matmul dtype variant. One of: fp32, fp16, "
          "int8_per_tensor, int8_chain, fp32_conv2d. Controls which "
          "op schema the flatbuffer builder emits, which in turn "
          "determines which GPU kernel the CL delegate selects.");
ABSL_FLAG(std::string, shapes, "",
          "[P3] Inline comma-separated MxNxK list, e.g. "
          "'1024x1536x6144,1x256x1536'. Empty means use --shapes_csv.");
ABSL_FLAG(std::string, shapes_csv, "",
          "[P3] Path to a CSV with at least m,n,k columns. Header "
          "row is assumed. Compatible with the roster CSV emitted by "
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
          "Phase 2 will wire this through kLiteRtEnvOptionTagRuntimeLibraryDir.");
ABSL_FLAG(std::string, cache_dir, "",
          "[P2] Optional directory for the LiteRT OpenCL program "
          "cache. Wiring is added in phase 2 together with "
          "runtime_lib_dir.");
ABSL_FLAG(uint64_t, seed, 0,
          "[P2] Deterministic PRNG seed for weight/activation bytes. "
          "The flatbuffer builder will hash (m,n,k,dtype,seed) so "
          "identical shapes produce byte-identical models across "
          "runs -- that keeps the GPU weight cache happy across "
          "restarts.");

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);

  const std::string backend = absl::GetFlag(FLAGS_backend);
  const std::string dtype = absl::GetFlag(FLAGS_dtype);
  const std::string runtime_lib_dir = absl::GetFlag(FLAGS_runtime_lib_dir);
  const std::string cache_dir = absl::GetFlag(FLAGS_cache_dir);
  const std::string shapes_inline = absl::GetFlag(FLAGS_shapes);
  const std::string shapes_csv = absl::GetFlag(FLAGS_shapes_csv);
  const int warmup = absl::GetFlag(FLAGS_warmup);
  const int iters = absl::GetFlag(FLAGS_iters);
  const bool enable_profile = absl::GetFlag(FLAGS_profile);
  const std::string csv_out = absl::GetFlag(FLAGS_csv_out);
  const uint64_t seed = absl::GetFlag(FLAGS_seed);

  std::fprintf(stderr,
               "[matmul_bench] phase=1-scaffolding backend=%s dtype=%s "
               "runtime_lib_dir=%s cache_dir=%s shapes=%s shapes_csv=%s "
               "warmup=%d iters=%d profile=%d csv_out=%s seed=%llu\n",
               backend.c_str(), dtype.c_str(), runtime_lib_dir.c_str(),
               cache_dir.empty() ? "(none)" : cache_dir.c_str(),
               shapes_inline.empty() ? "(none)" : shapes_inline.c_str(),
               shapes_csv.empty() ? "(none)" : shapes_csv.c_str(),
               warmup, iters, enable_profile ? 1 : 0,
               csv_out.empty() ? "(none)" : csv_out.c_str(),
               static_cast<unsigned long long>(seed));

  // ---------------------------------------------------------------------
  // Environment creation
  // ---------------------------------------------------------------------
  // The LiteRT C++ environment is what triggers dlopen() of the
  // accelerator shlibs (libLiteRtGpuAccelerator.so ->
  // libLiteRtOpenClAccelerator.so on Adreno). Without a correct
  // runtime library directory that dlopen can silently fail and every
  // CompiledModel falls through to the CPU interpreter.
  //
  // Phase 1 relies on the runner script exporting LD_LIBRARY_PATH to
  // the push directory, matching how the existing
  // runtime/engine/matmul_micro_benchmark.cc is launched today. Phase
  // 2 will promote --runtime_lib_dir to a first-class env option and
  // drop the LD_LIBRARY_PATH dependency.
  auto env_exp =
      litert::Environment::Create(std::vector<litert::Environment::Option>());
  if (!env_exp.HasValue()) {
    std::fprintf(stderr,
                 "[matmul_bench] ERR litert::Environment::Create: %s\n",
                 env_exp.Error().Message().data());
    return 2;
  }
  litert::Environment env = std::move(env_exp.Value());

  std::fprintf(stderr,
               "[matmul_bench] env ready. benchmark loop not implemented "
               "in phase 1 -- exiting 0.\n");
  return 0;
}
