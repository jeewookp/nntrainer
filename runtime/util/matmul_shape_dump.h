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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_MATMUL_SHAPE_DUMP_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_MATMUL_SHAPE_DUMP_H_

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl

namespace litert::lm {

// One entry in the matmul roster dumped by DumpMatmulShapes. One entry per
// matmul-class op in the model, used both for the human-readable stderr
// dump and for downstream tooling that consumes the CSV written by
// WriteMatmulShapeRosterCsv (e.g. spreadsheets, grep, microbench drivers).
struct MatmulShapeEntry {
  int subgraph_index = 0;
  std::string subgraph_name;
  int op_index = 0;
  // Human-readable TFLite BuiltinOperator name, e.g. "FULLY_CONNECTED",
  // "BATCH_MATMUL", "MATMUL", "CONV_2D" (the GPU delegate implements
  // matmul as a 1x1 conv under `conv_wave_memory`, so we also surface
  // small-kernel Conv2D rows).
  std::string op_type;
  // Shape vectors. Outer vector is per-tensor, inner vector is the TFLite
  // shape (-1 means dynamic).
  std::vector<std::vector<int32_t>> input_shapes;
  std::vector<std::vector<int32_t>> output_shapes;
  // Convenience canonical fields for the common FULLY_CONNECTED / MatMul
  // case: the "batch x contraction" dimensions extracted from the input
  // and weights. For other op types these stay -1.
  int64_t m = -1;  // rows of LHS / FC input elements in batch dim
  int64_t n = -1;  // cols of RHS / FC output features
  int64_t k = -1;  // contraction dim
};

// Walks the (already-verified) tflite flatbuffer model, collects every
// matmul-class op, and returns the roster. "Matmul-class" = any op that
// ends up lowered to a matmul by the GPU/CPU kernel:
//   - FULLY_CONNECTED
//   - BATCH_MATMUL
//   - (explicit) MATMUL / MAT_MUL (TFLite has experimental STABLEHLO_DOT_GENERAL)
//   - CONV_2D with 1x1 kernel (this is how the LiteRT CL delegate
//     implements FC under the `conv_wave_memory` kernel fusion, and is
//     therefore relevant when correlating profile rows to matmul shapes).
//
// Returns an InvalidArgumentError if the buffer is not a valid tflite model.
absl::StatusOr<std::vector<MatmulShapeEntry>> CollectMatmulShapes(
    absl::string_view tflite_buffer);

// Dumps the collected matmul roster to `out` (nullptr -> stderr) as a
// multi-line table. The header includes `label` so multiple models in
// the same run (prefill-decode / embedder / vision_adapter / ...) can
// be told apart. Emits exactly one block per call, surrounded by banner
// lines.
//
// The format is designed to stand next to the LiteRT profiler's
// "[PROFILE LiteRT CompiledModel per-op summary]" output so the reviewer
// can correlate each fused GPU kernel (e.g. `convolution_int8(conv_wave_memory)`
// on the per-op table) against the actual M/N/K of the source matmul op.
void DumpMatmulShapes(absl::string_view tflite_buffer, absl::string_view label,
                      FILE* out);

// Same as above but writes a machine-readable CSV with one row per matmul
// op. The first row is the header. Columns:
//   label,subgraph_idx,subgraph_name,op_idx,op_type,m,n,k,input_shapes,output_shapes
// `input_shapes` / `output_shapes` are pipe-separated shape lists per
// tensor, each shape comma-separated, so for e.g. BATCH_MATMUL with two
// inputs of rank 4 and one output of rank 4 the input_shapes column looks
// like "1,8,1024,128|1,8,128,1024". This CSV is emitted opt-in via the
// LITERT_LM_MATMUL_ROSTER_CSV env var and is meant to be grep-able /
// spreadsheet-loadable.
absl::Status WriteMatmulShapeRosterCsv(absl::string_view tflite_buffer,
                                       absl::string_view label,
                                       absl::string_view out_path);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_MATMUL_SHAPE_DUMP_H_
