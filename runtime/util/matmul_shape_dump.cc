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

#include "runtime/util/matmul_shape_dump.h"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tflite/schema/schema_generated.h"  // from @litert

namespace litert::lm {

namespace {

// Resolves the builtin opcode from an OperatorCode, respecting both the
// deprecated 8-bit field and the 32-bit field used for newer opcode values.
// This is the same logic as tflite::GetBuiltinCode in schema_utils.h; we
// inline it here to avoid a new schema_utils dependency.
tflite::BuiltinOperator ResolveBuiltinCode(const tflite::OperatorCode* oc) {
  if (oc == nullptr) return tflite::BuiltinOperator_CUSTOM;
  const auto full = oc->builtin_code();
  const auto deprecated =
      static_cast<tflite::BuiltinOperator>(oc->deprecated_builtin_code());
  return full > deprecated ? full : deprecated;
}

// Turns a flatbuffers Vector<int32_t>* into a std::vector<int32_t>. Handles
// nullptr shape (scalar / opaque).
std::vector<int32_t> ShapeToVec(
    const flatbuffers::Vector<int32_t>* shape) {
  std::vector<int32_t> out;
  if (shape == nullptr) {
    return out;
  }
  out.reserve(shape->size());
  for (int32_t d : *shape) {
    out.push_back(d);
  }
  return out;
}

// Renders a shape as "[d0,d1,...,dN]". Dynamic dims are reported as "-1"
// (same convention as the rest of this repo, see kDynamicDimValue).
std::string ShapeToString(const std::vector<int32_t>& shape) {
  std::string s = "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i != 0) s.push_back(',');
    absl::StrAppend(&s, shape[i]);
  }
  s.push_back(']');
  return s;
}

// Tries to fill m/n/k for common matmul lowerings. Returns true on success.
// For FULLY_CONNECTED: TFLite spec is output = input * transpose(weights).
//   input  : [..., K]         (last dim is reduction)
//   weights: [N, K]           (rank 2, transposed inside the op)
//   output : [..., N]
//   M = product of all input dims except the last.
// For BATCH_MATMUL: TFLite spec is output = lhs @ rhs (optionally transposed).
//   Without adj_x / adj_y we have lhs[..., M, K] x rhs[..., K, N] = out[..., M, N].
//   We don't inspect the BatchMatMulOptions transpose flags here because the
//   purpose is a shape-roster / correlation view, not a semantically-perfect
//   reconstruction. The caller is expected to read the raw shapes too.
// For CONV_2D (1x1 kernel): input[B,H,W,Cin], filter[Cout,1,1,Cin], out[B,H,W,Cout].
//   We flatten this into matmul as M = B*H*W, K = Cin, N = Cout.
void InferMNK(const tflite::BuiltinOperator op_code,
              const std::vector<std::vector<int32_t>>& inputs,
              const std::vector<std::vector<int32_t>>& outputs,
              MatmulShapeEntry& entry) {
  auto product = [](const std::vector<int32_t>& v, int from,
                    int to) -> int64_t {
    int64_t p = 1;
    for (int i = from; i < to; ++i) {
      if (i < 0 || i >= static_cast<int>(v.size())) continue;
      if (v[i] <= 0) return -1;  // dynamic or invalid
      p *= v[i];
    }
    return p;
  };

  switch (op_code) {
    case tflite::BuiltinOperator_FULLY_CONNECTED: {
      if (inputs.size() < 2 || outputs.empty()) return;
      const auto& in = inputs[0];
      const auto& w = inputs[1];
      if (in.empty() || w.size() != 2) return;
      entry.k = in.back();
      entry.n = w[0];
      entry.m = product(in, 0, static_cast<int>(in.size()) - 1);
      break;
    }
    case tflite::BuiltinOperator_BATCH_MATMUL: {
      if (inputs.size() < 2 || outputs.empty()) return;
      const auto& lhs = inputs[0];
      const auto& rhs = inputs[1];
      if (lhs.size() < 2 || rhs.size() < 2) return;
      entry.m = lhs[lhs.size() - 2];
      entry.k = lhs[lhs.size() - 1];
      entry.n = rhs[rhs.size() - 1];
      break;
    }
    case tflite::BuiltinOperator_CONV_2D: {
      if (inputs.size() < 2 || outputs.empty()) return;
      const auto& in = inputs[0];
      const auto& filt = inputs[1];
      const auto& out = outputs[0];
      // Only flag 1x1 convs as "matmul-equivalent". Conv with kernel > 1 is
      // not a matmul.
      if (filt.size() != 4) return;
      if (filt[1] != 1 || filt[2] != 1) return;
      if (in.size() != 4 || out.size() != 4) return;
      entry.k = in[3];
      entry.n = filt[0];
      entry.m = product(out, 0, 3);  // B * H_out * W_out
      break;
    }
    default:
      break;
  }
}

// Reports whether a BuiltinOperator is matmul-equivalent from the LLM prefill
// point of view. Conv2D is only reported when the filter is 1x1 (inferred in
// InferMNK by leaving m/n/k unset for kernel > 1; we still emit the roster
// entry for visibility but tag non-1x1 conv as such).
bool IsMatmulClass(tflite::BuiltinOperator op) {
  switch (op) {
    case tflite::BuiltinOperator_FULLY_CONNECTED:
    case tflite::BuiltinOperator_BATCH_MATMUL:
      return true;
    case tflite::BuiltinOperator_CONV_2D:
      return true;  // 1x1 filter is matmul; we print shape so the reader
                    // can spot non-1x1.
    default:
      return false;
  }
}

const char* BuiltinOperatorName(tflite::BuiltinOperator op) {
  // EnumNameBuiltinOperator is generated by flatc and returns the short
  // enum name (e.g. "FULLY_CONNECTED"). This gives us stable names without
  // maintaining our own switch.
  return tflite::EnumNameBuiltinOperator(op);
}

}  // namespace

absl::StatusOr<std::vector<MatmulShapeEntry>> CollectMatmulShapes(
    absl::string_view tflite_buffer) {
  if (tflite_buffer.empty()) {
    return absl::InvalidArgumentError("CollectMatmulShapes: empty buffer");
  }
  // Verify the flatbuffer before dereferencing anything. On Adreno we run
  // with unverified buffers from .litertlm, so we explicitly verify here
  // to protect the profiling code path from a corrupt model.
  flatbuffers::Verifier verifier(
      reinterpret_cast<const uint8_t*>(tflite_buffer.data()),
      tflite_buffer.size());
  if (!tflite::VerifyModelBuffer(verifier)) {
    return absl::InvalidArgumentError(
        "CollectMatmulShapes: not a valid tflite flatbuffer");
  }
  const tflite::Model* model = tflite::GetModel(tflite_buffer.data());
  if (model == nullptr || model->subgraphs() == nullptr) {
    return absl::InvalidArgumentError(
        "CollectMatmulShapes: tflite model has no subgraphs");
  }

  std::vector<MatmulShapeEntry> roster;

  // Resolve opcode_index -> BuiltinOperator once. Custom ops have
  // builtin_code == CUSTOM and a custom_code string; the GPU delegate
  // doesn't run custom matmul ops so we skip them.
  const auto* op_codes = model->operator_codes();
  std::vector<tflite::BuiltinOperator> builtin_by_index;
  if (op_codes != nullptr) {
    builtin_by_index.reserve(op_codes->size());
    for (const tflite::OperatorCode* oc : *op_codes) {
      // TFLite deprecated the 8-bit builtin_code in favor of
      // deprecated_builtin_code for large enum values. Pick the newer field
      // first, fall back to the deprecated one when it's < 127.
      builtin_by_index.push_back(ResolveBuiltinCode(oc));
    }
  }

  for (int sg_idx = 0; sg_idx < static_cast<int>(model->subgraphs()->size());
       ++sg_idx) {
    const tflite::SubGraph* sg = model->subgraphs()->Get(sg_idx);
    if (sg == nullptr) continue;
    std::string sg_name;
    if (sg->name() != nullptr) sg_name = sg->name()->str();

    const auto* tensors = sg->tensors();
    const auto* operators = sg->operators();
    if (tensors == nullptr || operators == nullptr) continue;

    for (int op_idx = 0; op_idx < static_cast<int>(operators->size());
         ++op_idx) {
      const tflite::Operator* op = operators->Get(op_idx);
      if (op == nullptr) continue;
      const uint32_t oc_idx = op->opcode_index();
      if (oc_idx >= builtin_by_index.size()) continue;
      const tflite::BuiltinOperator builtin = builtin_by_index[oc_idx];
      if (!IsMatmulClass(builtin)) continue;

      MatmulShapeEntry e;
      e.subgraph_index = sg_idx;
      e.subgraph_name = sg_name;
      e.op_index = op_idx;
      e.op_type = BuiltinOperatorName(builtin);

      auto fill_shapes = [&](const flatbuffers::Vector<int32_t>* indices,
                             std::vector<std::vector<int32_t>>& out_shapes) {
        if (indices == nullptr) return;
        out_shapes.reserve(indices->size());
        for (int32_t t_idx : *indices) {
          if (t_idx < 0 || t_idx >= static_cast<int32_t>(tensors->size())) {
            out_shapes.push_back({});
            continue;
          }
          const tflite::Tensor* t = tensors->Get(t_idx);
          out_shapes.push_back(t == nullptr ? std::vector<int32_t>{}
                                            : ShapeToVec(t->shape()));
        }
      };
      fill_shapes(op->inputs(), e.input_shapes);
      fill_shapes(op->outputs(), e.output_shapes);

      InferMNK(builtin, e.input_shapes, e.output_shapes, e);
      roster.push_back(std::move(e));
    }
  }
  return roster;
}

void DumpMatmulShapes(absl::string_view tflite_buffer,
                      absl::string_view label, FILE* out) {
  if (out == nullptr) out = stderr;
  auto status_or_roster = CollectMatmulShapes(tflite_buffer);
  if (!status_or_roster.ok()) {
    const std::string err_msg(status_or_roster.status().message());
    std::fprintf(out,
                 "\n[PROFILE MATMUL SHAPES] label=%.*s (SKIPPED: %s)\n",
                 static_cast<int>(label.size()), label.data(),
                 err_msg.c_str());
    return;
  }
  const auto& roster = *status_or_roster;

  // Summary banner.
  int64_t fc_count = 0;
  int64_t bmm_count = 0;
  int64_t conv_count = 0;
  int64_t other_count = 0;
  for (const auto& e : roster) {
    if (e.op_type == "FULLY_CONNECTED") {
      ++fc_count;
    } else if (e.op_type == "BATCH_MATMUL") {
      ++bmm_count;
    } else if (e.op_type == "CONV_2D") {
      ++conv_count;
    } else {
      ++other_count;
    }
  }
  std::fprintf(
      out,
      "\n[PROFILE MATMUL SHAPES] label=%.*s total=%zu "
      "(FULLY_CONNECTED=%lld BATCH_MATMUL=%lld CONV_2D=%lld other=%lld)\n"
      "  Use this roster to correlate rows in "
      "[PROFILE LiteRT CompiledModel per-op summary] and "
      "Delegate Statistics (e.g. `convolution_int8(conv_wave_memory)`)\n"
      "  against the actual M/N/K of the source matmul op.\n",
      static_cast<int>(label.size()), label.data(), roster.size(),
      static_cast<long long>(fc_count), static_cast<long long>(bmm_count),
      static_cast<long long>(conv_count),
      static_cast<long long>(other_count));

  if (roster.empty()) {
    std::fprintf(out,
                 "  (no matmul-class ops found -- is this really a prefill"
                 "/decode model?)\n");
    return;
  }

  // Column widths are fixed so the output aligns even on long shape names.
  std::fprintf(
      out,
      "  %-4s %-12s %-20s %-18s %-35s %-30s  M=       N=       K=\n",
      "sg", "sg_name", "op_type", "op_idx", "inputs", "outputs");
  for (const auto& e : roster) {
    std::string inputs_str;
    for (size_t i = 0; i < e.input_shapes.size(); ++i) {
      if (i != 0) inputs_str.push_back('|');
      inputs_str += ShapeToString(e.input_shapes[i]);
    }
    std::string outputs_str;
    for (size_t i = 0; i < e.output_shapes.size(); ++i) {
      if (i != 0) outputs_str.push_back('|');
      outputs_str += ShapeToString(e.output_shapes[i]);
    }

    std::fprintf(out,
                 "  %-4d %-12s %-20s op=%-14d %-35s %-30s  %-8lld %-8lld %-8lld\n",
                 e.subgraph_index, e.subgraph_name.c_str(), e.op_type.c_str(),
                 e.op_index, inputs_str.c_str(), outputs_str.c_str(),
                 static_cast<long long>(e.m), static_cast<long long>(e.n),
                 static_cast<long long>(e.k));
  }
  std::fprintf(out, "\n");
}

absl::Status WriteMatmulShapeRosterCsv(absl::string_view tflite_buffer,
                                       absl::string_view label,
                                       absl::string_view out_path) {
  auto status_or_roster = CollectMatmulShapes(tflite_buffer);
  if (!status_or_roster.ok()) return status_or_roster.status();
  const auto& roster = *status_or_roster;

  std::ofstream ofs{std::string(out_path)};
  if (!ofs) {
    return absl::UnavailableError(
        absl::StrCat("WriteMatmulShapeRosterCsv: cannot open ", out_path));
  }
  ofs << "label,subgraph_idx,subgraph_name,op_idx,op_type,m,n,k,"
         "input_shapes,output_shapes\n";
  const std::string label_str(label);
  for (const auto& e : roster) {
    auto encode_shapes = [](const std::vector<std::vector<int32_t>>& ss) {
      std::string s;
      for (size_t i = 0; i < ss.size(); ++i) {
        if (i != 0) s.push_back('|');
        for (size_t j = 0; j < ss[i].size(); ++j) {
          if (j != 0) s.push_back(',');
          s += std::to_string(ss[i][j]);
        }
      }
      return s;
    };
    ofs << label_str << ',' << e.subgraph_index << ',' << e.subgraph_name
        << ',' << e.op_index << ',' << e.op_type << ',' << e.m << ',' << e.n
        << ',' << e.k << ',' << encode_shapes(e.input_shapes) << ','
        << encode_shapes(e.output_shapes) << '\n';
  }
  return absl::OkStatus();
}

}  // namespace litert::lm
