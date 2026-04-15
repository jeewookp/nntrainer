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

#include "runtime/util/single_op_tflite_builder.h"

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tflite/schema/schema_generated.h"  // from @litert

namespace litert::lm {

namespace {

// IEEE-754 binary16 (fp16) bit patterns for a handful of small values.
// We only need a deterministic, bounded-magnitude pattern so every
// benchmark run uses the same weight bytes (which keeps the GPU weight
// cache happy) and so the numbers don't overflow / underflow during
// the FC dispatch.
constexpr uint16_t kFp16Pattern[] = {
    0x0000,  // +0.0
    0x2400,  // +0.0625 / 2 = 0.03125
    0x2800,  // +0.0625
    0x2c00,  // +0.09375
    0x3000,  // +0.125
    0x3400,  // +0.25
    0x3800,  // +0.5
    0x3c00,  // +1.0
    0xa400,  // -0.03125
    0xa800,  // -0.0625
    0xac00,  // -0.09375
    0xb000,  // -0.125
    0xb400,  // -0.25
    0xb800,  // -0.5
    0xbc00,  // -1.0
    0x3c00,  // +1.0 again (rotating)
};
constexpr int kFp16PatternLen =
    sizeof(kFp16Pattern) / sizeof(kFp16Pattern[0]);

// Fills a vector with `num_elements` fp16 bit patterns from kFp16Pattern,
// offset by `seed` to give different tensors distinct-but-deterministic
// values.
std::vector<uint16_t> MakeFp16Pattern(int64_t num_elements, int seed) {
  std::vector<uint16_t> data;
  data.reserve(num_elements);
  for (int64_t i = 0; i < num_elements; ++i) {
    data.push_back(kFp16Pattern[(i + seed) % kFp16PatternLen]);
  }
  return data;
}

}  // namespace

absl::StatusOr<SingleFcBuildResult> BuildSingleFullyConnectedTfliteModel(
    int64_t m, int64_t n, int64_t k, MatmulDtype dtype) {
  if (m <= 0 || n <= 0 || k <= 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("BuildSingleFullyConnectedTfliteModel: non-positive ",
                     "M/N/K (m=", m, " n=", n, " k=", k, ")"));
  }
  if (dtype != MatmulDtype::kFp16) {
    return absl::InvalidArgumentError(
        "BuildSingleFullyConnectedTfliteModel: only fp16 dtype is "
        "implemented");
  }
  // Sanity cap: 1 GiB of weights is way more than any matmul in Gemma4.
  constexpr int64_t kMaxBytes = int64_t{1} << 30;
  const int64_t weight_bytes = n * k * static_cast<int64_t>(sizeof(uint16_t));
  if (weight_bytes <= 0 || weight_bytes > kMaxBytes) {
    return absl::InvalidArgumentError(
        absl::StrCat("BuildSingleFullyConnectedTfliteModel: weight tensor ",
                     "too large (", weight_bytes, " bytes > 1 GiB)"));
  }

  flatbuffers::FlatBufferBuilder fbb(/*initial_size=*/static_cast<size_t>(
      weight_bytes + 64 * 1024));

  // ---- 1. Build the data buffers ----
  //
  // Buffer indexing convention:
  //   buffers[0] = empty sentinel (TFLite requires buffer 0 to exist and
  //                be referenced by activation tensors that have no
  //                constant data).
  //   buffers[1] = constant weights (fp16, n*k elements).
  //   buffers[2] = constant bias    (fp16 zeros, n elements).
  //
  // Note: CreateBuffer has two overloads depending on whether a data
  // vector is supplied. For activation tensors we use the no-data form
  // (offset 0 equivalent). For the weight / bias tensors we pass raw
  // bytes through CreateVector(uint8_t*, size).
  auto buffer_empty = tflite::CreateBuffer(fbb);

  const auto weights = MakeFp16Pattern(n * k, /*seed=*/1);
  auto weights_data_vec = fbb.CreateVector(
      reinterpret_cast<const uint8_t*>(weights.data()),
      weights.size() * sizeof(uint16_t));
  auto buffer_weights = tflite::CreateBuffer(fbb, weights_data_vec);

  const std::vector<uint16_t> bias(n, 0x0000);  // fp16 zero
  auto bias_data_vec = fbb.CreateVector(
      reinterpret_cast<const uint8_t*>(bias.data()),
      bias.size() * sizeof(uint16_t));
  auto buffer_bias = tflite::CreateBuffer(fbb, bias_data_vec);

  std::vector<flatbuffers::Offset<tflite::Buffer>> buffers_vec = {
      buffer_empty, buffer_weights, buffer_bias};
  auto buffers_fb = fbb.CreateVector(buffers_vec);

  // ---- 2. Build the tensors ----
  //
  // Tensor indices used below in the operator / subgraph:
  //   0 : input  x  [1, m, k]  fp16, buffer=0 (activation)
  //   1 : weight w  [n, k]     fp16, buffer=1 (constant)
  //   2 : bias   b  [n]        fp16, buffer=2 (constant)
  //   3 : output y  [1, m, n]  fp16, buffer=0 (activation)
  //
  // Note: we pass std::vector<int32_t> (not initializer_list) to
  // CreateVector because some pinned flatbuffers revisions in @litert
  // don't ship the initializer_list overload.
  const std::vector<int32_t> input_shape_v = {
      1, static_cast<int32_t>(m), static_cast<int32_t>(k)};
  auto input_shape = fbb.CreateVector(input_shape_v);
  auto input_name = fbb.CreateString("x");
  auto tensor_input = tflite::CreateTensor(
      fbb, input_shape, tflite::TensorType_FLOAT16,
      /*buffer=*/0, input_name);

  const std::vector<int32_t> weights_shape_v = {
      static_cast<int32_t>(n), static_cast<int32_t>(k)};
  auto weights_shape = fbb.CreateVector(weights_shape_v);
  auto weights_name = fbb.CreateString("w");
  auto tensor_weights = tflite::CreateTensor(
      fbb, weights_shape, tflite::TensorType_FLOAT16,
      /*buffer=*/1, weights_name);

  const std::vector<int32_t> bias_shape_v = {static_cast<int32_t>(n)};
  auto bias_shape = fbb.CreateVector(bias_shape_v);
  auto bias_name = fbb.CreateString("b");
  auto tensor_bias = tflite::CreateTensor(
      fbb, bias_shape, tflite::TensorType_FLOAT16,
      /*buffer=*/2, bias_name);

  const std::vector<int32_t> output_shape_v = {
      1, static_cast<int32_t>(m), static_cast<int32_t>(n)};
  auto output_shape = fbb.CreateVector(output_shape_v);
  auto output_name = fbb.CreateString("y");
  auto tensor_output = tflite::CreateTensor(
      fbb, output_shape, tflite::TensorType_FLOAT16,
      /*buffer=*/0, output_name);

  std::vector<flatbuffers::Offset<tflite::Tensor>> tensors_vec = {
      tensor_input, tensor_weights, tensor_bias, tensor_output};
  auto tensors_fb = fbb.CreateVector(tensors_vec);

  // ---- 3. Build the operator code + operator ----
  //
  // We use the modern builtin_code field (int32) and also populate the
  // deprecated_builtin_code field for backward compat with older
  // schema consumers. Version 9 covers keep_num_dims +
  // asymmetric_quantize_inputs which we set below.
  auto opcode = tflite::CreateOperatorCode(
      fbb,
      /*deprecated_builtin_code=*/static_cast<int8_t>(
          tflite::BuiltinOperator_FULLY_CONNECTED),
      /*custom_code=*/0,
      /*version=*/9,
      /*builtin_code=*/tflite::BuiltinOperator_FULLY_CONNECTED);
  std::vector<flatbuffers::Offset<tflite::OperatorCode>> opcodes_vec = {
      opcode};
  auto opcodes_fb = fbb.CreateVector(opcodes_vec);

  auto fc_options = tflite::CreateFullyConnectedOptions(
      fbb, tflite::ActivationFunctionType_NONE,
      tflite::FullyConnectedOptionsWeightsFormat_DEFAULT,
      /*keep_num_dims=*/true,
      /*asymmetric_quantize_inputs=*/false);

  const std::vector<int32_t> op_inputs_v = {0, 1, 2};
  const std::vector<int32_t> op_outputs_v = {3};
  auto op_inputs_fb = fbb.CreateVector(op_inputs_v);
  auto op_outputs_fb = fbb.CreateVector(op_outputs_v);
  auto op = tflite::CreateOperator(
      fbb, /*opcode_index=*/0, op_inputs_fb, op_outputs_fb,
      tflite::BuiltinOptions_FullyConnectedOptions, fc_options.Union());
  std::vector<flatbuffers::Offset<tflite::Operator>> operators_vec = {op};
  auto operators_fb = fbb.CreateVector(operators_vec);

  // ---- 4. Build the subgraph ----
  const std::vector<int32_t> sg_inputs_v = {0};
  const std::vector<int32_t> sg_outputs_v = {3};
  auto sg_inputs = fbb.CreateVector(sg_inputs_v);
  auto sg_outputs = fbb.CreateVector(sg_outputs_v);
  auto sg_name = fbb.CreateString("main");
  auto subgraph = tflite::CreateSubGraph(fbb, tensors_fb, sg_inputs,
                                          sg_outputs, operators_fb, sg_name);
  std::vector<flatbuffers::Offset<tflite::SubGraph>> subgraphs_vec = {subgraph};
  auto subgraphs_fb = fbb.CreateVector(subgraphs_vec);

  // ---- 5. Build the signature def ----
  //
  // LiteRT's CompiledModel API is signature-keyed, so we expose a single
  // signature named "main" with one input "x" -> tensor 0 and one output
  // "y" -> tensor 3. The microbenchmark binary uses these names to find
  // the input/output tensor buffers.
  //
  // We use SignatureDefBuilder instead of the positional
  // CreateSignatureDef helper because the schema has a vestigial
  // `deprecated_tag` field between `signature_key` and `subgraph_index`,
  // and getting the positional ordering wrong there silently shifts
  // subgraph_index into the wrong slot.
  auto sig_in_name = fbb.CreateString("x");
  auto sig_in = tflite::CreateTensorMap(fbb, sig_in_name, /*tensor_index=*/0);
  auto sig_out_name = fbb.CreateString("y");
  auto sig_out = tflite::CreateTensorMap(fbb, sig_out_name, /*tensor_index=*/3);
  std::vector<flatbuffers::Offset<tflite::TensorMap>> sig_ins_vec = {sig_in};
  std::vector<flatbuffers::Offset<tflite::TensorMap>> sig_outs_vec = {sig_out};
  auto sig_ins = fbb.CreateVector(sig_ins_vec);
  auto sig_outs = fbb.CreateVector(sig_outs_vec);
  auto sig_key = fbb.CreateString("main");
  flatbuffers::Offset<tflite::SignatureDef> sig_def;
  {
    tflite::SignatureDefBuilder sig_def_builder(fbb);
    sig_def_builder.add_inputs(sig_ins);
    sig_def_builder.add_outputs(sig_outs);
    sig_def_builder.add_signature_key(sig_key);
    sig_def_builder.add_subgraph_index(0);
    sig_def = sig_def_builder.Finish();
  }
  std::vector<flatbuffers::Offset<tflite::SignatureDef>> sig_defs_vec = {
      sig_def};
  auto sig_defs_fb = fbb.CreateVector(sig_defs_vec);

  // ---- 6. Build the top-level Model ----
  auto description = fbb.CreateString(
      absl::StrCat("matmul_micro_benchmark fc m=", m, " n=", n, " k=", k));
  auto model = tflite::CreateModel(
      fbb,
      /*version=*/3,
      /*operator_codes=*/opcodes_fb,
      /*subgraphs=*/subgraphs_fb,
      /*description=*/description,
      /*buffers=*/buffers_fb,
      /*metadata_buffer=*/0,
      /*metadata=*/0,
      /*signature_defs=*/sig_defs_fb);
  tflite::FinishModelBuffer(fbb, model);

  SingleFcBuildResult result;
  result.flatbuffer.assign(
      reinterpret_cast<const char*>(fbb.GetBufferPointer()),
      fbb.GetSize());
  result.signature_key = "main";
  result.input_name = "x";
  result.output_name = "y";
  return result;
}

}  // namespace litert::lm
