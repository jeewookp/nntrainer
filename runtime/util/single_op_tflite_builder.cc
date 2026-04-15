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

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tflite/schema/schema_generated.h"  // from @litert

namespace litert::lm {

absl::StatusOr<SingleFcBuildResult> BuildSingleFullyConnectedTfliteModel(
    int64_t m, int64_t n, int64_t k, MatmulDtype dtype) {
  if (m <= 0 || n <= 0 || k <= 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("BuildSingleFullyConnectedTfliteModel: non-positive ",
                     "M/N/K (m=", m, " n=", n, " k=", k, ")"));
  }

  // Per-mode tensor type and element byte width. Four layouts:
  //
  //   kInt8              : input/output INT8, weights INT8 per-channel,
  //                        bias INT32 per-channel, asymmetric_quantize_inputs
  //                        false. Fully-quantized int8 FC. This is the
  //                        layout the LiteRT CL delegate lowers to its
  //                        `convolution_int8(conv_wave_memory)` kernel,
  //                        which is what Gemma4 prefill uses for ~80%
  //                        of its matmul time.
  //
  //   kInt8WeightFp32Act : input/output FLOAT32, weights INT8 with
  //                        per-channel symmetric quant, bias FLOAT32,
  //                        asymmetric_quantize_inputs=true. Hybrid
  //                        path -- the delegate accepts it but does NOT
  //                        pick the conv_wave_memory int8 kernel, so
  //                        timings stay close to fp32. Kept as a
  //                        comparison point.
  //
  //   kFp32              : everything FLOAT32. GPU compiles fp16
  //                        reduction kernels internally via
  //                        SetPrecision(kFp16). Matches the smaller
  //                        `convolution(conv_wave_memory)` rows in
  //                        prefill.
  //
  //   kFp16              : everything FLOAT16. Currently broken (CPU
  //                        FC reference kernel asserts FLOAT32 input
  //                        during prepare).
  //
  // The "act" type drives input/output tensor type. The "weight" type
  // drives the weight tensor + its per-element byte stride (and, for
  // int8 modes, its quantization params). Bias is separate again
  // because in fully-int8 the bias is INT32 per-channel, in hybrid
  // int8 it's FLOAT32, and in fp32/fp16 it follows the activations.
  tflite::TensorType act_tflite_dtype = tflite::TensorType_FLOAT32;
  tflite::TensorType weight_tflite_dtype = tflite::TensorType_FLOAT32;
  tflite::TensorType bias_tflite_dtype = tflite::TensorType_FLOAT32;
  size_t weight_elem_bytes = 0;
  size_t bias_elem_bytes = 0;
  bool asymmetric_quantize_inputs = false;
  bool weights_are_per_channel_quant = false;
  bool acts_are_per_tensor_int8_quant = false;  // input + output
  bool bias_is_per_channel_int32_quant = false;
  switch (dtype) {
    case MatmulDtype::kInt8:
      act_tflite_dtype = tflite::TensorType_INT8;
      weight_tflite_dtype = tflite::TensorType_INT8;
      bias_tflite_dtype = tflite::TensorType_INT32;
      weight_elem_bytes = sizeof(int8_t);
      bias_elem_bytes = sizeof(int32_t);
      asymmetric_quantize_inputs = false;
      weights_are_per_channel_quant = true;
      acts_are_per_tensor_int8_quant = true;
      bias_is_per_channel_int32_quant = true;
      break;
    case MatmulDtype::kInt8WeightFp32Act:
      act_tflite_dtype = tflite::TensorType_FLOAT32;
      weight_tflite_dtype = tflite::TensorType_INT8;
      bias_tflite_dtype = tflite::TensorType_FLOAT32;
      weight_elem_bytes = sizeof(int8_t);
      bias_elem_bytes = sizeof(float);
      asymmetric_quantize_inputs = true;
      weights_are_per_channel_quant = true;
      break;
    case MatmulDtype::kFp32:
      act_tflite_dtype = tflite::TensorType_FLOAT32;
      weight_tflite_dtype = tflite::TensorType_FLOAT32;
      bias_tflite_dtype = tflite::TensorType_FLOAT32;
      weight_elem_bytes = sizeof(float);
      bias_elem_bytes = sizeof(float);
      break;
    case MatmulDtype::kFp16:
      act_tflite_dtype = tflite::TensorType_FLOAT16;
      weight_tflite_dtype = tflite::TensorType_FLOAT16;
      bias_tflite_dtype = tflite::TensorType_FLOAT16;
      weight_elem_bytes = sizeof(uint16_t);
      bias_elem_bytes = sizeof(uint16_t);
      break;
    default:
      return absl::InvalidArgumentError(
          "BuildSingleFullyConnectedTfliteModel: unknown dtype");
  }

  // Symmetric quant constants used by the int8 paths. We use a
  // uniform 1/127 scale across input, weights, and output -- the
  // exact values don't matter for a latency-only benchmark, what
  // matters is that the bias INT32 quant scale is consistent
  // (input_scale * weight_scale[i]) so the delegate's int8 kernel
  // accepts the model without complaint.
  constexpr float kInputScale = 1.0f / 127.0f;
  constexpr float kWeightScale = 1.0f / 127.0f;
  constexpr float kOutputScale = 1.0f / 127.0f;

  // Sanity cap: 1 GiB of weights is way more than any matmul in Gemma4.
  // For fp32 this means at most ~268M weight elements per FC, which is
  // larger than the lm_head (~49M for vocab_size=32003 x hidden=1536).
  // For int8 the same byte budget covers ~1G weight elements, so all
  // Gemma4 shapes fit comfortably.
  constexpr int64_t kMaxBytes = int64_t{1} << 30;
  const int64_t weight_bytes =
      n * k * static_cast<int64_t>(weight_elem_bytes);
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
  //   buffers[1] = constant weights (n*k elements of `weight_tflite_dtype`).
  //   buffers[2] = constant bias    (n elements of `bias_tflite_dtype`).
  //
  // Note: CreateBuffer has two overloads depending on whether a data
  // vector is supplied. For activation tensors we use the no-data form
  // (offset 0 equivalent). For the weight / bias tensors we pass raw
  // bytes through CreateVector(uint8_t*, size).
  auto buffer_empty = tflite::CreateBuffer(fbb);

  // Build the weight buffer as raw zero bytes. We don't care about
  // exact numerical values for a latency-only benchmark: the GPU CL
  // delegate dispatch time depends on shape / precision / weight
  // dtype, not on weight content.
  const std::vector<uint8_t> weight_bytes_vec(
      static_cast<size_t>(weight_bytes), 0);
  auto weights_data_vec = fbb.CreateVector(weight_bytes_vec);
  auto buffer_weights = tflite::CreateBuffer(fbb, weights_data_vec);

  const int64_t bias_bytes = n * static_cast<int64_t>(bias_elem_bytes);
  const std::vector<uint8_t> bias_bytes_vec(
      static_cast<size_t>(bias_bytes), 0);
  auto bias_data_vec = fbb.CreateVector(bias_bytes_vec);
  auto buffer_bias = tflite::CreateBuffer(fbb, bias_data_vec);

  std::vector<flatbuffers::Offset<tflite::Buffer>> buffers_vec = {
      buffer_empty, buffer_weights, buffer_bias};
  auto buffers_fb = fbb.CreateVector(buffers_vec);

  // ---- 1b. Build quantization params for int8 modes ----
  //
  //   weight_quant_params : per-output-channel symmetric quant on the
  //                         weights tensor [N, K]. scale[N] = 1/127,
  //                         zero_point[N] = 0, quantized_dimension = 0.
  //   input_quant_params  : per-tensor symmetric quant on the input
  //                         activation [1, M, K]. scale = 1/127,
  //                         zero_point = 0. Only set in fully-int8.
  //   output_quant_params : per-tensor symmetric quant on the output
  //                         activation [1, M, N]. Same shape as input
  //                         quant. Only set in fully-int8.
  //   bias_quant_params   : per-channel quant on the bias tensor [N]
  //                         with INT32 storage. scale[i] = input_scale
  //                         * weight_scale[i], zero_point[i] = 0,
  //                         quantized_dimension = 0. Only set in
  //                         fully-int8 (hybrid uses fp32 bias, no
  //                         quant params).
  //
  // The exact scale values don't affect latency: what matters is
  // that the schema is valid for the delegate's int8 conv path.
  flatbuffers::Offset<tflite::QuantizationParameters> weight_quant_params = 0;
  flatbuffers::Offset<tflite::QuantizationParameters> input_quant_params = 0;
  flatbuffers::Offset<tflite::QuantizationParameters> output_quant_params = 0;
  flatbuffers::Offset<tflite::QuantizationParameters> bias_quant_params = 0;

  if (weights_are_per_channel_quant) {
    std::vector<float> scales(n, kWeightScale);
    std::vector<int64_t> zero_points(n, 0);
    auto scales_fb = fbb.CreateVector(scales);
    auto zero_points_fb = fbb.CreateVector(zero_points);
    tflite::QuantizationParametersBuilder qp_builder(fbb);
    qp_builder.add_scale(scales_fb);
    qp_builder.add_zero_point(zero_points_fb);
    qp_builder.add_quantized_dimension(0);
    weight_quant_params = qp_builder.Finish();
  }

  if (acts_are_per_tensor_int8_quant) {
    // Input quant: single per-tensor scale.
    {
      const std::vector<float> scales = {kInputScale};
      const std::vector<int64_t> zero_points = {0};
      auto scales_fb = fbb.CreateVector(scales);
      auto zero_points_fb = fbb.CreateVector(zero_points);
      tflite::QuantizationParametersBuilder qp_builder(fbb);
      qp_builder.add_scale(scales_fb);
      qp_builder.add_zero_point(zero_points_fb);
      qp_builder.add_quantized_dimension(0);
      input_quant_params = qp_builder.Finish();
    }
    // Output quant: single per-tensor scale.
    {
      const std::vector<float> scales = {kOutputScale};
      const std::vector<int64_t> zero_points = {0};
      auto scales_fb = fbb.CreateVector(scales);
      auto zero_points_fb = fbb.CreateVector(zero_points);
      tflite::QuantizationParametersBuilder qp_builder(fbb);
      qp_builder.add_scale(scales_fb);
      qp_builder.add_zero_point(zero_points_fb);
      qp_builder.add_quantized_dimension(0);
      output_quant_params = qp_builder.Finish();
    }
  }

  if (bias_is_per_channel_int32_quant) {
    // For fully-int8 FC the bias INT32 storage requires per-channel
    // scales = input_scale * weight_scale[i]. With our uniform 1/127
    // weight scale this collapses to a constant 1/(127*127) per
    // channel -- but we still emit n entries to keep the
    // quantized_dimension=0 contract consistent with the weights.
    std::vector<float> scales(n, kInputScale * kWeightScale);
    std::vector<int64_t> zero_points(n, 0);
    auto scales_fb = fbb.CreateVector(scales);
    auto zero_points_fb = fbb.CreateVector(zero_points);
    tflite::QuantizationParametersBuilder qp_builder(fbb);
    qp_builder.add_scale(scales_fb);
    qp_builder.add_zero_point(zero_points_fb);
    qp_builder.add_quantized_dimension(0);
    bias_quant_params = qp_builder.Finish();
  }

  // ---- 2. Build the tensors ----
  //
  // Tensor indices used below in the operator / subgraph:
  //   0 : input  x  [1, m, k]  act dtype, buffer=0 (activation)
  //   1 : weight w  [n, k]     weight dtype (+ quant for int8), buffer=1
  //   2 : bias   b  [n]        bias dtype, buffer=2
  //   3 : output y  [1, m, n]  act dtype, buffer=0 (activation)
  //
  // Note: we pass std::vector<int32_t> (not initializer_list) to
  // CreateVector because some pinned flatbuffers revisions in @litert
  // don't ship the initializer_list overload.
  const std::vector<int32_t> input_shape_v = {
      1, static_cast<int32_t>(m), static_cast<int32_t>(k)};
  auto input_shape = fbb.CreateVector(input_shape_v);
  auto input_name = fbb.CreateString("x");
  auto tensor_input = tflite::CreateTensor(
      fbb, input_shape, act_tflite_dtype,
      /*buffer=*/0, input_name,
      /*quantization=*/input_quant_params);

  const std::vector<int32_t> weights_shape_v = {
      static_cast<int32_t>(n), static_cast<int32_t>(k)};
  auto weights_shape = fbb.CreateVector(weights_shape_v);
  auto weights_name = fbb.CreateString("w");
  auto tensor_weights = tflite::CreateTensor(
      fbb, weights_shape, weight_tflite_dtype,
      /*buffer=*/1, weights_name,
      /*quantization=*/weight_quant_params);

  const std::vector<int32_t> bias_shape_v = {static_cast<int32_t>(n)};
  auto bias_shape = fbb.CreateVector(bias_shape_v);
  auto bias_name = fbb.CreateString("b");
  auto tensor_bias = tflite::CreateTensor(
      fbb, bias_shape, bias_tflite_dtype,
      /*buffer=*/2, bias_name,
      /*quantization=*/bias_quant_params);

  const std::vector<int32_t> output_shape_v = {
      1, static_cast<int32_t>(m), static_cast<int32_t>(n)};
  auto output_shape = fbb.CreateVector(output_shape_v);
  auto output_name = fbb.CreateString("y");
  auto tensor_output = tflite::CreateTensor(
      fbb, output_shape, act_tflite_dtype,
      /*buffer=*/0, output_name,
      /*quantization=*/output_quant_params);

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

  // For the int8-weight hybrid path, asymmetric_quantize_inputs=true
  // tells the LiteRT runtime / GPU delegate to dynamically quantize
  // the float input to int8 at dispatch time and run the int8
  // matmul kernel. For the fp32 / fp16 paths it stays false.
  auto fc_options = tflite::CreateFullyConnectedOptions(
      fbb, tflite::ActivationFunctionType_NONE,
      tflite::FullyConnectedOptionsWeightsFormat_DEFAULT,
      /*keep_num_dims=*/true,
      /*asymmetric_quantize_inputs=*/asymmetric_quantize_inputs);

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
