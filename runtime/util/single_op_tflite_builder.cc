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
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tflite/schema/schema_generated.h"  // from @litert

namespace litert::lm {

namespace {

// Helpers shared between the single-op and the wrapped (3-op) builder
// paths. They keep the flatbuffer field-ordering noise out of the main
// functions, which makes the data-flow easier to follow.

// Builds a per-tensor symmetric quantization params table:
//   scale = [scale_value], zero_point = [0], quantized_dimension = 0.
// Used for the int8 input / int8 output activation tensors.
flatbuffers::Offset<tflite::QuantizationParameters> BuildPerTensorSymQuant(
    flatbuffers::FlatBufferBuilder& fbb, float scale_value) {
  const std::vector<float> scales = {scale_value};
  const std::vector<int64_t> zero_points = {0};
  auto scales_fb = fbb.CreateVector(scales);
  auto zero_points_fb = fbb.CreateVector(zero_points);
  tflite::QuantizationParametersBuilder qp_builder(fbb);
  qp_builder.add_scale(scales_fb);
  qp_builder.add_zero_point(zero_points_fb);
  qp_builder.add_quantized_dimension(0);
  return qp_builder.Finish();
}

// Builds a per-output-channel symmetric quant params table with `n`
// uniform scales (`scale_value` per channel) and zero zero-points.
// Used for the int8 weights tensor [n, k] with quantized_dimension=0.
flatbuffers::Offset<tflite::QuantizationParameters> BuildPerChannelSymQuant(
    flatbuffers::FlatBufferBuilder& fbb, int64_t n, float scale_value) {
  std::vector<float> scales(static_cast<size_t>(n), scale_value);
  std::vector<int64_t> zero_points(static_cast<size_t>(n), 0);
  auto scales_fb = fbb.CreateVector(scales);
  auto zero_points_fb = fbb.CreateVector(zero_points);
  tflite::QuantizationParametersBuilder qp_builder(fbb);
  qp_builder.add_scale(scales_fb);
  qp_builder.add_zero_point(zero_points_fb);
  qp_builder.add_quantized_dimension(0);
  return qp_builder.Finish();
}

// Builds an OperatorCode entry for a builtin op. Sets both the
// deprecated 8-bit field and the modern 32-bit field so the model
// stays compatible with both old and new schema readers.
flatbuffers::Offset<tflite::OperatorCode> BuildBuiltinOpCode(
    flatbuffers::FlatBufferBuilder& fbb, tflite::BuiltinOperator op,
    int32_t version) {
  return tflite::CreateOperatorCode(
      fbb,
      /*deprecated_builtin_code=*/static_cast<int8_t>(op),
      /*custom_code=*/0,
      /*version=*/version,
      /*builtin_code=*/op);
}

// Builds the SignatureDef table via the SignatureDefBuilder helper.
// The vestigial `deprecated_tag` field between `signature_key` and
// `subgraph_index` makes the positional CreateSignatureDef helper
// fragile, so the explicit builder is the safer choice.
flatbuffers::Offset<tflite::SignatureDef> BuildSignatureDef(
    flatbuffers::FlatBufferBuilder& fbb,
    flatbuffers::Offset<flatbuffers::String> signature_key,
    flatbuffers::Offset<flatbuffers::String> in_name, int32_t in_tensor_idx,
    flatbuffers::Offset<flatbuffers::String> out_name,
    int32_t out_tensor_idx) {
  auto sig_in = tflite::CreateTensorMap(fbb, in_name, in_tensor_idx);
  auto sig_out = tflite::CreateTensorMap(fbb, out_name, out_tensor_idx);
  std::vector<flatbuffers::Offset<tflite::TensorMap>> sig_ins_vec = {sig_in};
  std::vector<flatbuffers::Offset<tflite::TensorMap>> sig_outs_vec = {sig_out};
  auto sig_ins = fbb.CreateVector(sig_ins_vec);
  auto sig_outs = fbb.CreateVector(sig_outs_vec);
  tflite::SignatureDefBuilder sig_def_builder(fbb);
  sig_def_builder.add_inputs(sig_ins);
  sig_def_builder.add_outputs(sig_outs);
  sig_def_builder.add_signature_key(signature_key);
  sig_def_builder.add_subgraph_index(0);
  return sig_def_builder.Finish();
}

// Forward declaration of the wrapped int8 path. Defined below the
// main BuildSingleFullyConnectedTfliteModel function.
absl::StatusOr<SingleFcBuildResult> BuildWrappedInt8Fc(int64_t m, int64_t n,
                                                        int64_t k);

// Forward declaration of the CONV_2D 1x1 path. Defined below the
// main BuildSingleFullyConnectedTfliteModel function.
absl::StatusOr<SingleFcBuildResult> BuildFp32Conv2d1x1(int64_t m, int64_t n,
                                                        int64_t k);

}  // namespace

absl::StatusOr<SingleFcBuildResult> BuildSingleFullyConnectedTfliteModel(
    int64_t m, int64_t n, int64_t k, MatmulDtype dtype) {
  if (m <= 0 || n <= 0 || k <= 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("BuildSingleFullyConnectedTfliteModel: non-positive ",
                     "M/N/K (m=", m, " n=", n, " k=", k, ")"));
  }

  // The fully-int8 path is implemented in BuildWrappedInt8Fc as a
  // 3-op subgraph (QUANTIZE -> FC -> DEQUANTIZE) with FLOAT32 signature
  // I/O. It's structurally different enough from the single-op modes
  // (different tensor count, different operator list, different
  // signature wiring) that inlining it here would obscure the data
  // flow, so it lives in its own helper function.
  if (dtype == MatmulDtype::kInt8) {
    return BuildWrappedInt8Fc(m, n, k);
  }

  // CONV_2D 1x1 path. Different op type from FULLY_CONNECTED so the
  // GPU CL delegate goes through a different lowering pipeline and
  // (per profile data) hits the faster `convolution(conv_wave_memory)`
  // entry point instead of the FC-rewritten `convolution1x1(...)`.
  if (dtype == MatmulDtype::kFp32Conv2d) {
    return BuildFp32Conv2d1x1(m, n, k);
  }

  // Per-mode tensor type and element byte width for the SINGLE-OP path.
  // The fully-int8 path was redirected above to BuildWrappedInt8Fc.
  //
  //   kInt8WeightFp32Act : input/output FLOAT32, weights INT8 with
  //                        per-channel symmetric quant, bias FLOAT32,
  //                        asymmetric_quantize_inputs=true. Hybrid
  //                        quant -- the delegate accepts it but does
  //                        NOT pick the conv_wave_memory int8 kernel,
  //                        so timings stay close to fp32.
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
  tflite::TensorType act_tflite_dtype = tflite::TensorType_FLOAT32;
  tflite::TensorType weight_tflite_dtype = tflite::TensorType_FLOAT32;
  tflite::TensorType bias_tflite_dtype = tflite::TensorType_FLOAT32;
  size_t weight_elem_bytes = 0;
  size_t bias_elem_bytes = 0;
  bool asymmetric_quantize_inputs = false;
  bool weights_are_per_channel_quant = false;
  switch (dtype) {
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
    case MatmulDtype::kInt8:
      // Already routed to BuildWrappedInt8Fc above; this case is
      // present only to silence -Wswitch.
      return absl::InternalError(
          "kInt8 should have been dispatched to BuildWrappedInt8Fc");
    case MatmulDtype::kFp32Conv2d:
      // Already routed to BuildFp32Conv2d1x1 above.
      return absl::InternalError(
          "kFp32Conv2d should have been dispatched to BuildFp32Conv2d1x1");
    default:
      return absl::InvalidArgumentError(
          "BuildSingleFullyConnectedTfliteModel: unknown dtype");
  }

  // Hybrid int8: per-channel weight scale of 1/127. Exact value
  // doesn't affect latency.
  constexpr float kWeightScale = 1.0f / 127.0f;

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

  // ---- 1b. Build quantization params for the int8 hybrid path ----
  //
  // Only the weights tensor gets quant params in this path; the
  // input/output/bias remain plain fp32. The fully-int8 path
  // (kInt8) lives in BuildWrappedInt8Fc and builds its own params.
  flatbuffers::Offset<tflite::QuantizationParameters> weight_quant_params = 0;
  if (weights_are_per_channel_quant) {
    weight_quant_params = BuildPerChannelSymQuant(fbb, n, kWeightScale);
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
      /*buffer=*/0, input_name);

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
      /*buffer=*/2, bias_name);

  const std::vector<int32_t> output_shape_v = {
      1, static_cast<int32_t>(m), static_cast<int32_t>(n)};
  auto output_shape = fbb.CreateVector(output_shape_v);
  auto output_name = fbb.CreateString("y");
  auto tensor_output = tflite::CreateTensor(
      fbb, output_shape, act_tflite_dtype,
      /*buffer=*/0, output_name);

  std::vector<flatbuffers::Offset<tflite::Tensor>> tensors_vec = {
      tensor_input, tensor_weights, tensor_bias, tensor_output};
  auto tensors_fb = fbb.CreateVector(tensors_vec);

  // ---- 3. Build the operator code + operator ----
  //
  // FullyConnected version 9 covers keep_num_dims +
  // asymmetric_quantize_inputs.
  auto opcode = BuildBuiltinOpCode(
      fbb, tflite::BuiltinOperator_FULLY_CONNECTED, /*version=*/9);
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
  // Single signature "main" with input "x" -> tensor 0 and output
  // "y" -> tensor 3. The microbenchmark binary uses these names to
  // find the input/output tensor buffers.
  auto sig_key = fbb.CreateString("main");
  auto sig_in_name = fbb.CreateString("x");
  auto sig_out_name = fbb.CreateString("y");
  auto sig_def = BuildSignatureDef(
      fbb, sig_key, sig_in_name, /*in_tensor_idx=*/0, sig_out_name,
      /*out_tensor_idx=*/3);
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

namespace {

// Builds a 3-op subgraph that wraps a fully-quantized int8
// FULLY_CONNECTED with QUANTIZE on the input side and DEQUANTIZE on
// the output side. The signature keeps FLOAT32 input/output so
// litert::CompiledModel can negotiate buffer requirements normally
// (a previous experiment with INT8 signature I/O hit
// `Failed to get buffer requirements for tensor` warnings inside
// compiled_model.cc and ran 2x slower than the fp32 path). The FC
// itself sees INT8 input + INT8 weights + INT32 bias + INT8 output,
// which is the schema the LiteRT GPU CL delegate lowers to its
// `convolution_int8(conv_wave_memory)` kernel. The delegate fuses
// QUANTIZE + FC + DEQUANTIZE into one dispatch (the same way Gemma4
// prefill's profile output shows the fused chain
// `convolution_int8(conv_wave_memory) -> dequantize_to_float16 ->
// quantize_and_dequantize`).
absl::StatusOr<SingleFcBuildResult> BuildWrappedInt8Fc(int64_t m, int64_t n,
                                                        int64_t k) {
  // Sanity cap: 256 MiB of int8 weights. Tighter than the single-op
  // path's 1 GiB cap because the LiteRT GPU CL delegate appears to
  // expand int8 weights into a 2x larger internal layout for
  // SIMD/wave packing -- on the bogus n=k=32003 shapes from the
  // matmul roster the delegate tries to allocate ~2 GB of device
  // memory (32003 * 32003 * 2 bytes) and clCreateBuffer fails. Real
  // Gemma4 matmul shapes top out around vocab_size * hidden_dim
  // (~49 M weight elements = ~50 MB int8), well under the cap.
  constexpr int64_t kMaxBytes = int64_t{256} << 20;
  const int64_t weight_bytes = n * k;  // 1 byte per int8
  if (weight_bytes <= 0 || weight_bytes > kMaxBytes) {
    return absl::InvalidArgumentError(absl::StrCat(
        "BuildWrappedInt8Fc: weight tensor too large (", weight_bytes,
        " bytes > 256 MiB; the GPU CL delegate uses ~2x this for an "
        "internal SIMD layout and clCreateBuffer fails on Adreno)"));
  }

  flatbuffers::FlatBufferBuilder fbb(/*initial_size=*/static_cast<size_t>(
      weight_bytes + 64 * 1024));

  // Symmetric quant scale used everywhere. Exact value doesn't affect
  // GPU kernel selection or latency; what matters is that the bias
  // INT32 quant scale is consistent with input_scale * weight_scale[i]
  // so the delegate's int8 prepare path accepts the model.
  constexpr float kInputScale = 1.0f / 127.0f;
  constexpr float kWeightScale = 1.0f / 127.0f;
  constexpr float kOutputScale = 1.0f / 127.0f;
  constexpr float kBiasScale = kInputScale * kWeightScale;

  // ---- Buffers ----
  //   buffers[0] : empty sentinel (referenced by activation tensors)
  //   buffers[1] : int8 weights, n*k bytes
  //   buffers[2] : int32 bias, n*4 bytes
  auto buffer_empty = tflite::CreateBuffer(fbb);

  const std::vector<uint8_t> weight_bytes_vec(static_cast<size_t>(weight_bytes),
                                                0);
  auto weights_data_vec = fbb.CreateVector(weight_bytes_vec);
  auto buffer_weights = tflite::CreateBuffer(fbb, weights_data_vec);

  const std::vector<uint8_t> bias_bytes_vec(
      static_cast<size_t>(n) * sizeof(int32_t), 0);
  auto bias_data_vec = fbb.CreateVector(bias_bytes_vec);
  auto buffer_bias = tflite::CreateBuffer(fbb, bias_data_vec);

  std::vector<flatbuffers::Offset<tflite::Buffer>> buffers_vec = {
      buffer_empty, buffer_weights, buffer_bias};
  auto buffers_fb = fbb.CreateVector(buffers_vec);

  // ---- Quant params ----
  auto input_qp = BuildPerTensorSymQuant(fbb, kInputScale);
  auto weight_qp = BuildPerChannelSymQuant(fbb, n, kWeightScale);
  auto output_qp = BuildPerTensorSymQuant(fbb, kOutputScale);
  // Bias INT32 per-channel: scale[i] = input_scale * weight_scale[i].
  // With uniform weight scale this collapses to a constant per channel,
  // but we still emit n entries to match quantized_dimension=0.
  auto bias_qp = BuildPerChannelSymQuant(fbb, n, kBiasScale);

  // ---- Tensors ----
  //   0 : input_fp32  [1, m, k] FLOAT32 (signature input "x")
  //   1 : weights_int8 [n, k]   INT8    (constant, buffer 1, weight_qp)
  //   2 : bias_int32   [n]      INT32   (constant, buffer 2, bias_qp)
  //   3 : input_int8   [1, m, k] INT8   (intermediate, input_qp)
  //   4 : output_int8  [1, m, n] INT8   (intermediate, output_qp)
  //   5 : output_fp32  [1, m, n] FLOAT32 (signature output "y")
  std::vector<flatbuffers::Offset<tflite::Tensor>> tensors_vec;

  // Tensor 0 : input_fp32 "x"
  {
    const std::vector<int32_t> shape = {1, static_cast<int32_t>(m),
                                          static_cast<int32_t>(k)};
    auto shape_fb = fbb.CreateVector(shape);
    auto name = fbb.CreateString("x");
    tensors_vec.push_back(tflite::CreateTensor(
        fbb, shape_fb, tflite::TensorType_FLOAT32, /*buffer=*/0, name));
  }
  // Tensor 1 : weights_int8
  {
    const std::vector<int32_t> shape = {static_cast<int32_t>(n),
                                          static_cast<int32_t>(k)};
    auto shape_fb = fbb.CreateVector(shape);
    auto name = fbb.CreateString("w");
    tensors_vec.push_back(tflite::CreateTensor(
        fbb, shape_fb, tflite::TensorType_INT8, /*buffer=*/1, name,
        /*quantization=*/weight_qp));
  }
  // Tensor 2 : bias_int32
  {
    const std::vector<int32_t> shape = {static_cast<int32_t>(n)};
    auto shape_fb = fbb.CreateVector(shape);
    auto name = fbb.CreateString("b");
    tensors_vec.push_back(tflite::CreateTensor(
        fbb, shape_fb, tflite::TensorType_INT32, /*buffer=*/2, name,
        /*quantization=*/bias_qp));
  }
  // Tensor 3 : input_int8 (intermediate, after QUANTIZE)
  {
    const std::vector<int32_t> shape = {1, static_cast<int32_t>(m),
                                          static_cast<int32_t>(k)};
    auto shape_fb = fbb.CreateVector(shape);
    auto name = fbb.CreateString("x_int8");
    tensors_vec.push_back(tflite::CreateTensor(
        fbb, shape_fb, tflite::TensorType_INT8, /*buffer=*/0, name,
        /*quantization=*/input_qp));
  }
  // Tensor 4 : output_int8 (intermediate, before DEQUANTIZE)
  {
    const std::vector<int32_t> shape = {1, static_cast<int32_t>(m),
                                          static_cast<int32_t>(n)};
    auto shape_fb = fbb.CreateVector(shape);
    auto name = fbb.CreateString("y_int8");
    tensors_vec.push_back(tflite::CreateTensor(
        fbb, shape_fb, tflite::TensorType_INT8, /*buffer=*/0, name,
        /*quantization=*/output_qp));
  }
  // Tensor 5 : output_fp32 "y"
  {
    const std::vector<int32_t> shape = {1, static_cast<int32_t>(m),
                                          static_cast<int32_t>(n)};
    auto shape_fb = fbb.CreateVector(shape);
    auto name = fbb.CreateString("y");
    tensors_vec.push_back(tflite::CreateTensor(
        fbb, shape_fb, tflite::TensorType_FLOAT32, /*buffer=*/0, name));
  }
  auto tensors_fb = fbb.CreateVector(tensors_vec);

  // ---- Operator codes ----
  //   index 0 : QUANTIZE
  //   index 1 : FULLY_CONNECTED
  //   index 2 : DEQUANTIZE
  std::vector<flatbuffers::Offset<tflite::OperatorCode>> opcodes_vec;
  opcodes_vec.push_back(
      BuildBuiltinOpCode(fbb, tflite::BuiltinOperator_QUANTIZE, /*version=*/2));
  opcodes_vec.push_back(BuildBuiltinOpCode(
      fbb, tflite::BuiltinOperator_FULLY_CONNECTED, /*version=*/9));
  opcodes_vec.push_back(BuildBuiltinOpCode(
      fbb, tflite::BuiltinOperator_DEQUANTIZE, /*version=*/2));
  auto opcodes_fb = fbb.CreateVector(opcodes_vec);

  // ---- Operators ----
  //   op 0 : QUANTIZE        (tensor 0 -> tensor 3)
  //   op 1 : FULLY_CONNECTED (tensors 3, 1, 2 -> tensor 4)
  //   op 2 : DEQUANTIZE      (tensor 4 -> tensor 5)
  std::vector<flatbuffers::Offset<tflite::Operator>> operators_vec;

  // QUANTIZE
  {
    const std::vector<int32_t> ins = {0};
    const std::vector<int32_t> outs = {3};
    auto ins_fb = fbb.CreateVector(ins);
    auto outs_fb = fbb.CreateVector(outs);
    operators_vec.push_back(tflite::CreateOperator(fbb, /*opcode_index=*/0,
                                                     ins_fb, outs_fb));
  }
  // FULLY_CONNECTED
  {
    const std::vector<int32_t> ins = {3, 1, 2};
    const std::vector<int32_t> outs = {4};
    auto ins_fb = fbb.CreateVector(ins);
    auto outs_fb = fbb.CreateVector(outs);
    auto fc_options = tflite::CreateFullyConnectedOptions(
        fbb, tflite::ActivationFunctionType_NONE,
        tflite::FullyConnectedOptionsWeightsFormat_DEFAULT,
        /*keep_num_dims=*/true,
        /*asymmetric_quantize_inputs=*/false);
    operators_vec.push_back(tflite::CreateOperator(
        fbb, /*opcode_index=*/1, ins_fb, outs_fb,
        tflite::BuiltinOptions_FullyConnectedOptions, fc_options.Union()));
  }
  // DEQUANTIZE
  {
    const std::vector<int32_t> ins = {4};
    const std::vector<int32_t> outs = {5};
    auto ins_fb = fbb.CreateVector(ins);
    auto outs_fb = fbb.CreateVector(outs);
    operators_vec.push_back(tflite::CreateOperator(fbb, /*opcode_index=*/2,
                                                     ins_fb, outs_fb));
  }
  auto operators_fb = fbb.CreateVector(operators_vec);

  // ---- Subgraph ----
  //   inputs  : [0]    (input_fp32 "x")
  //   outputs : [5]    (output_fp32 "y")
  const std::vector<int32_t> sg_inputs_v = {0};
  const std::vector<int32_t> sg_outputs_v = {5};
  auto sg_inputs = fbb.CreateVector(sg_inputs_v);
  auto sg_outputs = fbb.CreateVector(sg_outputs_v);
  auto sg_name = fbb.CreateString("main");
  auto subgraph = tflite::CreateSubGraph(fbb, tensors_fb, sg_inputs,
                                          sg_outputs, operators_fb, sg_name);
  std::vector<flatbuffers::Offset<tflite::SubGraph>> subgraphs_vec = {subgraph};
  auto subgraphs_fb = fbb.CreateVector(subgraphs_vec);

  // ---- Signature ----
  auto sig_key = fbb.CreateString("main");
  auto sig_in_name = fbb.CreateString("x");
  auto sig_out_name = fbb.CreateString("y");
  auto sig_def = BuildSignatureDef(
      fbb, sig_key, sig_in_name, /*in_tensor_idx=*/0, sig_out_name,
      /*out_tensor_idx=*/5);
  std::vector<flatbuffers::Offset<tflite::SignatureDef>> sig_defs_vec = {
      sig_def};
  auto sig_defs_fb = fbb.CreateVector(sig_defs_vec);

  // ---- Top-level Model ----
  auto description = fbb.CreateString(absl::StrCat(
      "matmul_micro_benchmark wrapped int8 fc m=", m, " n=", n, " k=", k));
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

// Builds a single CONV_2D op with a 1x1 filter, NHWC layout. This is
// mathematically a [M, K] @ [N, K]^T matmul but uses
// BuiltinOperator_CONV_2D instead of FULLY_CONNECTED so the LiteRT
// GPU CL delegate goes through its native CONV_2D lowering pipeline
// and hits the `convolution(conv_wave_memory)` entry point (no `1x1`
// suffix) rather than the FC-rewritten `convolution1x1(...)` path.
//
// Tensor layout:
//   input  : [1, M, 1, K]  FLOAT32  (N=1, H=M, W=1, C=K)
//   filter : [N, 1, 1, K]  FLOAT32  (out_channels, kernel_h, kernel_w, in_channels)
//   bias   : [N]           FLOAT32
//   output : [1, M, 1, N]  FLOAT32  (N=1, H=M, W=1, C=N)
//   stride : 1 x 1
//   pad    : VALID (no padding; K must equal C, which it does)
//   fused  : NONE
absl::StatusOr<SingleFcBuildResult> BuildFp32Conv2d1x1(int64_t m, int64_t n,
                                                        int64_t k) {
  // Sanity cap: 1 GiB of fp32 weights, same as the single-op FC path.
  constexpr int64_t kMaxBytes = int64_t{1} << 30;
  const int64_t weight_bytes = n * k * static_cast<int64_t>(sizeof(float));
  if (weight_bytes <= 0 || weight_bytes > kMaxBytes) {
    return absl::InvalidArgumentError(
        absl::StrCat("BuildFp32Conv2d1x1: weight tensor too large (",
                     weight_bytes, " bytes > 1 GiB)"));
  }

  flatbuffers::FlatBufferBuilder fbb(/*initial_size=*/static_cast<size_t>(
      weight_bytes + 64 * 1024));

  // ---- Buffers ----
  auto buffer_empty = tflite::CreateBuffer(fbb);

  const std::vector<uint8_t> weight_bytes_vec(static_cast<size_t>(weight_bytes),
                                                0);
  auto weights_data_vec = fbb.CreateVector(weight_bytes_vec);
  auto buffer_weights = tflite::CreateBuffer(fbb, weights_data_vec);

  const std::vector<uint8_t> bias_bytes_vec(
      static_cast<size_t>(n) * sizeof(float), 0);
  auto bias_data_vec = fbb.CreateVector(bias_bytes_vec);
  auto buffer_bias = tflite::CreateBuffer(fbb, bias_data_vec);

  std::vector<flatbuffers::Offset<tflite::Buffer>> buffers_vec = {
      buffer_empty, buffer_weights, buffer_bias};
  auto buffers_fb = fbb.CreateVector(buffers_vec);

  // ---- Tensors ----
  //   0 : input  [1, M, 1, K] fp32
  //   1 : filter [N, 1, 1, K] fp32 constant
  //   2 : bias   [N]          fp32 constant
  //   3 : output [1, M, 1, N] fp32
  std::vector<flatbuffers::Offset<tflite::Tensor>> tensors_vec;
  // Tensor 0 : input
  {
    const std::vector<int32_t> shape = {1, static_cast<int32_t>(m), 1,
                                          static_cast<int32_t>(k)};
    auto shape_fb = fbb.CreateVector(shape);
    auto name = fbb.CreateString("x");
    tensors_vec.push_back(tflite::CreateTensor(
        fbb, shape_fb, tflite::TensorType_FLOAT32, /*buffer=*/0, name));
  }
  // Tensor 1 : filter
  {
    const std::vector<int32_t> shape = {static_cast<int32_t>(n), 1, 1,
                                          static_cast<int32_t>(k)};
    auto shape_fb = fbb.CreateVector(shape);
    auto name = fbb.CreateString("w");
    tensors_vec.push_back(tflite::CreateTensor(
        fbb, shape_fb, tflite::TensorType_FLOAT32, /*buffer=*/1, name));
  }
  // Tensor 2 : bias
  {
    const std::vector<int32_t> shape = {static_cast<int32_t>(n)};
    auto shape_fb = fbb.CreateVector(shape);
    auto name = fbb.CreateString("b");
    tensors_vec.push_back(tflite::CreateTensor(
        fbb, shape_fb, tflite::TensorType_FLOAT32, /*buffer=*/2, name));
  }
  // Tensor 3 : output
  {
    const std::vector<int32_t> shape = {1, static_cast<int32_t>(m), 1,
                                          static_cast<int32_t>(n)};
    auto shape_fb = fbb.CreateVector(shape);
    auto name = fbb.CreateString("y");
    tensors_vec.push_back(tflite::CreateTensor(
        fbb, shape_fb, tflite::TensorType_FLOAT32, /*buffer=*/0, name));
  }
  auto tensors_fb = fbb.CreateVector(tensors_vec);

  // ---- Operator code: CONV_2D version 5 covers the modern Conv2DOptions
  // (dilation_w_factor / dilation_h_factor) we use below. ----
  std::vector<flatbuffers::Offset<tflite::OperatorCode>> opcodes_vec;
  opcodes_vec.push_back(
      BuildBuiltinOpCode(fbb, tflite::BuiltinOperator_CONV_2D, /*version=*/5));
  auto opcodes_fb = fbb.CreateVector(opcodes_vec);

  // ---- Operator: CONV_2D(input, filter, bias) -> output ----
  const std::vector<int32_t> op_inputs_v = {0, 1, 2};
  const std::vector<int32_t> op_outputs_v = {3};
  auto op_inputs_fb = fbb.CreateVector(op_inputs_v);
  auto op_outputs_fb = fbb.CreateVector(op_outputs_v);
  auto conv_options = tflite::CreateConv2DOptions(
      fbb, tflite::Padding_VALID,
      /*stride_w=*/1, /*stride_h=*/1,
      tflite::ActivationFunctionType_NONE,
      /*dilation_w_factor=*/1, /*dilation_h_factor=*/1);
  auto op = tflite::CreateOperator(
      fbb, /*opcode_index=*/0, op_inputs_fb, op_outputs_fb,
      tflite::BuiltinOptions_Conv2DOptions, conv_options.Union());
  std::vector<flatbuffers::Offset<tflite::Operator>> operators_vec = {op};
  auto operators_fb = fbb.CreateVector(operators_vec);

  // ---- Subgraph + signature + model ----
  const std::vector<int32_t> sg_inputs_v = {0};
  const std::vector<int32_t> sg_outputs_v = {3};
  auto sg_inputs = fbb.CreateVector(sg_inputs_v);
  auto sg_outputs = fbb.CreateVector(sg_outputs_v);
  auto sg_name = fbb.CreateString("main");
  auto subgraph = tflite::CreateSubGraph(fbb, tensors_fb, sg_inputs,
                                          sg_outputs, operators_fb, sg_name);
  std::vector<flatbuffers::Offset<tflite::SubGraph>> subgraphs_vec = {subgraph};
  auto subgraphs_fb = fbb.CreateVector(subgraphs_vec);

  auto sig_key = fbb.CreateString("main");
  auto sig_in_name = fbb.CreateString("x");
  auto sig_out_name = fbb.CreateString("y");
  auto sig_def = BuildSignatureDef(
      fbb, sig_key, sig_in_name, /*in_tensor_idx=*/0, sig_out_name,
      /*out_tensor_idx=*/3);
  std::vector<flatbuffers::Offset<tflite::SignatureDef>> sig_defs_vec = {
      sig_def};
  auto sig_defs_fb = fbb.CreateVector(sig_defs_vec);

  auto description = fbb.CreateString(
      absl::StrCat("matmul_micro_benchmark conv2d_1x1 m=", m, " n=", n,
                   " k=", k));
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

}  // namespace

}  // namespace litert::lm
