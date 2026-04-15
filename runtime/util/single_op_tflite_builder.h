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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_SINGLE_OP_TFLITE_BUILDER_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_SINGLE_OP_TFLITE_BUILDER_H_

#include <cstdint>
#include <string>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl

namespace litert::lm {

// Data type selector for the single-op tflite builder.
//
// kFp32Conv2d:
//   Single CONV_2D op with a 1x1 filter, NHWC layout. Input
//   [1, M, 1, K] FLOAT32, filter [N, 1, 1, K] FLOAT32, bias [N] FLOAT32,
//   output [1, M, 1, N] FLOAT32, stride 1, padding VALID, no fused
//   activation. This is mathematically equivalent to a [M, K] x [K, N]
//   matmul but uses BuiltinOperator_CONV_2D instead of
//   BuiltinOperator_FULLY_CONNECTED. The motivation is that the
//   LiteRT GPU CL delegate lowers FC and CONV_2D through different
//   pipelines: FC is rewritten to `convolution1x1(conv_wave_memory)`,
//   while a native CONV_2D goes to `convolution(conv_wave_memory)`
//   (no `1x1` suffix). Profile data on Adreno 830 shows
//   `convolution(...)` is ~4x faster per dispatch than
//   `convolution1x1(...)` for matmul-shaped inputs, and prefill's
//   `convolution(conv_wave_memory)` row -- which dominates the fp16
//   share of the matmul total -- comes from native CONV_2D ops in
//   Gemma4's .litertlm graph.
//
// kInt8 (default for matching Gemma4 prefill):
//   3-op subgraph: QUANTIZE(fp32 -> int8) -> FULLY_CONNECTED(int8) ->
//   DEQUANTIZE(int8 -> fp32). The signature exposes FLOAT32 input/output
//   so litert::CompiledModel can negotiate buffer requirements correctly
//   (a previous experiment with INT8 signature I/O hit
//   `Failed to get buffer requirements for tensor` warnings inside
//   compiled_model.cc and ran 2x slower than fp32). The internal FC
//   sees INT8 input + INT8 weights + INT32 bias + INT8 output. The
//   delegate fuses the 3 ops but compiles the conv as fp16
//   (`convolution1x1(conv_wave_memory) -> quantize_and_dequantize ->
//   quantize_and_dequantize`); the int8 conv kernel is NOT picked. So
//   this mode is currently equivalent to kFp32 + a bit of QUANTIZE
//   overhead. Kept as a comparison point.
//
// kInt8WeightFp32Act:
//   Hybrid quantization (single FC op). Weights tensor is INT8 with
//   per-output-channel quant, bias FLOAT32, input/output FLOAT32,
//   and FullyConnectedOptions.asymmetric_quantize_inputs=true. The
//   LiteRT GPU delegate accepts this but does NOT pick the
//   convolution_int8(conv_wave_memory) kernel for it -- timings stay
//   close to the fp32 path. Kept as a comparison point for
//   understanding what the delegate accepts vs. what it actually
//   optimizes.
//
// kFp32:
//   FLOAT32 weights and activations. The GPU CL delegate compiles
//   fp16 reduction kernels internally via GpuOptions::SetPrecision(kFp16),
//   so what runs on device is the fp16-weight path -- this matches the
//   smaller `convolution(conv_wave_memory)` rows in prefill (~20% share
//   of the matmul total).
//
// kFp16:
//   FLOAT16 schema. Currently broken because the CPU FC reference
//   kernel asserts `input->type == kTfLiteFloat32` during prepare,
//   and any path that falls back to CPU rejects the model. Kept for
//   forward compatibility.
//
// kInt8PerTensor:
//   Single FULLY_CONNECTED op with INT8 weights using PER-TENSOR
//   symmetric quantization (one scale + one zero_point for the whole
//   weight tensor, NOT one-per-output-channel), FLOAT32 bias, and
//   FLOAT32 signature input/output. This matches the schema the open
//   source TFLite GPU CL delegate's lstm_parser.cc:62 detection
//   uses to construct a `FullyConnectedInt8Attributes` and emit the
//   `OperationType::FULLY_CONNECTED_INT8` node:
//       if (weights_tensor->type == kTfLiteInt8 &&
//           quant_params->scale->size == 1) { ... }
//   The closed-source ml_drift CL accelerator (libLiteRtGpuAccelerator.so)
//   that ships with the prebuilts is a fork of the open source TFLite
//   GPU code and almost certainly uses the same trigger condition,
//   which is the only path that ends up at the
//   `convolution_int8(conv_wave_memory)` kernel name we see in
//   Gemma4 prefill's profile output. Earlier int8 attempts in this
//   builder all used PER-CHANNEL quant (n scales, one per output
//   channel) which the trigger explicitly rejects.
//
// kInt8Chain:
//   4-op subgraph: QUANTIZE -> FC1(int8) -> FC2(int8) -> DEQUANTIZE.
//   Two int8 FullyConnected ops in series share an int8 intermediate
//   tensor that's read by FC2 and written by FC1 -- the GPU CL
//   delegate optimizer can NOT collapse the int8 intermediate to fp16
//   because two real ops depend on it. The hypothesis is that
//   single-op micros (kInt8 above) get compiled as fp16 conv even
//   though the FC tensor types are INT8, because the wrapped
//   QUANTIZE/DEQUANTIZE round-trip leaves no live int8 tensor; in a
//   chain there's a live int8 tensor between the two FCs and the
//   delegate has to commit to int8 GEMM. Kernel name should change
//   from `convolution1x1(conv_wave_memory)` (fp16 1x1 conv) to
//   `convolution_int8(conv_wave_memory) -> ...` (matching prefill's
//   int8 row).
//
//   Shape semantics: FC1 maps [M, K] -> [M, N], FC2 maps [M, N] ->
//   [M, K] (so the sub-graph is a closed loop that emits the same
//   shape it consumed). Both FCs do the same number of MACs as the
//   user-requested matmul, so the chain's wall time is ~2x a single
//   conv. matmul_micro_benchmark divides the reported per-Run
//   samples by 2 in this mode so the CSV stays comparable to
//   single-op modes and to prefill numbers.
enum class MatmulDtype {
  kInt8,
  kInt8WeightFp32Act,
  kFp32,
  kFp16,
  kFp32Conv2d,
  kInt8Chain,
  kInt8PerTensor,
};

// Pure-data result of BuildSingleFullyConnectedTfliteModel. The caller
// owns the flatbuffer bytes; they stay valid until the string is moved
// from or destroyed.
struct SingleFcBuildResult {
  // Raw serialized tflite flatbuffer, ready to feed into
  // litert::Model::CreateFromBuffer.
  std::string flatbuffer;
  // Name of the (single) signature exposed by this model. Use it with
  // CompiledModel::CreateInputBuffers / Run / etc.
  std::string signature_key;
  // Name of the lone input tensor in the signature.
  std::string input_name;
  // Name of the lone output tensor in the signature.
  std::string output_name;
};

// Builds a minimal tflite flatbuffer containing a single FULLY_CONNECTED
// op, shaped for a per-shape microbenchmark run:
//
//   x : [1, m, k]  input activations     (fp16, read at Run time)
//   w : [n, k]     weights               (fp16, baked into the model
//                                          as a constant Buffer)
//   b : [n]        bias                  (fp16 zeros, baked into the model)
//   y : [1, m, n]  output activations    (fp16, written at Run time)
//
// The FullyConnectedOptions flags are:
//   fused_activation_function = NONE
//   weights_format            = DEFAULT
//   keep_num_dims             = true        (preserve the leading [1,m,_]
//                                            dims instead of flattening,
//                                            so output is [1,m,n] not [m,n])
//   asymmetric_quantize_inputs = false
//
// Weight and bias data are filled with a deterministic pseudo-random
// pattern so repeated invocations at the same shape produce identical
// bytes, which keeps the weight cache happy when the LiteRT GPU backend
// re-compiles the same model across benchmark iterations.
//
// Returns an InvalidArgumentError if m/n/k is non-positive or the
// resulting buffer would exceed 1 GiB (sanity check).
absl::StatusOr<SingleFcBuildResult> BuildSingleFullyConnectedTfliteModel(
    int64_t m, int64_t n, int64_t k, MatmulDtype dtype);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_SINGLE_OP_TFLITE_BUILDER_H_
