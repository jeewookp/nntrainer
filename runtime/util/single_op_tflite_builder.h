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
// kInt8WeightFp32Act:
//   Hybrid quantization: weights tensor is INT8 with per-output-channel
//   symmetric quantization (scale[N], zero_point all 0,
//   quantized_dimension=0), bias is FLOAT32, input/output are FLOAT32,
//   and FullyConnectedOptions.asymmetric_quantize_inputs=true. This is
//   the exact pattern Gemma4 prefill uses: the LiteRT CL delegate
//   recognizes the hybrid quant FC and lowers it to its
//   `convolution_int8(conv_wave_memory)` kernel, which is what the
//   prefill profile shows as the dominant matmul cost (~80% of the
//   matmul total). Use this mode for apples-to-apples comparison
//   against prefill numbers.
//
// kFp32:
//   FLOAT32 weights and activations. The GPU CL delegate compiles
//   fp16 reduction kernels internally via GpuOptions::SetPrecision(kFp16),
//   so what runs on device is the fp16-weight path -- this matches the
//   `convolution(conv_wave_memory)` rows in prefill (the smaller, ~20%
//   share of the matmul total). Useful as a baseline / upper bound and
//   for sanity-checking the schema before touching int8 quantization.
//
// kFp16:
//   The model's tensors are FLOAT16. This is broken at the moment
//   because the CPU FC reference kernel asserts
//   `input->type != kTfLiteFloat32`, and any time the GPU CL delegate
//   doesn't claim the FC op the interpreter falls back to the CPU
//   FC kernel which immediately fails to prepare. Kept as an enum
//   value for forward compatibility with future LiteRT builds that
//   add a fp16 CPU FC kernel.
enum class MatmulDtype {
  kInt8WeightFp32Act,
  kFp32,
  kFp16,
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
