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

#include "runtime/executor/llm_litert_compiled_model_executor_factory.h"

#include <algorithm>
#include <memory>
#include <string>

#include "absl/algorithm/container.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_model.h"  // from @litert
#include "litert/cc/litert_ranked_tensor_type.h"  // from @litert
#include "runtime/components/model_resources.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/litert_compiled_model_executor_utils.h"
#include "runtime/executor/llm_executor.h"
#include "runtime/executor/llm_executor_settings.h"
#include <cstdio>
#include <cstdlib>
#include <string>

#include "runtime/executor/llm_litert_compiled_model_executor.h"
#include "runtime/util/matmul_shape_dump.h"
#include "runtime/util/status_macros.h"

#if !defined(LITERT_DISABLE_NPU)
#include "runtime/executor/llm_litert_npu_compiled_model_executor.h"
#endif  // !defined(LITERT_DISABLE_NPU)

namespace litert::lm {
namespace {

// The value used by the converter to annotate a dimension as dynamic in models.
constexpr int kDynamicDimValue = -1;

// Given a tensor object, inspects its ranked tensor type and returns true if
// any of its dimensions are dynamic (i.e., have value kDynamicDimValue).
absl::StatusOr<bool> IsDynamicTensor(const SimpleTensor& tensor) {
  LITERT_ASSIGN_OR_RETURN(RankedTensorType ranked_tensor_type,
                          tensor.RankedTensorType());
  auto dimensions = ranked_tensor_type.Layout().Dimensions();
  int n_dynamic_dims =
      std::count(dimensions.begin(), dimensions.end(), kDynamicDimValue);
  return n_dynamic_dims > 0;
}

// Returns true if the model is a dynamic model.
// Model dynamism is determined based on the prefill signature and whether the
// KV cache and sequence length are dynamic.
absl::StatusOr<bool> IsDynamicModel(const Model& model) {
  absl::string_view prefill_signature_key;
  for (int i = 0; i < model.GetNumSignatures(); ++i) {
    LITERT_ASSIGN_OR_RETURN(auto sig, model.GetSignature(i));
    absl::string_view key = sig.Key();
    if (absl::StartsWith(key, "prefill")) {
      prefill_signature_key = key;
      break;
    }
  }
  LITERT_ASSIGN_OR_RETURN(const SimpleSignature& prefill_signature,
                          model.FindSignature(prefill_signature_key));

  bool is_kv_cache_dynamic = false;
  {
    std::string kv_cache_k_root_name;
    std::string kv_cache_v_root_name;
    RETURN_IF_ERROR(GetKVCacheRootNames(
        prefill_signature.InputNames(), prefill_signature.OutputNames(),
        kv_cache_k_root_name, kv_cache_v_root_name));

    // TODO(b/477657050): Investigate support for dynamic model with optimized
    // gpu cache.
    if (!absl::c_any_of(prefill_signature.InputNames(),
                        [&](absl::string_view input_name) {
                          return absl::StartsWith(input_name,
                                                  kv_cache_k_root_name) ||
                                 absl::StartsWith(input_name,
                                                  kv_cache_v_root_name);
                        })) {
      return false;
    }

    std::string first_kv_cache_k_input_name = kv_cache_k_root_name + "0";
    LITERT_ASSIGN_OR_RETURN(
        SimpleTensor k_tensor,
        prefill_signature.InputTensor(first_kv_cache_k_input_name));
    ASSIGN_OR_RETURN(bool is_k_dynamic, IsDynamicTensor(k_tensor));

    std::string first_kv_cache_v_input_name = kv_cache_v_root_name + "0";
    LITERT_ASSIGN_OR_RETURN(
        SimpleTensor v_tensor,
        prefill_signature.InputTensor(first_kv_cache_v_input_name));
    ASSIGN_OR_RETURN(bool is_v_dynamic, IsDynamicTensor(v_tensor));

    RET_CHECK(is_k_dynamic == is_v_dynamic)
        << "KV cache k and v need to be dynamic or static at the same time.";
    is_kv_cache_dynamic = is_k_dynamic && is_v_dynamic;
  }

  bool is_seq_len_dynamic = false;
  {
    ASSIGN_OR_RETURN(ModelSignatures signatures,
                     GetModelSignaturesFromInputOutputNames(
                         prefill_signature.InputNames(),
                         prefill_signature.OutputNames(), /*strict=*/false));
    LITERT_ASSIGN_OR_RETURN(
        SimpleTensor position_tensor,
        prefill_signature.InputTensor(signatures.input_positions));
    ASSIGN_OR_RETURN(is_seq_len_dynamic, IsDynamicTensor(position_tensor));
  }
  RET_CHECK(is_kv_cache_dynamic == is_seq_len_dynamic)
      << "KV cache and seq len need to be dynamic or static at the same time.";

  return is_kv_cache_dynamic;
}

absl::StatusOr<std::unique_ptr<LlmExecutor>>
CreateCpuOrGpuLlmLiteRtCompiledModelExecutor(
    LlmExecutorSettings executor_settings, Environment& lrt_env,
    ModelResources& resources) {
  ASSIGN_OR_RETURN(const litert::Model* litert_model,
                   resources.GetTFLiteModel(ModelType::kTfLitePrefillDecode));

  // When --enable_op_profiling is set we also dump a matmul roster for the
  // prefill/decode model before creating the compiled model. The LiteRT GPU
  // delegate's own per-op profile table reports fused kernel names
  // (e.g. `convolution_int8(conv_wave_memory)`) with no shape info, which
  // makes it hard to identify which source matmul is the bottleneck. The
  // roster here lists every FULLY_CONNECTED / BATCH_MATMUL / 1x1 CONV_2D
  // op in the graph together with its M/N/K, so the reviewer can correlate
  // the two tables when reading temp_litert_run.log. We only walk the
  // flatbuffer when profiling is explicitly requested so normal runs don't
  // pay the cost, and we tolerate failures so a corrupt model never turns
  // a profile run into a crash.
  if (executor_settings.GetAdvancedSettings().has_value() &&
      executor_settings.GetAdvancedSettings()->enable_op_profiling) {
    auto buffer = resources.GetTFLiteModelBuffer(ModelType::kTfLitePrefillDecode);
    if (buffer.ok()) {
      DumpMatmulShapes(*buffer, "prefill-decode", stderr);
    }
    // Optional machine-readable CSV export. Off by default; set the env
    // var LITERT_LM_MATMUL_ROSTER_CSV=/path/to/roster.csv to enable. The
    // CSV has one row per matmul op and can be consumed by any downstream
    // tool (grep, spreadsheet, microbench). Failures are non-fatal
    // (e.g. no write permission on device tmpfs).
    const char* csv_path = std::getenv("LITERT_LM_MATMUL_ROSTER_CSV");
    if (csv_path != nullptr && *csv_path != '\0' && buffer.ok()) {
      auto status = WriteMatmulShapeRosterCsv(*buffer, "prefill-decode",
                                              csv_path);
      if (!status.ok()) {
        const std::string err_msg(status.message());
        std::fprintf(stderr,
                     "[PROFILE MATMUL SHAPES] CSV export to %s failed: %s\n",
                     csv_path, err_msg.c_str());
      } else {
        std::fprintf(stderr,
                     "[PROFILE MATMUL SHAPES] CSV roster written to %s\n",
                     csv_path);
      }
    }
  }

  std::unique_ptr<LlmExecutor> executor;
  ASSIGN_OR_RETURN(bool is_dynamic_model, IsDynamicModel(*litert_model));
  if (is_dynamic_model) {
    ASSIGN_OR_RETURN(executor, LlmLiteRtCompiledModelExecutorDynamic::Create(
                                   executor_settings, lrt_env, resources));
  } else {
    ASSIGN_OR_RETURN(executor, LlmLiteRtCompiledModelExecutorStatic::Create(
                                   executor_settings, lrt_env, resources));
  }

  return executor;
}

absl::StatusOr<std::unique_ptr<LlmExecutor>>
CreateNpuLlmLiteRtCompiledModelExecutor(LlmExecutorSettings executor_settings,
                                        Environment& lrt_env,
                                        ModelResources& resources) {
#if defined(LITERT_DISABLE_NPU)
  return absl::InvalidArgumentError("Only CPU and GPU backends are supported.");
#else
  return LlmLiteRtNpuCompiledModelExecutor::Create(executor_settings, resources,
                                                   lrt_env);
#endif  // defined(LITERT_DISABLE_NPU)
}

}  // namespace

absl::StatusOr<std::unique_ptr<LlmExecutor>>
CreateLlmLiteRtCompiledModelExecutor(LlmExecutorSettings executor_settings,
                                     Environment& lrt_env,
                                     ModelResources& resources) {
  Backend backend = executor_settings.GetBackend();
  switch (backend) {
    case Backend::CPU:
    case Backend::GPU:
      return CreateCpuOrGpuLlmLiteRtCompiledModelExecutor(executor_settings,
                                                          lrt_env, resources);
    case Backend::NPU:
      return CreateNpuLlmLiteRtCompiledModelExecutor(executor_settings, lrt_env,
                                                     resources);
    default:
      return absl::InvalidArgumentError(
          absl::StrCat("Unsupported backend: ", backend));
  }
};

}  // namespace litert::lm
