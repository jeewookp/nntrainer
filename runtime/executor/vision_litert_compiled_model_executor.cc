// Copyright 2025 The ODML Authors.
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

#include "runtime/executor/vision_litert_compiled_model_executor.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>  //NOLINT
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_common.h"  // from @litert
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/cc/litert_ranked_tensor_type.h"  // from @litert
#include "litert/cc/litert_tensor_buffer_types.h"  // from @litert
#include "runtime/engine/io_types.h"
#include "runtime/executor/vision_executor_utils.h"
#include "runtime/util/scoped_file.h"
#if !defined(LITERT_DISABLE_NPU)
#include "litert/cc/options/litert_qualcomm_options.h"  // from @litert
#endif  // !defined(LITERT_DISABLE_NPU)
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_model.h"  // from @litert
#include "litert/cc/litert_options.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "litert/cc/options/litert_cpu_options.h"  // from @litert
#include "litert/cc/options/litert_gpu_options.h"  // from @litert
#include "litert/cc/options/litert_runtime_options.h"  // from @litert
#include "runtime/components/model_resources.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/litert_compiled_model_executor_utils.h"
#include "runtime/executor/llm_executor_io_types.h"
#include "runtime/executor/vision_executor_settings.h"
#include "runtime/util/convert_tensor_buffer.h"
#include "runtime/util/file_util.h"
#include "runtime/util/status_macros.h"  // NOLINT

namespace litert::lm {

namespace {

// The position input tensor name for ViT encoder.
constexpr absl::string_view kPositionsXy = "positions_xy";
// The image patch input tensor name for ViT encoder.
constexpr absl::string_view kImages = "images";
// The features output tensor name for ViT encoder.
constexpr absl::string_view kFeatures = "features";
// The mask input tensor name for ViT encoder.
constexpr absl::string_view kMask = "mask";

absl::Status SetCpuCacheOptions(
    const absl::StatusOr<std::string>& weight_cache_file,
    std::shared_ptr<litert::lm::ScopedFile> scoped_cache_file,
    litert::CpuOptions& cpu_options, absl::string_view logging_prefix) {
  if (scoped_cache_file != nullptr) {
    ASSIGN_OR_RETURN(auto duplicated, scoped_cache_file->Duplicate());
    ASSIGN_OR_RETURN(int fd, duplicated.Release());
    cpu_options.SetXNNPackWeightCacheFileDescriptor(fd);
    ABSL_LOG(INFO) << logging_prefix
                   << " use provided cache file descriptor: " << fd;
  } else if (weight_cache_file.ok()) {
    const std::string& weight_cache_path = *weight_cache_file;
    cpu_options.SetXNNPackWeightCachePath(weight_cache_path.c_str());
    ABSL_LOG(INFO) << logging_prefix
                   << " use cache path: " << weight_cache_path;
  } else {
    ABSL_LOG(INFO) << logging_prefix << " does not use cache.";
  }
  return absl::OkStatus();
}

absl::Status SetGpuOptions(
    const std::string& weight_cache_path,
    std::shared_ptr<litert::lm::ScopedFile> scoped_cache_file,
    const absl::StatusOr<
        std::variant<std::string, std::shared_ptr<litert::lm::ScopedFile>>>&
        program_cache_file,
    const VisionExecutorSettings& executor_settings,
    absl::string_view cache_key, absl::string_view logging_prefix,
    litert::GpuOptions& gpu_options) {
#if defined(LITERT_USE_WEBGPU_ACCELERATOR)
  gpu_options.SetBackend(GpuOptions::Backend::kWebGpu);
#endif  // defined(LITERT_USE_WEBGPU_ACCELERATOR)
  gpu_options.EnableConstantTensorSharing(true);
  // TODO(b/484646529): Re-enable precision setting once the GPU vision
  // encoder precision is fixed.
  // if (activation_data_type == ActivationDataType::FLOAT32) {
  //   gpu_options.SetPrecision(GpuOptions::Precision::kFp32);
  // } else {
  //   gpu_options.SetPrecision(GpuOptions::Precision::kFp16);
  // }
  gpu_options.SetPrecision(GpuOptions::Precision::kFp32);
#if defined(__APPLE__)
  gpu_options.SetPreferTextureWeights(false);
  gpu_options.SetUseMetalArgumentBuffers(true);
#else   // !__APPLE__
  gpu_options.SetPreferTextureWeights(true);
#endif  // !__APPLE__
  gpu_options.SetModelCacheKey(cache_key.data());
  std::string cache_path = weight_cache_path;
  bool serialization_dir_set = false;
  if (cache_path != ":nocache") {
    if (cache_path.empty()) {
      ASSIGN_OR_RETURN(auto model_path,
                       executor_settings.GetModelAssets().GetPath());
      cache_path =
          std::filesystem::path(std::string(model_path)).parent_path().string();
      if (cache_path.empty()) {
        cache_path = std::filesystem::current_path().string();
      }
    }
    gpu_options.SetSerializationDir(cache_path.c_str());
    gpu_options.SetSerializeExternalTensors(true);
    serialization_dir_set = true;
  }
  if (program_cache_file.ok()) {
    if (std::holds_alternative<std::string>(*program_cache_file)) {
      if (!serialization_dir_set) {
        cache_path =
            std::filesystem::path(std::get<std::string>(*program_cache_file))
                .parent_path()
                .string();
        gpu_options.SetSerializationDir(cache_path.c_str());
      }
    } else {
      auto scoped_cache_file =
          std::get<std::shared_ptr<lm::ScopedFile>>(*program_cache_file);
      ASSIGN_OR_RETURN(auto duplicated, scoped_cache_file->Duplicate());
      ASSIGN_OR_RETURN(int fd, duplicated.Release());
      gpu_options.SetProgramCacheFd(fd);
    }
    gpu_options.SetSerializeProgramCache(true);
  } else {
    gpu_options.SetSerializeProgramCache(false);
  }
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<
    std::unique_ptr<VisionLiteRtCompiledModelExecutor::VisionEncoder>>
VisionLiteRtCompiledModelExecutor::VisionEncoder::Create(
    Environment& env, const Model* absl_nonnull model,
    const VisionExecutorSettings& vision_executor_settings) {
  auto handler = std::unique_ptr<VisionEncoder>(
      new VisionEncoder(env, model, vision_executor_settings));
  RETURN_IF_ERROR(handler->Initialize());
  return handler;
}

absl::Status VisionLiteRtCompiledModelExecutor::VisionEncoder::Initialize() {
  // TODO(b/405424188): - Add support for NPU backends.
  LITERT_ASSIGN_OR_RETURN(auto options, Options::Create());
  auto weight_cache_file = vision_executor_settings_.GetWeightCacheFile(
      ".vision_encoder.xnnpack_cache");
  std::string weight_cache_path = vision_executor_settings_.GetCacheDir();
  auto activation_data_type = ActivationDataType::FLOAT16;
  if (vision_executor_settings_.GetActivationDataType().has_value()) {
    activation_data_type =
        vision_executor_settings_.GetActivationDataType().value();
  }
  switch (backend_) {
    case Backend::CPU: {
      // TODO: b/403132820 - Add accelerator compilation options for XNNPACK.
      LITERT_ASSIGN_OR_RETURN(auto& cpu_options, options.GetCpuOptions());
      // Set the number of threads to 4 by default.
      cpu_options.SetNumThreads(4);
      std::shared_ptr<ScopedFile> scoped_encoder_cache_file =
          vision_executor_settings_.GetScopedEncoderCacheFile();
      RETURN_IF_ERROR(SetCpuCacheOptions(weight_cache_file,
                                         scoped_encoder_cache_file, cpu_options,
                                         "vision_encoder"));
      options.SetHardwareAccelerators(litert::HwAccelerators::kCpu);
      break;
    }
    case Backend::GPU: {
      // TODO: b/403132820 - Add accelerator compilation options for ML_DRIFT.
      LITERT_ASSIGN_OR_RETURN(auto& gpu_options, options.GetGpuOptions());
      ASSIGN_OR_RETURN(auto model_path,
                       vision_executor_settings_.GetModelAssets().GetPath());
      absl::string_view model_basename = Basename(model_path);
      auto program_cache_file = vision_executor_settings_.GetProgramCacheFile(
          ".mldrift_program_cache.vision_encoder.bin");
      RETURN_IF_ERROR(
          SetGpuOptions(weight_cache_path,
                        vision_executor_settings_.GetScopedEncoderCacheFile(),
                        program_cache_file, vision_executor_settings_,
                        absl::StrCat(model_basename, ".vision_encoder"),
                        "vision_encoder", gpu_options));
      options.SetHardwareAccelerators(litert::HwAccelerators::kGpu);
      break;
    }
#if !defined(LITERT_DISABLE_NPU)
    case Backend::NPU: {
      LITERT_ASSIGN_OR_RETURN(auto& qualcomm_options,
                              options.GetQualcommOptions());
      qualcomm_options.SetLogLevel(qualcomm::QualcommOptions::LogLevel::kOff);
      qualcomm_options.SetHtpPerformanceMode(
          qualcomm::QualcommOptions::HtpPerformanceMode::kBurst);
      // TODO: yunandrew - Add support for other NPU backends.
      options.SetHardwareAccelerators(litert::HwAccelerators::kCpu);
      break;
    }
#endif  // !defined(LITERT_DISABLE_NPU)
    default:
      return absl::InvalidArgumentError(
          absl::StrCat("Unsupported encoder backend: ", backend_));
  }

  LITERT_ASSIGN_OR_RETURN(compiled_model_,
                          CompiledModel::Create(env_, model_.Get(), options));
  if (auto num_signatures = model_.GetNumSignatures(); num_signatures != 1) {
    return absl::InvalidArgumentError(absl::StrCat(
        "The Vision Encoder model must have exactly one signature but got ",
        num_signatures));
  }
  LITERT_ASSIGN_OR_RETURN(input_buffers_, compiled_model_.CreateInputBuffers(
                                              /*signature_index=*/0));
  LITERT_ASSIGN_OR_RETURN(output_buffers_, compiled_model_.CreateOutputBuffers(
                                               /*signature_index=*/0));

  return absl::OkStatus();
}

absl::StatusOr<
    std::unique_ptr<VisionLiteRtCompiledModelExecutor::VisionAdapter>>
VisionLiteRtCompiledModelExecutor::VisionAdapter::Create(
    Environment& env, const Model* absl_nonnull model,
    const VisionExecutorSettings& vision_executor_settings) {
  auto handler = std::unique_ptr<VisionAdapter>(
      new VisionAdapter(env, model, vision_executor_settings));
  RETURN_IF_ERROR(handler->Initialize());
  return handler;
}

absl::Status VisionLiteRtCompiledModelExecutor::VisionAdapter::Initialize() {
  // TODO(b/405424188): - Add support for NPU backends.
  LITERT_ASSIGN_OR_RETURN(auto options, Options::Create());
  auto weight_cache_file = vision_executor_settings_.GetWeightCacheFile(
      ".vision_adapter.xnnpack_cache");
  std::string weight_cache_path = vision_executor_settings_.GetCacheDir();
  switch (backend_) {
    case Backend::CPU: {
      // TODO: b/403132820 - Add accelerator compilation options for XNNPACK.
      LITERT_ASSIGN_OR_RETURN(auto& cpu_options, options.GetCpuOptions());
      // Set the number of threads to 4 by default.
      cpu_options.SetNumThreads(4);
      std::shared_ptr<ScopedFile> scoped_adapter_cache_file =
          vision_executor_settings_.GetScopedAdapterCacheFile();
      RETURN_IF_ERROR(SetCpuCacheOptions(weight_cache_file,
                                         scoped_adapter_cache_file, cpu_options,
                                         "vision_adapter"));
      options.SetHardwareAccelerators(litert::HwAccelerators::kCpu);
      break;
    }
    case Backend::GPU: {
      // TODO: b/403132820 - Add accelerator compilation options for ML_DRIFT.
      LITERT_ASSIGN_OR_RETURN(auto& gpu_options, options.GetGpuOptions());
      gpu_options.EnableConstantTensorSharing(true);
      gpu_options.EnableAllowSrcQuantizedFcConvOps(true);

      gpu_options.SetPrecision(GpuOptions::Precision::kFp16);
      gpu_options.SetPreferTextureWeights(true);
      options.SetHardwareAccelerators(litert::HwAccelerators::kGpu);
      break;
    }
    default:
      return absl::InvalidArgumentError(
          absl::StrCat("Unsupported adapter backend: ", backend_));
  }

  LITERT_ASSIGN_OR_RETURN(compiled_model_,
                          CompiledModel::Create(env_, model_.Get(), options));
  if (auto num_signatures = model_.GetNumSignatures(); num_signatures != 1) {
    return absl::InvalidArgumentError(absl::StrCat(
        "The Vision Adapter model must have exactly one signature but got ",
        num_signatures));
  }
  LITERT_ASSIGN_OR_RETURN(input_buffers_, compiled_model_.CreateInputBuffers(
                                              /*signature_index=*/0));
  if (input_buffers_.size() != 1) {
    return absl::InvalidArgumentError(
        absl::StrCat("The Vision Adapter model must have exactly one input "
                     "buffer but got ",
                     input_buffers_.size()));
  }

  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<VisionLiteRtCompiledModelExecutor>>
litert::lm::VisionLiteRtCompiledModelExecutor::Create(
    const VisionExecutorSettings& vision_executor_settings, Environment& env) {
  LITERT_ASSIGN_OR_RETURN(auto resources,
                          BuildLiteRtCompiledModelResources(
                              vision_executor_settings.GetModelAssets()));

  ASSIGN_OR_RETURN(auto vision_encoder_model,
                   resources->GetTFLiteModel(ModelType::kTfLiteVisionEncoder));
  if (!vision_encoder_model) {
    return absl::InternalError("Failed to build LiteRt encoder model.");
  }
  ASSIGN_OR_RETURN(auto vision_adapter_model,
                   resources->GetTFLiteModel(ModelType::kTfLiteVisionAdapter));
  if (!vision_adapter_model) {
    return absl::InternalError("Failed to build LiteRt adapter model.");
  }

  ASSIGN_OR_RETURN(auto vision_encoder,
                   VisionEncoder::Create(env, vision_encoder_model,
                                         vision_executor_settings));

  ASSIGN_OR_RETURN(auto vision_adapter,
                   VisionAdapter::Create(env, vision_adapter_model,
                                         vision_executor_settings));

  LITERT_ASSIGN_OR_RETURN(auto tensor_type,
                          vision_encoder_model->GetInputTensorType(0, 0));
  const auto& dimensions = tensor_type.Layout().Dimensions();
  if (dimensions.size() == 4) {
    if (dimensions[3] < 3 || dimensions[3] > 4) {
      return absl::FailedPreconditionError(
          absl::StrCat("Expected encoder input tensor to have 3 or 4 channels",
                       " but got ", dimensions[3]));
    }
  } else if (dimensions.size() != 3) {
    return absl::FailedPreconditionError(absl::StrCat(
        "Expected encoder input tensor to have 3 or 4 dimensions, but got ",
        dimensions.size()));
  }
  auto expected_input_dimension =
      std::vector<int>(dimensions.begin(), dimensions.end());

  ASSIGN_OR_RETURN(
      auto vision_executor_properties,
      GetVisionExecutorPropertiesFromModelResources(*resources.get()));

  return absl::WrapUnique(new VisionLiteRtCompiledModelExecutor(
      vision_executor_settings, env, std::move(resources),
      std::move(vision_encoder), std::move(vision_adapter),
      expected_input_dimension, vision_executor_properties));
}

absl::StatusOr<ExecutorVisionData> VisionLiteRtCompiledModelExecutor::Encode(
    const litert::TensorBuffer& input_image_tensor) {
  LITERT_ASSIGN_OR_RETURN(
      auto output_tensor_buffers,
      vision_adapter_->GetCompiledModel().CreateOutputBuffers(
          /*signature_index=*/0));
  if (output_tensor_buffers.size() != 1) {
    return absl::InternalError(
        absl::StrCat("The Vision Adapter model must have exactly one output "
                     "buffer but got ",
                     output_tensor_buffers.size()));
  }

  LITERT_ASSIGN_OR_RETURN(auto input_image_data,
                          ReferTensorBufferAsSpan<float>(input_image_tensor));
  LITERT_RETURN_IF_ERROR(
      vision_encoder_->GetMutableInputBuffers()[0].Write<float>(
          input_image_data));
  auto& encoder_outputs = vision_encoder_->GetMutableOutputBuffers();
  if (encoder_outputs[0].IsWebGpuMemory() ||
      encoder_outputs[0].IsMetalMemory()) {
    // For WebGPU and Metal memory, we need to create a new output buffer to
    // hold the data, otherwise we will get failed to lock TensorBuffer error on
    // the second call to `Encode`. See b/457483190
    LITERT_ASSIGN_OR_RETURN(
        encoder_outputs,
        vision_encoder_->GetCompiledModel().CreateOutputBuffers(
            /*signature_index=*/0));
  }

  LITERT_RETURN_IF_ERROR(vision_encoder_->GetCompiledModel().Run(
      /*input_buffers=*/vision_encoder_->GetInputBuffers(),
      /*output_buffers=*/encoder_outputs));

  LITERT_RETURN_IF_ERROR(vision_adapter_->GetCompiledModel().Run(
      /*input_buffers=*/encoder_outputs,
      /*output_buffers=*/output_tensor_buffers));

  return ExecutorVisionData(std::move(output_tensor_buffers[0]),
                            /*per_layer_embeddings=*/std::nullopt);
}

absl::StatusOr<std::vector<int>>
VisionLiteRtCompiledModelExecutor::GetExpectedInputDimension() const {
  return expected_input_dimension_;
}

absl::StatusOr<ExecutorVisionData> VisionLiteRtCompiledModelExecutor::Encode(
    const absl::flat_hash_map<std::string, litert::TensorBuffer>& input_maps) {

  if (!input_maps.contains(kPositionsXy)) {
    return absl::InvalidArgumentError(
        absl::StrCat(kPositionsXy, " is not found in the input maps."));
  }
  if (!input_maps.contains(kImages)) {
    return absl::InvalidArgumentError(
        absl::StrCat(kImages, " is not found in the input maps."));
  }

  LITERT_ASSIGN_OR_RETURN(
      auto adapter_output_tensor_buffers,
      vision_adapter_->GetCompiledModel().CreateOutputBuffers(
          /*signature_index=*/0));
  if (adapter_output_tensor_buffers.size() != 1) {
    return absl::InternalError(
        absl::StrCat("The Vision Adapter model must have exactly one output "
                     "buffer but got ",
                     adapter_output_tensor_buffers.size()));
  }

  auto& input_buffers = vision_encoder_->GetMutableInputBuffers();

  absl::flat_hash_map<absl::string_view, litert::TensorBuffer>
      encoder_input_maps;
  for (const auto& [key, value] : input_maps) {
    LITERT_ASSIGN_OR_RETURN(auto tensor_type, value.TensorType());
    LITERT_ASSIGN_OR_RETURN(auto input_index,
                            vision_encoder_->GetCompiledModel().FindInputIndex(
                                /*signature_index=*/0, key));
    input_buffers[input_index].Clear();
    if (tensor_type.ElementType() == ElementType::Float32) {
      LITERT_ASSIGN_OR_RETURN(auto input_data,
                              ReferTensorBufferAsSpan<float>(value));
      LITERT_RETURN_IF_ERROR(
          input_buffers[input_index].Write<float>(input_data));

      // Force clearing padding beyond the written data
      LITERT_ASSIGN_OR_RETURN(auto lock_and_addr,
                              litert::TensorBufferScopedLock::Create<float>(
                                  input_buffers[input_index],
                                  litert::TensorBuffer::LockMode::kWrite));
      LITERT_ASSIGN_OR_RETURN(size_t packed_size,
                              input_buffers[input_index].PackedSize());
      size_t written_size = input_data.size() * sizeof(float);
      if (packed_size > written_size) {
        std::memset(reinterpret_cast<uint8_t*>(lock_and_addr.second) +
                        written_size,
                    0, packed_size - written_size);
      }
    } else if (tensor_type.ElementType() == ElementType::Int32) {
      // Initialize the position buffer to -1 since the input image tensor
      // might have different size as the encoder input tensor.
      LITERT_ASSIGN_OR_RETURN(auto encoder_tensor_type,
                              input_buffers[input_index].TensorType());
      LITERT_ASSIGN_OR_RETURN(auto position_num_elements,
                              encoder_tensor_type.Layout().NumElements());
      std::vector<int32_t> encoder_input_positions(position_num_elements, -1);
      LITERT_RETURN_IF_ERROR(
          input_buffers[input_index].Write<int32_t>(encoder_input_positions));
      LITERT_ASSIGN_OR_RETURN(auto input_data,
                              ReferTensorBufferAsSpan<int32_t>(value));
      LITERT_RETURN_IF_ERROR(
          input_buffers[input_index].Write<int32_t>(input_data));
    } else {
      return absl::InvalidArgumentError("Unsupported input tensor type");
    }
  }

  auto& encoder_output_buffers = vision_encoder_->GetMutableOutputBuffers();
  if (encoder_output_buffers[0].IsWebGpuMemory() ||
      encoder_output_buffers[0].IsMetalMemory()) {
    // For WebGPU and Metal memory, we need to create a new output buffer to
    // hold the data, otherwise we will get failed to lock TensorBuffer error on
    // the second call to `Encode`. See b/457483190
    LITERT_ASSIGN_OR_RETURN(
        encoder_output_buffers,
        vision_encoder_->GetCompiledModel().CreateOutputBuffers(
            /*signature_index=*/0));
  }
  LITERT_RETURN_IF_ERROR(vision_encoder_->GetCompiledModel().Run(
      input_buffers, encoder_output_buffers));

  int num_patches = 0;
  auto mask_index = vision_encoder_->GetCompiledModel().FindOutputIndex(
      /*signature_index=*/0, kMask);

  if (!mask_index.HasValue()) {
    // If the mask is not in the encoder output, we need to estimate the number
    // of patches from the input image tensor.
    if (!vision_executor_properties_.patch_num_shrink_factor.has_value()) {
      return absl::InternalError(
          "patch_num_shrink_factor is not set in the vision executor "
          "properties.");
    }
    LITERT_ASSIGN_OR_RETURN(auto positions_tensor_type,
                            input_maps.at(kPositionsXy).TensorType());
    const int& num_patches_from_input =
        positions_tensor_type.Layout().Dimensions()[1];
    const int& patch_num_shrink_factor =
        vision_executor_properties_.patch_num_shrink_factor.value();
    // Round up the number of patches so we have at least one patch.
    num_patches = (num_patches_from_input + patch_num_shrink_factor - 1) /
                  patch_num_shrink_factor;
  } else {
    LITERT_ASSIGN_OR_RETURN(
        auto mask_tensor_type,
        encoder_output_buffers[mask_index.Value()].TensorType());
    LITERT_ASSIGN_OR_RETURN(int mask_num_elements,
                            mask_tensor_type.Layout().NumElements());
    std::vector<uint8_t> encoder_output_mask(mask_num_elements, 0);
    LITERT_RETURN_IF_ERROR(
        encoder_output_buffers[mask_index.Value()].Read<uint8_t>(
            absl::MakeSpan(encoder_output_mask)));
    num_patches = std::count(encoder_output_mask.begin(),
                             encoder_output_mask.end(), true);
  }

  LITERT_ASSIGN_OR_RETURN(auto features_index,
                          vision_encoder_->GetCompiledModel().FindOutputIndex(
                              /*signature_index=*/0, kFeatures));
  LITERT_ASSIGN_OR_RETURN(auto encoder_output_tensor_type,
                          encoder_output_buffers[features_index].TensorType());
  const int& encoder_output_dim =
      encoder_output_tensor_type.Layout().Dimensions()
          [encoder_output_tensor_type.Layout().Dimensions().size() - 1];
  LITERT_ASSIGN_OR_RETURN(int encoder_output_num_elements,
                          encoder_output_tensor_type.Layout().NumElements());
  std::vector<float> encoder_output_data(encoder_output_num_elements);
  LITERT_RETURN_IF_ERROR(encoder_output_buffers[features_index].Read<float>(
      absl::MakeSpan(encoder_output_data)));

  auto& adapter_input_buffers = vision_adapter_->GetMutableInputBuffers();
  adapter_input_buffers[0].Clear();
  LITERT_RETURN_IF_ERROR(adapter_input_buffers[0].Write<float>(absl::MakeSpan(
      encoder_output_data.data(), num_patches * encoder_output_dim)));

  LITERT_RETURN_IF_ERROR(vision_adapter_->GetCompiledModel().Run(
      /*input_buffers=*/adapter_input_buffers,
      /*output_buffers=*/adapter_output_tensor_buffers));

  // Create the final output tensor with the correct number of patches.
  LITERT_ASSIGN_OR_RETURN(auto adapter_output_tensor_type,
                          adapter_output_tensor_buffers[0].TensorType());
  RankedTensorType output_tensor_type(
      GetElementType<float>(),
      Layout(
          Dimensions({1, num_patches,
                      adapter_output_tensor_type.Layout().Dimensions()[2]})));
  LITERT_ASSIGN_OR_RETURN(
      auto output_tensor,
      TensorBuffer::CreateManaged(
          env_, TensorBufferType::kHostMemory, output_tensor_type,
          output_tensor_type.Layout().Dimensions()[1] *
              output_tensor_type.Layout().Dimensions()[2] * sizeof(float)));
  LITERT_ASSIGN_OR_RETURN(
      auto adapter_output_data,
      ReferTensorBufferAsSpan<float>(adapter_output_tensor_buffers[0]));

  LITERT_RETURN_IF_ERROR(output_tensor.Write<float>(adapter_output_data.subspan(
      0, num_patches * output_tensor_type.Layout().Dimensions()[2])));
  return ExecutorVisionData(std::move(output_tensor),
                            /*per_layer_embeddings=*/std::nullopt);
}

absl::StatusOr<VisionExecutorProperties>
VisionLiteRtCompiledModelExecutor::GetVisionExecutorProperties() const {
  return vision_executor_properties_;
}

}  // namespace litert::lm
