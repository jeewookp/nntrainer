// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Samsung Electronics Co., Ltd.
 *
 * @file   embedding_layer_cl.cpp
 * @date   10 April 2026
 * @brief  Embedding Layer Class with OpenCL implementation
 * @see    https://github.com/nntrainer/nntrainer
 * @bug    No known bugs except for NYI items
 *
 */

#include <embedding_layer_cl.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>

#include <cl_kernels/embedding.h>
#ifdef ENABLE_FP16
#include <cl_kernels/embedding_fp16.h>
#endif

#include <cpu_backend.h>

namespace nntrainer {

static constexpr size_t SINGLE_INOUT_IDX = 0;

enum EmbeddingParams { weight };

EmbeddingLayerCl::EmbeddingLayerCl() :
  LayerImplCl(),
  embedding_props(props::InDim(), props::OutDim(), props::Scale()),
  weight_idx(std::numeric_limits<unsigned>::max()) {}

bool EmbeddingLayerCl::registerClKernels(ClContext &cl_context) {
  auto &layer_kernel_ptrs = getLayerKernelPtrs();

  if (!layer_kernel_ptrs.empty()) {
    ml_loge("kernels for embedding_layer_cl are already registered.");
    return false;
  }

  do {
    ClContext::SharedPtrClKernel kernel_embedding_ptr =
      cl_context.registerClKernel(embedding_kernel, "embedding_cl");

    if (!kernel_embedding_ptr) {
      ml_loge("OpenCL Error: Fail to register embedding_cl kernel");
      break;
    }
    layer_kernel_ptrs.emplace_back(kernel_embedding_ptr);

#ifdef ENABLE_FP16
    ClContext::SharedPtrClKernel kernel_embedding_fp16_ptr =
      cl_context.registerClKernel(embedding_fp16_kernel, "embedding_cl_fp16");

    if (!kernel_embedding_fp16_ptr) {
      ml_loge("OpenCL Error: Fail to register embedding_cl_fp16 kernel");
      break;
    }
    layer_kernel_ptrs.emplace_back(kernel_embedding_fp16_ptr);
#endif

    return true;
  } while (false);

  layer_kernel_ptrs.clear();
  return false;
}

void EmbeddingLayerCl::finalize(InitLayerContext &context) {
  NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
    << "Embedding layer takes only one input";

  const TensorDim &input_dim =
    context.getInputDimensions()[SINGLE_INOUT_IDX];
  NNTR_THROW_IF(input_dim.channel() != 1, std::invalid_argument)
    << "Embedding layer takes only one for channel size";

  NNTR_THROW_IF(input_dim.getDataType() != TensorDim::DataType::FP32,
                std::invalid_argument)
    << "Embedding layer takes only FP32 input data";

  auto &weight_regularizer =
    std::get<props::WeightRegularizer>(*layer_impl_props);
  auto &weight_regularizer_constant =
    std::get<props::WeightRegularizerConstant>(*layer_impl_props);
  auto weight_initializer = props::InitializerInfo::Enum::NONE;
  auto &weight_decay = std::get<props::WeightDecay>(*layer_impl_props);

  size_t in_dim =
    static_cast<size_t>(std::get<props::InDim>(embedding_props));
  size_t out_dim =
    static_cast<size_t>(std::get<props::OutDim>(embedding_props));

  TensorDim output_dim = input_dim;
  output_dim.height(input_dim.width());
  output_dim.width(out_dim);
  output_dim.setTensorType(
    {context.getFormat(), context.getActivationDataType()});
  context.setOutputDimensions({output_dim});

  TensorDim dim = output_dim;
  dim.setTensorType({context.getFormat(), context.getWeightDataType()});
  dim.height(in_dim);
  dim.width(out_dim);
  dim.batch(1);

  weight_idx = context.requestWeight(
    dim, weight_initializer, weight_regularizer, weight_regularizer_constant,
    weight_decay, "Embedding", true);
}

void EmbeddingLayerCl::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, embedding_props);
  LayerImplCl::setProperty(remain_props);
}

void EmbeddingLayerCl::forwarding(RunLayerContext &context, bool training) {}

void EmbeddingLayerCl::incremental_forwarding(RunLayerContext &context,
                                              unsigned int from,
                                              unsigned int to,
                                              bool training) {
  unsigned int in_dim = std::get<props::InDim>(embedding_props);
  unsigned int out_dim = std::get<props::OutDim>(embedding_props);
  float scale = std::get<props::Scale>(embedding_props).empty()
                  ? 1.0f
                  : std::get<props::Scale>(embedding_props).get();

  Tensor &weight = context.getWeight(weight_idx);
  Tensor &hidden_ = context.getOutput(SINGLE_INOUT_IDX);
  Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);

  if (from) {
    NNTR_THROW_IF(to - from != 1, std::invalid_argument)
      << "incremental step size is not 1";
    from = 0;
    to = 1;
  }

  unsigned int num_tokens = to - from;

  if (weight.getDataType() == ml::train::TensorDim::DataType::FP32) {
    embedding_cl(input_.getData<float>(), weight.getData<float>(),
                 hidden_.getData<float>(), num_tokens, out_dim, scale);
  } else if (weight.getDataType() == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    embedding_cl_fp16(input_.getData<float>(), weight.getData<_FP16>(),
                      hidden_.getData<_FP16>(), num_tokens, out_dim, scale);
#else
    throw std::invalid_argument("Error: enable-fp16 is not enabled");
#endif
  } else {
    /// Q6_K / Q4_0 fallback on CPU
    TensorDim out_tensor_dim =
      TensorDim({1, 1, 1, out_dim}, hidden_.getTensorType());

    for (unsigned int i = 0; i < num_tokens; ++i) {
      size_t embed_idx = static_cast<size_t>(input_.getData<float>()[i]);
      NNTR_THROW_IF(embed_idx >= in_dim, std::invalid_argument)
        << "input word index is greater than in_dim";

      Tensor out_tensor =
        hidden_.getSharedDataTensor(out_tensor_dim, out_dim * i);

      if (weight.getDataType() == TensorDim::DataType::Q6_K) {
        int num_blocks_per_row = (weight.width() + 256 - 1) / 256;
        dequantize_row_q6_K(
          (void *)((char *)weight.getData<uint8_t>() +
                   (210 * num_blocks_per_row) * embed_idx),
          out_tensor.getData(), out_dim);
      } else if (weight.getDataType() == TensorDim::DataType::Q4_0) {
        int num_blocks_per_row = (weight.width() + 32 - 1) / 32;
        dequantize_row_q4_0(
          (void *)((char *)weight.getData<uint8_t>() +
                   (18 * num_blocks_per_row) * embed_idx),
          out_tensor.getData(), out_dim);
      }

      if (scale != 1.0f) {
        out_tensor.multiply_i(scale);
      }
    }
  }
}

void EmbeddingLayerCl::embedding_cl(float *input, float *weight, float *output,
                                    unsigned int num_tokens,
                                    unsigned int out_dim, float scale,
                                    bool svm) {
  auto *global_cl_context =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  auto &clbuffInstance = ClBufferManager::Global();

  do {
    const auto &kernel_ptr = getLayerKernelPtrs()[Kernels::EMBEDDING_CL];
    int total_dim = int(num_tokens * out_dim);
    int od = int(out_dim);

    if (!svm) {
      bool write_result = true;

      write_result &= clbuffInstance.getInBufferA()->WriteDataRegion(
        global_cl_context->command_queue_inst_,
        num_tokens * sizeof(float), input);
      write_result &= clbuffInstance.getInBufferB()->WriteDataRegion(
        global_cl_context->command_queue_inst_,
        total_dim * sizeof(float), weight);
      if (!write_result) {
        break;
      }

      auto bufferInA = clbuffInstance.getInBufferA()->GetBuffer();
      auto bufferInB = clbuffInstance.getInBufferB()->GetBuffer();
      auto bufferOutA = clbuffInstance.getOutBufferA()->GetBuffer();

      bool set_result = true;
      set_result &=
        kernel_ptr->SetKernelArguments(0, &bufferInA, sizeof(cl_mem));
      set_result &=
        kernel_ptr->SetKernelArguments(1, &bufferInB, sizeof(cl_mem));
      set_result &=
        kernel_ptr->SetKernelArguments(2, &bufferOutA, sizeof(cl_mem));
      set_result &=
        kernel_ptr->SetKernelArguments(3, &od, sizeof(int));
      set_result &=
        kernel_ptr->SetKernelArguments(4, &scale, sizeof(float));
      if (!set_result) {
        break;
      }
    } else {
      bool map_result = true;
      map_result &=
        global_cl_context->command_queue_inst_.enqueueSVMUnmap(input);
      map_result &=
        global_cl_context->command_queue_inst_.enqueueSVMUnmap(weight);
      if (!map_result) {
        ml_loge("Failed to unmap svm");
        break;
      }

      bool set_svm_result = true;
      set_svm_result &= kernel_ptr->SetKernelSVMArguments(0, input);
      set_svm_result &= kernel_ptr->SetKernelSVMArguments(1, weight);
      set_svm_result &= kernel_ptr->SetKernelSVMArguments(2, output);
      set_svm_result &=
        kernel_ptr->SetKernelArguments(3, &od, sizeof(int));
      set_svm_result &=
        kernel_ptr->SetKernelArguments(4, &scale, sizeof(float));
      if (!set_svm_result) {
        ml_loge("Failed to set svm");
        break;
      }
    }

    const int32_t desired_local = 64;
    const bool can_use_desired = total_dim >= desired_local;
    const int32_t chosen_local = can_use_desired ? desired_local : total_dim;

    const int work_groups_count[3] = {total_dim, 1, 1};
    const int work_group_size[3] = {chosen_local, 1, 1};

    if (!global_cl_context->command_queue_inst_.DispatchCommand(
          kernel_ptr, work_groups_count, work_group_size)) {
      ml_loge("Failed to run embedding_cl");
      break;
    }

    if (!svm) {
      if (!clbuffInstance.getOutBufferA()->ReadDataRegion(
            global_cl_context->command_queue_inst_,
            total_dim * sizeof(float), output)) {
        break;
      }
    } else {
      if (!global_cl_context->command_queue_inst_.enqueueSVMMap(
            output, total_dim * sizeof(float), true)) {
        ml_loge("Failed to map svm");
        break;
      }
    }

  } while (false);
}

#ifdef ENABLE_FP16
void EmbeddingLayerCl::embedding_cl_fp16(float *input, _FP16 *weight,
                                         _FP16 *output,
                                         unsigned int num_tokens,
                                         unsigned int out_dim, float scale,
                                         bool svm) {
  auto *global_cl_context =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  auto &clbuffInstance = ClBufferManager::Global();

  do {
    const auto &kernel_ptr = getLayerKernelPtrs()[Kernels::EMBEDDING_CL_FP16];
    int total_dim = int(num_tokens * out_dim);
    int od = int(out_dim);

    bool write_result = true;

    write_result &= clbuffInstance.getInBufferA()->WriteDataRegion(
      global_cl_context->command_queue_inst_,
      num_tokens * sizeof(float), input);
    write_result &= clbuffInstance.getInBufferB()->WriteDataRegion(
      global_cl_context->command_queue_inst_,
      total_dim * sizeof(_FP16), weight);
    if (!write_result) {
      break;
    }

    auto bufferInA = clbuffInstance.getInBufferA()->GetBuffer();
    auto bufferInB = clbuffInstance.getInBufferB()->GetBuffer();
    auto bufferOutA = clbuffInstance.getOutBufferA()->GetBuffer();

    bool set_result = true;
    set_result &=
      kernel_ptr->SetKernelArguments(0, &bufferInA, sizeof(cl_mem));
    set_result &=
      kernel_ptr->SetKernelArguments(1, &bufferInB, sizeof(cl_mem));
    set_result &=
      kernel_ptr->SetKernelArguments(2, &bufferOutA, sizeof(cl_mem));
    set_result &=
      kernel_ptr->SetKernelArguments(3, &od, sizeof(int));
    set_result &=
      kernel_ptr->SetKernelArguments(4, &scale, sizeof(float));
    if (!set_result) {
      break;
    }

    const int32_t desired_local = 64;
    const bool can_use_desired = total_dim >= desired_local;
    const int32_t chosen_local = can_use_desired ? desired_local : total_dim;

    const int work_groups_count[3] = {total_dim, 1, 1};
    const int work_group_size[3] = {chosen_local, 1, 1};

    if (!global_cl_context->command_queue_inst_.DispatchCommand(
          kernel_ptr, work_groups_count, work_group_size)) {
      ml_loge("Failed to run embedding_cl_fp16");
      break;
    }

    if (!clbuffInstance.getOutBufferA()->ReadDataRegion(
          global_cl_context->command_queue_inst_,
          total_dim * sizeof(_FP16), output)) {
      break;
    }

  } while (false);
}
#endif

void EmbeddingLayerCl::calcDerivative(RunLayerContext &context) {
  std::throw_with_nested(
    std::runtime_error("calcDerivative for Embedding layer is not supported"));
}

void EmbeddingLayerCl::exportTo(
  Exporter &exporter, const ml::train::ExportMethods &method) const {
  LayerImplCl::exportTo(exporter, method);
  exporter.saveResult(embedding_props, method, this);
}

void EmbeddingLayerCl::save(std::ofstream &file, RunLayerContext &run_context,
                            bool opt_var, ml::train::ExecutionMode mode,
                            bool trainable, TensorDim::DataType dtype) const {
  for (unsigned int i = 0; i < run_context.getNumWeights(); ++i) {
    if (run_context.isGradientFirstAccess(i)) {
      auto &w = run_context.getWeight(i);
      if (dtype == TensorDim::DataType::NONE || w.getDataType() == dtype) {
        w.save(file);
      } else {
        NNTR_THROW_IF(w.getDataType() != TensorDim::DataType::FP32,
                      std::runtime_error)
          << "Save with quantization only supports for FP32 weight.";

        TensorDim dim = w.getDim();
        unsigned int K = dim.height();
        unsigned int N = dim.width();

        if (dtype == TensorDim::DataType::Q4_0) {
          if (K == 1) {
            w.save(file);
          } else {
            NNTR_THROW_IF(N % 32 != 0 || K % 32 != 0, std::invalid_argument)
              << "Q4_0 quantization requires both width and height to be "
                 "divisible by 32, but got height="
              << K << ", width=" << N;
            Tensor quant_weight(dim.batch(), dim.channel(), K, N,
                                {Tformat::NCHW, dtype});
            quantize_q4_0(w.getData<float>(), quant_weight.getData<uint8_t>(),
                          K, N, nullptr);
            quant_weight.save(file);
          }
        } else if (dtype == TensorDim::DataType::Q6_K) {
          Tensor quant_weight(dim.batch(), dim.channel(), K, N,
                              {Tformat::NCHW, dtype});
          quantize_q6_K(w.getData<float>(), quant_weight.getData<uint8_t>(), K,
                        N, nullptr);
          quant_weight.save(file);
        } else {
          NNTR_THROW_IF(true, std::runtime_error)
            << "This dtype is not supported in save with quantization";
        }
      }
    }
  }
}

std::vector<ClContext::SharedPtrClKernel> &
EmbeddingLayerCl::getLayerKernelPtrs() {
  static std::vector<ClContext::SharedPtrClKernel> layer_kernel_ptrs;
  return layer_kernel_ptrs;
}

} // namespace nntrainer
