// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Samsung Electronics Co., Ltd.
 *
 * @file   tie_word_embedding_cl.cpp
 * @date   10 April 2026
 * @brief  Tie Word Embedding Layer Class with OpenCL implementation
 * @see    https://github.com/nntrainer/nntrainer
 * @bug    No known bugs except for NYI items
 *
 */

#include <tie_word_embedding_cl.h>

#include <blas_kernel_interface.h>
#include <cpu_backend.h>
#include <layer_context.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <tensor.h>
#include <tensor_dim.h>

#include <cl_kernels/embedding.h>
#ifdef ENABLE_FP16
#include <cl_kernels/embedding_fp16.h>
#endif

namespace nntrainer {

static constexpr size_t SINGLE_INOUT_IDX = 0;

enum TieWordEmbeddingParams {
  weight,
  bias,
  candidate_weight,
  candidate_hidden_step
};

TieWordEmbeddingCl::TieWordEmbeddingCl() :
  LayerImplCl(),
  tieword_embedding_props(props::InDim(), props::OutDim(), props::Unit(),
                          props::Scale()) {
  weight_idx.fill(std::numeric_limits<unsigned>::max());
}

bool TieWordEmbeddingCl::registerClKernels(ClContext &cl_context) {
  auto &layer_kernel_ptrs = getLayerKernelPtrs();

  if (!layer_kernel_ptrs.empty()) {
    ml_loge("kernels for tie_word_embedding_cl are already registered.");
    return false;
  }

  do {
    ClContext::SharedPtrClKernel kernel_ptr =
      cl_context.registerClKernel(embedding_kernel, "embedding_cl");
    if (!kernel_ptr) {
      ml_loge("OpenCL Error: Fail to register embedding_cl kernel");
      break;
    }
    layer_kernel_ptrs.emplace_back(kernel_ptr);

#ifdef ENABLE_FP16
    ClContext::SharedPtrClKernel kernel_fp16_ptr =
      cl_context.registerClKernel(embedding_fp16_kernel, "embedding_cl_fp16");
    if (!kernel_fp16_ptr) {
      ml_loge("OpenCL Error: Fail to register embedding_cl_fp16 kernel");
      break;
    }
    layer_kernel_ptrs.emplace_back(kernel_fp16_ptr);
#endif

    return true;
  } while (false);

  layer_kernel_ptrs.clear();
  return false;
}

void TieWordEmbeddingCl::finalize(InitLayerContext &context) {
  mode_ = std::get<props::Unit>(tieword_embedding_props).empty()
            ? mode::embedding
            : mode::lm_head;
  if (mode_ == mode::embedding)
    finalize_embedding(context);
  else if (mode_ == mode::lm_head)
    finalize_lmhead(context);
}

void TieWordEmbeddingCl::finalize_embedding(InitLayerContext &context) {
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

  unsigned int in_dim = std::get<props::InDim>(tieword_embedding_props);
  unsigned int out_dim = std::get<props::OutDim>(tieword_embedding_props);

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

  weight_idx[TieWordEmbeddingParams::weight] = context.requestWeight(
    dim, weight_initializer, weight_regularizer, weight_regularizer_constant,
    weight_decay, "Embedding", true);
}

void TieWordEmbeddingCl::finalize_lmhead(InitLayerContext &context) {
  auto &weight_regularizer =
    std::get<props::WeightRegularizer>(*layer_impl_props);
  auto &weight_regularizer_constant =
    std::get<props::WeightRegularizerConstant>(*layer_impl_props);
  auto weight_initializer = props::InitializerInfo::Enum::NONE;
  auto &weight_decay = std::get<props::WeightDecay>(*layer_impl_props);
  auto &bias_decay = std::get<props::BiasDecay>(*layer_impl_props);
  auto &bias_initializer = std::get<props::BiasInitializer>(*layer_impl_props);
  auto &disable_bias = std::get<props::DisableBias>(*layer_impl_props);

  auto unit = std::get<props::Unit>(tieword_embedding_props).get();

  NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
    << "lm head layer takes only one input";

  std::vector<TensorDim> output_dims(1);

  context.setEffDimFlagInputDimension(0, 0b1001);
  context.setDynDimFlagInputDimension(0, 0b1000);
  bool is_nchw = (context.getFormat() == Tformat::NCHW);

  auto const &in_dim = context.getInputDimensions()[0];
  output_dims[0] = in_dim;
  is_nchw ? output_dims[0].width(unit) : output_dims[0].channel(unit);
  output_dims[0].height(1);

  output_dims[0].setTensorType(
    {context.getFormat(), context.getActivationDataType()});

  context.setOutputDimensions(output_dims);

  TensorDim bias_dim(
    1, is_nchw ? 1 : unit, 1, is_nchw ? unit : 1,
    TensorDim::TensorType(context.getFormat(), context.getWeightDataType()),
    is_nchw ? 0b0001 : 0b0100);

  TensorDim weight_dim(
    1, is_nchw ? 1 : in_dim.channel(), is_nchw ? unit : 1,
    is_nchw ? in_dim.width() : unit,
    TensorDim::TensorType(context.getFormat(), context.getWeightDataType()),
    is_nchw ? 0b0011 : 0b0101);

  weight_idx[TieWordEmbeddingParams::weight] = context.requestWeight(
    weight_dim, weight_initializer, weight_regularizer,
    weight_regularizer_constant, weight_decay, "Embedding", true);

  if (disable_bias.empty() || disable_bias.get() == false) {
    weight_idx[TieWordEmbeddingParams::bias] = context.requestWeight(
      bias_dim, bias_initializer, WeightRegularizer::NONE, 1.0f, bias_decay,
      "bias", true);
  }
}

void TieWordEmbeddingCl::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, tieword_embedding_props);
  LayerImplCl::setProperty(remain_props);
}

void TieWordEmbeddingCl::forwarding(RunLayerContext &context, bool training) {}

void TieWordEmbeddingCl::incremental_forwarding(RunLayerContext &context,
                                                unsigned int from,
                                                unsigned int to,
                                                bool training) {
  if (mode_ == mode::embedding)
    incremental_forwarding_embedding(context, from, to, training);
  else if (mode_ == mode::lm_head)
    incremental_forwarding_lmhead(context, from, to, training);
  else
    throw std::invalid_argument("unknown mode");
}

void TieWordEmbeddingCl::incremental_forwarding_embedding(
  RunLayerContext &context, unsigned int from, unsigned int to, bool training) {

  unsigned int in_dim = std::get<props::InDim>(tieword_embedding_props);
  unsigned int out_dim = std::get<props::OutDim>(tieword_embedding_props);
  float scale = std::get<props::Scale>(tieword_embedding_props).empty()
                  ? 1.0f
                  : std::get<props::Scale>(tieword_embedding_props).get();

  Tensor &weight = context.getWeight(weight_idx[TieWordEmbeddingParams::weight]);
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
    embedding_cl_kernel(input_.getData<float>(), weight.getData<float>(),
                        hidden_.getData<float>(), num_tokens, out_dim, scale);
  } else if (weight.getDataType() == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    embedding_cl_fp16_kernel(input_.getData<float>(), weight.getData<_FP16>(),
                             hidden_.getData<_FP16>(), num_tokens, out_dim,
                             scale);
#else
    throw std::invalid_argument("Error: enable-fp16 is not enabled");
#endif
  } else {
    /// Q6_K fallback on CPU
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

void TieWordEmbeddingCl::incremental_forwarding_lmhead(
  RunLayerContext &context, unsigned int from, unsigned int to, bool training) {

  Tensor w;
  Tensor &weight = w;
  context.getWeight(weight, weight_idx[TieWordEmbeddingParams::weight]);

  Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);
  Tensor &hidden_ = context.getOutput(SINGLE_INOUT_IDX);

  TensorDim input_dim = input_.getDim();
  TensorDim hidden_dim = hidden_.getDim();

  TensorDim input_step_dim = input_dim;
  TensorDim hidden_step_dim = hidden_dim;

  input_step_dim.batch(1);
  input_step_dim.height(1);
  hidden_step_dim.batch(1);

  unsigned int b_size = input_dim.batch();

  for (unsigned int b = 0; b < b_size; ++b) {
    Tensor input_step = input_.getSharedDataTensor(
      input_step_dim,
      b * input_dim.getFeatureLen() + (to - from - 1) * input_.width(), true);
    Tensor hidden_step = hidden_.getSharedDataTensor(
      hidden_step_dim, b * hidden_dim.getFeatureLen(), true);

    NNTR_THROW_IF(weight.getDataType() == TensorDim::DataType::BCQ,
                  std::invalid_argument)
      << "weight type is not supported for custom tie word embedding layer";

    /// @note weight is transposed because it's shared with embedding
    dotCl(input_step, weight, hidden_step, false, true);

    if (auto &disable_bias = std::get<props::DisableBias>(*layer_impl_props);
        disable_bias.empty() || disable_bias.get() == false) {
      Tensor &bias =
        context.getWeight(weight_idx[TieWordEmbeddingParams::bias]);
      hidden_step.add_i(bias);
    }
  }
}

void TieWordEmbeddingCl::embedding_cl_kernel(float *input, float *weight,
                                             float *output,
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
      if (!write_result)
        break;

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
      set_result &= kernel_ptr->SetKernelArguments(3, &od, sizeof(int));
      set_result &= kernel_ptr->SetKernelArguments(4, &scale, sizeof(float));
      if (!set_result)
        break;
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
      set_svm_result &= kernel_ptr->SetKernelArguments(3, &od, sizeof(int));
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
            total_dim * sizeof(float), output))
        break;
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
void TieWordEmbeddingCl::embedding_cl_fp16_kernel(
  float *input, _FP16 *weight, _FP16 *output, unsigned int num_tokens,
  unsigned int out_dim, float scale, bool svm) {

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
    if (!write_result)
      break;

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
    set_result &= kernel_ptr->SetKernelArguments(3, &od, sizeof(int));
    set_result &= kernel_ptr->SetKernelArguments(4, &scale, sizeof(float));
    if (!set_result)
      break;

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
          total_dim * sizeof(_FP16), output))
      break;

  } while (false);
}
#endif

void TieWordEmbeddingCl::calcDerivative(RunLayerContext &context) {
  std::throw_with_nested(std::runtime_error(
    "calcDerivative for TieWordEmbedding layer is not supported"));
}

void TieWordEmbeddingCl::exportTo(
  Exporter &exporter, const ml::train::ExportMethods &method) const {
  LayerImplCl::exportTo(exporter, method);
  exporter.saveResult(tieword_embedding_props, method, this);
}

void TieWordEmbeddingCl::updateTensorsByInputDimensions(
  RunLayerContext &context, std::vector<TensorDim> input_dimensions) {
  TensorDim in_dim = context.getInput(SINGLE_INOUT_IDX).getDim();
  TensorDim out_dim = context.getOutput(SINGLE_INOUT_IDX).getDim();

  unsigned int height = input_dimensions[0].height();

  if (mode_ == mode::embedding) {
    in_dim.width(height);
  } else {
    in_dim.height(height);
  }
  out_dim.height(height);

  context.updateInput(SINGLE_INOUT_IDX, in_dim);
  context.updateOutput(SINGLE_INOUT_IDX, out_dim);
}

void TieWordEmbeddingCl::read(
  std::ifstream &file, RunLayerContext &context, bool opt_var,
  ml::train::ExecutionMode exec_mode, bool trainable,
  TensorDim::DataType definedWeightDataType, bool fsu, size_t start_offset,
  bool read_from_offset, int file_fd) {
  if (mode_ == mode::embedding) {
    for (unsigned int i = 0; i < context.getNumWeights(); ++i) {
      if (context.isGradientFirstAccess(i)) {
        context.getWeight(i).read(file, start_offset, read_from_offset);
        if (context.isMixedPrecision(i) && trainable &&
            !context.getWeightFP32(i).empty()) {
          context.getWeightFP32(i).copyData(context.getWeight(i));
        }
      }
    }
  }
}

void TieWordEmbeddingCl::read(
  ReadSource src, RunLayerContext &context, bool opt_var,
  ml::train::ExecutionMode exec_mode, bool trainable,
  TensorDim::DataType definedWeightDataType, bool fsu, size_t start_offset,
  bool read_from_offset) {
  if (mode_ == mode::embedding) {
    for (unsigned int i = 0; i < context.getNumWeights(); ++i) {
      if (context.isGradientFirstAccess(i)) {
        context.getWeight(i).read(src, start_offset, read_from_offset);
        if (context.isMixedPrecision(i) && trainable &&
            !context.getWeightFP32(i).empty()) {
          context.getWeightFP32(i).copyData(context.getWeight(i));
        }
      }
    }
  }
}

void TieWordEmbeddingCl::save(std::ofstream &file, RunLayerContext &run_context,
                              bool opt_var, ml::train::ExecutionMode exec_mode,
                              bool trainable, TensorDim::DataType dtype) const {
  if (mode_ == mode::embedding) {
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

          if (dtype == TensorDim::DataType::Q6_K) {
            Tensor quant_weight(dim.batch(), dim.channel(), K, N,
                                {Tformat::NCHW, dtype});
            quantize_q6_K(w.getData<float>(), quant_weight.getData<uint8_t>(),
                          K, N, nullptr);
            quant_weight.save(file);
          } else {
            NNTR_THROW_IF(true, std::runtime_error)
              << "This dtype is not supported in save with quantization";
          }
        }
      }
    }
  }
}

std::vector<ClContext::SharedPtrClKernel> &
TieWordEmbeddingCl::getLayerKernelPtrs() {
  static std::vector<ClContext::SharedPtrClKernel> layer_kernel_ptrs;
  return layer_kernel_ptrs;
}

} // namespace nntrainer
