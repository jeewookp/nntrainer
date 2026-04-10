// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Samsung Electronics Co., Ltd.
 *
 * @file   lm_head_cl.cpp
 * @date   10 April 2026
 * @brief  LM Head Layer Class with OpenCL implementation
 * @see    https://github.com/nntrainer/nntrainer
 * @bug    No known bugs except for NYI items
 *
 */

#include <blas_kernel_interface.h>
#include <common_properties.h>
#include <layer_context.h>
#include <lm_head_cl.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>

namespace nntrainer {

static constexpr size_t SINGLE_INOUT_IDX = 0;

enum LmHeadParams {
  weight,
  bias,
};

LmHeadLayerCl::LmHeadLayerCl() :
  LayerImplCl(), lmhead_props(props::Unit()) {
  weight_idx.fill(std::numeric_limits<unsigned>::max());
}

void LmHeadLayerCl::finalize(InitLayerContext &context) {
  auto &weight_regularizer =
    std::get<props::WeightRegularizer>(*layer_impl_props);
  auto &weight_regularizer_constant =
    std::get<props::WeightRegularizerConstant>(*layer_impl_props);
  auto weight_initializer = props::InitializerInfo::Enum::NONE;
  auto &weight_decay = std::get<props::WeightDecay>(*layer_impl_props);
  auto &bias_decay = std::get<props::BiasDecay>(*layer_impl_props);
  auto &bias_initializer = std::get<props::BiasInitializer>(*layer_impl_props);
  auto &disable_bias = std::get<props::DisableBias>(*layer_impl_props);

  auto unit = std::get<props::Unit>(lmhead_props).get();

  NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
    << "lm head layer takes only one input";

  std::vector<TensorDim> output_dims(1);

  context.setEffDimFlagInputDimension(0, 0b1001);
  context.setDynDimFlagInputDimension(0, 0b1000);
  bool is_nchw = (context.getFormat() == Tformat::NCHW);

  auto const &in_dim = context.getInputDimensions()[0];
  output_dims[0] = in_dim;
  if (is_nchw)
    output_dims[0].width(unit);
  else
    output_dims[0].channel(unit);
  output_dims[0].height(1);

  output_dims[0].setTensorType(
    {context.getFormat(), context.getActivationDataType()});

  context.setOutputDimensions(output_dims);

  TensorDim bias_dim(
    1, is_nchw ? 1 : unit, 1, is_nchw ? unit : 1,
    TensorDim::TensorType(context.getFormat(), context.getWeightDataType()),
    is_nchw ? 0b0001 : 0b0100);

  TensorDim weight_dim(
    1, is_nchw ? 1 : unit, is_nchw ? in_dim.width() : 1,
    is_nchw ? unit : in_dim.channel(),
    TensorDim::TensorType(context.getFormat(), context.getWeightDataType()),
    is_nchw ? 0b0011 : 0b0101);

  weight_idx[LmHeadParams::weight] = context.requestWeight(
    weight_dim, weight_initializer, weight_regularizer,
    weight_regularizer_constant, weight_decay, "weight", true);

  if (disable_bias.empty() || disable_bias.get() == false) {
    weight_idx[LmHeadParams::bias] = context.requestWeight(
      bias_dim, bias_initializer, WeightRegularizer::NONE, 1.0f, bias_decay,
      "bias", true);
  }
}

void LmHeadLayerCl::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, lmhead_props);
  LayerImplCl::setProperty(remain_props);
}

void LmHeadLayerCl::forwarding(RunLayerContext &context, bool training) {
  throw exception::not_supported(
    "Forwarding for LMHead layer is not supported");
}

void LmHeadLayerCl::incremental_forwarding(RunLayerContext &context,
                                           unsigned int from, unsigned int to,
                                           bool training) {
  Tensor w;
  Tensor &weight = w;
  context.getWeight(weight, weight_idx[LmHeadParams::weight]);

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

    dotCl(input_step, weight, hidden_step);

    if (auto &disable_bias = std::get<props::DisableBias>(*layer_impl_props);
        disable_bias.empty() || disable_bias.get() == false) {
      Tensor &bias = context.getWeight(weight_idx[LmHeadParams::bias]);
      hidden_step.add_i(bias);
    }
  }
}

void LmHeadLayerCl::calcDerivative(RunLayerContext &context) {
  std::throw_with_nested(
    std::runtime_error("calcDerivative for LMHead layer is not supported"));
}

void LmHeadLayerCl::calcGradient(RunLayerContext &context) {
  std::throw_with_nested(
    std::runtime_error("calcGradient for LMHead layer is not supported"));
}

void LmHeadLayerCl::exportTo(Exporter &exporter,
                             const ml::train::ExportMethods &method) const {
  LayerImplCl::exportTo(exporter, method);
  exporter.saveResult(lmhead_props, method, this);
}

void LmHeadLayerCl::updateTensorsByInputDimensions(
  RunLayerContext &context, std::vector<TensorDim> input_dimensions) {
  TensorDim in_dim = context.getInput(SINGLE_INOUT_IDX).getDim();

  unsigned int height = input_dimensions[0].height();

  // output dim's height is always 1
  in_dim.height(height);
  context.updateInput(SINGLE_INOUT_IDX, in_dim);
}

} // namespace nntrainer
