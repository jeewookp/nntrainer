// SPDX-License-Identifier: Apache-2.0
/**
 * @file   gate_up_layer.cpp
 * @brief  Batched MLP gate/up projection.  See gate_up_layer.h for
 *         the weight ordering rationale.
 */

#include <gate_up_layer.h>

#include <bs_thread_pool_manager.hpp>
#include <engine.h>
#include <layer_context.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <util_func.h>

namespace causallm {

static constexpr size_t SINGLE_INOUT_IDX = 0;

// Output index convention: matches weight registration order (Up, Gate)
// so the on-disk byte layout is preserved relative to the legacy
// ffn_up + ffn_gate FC pair.
enum GateUpParams { Up, Gate };

GateUpLayer::GateUpLayer() :
  LayerImpl(), gate_up_props(props::UpUnit(), props::GateUnit()) {
  weight_idx.fill(std::numeric_limits<unsigned>::max());
}

void GateUpLayer::finalize(nntrainer::InitLayerContext &context) {
  NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
    << "GateUp layer takes only one input";

  auto &weight_regularizer =
    std::get<nntrainer::props::WeightRegularizer>(*layer_impl_props);
  auto &weight_regularizer_constant =
    std::get<nntrainer::props::WeightRegularizerConstant>(*layer_impl_props);
  auto weight_initializer = nntrainer::props::InitializerInfo::Enum::NONE;
  auto &weight_decay =
    std::get<nntrainer::props::WeightDecay>(*layer_impl_props);

  const auto &up_unit = std::get<props::UpUnit>(gate_up_props).get();
  const auto &gate_unit = std::get<props::GateUnit>(gate_up_props).get();

  std::vector<nntrainer::TensorDim> output_dims(2);

  context.setEffDimFlagInputDimension(0, 0b1001);
  context.setDynDimFlagInputDimension(0, 0b1000);

  bool is_nchw = (context.getFormat() == nntrainer::Tformat::NCHW);
  auto const &in_dim = context.getInputDimensions()[0];

  /** Up out (output index 0) */
  output_dims[GateUpParams::Up] = in_dim;
  is_nchw ? output_dims[GateUpParams::Up].width(up_unit)
          : output_dims[GateUpParams::Up].channel(up_unit);
  output_dims[GateUpParams::Up].setTensorType(
    {context.getFormat(), context.getActivationDataType()});

  /** Gate out (output index 1) */
  output_dims[GateUpParams::Gate] = in_dim;
  is_nchw ? output_dims[GateUpParams::Gate].width(gate_unit)
          : output_dims[GateUpParams::Gate].channel(gate_unit);
  output_dims[GateUpParams::Gate].setTensorType(
    {context.getFormat(), context.getActivationDataType()});

  context.setOutputDimensions(output_dims);

  /** Up weight */
  nntrainer::TensorDim weight_dim(
    1, is_nchw ? 1 : up_unit, is_nchw ? in_dim.width() : 1,
    is_nchw ? up_unit : in_dim.channel(),
    nntrainer::TensorDim::TensorType(context.getFormat(),
                                     context.getWeightDataType()),
    is_nchw ? 0b0011 : 0b0101);
  weight_idx[GateUpParams::Up] = context.requestWeight(
    weight_dim, weight_initializer, weight_regularizer,
    weight_regularizer_constant, weight_decay, "upweight", true);

  /** Gate weight */
  weight_dim.width(gate_unit);
  weight_idx[GateUpParams::Gate] = context.requestWeight(
    weight_dim, weight_initializer, weight_regularizer,
    weight_regularizer_constant, weight_decay, "gateweight", true);
}

void GateUpLayer::exportTo(nntrainer::Exporter &exporter,
                           const ml::train::ExportMethods &method) const {
  LayerImpl::exportTo(exporter, method);
  exporter.saveResult(gate_up_props, method, this);
}

void GateUpLayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, gate_up_props);
  LayerImpl::setProperty(remain_props);
}

void GateUpLayer::forwarding(nntrainer::RunLayerContext &context,
                             bool training) {
  return;
}

void GateUpLayer::incremental_forwarding(nntrainer::RunLayerContext &context,
                                         unsigned int from, unsigned int to,
                                         bool training) {
  nntrainer::Tensor &Uweight = context.getWeight(weight_idx[GateUpParams::Up]);
  nntrainer::Tensor &Gweight =
    context.getWeight(weight_idx[GateUpParams::Gate]);
  nntrainer::Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &Uhidden_ = context.getOutput(GateUpParams::Up);
  nntrainer::Tensor &Ghidden_ = context.getOutput(GateUpParams::Gate);

  nntrainer::TensorDim input_dim = input_.getDim();
  nntrainer::TensorDim input_step_dim = input_dim;
  input_step_dim.batch(1);
  input_step_dim.height(to - from);

  nntrainer::Tensor input_step =
    input_.getSharedDataTensor(input_step_dim, 0, true);

  nntrainer::TensorDim Uhidden_step_dim = Uhidden_.getDim();
  Uhidden_step_dim.batch(1);
  Uhidden_step_dim.height(to - from);
  nntrainer::Tensor Uhidden_step =
    Uhidden_.getSharedDataTensor(Uhidden_step_dim, 0, true);

  nntrainer::TensorDim Ghidden_step_dim = Ghidden_.getDim();
  Ghidden_step_dim.batch(1);
  Ghidden_step_dim.height(to - from);
  nntrainer::Tensor Ghidden_step =
    Ghidden_.getSharedDataTensor(Ghidden_step_dim, 0, true);

  // Match the per-FC ordering (up first, gate second) so the
  // batched dot path's fused_gemv_int4_cl wiring sees the same weight
  // and output layout that finalize() registered.
  std::vector<nntrainer::Tensor *> Weights({&Uweight, &Gweight});
  std::vector<nntrainer::Tensor *> Outputs({&Uhidden_step, &Ghidden_step});

  input_step.dot(Weights, Outputs);
}

void GateUpLayer::calcDerivative(nntrainer::RunLayerContext &context) {
  return;
}

void GateUpLayer::calcGradient(nntrainer::RunLayerContext &context) { return; }

void GateUpLayer::updateTensorsByInputDimensions(
  nntrainer::RunLayerContext &context,
  std::vector<nntrainer::TensorDim> input_dimensions) {
  ml::train::TensorDim input_dim = context.getInput(SINGLE_INOUT_IDX).getDim();
  ml::train::TensorDim Uoutput_dim =
    context.getOutput(GateUpParams::Up).getDim();
  ml::train::TensorDim Goutput_dim =
    context.getOutput(GateUpParams::Gate).getDim();

  input_dim.height(input_dimensions[0].height());
  Uoutput_dim.height(input_dimensions[0].height());
  Goutput_dim.height(input_dimensions[0].height());

  context.updateInput(SINGLE_INOUT_IDX, input_dim);
  context.updateOutput(GateUpParams::Up, Uoutput_dim);
  context.updateOutput(GateUpParams::Gate, Goutput_dim);
}

} // namespace causallm
