// SPDX-License-Identifier: Apache-2.0
/**
 * @file   swiglu_down_layer.cpp
 * @brief  See swiglu_down_layer.h.  M=1 decode dispatches the fused
 *         swiglu_down_int4 kernel; M>1 prefill falls back to the
 *         existing swiglu + dot sequence.
 */

#include <swiglu_down_layer.h>

#include <blas_kernels.h>
#include <cl_context.h>
#include <engine.h>
#include <layer_context.h>
#include <nntrainer_error.h>
#include <node_exporter.h>
#include <util_func.h>

#include <cmath>

namespace causallm {

static constexpr size_t SINGLE_INOUT_IDX = 0;
static constexpr size_t INPUT_GATE = 0;
static constexpr size_t INPUT_UP = 1;

SwigluDownLayer::SwigluDownLayer() :
  LayerImpl(), swiglu_down_props(props::DownUnit()) {
  weight_idx = std::numeric_limits<unsigned>::max();
}

void SwigluDownLayer::finalize(nntrainer::InitLayerContext &context) {
  NNTR_THROW_IF(context.getNumInputs() != 2, std::invalid_argument)
    << "SwigluDown layer takes exactly two inputs (gate, up).";

  auto &weight_regularizer =
    std::get<nntrainer::props::WeightRegularizer>(*layer_impl_props);
  auto &weight_regularizer_constant =
    std::get<nntrainer::props::WeightRegularizerConstant>(*layer_impl_props);
  auto weight_initializer = nntrainer::props::InitializerInfo::Enum::NONE;
  auto &weight_decay =
    std::get<nntrainer::props::WeightDecay>(*layer_impl_props);

  const auto &down_unit = std::get<props::DownUnit>(swiglu_down_props).get();

  bool is_nchw = (context.getFormat() == nntrainer::Tformat::NCHW);
  // Gate and up have the same shape; we use the gate dim to size output.
  auto const &in_dim = context.getInputDimensions()[INPUT_GATE];

  std::vector<nntrainer::TensorDim> output_dims(1);
  output_dims[0] = in_dim;
  is_nchw ? output_dims[0].width(down_unit)
          : output_dims[0].channel(down_unit);
  output_dims[0].setTensorType(
    {context.getFormat(), context.getActivationDataType()});
  context.setOutputDimensions(output_dims);

  context.setEffDimFlagInputDimension(0, 0b1001);
  context.setDynDimFlagInputDimension(0, 0b1000);
  context.setEffDimFlagInputDimension(1, 0b1001);
  context.setDynDimFlagInputDimension(1, 0b1000);

  /** Down weight (int4 channel-wise).  Registered as the layer's only
   *  weight so the bundle byte position matches the legacy ffn_down.
   */
  nntrainer::TensorDim weight_dim(
    1, is_nchw ? 1 : down_unit, is_nchw ? in_dim.width() : 1,
    is_nchw ? down_unit : in_dim.channel(),
    nntrainer::TensorDim::TensorType(context.getFormat(),
                                     context.getWeightDataType()),
    is_nchw ? 0b0011 : 0b0101);
  weight_idx = context.requestWeight(
    weight_dim, weight_initializer, weight_regularizer,
    weight_regularizer_constant, weight_decay, "downweight", true);
}

void SwigluDownLayer::exportTo(nntrainer::Exporter &exporter,
                               const ml::train::ExportMethods &method) const {
  LayerImpl::exportTo(exporter, method);
  exporter.saveResult(swiglu_down_props, method, this);
}

void SwigluDownLayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, swiglu_down_props);
  LayerImpl::setProperty(remain_props);
}

void SwigluDownLayer::forwarding(nntrainer::RunLayerContext &context,
                                 bool training) {
  return;
}

void SwigluDownLayer::incremental_forwarding(
  nntrainer::RunLayerContext &context, unsigned int from, unsigned int to,
  bool training) {
  nntrainer::Tensor &down_w = context.getWeight(weight_idx);
  nntrainer::Tensor &gate_in = context.getInput(INPUT_GATE);
  nntrainer::Tensor &up_in = context.getInput(INPUT_UP);
  nntrainer::Tensor &out_ = context.getOutput(SINGLE_INOUT_IDX);

  nntrainer::TensorDim gate_dim = gate_in.getDim();
  nntrainer::TensorDim gate_step_dim = gate_dim;
  gate_step_dim.batch(1);
  gate_step_dim.height(to - from);
  nntrainer::Tensor gate_step =
    gate_in.getSharedDataTensor(gate_step_dim, 0, true);

  nntrainer::TensorDim up_dim = up_in.getDim();
  nntrainer::TensorDim up_step_dim = up_dim;
  up_step_dim.batch(1);
  up_step_dim.height(to - from);
  nntrainer::Tensor up_step = up_in.getSharedDataTensor(up_step_dim, 0, true);

  nntrainer::TensorDim out_dim = out_.getDim();
  nntrainer::TensorDim out_step_dim = out_dim;
  out_step_dim.batch(1);
  out_step_dim.height(to - from);
  nntrainer::Tensor out_step =
    out_.getSharedDataTensor(out_step_dim, 0, true);

  const unsigned int M = (unsigned int)(to - from);
  const unsigned int K = gate_step_dim.width();
  const unsigned int N = out_step_dim.width();

#if defined(ENABLE_OPENCL) && ENABLE_OPENCL == 1
  // M=1 fused path.  Requires gate/up/out all fp16 SVM and weight
  // QINT4.  Saves 1 dispatch + the swiglu output SVM round-trip
  // per layer.
  if (M == 1 &&
      gate_in.getMemoryData() && gate_in.getMemoryData()->isSVM() &&
      up_in.getMemoryData() && up_in.getMemoryData()->isSVM() &&
      out_.getMemoryData() && out_.getMemoryData()->isSVM() &&
      gate_in.getDataType() == ml::train::TensorDim::DataType::FP16 &&
      up_in.getDataType() == ml::train::TensorDim::DataType::FP16 &&
      down_w.getDataType() == ml::train::TensorDim::DataType::QINT4 &&
      (K & 3u) == 0 && (N & 3u) == 0) {
    auto *gate_svm =
      reinterpret_cast<uint16_t *>(gate_step.getData<_FP16>());
    auto *up_svm = reinterpret_cast<uint16_t *>(up_step.getData<_FP16>());
    auto *w_svm = reinterpret_cast<uint16_t *>(down_w.getData<char>());
    auto *s_svm = down_w.getScale<uint16_t>();
    auto *out_svm =
      reinterpret_cast<uint16_t *>(out_step.getData<_FP16>());

    if (nntrainer::swiglu_down_int4_cl(gate_svm, up_svm, w_svm, s_svm,
                                        out_svm, K, N,
                                        /*sync_output=*/true)) {
      return;
    }
    // Helper rejected; fall through to the prefill-style path.
  }
#endif

  // M>1 prefill (or M=1 fallback): apply swiglu CPU NEON to a temp
  // tensor (use out_step as scratch -- it's SVM-backed and width N >=
  // K_intermediate is NOT guaranteed... actually out width = down_unit
  // = hidden, and K = intermediate (larger).  We can't use out_step
  // as scratch because it's smaller.  Use up_step in-place instead:
  // compute silu(gate)*up into up_step (overwriting up's input), then
  // dot up_step with weight -> out_step.  Caller mustn't reuse up_in
  // after this call (network_graph guarantees this since up output
  // has only this layer as consumer in the standard MLP graph).
  {
    _FP16 *gate_ptr = gate_step.getData<_FP16>();
    _FP16 *up_ptr = up_step.getData<_FP16>();
    const std::size_t total = (std::size_t)gate_step_dim.batch() *
                               gate_step_dim.channel() *
                               gate_step_dim.height() * K;
    for (std::size_t i = 0; i < total; ++i) {
      const float g = (float)gate_ptr[i];
      const float u = (float)up_ptr[i];
      const float silu_g = g / (1.0f + std::exp(-g));
      up_ptr[i] = (_FP16)(silu_g * u);
    }
  }
  // up_step now holds silu(gate)*up.  Dispatch the down FC.
  up_step.dot(down_w, out_step);
}

void SwigluDownLayer::calcDerivative(nntrainer::RunLayerContext &context) {
  return;
}

void SwigluDownLayer::calcGradient(nntrainer::RunLayerContext &context) {
  return;
}

void SwigluDownLayer::updateTensorsByInputDimensions(
  nntrainer::RunLayerContext &context,
  std::vector<nntrainer::TensorDim> input_dimensions) {
  ml::train::TensorDim gate_dim = context.getInput(INPUT_GATE).getDim();
  ml::train::TensorDim up_dim = context.getInput(INPUT_UP).getDim();
  ml::train::TensorDim out_dim = context.getOutput(SINGLE_INOUT_IDX).getDim();

  gate_dim.height(input_dimensions[0].height());
  up_dim.height(input_dimensions[1].height());
  out_dim.height(input_dimensions[0].height());

  context.updateInput(INPUT_GATE, gate_dim);
  context.updateInput(INPUT_UP, up_dim);
  context.updateOutput(SINGLE_INOUT_IDX, out_dim);
}

} // namespace causallm
