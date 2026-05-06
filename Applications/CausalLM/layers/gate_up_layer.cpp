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

#if defined(ENABLE_OPENCL) && ENABLE_OPENCL == 1
#include <blas_kernels.h>
#endif

namespace causallm {

static constexpr size_t SINGLE_INOUT_IDX = 0;

// Weight slot enum.  When fused_rmsnorm is enabled gamma occupies
// slot 0, pushing Up/Gate to slots 1/2 so the bundle byte order
// matches the original (rms_norm gamma + ffn_up + ffn_gate) layout.
enum GateUpParams { Up = 0, Gate = 1 };
enum GateUpFusedParams { FusedGamma = 0, FusedUp = 1, FusedGate = 2 };

GateUpLayer::GateUpLayer() :
  LayerImpl(),
  gate_up_props(props::UpUnit(), props::GateUnit(),
                props::FusedRmsnorm(false),
                props::FusedRmsnormEpsilon(1.0e-6f)) {
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

  const bool fused_rmsnorm =
    std::get<props::FusedRmsnorm>(gate_up_props).get();
  if (fused_rmsnorm) {
    // Phase A: register gamma FIRST so the weight bundle byte order
    // matches the original (rms_norm gamma + ffn_up + ffn_gate)
    // layout.  Gamma shape = (1,1,1,hidden) FP32, identical to
    // RMSNormLayer::finalize.  Without this prefix the model loader
    // (positional) would slide all subsequent weights by gamma_size.
    nntrainer::TensorDim gamma_dim(
      1, 1, 1, in_dim.width(),
      nntrainer::TensorDim::TensorType(
        context.getFormat(), nntrainer::TensorDim::DataType::FP32));
    weight_idx[GateUpFusedParams::FusedGamma] = context.requestWeight(
      gamma_dim, nntrainer::props::InitializerInfo::Enum::NONE,
      nntrainer::WeightRegularizer::NONE, 1.0f, 0.0f, "gamma",
      /*trainable=*/false);
  }

  /** Up weight */
  nntrainer::TensorDim weight_dim(
    1, is_nchw ? 1 : up_unit, is_nchw ? in_dim.width() : 1,
    is_nchw ? up_unit : in_dim.channel(),
    nntrainer::TensorDim::TensorType(context.getFormat(),
                                     context.getWeightDataType()),
    is_nchw ? 0b0011 : 0b0101);
  weight_idx[fused_rmsnorm ? GateUpFusedParams::FusedUp
                            : GateUpParams::Up] =
    context.requestWeight(weight_dim, weight_initializer, weight_regularizer,
                          weight_regularizer_constant, weight_decay,
                          "upweight", true);

  /** Gate weight */
  weight_dim.width(gate_unit);
  weight_idx[fused_rmsnorm ? GateUpFusedParams::FusedGate
                            : GateUpParams::Gate] =
    context.requestWeight(weight_dim, weight_initializer, weight_regularizer,
                          weight_regularizer_constant, weight_decay,
                          "gateweight", true);
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
  const bool fused_rmsnorm =
    std::get<props::FusedRmsnorm>(gate_up_props).get();
  const auto up_idx =
    fused_rmsnorm ? GateUpFusedParams::FusedUp : GateUpParams::Up;
  const auto gate_idx =
    fused_rmsnorm ? GateUpFusedParams::FusedGate : GateUpParams::Gate;
  nntrainer::Tensor &Uweight = context.getWeight(weight_idx[up_idx]);
  nntrainer::Tensor &Gweight = context.getWeight(weight_idx[gate_idx]);
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

#if defined(ENABLE_OPENCL) && ENABLE_OPENCL == 1
  // Phase A wire-in: when fused_rmsnorm is set, our `input_` is the
  // raw post-residual tensor (transformer.cpp createMlp wires us
  // directly to decoder_add).  Fold the rmsnorm + gate proj + up proj
  // into a single dispatch via fused_rmsnorm_gate_up_cl.  Saves one
  // kernel dispatch per layer per token AND eliminates the
  // intermediate rms_norm SVM tensor + drain.
  if (fused_rmsnorm) {
    nntrainer::Tensor &gamma_ =
      context.getWeight(weight_idx[GateUpFusedParams::FusedGamma]);
    const float epsilon =
      std::get<props::FusedRmsnormEpsilon>(gate_up_props).get();
    if (input_step.getDataType() ==
          ml::train::TensorDim::DataType::FP16 &&
        input_step.getMemoryData() && input_step.getMemoryData()->isSVM() &&
        Uhidden_step.getMemoryData() &&
        Uhidden_step.getMemoryData()->isSVM() &&
        Ghidden_step.getMemoryData() &&
        Ghidden_step.getMemoryData()->isSVM()) {
      const unsigned int K_in = input_step.width();
      const unsigned int N_up = Uhidden_step.width();
      const unsigned int N_gate = Ghidden_step.width();
      // Helper signature is (gate, up) -- match it.
      const bool fused_ok = nntrainer::fused_rmsnorm_gate_up_cl(
        reinterpret_cast<uint16_t *>(input_step.getData<char>()),
        gamma_.getData<float>(),
        reinterpret_cast<uint16_t *>(Gweight.getData<char>()),
        Gweight.getScale<uint16_t>(),
        reinterpret_cast<uint16_t *>(Ghidden_step.getData<char>()),
        reinterpret_cast<uint16_t *>(Uweight.getData<char>()),
        Uweight.getScale<uint16_t>(),
        reinterpret_cast<uint16_t *>(Uhidden_step.getData<char>()),
        K_in, N_gate, N_up, epsilon);
      if (fused_ok) {
        return;
      }
      // Fall through to the legacy dot path on shape constraint
      // violation (helper returns false e.g. when N % 64 != 0).
    }
  }
#endif

  // Legacy path (also serves as fused-mode fallback for unsupported
  // shapes / non-SVM tensors).  Match the per-FC ordering (up first,
  // gate second) so the batched dot path's fused_gemv_int4_cl wiring
  // sees the same weight and output layout that finalize() registered.
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
