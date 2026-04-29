// SPDX-License-Identifier: Apache-2.0
/**
 * @file   gate_up_layer.cpp
 * @brief  Batched MLP gate/up projection.  See gate_up_layer.h for
 *         the weight ordering rationale.
 */

#include <gate_up_layer.h>

#include <blas_kernels.h>
#include <bs_thread_pool_manager.hpp>
#include <cl_context.h>
#include <engine.h>
#include <layer_context.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <rmsnorm_fused_fp16.h>
#include <util_func.h>

namespace causallm {

static constexpr size_t SINGLE_INOUT_IDX = 0;

// Weight registration order matches the legacy byte layout
// (ffn_norm.gamma | ffn_up.weight | ffn_gate.weight) so existing
// pytorch_model.bin loads without repackaging.  Output index 0 is up,
// 1 is gate -- swiglu is referenced as gate_up_layer(1),gate_up_layer(0)
// to match its (gate, up) input order convention.
enum GateUpWeightIdx { Gamma, UpWeight, GateWeight };
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

  /** RMSNorm gamma (registered FIRST to preserve legacy byte order:
   *  ffn_norm.gamma | ffn_up.weight | ffn_gate.weight).  Force fp32
   *  to match the canonical RMSNormLayer storage; see rms_norm.cpp
   *  for why fp16 gamma corrupts the input.
   */
  const unsigned int K_in =
    is_nchw ? in_dim.width() : in_dim.channel();
  nntrainer::TensorDim gamma_dim(
    1, 1, 1, K_in,
    nntrainer::TensorDim::TensorType(
      context.getFormat(), nntrainer::TensorDim::DataType::FP32));
  weight_idx[GateUpWeightIdx::Gamma] = context.requestWeight(
    gamma_dim, nntrainer::props::InitializerInfo::Enum::NONE,
    nntrainer::WeightRegularizer::NONE, 1.0f, 0.0f, "gamma", false);

  /** Up weight */
  nntrainer::TensorDim weight_dim(
    1, is_nchw ? 1 : up_unit, is_nchw ? in_dim.width() : 1,
    is_nchw ? up_unit : in_dim.channel(),
    nntrainer::TensorDim::TensorType(context.getFormat(),
                                     context.getWeightDataType()),
    is_nchw ? 0b0011 : 0b0101);
  weight_idx[GateUpWeightIdx::UpWeight] = context.requestWeight(
    weight_dim, weight_initializer, weight_regularizer,
    weight_regularizer_constant, weight_decay, "upweight", true);

  /** Gate weight */
  weight_dim.width(gate_unit);
  weight_idx[GateUpWeightIdx::GateWeight] = context.requestWeight(
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
  nntrainer::Tensor &gamma =
    context.getWeight(weight_idx[GateUpWeightIdx::Gamma]);
  nntrainer::Tensor &Uweight =
    context.getWeight(weight_idx[GateUpWeightIdx::UpWeight]);
  nntrainer::Tensor &Gweight =
    context.getWeight(weight_idx[GateUpWeightIdx::GateWeight]);
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

  const float epsilon =
    std::get<props::GateUpEpsilon>(gate_up_props).get();
  const unsigned int M = (unsigned int)(to - from);
  const unsigned int K_in = input_step_dim.width();
  const unsigned int N_up = Uhidden_step_dim.width();
  const unsigned int N_gate = Ghidden_step_dim.width();

#if defined(ENABLE_OPENCL) && ENABLE_OPENCL == 1
  // M=1 decode fused path: rmsnorm + 2 FCs in a single GPU dispatch.
  // Mirrors the unittest nntrainer_fused_rmsnorm_gate_up.qwen3_4b_shapes
  // (FUSED 0.95 ms vs ~1.32 ms baseline at K_in=2560 N=9728).
  // Requires K_in <= 2560 and N partitions divisible by 64 (kernel
  // local-mem cache + WG=64 dispatch alignment).
  if (M == 1 &&
      input_.getMemoryData() && input_.getMemoryData()->isSVM() &&
      Uhidden_.getMemoryData() && Uhidden_.getMemoryData()->isSVM() &&
      Ghidden_.getMemoryData() && Ghidden_.getMemoryData()->isSVM() &&
      input_.getDataType() == ml::train::TensorDim::DataType::FP16 &&
      Uweight.getDataType() == ml::train::TensorDim::DataType::QINT4 &&
      Gweight.getDataType() == ml::train::TensorDim::DataType::QINT4 &&
      gamma.getDataType() == ml::train::TensorDim::DataType::FP32 &&
      K_in <= 2560 && (K_in % 4) == 0 &&
      (N_up % 64) == 0 && (N_gate % 64) == 0) {
    auto *cl_ctx = static_cast<nntrainer::ClContext *>(
      nntrainer::Engine::Global().getRegisteredContext("gpu"));
    if (cl_ctx) {
      // Entry fence: drain upstream addition layer's GPU writes to
      // input_step's SVM region.  PROFILE_LAYER_SYNC=1 normally hides
      // this race via a per-layer clFinish, but the explicit map keeps
      // the layer correct under SYNC=0 too.
      cl_ctx->command_queue_inst_.enqueueSVMMap(
        input_step.getData<char>(), input_step.bytes(), /*read_only=*/true);
    }

    auto *in_svm =
      reinterpret_cast<uint16_t *>(input_step.getData<_FP16>());
    auto *gamma_svm = gamma.getData<float>();
    auto *u_w = reinterpret_cast<uint16_t *>(Uweight.getData<char>());
    auto *u_s = Uweight.getScale<uint16_t>();
    auto *u_out = reinterpret_cast<uint16_t *>(Uhidden_step.getData<_FP16>());
    auto *g_w = reinterpret_cast<uint16_t *>(Gweight.getData<char>());
    auto *g_s = Gweight.getScale<uint16_t>();
    auto *g_out = reinterpret_cast<uint16_t *>(Ghidden_step.getData<_FP16>());

    // fused_rmsnorm_gate_up_cl signature: input, gamma, gate slot,
    // up slot, K, N_gate, N_up, eps.  We map our (Up, Gate) layer
    // ordering into the kernel's (gate, up) slot ordering: kernel
    // produces 2 outputs, which we connect downstream as
    // gate_up(0)=up_result, gate_up(1)=gate_result.
    if (nntrainer::fused_rmsnorm_gate_up_cl(
          in_svm, gamma_svm,
          /*gate_slot*/ u_w, u_s, u_out,
          /*up_slot*/   g_w, g_s, g_out,
          K_in, N_up, N_gate, epsilon)) {
      return;
    }
    // Helper rejected (rare); fall through to the unfused path below.
  }
#endif

  // M>1 prefill fallback: do CPU NEON rmsnorm into a temporary SVM
  // buffer, then dispatch the existing batched dot which routes
  // through HalfTensor::dot M>1 (per-weight gemm_delegate_fp16_cl
  // with GpuImagePool cache).  This preserves the legacy prefill
  // perf since the rmsnorm step matches RMSNormLayer's CPU NEON path
  // and the FCs go through the same delegate kernels they used to.
  // Produces a temporary stack-allocated tensor for the normalized
  // input -- only on prefill (per-token rare).
  nntrainer::Tensor norm_in_step(input_step_dim);
  norm_in_step.allocate();
  {
    const _FP16 *in_ptr = input_step.getData<_FP16>();
    _FP16 *out_ptr = norm_in_step.getData<_FP16>();
    const float *gamma_ptr = gamma.getData<float>();
    const std::size_t H_rows = (std::size_t)input_step_dim.batch() *
                                input_step_dim.channel() *
                                input_step_dim.height();
    nntrainer::rmsnorm_fused_fp16(in_ptr, out_ptr, gamma_ptr, H_rows, K_in,
                                   epsilon);
  }
  std::vector<nntrainer::Tensor *> Weights({&Uweight, &Gweight});
  std::vector<nntrainer::Tensor *> Outputs({&Uhidden_step, &Ghidden_step});
  norm_in_step.dot(Weights, Outputs);
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
