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

// weight_idx[0]=gamma is registered first so the bundle byte order
// matches the legacy ffn_norm.gamma | ffn_up.weight | ffn_gate.weight
// layout.  Output index convention unchanged: gate_up(0)=up,
// gate_up(1)=gate (swiglu reads them as "(1),(0)").
enum GateUpWeightIdx { Gamma, UpWeight, GateWeight };
enum GateUpParams { Up, Gate };

GateUpLayer::GateUpLayer() :
  LayerImpl(),
  gate_up_props(props::UpUnit(), props::GateUnit(), props::GateUpEpsilon()) {
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

  /** RMSNorm gamma (registered FIRST to preserve legacy bundle byte
   *  order: ffn_norm.gamma | ffn_up.weight | ffn_gate.weight).
   *  Force fp32 to match RMSNormLayer storage (rms_norm.cpp explains
   *  why fp16 gamma corrupts via raw byte read of the fp32 file).
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
  // M=1 decode: rmsnorm + 2 FCs in a single GPU dispatch.  Mirrors the
  // unittest nntrainer_fused_rmsnorm_gate_up.qwen3_4b_shapes (FUSED
  // 0.95 ms vs ~1.32 ms baseline at K_in=2560 N=9728).
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
      // Drain upstream addition layer's GPU writes before the kernel
      // reads input_step's SVM region.
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

    if (nntrainer::fused_rmsnorm_gate_up_cl(
          in_svm, gamma_svm,
          /*kernel gate slot*/ u_w, u_s, u_out,
          /*kernel up slot*/   g_w, g_s, g_out,
          K_in, N_up, N_gate, epsilon)) {
      return;
    }
    // Helper rejected; fall through.
  }
#endif

  // M>1 prefill fallback: do CPU NEON rmsnorm INTO the layer's Up output
  // tensor (Uhidden_step) -- it's the only SVM-backed scratch we have
  // access to and its width (N_up=9728) >= K_in (2560).  Then dispatch
  // the two FCs in order:
  //   1) Gate FC  reads Uhidden_step (rmsnorm result), writes Ghidden_step
  //      -- Uhidden_step is left intact for the next call.
  //   2) Up FC    reads Uhidden_step, writes Uhidden_step (overwrite).
  //      dotQInteger M>1 stages input to ClBufferManager::getSVMInput()
  //      svm_in scratch via a CPU loop BEFORE the kernel runs, so the
  //      same-buffer in/out aliasing is safe -- the kernel never reads
  //      from the output buffer directly.
  {
    const _FP16 *in_ptr = input_step.getData<_FP16>();
    _FP16 *scratch_ptr = Uhidden_step.getData<_FP16>();
    const float *gamma_ptr = gamma.getData<float>();
    const std::size_t H_rows = (std::size_t)input_step_dim.batch() *
                                input_step_dim.channel() *
                                input_step_dim.height();
    rmsnorm_fused_fp16(in_ptr, scratch_ptr, gamma_ptr, H_rows, K_in,
                       epsilon);
  }
  // Uhidden_step now holds the rmsnorm result (shape: M x K_in_padded).
  // It's allocated for M x N_up which is larger; only the first
  // M x K_in elements are written; the rest is undefined but unread.
  // Construct a (M x K_in) view into Uhidden_step for the FC inputs.
  nntrainer::TensorDim norm_view_dim = input_step_dim;
  nntrainer::Tensor norm_in =
    Uhidden_step.getSharedDataTensor(norm_view_dim, 0, true);

  // Gate FC first (Uhidden_step intact for the second FC).
  norm_in.dot(Gweight, Ghidden_step);
  // Up FC second (overwrites the scratch view; safe because dotQInteger
  // M>1 stages input via getSVMInput() before the kernel runs).
  norm_in.dot(Uweight, Uhidden_step);
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
