// SPDX-License-Identifier: Apache-2.0
/**
 * @file   qkv_norm_layer.cpp
 * @brief  Fused QKV + Q_norm + K_norm layer implementation.
 */

#include <qkv_norm_layer.h>

#include "rmsnorm_fused_fp16.h"
#include <engine.h>
#include <layer_context.h>
#include <nntrainer_error.h>
#include <node_exporter.h>

#if defined(ENABLE_OPENCL) && ENABLE_OPENCL == 1
#include <blas_kernels.h>
#endif

namespace causallm {

static constexpr size_t SINGLE_INOUT_IDX = 0;
enum QKVNormParams { Q = 0, K = 1, V = 2 };
enum WeightSlots {
  W_Q = 0,
  W_GAMMA_Q = 1,
  W_K = 2,
  W_GAMMA_K = 3,
  W_V = 4,
};

QKVNormLayer::QKVNormLayer() :
  LayerImpl(),
  qkv_norm_props(qkvnorm_props::QUnit(), qkvnorm_props::KUnit(),
                  qkvnorm_props::VUnit(), qkvnorm_props::FeatureSize(),
                  qkvnorm_props::Epsilon()) {
  weight_idx.fill(std::numeric_limits<unsigned>::max());
}

void QKVNormLayer::finalize(nntrainer::InitLayerContext &context) {
  NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
    << "QKVNormLayer takes only one input";

  auto &weight_regularizer =
    std::get<nntrainer::props::WeightRegularizer>(*layer_impl_props);
  auto &weight_regularizer_constant =
    std::get<nntrainer::props::WeightRegularizerConstant>(*layer_impl_props);
  auto weight_initializer = nntrainer::props::InitializerInfo::Enum::NONE;
  auto &weight_decay =
    std::get<nntrainer::props::WeightDecay>(*layer_impl_props);

  const auto &q_unit = std::get<qkvnorm_props::QUnit>(qkv_norm_props).get();
  const auto &k_unit = std::get<qkvnorm_props::KUnit>(qkv_norm_props).get();
  const auto &v_unit = std::get<qkvnorm_props::VUnit>(qkv_norm_props).get();
  const auto &feature_size =
    std::get<qkvnorm_props::FeatureSize>(qkv_norm_props).get();

  bool is_nchw = (context.getFormat() == nntrainer::Tformat::NCHW);
  auto const &in_dim = context.getInputDimensions()[0];

  context.setEffDimFlagInputDimension(0, 0b1001);
  context.setDynDimFlagInputDimension(0, 0b1000);

  std::vector<nntrainer::TensorDim> output_dims(3);
  output_dims[Q] = in_dim;
  is_nchw ? output_dims[Q].width(q_unit) : output_dims[Q].channel(q_unit);
  output_dims[Q].setTensorType(
    {context.getFormat(), context.getActivationDataType()});

  output_dims[K] = in_dim;
  is_nchw ? output_dims[K].width(k_unit) : output_dims[K].channel(k_unit);
  output_dims[K].setTensorType(
    {context.getFormat(), context.getActivationDataType()});

  output_dims[V] = in_dim;
  is_nchw ? output_dims[V].width(v_unit) : output_dims[V].channel(v_unit);
  output_dims[V].setTensorType(
    {context.getFormat(), context.getActivationDataType()});

  context.setOutputDimensions(output_dims);

  // Q weight (QINT4)
  nntrainer::TensorDim q_weight_dim(
    1, is_nchw ? 1 : q_unit, is_nchw ? in_dim.width() : 1,
    is_nchw ? q_unit : in_dim.channel(),
    nntrainer::TensorDim::TensorType(context.getFormat(),
                                      context.getWeightDataType()),
    is_nchw ? 0b0011 : 0b0101);
  weight_idx[W_Q] = context.requestWeight(
    q_weight_dim, weight_initializer, weight_regularizer,
    weight_regularizer_constant, weight_decay, "qweight", true);

  // gamma_q (FP32, feature_size = head_dim)
  // Mirrors ReshapedRMSNormLayer's gamma registration so bundle bytes
  // line up with the original [Q_FC, Q_norm, K_FC, K_norm, V_FC] order.
  nntrainer::TensorDim gamma_q_dim(
    1, 1, 1, feature_size,
    nntrainer::TensorDim::TensorType(context.getFormat(),
                                      nntrainer::TensorDim::DataType::FP32));
  weight_idx[W_GAMMA_Q] = context.requestWeight(
    gamma_q_dim, nntrainer::props::InitializerInfo::Enum::NONE,
    nntrainer::WeightRegularizer::NONE, 1.0f, 0.0f, "gamma_q",
    /*trainable=*/false);

  // K weight
  nntrainer::TensorDim k_weight_dim = q_weight_dim;
  is_nchw ? k_weight_dim.width(k_unit) : k_weight_dim.channel(k_unit);
  weight_idx[W_K] = context.requestWeight(
    k_weight_dim, weight_initializer, weight_regularizer,
    weight_regularizer_constant, weight_decay, "kweight", true);

  // gamma_k
  weight_idx[W_GAMMA_K] = context.requestWeight(
    gamma_q_dim, nntrainer::props::InitializerInfo::Enum::NONE,
    nntrainer::WeightRegularizer::NONE, 1.0f, 0.0f, "gamma_k",
    /*trainable=*/false);

  // V weight
  nntrainer::TensorDim v_weight_dim = q_weight_dim;
  is_nchw ? v_weight_dim.width(v_unit) : v_weight_dim.channel(v_unit);
  weight_idx[W_V] = context.requestWeight(
    v_weight_dim, weight_initializer, weight_regularizer,
    weight_regularizer_constant, weight_decay, "vweight", true);
}

void QKVNormLayer::exportTo(nntrainer::Exporter &exporter,
                             const ml::train::ExportMethods &method) const {
  LayerImpl::exportTo(exporter, method);
  exporter.saveResult(qkv_norm_props, method, this);
}

void QKVNormLayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, qkv_norm_props);
  LayerImpl::setProperty(remain_props);
}

void QKVNormLayer::forwarding(nntrainer::RunLayerContext &context,
                               bool training) {
  return;
}

void QKVNormLayer::incremental_forwarding(
  nntrainer::RunLayerContext &context, unsigned int from, unsigned int to,
  bool training) {
  nntrainer::Tensor &Qweight = context.getWeight(weight_idx[W_Q]);
  nntrainer::Tensor &gamma_q = context.getWeight(weight_idx[W_GAMMA_Q]);
  nntrainer::Tensor &Kweight = context.getWeight(weight_idx[W_K]);
  nntrainer::Tensor &gamma_k = context.getWeight(weight_idx[W_GAMMA_K]);
  nntrainer::Tensor &Vweight = context.getWeight(weight_idx[W_V]);

  nntrainer::Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &Qhidden_ = context.getOutput(Q);
  nntrainer::Tensor &Khidden_ = context.getOutput(K);
  nntrainer::Tensor &Vhidden_ = context.getOutput(V);

  const auto &feature_size =
    std::get<qkvnorm_props::FeatureSize>(qkv_norm_props).get();
  const auto &epsilon = std::get<qkvnorm_props::Epsilon>(qkv_norm_props).get();

  nntrainer::TensorDim input_dim = input_.getDim();
  nntrainer::TensorDim input_step_dim = input_dim;
  input_step_dim.batch(1);
  input_step_dim.height(to - from);

  nntrainer::Tensor input_step =
    input_.getSharedDataTensor(input_step_dim, 0, true);

  nntrainer::TensorDim Qhidden_step_dim = Qhidden_.getDim();
  Qhidden_step_dim.batch(1);
  Qhidden_step_dim.height(to - from);
  nntrainer::Tensor Qhidden_step =
    Qhidden_.getSharedDataTensor(Qhidden_step_dim, 0, true);

  nntrainer::TensorDim Khidden_step_dim = Khidden_.getDim();
  Khidden_step_dim.batch(1);
  Khidden_step_dim.height(to - from);
  nntrainer::Tensor Khidden_step =
    Khidden_.getSharedDataTensor(Khidden_step_dim, 0, true);

  nntrainer::TensorDim Vhidden_step_dim = Vhidden_.getDim();
  Vhidden_step_dim.batch(1);
  Vhidden_step_dim.height(to - from);
  nntrainer::Tensor Vhidden_step =
    Vhidden_.getSharedDataTensor(Vhidden_step_dim, 0, true);

  // Step 1: fused QKV projection via HalfTensor::dot batched path.
  // For QINT4 weights + SVM activations + M=1, this routes to
  // fused_gemv_int4_image2d_v2_cl, the same path gate_up_layer uses
  // (kernel runs at ~64 GB/s = BW-bound).
  std::vector<nntrainer::Tensor *> Weights({&Qweight, &Kweight, &Vweight});
  std::vector<nntrainer::Tensor *> Outputs(
    {&Qhidden_step, &Khidden_step, &Vhidden_step});
  input_step.dot(Weights, Outputs);

  // Step 2: in-place per-head rmsnorm on Q and K with their gammas.
  // Reshape to (rows, head_dim) where rows = (to-from) * num_heads.
  // Q rows = (to-from) * (q_unit / head_dim)
  // K rows = (to-from) * (k_unit / head_dim)
  // V passes through with no norm.
  const unsigned int M = to - from;
  const unsigned int q_unit =
    std::get<qkvnorm_props::QUnit>(qkv_norm_props).get();
  const unsigned int k_unit =
    std::get<qkvnorm_props::KUnit>(qkv_norm_props).get();
  const unsigned int q_rows = M * (q_unit / feature_size);
  const unsigned int k_rows = M * (k_unit / feature_size);

#if defined(ENABLE_OPENCL) && ENABLE_OPENCL == 1
  const bool q_svm =
    Qhidden_step.getMemoryData() && Qhidden_step.getMemoryData()->isSVM();
  const bool k_svm =
    Khidden_step.getMemoryData() && Khidden_step.getMemoryData()->isSVM();
  if (q_svm && Qhidden_step.getDataType() ==
                  nntrainer::TensorDim::DataType::FP16) {
    nntrainer::rmsnorm_fp16_svm_cl(
      Qhidden_step.getData<_FP16>(), Qhidden_step.getData<_FP16>(),
      gamma_q.getData<float>(), q_rows, feature_size, epsilon);
  } else {
    // CPU fallback for fp32 or non-SVM.
    Qhidden_step.reshape(nntrainer::TensorDim(
      {1, 1, q_rows, feature_size}, Qhidden_step.getTensorType()));
    if (Qhidden_step.getDataType() ==
        nntrainer::TensorDim::DataType::FP16) {
      // Use the same fused kernel even for non-SVM — fp16 NEON.
      causallm::rmsnorm_fused_fp16(
        Qhidden_step.getData<_FP16>(), Qhidden_step.getData<_FP16>(),
        gamma_q.getData<float>(), q_rows, feature_size, epsilon);
    }
  }
  if (k_svm && Khidden_step.getDataType() ==
                  nntrainer::TensorDim::DataType::FP16) {
    nntrainer::rmsnorm_fp16_svm_cl(
      Khidden_step.getData<_FP16>(), Khidden_step.getData<_FP16>(),
      gamma_k.getData<float>(), k_rows, feature_size, epsilon);
  } else {
    Khidden_step.reshape(nntrainer::TensorDim(
      {1, 1, k_rows, feature_size}, Khidden_step.getTensorType()));
    if (Khidden_step.getDataType() ==
        nntrainer::TensorDim::DataType::FP16) {
      causallm::rmsnorm_fused_fp16(
        Khidden_step.getData<_FP16>(), Khidden_step.getData<_FP16>(),
        gamma_k.getData<float>(), k_rows, feature_size, epsilon);
    }
  }
#else
  // Pure CPU build path -- TODO if needed.
  (void)q_rows;
  (void)k_rows;
#endif
}

void QKVNormLayer::calcDerivative(nntrainer::RunLayerContext &context) {
  return;
}
void QKVNormLayer::calcGradient(nntrainer::RunLayerContext &context) {
  return;
}

void QKVNormLayer::updateTensorsByInputDimensions(
  nntrainer::RunLayerContext &context,
  std::vector<nntrainer::TensorDim> input_dimensions) {
  ml::train::TensorDim input_dim = context.getInput(SINGLE_INOUT_IDX).getDim();
  ml::train::TensorDim Qoutput_dim = context.getOutput(Q).getDim();
  ml::train::TensorDim Koutput_dim = context.getOutput(K).getDim();
  ml::train::TensorDim Voutput_dim = context.getOutput(V).getDim();

  input_dim.height(input_dimensions[0].height());
  Qoutput_dim.height(input_dimensions[0].height());
  Koutput_dim.height(input_dimensions[0].height());
  Voutput_dim.height(input_dimensions[0].height());

  context.updateInput(SINGLE_INOUT_IDX, input_dim);
  context.updateOutput(Q, Qoutput_dim);
  context.updateOutput(K, Koutput_dim);
  context.updateOutput(V, Voutput_dim);
}

} // namespace causallm
