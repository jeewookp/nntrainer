// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file   custom_rms_norm.cpp
 * @date   19 July 2023
 * @brief  Implementation of custom RMS normalization function
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cpu_backend.h>
#include <reshaped_rms_norm.h>

#if defined(ENABLE_OPENCL) && ENABLE_OPENCL == 1
#include <cl_context.h>
#include <engine.h>
#endif

namespace causallm {

namespace {

struct ReshapedRMSNormProfile {
  std::atomic<uint64_t> calls{0};
  std::atomic<uint64_t> ns{0};
  std::atomic<uint64_t> ns_svm_map{0};
  std::atomic<uint64_t> ns_alloc_in{0};
  std::atomic<uint64_t> ns_fp16_to_fp32{0};
  std::atomic<uint64_t> ns_alloc_out{0};
  std::atomic<uint64_t> ns_neon{0};
  std::atomic<uint64_t> ns_gamma_mul{0};
  std::atomic<uint64_t> ns_fp32_to_fp16{0};

  ~ReshapedRMSNormProfile() {
    const uint64_t c = calls.load();
    if (c == 0)
      return;
    const uint64_t t = ns.load();
    const double T = t / 1.0e6;
    auto pct = [&](uint64_t v) {
      return t == 0 ? 0.0 : (v / 1.0e6) / T * 100.0;
    };
    std::fprintf(stderr,
                 "[PROFILE ReshapedRMSNormLayer prefill (M>1)] "
                 "total=%.2f ms calls=%llu avg=%.3f ms\n",
                 T, (unsigned long long)c, T / static_cast<double>(c));
    std::fprintf(stderr,
                 "  svm_map      : %8.2f ms (%5.1f%%)\n",
                 ns_svm_map / 1.0e6, pct(ns_svm_map));
    std::fprintf(stderr,
                 "  alloc_in     : %8.2f ms (%5.1f%%)\n",
                 ns_alloc_in / 1.0e6, pct(ns_alloc_in));
    std::fprintf(stderr,
                 "  fp16->fp32   : %8.2f ms (%5.1f%%)\n",
                 ns_fp16_to_fp32 / 1.0e6, pct(ns_fp16_to_fp32));
    std::fprintf(stderr,
                 "  alloc_out    : %8.2f ms (%5.1f%%)\n",
                 ns_alloc_out / 1.0e6, pct(ns_alloc_out));
    std::fprintf(stderr,
                 "  neon_rms     : %8.2f ms (%5.1f%%)  "
                 "[rms_norm_wrt_width_fp16_intrinsic]\n",
                 ns_neon / 1.0e6, pct(ns_neon));
    std::fprintf(stderr,
                 "  gamma_mul    : %8.2f ms (%5.1f%%)\n",
                 ns_gamma_mul / 1.0e6, pct(ns_gamma_mul));
    std::fprintf(stderr,
                 "  fp32->fp16   : %8.2f ms (%5.1f%%)\n",
                 ns_fp32_to_fp16 / 1.0e6, pct(ns_fp32_to_fp16));
  }
};

ReshapedRMSNormProfile g_reshaped_rms_norm_profile;

inline uint64_t now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
           std::chrono::steady_clock::now().time_since_epoch())
    .count();
}

} // namespace

static constexpr size_t SINGLE_INOUT_IDX = 0;

void ReshapedRMSNormLayer::finalize(nntrainer::InitLayerContext &context) {
  std::vector<nntrainer::TensorDim> dim = context.getInputDimensions();
  context.setOutputDimensions(dim);
  feature_size = std::get<props::FeatureSize>(rms_props);

  NNTR_THROW_IF(dim[0].width() % feature_size != 0, std::invalid_argument)
    << "feature size must be a divisor of width";

  // Force gamma to FP32 regardless of activation dtype. See RMSNormLayer::
  // finalize for the rationale -- TensorBase::read() is a raw byte copy
  // that does not convert from the file's fp32 layout to fp16, so the
  // loader silently produces an alternating [0x0000, high16_of_fp32]
  // pattern when we request gamma as fp16.
  nntrainer::TensorDim gamma_dim(
    1, 1, 1, feature_size,
    nntrainer::TensorDim::TensorType(
      context.getFormat(), nntrainer::TensorDim::DataType::FP32));
  wt_idx[RMSParams::gamma] = context.requestWeight(
    gamma_dim, nntrainer::props::InitializerInfo::Enum::NONE,
    nntrainer::WeightRegularizer::NONE, 1.0f, 0.0f, "gamma", false);
}

void ReshapedRMSNormLayer::forwarding(nntrainer::RunLayerContext &context,
                                      bool training) {}

void ReshapedRMSNormLayer::incremental_forwarding(
  nntrainer::RunLayerContext &context, unsigned int from, unsigned int to,
  bool training) {
  const bool profile_this_call = (to - from) > 1;
  const uint64_t t_layer_start = profile_this_call ? now_ns() : 0;

  auto &epsilon = std::get<nntrainer::props::Epsilon>(rms_props).get();

  nntrainer::Tensor &in = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &out = context.getOutput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &gamma = context.getWeight(wt_idx[RMSParams::gamma]);

#if defined(ENABLE_OPENCL) && ENABLE_OPENCL == 1
  // This layer runs purely on the CPU (NEON rms_norm_wrt_width_*_intrinsic).
  // When input lives in the tensor_pool's SVM region, the upstream GPU
  // gemm may still have its image2d_to_svm write in flight with only a
  // non-blocking SVMMap enqueued. Drain the OpenCL queue here with a
  // blocking SVMMap so the CPU reads that follow see coherent data.
  const uint64_t t_svm = profile_this_call ? now_ns() : 0;
  if (in.getMemoryData() && in.getMemoryData()->isSVM()) {
    auto *cl_ctx = static_cast<nntrainer::ClContext *>(
      nntrainer::Engine::Global().getRegisteredContext("gpu"));
    if (cl_ctx) {
      cl_ctx->command_queue_inst_.enqueueSVMMap(
        in.getData<char>(), in.bytes(), /*read_only=*/true);
    }
  }
  if (profile_this_call)
    g_reshaped_rms_norm_profile.ns_svm_map += now_ns() - t_svm;
#endif

  ml::train::TensorDim in_dim = in.getDim();
  ml::train::TensorDim out_dim = out.getDim();

  ml::train::TensorDim in_step_dim = in_dim;
  ml::train::TensorDim out_step_dim = out_dim;

  unsigned int _from = from;

  in_step_dim.batch(1);
  in_step_dim.height(to - from);
  out_step_dim.batch(1);
  out_step_dim.height(to - from);

  // set reshaped dim to (1, 1, -1, feature_size)
  ml::train::TensorDim step_reshaped_dim = in_step_dim;

  step_reshaped_dim.width(feature_size);
  step_reshaped_dim.height(in_step_dim.height() *
                           (in_dim.width() / feature_size));

  unsigned int b_size = in_dim.batch();

  for (unsigned int b = 0; b < b_size; ++b) {
    nntrainer::Tensor in_step =
      in.getSharedDataTensor(in_step_dim, b * in_dim.getFeatureLen(), true);
    nntrainer::Tensor out_step =
      out.getSharedDataTensor(out_step_dim, b * out_dim.getFeatureLen(), true);

    // reshape in_step
    // reshape out_step
    in_step.reshape(step_reshaped_dim);
    out_step.reshape(step_reshaped_dim);

    if (in_step.getDataType() == ml::train::TensorDim::DataType::FP32) {
      ///@todo rms_norm_wrt_width_something() should be refactored to
      /// nntrainer::Tensor operation.
#ifdef ENABLE_FP16
      nntrainer::rms_norm_wrt_width_fp16_intrinsic(
        in_step.getData<float>(), out_step.getData<float>(),
        in_step.getDim().height(), in_step.getDim().width(), epsilon);
#else
      nntrainer::rms_norm_wrt_width_fp32_intrinsic(
        in_step.getData<float>(), out_step.getData<float>(),
        in_step.getDim().height(), in_step.getDim().width(), epsilon);
#endif
      // gamma is fp32 (forced in finalize), out_step is fp32 here.
      out_step.multiply_i(gamma);
    } else if (in_step.getDataType() ==
               ml::train::TensorDim::DataType::FP16) {
      // Mixed precision: the `_FP16` template specialization of
      // rms_norm_wrt_width_fp16_intrinsic is NYI in the fallback
      // (arm_compute_backend_fp16.cpp:404 just delegates to
      // __fallback_rms_norm_wrt_width_fp16_intrinsic which throws).
      // Only the `float*` variant has a real NEON implementation, so
      // we stage through a fp32 temp like RMSNormLayer does. Q/K norm
      // width here is small (head_dim, e.g. 128 for Qwen3-4B) so the
      // temp allocation is cheap.
      nntrainer::TensorDim fp32_dim = in_step.getDim();
      fp32_dim.setDataType(ml::train::TensorDim::DataType::FP32);

      const uint64_t t_a_in = profile_this_call ? now_ns() : 0;
      nntrainer::Tensor in_fp32(fp32_dim, /*alloc_now=*/true);
      if (profile_this_call)
        g_reshaped_rms_norm_profile.ns_alloc_in += now_ns() - t_a_in;

      const uint64_t t_cast_in = profile_this_call ? now_ns() : 0;
      in_fp32.copyData(in_step); // fp16 -> fp32
      if (profile_this_call)
        g_reshaped_rms_norm_profile.ns_fp16_to_fp32 += now_ns() - t_cast_in;

      const uint64_t t_a_out = profile_this_call ? now_ns() : 0;
      nntrainer::Tensor out_fp32(fp32_dim, /*alloc_now=*/true);
      if (profile_this_call)
        g_reshaped_rms_norm_profile.ns_alloc_out += now_ns() - t_a_out;

      const uint64_t t_neon = profile_this_call ? now_ns() : 0;
#ifdef ENABLE_FP16
      nntrainer::rms_norm_wrt_width_fp16_intrinsic(
        in_fp32.getData<float>(), out_fp32.getData<float>(),
        in_step.getDim().height(), in_step.getDim().width(), epsilon);
#else
      nntrainer::rms_norm_wrt_width_fp32_intrinsic(
        in_fp32.getData<float>(), out_fp32.getData<float>(),
        in_step.getDim().height(), in_step.getDim().width(), epsilon);
#endif
      if (profile_this_call)
        g_reshaped_rms_norm_profile.ns_neon += now_ns() - t_neon;

      const uint64_t t_gamma = profile_this_call ? now_ns() : 0;
      // gamma is fp32 (forced in finalize) -- multiply in the fp32 temp
      // before converting back to fp16.
      out_fp32.multiply_i(gamma);
      if (profile_this_call)
        g_reshaped_rms_norm_profile.ns_gamma_mul += now_ns() - t_gamma;

      const uint64_t t_cast_out = profile_this_call ? now_ns() : 0;
      out_step.copyData(out_fp32); // fp32 -> fp16
      if (profile_this_call)
        g_reshaped_rms_norm_profile.ns_fp32_to_fp16 += now_ns() - t_cast_out;
    } else {
      throw std::invalid_argument(
        "Error: not yet implemented for this data type");
    }

    // reshape again out_step
    out_step.reshape(out_step_dim);

#ifdef DEBUG
    std::cout << context.getName() << " \n input:" << in_step
              << "output:" << out_step << "gamma:" << gamma << std::endl;
#endif
  }

  if (profile_this_call) {
    g_reshaped_rms_norm_profile.ns += now_ns() - t_layer_start;
    g_reshaped_rms_norm_profile.calls++;
  }
}

void ReshapedRMSNormLayer::updateTensorsByInputDimensions(
  nntrainer::RunLayerContext &context,
  std::vector<nntrainer::TensorDim> input_dimensions) {
  context.updateInput(SINGLE_INOUT_IDX, input_dimensions[0]);
  context.updateOutput(SINGLE_INOUT_IDX, input_dimensions[0]);
}

void ReshapedRMSNormLayer::calcDerivative(nntrainer::RunLayerContext &context) {
  std::throw_with_nested(std::runtime_error("Training is not supported yet."));
}

#ifdef PLUGGABLE

nntrainer::Layer *create_rms_norm_layer() {
  auto layer = new ReshapedRMSNormLayer();
  return layer;
}

void destroy_rms_norm_layer(nntrainer::Layer *layer) { delete layer; }

extern "C" {
nntrainer::LayerPluggable ml_train_layer_pluggable{create_rms_norm_layer,
                                                   destroy_rms_norm_layer};
}

#endif

} // namespace causallm
