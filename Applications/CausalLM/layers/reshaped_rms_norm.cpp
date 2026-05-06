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
#include <profile_gate.h>
#include <reshaped_rms_norm.h>
#include "rmsnorm_fused_fp16.h"

#if defined(ENABLE_OPENCL) && ENABLE_OPENCL == 1
#include <blas_kernels.h>
#include <cl_context.h>
#include <engine.h>
#endif

namespace causallm {

namespace {

struct ReshapedRMSNormProfile {
  std::atomic<uint64_t> calls{0};
  std::atomic<uint64_t> ns{0};
  std::atomic<uint64_t> ns_svm_map{0};
  std::atomic<uint64_t> ns_fused{0};

  ~ReshapedRMSNormProfile() {
    const uint64_t c = calls.load();
    if (c == 0)
      return;
    if (nntrainer::prefill_profile_suppressed())
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
                 "  fused        : %8.2f ms (%5.1f%%)  "
                 "[rmsnorm_fused_fp16 NEON]\n",
                 ns_fused / 1.0e6, pct(ns_fused));
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
  // Default path runs CPU NEON (rms_norm_wrt_width_*_intrinsic) so we
  // drain the GPU queue first to make upstream Q/K projection writes
  // coherent with the host read. NNTRAINER_RESHAPED_RMSNORM_DECODE_SVM=1
  // routes the M==1 FP16 decode case through rmsnorm_fp16_svm_cl on
  // the same OpenCL queue (skipping the drain since same-queue
  // ordering covers it).
  static const bool s_reshaped_decode_svm =
    std::getenv("NNTRAINER_RESHAPED_RMSNORM_DECODE_SVM") != nullptr;
  const bool decode_h1 = ((to - from) == 1);
  const bool in_is_svm = in.getMemoryData() && in.getMemoryData()->isSVM();
  const bool out_is_svm =
    out.getMemoryData() && out.getMemoryData()->isSVM();
  const bool route_decode_svm =
    s_reshaped_decode_svm && decode_h1 && in_is_svm && out_is_svm &&
    in.getDataType() == ml::train::TensorDim::DataType::FP16;
  const uint64_t t_svm = profile_this_call ? now_ns() : 0;
  if (!route_decode_svm && in_is_svm) {
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
      // Stage 1b: fused fp16-in fp16-out RMSNorm + gamma.  Head-dim
      // (W=feature_size) is small (128 on Qwen3-4B) but we run the
      // normalisation H_rows = (height after reshape) times, which is
      // (to-from) * num_heads per call.  Prior round-trip spent ~148 ms
      // of 172 ms on fp32 temps + gamma multiply_i as a separate pass.
      const uint64_t t_fused = profile_this_call ? now_ns() : 0;
      const _FP16 *in_ptr = in_step.getData<_FP16>();
      _FP16 *out_ptr = out_step.getData<_FP16>();
      const float *gamma_ptr = gamma.getData<float>();
      const ml::train::TensorDim sd = in_step.getDim();
      const std::size_t H_rows =
        (std::size_t)sd.batch() * sd.channel() * sd.height();
      const std::size_t W = sd.width();
#if defined(ENABLE_OPENCL) && ENABLE_OPENCL == 1
      if (route_decode_svm) {
        // SVM-direct decode path: same kernel rmsnorm_fp16_svm
        // serves both rms_norm (W=2560) and reshaped_rms_norm
        // (W=head_dim=128, H_rows=num_heads). Single kernel
        // dispatch on the same OpenCL queue as the upstream Q/K
        // projection, no host drain, no image2d round-trip.
        nntrainer::rmsnorm_fp16_svm_cl(
          (void *)in_ptr, (void *)out_ptr, gamma_ptr,
          (unsigned int)H_rows, (unsigned int)W,
          static_cast<float>(epsilon));
      } else {
        rmsnorm_fused_fp16(in_ptr, out_ptr, gamma_ptr, H_rows, W,
                           static_cast<float>(epsilon));
      }
#else
      rmsnorm_fused_fp16(in_ptr, out_ptr, gamma_ptr, H_rows, W,
                         static_cast<float>(epsilon));
#endif
      if (profile_this_call)
        g_reshaped_rms_norm_profile.ns_fused += now_ns() - t_fused;
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
