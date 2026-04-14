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
#include <cmath>
#include <cstdio>
#include <iostream>

#include "rms_norm.h"

namespace causallm {

static constexpr size_t SINGLE_INOUT_IDX = 0;

void RMSNormLayer::finalize(nntrainer::InitLayerContext &context) {
  std::vector<nntrainer::TensorDim> dim = context.getInputDimensions();
  context.setOutputDimensions(dim);
  nntrainer::TensorDim gamma_dim(
    1, 1, 1, dim[0].width(),
    nntrainer::TensorDim::TensorType(context.getFormat(),
                                     context.getWeightDataType()));
  wt_idx[RMSParams::gamma] = context.requestWeight(
    gamma_dim, nntrainer::props::InitializerInfo::Enum::NONE,
    nntrainer::WeightRegularizer::NONE, 1.0f, 0.0f, "gamma", false);
}

void RMSNormLayer::forwarding(nntrainer::RunLayerContext &context,
                              bool training) {}

void RMSNormLayer::incremental_forwarding(nntrainer::RunLayerContext &context,
                                          unsigned int from, unsigned int to,
                                          bool training) {
  auto &epsilon = std::get<nntrainer::props::Epsilon>(rms_props).get();

  nntrainer::Tensor &in = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &out = context.getOutput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &gamma = context.getWeight(wt_idx[RMSParams::gamma]);

  ml::train::TensorDim in_dim = in.getDim();
  ml::train::TensorDim out_dim = out.getDim();

  ml::train::TensorDim in_step_dim = in_dim;
  ml::train::TensorDim out_step_dim = out_dim;

  unsigned int _from = from;

  in_step_dim.batch(1);
  in_step_dim.height(to - from);
  out_step_dim.batch(1);
  out_step_dim.height(to - from);

  unsigned int b_size = in_dim.batch();

  for (unsigned int b = 0; b < b_size; ++b) {
    nntrainer::Tensor in_step =
      in.getSharedDataTensor(in_step_dim, b * in_dim.getFeatureLen(), true);
    nntrainer::Tensor out_step =
      out.getSharedDataTensor(out_step_dim, b * out_dim.getFeatureLen(), true);

    if (in_step.getDataType() == ml::train::TensorDim::DataType::FP32) {
      auto t = in_step.multiply(in_step).average(3).add(epsilon);
      t.inv_sqrt_i();
      in_step.multiply(t, out_step);
    } else if (in_step.getDataType() ==
               ml::train::TensorDim::DataType::FP16) {
      // Mixed precision: fp16 activations are prone to overflow when
      // we accumulate x*x over the hidden dimension (K in [2560..9728]
      // for Qwen3-4B with per-element values easily >1). Stage the
      // variance computation through a fp32 temp, then write the
      // normalized result back as fp16.
      nntrainer::TensorDim fp32_dim = in_step.getDim();
      fp32_dim.setDataType(ml::train::TensorDim::DataType::FP32);

      nntrainer::Tensor in_fp32(fp32_dim, /*alloc_now=*/true);
      in_fp32.copyData(in_step); // fp16 -> fp32 via FloatTensor::copyData

      // DIAG: dump first 16 fp16 values of in_step (= previous layer's
      // output) to see whether the alternating [0, X, 0, X, ...] pattern
      // observed downstream in HalfTensor::dotQInteger is already present
      // at rms_norm's input. If so, the bug is upstream (embedding); if
      // not, the bug is in this layer's fp32<->fp16 round trip.
      static std::atomic<int> diag_calls{0};
      const int call_no = diag_calls.fetch_add(1, std::memory_order_relaxed);
      if (call_no < 2) {
#ifdef ENABLE_FP16
        auto *in_hex =
          reinterpret_cast<const uint16_t *>(in_step.getData<_FP16>());
        std::fprintf(stderr,
                     "[DIAG rms_norm #%d name=%s] in_step_fp16[0..15]=",
                     call_no, context.getName().c_str());
        for (int i = 0; i < 16; ++i) {
          std::fprintf(stderr, "%04x%s", in_hex[i], i + 1 < 16 ? " " : "");
        }
        std::fprintf(stderr, "\n");
        // After converting to fp32 via copyData, dump the fp32 temp too.
        auto *in_f32_ptr = in_fp32.getData<float>();
        std::fprintf(stderr,
                     "[DIAG rms_norm #%d name=%s] in_fp32[0..7]=", call_no,
                     context.getName().c_str());
        for (int i = 0; i < 8; ++i) {
          std::fprintf(stderr, "%+.4g%s",
                       static_cast<double>(in_f32_ptr[i]), i + 1 < 8 ? " " : "");
        }
        std::fprintf(stderr, "\n");
        std::fflush(stderr);
#endif
      }

      auto t = in_fp32.multiply(in_fp32).average(3).add(epsilon);
      t.inv_sqrt_i();

      nntrainer::Tensor out_fp32(fp32_dim, /*alloc_now=*/true);
      in_fp32.multiply(t, out_fp32);

      out_step.copyData(out_fp32); // fp32 -> fp16 via HalfTensor::copyData

      // DIAG: dump first 16 fp16 values of out_step AFTER copyData so we
      // can see whether the fp32 -> fp16 conversion introduced the
      // alternating pattern.
      if (call_no < 2) {
#ifdef ENABLE_FP16
        auto *out_hex =
          reinterpret_cast<const uint16_t *>(out_step.getData<_FP16>());
        std::fprintf(stderr,
                     "[DIAG rms_norm #%d name=%s] out_step_fp16[0..15]=",
                     call_no, context.getName().c_str());
        for (int i = 0; i < 16; ++i) {
          std::fprintf(stderr, "%04x%s", out_hex[i], i + 1 < 16 ? " " : "");
        }
        std::fprintf(stderr, "\n");
        std::fflush(stderr);
#endif
      }
    } else {
      throw std::invalid_argument(
        "Error: not yet implemented for this data type");
    }
    out_step.multiply_i(gamma);

#ifdef DEBUG
    std::cout << context.getName() << " \n input:" << in_step
              << "output:" << out_step << "gamma:" << gamma << std::endl;
#endif
  }
}

void RMSNormLayer::updateTensorsByInputDimensions(
  nntrainer::RunLayerContext &context,
  std::vector<nntrainer::TensorDim> input_dimensions) {
  context.updateInput(SINGLE_INOUT_IDX, input_dimensions[0]);
  context.updateOutput(SINGLE_INOUT_IDX, input_dimensions[0]);
}

void RMSNormLayer::calcDerivative(nntrainer::RunLayerContext &context) {
  std::throw_with_nested(std::runtime_error("Training is not supported yet."));
}

#ifdef PLUGGABLE

nntrainer::Layer *create_rms_norm_layer() {
  auto layer = new RMSNormLayer();
  return layer;
}

void destroy_rms_norm_layer(nntrainer::Layer *layer) { delete layer; }

extern "C" {
nntrainer::LayerPluggable ml_train_layer_pluggable{create_rms_norm_layer,
                                                   destroy_rms_norm_layer};
}

#endif

} // namespace causallm
