// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   addition_layer.cpp
 * @date   30 July 2020
 * @see    https://github.com/nntrainer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is Addition Layer Class for Neural Network
 *
 */

#include <addition_layer.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <util_func.h>

#include <layer_context.h>

#if defined(ENABLE_OPENCL) && ENABLE_OPENCL == 1
#include <blas_kernels.h>
#include <cl_context.h>
#include <engine.h>
#endif

namespace nntrainer {

static constexpr size_t SINGLE_INOUT_IDX = 0;

void AdditionLayer::finalize(InitLayerContext &context) {
  context.setOutputDimensions({context.getInputDimensions()[0]});
}

void AdditionLayer::forwarding(RunLayerContext &context, bool training) {
  Tensor &hidden_ = context.getOutput(SINGLE_INOUT_IDX);

  /** @todo check possibility for in-place of addition layer */
  for (unsigned int idx = 0; idx < context.getNumInputs(); ++idx) {
    const Tensor &input_ = context.getInput(idx);
    if (!idx) {
      hidden_.copy(input_);
    } else {
      hidden_.add_i(input_);
    }
  }
}

void AdditionLayer::incremental_forwarding(RunLayerContext &context,
                                           unsigned int from, unsigned int to,
                                           bool training) {
  Tensor &hidden_ = context.getOutput(SINGLE_INOUT_IDX);

#if defined(ENABLE_OPENCL) && ENABLE_OPENCL == 1
  // Phase B fast path: two-input SVM fp16 residual add goes to the
  // add2_fp16_svm GPU kernel. Keeps the residual on the OpenCL queue
  // so the upstream gemm's async writes are naturally serialised
  // before this add, and the downstream RMSNorm (CPU) fence is the
  // only place that has to drain. Much faster than the NEON
  // Tensor::copy + add_i path, which also had to block on the
  // preceding gemm output.
  if (context.getNumInputs() == 2 &&
      hidden_.getDataType() == ml::train::TensorDim::DataType::FP16 &&
      hidden_.getMemoryData() && hidden_.getMemoryData()->isSVM()) {
    const Tensor &in0 = context.getInput(0);
    const Tensor &in1 = context.getInput(1);
    if (in0.getDataType() == ml::train::TensorDim::DataType::FP16 &&
        in1.getDataType() == ml::train::TensorDim::DataType::FP16 &&
        in0.getMemoryData() && in0.getMemoryData()->isSVM() &&
        in1.getMemoryData() && in1.getMemoryData()->isSVM() &&
        hidden_.batch() == 1) {
      const size_t step_total =
        (size_t)hidden_.channel() * (size_t)(to - from) *
        (size_t)hidden_.width();
      nntrainer::add2_fp16_svm_cl(in0.getData<char>(),
                                  in1.getData<char>(),
                                  hidden_.getData<char>(),
                                  step_total);
      // Exit drain for sync=0 zero-copy mode: when
      // NNTRAINER_PROFILE_LAYER_SYNC is unset there is no per-layer
      // clFinish to flush the GPU writes done by add2_fp16_svm_cl.
      // The downstream rmsnorm has its own entry SVMMap drain so
      // synchronous CPU consumers are covered, but a downstream GPU
      // kernel reading `hidden_` via SetKernelSVMArguments on Adreno
      // coarse-grained SVM may see stale values without an explicit
      // kernel-boundary barrier.  Use enqueueSVMMap on `hidden_`
      // (blocking=true) so the queue drains and the buffer cache
      // becomes coherent before the next dispatch reads it.
      // Keep this off when PROFILE_LAYER_SYNC=1 -- redundant drain
      // and we want the clFinish to do the work for honest profiling.
      static const bool s_zerocopy =
        std::getenv("NNTRAINER_GEMV_ZEROCOPY") != nullptr;
      static const bool s_layer_sync =
        std::getenv("NNTRAINER_PROFILE_LAYER_SYNC") != nullptr;
      if (s_zerocopy && !s_layer_sync) {
        auto *cl_ctx = static_cast<ClContext *>(
          Engine::Global().getRegisteredContext("gpu"));
        if (cl_ctx) {
          cl_ctx->command_queue_inst_.enqueueSVMMap(
            hidden_.getData<char>(), hidden_.bytes(), /*read_only=*/true);
        }
      }
      return;
    }
  }

  {
    auto *cl_ctx = static_cast<ClContext *>(
      Engine::Global().getRegisteredContext("gpu"));
    if (cl_ctx) {
      auto map_if_svm = [&](const Tensor &t) {
        if (t.getMemoryData() && t.getMemoryData()->isSVM()) {
          cl_ctx->command_queue_inst_.enqueueSVMMap(
            const_cast<char *>(t.getData<char>()), t.bytes(),
            /*read_only=*/true);
        }
      };
      for (unsigned int idx = 0; idx < context.getNumInputs(); ++idx) {
        map_if_svm(context.getInput(idx));
      }
    }
  }
#endif

  TensorDim hidden_dim = hidden_.getDim();
  TensorDim hidden_step_dim = hidden_dim;

  hidden_step_dim.batch(1);
  hidden_step_dim.height(to - from);

  for (unsigned int b = 0; b < hidden_.batch(); ++b) {
    Tensor hidden_step = hidden_.getSharedDataTensor(
      hidden_step_dim, b * hidden_dim.getFeatureLen(), true);

    /** @todo check possibility for in-place of addition layer */
    for (unsigned int idx = 0; idx < context.getNumInputs(); ++idx) {
      const Tensor &input_ = context.getInput(idx);
      TensorDim input_dim = input_.getDim();

      TensorDim input_step_dim = input_dim;
      input_step_dim.batch(1);
      input_step_dim.height(to - from);

      Tensor input_step = input_.getSharedDataTensor(
        input_step_dim, b * input_dim.getFeatureLen(), true);
      if (!idx) {
        hidden_step.copy(input_step);
      } else {
        hidden_step.add_i(input_step);
      }
    }
  }
}

void AdditionLayer::calcDerivative(RunLayerContext &context) {

  for (unsigned int idx = 0; idx < context.getNumInputs(); ++idx) {
    /**
     * TODO: replace this with tensor assignment during optimization.
     * Tensor assignment needs to make sure that the previous connected layers
     * are not inplace
     */
    context.getOutgoingDerivative(idx).copy(
      context.getIncomingDerivative(SINGLE_INOUT_IDX));
  }
}

void AdditionLayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, add_props);
  if (!remain_props.empty()) {
    std::string msg = "[AdditionLayer] Unknown Layer Properties count " +
                      std::to_string(values.size());
    throw exception::not_supported(msg);
  }
}

void AdditionLayer::updateTensorsByInputDimensions(
  nntrainer::RunLayerContext &context,
  std::vector<nntrainer::TensorDim> input_dimensions) {
  for (size_t i = 0; i < context.getNumInputs(); ++i) {
    context.updateInput(i, input_dimensions[0]);
  }
  context.updateOutput(SINGLE_INOUT_IDX, input_dimensions[0]);
}

} /* namespace nntrainer */
