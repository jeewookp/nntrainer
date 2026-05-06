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
#include <atomic>
#include <chrono>
#include <cstdio>

#include <blas_kernels.h>
#include <cl_context.h>
#include <engine.h>
#endif

namespace nntrainer {

static constexpr size_t SINGLE_INOUT_IDX = 0;

#if defined(ENABLE_OPENCL) && ENABLE_OPENCL == 1
namespace {
// Decode-path profile for AdditionLayer::incremental_forwarding fast
// path (the upstream AdditionLayer::incremental_forwarding, NOT
// AdditionLayerCL -- the SVM fp16 fast path here is what actually
// fires in production: 2 inputs, fp16, both SVM, batch=1 -> route to
// add2_fp16_svm_cl GPU kernel + optional exit drain).
//
// PROFILE NetworkGraph reported addition takes 28.8% of decode
// (2090 / 7251 ms, 2304 calls = 2 residual adds * 36 layers * 32
// tokens, 0.91 ms/call) -- 100x bigger than the actual fp16 add
// compute. Suspect: the per-call enqueueSVMMap exit drain blocks on
// upstream layer GPU work, attributing the wait to addition.
//
// Splits each call into:
//   ns_kernel  : add2_fp16_svm_cl (kernel arg setup + DispatchCommand,
//                async, NO sync inside)
//   ns_drain   : enqueueSVMMap(hidden_, blocking=true) exit drain
//                (THIS is the queue-flushing blocker)
//   ns_misc    : type checks, isSVM checks, function entry/exit
struct AdditionDecodeProfile {
  std::atomic<uint64_t> calls_fast{0};
  std::atomic<uint64_t> calls_fallback{0};
  std::atomic<uint64_t> ns_kernel{0};
  std::atomic<uint64_t> ns_drain{0};
  std::atomic<uint64_t> ns_misc{0};

  ~AdditionDecodeProfile() {
    const uint64_t cf = calls_fast.load();
    const uint64_t cb = calls_fallback.load();
    if (cf == 0 && cb == 0) return;
    const uint64_t kernel = ns_kernel.load();
    const uint64_t drain = ns_drain.load();
    const uint64_t misc = ns_misc.load();
    const uint64_t total = kernel + drain + misc;
    auto ms = [](uint64_t v) -> double { return v / 1.0e6; };
    auto pct = [&](uint64_t v) -> double {
      return total == 0 ? 0.0 : (double)v / (double)total * 100.0;
    };
    std::fprintf(stderr,
                 "\n[PROFILE AdditionLayer decode] total=%.2f ms "
                 "fast=%llu fallback=%llu avg_fast=%.3f us\n",
                 ms(total), (unsigned long long)cf, (unsigned long long)cb,
                 cf == 0 ? 0.0 : (double)total / (double)cf / 1000.0);
    std::fprintf(stderr,
                 "  kernel : %8.2f ms (%5.1f%%)  [add2_fp16_svm_cl async]\n",
                 ms(kernel), pct(kernel));
    std::fprintf(stderr,
                 "  drain  : %8.2f ms (%5.1f%%)  "
                 "[enqueueSVMMap(hidden_, blocking=true) exit -- queue "
                 "flush, blames upstream stalls on addition]\n",
                 ms(drain), pct(drain));
    std::fprintf(stderr,
                 "  misc   : %8.2f ms (%5.1f%%)  [type checks + entry/exit]\n",
                 ms(misc), pct(misc));
  }
};
AdditionDecodeProfile g_addition_decode_profile;

inline uint64_t addition_now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
           std::chrono::steady_clock::now().time_since_epoch())
    .count();
}
} // namespace
#endif

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
      const uint64_t t_kern0 = addition_now_ns();
      nntrainer::add2_fp16_svm_cl(in0.getData<char>(),
                                  in1.getData<char>(),
                                  hidden_.getData<char>(),
                                  step_total);
      const uint64_t t_kern1 = addition_now_ns();
      g_addition_decode_profile.ns_kernel.fetch_add(t_kern1 - t_kern0,
                                                    std::memory_order_relaxed);
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
      // Phase B.2 (decode addition optimization): the per-call exit
      // SVMMap drain accounted for 99.5% of addition's wall (3244 ms
      // / 2376 calls = 1373 us each) per the [PROFILE AdditionLayer
      // decode] probe. The drain blocks until ALL queued GPU work
      // finishes, so addition was paying for upstream layers'
      // (mha_core, FCs) GPU latency. Downstream consumer of `hidden_`
      // is rms_norm -- if it's the GPU image2d path
      // (NNTRAINER_RMSNORM_GPU=1) same-queue ordering is automatic
      // per OpenCL spec. NNTRAINER_ADDITION_NO_DRAIN=1 skips the
      // explicit drain. If that turns the output garbage we know a
      // specific addition->X boundary needs the fix at the consumer's
      // entry instead.
      static const bool s_no_drain =
        std::getenv("NNTRAINER_ADDITION_NO_DRAIN") != nullptr;
      if (s_zerocopy && !s_layer_sync && !s_no_drain) {
        auto *cl_ctx = static_cast<ClContext *>(
          Engine::Global().getRegisteredContext("gpu"));
        if (cl_ctx) {
          const uint64_t t_drain0 = addition_now_ns();
          cl_ctx->command_queue_inst_.enqueueSVMMap(
            hidden_.getData<char>(), hidden_.bytes(), /*read_only=*/true);
          const uint64_t t_drain1 = addition_now_ns();
          g_addition_decode_profile.ns_drain.fetch_add(
            t_drain1 - t_drain0, std::memory_order_relaxed);
        }
      }
      g_addition_decode_profile.calls_fast.fetch_add(
        1, std::memory_order_relaxed);
      return;
    }
  }
  g_addition_decode_profile.calls_fallback.fetch_add(
    1, std::memory_order_relaxed);

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
