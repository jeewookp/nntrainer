// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Yash Singh <yash.singh@samsung.com>
 *
 * @file   addition_layer_cl.cpp
 * @date   28 May 2024
 * @see    https://github.com/nntrainer/nntrainer
 * @author Yash Singh yash.singh@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief	 This is Addition Layer Class Class for Neural Network with OpenCl
 * implementation
 */

#include <addition_layer_cl.h>
#include <blas_kernel_interface.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <util_func.h>

#include <atomic>
#include <chrono>
#include <cstdio>

#include <layer_context.h>

namespace nntrainer {

static constexpr size_t SINGLE_INOUT_IDX = 0;

namespace {
// Decode-path profile for AdditionLayerCL::incremental_forwarding.
// Decode dispatches a residual addition twice per transformer layer
// (post-attention residual + post-MLP residual) so for Qwen3-4B:
//   2 * 36 layers * 32 tokens = 2304 calls per generation run.
// The PROFILE NetworkGraph block reported addition takes 28.8% of
// decode wall time (2090 / 7251 ms) at 0.91 ms/call -- 100x bigger
// than a 2560-elem fp16 add should be on this device. Likely culprit
// is addition_cl_internal taking the stale clEnqueueWriteBuffer +
// clEnqueueReadBuffer round-trip path instead of SVM zero-copy.
// This probe splits each call into:
//   ns_copy   : hidden_step.copy(input_step) on the FIRST input idx
//   ns_add    : add_i_cl(hidden_step, input_step) on subsequent idx
//   ns_misc   : everything else inside incremental_forwarding (loop
//               setup, getSharedDataTensor, etc.)
// Atomic so multi-threaded prefill won't corrupt the counters; decode
// is single-threaded but we keep the same pattern as
// HalfDotQInt4DecodeProfile in half_tensor.cpp for consistency.
struct AdditionDecodeProfile {
  std::atomic<uint64_t> calls{0};
  std::atomic<uint64_t> ns_copy{0};
  std::atomic<uint64_t> ns_add{0};
  std::atomic<uint64_t> ns_misc{0};

  ~AdditionDecodeProfile() {
    const uint64_t c = calls.load();
    if (c == 0)
      return;
    const uint64_t copy = ns_copy.load();
    const uint64_t add = ns_add.load();
    const uint64_t misc = ns_misc.load();
    const uint64_t total = copy + add + misc;
    auto ms = [](uint64_t v) -> double { return v / 1.0e6; };
    auto pct = [&](uint64_t v) -> double {
      return total == 0 ? 0.0 : (double)v / (double)total * 100.0;
    };
    std::fprintf(stderr,
                 "\n[PROFILE AdditionLayerCL decode (M==1)] "
                 "total=%.2f ms calls=%llu avg=%.3f us/call\n",
                 ms(total), (unsigned long long)c,
                 c == 0 ? 0.0 : (double)total / (double)c / 1000.0);
    std::fprintf(stderr,
                 "  copy      : %8.2f ms (%5.1f%%)  "
                 "[hidden_step.copy(input_step) on first idx]\n",
                 ms(copy), pct(copy));
    std::fprintf(stderr,
                 "  add_i_cl  : %8.2f ms (%5.1f%%)  "
                 "[GPU add via addition_cl_internal -> WriteBuffer + "
                 "Enqueue + ReadBuffer round-trip on subsequent idx]\n",
                 ms(add), pct(add));
    std::fprintf(stderr,
                 "  misc      : %8.2f ms (%5.1f%%)  "
                 "[loop setup + getSharedDataTensor + layer wrapper]\n",
                 ms(misc), pct(misc));
  }
};
AdditionDecodeProfile g_addition_decode_profile;

inline uint64_t now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
           std::chrono::steady_clock::now().time_since_epoch())
    .count();
}
} // namespace

void AdditionLayerCL::finalize(InitLayerContext &context) {
  context.setOutputDimensions({context.getInputDimensions()[0]});
}

void AdditionLayerCL::forwarding(RunLayerContext &context, bool training) {
  Tensor &hidden_ = context.getOutput(SINGLE_INOUT_IDX);

  /** @todo check possibility for in-place of addition layer */
  for (unsigned int idx = 0; idx < context.getNumInputs(); ++idx) {
    const Tensor &input_ = context.getInput(idx);
    if (!idx) {
      hidden_.copy(input_);
    } else {
      add_i_cl(hidden_, input_);
    }
  }
}

void AdditionLayerCL::incremental_forwarding(RunLayerContext &context,
                                             unsigned int from, unsigned int to,
                                             bool training) {
  const uint64_t t_enter = now_ns();
  Tensor &hidden_ = context.getOutput(SINGLE_INOUT_IDX);
  TensorDim hidden_dim = hidden_.getDim();
  TensorDim hidden_step_dim = hidden_dim;

  if (from) {
    NNTR_THROW_IF(to - from != 1, std::invalid_argument)
      << "incremental step size is not 1";
    from = 0;
    to = 1;
  }

  hidden_step_dim.batch(1);
  hidden_step_dim.height(to - from);

  uint64_t local_copy = 0;
  uint64_t local_add = 0;

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
        const uint64_t t0 = now_ns();
        hidden_step.copy(input_step);
        local_copy += now_ns() - t0;
      } else {
        const uint64_t t0 = now_ns();
        add_i_cl(hidden_step, input_step);
        local_add += now_ns() - t0;
      }
    }
  }

  const uint64_t t_exit = now_ns();
  const uint64_t total = t_exit - t_enter;
  const uint64_t misc =
    total > (local_copy + local_add) ? total - local_copy - local_add : 0;
  g_addition_decode_profile.ns_copy.fetch_add(local_copy,
                                              std::memory_order_relaxed);
  g_addition_decode_profile.ns_add.fetch_add(local_add,
                                             std::memory_order_relaxed);
  g_addition_decode_profile.ns_misc.fetch_add(misc, std::memory_order_relaxed);
  g_addition_decode_profile.calls.fetch_add(1, std::memory_order_relaxed);
}

void AdditionLayerCL::calcDerivative(RunLayerContext &context) {

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

void AdditionLayerCL::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, add_props);
  if (!remain_props.empty()) {
    std::string msg = "[AdditionLayer] Unknown Layer Properties count " +
                      std::to_string(values.size());
    throw exception::not_supported(msg);
  }
}
} /* namespace nntrainer */
