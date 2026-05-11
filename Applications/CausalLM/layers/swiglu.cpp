// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file   swiglu.cpp
 * @date   14 July 2023
 * @brief  Implementation of SwiGLU activation function
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>

#include <profile_gate.h>
#include <util_simd.h>

#include "swiglu.h"

#if defined(ENABLE_OPENCL) && ENABLE_OPENCL == 1
#include <blas_kernels.h>
#include <cl_context.h>
#include <engine.h>
#endif

namespace causallm {

namespace {

struct SwiGLUProfile {
  std::atomic<uint64_t> calls{0};
  std::atomic<uint64_t> ns{0};

  ~SwiGLUProfile() {
    const uint64_t c = calls.load();
    if (c == 0)
      return;
    if (nntrainer::prefill_profile_suppressed())
      return;
    const uint64_t t = ns.load();
    std::fprintf(stderr,
                 "[PROFILE SwiGLULayer prefill (M>1)] total=%.2f ms "
                 "calls=%llu avg=%.3f ms\n",
                 t / 1.0e6, (unsigned long long)c,
                 (t / 1.0e6) / static_cast<double>(c));
  }
};

SwiGLUProfile g_swiglu_profile;

inline uint64_t now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
           std::chrono::steady_clock::now().time_since_epoch())
    .count();
}

} // namespace

static constexpr size_t OUT_IDX = 0;
static constexpr size_t INPUT_IDX_1 = 0;
static constexpr size_t INPUT_IDX_2 = 1;

namespace ActivationOp {
/**
 * @brief activation function swiglu
 * @param x input
 * @retval swiglu(x)
 */
float swiglu(float x) { return x / (1 + nntrainer::exp_util(-x)); }
} // namespace ActivationOp

void SwiGLULayer::finalize(nntrainer::InitLayerContext &context) {
  context.setOutputDimensions({context.getInputDimensions()[0]});
}

void SwiGLULayer::forwarding(nntrainer::RunLayerContext &context,
                             bool training) {}

void SwiGLULayer::incremental_forwarding(nntrainer::RunLayerContext &context,
                                         unsigned int from, unsigned int to,
                                         bool training) {
  const bool profile_this_call = (to - from) > 1;
  const uint64_t t_layer_start = profile_this_call ? now_ns() : 0;

  nntrainer::Tensor &in1 = context.getInput(INPUT_IDX_1);
  nntrainer::Tensor &in2 = context.getInput(INPUT_IDX_2);
  nntrainer::Tensor &out = context.getOutput(OUT_IDX);

  unsigned int _from = from;

  int iter = to - from;

#if defined(ENABLE_OPENCL) && ENABLE_OPENCL == 1
  // Plan 3 Stage 1: read gate/up from cl_mem buffers written by GateUpLayer.
  // Always-on for decode (iter==1) + FP16. Matches the gate_up_layer
  // cl_mem output path (also always-on as of Stage 1).
  if (iter == 1 &&
      in1.getDataType() == ml::train::TensorDim::DataType::FP16 &&
      in2.getDataType() == ml::train::TensorDim::DataType::FP16 &&
      out.getDataType() == ml::train::TensorDim::DataType::FP16 &&
      out.getMemoryData() && out.getMemoryData()->isSVM()) {
    // SwiGLU's input convention: in1 = gate, in2 = up.
    // Match GateUpLayer's writes (gate_buf <- Gweight slot,
    // up_buf <- Uweight slot).
    const unsigned int N = (unsigned int)in1.width();
    cl_mem gate_buf = nntrainer::clmem_poc_gate_buf(N);
    cl_mem up_buf = nntrainer::clmem_poc_up_buf(N);
    if (gate_buf && up_buf) {
      const size_t step_total =
        (size_t)iter * (size_t)in1.channel() * (size_t)in1.width();
      if (nntrainer::swiglu_fp16_clmem_in_cl(gate_buf, up_buf,
                                              out.getData<_FP16>(),
                                              step_total)) {
        if ((out.width() % 4) == 0) {
          const int pub_M = (int)out.channel() * (int)iter;
          nntrainer::svm_to_image2d_publish(
            out.getData<char>(), pub_M, (unsigned int)out.width());
        }
        if (profile_this_call) {
          g_swiglu_profile.ns += now_ns() - t_layer_start;
          g_swiglu_profile.calls++;
        }
        return;
      }
    }
  }

  // Phase M pass-through: when GateUpLayer fused SwiGLU inline (env
  // NNTRAINER_GATEUP_SWIGLU_FUSED + decode M==1), in2 already
  // contains the final swiglu_out. Just copy it into our output via
  // the existing copy_cl_fp16 dispatch and skip the silu(g)*u math.
  static const bool s_gateup_swiglu_fused =
    std::getenv("NNTRAINER_GATEUP_SWIGLU_FUSED") != nullptr;
  if (s_gateup_swiglu_fused && iter == 1 &&
      in1.getDataType() == ml::train::TensorDim::DataType::FP16 &&
      in2.getDataType() == ml::train::TensorDim::DataType::FP16 &&
      out.getDataType() == ml::train::TensorDim::DataType::FP16 &&
      in2.getMemoryData() && in2.getMemoryData()->isSVM() &&
      out.getMemoryData() && out.getMemoryData()->isSVM()) {
    const size_t bytes = in2.bytes();
    if (out.bytes() == bytes &&
        nntrainer::svm_memcpy_fp16_cl(in2.getData<char>(),
                                       out.getData<char>(), bytes)) {
      // Publish swiglu_out to GpuImagePool so the down_proj pool-hits
      // (matching the legacy fast-path's behaviour at line ~131).
      if ((out.width() % 4) == 0) {
        const int pub_M = (int)out.channel() * (int)iter;
        nntrainer::svm_to_image2d_publish(
          out.getData<char>(), pub_M, (unsigned int)out.width());
      }
      if (profile_this_call) {
        g_swiglu_profile.ns += now_ns() - t_layer_start;
        g_swiglu_profile.calls++;
      }
      return;
    }
    // Fall through to legacy fast-path on helper failure.
  }

  // Phase B fast path: if all three tensors are fp16 SVM and the
  // flat element count is a safe multiple of the dispatch local size,
  // run swiglu_cl_fp16 on the GPU queue instead of the NEON CPU
  // path. This eliminates the blocking SVMMap fence that used to
  // drain the upstream gemm_delegate output, since the GPU kernel
  // reads the SVM directly in queue order.
  //
  // DEBUG toggle: NNTRAINER_DEBUG_CPU_SWIGLU=1 forces the CPU NEON
  // path (with a blocking SVMMap fence on both inputs) so we can
  // bisect whether the GPU dispatch is the garbage source.
  static const bool s_debug_cpu_swiglu =
    std::getenv("NNTRAINER_DEBUG_CPU_SWIGLU") != nullptr;
  if (!s_debug_cpu_swiglu &&
      in1.getDataType() == ml::train::TensorDim::DataType::FP16 &&
      in2.getDataType() == ml::train::TensorDim::DataType::FP16 &&
      out.getDataType() == ml::train::TensorDim::DataType::FP16 &&
      in1.getMemoryData() && in1.getMemoryData()->isSVM() &&
      in2.getMemoryData() && in2.getMemoryData()->isSVM() &&
      out.getMemoryData() && out.getMemoryData()->isSVM() &&
      in1.batch() == 1 && _from == 0) {
#ifdef ENABLE_FP16
    const size_t step_total =
      (size_t)iter * (size_t)in1.channel() * (size_t)in1.width();
    nntrainer::swiglu_fp16_svm_cl(
      in1.getData<_FP16>(), in2.getData<_FP16>(), out.getData<_FP16>(),
      step_total);
    // Phase B publish: SwiGLU output -> GpuImagePool so the
    // following down_proj pool-hits, dropping its per-gemm blocking
    // SVMMap + scalar staging copy. out shape here is
    // (1, C, iter, W); publish M = C*iter, K = W.
    if ((out.width() % 4) == 0) {
      const int pub_M =
        (int)out.channel() * (int)iter;
      nntrainer::svm_to_image2d_publish(
        out.getData<char>(), pub_M, (unsigned int)out.width());
    }
    if (profile_this_call) {
      g_swiglu_profile.ns += now_ns() - t_layer_start;
      g_swiglu_profile.calls++;
    }
    return;
#endif
  }

  if (s_debug_cpu_swiglu) {
    // CPU path needs the GPU queue drained first, since upstream
    // gate/up gemms no longer block-sync on their outputs.
    auto *cl_ctx = static_cast<nntrainer::ClContext *>(
      nntrainer::Engine::Global().getRegisteredContext("gpu"));
    if (cl_ctx) {
      auto map_if_svm = [&](nntrainer::Tensor &t) {
        if (t.getMemoryData() && t.getMemoryData()->isSVM()) {
          cl_ctx->command_queue_inst_.enqueueSVMMap(
            t.getData<char>(), t.bytes(), /*read_only=*/true);
        }
      };
      map_if_svm(in1);
      map_if_svm(in2);
    }
  }
#endif

  if (in1.getDataType() == ml::train::TensorDim::DataType::FP32) {
    for (unsigned int b = 0; b < in1.batch(); b++) {
      for (unsigned int c = 0; c < in1.channel(); c++) {
        for (unsigned int h = 0; h < iter; h++) {
          nntrainer::swiglu(in1.width(),
                            out.getData<float>() + out.getIndex(b, c, h, 0),
                            in1.getData<float>() + in1.getIndex(b, c, h, 0),
                            in2.getData<float>() + in2.getIndex(b, c, h, 0));
        }
      }
    }
  } else if (in1.getDataType() == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    for (unsigned int b = 0; b < in1.batch(); b++) {
      for (unsigned int c = 0; c < in1.channel(); c++) {
        for (unsigned int h = 0; h < iter; h++) {
          nntrainer::swiglu(in1.width(),
                            out.getData<_FP16>() + out.getIndex(b, c, h, 0),
                            in1.getData<_FP16>() + in1.getIndex(b, c, h, 0),
                            in2.getData<_FP16>() + in2.getIndex(b, c, h, 0));
        }
      }
    }
#else
    NNTR_THROW_IF(true, std::invalid_argument) << "enable-fp16 is not set!";
#endif
  }

  if (profile_this_call) {
    g_swiglu_profile.ns += now_ns() - t_layer_start;
    g_swiglu_profile.calls++;
  }
}

void SwiGLULayer::updateTensorsByInputDimensions(
  nntrainer::RunLayerContext &context,
  std::vector<nntrainer::TensorDim> input_dimensions) {
  ml::train::TensorDim input_dim1 = context.getInput(INPUT_IDX_1).getDim();
  ml::train::TensorDim input_dim2 = context.getInput(INPUT_IDX_2).getDim();
  ml::train::TensorDim output_dim = context.getOutput(OUT_IDX).getDim();

  input_dim1.height(input_dimensions[0].height());
  input_dim2.height(input_dimensions[0].height());
  output_dim.height(input_dimensions[0].height());

  context.updateInput(INPUT_IDX_1, input_dim1);
  context.updateInput(INPUT_IDX_2, input_dim2);
  context.updateOutput(OUT_IDX, output_dim);
}

void SwiGLULayer::calcDerivative(nntrainer::RunLayerContext &context) {
  // std::throw_with_nested(std::runtime_error("Training is not supported
  // yet."));
}

#ifdef PLUGGABLE

nntrainer::Layer *create_swiglu_layer() {
  auto layer = new SwiGLULayer();
  return layer;
}

void destroy_swiglu_layer(nntrainer::Layer *layer) { delete layer; }

extern "C" {
nntrainer::LayerPluggable ml_train_layer_pluggable{create_swiglu_layer,
                                                   destroy_swiglu_layer};
}

#endif

} // namespace causallm
