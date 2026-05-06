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
#include <iostream>
#include <unordered_map>

#include <profile_gate.h>

#include "rms_norm.h"
#include "rmsnorm_fused_fp16.h"

#if defined(ENABLE_OPENCL) && ENABLE_OPENCL == 1
#include <blas_kernels.h>
#include <cl_context.h>
#include <engine.h>
#include <gpu_image_pool.h>
#endif

namespace causallm {

namespace {

// Per-layer cumulative prefill wall-clock profiler. Only counts calls
// where (to - from) > 1 so decode (M = 1) overhead is NOT mixed into
// the prefill-bottleneck breakdown. Prints a one-shot summary at
// process exit via the global's destructor.
struct RMSNormProfile {
  std::atomic<uint64_t> calls{0};
  std::atomic<uint64_t> ns{0};
  // Stage 1b sub-timers for FP16 activation path (where prefill lives).
  // Each bucket is cumulative nanoseconds across all prefill calls.
  std::atomic<uint64_t> ns_svm_map{0};   // SVMMap barrier (upstream GPU drain)
  std::atomic<uint64_t> ns_fused{0};     // rmsnorm_fused_fp16 (NEON)
  std::atomic<uint64_t> ns_publish{0};   // svm_to_image2d_publish

  ~RMSNormProfile() {
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
                 "[PROFILE RMSNormLayer prefill (M>1)] total=%.2f ms "
                 "calls=%llu avg=%.3f ms\n",
                 T, (unsigned long long)c, T / static_cast<double>(c));
    std::fprintf(stderr,
                 "  svm_map      : %8.2f ms (%5.1f%%)  "
                 "[upstream GPU drain]\n",
                 ns_svm_map / 1.0e6, pct(ns_svm_map));
    std::fprintf(stderr,
                 "  fused        : %8.2f ms (%5.1f%%)  "
                 "[rmsnorm_fused_fp16 NEON]\n",
                 ns_fused / 1.0e6, pct(ns_fused));
    std::fprintf(stderr,
                 "  publish      : %8.2f ms (%5.1f%%)  "
                 "[svm_to_image2d_publish]\n",
                 ns_publish / 1.0e6, pct(ns_publish));
  }
};

RMSNormProfile g_rms_norm_profile;

inline uint64_t now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
           std::chrono::steady_clock::now().time_since_epoch())
    .count();
}

} // namespace

static constexpr size_t SINGLE_INOUT_IDX = 0;

void RMSNormLayer::finalize(nntrainer::InitLayerContext &context) {
  std::vector<nntrainer::TensorDim> dim = context.getInputDimensions();
  context.setOutputDimensions(dim);

  // Force gamma to FP32 regardless of activation dtype. TensorBase::read()
  // is a raw byte copy that does NOT perform any dtype conversion from the
  // model file, so if the file stores gamma as fp32 (e.g. the Qwen3-4B
  // "...-fp32-arm.bin" checkpoint) and we request gamma as fp16 (via
  // packed=false + activation_dtype=FP16), the loader copies only the
  // first half of the fp32 bytes and reinterprets them as fp16 -- which
  // produces an alternating [0x0000, high16_of_fp32] pattern with every
  // even slot zero. That corruption propagates through the first
  // multiply_i(gamma) and turns every FC input into garbage.
  //
  // Keep gamma fp32 and do the per-channel scaling in the fp32 temp
  // inside incremental_forwarding before converting back to fp16 (see
  // the FP16 branch below). For the FP32 activation mode this is a no-op
  // change since activation_dtype == FP32 already.
  nntrainer::TensorDim gamma_dim(
    1, 1, 1, dim[0].width(),
    nntrainer::TensorDim::TensorType(
      context.getFormat(), nntrainer::TensorDim::DataType::FP32));
  wt_idx[RMSParams::gamma] = context.requestWeight(
    gamma_dim, nntrainer::props::InitializerInfo::Enum::NONE,
    nntrainer::WeightRegularizer::NONE, 1.0f, 0.0f, "gamma", false);
}

void RMSNormLayer::forwarding(nntrainer::RunLayerContext &context,
                              bool training) {}

void RMSNormLayer::incremental_forwarding(nntrainer::RunLayerContext &context,
                                          unsigned int from, unsigned int to,
                                          bool training) {
  const bool profile_this_call = (to - from) > 1;
  const uint64_t t_layer_start = profile_this_call ? now_ns() : 0;

  auto &epsilon = std::get<nntrainer::props::Epsilon>(rms_props).get();

  nntrainer::Tensor &in = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &out = context.getOutput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &gamma = context.getWeight(wt_idx[RMSParams::gamma]);

#if defined(ENABLE_OPENCL) && ENABLE_OPENCL == 1
  // The drain skip (NNTRAINER_RMSNORM_NO_DRAIN=1) only applies when we
  // route to GPU image2d (same-queue serialisation handles ordering).
  // For the CPU NEON path the drain is required so the NEON code reads
  // coherent data. Compute the route gate up-front, mirroring the same
  // logic used inside the per-step loop below for the actual dispatch.
  //
  // Decode-NEON gate: NNTRAINER_RMSNORM_DECODE_NEON=1 forces the CPU
  // NEON path even when NNTRAINER_RMSNORM_GPU=1, but only when this is
  // a single-token (M==1) decode step. The image2d_cl path's per-call
  // overhead (4 kernel dispatches + blocking fence) wipes the GPU
  // compute win for H_rows==1; CPU NEON ends up faster
  // (reshaped_rms_norm at 0.45 ms/call vs image2d_cl at 1.02 ms/call
  // measured on the same Qwen3-4B decode shape).
  static const bool s_rmsnorm_no_drain =
    std::getenv("NNTRAINER_RMSNORM_NO_DRAIN") != nullptr;
  static const bool s_rmsnorm_gpu_for_drain =
    std::getenv("NNTRAINER_RMSNORM_GPU") != nullptr;
  static const bool s_rmsnorm_decode_neon_for_drain =
    std::getenv("NNTRAINER_RMSNORM_DECODE_NEON") != nullptr;
  const bool drain_step_is_decode_h1 = ((to - from) == 1);
  const bool drain_route_will_be_gpu =
    s_rmsnorm_gpu_for_drain &&
    in.getDataType() == ml::train::TensorDim::DataType::FP16 &&
    in.getMemoryData() && in.getMemoryData()->isSVM() &&
    !(s_rmsnorm_decode_neon_for_drain && drain_step_is_decode_h1);
  const bool skip_drain = s_rmsnorm_no_drain && drain_route_will_be_gpu;
  const uint64_t t_svm = profile_this_call ? now_ns() : 0;
  if (!skip_drain && in.getMemoryData() && in.getMemoryData()->isSVM()) {
    auto *cl_ctx = static_cast<nntrainer::ClContext *>(
      nntrainer::Engine::Global().getRegisteredContext("gpu"));
    if (cl_ctx) {
      cl_ctx->command_queue_inst_.enqueueSVMMap(
        in.getData<char>(), in.bytes(), /*read_only=*/true);
    }
  }
  if (profile_this_call)
    g_rms_norm_profile.ns_svm_map += now_ns() - t_svm;
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
      // gamma is fp32 here too, standard fp32 * fp32 multiply.
      out_step.multiply_i(gamma);
    } else if (in_step.getDataType() ==
               ml::train::TensorDim::DataType::FP16) {
      // Stage 1b: fused fp16-in fp16-out RMSNorm with fp32 accumulator
      // and fp32-pinned gamma. Replaces the prior fp16 -> fp32 copy ->
      // Tensor chain (multiply/average/add/inv_sqrt/multiply) -> gamma
      // multiply_i -> fp32 -> fp16 copy round-trip, which was measured
      // at ~295 ms of this layer's 318 ms prefill cost.
      const uint64_t t_fused = profile_this_call ? now_ns() : 0;
      const _FP16 *in_ptr = in_step.getData<_FP16>();
      _FP16 *out_ptr = out_step.getData<_FP16>();
      const float *gamma_ptr = gamma.getData<float>();
      const ml::train::TensorDim sd = in_step.getDim();
      const std::size_t H_rows =
        (std::size_t)sd.batch() * sd.channel() * sd.height();
      const std::size_t W = sd.width();
#if defined(ENABLE_OPENCL) && ENABLE_OPENCL == 1
      // NNTRAINER_RMSNORM_GPU=1 dispatches rmsnorm_image2d_v2 on the
      // GPU queue instead of the NEON CPU loop. See blas_kernels.cpp
      // for semantics. Defaults off so existing CPU path stays active.
      static const bool s_rmsnorm_gpu =
        std::getenv("NNTRAINER_RMSNORM_GPU") != nullptr;
      static bool s_logged_variant = false;
      if (!s_logged_variant) {
        std::fprintf(stderr,
                     "[rms_norm] NNTRAINER_RMSNORM_GPU=%s -> %s\n",
                     std::getenv("NNTRAINER_RMSNORM_GPU")
                       ? std::getenv("NNTRAINER_RMSNORM_GPU")
                       : "(unset)",
                     s_rmsnorm_gpu ? "GPU image2d path"
                                   : "NEON fused path");
        s_logged_variant = true;
      }
      // For decode (H_rows == 1) the image2d_cl path's per-call cost
      // is dominated by 4 kernel dispatches (svm_to_image2d input
      // reformat + rmsnorm_image2d_v2 + image2d_to_svm output readback
      // + svm_to_image2d_publish to pool) plus a blocking SVMMap fence
      // at the end. Measured ~1.02 ms/call * 2336 calls = 2380 ms /
      // 35% of decode wall, vs the same shape via CPU NEON
      // (rmsnorm_fused_fp16) at ~0.45 ms/call as seen in
      // reshaped_rms_norm (which is purely NEON). For prefill the
      // image2d cost amortises over many rows so it stays useful, but
      // for M==1 decode it's pure overhead. Force NEON for H_rows==1
      // even when NNTRAINER_RMSNORM_GPU=1 -- gated on
      // NNTRAINER_RMSNORM_DECODE_NEON=1 so the fix is opt-in until
      // measured.
      static const bool s_rmsnorm_decode_neon =
        std::getenv("NNTRAINER_RMSNORM_DECODE_NEON") != nullptr;
      const bool route_gpu = s_rmsnorm_gpu && in_step.getMemoryData() &&
                             in_step.getMemoryData()->isSVM() &&
                             out_step.getMemoryData() &&
                             out_step.getMemoryData()->isSVM() && (W % 4) == 0 &&
                             !(s_rmsnorm_decode_neon && H_rows == 1);
      if (route_gpu) {
        nntrainer::rmsnorm_image2d_cl(
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
        g_rms_norm_profile.ns_fused += now_ns() - t_fused;
    } else {
      throw std::invalid_argument(
        "Error: not yet implemented for this data type");
    }

#ifdef DEBUG
    std::cout << context.getName() << " \n input:" << in_step
              << "output:" << out_step << "gamma:" << gamma << std::endl;
#endif
  }

#if defined(ENABLE_OPENCL) && ENABLE_OPENCL == 1
  // Phase A publish: RMSNorm's output has just been written to SVM by the
  // CPU path above. If we also convert it to an image2d and register the
  // image2d in GpuImagePool, the immediate next gemm_delegate call
  // (q_proj / k_proj / v_proj for rmsnorm_1, gate_proj / up_proj for
  // rmsnorm_2) picks it up and skips its own svm_to_image2d reformat.
  // Adds one image_reformat::svm_to_image2d dispatch per RMSNorm
  // (73/prefill), but saves ~5 per-decoder-layer reformats (180/prefill),
  // net positive.
  //
  // The current RMSNormLayer output shape is (B, 1, H, W) with
  // hidden_dim = W. Pool image2d layout: width = B*H (total positions),
  // height = W/4 (RGBA half slices). This matches gemm_delegate's
  // src image expectation.
  const uint64_t t_pub = profile_this_call ? now_ns() : 0;
  // Skip the separate publish when the GPU path already put out_img in
  // GpuImagePool as part of rmsnorm_image2d_cl.
  static const bool s_rmsnorm_gpu_publish_skip =
    std::getenv("NNTRAINER_RMSNORM_GPU") != nullptr;
  if (!s_rmsnorm_gpu_publish_skip &&
      out.getMemoryData() && out.getMemoryData()->isSVM() &&
      out.getDataType() == ml::train::TensorDim::DataType::FP16 &&
      (out.width() % 4) == 0) {
    // Use the step size (rows we actually just wrote). The tensor is
    // allocated for init_seq_len but prefill only normalises
    // (to - from) rows — the full-height path would copy junk rows
    // and the downstream gemm's shape check would fail.
    const int step_M =
      (int)out.batch() * (int)out.channel() * (int)(to - from);
    nntrainer::svm_to_image2d_publish(
      out.getData<char>(), step_M, (unsigned int)out.width());
  }
  if (profile_this_call)
    g_rms_norm_profile.ns_publish += now_ns() - t_pub;
#endif

  if (profile_this_call) {
    g_rms_norm_profile.ns += now_ns() - t_layer_start;
    g_rms_norm_profile.calls++;
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
