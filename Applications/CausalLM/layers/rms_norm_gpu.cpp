// SPDX-License-Identifier: Apache-2.0
/**
 * @file   rms_norm_gpu.cpp
 * @date   29 May 2026
 * @brief  GPU-routed RMSNorm. See rms_norm_gpu.h for the rationale.
 */

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(__aarch64__) || defined(__ARM_NEON)
#include <arm_neon.h>
#define RMSNORM_GPU_HAVE_NEON 1
#else
#define RMSNORM_GPU_HAVE_NEON 0
#endif

#include "rms_norm_gpu.h"

#include <blas_kernel_interface.h>

namespace causallm {

static constexpr size_t SINGLE_INOUT_IDX = 0;

enum class RMSParamsGPU : unsigned int { GAMMA = 0 };

void RMSNormLayerGPU::finalize(nntrainer::InitLayerContext &context) {
  std::vector<nntrainer::TensorDim> dim = context.getInputDimensions();
  context.setOutputDimensions(dim);
  nntrainer::TensorDim gamma_dim(
    1, 1, 1, dim[0].width(),
    nntrainer::TensorDim::TensorType(context.getFormat(),
                                     context.getWeightDataType()));
  wt_idx[(unsigned int)RMSParamsGPU::GAMMA] = context.requestWeight(
    gamma_dim, nntrainer::props::InitializerInfo::Enum::NONE,
    nntrainer::WeightRegularizer::NONE, 1.0f, 0.0f, "gamma", false);
}

void RMSNormLayerGPU::updateTensorsByInputDimensions(
  nntrainer::RunLayerContext &context,
  std::vector<nntrainer::TensorDim> input_dimensions) {
  context.updateInput(SINGLE_INOUT_IDX, input_dimensions[0]);
  context.updateOutput(SINGLE_INOUT_IDX, input_dimensions[0]);
}

// Raw-pointer host RMSNorm. Used because Tensor::multiply / add_i /
// inv_sqrt_i crash on gpu-context-allocated tensors (same family as
// the AdditionLayerCL workaround). NEON-SIMD on aarch64 for ~10×
// throughput vs the scalar fallback; scalar code is correct for
// reference / non-ARM builds.
static void rms_norm_host_fp32(const float *in, const float *gamma,
                               float *out, float eps, unsigned int rows,
                               unsigned int cols) {
#if RMSNORM_GPU_HAVE_NEON
  for (unsigned int r = 0; r < rows; ++r) {
    const float *in_row = in + (size_t)r * cols;
    float *out_row = out + (size_t)r * cols;

    // Pass 1: sum-of-squares with 2-lane parallel accumulation.
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    unsigned int k = 0;
    for (; k + 8 <= cols; k += 8) {
      float32x4_t v0 = vld1q_f32(in_row + k);
      float32x4_t v1 = vld1q_f32(in_row + k + 4);
      acc0 = vmlaq_f32(acc0, v0, v0);
      acc1 = vmlaq_f32(acc1, v1, v1);
    }
    float sumsq = vaddvq_f32(vaddq_f32(acc0, acc1));
    for (; k < cols; ++k) sumsq += in_row[k] * in_row[k];

    const float inv_rms = 1.0f / std::sqrt(sumsq / (float)cols + eps);
    const float32x4_t inv_rms_v = vdupq_n_f32(inv_rms);

    // Pass 2: out[k] = in[k] * inv_rms * gamma[k].
    for (k = 0; k + 4 <= cols; k += 4) {
      float32x4_t v = vld1q_f32(in_row + k);
      float32x4_t g = vld1q_f32(gamma + k);
      vst1q_f32(out_row + k, vmulq_f32(vmulq_f32(v, inv_rms_v), g));
    }
    for (; k < cols; ++k) out_row[k] = in_row[k] * inv_rms * gamma[k];
  }
#else
  for (unsigned int r = 0; r < rows; ++r) {
    const float *in_row = in + (size_t)r * cols;
    float *out_row = out + (size_t)r * cols;
    double sumsq = 0.0;
    for (unsigned int k = 0; k < cols; ++k)
      sumsq += (double)in_row[k] * in_row[k];
    const float mean_sq = (float)(sumsq / cols);
    const float inv_rms = 1.0f / std::sqrt(mean_sq + eps);
    for (unsigned int k = 0; k < cols; ++k)
      out_row[k] = in_row[k] * inv_rms * gamma[k];
  }
#endif
}

void RMSNormLayerGPU::incremental_forwarding(
  nntrainer::RunLayerContext &context, unsigned int from, unsigned int to,
  bool training) {
  auto &epsilon = std::get<nntrainer::props::Epsilon>(rms_props).get();

  nntrainer::Tensor &in = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &out = context.getOutput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &gamma = context.getWeight(wt_idx[(unsigned int)RMSParamsGPU::GAMMA]);

  const ml::train::TensorDim in_dim = in.getDim();
  const ml::train::TensorDim out_dim = out.getDim();
  const unsigned int b_size = in_dim.batch();
  const unsigned int H = to - from;
  const unsigned int W = in_dim.width();

  // FP32 only for now (Qwen3 residual stream). FP16 fall-through to
  // the host computation as a safety net.
  const bool fp32 =
    in.getDataType() == ml::train::TensorDim::DataType::FP32 &&
    gamma.getDataType() == ml::train::TensorDim::DataType::FP32 &&
    out.getDataType() == ml::train::TensorDim::DataType::FP32;

  if (!fp32) {
    // Host fp32 fallback only handles fp32; for other dtypes raise.
    throw std::runtime_error(
      "RMSNormLayerGPU: only FP32 inputs supported in this build");
  }

  // Try the fused-rmsq path first (paper §3.6 #2): it writes int8 +
  // scale + zp + row_sum into pool backings keyed by
  // ptr:<out_host>:fused_{i8,scale,zp,rs}. The v8c FC consumer
  // (NNTR_V8C_CONSUME_FUSED_RMSQ=1) picks those up directly,
  // bypassing the rmsnorm→FC fp32 boundary. Falls back to plain
  // GPU rmsnorm or host rmsnorm depending on env.
  for (unsigned int b = 0; b < b_size; ++b) {
    // Sliced views: in and out are shared with the parent at offset
    // b * featureLen. Operate on the raw float* with explicit offsets
    // rather than calling Tensor::getSharedDataTensor (which would
    // return another gpu-context tensor and risk the same crashes).
    const size_t in_off =
      (size_t)b * in_dim.getFeatureLen();
    const size_t out_off =
      (size_t)b * out_dim.getFeatureLen();
    const float *in_p_root = in.getData<float>();
    float *out_p_root = out.getData<float>();
    const float *gamma_p = gamma.getData<float>();
    const float *in_p = in_p_root + in_off;
    float *out_p = out_p_root + out_off;

    // NEON SIMD host RMSNorm is the primary compute path because
    // measurement showed the fused-rmsq GPU dispatch is NET NEGATIVE
    // for TPS even when paired with the v8c FC consumer that skips
    // its own quant. The 56 dispatches × ~3 ms per forward outweigh
    // the saved per-FC quant time.
    //
    // The fused-rmsq dispatch is therefore OPT-IN via two envs:
    //   NNTR_FUSED_RMSQ=1          — base producer env (legacy)
    //   NNTR_RMSQ_GPU_DISPATCH=1   — engage from this GPU layer
    // PLUS H >= NNTR_RMSQ_GPU_MIN_H (default 8) for decode skip.
    //
    // The NEON host norm always runs because the v8c FC consumer
    // path is correct without the fused entries (it falls back to
    // its own act-quant). So host norm gives correctness; the
    // fused dispatch is now purely a perf experiment.
    static const bool dispatch_enabled =
      std::getenv("NNTR_RMSQ_GPU_DISPATCH") != nullptr;
    static const unsigned int min_h_for_gpu = []() {
      const char *e = std::getenv("NNTR_RMSQ_GPU_MIN_H");
      return e != nullptr ? (unsigned int)std::atoi(e) : 8u;
    }();
    if (dispatch_enabled && b_size == 1 && H >= min_h_for_gpu) {
      nntrainer::fused_rmsnorm_quant_resident_fp32(
        in, gamma, epsilon, H, W, out.getName(), (const void *)out_p);
    }
    rms_norm_host_fp32(in_p, gamma_p, out_p, epsilon, H, W);
  }
}

} // namespace causallm
