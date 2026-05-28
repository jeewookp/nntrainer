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

// Raw-pointer host RMSNorm fallback. Used when the GPU dispatch
// returns false (env disabled or precondition fails). Operates
// directly on getData() pointers, no Tensor::multiply / add /
// inv_sqrt_i calls — those crash on gpu-context-allocated tensors.
static void rms_norm_host_fp32(const float *in, const float *gamma,
                               float *out, float eps, unsigned int rows,
                               unsigned int cols) {
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

    // Two outputs need to be produced:
    //   (a) The fused int8/scale/zp/rs pool entries that v8c FC's
    //       consumer path (NNTR_V8C_CONSUME_FUSED_RMSQ=1) picks up
    //       — this is the GPU compute path.
    //   (b) An fp32 host buffer at `out_p` for any consumer that
    //       reads the un-quantized output (debug/profile, layers
    //       that didn't get the fused consumer treatment yet).
    //
    // For (a) call fused_rmsnorm_quant_resident_fp32. For (b)
    // compute on host directly — calling rmsnorm_resident_fp32 +
    // readback_backing_to_host would double the GPU work without
    // saving anything since the fused kernel already did the
    // bandwidth-heavy reduction. Host fp32 norm on K=1024 × ~282
    // rows is ~5 ms total which is cheap.
    if (b_size == 1) {
      nntrainer::fused_rmsnorm_quant_resident_fp32(
        in, gamma, epsilon, H, W, out.getName(), (const void *)out_p);
    }
    rms_norm_host_fp32(in_p, gamma_p, out_p, epsilon, H, W);
  }
}

} // namespace causallm
