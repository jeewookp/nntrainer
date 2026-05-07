// SPDX-License-Identifier: Apache-2.0
/**
 * @file   qkv_norm_layer.h
 * @date   May 2026
 * @brief  Fused QKV projection + Q_norm + K_norm for Qwen3-style models.
 *
 *         Replaces 3 separate fully_connected layers (q_proj, k_proj,
 *         v_proj) plus 2 reshape_rms_norm layers (Q_norm, K_norm) with a
 *         single layer. The 5 weights are registered in the same order as
 *         the original sequence appears in the bundle file:
 *           [0] q_weight       (QINT4)
 *           [1] gamma_q        (FP32, head_dim entries)
 *           [2] k_weight       (QINT4)
 *           [3] gamma_k        (FP32, head_dim entries)
 *           [4] v_weight       (QINT4)
 *         so positional bundle loading sees identical bytes to the
 *         legacy 3-FC + 2-norm sequence.
 *
 *         Forward dispatches the QKV projections via HalfTensor::dot's
 *         batched 3-weight path (same fused_gemv_int4_image2d path that
 *         gate_up_layer uses), then applies in-place per-head rmsnorm
 *         to Q and K with their respective gammas. V passes through.
 *
 *         Outputs:
 *           [0] Q_normed   (B, 1, M, q_unit)
 *           [1] K_normed   (B, 1, M, k_unit)
 *           [2] V          (B, 1, M, v_unit)
 */

#ifndef __QKV_NORM_LAYER_H__
#define __QKV_NORM_LAYER_H__
#ifdef __cplusplus

#pragma once
#ifdef _WIN32
#define WIN_EXPORT __declspec(dllexport)
#else
#define WIN_EXPORT
#endif

#include <common_properties.h>
#include <layer_impl.h>

namespace causallm {

namespace qkvnorm_props {

class QUnit : public nntrainer::PositiveIntegerProperty {
public:
  static constexpr const char *key = "q_unit";
  using prop_tag = nntrainer::uint_prop_tag;
};

class KUnit : public nntrainer::PositiveIntegerProperty {
public:
  static constexpr const char *key = "k_unit";
  using prop_tag = nntrainer::uint_prop_tag;
};

class VUnit : public nntrainer::PositiveIntegerProperty {
public:
  static constexpr const char *key = "v_unit";
  using prop_tag = nntrainer::uint_prop_tag;
};

class FeatureSize : public nntrainer::PositiveIntegerProperty {
public:
  static constexpr const char *key = "feature_size";  // = head_dim
  using prop_tag = nntrainer::uint_prop_tag;
};

class Epsilon : public nntrainer::Property<float> {
public:
  static constexpr const char *key = "epsilon";
  using prop_tag = nntrainer::float_prop_tag;
};

} // namespace qkvnorm_props

WIN_EXPORT class QKVNormLayer : public nntrainer::LayerImpl {
public:
  WIN_EXPORT QKVNormLayer();
  WIN_EXPORT ~QKVNormLayer() = default;
  WIN_EXPORT QKVNormLayer(QKVNormLayer &&rhs) noexcept = default;
  WIN_EXPORT QKVNormLayer &operator=(QKVNormLayer &&rhs) = default;

  WIN_EXPORT void finalize(nntrainer::InitLayerContext &context) override;
  WIN_EXPORT void forwarding(nntrainer::RunLayerContext &context,
                             bool training) override;
  WIN_EXPORT void incremental_forwarding(nntrainer::RunLayerContext &context,
                                         unsigned int from, unsigned int to,
                                         bool training) override;

  WIN_EXPORT void calcDerivative(nntrainer::RunLayerContext &context) override;
  WIN_EXPORT void calcGradient(nntrainer::RunLayerContext &context) override;

  WIN_EXPORT void exportTo(nntrainer::Exporter &exporter,
                            const ml::train::ExportMethods &method) const override;

  WIN_EXPORT const std::string getType() const override {
    return QKVNormLayer::type;
  };

  WIN_EXPORT bool supportBackwarding() const override { return false; }

  WIN_EXPORT void setProperty(const std::vector<std::string> &values) override;

  WIN_EXPORT void updateTensorsByInputDimensions(
    nntrainer::RunLayerContext &context,
    std::vector<nntrainer::TensorDim> input_dimensions) override;

  inline static const std::string type = "qkv_norm_layer";

private:
  std::tuple<qkvnorm_props::QUnit, qkvnorm_props::KUnit, qkvnorm_props::VUnit,
              qkvnorm_props::FeatureSize, qkvnorm_props::Epsilon>
    qkv_norm_props;

  std::array<unsigned int, 5> weight_idx; /**< q_w, gamma_q, k_w, gamma_k, v_w */
};

} // namespace causallm

#endif /* __cplusplus */
#endif /* __QKV_NORM_LAYER_H__ */
