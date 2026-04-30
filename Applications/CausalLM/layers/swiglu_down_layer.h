// SPDX-License-Identifier: Apache-2.0
/**
 * @file   swiglu_down_layer.h
 * @brief  Fused SwiGLU + down projection for M=1 decode MLP.  v2 with
 *         __local memory tiling -- avoids the per-lane redundant silu
 *         computation that made the v1 attempt slower than separate
 *         swiglu + gemv.  Bundle byte order preserved: registers one
 *         weight (down) in the slot the standalone ffn_down used to
 *         occupy.  M>1 prefill falls back to GPU swiglu + dot path.
 */
#ifndef __SWIGLU_DOWN_LAYER_H__
#define __SWIGLU_DOWN_LAYER_H__
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

namespace props {

class DownUnit : public nntrainer::PositiveIntegerProperty {
public:
  static constexpr const char *key = "down_unit";
  using prop_tag = nntrainer::uint_prop_tag;
};

} // namespace props

WIN_EXPORT class SwigluDownLayer : public nntrainer::LayerImpl {
public:
  WIN_EXPORT SwigluDownLayer();
  WIN_EXPORT ~SwigluDownLayer() = default;

  WIN_EXPORT SwigluDownLayer(SwigluDownLayer &&rhs) noexcept = default;
  WIN_EXPORT SwigluDownLayer &operator=(SwigluDownLayer &&rhs) = default;

  WIN_EXPORT void finalize(nntrainer::InitLayerContext &context) override;
  WIN_EXPORT void forwarding(nntrainer::RunLayerContext &context,
                             bool training) override;
  WIN_EXPORT void incremental_forwarding(nntrainer::RunLayerContext &context,
                                         unsigned int from, unsigned int to,
                                         bool training) override;
  WIN_EXPORT void calcDerivative(nntrainer::RunLayerContext &context) override;
  WIN_EXPORT void calcGradient(nntrainer::RunLayerContext &context) override;
  WIN_EXPORT void
  exportTo(nntrainer::Exporter &exporter,
           const ml::train::ExportMethods &method) const override;
  WIN_EXPORT const std::string getType() const override {
    return SwigluDownLayer::type;
  };
  WIN_EXPORT bool supportBackwarding() const override { return true; }
  WIN_EXPORT void setProperty(const std::vector<std::string> &values) override;
  WIN_EXPORT void updateTensorsByInputDimensions(
    nntrainer::RunLayerContext &context,
    std::vector<nntrainer::TensorDim> input_dimensions) override;

  inline static const std::string type = "swiglu_down_layer";

private:
  std::tuple<props::DownUnit> swiglu_down_props;
  unsigned int weight_idx;
};

} // namespace causallm

#endif // __cplusplus
#endif // __SWIGLU_DOWN_LAYER_H__
