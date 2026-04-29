// SPDX-License-Identifier: Apache-2.0
/**
 * @file   gate_up_layer.h
 * @brief  Batched MLP gate/up projection layer.  Mirrors QKVLayer with
 *         two weights instead of three.  Weight registration order
 *         intentionally matches the existing per-FC bundle byte
 *         layout: createMlp() in transformer.cpp registers ffn_up
 *         BEFORE ffn_gate, so we register up first then gate.  This
 *         keeps the existing pytorch_model.bin valid; no bundle
 *         repackaging required.
 *
 *         Output index convention:
 *           gate_up_layer(0) = up projection  (matches former ffn_up)
 *           gate_up_layer(1) = gate projection (matches former ffn_gate)
 *
 *         Callers feeding swiglu must reference these as
 *         "gate_up_name(1),gate_up_name(0)" since swiglu expects
 *         (gate, up) input order.
 */

#ifndef __GATE_UP_LAYER_H__
#define __GATE_UP_LAYER_H__
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

class UpUnit : public nntrainer::PositiveIntegerProperty {
public:
  static constexpr const char *key = "up_unit";
  using prop_tag = nntrainer::uint_prop_tag;
};

class GateUnit : public nntrainer::PositiveIntegerProperty {
public:
  static constexpr const char *key = "gate_unit";
  using prop_tag = nntrainer::uint_prop_tag;
};

class GateUpEpsilon : public nntrainer::Property<float> {
public:
  static constexpr const char *key = "epsilon";
  using prop_tag = nntrainer::float_prop_tag;
  GateUpEpsilon(float val = 1e-6f) : Property(val) {}
};

} // namespace props

WIN_EXPORT class GateUpLayer : public nntrainer::LayerImpl {
public:
  WIN_EXPORT GateUpLayer();
  WIN_EXPORT ~GateUpLayer() = default;

  WIN_EXPORT GateUpLayer(GateUpLayer &&rhs) noexcept = default;
  WIN_EXPORT GateUpLayer &operator=(GateUpLayer &&rhs) = default;

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
    return GateUpLayer::type;
  };
  WIN_EXPORT bool supportBackwarding() const override { return true; }
  WIN_EXPORT void setProperty(const std::vector<std::string> &values) override;
  WIN_EXPORT void updateTensorsByInputDimensions(
    nntrainer::RunLayerContext &context,
    std::vector<nntrainer::TensorDim> input_dimensions) override;

  inline static const std::string type = "gate_up_layer";

private:
  std::tuple<props::UpUnit, props::GateUnit, props::GateUpEpsilon> gate_up_props;
  // Layout: weight_idx[0] = rmsnorm gamma (fp32, K_in elements)
  //         weight_idx[1] = up weight (channel-wise int4, K_in -> N_up)
  //         weight_idx[2] = gate weight (channel-wise int4, K_in -> N_gate)
  // Order matches the legacy bundle byte layout
  // (ffn_norm.gamma | ffn_up.weight | ffn_gate.weight) so the existing
  // pytorch_model.bin loads without repackaging.
  std::array<unsigned int, 3> weight_idx;
};

} // namespace causallm

#endif // __cplusplus
#endif // __GATE_UP_LAYER_H__
