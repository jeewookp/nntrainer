// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Samsung Electronics Co., Ltd.
 *
 * @file   lm_head_cl.h
 * @date   10 April 2026
 * @brief  LM Head Layer Class with OpenCL implementation
 * @see    https://github.com/nntrainer/nntrainer
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef __LM_HEAD_CL_H__
#define __LM_HEAD_CL_H__
#ifdef __cplusplus

#include <cl_context.h>
#include <common_properties.h>
#include <layer_impl_cl.h>

namespace nntrainer {

/**
 * @class   LmHeadLayerCl
 * @brief   LM Head layer with OpenCL support (uses dotCl for GEMM)
 */
class LmHeadLayerCl : public LayerImplCl {
public:
  /**
   * @brief Constructor
   */
  LmHeadLayerCl();

  /**
   * @brief Destructor
   */
  ~LmHeadLayerCl() = default;

  /**
   * @copydoc Layer::finalize(InitLayerContext &context)
   */
  void finalize(InitLayerContext &context) override;

  /**
   * @copydoc Layer::forwarding(RunLayerContext &context, bool training)
   */
  void forwarding(RunLayerContext &context, bool training) override;

  /**
   * @copydoc Layer::incremental_forwarding(RunLayerContext &context, unsigned
   * int from, unsigned int to, bool training)
   */
  void incremental_forwarding(RunLayerContext &context, unsigned int from,
                              unsigned int to, bool training) override;

  /**
   * @copydoc Layer::calcDerivative(RunLayerContext &context)
   */
  void calcDerivative(RunLayerContext &context) override;

  /**
   * @copydoc Layer::calcGradient(RunLayerContext &context)
   */
  void calcGradient(RunLayerContext &context) override;

  /**
   * @copydoc Layer::exportTo(Exporter &exporter, ExportMethods method)
   */
  void exportTo(Exporter &exporter,
                const ml::train::ExportMethods &method) const override;

  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const override { return LmHeadLayerCl::type; };

  /**
   * @copydoc Layer::supportBackwarding()
   */
  bool supportBackwarding() const override { return false; }

  /**
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   */
  void setProperty(const std::vector<std::string> &values) override;

  /**
   * @copydoc Layer::updateTensorsByInputDimensions()
   */
  void updateTensorsByInputDimensions(
    RunLayerContext &context,
    std::vector<TensorDim> input_dimensions) override;

  static bool registerClKernels([[maybe_unused]] ClContext &cl_context) {
    return true;
  };

  static constexpr const char *type = "lm_head";

private:
  std::tuple<props::Unit> lmhead_props;
  std::array<unsigned int, 2> weight_idx;
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __LM_HEAD_CL_H__ */
