// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Samsung Electronics Co., Ltd.
 *
 * @file   embedding_layer_cl.h
 * @date   10 April 2026
 * @brief  Embedding Layer Class with OpenCL implementation
 * @see    https://github.com/nntrainer/nntrainer
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef __EMBEDDING_LAYER_CL_H__
#define __EMBEDDING_LAYER_CL_H__
#ifdef __cplusplus

#include <cl_context.h>
#include <common_properties.h>
#include <layer_context.h>
#include <layer_devel.h>
#include <layer_impl_cl.h>
#include <node_exporter.h>
#include <opencl_buffer.h>
#include <opencl_kernel.h>

namespace nntrainer {

/**
 * @class   EmbeddingLayerCl
 * @brief   Embedding Layer with OpenCL support
 */
class EmbeddingLayerCl final : public LayerImplCl {
public:
  /**
   * @brief Construct a new Embedding Layer Cl object
   */
  EmbeddingLayerCl();

  /**
   * @brief Destroy the Embedding Layer Cl object
   */
  ~EmbeddingLayerCl() = default;

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
  void calcGradient(RunLayerContext &context) override {};

  /**
   * @copydoc bool supportBackwarding() const
   */
  bool supportBackwarding() const override { return false; };

  /**
   * @copydoc Layer::exportTo(Exporter &exporter, ExportMethods method)
   */
  void exportTo(Exporter &exporter,
                const ml::train::ExportMethods &method) const override;

  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const override { return EmbeddingLayerCl::type; };

  /**
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   */
  void setProperty(const std::vector<std::string> &values) override;

  /**
   * @copydoc Layer::save()
   */
  void save(std::ofstream &file, RunLayerContext &run_context, bool opt_var,
            ml::train::ExecutionMode mode, bool trainable,
            TensorDim::DataType dtype = TensorDim::DataType::NONE) const override;

  /**
   * @brief Register OpenCL kernels for Embedding layer
   */
  static bool registerClKernels(ClContext &cl_context);

  static constexpr const char *type = "embedding_layer";

  /**
   * @brief embedding lookup (FP32) - public for testing
   */
  void embedding_cl(float *input, float *weight, float *output,
                    unsigned int num_tokens, unsigned int out_dim, float scale,
                    bool svm = true);

#ifdef ENABLE_FP16
  /**
   * @brief embedding lookup (FP16) - public for testing
   */
  void embedding_cl_fp16(float *input, _FP16 *weight, _FP16 *output,
                         unsigned int num_tokens, unsigned int out_dim,
                         float scale, bool svm = true);
#endif

private:
  std::tuple<props::InDim, props::OutDim, props::Scale> embedding_props;
  unsigned int weight_idx;

  static std::vector<ClContext::SharedPtrClKernel> &getLayerKernelPtrs();

  enum Kernels { EMBEDDING_CL, EMBEDDING_CL_FP16 };
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __EMBEDDING_LAYER_CL_H__ */
