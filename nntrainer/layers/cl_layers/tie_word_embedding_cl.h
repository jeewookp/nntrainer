// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Samsung Electronics Co., Ltd.
 *
 * @file   tie_word_embedding_cl.h
 * @date   10 April 2026
 * @brief  Tie Word Embedding Layer Class with OpenCL implementation
 * @see    https://github.com/nntrainer/nntrainer
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef __TIE_WORD_EMBEDDING_CL_H__
#define __TIE_WORD_EMBEDDING_CL_H__
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
 * @class   TieWordEmbeddingCl
 * @brief   Tie Word Embedding layer with OpenCL support
 *          Acts as embedding (no Unit) or lm_head (with Unit)
 */
class TieWordEmbeddingCl final : public LayerImplCl {
public:
  /**
   * @brief Constructor
   */
  TieWordEmbeddingCl();

  /**
   * @brief Destructor
   */
  ~TieWordEmbeddingCl() = default;

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
  const std::string getType() const override {
    return TieWordEmbeddingCl::type;
  };

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

  /**
   * @copydoc Layer::read() (ifstream variant)
   */
  void read(std::ifstream &file, RunLayerContext &context, bool opt_var,
            ml::train::ExecutionMode mode, bool trainable,
            TensorDim::DataType definedWeightDataType, bool fsu = false,
            size_t start_offset = 0, bool read_from_offset = false,
            int file_fd = -1) override;

  /**
   * @copydoc Layer::read() (ReadSource/mmap variant)
   */
  void read(ReadSource src, RunLayerContext &context, bool opt_var,
            ml::train::ExecutionMode mode, bool trainable,
            TensorDim::DataType definedWeightDataType, bool fsu,
            size_t start_offset = 0, bool read_from_offset = false) override;

  /**
   * @copydoc Layer::save()
   */
  void save(std::ofstream &file, RunLayerContext &run_context, bool opt_var,
            ml::train::ExecutionMode mode, bool trainable,
            TensorDim::DataType dtype = TensorDim::DataType::NONE) const override;

  /**
   * @brief Register OpenCL kernels
   */
  static bool registerClKernels(ClContext &cl_context);

  static constexpr const char *type = "tie_word_embeddings";

  /**
   * @brief embedding lookup kernel (FP32) - public for testing
   */
  void embedding_cl_kernel(float *input, float *weight, float *output,
                           unsigned int num_tokens, unsigned int out_dim,
                           float scale, bool svm = true);

#ifdef ENABLE_FP16
  /**
   * @brief embedding lookup kernel (FP16) - public for testing
   */
  void embedding_cl_fp16_kernel(float *input, _FP16 *weight, _FP16 *output,
                                unsigned int num_tokens, unsigned int out_dim,
                                float scale, bool svm = true);
#endif

private:
  std::tuple<props::InDim, props::OutDim, props::Unit, props::Scale>
    tieword_embedding_props;
  enum mode { embedding, lm_head };
  enum mode mode_;
  std::array<unsigned int, 4> weight_idx;

  static std::vector<ClContext::SharedPtrClKernel> &getLayerKernelPtrs();

  enum Kernels { EMBEDDING_CL, EMBEDDING_CL_FP16 };

  void finalize_embedding(InitLayerContext &context);
  void finalize_lmhead(InitLayerContext &context);

  void incremental_forwarding_embedding(RunLayerContext &context,
                                        unsigned int from, unsigned int to,
                                        bool training);
  void incremental_forwarding_lmhead(RunLayerContext &context,
                                     unsigned int from, unsigned int to,
                                     bool training);
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __TIE_WORD_EMBEDDING_CL_H__ */
