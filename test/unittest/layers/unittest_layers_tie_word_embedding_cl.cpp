// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Samsung Electronics Co., Ltd.
 *
 * @file unittest_layers_tie_word_embedding_cl.cpp
 * @date 10 April 2026
 * @brief Tie Word Embedding Layer CL Test (self-contained, no golden files)
 * @see	https://github.com/nntrainer/nntrainer
 * @bug No known bugs except for NYI items
 */
#include <cmath>
#include <cstring>
#include <gtest/gtest.h>

#include <blas_kernel_interface.h>
#include <cl_context.h>
#include <engine.h>
#include <layers_common_tests.h>
#include <tensor.h>
#include <tie_word_embedding_cl.h>

/// Semantics test: embedding mode (no unit property)
auto semantic_tie_word_embedding_gpu = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::TieWordEmbeddingCl>,
  nntrainer::TieWordEmbeddingCl::type, {"in_dim=10", "out_dim=5"},
  LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false, 1);

GTEST_PARAMETER_TEST(TieWordEmbeddingGPU, LayerSemanticsGpu,
                     ::testing::Values(semantic_tie_word_embedding_gpu));

/**
 * @brief Test embedding mode: GPU kernel lookup vs CPU reference
 *
 * Same logic as EmbeddingLayerCl — token ID -> weight row lookup * scale
 */
TEST(TieWordEmbeddingGPU_Kernel, embedding_mode_fp32) {
  const unsigned int vocab_size = 8;
  const unsigned int embed_dim = 4;
  const unsigned int num_tokens = 3;
  const float scale = 1.5f;

  nntrainer::TensorDim::TensorType t_fp32 = {nntrainer::Tformat::NCHW,
                                              nntrainer::Tdatatype::FP32};

  nntrainer::Tensor weight(1, 1, vocab_size, embed_dim, t_fp32);
  for (unsigned int r = 0; r < vocab_size; ++r) {
    for (unsigned int c = 0; c < embed_dim; ++c) {
      weight.setValue(0, 0, r, c, (r + 1) * 0.2f + c * 0.1f);
    }
  }

  nntrainer::Tensor input(1, 1, 1, num_tokens, t_fp32);
  input.setValue(0, 0, 0, 0, 1.0f);
  input.setValue(0, 0, 0, 1, 4.0f);
  input.setValue(0, 0, 0, 2, 7.0f);

  nntrainer::Tensor output_gpu(1, 1, num_tokens, embed_dim, t_fp32);
  output_gpu.setZero();

  // Compute CPU expected
  float expected[3][4];
  for (unsigned int t = 0; t < num_tokens; ++t) {
    unsigned int idx = static_cast<unsigned int>(input.getValue(0, 0, 0, t));
    for (unsigned int d = 0; d < embed_dim; ++d) {
      expected[t][d] = weight.getValue(0, 0, idx, d) * scale;
    }
  }

  // Run GPU kernel (reuses embedding_cl internally)
  nntrainer::TieWordEmbeddingCl layer;
  layer.embedding_cl_kernel(input.getData<float>(), weight.getData<float>(),
                            output_gpu.getData<float>(), num_tokens, embed_dim,
                            scale, true);

  const float tolerance = 1e-5f;
  for (unsigned int t = 0; t < num_tokens; ++t) {
    for (unsigned int d = 0; d < embed_dim; ++d) {
      EXPECT_NEAR(output_gpu.getValue(0, 0, t, d), expected[t][d], tolerance)
        << "Mismatch at token=" << t << " dim=" << d;
    }
  }
}

/**
 * @brief Test lm_head mode: dotCl with transpose vs CPU dot transpose
 *
 * In tie_word_embeddings lm_head mode, weight is shared with embedding
 * so it's [vocab_size, hidden_dim] and used transposed:
 *   output = input * weight^T
 */
TEST(TieWordEmbeddingGPU_Kernel, lmhead_mode_fp32) {
  const unsigned int hidden_dim = 4;
  const unsigned int vocab_size = 6;

  nntrainer::TensorDim::TensorType t_fp32 = {nntrainer::Tformat::NCHW,
                                              nntrainer::Tdatatype::FP32};

  nntrainer::Tensor input(1, 1, 1, hidden_dim, t_fp32);
  for (unsigned int d = 0; d < hidden_dim; ++d) {
    input.setValue(0, 0, 0, d, (d + 1) * 0.4f);
  }

  // weight shape: [vocab_size, hidden_dim] — transposed for dot
  nntrainer::Tensor weight(1, 1, vocab_size, hidden_dim, t_fp32);
  for (unsigned int r = 0; r < vocab_size; ++r) {
    for (unsigned int c = 0; c < hidden_dim; ++c) {
      weight.setValue(0, 0, r, c, sinf(r * 0.5f + c * 0.3f));
    }
  }

  // CPU reference: input * weight^T
  nntrainer::Tensor output_cpu(1, 1, 1, vocab_size, t_fp32);
  input.dot(weight, output_cpu, false, true);

  // GPU: dotCl with trans_m=true
  nntrainer::Tensor output_gpu(1, 1, 1, vocab_size, t_fp32);
  output_gpu.setZero();
  nntrainer::dotCl(input, weight, output_gpu, false, true);

  const float tolerance = 1e-4f;
  for (unsigned int c = 0; c < vocab_size; ++c) {
    EXPECT_NEAR(output_gpu.getValue(0, 0, 0, c),
                output_cpu.getValue(0, 0, 0, c), tolerance)
      << "Mismatch at vocab index=" << c;
  }
}

/**
 * @brief Test embedding single token (decode step)
 */
TEST(TieWordEmbeddingGPU_Kernel, embedding_single_token) {
  const unsigned int vocab_size = 16;
  const unsigned int embed_dim = 8;
  const float scale = 1.0f;

  nntrainer::TensorDim::TensorType t_fp32 = {nntrainer::Tformat::NCHW,
                                              nntrainer::Tdatatype::FP32};

  nntrainer::Tensor weight(1, 1, vocab_size, embed_dim, t_fp32);
  for (unsigned int r = 0; r < vocab_size; ++r) {
    for (unsigned int c = 0; c < embed_dim; ++c) {
      weight.setValue(0, 0, r, c, cosf(r * 0.3f) + sinf(c * 0.5f));
    }
  }

  nntrainer::Tensor input(1, 1, 1, 1, t_fp32);
  input.setValue(0, 0, 0, 0, 11.0f);

  nntrainer::Tensor output_gpu(1, 1, 1, embed_dim, t_fp32);
  output_gpu.setZero();

  nntrainer::TieWordEmbeddingCl layer;
  layer.embedding_cl_kernel(input.getData<float>(), weight.getData<float>(),
                            output_gpu.getData<float>(), 1, embed_dim, scale,
                            true);

  const float tolerance = 1e-5f;
  for (unsigned int d = 0; d < embed_dim; ++d) {
    float expected = weight.getValue(0, 0, 11, d);
    EXPECT_NEAR(output_gpu.getValue(0, 0, 0, d), expected, tolerance)
      << "Mismatch at dim=" << d;
  }
}
