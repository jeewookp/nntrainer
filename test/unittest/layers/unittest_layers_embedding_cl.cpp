// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Samsung Electronics Co., Ltd.
 *
 * @file unittest_layers_embedding_cl.cpp
 * @date 10 April 2026
 * @brief Embedding Layer CL Test (self-contained, no golden files)
 * @see	https://github.com/nntrainer/nntrainer
 * @bug No known bugs except for NYI items
 */
#include <cmath>
#include <cstring>
#include <gtest/gtest.h>

#include <cl_context.h>
#include <embedding_layer_cl.h>
#include <engine.h>
#include <layers_common_tests.h>
#include <tensor.h>

/// Semantics test: layer creation, property setting, finalize
auto semantic_embedding_gpu = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::EmbeddingLayerCl>,
  nntrainer::EmbeddingLayerCl::type, {"in_dim=10", "out_dim=5"},
  LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false, 1);

GTEST_PARAMETER_TEST(EmbeddingGPU, LayerSemanticsGpu,
                     ::testing::Values(semantic_embedding_gpu));

/**
 * @brief Test embedding lookup on GPU vs CPU reference
 *
 * Weight table [vocab_size=8, embed_dim=4]:
 *   row i = [i*0.1, i*0.2, i*0.3, i*0.4]
 *
 * Input tokens: [2, 5, 0] (3 tokens)
 * Scale: 2.0
 *
 * Expected output:
 *   token 2 -> [0.2, 0.4, 0.6, 0.8] * 2.0 = [0.4, 0.8, 1.2, 1.6]
 *   token 5 -> [0.5, 1.0, 1.5, 2.0] * 2.0 = [1.0, 2.0, 3.0, 4.0]
 *   token 0 -> [0.0, 0.0, 0.0, 0.0] * 2.0 = [0.0, 0.0, 0.0, 0.0]
 */
TEST(EmbeddingGPU_Kernel, fp32_lookup) {
  const unsigned int vocab_size = 8;
  const unsigned int embed_dim = 4;
  const unsigned int num_tokens = 3;
  const float scale = 2.0f;

  nntrainer::TensorDim::TensorType t_fp32 = {nntrainer::Tformat::NCHW,
                                              nntrainer::Tdatatype::FP32};

  // Build weight table: row i = [i*0.1, i*0.2, i*0.3, i*0.4]
  nntrainer::Tensor weight(1, 1, vocab_size, embed_dim, t_fp32);
  for (unsigned int r = 0; r < vocab_size; ++r) {
    for (unsigned int c = 0; c < embed_dim; ++c) {
      weight.setValue(0, 0, r, c, r * 0.1f * (c + 1));
    }
  }

  // Input: token IDs as float
  nntrainer::Tensor input(1, 1, 1, num_tokens, t_fp32);
  input.setValue(0, 0, 0, 0, 2.0f);
  input.setValue(0, 0, 0, 1, 5.0f);
  input.setValue(0, 0, 0, 2, 0.0f);

  // GPU output
  nntrainer::Tensor output_gpu(1, 1, num_tokens, embed_dim, t_fp32);
  output_gpu.setZero();

  // Compute expected output on CPU
  float expected[num_tokens][embed_dim];
  for (unsigned int t = 0; t < num_tokens; ++t) {
    unsigned int idx = static_cast<unsigned int>(input.getValue(0, 0, 0, t));
    for (unsigned int d = 0; d < embed_dim; ++d) {
      expected[t][d] = weight.getValue(0, 0, idx, d) * scale;
    }
  }

  // Run GPU kernel
  nntrainer::EmbeddingLayerCl layer;
  layer.embedding_cl(input.getData<float>(), weight.getData<float>(),
                     output_gpu.getData<float>(), num_tokens, embed_dim, scale,
                     true);

  // Compare
  const float tolerance = 1e-5f;
  for (unsigned int t = 0; t < num_tokens; ++t) {
    for (unsigned int d = 0; d < embed_dim; ++d) {
      EXPECT_NEAR(output_gpu.getValue(0, 0, t, d), expected[t][d], tolerance)
        << "Mismatch at token=" << t << " dim=" << d;
    }
  }
}

/**
 * @brief Test embedding with scale=1.0 (no scaling)
 */
TEST(EmbeddingGPU_Kernel, fp32_no_scale) {
  const unsigned int vocab_size = 4;
  const unsigned int embed_dim = 8;
  const unsigned int num_tokens = 2;
  const float scale = 1.0f;

  nntrainer::TensorDim::TensorType t_fp32 = {nntrainer::Tformat::NCHW,
                                              nntrainer::Tdatatype::FP32};

  nntrainer::Tensor weight(1, 1, vocab_size, embed_dim, t_fp32);
  for (unsigned int r = 0; r < vocab_size; ++r) {
    for (unsigned int c = 0; c < embed_dim; ++c) {
      weight.setValue(0, 0, r, c, (r + 1) * 0.5f + c * 0.01f);
    }
  }

  nntrainer::Tensor input(1, 1, 1, num_tokens, t_fp32);
  input.setValue(0, 0, 0, 0, 3.0f);
  input.setValue(0, 0, 0, 1, 1.0f);

  nntrainer::Tensor output_gpu(1, 1, num_tokens, embed_dim, t_fp32);
  output_gpu.setZero();

  float expected[num_tokens][embed_dim];
  for (unsigned int t = 0; t < num_tokens; ++t) {
    unsigned int idx = static_cast<unsigned int>(input.getValue(0, 0, 0, t));
    for (unsigned int d = 0; d < embed_dim; ++d) {
      expected[t][d] = weight.getValue(0, 0, idx, d);
    }
  }

  nntrainer::EmbeddingLayerCl layer;
  layer.embedding_cl(input.getData<float>(), weight.getData<float>(),
                     output_gpu.getData<float>(), num_tokens, embed_dim, scale,
                     true);

  const float tolerance = 1e-5f;
  for (unsigned int t = 0; t < num_tokens; ++t) {
    for (unsigned int d = 0; d < embed_dim; ++d) {
      EXPECT_NEAR(output_gpu.getValue(0, 0, t, d), expected[t][d], tolerance)
        << "Mismatch at token=" << t << " dim=" << d;
    }
  }
}

/**
 * @brief Test single token embedding (incremental decode case)
 */
TEST(EmbeddingGPU_Kernel, fp32_single_token) {
  const unsigned int vocab_size = 16;
  const unsigned int embed_dim = 32;
  const unsigned int num_tokens = 1;
  const float scale = 1.0f;

  nntrainer::TensorDim::TensorType t_fp32 = {nntrainer::Tformat::NCHW,
                                              nntrainer::Tdatatype::FP32};

  nntrainer::Tensor weight(1, 1, vocab_size, embed_dim, t_fp32);
  for (unsigned int r = 0; r < vocab_size; ++r) {
    for (unsigned int c = 0; c < embed_dim; ++c) {
      weight.setValue(0, 0, r, c, sinf(r * 0.7f + c * 0.3f));
    }
  }

  nntrainer::Tensor input(1, 1, 1, 1, t_fp32);
  input.setValue(0, 0, 0, 0, 7.0f);

  nntrainer::Tensor output_gpu(1, 1, 1, embed_dim, t_fp32);
  output_gpu.setZero();

  nntrainer::EmbeddingLayerCl layer;
  layer.embedding_cl(input.getData<float>(), weight.getData<float>(),
                     output_gpu.getData<float>(), num_tokens, embed_dim, scale,
                     true);

  const float tolerance = 1e-5f;
  for (unsigned int d = 0; d < embed_dim; ++d) {
    float expected = weight.getValue(0, 0, 7, d);
    EXPECT_NEAR(output_gpu.getValue(0, 0, 0, d), expected, tolerance)
      << "Mismatch at dim=" << d;
  }
}
