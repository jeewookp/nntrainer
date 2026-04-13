// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Samsung Electronics Co., Ltd.
 *
 * @file unittest_layers_embedding_cl.cpp
 * @date 10 April 2026
 * @brief Embedding Layer CL Test (SVM-based, mirrors real model path)
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
#include <nntrainer_test_util.h>

/// Semantics test
auto semantic_embedding_gpu = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::EmbeddingLayerCl>,
  nntrainer::EmbeddingLayerCl::type, {"in_dim=10", "out_dim=5"},
  LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false, 1);

GTEST_PARAMETER_TEST(EmbeddingGPU, LayerSemanticsGpu,
                     ::testing::Values(semantic_embedding_gpu));

/**
 * @brief Run embedding lookup on GPU using SVM (mirrors real model memory)
 *        and compare against an inline CPU reference.
 */
static void runEmbeddingSVMTest(unsigned int vocab_size,
                                unsigned int embed_dim,
                                unsigned int num_tokens,
                                const std::vector<float> &token_ids,
                                float scale,
                                std::function<float(unsigned int, unsigned int)> weight_fn) {
  auto *cl_ctx = static_cast<nntrainer::ClContext *>(
    nntrainer::Engine::Global().getRegisteredContext("gpu"));

  size_t weight_bytes = vocab_size * embed_dim * sizeof(float);
  size_t input_bytes = num_tokens * sizeof(float);
  size_t output_bytes = num_tokens * embed_dim * sizeof(float);

  float *input_svm = (float *)allocateSVM(input_bytes);
  float *weight_svm = (float *)allocateSVM(weight_bytes);
  float *output_svm = (float *)allocateSVM(output_bytes);

  // Map input/weight for host write, fill, unmap (rmsnorm_cl test pattern)
  cl_ctx->command_queue_inst_.enqueueSVMMap(input_svm, input_bytes, false);
  cl_ctx->command_queue_inst_.enqueueSVMMap(weight_svm, weight_bytes, false);

  for (unsigned int i = 0; i < num_tokens; ++i) {
    input_svm[i] = token_ids[i];
  }
  for (unsigned int r = 0; r < vocab_size; ++r) {
    for (unsigned int c = 0; c < embed_dim; ++c) {
      weight_svm[r * embed_dim + c] = weight_fn(r, c);
    }
  }

  cl_ctx->command_queue_inst_.enqueueSVMUnmap(input_svm);
  cl_ctx->command_queue_inst_.enqueueSVMUnmap(weight_svm);

  // Run GPU kernel — input/weight unmapped, output never mapped.
  // embedding_cl maps output internally for read after dispatch.
  nntrainer::EmbeddingLayerCl layer;
  layer.embedding_cl(input_svm, weight_svm, output_svm, num_tokens, embed_dim,
                     scale, true);

  // CPU reference
  std::vector<float> expected(num_tokens * embed_dim);
  for (unsigned int t = 0; t < num_tokens; ++t) {
    unsigned int idx = static_cast<unsigned int>(token_ids[t]);
    for (unsigned int d = 0; d < embed_dim; ++d) {
      expected[t * embed_dim + d] = weight_fn(idx, d) * scale;
    }
  }

  // output_svm has been mapped for read by embedding_cl
  const float tol = 1e-5f;
  for (unsigned int t = 0; t < num_tokens; ++t) {
    for (unsigned int d = 0; d < embed_dim; ++d) {
      EXPECT_NEAR(output_svm[t * embed_dim + d],
                  expected[t * embed_dim + d], tol)
        << "Mismatch at token=" << t << " dim=" << d;
    }
  }

  freeSVM(input_svm);
  freeSVM(weight_svm);
  freeSVM(output_svm);
}

TEST(EmbeddingGPU_Kernel, fp32_lookup) {
  runEmbeddingSVMTest(
    /*vocab_size=*/8, /*embed_dim=*/4, /*num_tokens=*/3,
    /*token_ids=*/{2.0f, 5.0f, 0.0f}, /*scale=*/2.0f,
    /*weight_fn=*/[](unsigned int r, unsigned int c) {
      return r * 0.1f * (c + 1);
    });
}

TEST(EmbeddingGPU_Kernel, fp32_no_scale) {
  runEmbeddingSVMTest(
    /*vocab_size=*/4, /*embed_dim=*/8, /*num_tokens=*/2,
    /*token_ids=*/{3.0f, 1.0f}, /*scale=*/1.0f,
    /*weight_fn=*/[](unsigned int r, unsigned int c) {
      return (r + 1) * 0.5f + c * 0.01f;
    });
}

TEST(EmbeddingGPU_Kernel, fp32_single_token) {
  runEmbeddingSVMTest(
    /*vocab_size=*/16, /*embed_dim=*/32, /*num_tokens=*/1,
    /*token_ids=*/{7.0f}, /*scale=*/1.0f,
    /*weight_fn=*/[](unsigned int r, unsigned int c) {
      return sinf(r * 0.7f + c * 0.3f);
    });
}
