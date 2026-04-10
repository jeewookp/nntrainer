// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Samsung Electronics Co., Ltd.
 *
 * @file unittest_layers_tie_word_embedding_cl.cpp
 * @date 10 April 2026
 * @brief Tie Word Embedding Layer CL Test
 * @see	https://github.com/nntrainer/nntrainer
 * @bug No known bugs except for NYI items
 */
#include <tuple>

#include <gtest/gtest.h>

#include <layers_common_tests.h>
#include <tie_word_embedding_cl.h>

/// Test embedding mode (no unit property)
auto semantic_tie_word_embedding_gpu = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::TieWordEmbeddingCl>,
  nntrainer::TieWordEmbeddingCl::type, {"in_dim=10", "out_dim=5"},
  LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false, 1);

GTEST_PARAMETER_TEST(TieWordEmbeddingGPU, LayerSemanticsGpu,
                     ::testing::Values(semantic_tie_word_embedding_gpu));

/// Test embedding mode
auto tie_word_embedding_cl_embed = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::TieWordEmbeddingCl>,
  {"in_dim=10", "out_dim=5"}, "1:1:1:3",
  "tie_word_embedding_cl_embed.nnlayergolden",
  LayerGoldenTestParamOptions::SKIP_CALC_DERIV |
    LayerGoldenTestParamOptions::SKIP_CALC_GRAD |
    LayerGoldenTestParamOptions::USE_INC_FORWARD,
  "nchw", "fp32", "fp32");

/// Test lm_head mode (with unit property)
auto tie_word_embedding_cl_lmhead = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::TieWordEmbeddingCl>,
  {"unit=10", "disable_bias=true"}, "1:1:3:5",
  "tie_word_embedding_cl_lmhead.nnlayergolden",
  LayerGoldenTestParamOptions::SKIP_CALC_DERIV |
    LayerGoldenTestParamOptions::SKIP_CALC_GRAD |
    LayerGoldenTestParamOptions::USE_INC_FORWARD,
  "nchw", "fp32", "fp32");

GTEST_PARAMETER_TEST(TieWordEmbeddingGPU, LayerGoldenTest,
                     ::testing::Values(tie_word_embedding_cl_embed,
                                       tie_word_embedding_cl_lmhead));
