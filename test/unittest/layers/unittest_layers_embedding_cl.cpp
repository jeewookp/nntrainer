// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Samsung Electronics Co., Ltd.
 *
 * @file unittest_layers_embedding_cl.cpp
 * @date 10 April 2026
 * @brief Embedding Layer CL Test
 * @see	https://github.com/nntrainer/nntrainer
 * @bug No known bugs except for NYI items
 */
#include <tuple>

#include <gtest/gtest.h>

#include <embedding_layer_cl.h>
#include <layers_common_tests.h>

auto semantic_embedding_gpu = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::EmbeddingLayerCl>,
  nntrainer::EmbeddingLayerCl::type, {"in_dim=10", "out_dim=5"},
  LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false, 1);

GTEST_PARAMETER_TEST(EmbeddingGPU, LayerSemanticsGpu,
                     ::testing::Values(semantic_embedding_gpu));

auto embedding_cl_basic = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::EmbeddingLayerCl>,
  {"in_dim=10", "out_dim=5"}, "1:1:1:3",
  "embedding_cl.nnlayergolden",
  LayerGoldenTestParamOptions::SKIP_CALC_DERIV |
    LayerGoldenTestParamOptions::SKIP_CALC_GRAD |
    LayerGoldenTestParamOptions::USE_INC_FORWARD,
  "nchw", "fp32", "fp32");

GTEST_PARAMETER_TEST(EmbeddingGPU, LayerGoldenTest,
                     ::testing::Values(embedding_cl_basic));

#ifdef ENABLE_FP16
auto embedding_cl_fp16 = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::EmbeddingLayerCl>,
  {"in_dim=10", "out_dim=5"}, "1:1:1:3",
  "embedding_cl_fp16.nnlayergolden",
  LayerGoldenTestParamOptions::SKIP_CALC_DERIV |
    LayerGoldenTestParamOptions::SKIP_CALC_GRAD |
    LayerGoldenTestParamOptions::USE_INC_FORWARD,
  "nchw", "fp16", "fp16");

GTEST_PARAMETER_TEST(EmbeddingGPU16, LayerGoldenTest,
                     ::testing::Values(embedding_cl_fp16));
#endif
