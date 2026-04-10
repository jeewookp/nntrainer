// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Samsung Electronics Co., Ltd.
 *
 * @file unittest_layers_lm_head_cl.cpp
 * @date 10 April 2026
 * @brief LM Head Layer CL Test
 * @see	https://github.com/nntrainer/nntrainer
 * @bug No known bugs except for NYI items
 */
#include <tuple>

#include <gtest/gtest.h>

#include <layers_common_tests.h>
#include <lm_head_cl.h>

auto semantic_lm_head_gpu = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::LmHeadLayerCl>,
  nntrainer::LmHeadLayerCl::type, {"unit=10"},
  LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false, 1);

GTEST_PARAMETER_TEST(LmHeadGPU, LayerSemanticsGpu,
                     ::testing::Values(semantic_lm_head_gpu));

auto lm_head_cl_basic = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::LmHeadLayerCl>,
  {"unit=10", "disable_bias=true"}, "1:1:3:5",
  "lm_head_cl.nnlayergolden",
  LayerGoldenTestParamOptions::SKIP_CALC_DERIV |
    LayerGoldenTestParamOptions::SKIP_CALC_GRAD |
    LayerGoldenTestParamOptions::USE_INC_FORWARD,
  "nchw", "fp32", "fp32");

GTEST_PARAMETER_TEST(LmHeadGPU, LayerGoldenTest,
                     ::testing::Values(lm_head_cl_basic));

#ifdef ENABLE_FP16
auto lm_head_cl_fp16 = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::LmHeadLayerCl>,
  {"unit=10", "disable_bias=true"}, "1:1:3:5",
  "lm_head_cl_fp16.nnlayergolden",
  LayerGoldenTestParamOptions::SKIP_CALC_DERIV |
    LayerGoldenTestParamOptions::SKIP_CALC_GRAD |
    LayerGoldenTestParamOptions::USE_INC_FORWARD,
  "nchw", "fp16", "fp16");

GTEST_PARAMETER_TEST(LmHeadGPU16, LayerGoldenTest,
                     ::testing::Values(lm_head_cl_fp16));
#endif
