// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Samsung Electronics Co., Ltd.
 *
 * @file unittest_layers_lm_head_cl.cpp
 * @date 10 April 2026
 * @brief LM Head Layer CL Test (self-contained, no golden files)
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
#include <lm_head_cl.h>
#include <tensor.h>

/// Semantics test
auto semantic_lm_head_gpu = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::LmHeadLayerCl>,
  nntrainer::LmHeadLayerCl::type, {"unit=10"},
  LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false, 1);

GTEST_PARAMETER_TEST(LmHeadGPU, LayerSemanticsGpu,
                     ::testing::Values(semantic_lm_head_gpu));

/**
 * @brief Test dotCl (used by lm_head) vs CPU dot
 *
 * input: [1, 1, 1, 4]  (last token hidden state)
 * weight: [1, 1, 4, 6]  (projection to vocab=6)
 * output: [1, 1, 1, 6]  (logits)
 *
 * Verifies: dotCl(input, weight, output) == input.dot(weight)
 */
TEST(LmHeadGPU_Kernel, fp32_dot_basic) {
  const unsigned int hidden_dim = 4;
  const unsigned int vocab_size = 6;

  nntrainer::TensorDim::TensorType t_fp32 = {nntrainer::Tformat::NCHW,
                                              nntrainer::Tdatatype::FP32};

  // input: single token hidden state
  nntrainer::Tensor input(1, 1, 1, hidden_dim, t_fp32);
  for (unsigned int d = 0; d < hidden_dim; ++d) {
    input.setValue(0, 0, 0, d, (d + 1) * 0.5f);
  }

  // weight: [hidden_dim, vocab_size]
  nntrainer::Tensor weight(1, 1, hidden_dim, vocab_size, t_fp32);
  for (unsigned int r = 0; r < hidden_dim; ++r) {
    for (unsigned int c = 0; c < vocab_size; ++c) {
      weight.setValue(0, 0, r, c, sinf(r * 1.1f + c * 0.7f));
    }
  }

  // CPU reference
  nntrainer::Tensor output_cpu(1, 1, 1, vocab_size, t_fp32);
  input.dot(weight, output_cpu, false, false);

  // GPU via dotCl
  nntrainer::Tensor output_gpu(1, 1, 1, vocab_size, t_fp32);
  output_gpu.setZero();
  nntrainer::dotCl(input, weight, output_gpu, false, false);

  // Compare
  const float tolerance = 1e-4f;
  for (unsigned int c = 0; c < vocab_size; ++c) {
    EXPECT_NEAR(output_gpu.getValue(0, 0, 0, c),
                output_cpu.getValue(0, 0, 0, c), tolerance)
      << "Mismatch at vocab index=" << c;
  }
}

/**
 * @brief Test dotCl with larger dimensions (closer to real LLM sizes)
 */
TEST(LmHeadGPU_Kernel, fp32_dot_larger) {
  const unsigned int hidden_dim = 128;
  const unsigned int vocab_size = 256;

  nntrainer::TensorDim::TensorType t_fp32 = {nntrainer::Tformat::NCHW,
                                              nntrainer::Tdatatype::FP32};

  nntrainer::Tensor input(1, 1, 1, hidden_dim, t_fp32);
  for (unsigned int d = 0; d < hidden_dim; ++d) {
    input.setValue(0, 0, 0, d, cosf(d * 0.05f));
  }

  nntrainer::Tensor weight(1, 1, hidden_dim, vocab_size, t_fp32);
  for (unsigned int r = 0; r < hidden_dim; ++r) {
    for (unsigned int c = 0; c < vocab_size; ++c) {
      weight.setValue(0, 0, r, c, sinf(r * 0.03f + c * 0.02f) * 0.1f);
    }
  }

  nntrainer::Tensor output_cpu(1, 1, 1, vocab_size, t_fp32);
  input.dot(weight, output_cpu, false, false);

  nntrainer::Tensor output_gpu(1, 1, 1, vocab_size, t_fp32);
  output_gpu.setZero();
  nntrainer::dotCl(input, weight, output_gpu, false, false);

  const float tolerance = 1e-3f;
  for (unsigned int c = 0; c < vocab_size; ++c) {
    EXPECT_NEAR(output_gpu.getValue(0, 0, 0, c),
                output_cpu.getValue(0, 0, 0, c), tolerance)
      << "Mismatch at vocab index=" << c;
  }
}

/**
 * @brief Test dotCl with transpose (tie_word_embeddings lm_head mode)
 *
 * input: [1, 1, 1, 4]
 * weight: [1, 1, 6, 4]  (transposed: vocab_size x hidden_dim)
 * output = input * weight^T => [1, 1, 1, 6]
 */
TEST(LmHeadGPU_Kernel, fp32_dot_transpose) {
  const unsigned int hidden_dim = 4;
  const unsigned int vocab_size = 6;

  nntrainer::TensorDim::TensorType t_fp32 = {nntrainer::Tformat::NCHW,
                                              nntrainer::Tdatatype::FP32};

  nntrainer::Tensor input(1, 1, 1, hidden_dim, t_fp32);
  for (unsigned int d = 0; d < hidden_dim; ++d) {
    input.setValue(0, 0, 0, d, (d + 1) * 0.3f);
  }

  // weight shape is [vocab_size, hidden_dim] — will be transposed
  nntrainer::Tensor weight(1, 1, vocab_size, hidden_dim, t_fp32);
  for (unsigned int r = 0; r < vocab_size; ++r) {
    for (unsigned int c = 0; c < hidden_dim; ++c) {
      weight.setValue(0, 0, r, c, (r + 1) * 0.1f + c * 0.05f);
    }
  }

  // CPU: input * weight^T
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
