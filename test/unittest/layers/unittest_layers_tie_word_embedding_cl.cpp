// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Samsung Electronics Co., Ltd.
 *
 * @file unittest_layers_tie_word_embedding_cl.cpp
 * @date 10 April 2026
 * @brief Tie Word Embedding Layer CL Test (SVM-based for embedding mode,
 *        Tensor-based for lm_head mode via dotCl)
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
#include <nntrainer_test_util.h>
#include <tensor.h>
#include <tie_word_embedding_cl.h>
#include <timer.h>

/// Semantics test
auto semantic_tie_word_embedding_gpu = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::TieWordEmbeddingCl>,
  nntrainer::TieWordEmbeddingCl::type, {"in_dim=10", "out_dim=5"},
  LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false, 1);

GTEST_PARAMETER_TEST(TieWordEmbeddingGPU, LayerSemanticsGpu,
                     ::testing::Values(semantic_tie_word_embedding_gpu));

/**
 * @brief Run tie_word_embedding embedding-mode lookup on GPU using SVM
 *        and compare against an inline CPU reference.
 */
static void runTieWordEmbeddingSVMTest(
  unsigned int vocab_size, unsigned int embed_dim, unsigned int num_tokens,
  const std::vector<float> &token_ids, float scale,
  std::function<float(unsigned int, unsigned int)> weight_fn) {
  auto *cl_ctx = static_cast<nntrainer::ClContext *>(
    nntrainer::Engine::Global().getRegisteredContext("gpu"));

  size_t weight_bytes = vocab_size * embed_dim * sizeof(float);
  size_t input_bytes = num_tokens * sizeof(float);
  size_t output_bytes = num_tokens * embed_dim * sizeof(float);

  float *input_svm = (float *)allocateSVM(input_bytes);
  float *weight_svm = (float *)allocateSVM(weight_bytes);
  float *output_svm = (float *)allocateSVM(output_bytes);

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

  nntrainer::TieWordEmbeddingCl layer;
  layer.embedding_cl_kernel(input_svm, weight_svm, output_svm, num_tokens,
                            embed_dim, scale, true);

  std::vector<float> expected(num_tokens * embed_dim);
  for (unsigned int t = 0; t < num_tokens; ++t) {
    unsigned int idx = static_cast<unsigned int>(token_ids[t]);
    for (unsigned int d = 0; d < embed_dim; ++d) {
      expected[t * embed_dim + d] = weight_fn(idx, d) * scale;
    }
  }

  // output_svm has been mapped for read by embedding_cl_kernel
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

TEST(TieWordEmbeddingGPU_Kernel, embedding_mode_fp32) {
  runTieWordEmbeddingSVMTest(
    /*vocab_size=*/8, /*embed_dim=*/4, /*num_tokens=*/3,
    /*token_ids=*/{1.0f, 4.0f, 7.0f}, /*scale=*/1.5f,
    /*weight_fn=*/[](unsigned int r, unsigned int c) {
      return (r + 1) * 0.2f + c * 0.1f;
    });
}

TEST(TieWordEmbeddingGPU_Kernel, embedding_single_token) {
  runTieWordEmbeddingSVMTest(
    /*vocab_size=*/16, /*embed_dim=*/8, /*num_tokens=*/1,
    /*token_ids=*/{11.0f}, /*scale=*/1.0f,
    /*weight_fn=*/[](unsigned int r, unsigned int c) {
      return cosf(r * 0.3f) + sinf(c * 0.5f);
    });
}

/**
 * @brief Test lm_head mode: dotCl with transpose vs CPU dot transpose
 *
 * In tie_word_embeddings lm_head mode, weight is shared with embedding
 * so it's [vocab_size, hidden_dim] and used transposed:
 *   output = input * weight^T
 *
 * dotCl works on regular nntrainer::Tensor objects (it handles SVM
 * internally if available) so we use plain Tensors here.
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

  const float tol = 1e-4f;
  for (unsigned int c = 0; c < vocab_size; ++c) {
    EXPECT_NEAR(output_gpu.getValue(0, 0, 0, c),
                output_cpu.getValue(0, 0, 0, c), tol)
      << "Mismatch at vocab index=" << c;
  }
}

// ============================================================================
// Benchmarks (Qwen3-4B sized)
// See unittest_layers_embedding_cl.cpp for rationale on chosen dimensions.
// ============================================================================

namespace {
constexpr unsigned int kBenchVocab = 32000;
constexpr unsigned int kBenchEmbedDim = 2560; // Qwen3-4B hidden size
constexpr unsigned int kWarmupIters = 10;
constexpr unsigned int kBenchIters = 100;

void runTieWordEmbeddingBenchmark(const char *label, unsigned int num_tokens) {
  auto *cl_ctx = static_cast<nntrainer::ClContext *>(
    nntrainer::Engine::Global().getRegisteredContext("gpu"));

  size_t weight_bytes = kBenchVocab * kBenchEmbedDim * sizeof(float);
  size_t input_bytes = num_tokens * sizeof(float);
  size_t output_bytes = num_tokens * kBenchEmbedDim * sizeof(float);

  float *input_svm = (float *)allocateSVM(input_bytes);
  float *weight_svm = (float *)allocateSVM(weight_bytes);
  float *output_svm = (float *)allocateSVM(output_bytes);

  cl_ctx->command_queue_inst_.enqueueSVMMap(input_svm, input_bytes, false);
  cl_ctx->command_queue_inst_.enqueueSVMMap(weight_svm, weight_bytes, false);
  for (unsigned int i = 0; i < num_tokens; ++i) {
    input_svm[i] = static_cast<float>((i * 37) % kBenchVocab);
  }
  for (unsigned int r = 0; r < kBenchVocab; ++r) {
    for (unsigned int c = 0; c < kBenchEmbedDim; ++c) {
      weight_svm[r * kBenchEmbedDim + c] = (r * 0.001f) + (c * 0.0001f);
    }
  }
  cl_ctx->command_queue_inst_.enqueueSVMUnmap(input_svm);
  cl_ctx->command_queue_inst_.enqueueSVMUnmap(weight_svm);

  nntrainer::TieWordEmbeddingCl layer;

  for (unsigned int i = 0; i < kWarmupIters; ++i) {
    layer.embedding_cl_kernel(input_svm, weight_svm, output_svm, num_tokens,
                              kBenchEmbedDim, /*scale=*/1.0f, /*svm=*/true);
  }

  Timer timer{};
  for (unsigned int i = 0; i < kBenchIters; ++i) {
    layer.embedding_cl_kernel(input_svm, weight_svm, output_svm, num_tokens,
                              kBenchEmbedDim, /*scale=*/1.0f, /*svm=*/true);
  }
  const float total_ms = timer.GetElapsedMilliseconds();
  const float avg_ms = total_ms / static_cast<float>(kBenchIters);

  std::cout << "[Bench] TieWordEmbeddingGPU " << label
            << " (vocab=" << kBenchVocab << ", embed_dim=" << kBenchEmbedDim
            << ", num_tokens=" << num_tokens
            << ", iters=" << kBenchIters << "): "
            << avg_ms << " ms/call (total " << total_ms << " ms after "
            << kWarmupIters << " warmup)" << std::endl;

  freeSVM(input_svm);
  freeSVM(weight_svm);
  freeSVM(output_svm);
}
} // namespace

TEST(TieWordEmbeddingGPU_Bench, fp32_decode_single_token) {
  runTieWordEmbeddingBenchmark("decode", /*num_tokens=*/1);
}

TEST(TieWordEmbeddingGPU_Bench, fp32_prefill_128_tokens) {
  runTieWordEmbeddingBenchmark("prefill_128", /*num_tokens=*/128);
}

TEST(TieWordEmbeddingGPU_Bench, fp32_prefill_256_tokens) {
  runTieWordEmbeddingBenchmark("prefill_256", /*num_tokens=*/256);
}

TEST(TieWordEmbeddingGPU_Bench, fp32_prefill_512_tokens) {
  runTieWordEmbeddingBenchmark("prefill_512", /*num_tokens=*/512);
}
