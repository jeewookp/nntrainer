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
#include <layer_context.h>
#include <layers_common_tests.h>
#include <lm_head_cl.h>
#include <tensor.h>
#include <timer.h>
#include <var_grad.h>
#include <weight.h>

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
  const float tol = 1e-4f;
  for (unsigned int c = 0; c < vocab_size; ++c) {
    EXPECT_NEAR(output_gpu.getValue(0, 0, 0, c),
                output_cpu.getValue(0, 0, 0, c), tol)
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

  const float tol = 1e-3f;
  for (unsigned int c = 0; c < vocab_size; ++c) {
    EXPECT_NEAR(output_gpu.getValue(0, 0, 0, c),
                output_cpu.getValue(0, 0, 0, c), tol)
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

  const float tol = 1e-4f;
  for (unsigned int c = 0; c < vocab_size; ++c) {
    EXPECT_NEAR(output_gpu.getValue(0, 0, 0, c),
                output_cpu.getValue(0, 0, 0, c), tol)
      << "Mismatch at vocab index=" << c;
  }
}

// ============================================================================
// Benchmark (Qwen3-4B sized) — runs through LmHeadLayerCl::incremental_forwarding
// using a manually built RunLayerContext (option C).
//
// lm_head always processes only the LAST token regardless of input seq_len:
//   input  [1, 1, seq_len, hidden]
//   output [1, 1, 1,       vocab]
// So unlike embedding, lm_head's per-call cost depends on hidden_dim and
// vocab_size only — not num_tokens.
// ============================================================================

namespace {
constexpr unsigned int kBenchHidden = 2560; // Qwen3-4B hidden_size
constexpr unsigned int kBenchVocab = 32000; // reduced from 151936 (see embedding bench)
constexpr unsigned int kWarmupIters = 10;
constexpr unsigned int kBenchIters = 100;

void runLmHeadBenchmark(const char *label, unsigned int seq_len) {
  using namespace nntrainer;

  // 1. Create the layer
  auto layer = createLayer<LmHeadLayerCl>();
  layer->setProperty({"unit=" + std::to_string(kBenchVocab),
                      "disable_bias=true"});

  // 2. Build InitLayerContext: input shape [1, 1, seq_len, hidden]
  TensorDim in_dim({1, 1, seq_len, kBenchHidden},
                   {Tformat::NCHW, Tdatatype::FP32});
  std::array<std::string, 3> ttype = {"NCHW", "FP32", "FP32"};
  InitLayerContext init_ctx({in_dim}, {true}, false, "lm_head_bench", "", 0.0f,
                            ttype, 1.0f, ml::train::ExecutionMode::INFERENCE,
                            ml::train::LayerComputeEngine::GPU);

  // 3. Finalize → context is populated with weight specs + output dims
  layer->finalize(init_ctx);

  // 4. Allocate weights, inputs, outputs, tensors based on init_ctx specs.
  //    LmHead's finalize requests Initializer::NONE, but Weight ctor rejects
  //    that. Override to ZEROS so the bench can construct Weight standalone
  //    (real model goes through network_graph which avoids this path).
  std::vector<Weight> weights;
  for (auto &spec : init_ctx.getWeightsSpec()) {
    WeightSpec spec_ = spec;
    std::get<2>(spec_) = Initializer::ZEROS;
    weights.emplace_back(spec_, /*alloc_now=*/true);
    weights.back().getVariableRef().setRandUniform(-0.01f, 0.01f);
    weights.back().getGradientRef().setZero();
  }

  std::vector<Var_Grad> ins;
  for (auto &dim : init_ctx.getInputDimensions()) {
    ins.emplace_back(dim, Initializer::NONE, /*ng=*/true,
                     /*alloc_now=*/true, "in");
    ins.back().getVariableRef().setRandUniform(-1.0f, 1.0f);
  }

  std::vector<Var_Grad> outs;
  for (auto &spec : init_ctx.getOutSpecs()) {
    outs.emplace_back(spec.variable_spec.dim, Initializer::NONE,
                      /*ng=*/true, /*alloc_now=*/true, "out");
  }

  std::vector<Var_Grad> tensors;
  for (auto &spec : init_ctx.getTensorsSpec()) {
    tensors.emplace_back(spec, /*alloc_now=*/true);
  }

  // 5. Build RunLayerContext (view = vector of pointers)
  auto make_view = [](auto &vec) {
    using T = std::decay_t<decltype(vec[0])>;
    std::vector<T *> v;
    v.reserve(vec.size());
    for (auto &e : vec)
      v.push_back(&e);
    return v;
  };

  RunLayerContext rc("lm_head_bench", true, 0.0f, false, 1.0f, nullptr, false,
                     make_view(weights), make_view(ins), make_view(outs),
                     make_view(tensors));

  // 6. Warmup
  for (unsigned int i = 0; i < kWarmupIters; ++i) {
    layer->incremental_forwarding(rc, /*from=*/0, /*to=*/seq_len,
                                  /*training=*/false);
  }

  // 7. Measured runs
  Timer timer{};
  for (unsigned int i = 0; i < kBenchIters; ++i) {
    layer->incremental_forwarding(rc, /*from=*/0, /*to=*/seq_len,
                                  /*training=*/false);
  }
  const float total_ms = timer.GetElapsedMilliseconds();
  const float avg_ms = total_ms / static_cast<float>(kBenchIters);

  std::cout << "[Bench] LmHeadGPU " << label
            << " (hidden=" << kBenchHidden << ", vocab=" << kBenchVocab
            << ", seq_len=" << seq_len << ", iters=" << kBenchIters << "): "
            << avg_ms << " ms/call (total " << total_ms << " ms after "
            << kWarmupIters << " warmup)" << std::endl;
}
} // namespace

TEST(LmHeadGPU_Bench, fp32_decode) {
  // decode: single new token. lm_head only touches the last position.
  runLmHeadBenchmark("decode", /*seq_len=*/1);
}

TEST(LmHeadGPU_Bench, fp32_prefill_128) {
  runLmHeadBenchmark("prefill_128", /*seq_len=*/128);
}

TEST(LmHeadGPU_Bench, fp32_prefill_256) {
  runLmHeadBenchmark("prefill_256", /*seq_len=*/256);
}

TEST(LmHeadGPU_Bench, fp32_prefill_512) {
  runLmHeadBenchmark("prefill_512", /*seq_len=*/512);
}
