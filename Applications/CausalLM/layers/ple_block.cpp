// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   ple_block.cpp
 * @brief  Per-Layer Embedding gated block for Gemma4.
 *
 * Per-decoder-block PLE gate (mirrors Gemma4TextDecoderLayer):
 *   emb       = embed_slice[token_id]
 *   proj      = rms_norm(initial_embeds @ model_proj_w * proj_scale)
 *   ple_in    = (proj + emb) * input_scale
 *   gate      = gelu(hidden @ gate_w)
 *   out       = rms_norm(gate * ple_in @ down_proj_w)
 *   output    = hidden + out
 */

#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <layer_context.h>
#include <nntrainer_error.h>
#include <node_exporter.h>

#include "ple_block.h"

namespace causallm {

thread_local std::vector<unsigned int> Gemma4PleBlock::s_token_ids;
thread_local int Gemma4PleBlock::s_seq_len = 0;

void Gemma4PleBlock::setCurrentTokenIds(const unsigned int *ids, int seq_len) {
  s_seq_len = seq_len;
  s_token_ids.assign(ids, ids + seq_len);
}

Gemma4PleBlock::Gemma4PleBlock() :
  LayerImpl(),
  ple_props(nntrainer::props::InDim(), nntrainer::props::OutDim(),
            nntrainer::props::Epsilon()) {}

void Gemma4PleBlock::finalize(nntrainer::InitLayerContext &context) {
  NNTR_THROW_IF(context.getNumInputs() != 2, std::invalid_argument)
    << "Gemma4PleBlock requires 2 inputs: hidden_states and initial_embeds";

  const auto &hidden_dim = context.getInputDimensions()[0];
  const unsigned int H = hidden_dim.width(); // hidden_size
  const unsigned int V =
    std::get<nntrainer::props::InDim>(ple_props).get();   // vocab_per_layer
  const unsigned int P =
    std::get<nntrainer::props::OutDim>(ple_props).get();  // ple_dim

  NNTR_THROW_IF(V == 0 || P == 0, std::invalid_argument)
    << "Gemma4PleBlock: in_dim (vocab_per_layer) and out_dim (ple_dim) must be >0";

  // Set default scales based on ple_dim if not explicitly set
  if (proj_scale < 0.0f)
    proj_scale = 1.0f / std::sqrt((float)P);
  if (input_scale < 0.0f)
    input_scale = 1.0f / std::sqrt((float)P);

  context.setOutputDimensions({hidden_dim});

  auto fp32_type = nntrainer::TensorDim::TensorType(
    context.getFormat(), nntrainer::TensorDim::DataType::FP32);

  // Weights in file-byte order (must match weight_converter.py)
  embed_w_idx = context.requestWeight(
    nntrainer::TensorDim(1, 1, V, P, fp32_type),
    nntrainer::props::InitializerInfo::Enum::NONE,
    nntrainer::WeightRegularizer::NONE, 1.0f, 0.0f, "embed_slice", false);

  model_proj_w_idx = context.requestWeight(
    nntrainer::TensorDim(1, 1, H, P, fp32_type),
    nntrainer::props::InitializerInfo::Enum::NONE,
    nntrainer::WeightRegularizer::NONE, 1.0f, 0.0f, "model_proj_w", false);

  model_proj_norm_idx = context.requestWeight(
    nntrainer::TensorDim(1, 1, 1, P, fp32_type),
    nntrainer::props::InitializerInfo::Enum::NONE,
    nntrainer::WeightRegularizer::NONE, 1.0f, 0.0f, "model_proj_norm", false);

  gate_w_idx = context.requestWeight(
    nntrainer::TensorDim(1, 1, H, P, fp32_type),
    nntrainer::props::InitializerInfo::Enum::NONE,
    nntrainer::WeightRegularizer::NONE, 1.0f, 0.0f, "gate_w", false);

  down_proj_w_idx = context.requestWeight(
    nntrainer::TensorDim(1, 1, P, H, fp32_type),
    nntrainer::props::InitializerInfo::Enum::NONE,
    nntrainer::WeightRegularizer::NONE, 1.0f, 0.0f, "down_proj_w", false);

  post_norm_idx = context.requestWeight(
    nntrainer::TensorDim(1, 1, 1, H, fp32_type),
    nntrainer::props::InitializerInfo::Enum::NONE,
    nntrainer::WeightRegularizer::NONE, 1.0f, 0.0f, "post_norm", false);
}

void Gemma4PleBlock::forwarding(nntrainer::RunLayerContext &context,
                                bool training) {
  const auto &hidden = context.getInput(0);
  const unsigned int total =
    (unsigned int)(hidden.batch() * hidden.channel() * hidden.height());
  incremental_forwarding(context, 0, total, training);
}

static void rms_norm_f32(float *x, const float *gamma, unsigned int N,
                         float eps) {
  float sq = 0.0f;
  for (unsigned int i = 0; i < N; ++i)
    sq += x[i] * x[i];
  const float inv = 1.0f / std::sqrt(sq / (float)N + eps);
  for (unsigned int i = 0; i < N; ++i)
    x[i] = x[i] * inv * gamma[i];
}

// Tanh-approximated GELU (matches PyTorch gelu_pytorch_tanh)
static inline float gelu_tanh(float x) {
  const float c = 0.7978845608028654f;
  const float a = 0.044715f;
  const float t = std::tanh(c * (x + a * x * x * x));
  return 0.5f * x * (1.0f + t);
}

void Gemma4PleBlock::incremental_forwarding(
  nntrainer::RunLayerContext &context, unsigned int from, unsigned int to,
  bool training) {
  const float eps = std::get<nntrainer::props::Epsilon>(ple_props).get();
  const unsigned int P =
    std::get<nntrainer::props::OutDim>(ple_props).get();  // ple_dim

  const nntrainer::Tensor &hidden_t = context.getInput(0);
  const nntrainer::Tensor &init_emb_t = context.getInput(1);
  nntrainer::Tensor &out_t = context.getOutput(0);

  const float *embed_data = context.getWeight(embed_w_idx).getData<float>();
  const float *mproj_data = context.getWeight(model_proj_w_idx).getData<float>();
  const float *mproj_norm = context.getWeight(model_proj_norm_idx).getData<float>();
  const float *gate_data = context.getWeight(gate_w_idx).getData<float>();
  const float *dproj_data = context.getWeight(down_proj_w_idx).getData<float>();
  const float *post_norm_g = context.getWeight(post_norm_idx).getData<float>();

  const unsigned int H = hidden_t.width();
  const unsigned int M = to - from;

  std::vector<float> h_row(H), ie_row(H), proj_v(P), gate_v(P), out_v(H);

  // Helper lambdas to extract a row from FP16 or FP32 tensor
  auto load_row = [&](const nntrainer::Tensor &t, unsigned int row,
                      float *dst) {
    if (t.getDataType() == ml::train::TensorDim::DataType::FP16) {
      const _FP16 *p = t.getData<_FP16>() + (size_t)row * H;
      for (unsigned int i = 0; i < H; ++i)
        dst[i] = (float)p[i];
    } else {
      std::memcpy(dst, t.getData<float>() + (size_t)row * H, H * sizeof(float));
    }
  };

  auto store_row = [&](nntrainer::Tensor &t, unsigned int row,
                       const float *src) {
    if (t.getDataType() == ml::train::TensorDim::DataType::FP16) {
      _FP16 *p = t.getData<_FP16>() + (size_t)row * H;
      for (unsigned int i = 0; i < H; ++i)
        p[i] = (_FP16)src[i];
    } else {
      std::memcpy(t.getData<float>() + (size_t)row * H, src, H * sizeof(float));
    }
  };

  for (unsigned int m = 0; m < M; ++m) {
    const unsigned int pos = from + m;
    const unsigned int tok =
      (pos < (unsigned int)s_seq_len && !s_token_ids.empty())
        ? s_token_ids[pos]
        : 0u;

    load_row(hidden_t, pos, h_row.data());
    load_row(init_emb_t, pos, ie_row.data());

    // 1. Token embedding: emb[j] = embed_slice[tok, j]
    const float *emb = embed_data + (size_t)tok * P;

    // 2. Model projection: proj = init_emb @ model_proj_w  [H x P]
    for (unsigned int j = 0; j < P; ++j) {
      float v = 0.0f;
      for (unsigned int i = 0; i < H; ++i)
        v += ie_row[i] * mproj_data[i * P + j];
      proj_v[j] = v * proj_scale;
    }

    // 3. RMSNorm on projection
    rms_norm_f32(proj_v.data(), mproj_norm, P, eps);

    // 4. Combined PLE input: (proj + emb) * input_scale
    for (unsigned int j = 0; j < P; ++j)
      proj_v[j] = (proj_v[j] + emb[j]) * input_scale;

    // 5. Gate: gate = gelu(hidden @ gate_w)  [H x P]
    for (unsigned int j = 0; j < P; ++j) {
      float v = 0.0f;
      for (unsigned int i = 0; i < H; ++i)
        v += h_row[i] * gate_data[i * P + j];
      gate_v[j] = gelu_tanh(v);
    }

    // 6. Gated multiply
    for (unsigned int j = 0; j < P; ++j)
      gate_v[j] *= proj_v[j];

    // 7. Down projection: out = gated @ down_proj_w  [P x H]
    for (unsigned int i = 0; i < H; ++i) {
      float v = 0.0f;
      for (unsigned int j = 0; j < P; ++j)
        v += gate_v[j] * dproj_data[j * H + i];
      out_v[i] = v;
    }

    // 8. Post RMSNorm
    rms_norm_f32(out_v.data(), post_norm_g, H, eps);

    // 9. Residual add
    for (unsigned int i = 0; i < H; ++i)
      out_v[i] += h_row[i];

    store_row(out_t, pos, out_v.data());
  }
}

void Gemma4PleBlock::calcDerivative(nntrainer::RunLayerContext &) {
  std::throw_with_nested(std::runtime_error("Training not supported."));
}

void Gemma4PleBlock::exportTo(nntrainer::Exporter &exporter,
                              const ml::train::ExportMethods &method) const {
  LayerImpl::exportTo(exporter, method);
  exporter.saveResult(ple_props, method, this);
}

void Gemma4PleBlock::setProperty(const std::vector<std::string> &values) {
  std::vector<std::string> remaining;
  for (const auto &v : values) {
    if (v.find("proj_scale=") == 0) {
      proj_scale = std::stof(v.substr(11));
    } else if (v.find("input_scale=") == 0) {
      input_scale = std::stof(v.substr(12));
    } else {
      remaining.push_back(v);
    }
  }
  auto unparsed = loadProperties(remaining, ple_props);
  LayerImpl::setProperty(unparsed);
}

} // namespace causallm
