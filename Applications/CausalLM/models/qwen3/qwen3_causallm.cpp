/**
 * Copyright (C) 2025 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *   http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 *
 * @file	qwen3_causallm.cpp
 * @date	23 July 2025
 * @brief	This defines a qwen3 causal language model.
 * @see		https://github.com/nnstreamer/
 * @author	Eunju Yang <ej.yang@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */
#include <llm_util.hpp>
#include <model.h>
#include <qwen3_causallm.h>

#include <app_context.h>
#include <engine.h>
#include <reshaped_rms_norm.h>

namespace causallm {

std::vector<LayerHandle> Qwen3Transformer::createAttention(
  const int layer_id, int seq_len, int n_heads, int head_dim,
  std::string query_name, std::string key_name, std::string value_name) {

  std::vector<LayerHandle> layers;
  auto Q = "layer" + std::to_string(layer_id) + "_wq";
  auto Q_norm = "layer" + std::to_string(layer_id) + "_q_norm";
  auto K = "layer" + std::to_string(layer_id) + "_wk";
  auto K_norm = "layer" + std::to_string(layer_id) + "_k_norm";
  auto V = "layer" + std::to_string(layer_id) + "_wv";
  auto A = "layer" + std::to_string(layer_id) + "_attention";
  auto O = "layer" + std::to_string(layer_id) + "_attention_out";

  // NNTRAINER_QKV_FUSED=1 routes the 3 separate Q/K/V projections
  // through QKVLayer's batched dot, which fuses into a single
  // fused_gemv_int4 dispatch (same path used by gate_up_layer).
  // Output naming follows the multi-output convention:
  //   QKV(0) = Q  ->  reshaped_rms_norm (Q_norm)
  //   QKV(1) = K  ->  reshaped_rms_norm (K_norm)
  //   QKV(2) = V  ->  mha_core (no norm)
  // Bundle weight byte layout is preserved: QKVLayer registers
  // (q, k, v) weights in the same order as the legacy 3-FC sequence
  // so positional loading picks up the same bytes.
  static const bool s_qkv_fused =
    std::getenv("NNTRAINER_QKV_FUSED") != nullptr;
  if (s_qkv_fused) {
    auto QKV = "layer" + std::to_string(layer_id) + "_wqkv";
    std::vector<std::string> qkv_params = {
      withKey("name", QKV),
      withKey("q_unit", head_dim * n_heads),
      withKey("k_unit", head_dim * n_heads / GQA_SIZE),
      withKey("v_unit", head_dim * n_heads / GQA_SIZE),
      withKey("disable_bias", "true"),
      withKey("input_layers", query_name),
      withKey("weight_initializer", "ones")};
    layers.push_back(createLayer("qkv_layer", qkv_params));

    std::vector<std::string> q_norm_params = {
      withKey("name", Q_norm),
      withKey("input_layers", QKV + "(0)"),
      withKey("packed", "false"),
      withKey("epsilon", std::to_string(NORM_EPS)),
      withKey("feature_size", std::to_string(head_dim))};
    layers.push_back(createLayer("reshaped_rms_norm", q_norm_params));

    std::vector<std::string> k_norm_params = {
      withKey("name", K_norm),
      withKey("input_layers", QKV + "(1)"),
      withKey("packed", "false"),
      withKey("epsilon", std::to_string(NORM_EPS)),
      withKey("feature_size", std::to_string(head_dim))};
    layers.push_back(createLayer("reshaped_rms_norm", k_norm_params));

    // Rename V so the mha_core input list below picks up QKV(2) via V.
    V = QKV + "(2)";
  } else {
    // Q layer
    std::vector<std::string> q_params = {
      withKey("name", Q), withKey("unit", head_dim * n_heads),
      withKey("disable_bias", "true"), withKey("input_layers", query_name),
      withKey("weight_initializer", "ones")};
    layers.push_back(createLayer("fully_connected", q_params));

    // Q-reshaped-norm layer
    // q_norm(q_proj.view(hidden_shape))
    std::vector<std::string> q_norm_params = {
      withKey("name", Q_norm), withKey("input_layers", Q),
      withKey("packed", "false"), withKey("epsilon", std::to_string(NORM_EPS)),
      withKey("feature_size", std::to_string(head_dim))};
    layers.push_back(createLayer("reshaped_rms_norm", q_norm_params));

    // K layer
    std::vector<std::string> k_params = {
      withKey("name", K), withKey("unit", head_dim * n_heads / GQA_SIZE),
      withKey("disable_bias", "true"), withKey("input_layers", key_name),
      withKey("weight_initializer", "ones")};
    layers.push_back(createLayer("fully_connected", k_params));

    // K-reshaped-norm layer
    // k_norm(k_proj.view(hidden_shape))
    std::vector<std::string> k_norm_params = {
      withKey("name", K_norm), withKey("input_layers", K),
      withKey("packed", "false"), withKey("epsilon", std::to_string(NORM_EPS)),
      withKey("feature_size", std::to_string(head_dim))};
    layers.push_back(createLayer("reshaped_rms_norm", k_norm_params));

    // V layer
    std::vector<std::string> v_params = {
      withKey("name", V), withKey("unit", head_dim * n_heads / GQA_SIZE),
      withKey("disable_bias", "true"), withKey("input_layers", value_name),
      withKey("weight_initializer", "ones")};
    layers.push_back(createLayer("fully_connected", v_params));
  }

  // Attention core layer
  std::vector<std::string> a_params = {
    withKey("name", A),
    withKey("num_heads", n_heads),
    withKey("num_heads_kv", n_heads / GQA_SIZE),
    withKey("max_timestep", std::to_string(INIT_SEQ_LEN + NUM_TO_GENERATE)),
    withKey("sliding_window", SLIDING_WINDOW),
    withKey("rope_theta", ROPE_THETA),
    withKey("max_position_embeddings", MAX_POSITION_EMBEDDINGS),
    withKey("max_new_tokens", std::to_string(NUM_TO_GENERATE)),
    withKey("input_layers", {Q_norm, K_norm, V})};
  layers.push_back(createLayer("mha_core", a_params));

  // O layer
  std::vector<std::string> o_params = {
    withKey("name", O), withKey("unit", DIM), withKey("disable_bias", "true"),
    withKey("input_layers", A), withKey("weight_initializer", "ones")};
  layers.push_back(createLayer("fully_connected", o_params));

  return layers;
}

void Qwen3Transformer::registerCustomLayers() {
  ///
  auto &ct_engine = nntrainer::Engine::Global();
  auto app_context =
    static_cast<nntrainer::AppContext *>(ct_engine.getRegisteredContext("cpu"));

  try {
    app_context->registerFactory(
      nntrainer::createLayer<causallm::ReshapedRMSNormLayer>);
  } catch (std::invalid_argument &e) {
    std::cerr << "failed to register factory, reason: " << e.what()
              << std::endl;
  }
}

void Qwen3CausalLM::registerCustomLayers() {
  CausalLM::registerCustomLayers();
  Qwen3Transformer::registerCustomLayers();
}

} // namespace causallm
