// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   gemma4_causallm.cpp
 * @brief  Gemma4 CausalLM model for NNTrainer.
 */

#include <cmath>
#include <cstdio>
#include <stdexcept>

#include <app_context.h>
#include <engine.h>
#include <llm_util.hpp>

#include <gemma3_causallm.h>
#include <ple_block.h>
#include <reshaped_rms_norm.h>

#include "gemma4_causallm.h"

namespace causallm {

// ---------------------------------------------------------------------------
// Static helpers
// ---------------------------------------------------------------------------

json &Gemma4Transformer::sanitizeConfig(json &cfg) {
  // Gemma4 HuggingFace config nests fields under "text_config".
  // Promote them to top level so all downstream code finds them directly.
  if (cfg.contains("text_config") && cfg["text_config"].is_object()) {
    for (auto &[k, v] : cfg["text_config"].items()) {
      if (!cfg.contains(k) || cfg[k].is_null())
        cfg[k] = v;
    }
  }
  if (!cfg.contains("tie_word_embeddings"))
    cfg["tie_word_embeddings"] = true;
  // Ensure rope_theta is present (required by Transformer::setupParameters)
  if (!cfg.contains("rope_theta")) {
    // Local/sliding attention default
    cfg["rope_theta"] = 10000;
  }
  // Generate layer_types if not present (every sliding_window_pattern-th is global)
  if (!cfg.contains("layer_types") || cfg["layer_types"].is_null()) {
    const int n = cfg.value("num_hidden_layers", 30);
    const int pat = cfg.value("sliding_window_pattern", 6);
    std::vector<std::string> types;
    for (int i = 0; i < n; ++i) {
      bool is_global = ((i + 1) % pat == 0);
      types.push_back(is_global ? "full_attention" : "sliding_attention");
    }
    // Last layer is always global
    if (!types.empty())
      types.back() = "full_attention";
    cfg["layer_types"] = types;
  }
  return cfg;
}

json &Gemma4Transformer::sanitizeGenerationConfig(json &gen_cfg,
                                                  const json &cfg) {
  if (!gen_cfg.contains("eos_token_id")) {
    if (cfg.contains("eos_token_id")) {
      auto eos = cfg["eos_token_id"];
      if (eos.is_number())
        gen_cfg["eos_token_id"] = std::vector<unsigned int>{eos.get<unsigned int>()};
      else
        gen_cfg["eos_token_id"] = eos;
    }
  } else {
    auto eos = gen_cfg["eos_token_id"];
    if (eos.is_number())
      gen_cfg["eos_token_id"] = std::vector<unsigned int>{eos.get<unsigned int>()};
  }
  return gen_cfg;
}

void Gemma4Transformer::init_from_config(const json &cfg) {
  PLE_DIM = cfg.value("hidden_size_per_layer_input", 256);
  VOCAB_PER_LAYER =
    cfg.value("vocab_size_per_layer_input", cfg.value("vocab_size", 262144));
  GLOBAL_HEAD_DIM = cfg.value("global_head_dim", 512);
  GLOBAL_ROPE_THETA = cfg.value("global_rope_theta", 1000000.0f);
  NUM_GLOBAL_KEY_VALUE_HEADS =
    cfg.value("num_global_key_value_heads",
              cfg.value("num_key_value_heads", 4));
  NUM_KV_SHARED_LAYERS = cfg.value("num_kv_shared_layers", 0);

  if (cfg.contains("layer_types")) {
    // already set in sanitizeConfig, but re-read here for access
  }
}

void Gemma4Transformer::setupParameters(json &cfg, json &generation_cfg,
                                        json &nntr_cfg) {
  Transformer::setupParameters(cfg, generation_cfg, nntr_cfg);
  init_from_config(cfg);

  // Gemma4 uses attn_logit_softcapping (if any)
  if (cfg.contains("attn_logit_softcapping") &&
      !cfg["attn_logit_softcapping"].is_null()) {
    ATTN_LOGIT_SOFTCAPPING =
      cfg["attn_logit_softcapping"].get<float>();
  }
}

// ---------------------------------------------------------------------------
// Layer helpers
// ---------------------------------------------------------------------------

bool Gemma4Transformer::isGlobalLayer(int layer_id) const {
  if (layer_id == NUM_LAYERS - 1)
    return true; // last layer always global
  const unsigned int pat = SLIDING_WINDOW_PATTERN ? SLIDING_WINDOW_PATTERN : 6;
  return ((layer_id + 1) % pat == 0);
}

bool Gemma4Transformer::isKvSharedLayer(int layer_id) const {
  if (NUM_KV_SHARED_LAYERS <= 0)
    return false;
  const int first_shared = NUM_LAYERS - NUM_KV_SHARED_LAYERS;
  return layer_id >= first_shared;
}

// ---------------------------------------------------------------------------
// Model construction
// ---------------------------------------------------------------------------

std::vector<LayerHandle>
Gemma4Transformer::createTransformerDecoderBlock(const int layer_id,
                                                 std::string input_name) {
  std::vector<LayerHandle> layers;
  const std::string pfx = "layer" + std::to_string(layer_id);

  // ---- PLE gate block (prepended to every decoder block) ----
  const int vocab_per_layer = VOCAB_PER_LAYER > 0 ? VOCAB_PER_LAYER : NUM_VOCAB;
  const float ple_scale =
    (PLE_PROJ_SCALE < 0.0f) ? (1.0f / std::sqrt((float)PLE_DIM)) : PLE_PROJ_SCALE;
  const float inp_scale =
    (PLE_INPUT_SCALE < 0.0f) ? (1.0f / std::sqrt((float)PLE_DIM)) : PLE_INPUT_SCALE;

  // PLE takes: (hidden, initial_embeds=embedding0)
  // initial_embeds is the output of the global embedding0 layer
  const std::string ple_name = pfx + "_ple";
  std::vector<std::string> ple_params = {
    withKey("name", ple_name),
    withKey("in_dim", vocab_per_layer),
    withKey("out_dim", PLE_DIM),
    withKey("epsilon", std::to_string(NORM_EPS)),
    withKey("proj_scale", std::to_string(ple_scale)),
    withKey("input_scale", std::to_string(inp_scale)),
    withKey("input_layers", input_name + "," + INIT_EMBEDS_LAYER),
  };
  layers.push_back(createLayer("gemma4_ple_block", ple_params));

  // Attention pre-norm
  layers.push_back(createLayer(
    "rms_norm",
    {withKey("name", pfx + "_attention_norm"),
     withKey("input_layers", ple_name),
     withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("packed", "false")}));

  // Attention (handles global vs sliding, KV sharing)
  const bool global = isGlobalLayer(layer_id);
  const int head_dim = global ? GLOBAL_HEAD_DIM : HEAD_DIM;
  const int n_kv_heads =
    global ? NUM_GLOBAL_KEY_VALUE_HEADS : NUM_KEY_VALUE_HEADS;
  const int gqa = NUM_HEADS / n_kv_heads;

  auto att_norm_name = pfx + "_attention_norm";
  auto att_layers = createAttention(layer_id, INIT_SEQ_LEN, NUM_HEADS, head_dim,
                                    att_norm_name, att_norm_name, att_norm_name);
  layers.insert(layers.end(), att_layers.begin(), att_layers.end());

  // Post-attention norm + residual
  layers.push_back(createLayer(
    "rms_norm",
    {withKey("name", pfx + "_post_attention_norm"),
     withKey("input_layers", pfx + "_attention_out"),
     withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("packed", "false")}));

  layers.push_back(createLayer(
    "addition",
    {withKey("name", pfx + "_post_attention"),
     withKey("input_layers",
             ple_name + "," + pfx + "_post_attention_norm")}));

  // Pre-FFN norm
  layers.push_back(createLayer(
    "rms_norm",
    {withKey("name", pfx + "_pre_ffn_norm"),
     withKey("input_layers", pfx + "_post_attention"),
     withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("packed", "false")}));

  // FFN (same structure as Gemma3: gate + up + geglu + down)
  auto ffn = createMlp(layer_id, DIM, INTERMEDIATE_SIZE,
                       pfx + "_pre_ffn_norm");
  layers.insert(layers.end(), ffn.begin(), ffn.end());

  // Post-FFN norm + residual
  layers.push_back(createLayer(
    "rms_norm",
    {withKey("name", pfx + "_post_ffn_norm"),
     withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("packed", "false")}));

  layers.push_back(createLayer(
    "addition",
    {withKey("name", pfx + "_decoder_output"),
     withKey("input_layers",
             pfx + "_post_attention," + pfx + "_post_ffn_norm")}));

  return layers;
}

std::vector<LayerHandle>
Gemma4Transformer::createAttention(const int layer_id, int seq_len,
                                   int n_heads, int head_dim,
                                   std::string query_name,
                                   std::string key_name,
                                   std::string value_name) {
  std::vector<LayerHandle> layers;
  const std::string pfx = "layer" + std::to_string(layer_id);

  const bool global = isGlobalLayer(layer_id);
  const bool kv_shared = isKvSharedLayer(layer_id);
  const int n_kv = global ? NUM_GLOBAL_KEY_VALUE_HEADS : NUM_KEY_VALUE_HEADS;

  auto Q = pfx + "_wq";
  auto Q_norm = pfx + "_q_norm";
  auto K = pfx + "_wk";
  auto K_norm = pfx + "_k_norm";
  auto V = pfx + "_wv";
  auto A = pfx + "_attention";
  auto O = pfx + "_attention_out";

  // Q projection (always present, even for KV-shared layers)
  layers.push_back(createLayer(
    "fully_connected",
    {withKey("name", Q),
     withKey("unit", head_dim * n_heads),
     withKey("disable_bias", "true"),
     withKey("input_layers", query_name),
     withKey("weight_initializer", "ones"),
     withKey("weight_dtype", FC_LAYER_DTYPE)}));

  // K/V projections: skip for KV-shared layers, wire from non-shared source
  if (!kv_shared) {
    layers.push_back(createLayer(
      "fully_connected",
      {withKey("name", K),
       withKey("unit", head_dim * n_kv),
       withKey("disable_bias", "true"),
       withKey("input_layers", key_name),
       withKey("weight_initializer", "ones"),
       withKey("weight_dtype", FC_LAYER_DTYPE)}));

    layers.push_back(createLayer(
      "fully_connected",
      {withKey("name", V),
       withKey("unit", head_dim * n_kv),
       withKey("disable_bias", "true"),
       withKey("input_layers", value_name),
       withKey("weight_initializer", "ones"),
       withKey("weight_dtype", FC_LAYER_DTYPE)}));

    // Update last non-shared KV names for this attention type
    if (global)
      last_global_kv_layer_name = pfx;
    else
      last_sliding_kv_layer_name = pfx;
  }

  // Q norm (always present)
  layers.push_back(createLayer(
    "reshaped_rms_norm",
    {withKey("name", Q_norm),
     withKey("input_layers", Q),
     withKey("packed", "false"),
     withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("feature_size", std::to_string(head_dim))}));

  // K norm: only for non-KV-shared layers; shared layers reuse source k_norm output
  std::string k_norm_for_mha;
  if (!kv_shared) {
    layers.push_back(createLayer(
      "reshaped_rms_norm",
      {withKey("name", K_norm),
       withKey("input_layers", K),
       withKey("packed", "false"),
       withKey("epsilon", std::to_string(NORM_EPS)),
       withKey("feature_size", std::to_string(head_dim))}));
    k_norm_for_mha = K_norm;
  } else {
    k_norm_for_mha =
      (global ? last_global_kv_layer_name : last_sliding_kv_layer_name) +
      "_k_norm";
  }

  // MHA core
  const float rope_theta = global ? GLOBAL_ROPE_THETA : (float)ROPE_THETA;
  const unsigned int window =
    global ? UINT_MAX
           : (SLIDING_WINDOW < UINT_MAX ? SLIDING_WINDOW : UINT_MAX);

  const std::string v_src = kv_shared
    ? (global ? last_global_kv_layer_name : last_sliding_kv_layer_name) + "_wv"
    : V;

  std::vector<std::string> a_params = {
    withKey("name", A),
    withKey("num_heads", n_heads),
    withKey("num_heads_kv", n_kv),
    withKey("max_timestep", std::to_string(INIT_SEQ_LEN + NUM_TO_GENERATE)),
    withKey("sliding_window", window),
    withKey("rope_theta", std::to_string(rope_theta)),
    withKey("max_new_tokens", std::to_string(NUM_TO_GENERATE)),
    withKey("is_causal", IS_CAUSAL ? "true" : "false"),
    withKey("input_layers", {Q_norm, k_norm_for_mha, v_src})};
  layers.push_back(createLayer("mha_core", a_params));

  // O projection
  layers.push_back(createLayer(
    "fully_connected",
    {withKey("name", O),
     withKey("unit", DIM),
     withKey("disable_bias", "true"),
     withKey("input_layers", A),
     withKey("weight_initializer", "ones"),
     withKey("weight_dtype", FC_LAYER_DTYPE)}));

  return layers;
}

std::vector<LayerHandle>
Gemma4Transformer::createMlp(const int layer_id, int dim, int hidden_dim,
                             std::string input_name) {
  // Same structure as Gemma3 (gated GeLU MLP with separate gate/up/down)
  std::vector<LayerHandle> layers;
  const std::string pfx = "layer" + std::to_string(layer_id);

  // Gate projection + GeLU
  layers.push_back(createLayer(
    "fully_connected",
    {withKey("name", pfx + "_ffn_gate"),
     withKey("unit", hidden_dim),
     withKey("disable_bias", "true"),
     withKey("input_layers", input_name),
     withKey("weight_initializer", "ones"),
     withKey("weight_dtype", FC_LAYER_DTYPE)}));

  layers.push_back(createLayer(
    "activation",
    {withKey("name", pfx + "_ffn_gate_gelu"),
     withKey("activation", "tanh_gelu"),
     withKey("input_layers", pfx + "_ffn_gate")}));

  // Up projection
  layers.push_back(createLayer(
    "fully_connected",
    {withKey("name", pfx + "_ffn_up"),
     withKey("unit", hidden_dim),
     withKey("disable_bias", "true"),
     withKey("input_layers", input_name),
     withKey("weight_initializer", "ones"),
     withKey("weight_dtype", FC_LAYER_DTYPE)}));

  // Multiply (gate * up)
  layers.push_back(createLayer(
    "multiply",
    {withKey("name", pfx + "_ffn_geglu"),
     withKey("input_layers",
             pfx + "_ffn_gate_gelu," + pfx + "_ffn_up")}));

  // Down projection
  layers.push_back(createLayer(
    "fully_connected",
    {withKey("name", pfx + "_ffn_down"),
     withKey("unit", dim),
     withKey("disable_bias", "true"),
     withKey("input_layers", pfx + "_ffn_geglu"),
     withKey("weight_initializer", "ones"),
     withKey("weight_dtype", FC_LAYER_DTYPE)}));

  return layers;
}

void Gemma4Transformer::registerCustomLayers() {
  auto &ct_engine = nntrainer::Engine::Global();
  auto app_ctx =
    static_cast<nntrainer::AppContext *>(ct_engine.getRegisteredContext("cpu"));

  auto tryRegister = [&](auto factory) {
    try {
      app_ctx->registerFactory(factory);
    } catch (std::invalid_argument &) {
      // already registered, skip
    }
  };

  tryRegister(nntrainer::createLayer<causallm::ReshapedRMSNormLayer>);
  tryRegister(nntrainer::createLayer<causallm::Gemma4PleBlock>);
}

void Gemma4CausalLM::registerCustomLayers() {
  CausalLM::registerCustomLayers();
  Gemma4Transformer::registerCustomLayers();
}

} // namespace causallm
