// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   gemma4_causallm.h
 * @brief  Gemma4 CausalLM implementation for NNTrainer.
 *
 * Differences from Gemma3:
 *  1. Per-Layer Embeddings (PLE): each decoder block gets a gated auxiliary
 *     signal derived from token embeddings + a model-level projection.
 *  2. Variable head_dim: sliding-attention layers use head_dim (default 256),
 *     global (full) attention layers use global_head_dim (default 512).
 *  3. Dual RoPE theta: 10 000 for sliding layers, 1 000 000 for global layers.
 *  4. KV cache sharing (num_kv_shared_layers): the last N layers reuse K/V
 *     projections from the nearest preceding non-shared layer of the same type.
 */

#ifndef __GEMMA4_CAUSAL_LM_H__
#define __GEMMA4_CAUSAL_LM_H__

#include <causal_lm.h>
#include <gemma3_causallm.h>
#include <ple_block.h>

namespace causallm {

class Gemma4Transformer : virtual public Transformer {
public:
  static constexpr const char *architectures = "Gemma4Transformer";

  Gemma4Transformer(json &cfg, json &generation_cfg, json &nntr_cfg) :
    Transformer(sanitizeConfig(cfg),
                sanitizeGenerationConfig(generation_cfg, cfg), nntr_cfg) {
    init_from_config(cfg);
  }

  virtual ~Gemma4Transformer() = default;

protected:
  static json &sanitizeConfig(json &cfg);
  static json &sanitizeGenerationConfig(json &gen_cfg, const json &cfg);
  void init_from_config(const json &cfg);

  // PLE parameters
  int PLE_DIM = 256;           // hidden_size_per_layer_input
  int VOCAB_PER_LAYER = 0;     // vocab_size_per_layer_input (0 = use NUM_VOCAB)
  float PLE_PROJ_SCALE = -1.f; // <0 = auto: 1/sqrt(ple_dim)
  float PLE_INPUT_SCALE = -1.f;

  // Global attention parameters
  int GLOBAL_HEAD_DIM = 512;             // head_dim for full-attention layers
  int NUM_GLOBAL_KEY_VALUE_HEADS = 0;    // 0 = use NUM_KEY_VALUE_HEADS
  float GLOBAL_ROPE_THETA = 1000000.0f;

  // KV sharing
  int NUM_KV_SHARED_LAYERS = 0;

  // Track last non-shared layer per type for KV wiring
  std::string last_sliding_kv_layer_name;
  std::string last_global_kv_layer_name;

  bool isGlobalLayer(int layer_id) const;
  bool isKvSharedLayer(int layer_id) const;

public:
  void setupParameters(json &cfg, json &generation_cfg,
                       json &nntr_cfg) override;

  void registerCustomLayers() override;

  std::vector<LayerHandle>
  createTransformerDecoderBlock(const int layer_id,
                                std::string input_name) override;

  std::vector<LayerHandle> createAttention(const int layer_id, int seq_len,
                                           int n_heads, int head_dim,
                                           std::string query_name,
                                           std::string key_name,
                                           std::string value_name) override;

  std::vector<LayerHandle> createMlp(const int layer_id, int dim,
                                     int hidden_dim,
                                     std::string input_name) override;

  /** Must be called before each forward pass to provide token IDs for PLE. */
  void setTokenIdsForPle(const unsigned int *ids, int seq_len) {
    Gemma4PleBlock::setCurrentTokenIds(ids, seq_len);
  }

private:
  // The embedding0 layer name used as initial_embeds for all PLE blocks
  static constexpr const char *INIT_EMBEDS_LAYER = "embedding0";
};

class Gemma4CausalLM : public CausalLM, public Gemma4Transformer {
public:
  static constexpr const char *architectures = "Gemma4ForCausalLM";

  Gemma4CausalLM(json &cfg, json &generation_cfg, json &nntr_cfg) :
    Transformer(Gemma4Transformer::sanitizeConfig(cfg),
                Gemma4Transformer::sanitizeGenerationConfig(generation_cfg, cfg),
                nntr_cfg),
    CausalLM(Gemma4Transformer::sanitizeConfig(cfg),
             Gemma4Transformer::sanitizeGenerationConfig(generation_cfg, cfg),
             nntr_cfg),
    Gemma4Transformer(Gemma4Transformer::sanitizeConfig(cfg),
                      Gemma4Transformer::sanitizeGenerationConfig(generation_cfg,
                                                                  cfg),
                      nntr_cfg) {}

  virtual ~Gemma4CausalLM() = default;

  void setupParameters(json &cfg, json &generation_cfg,
                       json &nntr_cfg) override {
    CausalLM::setupParameters(cfg, generation_cfg, nntr_cfg);
    Gemma4Transformer::setupParameters(cfg, generation_cfg, nntr_cfg);
  }

  void registerCustomLayers() override;
};

} // namespace causallm

#endif /* __GEMMA4_CAUSAL_LM_H__ */
