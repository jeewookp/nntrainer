// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   ple_block.h
 * @brief  Per-Layer Embedding (PLE) gated block for Gemma4.
 *
 * Implements the decoder-block-level PLE gate:
 *   ple_emb  = embed_slice[token_ids]                  [B, S, ple_dim]
 *   ple_proj = rms_norm(initial_embeds @ model_proj_w * proj_scale)
 *   ple_in   = (ple_proj + ple_emb) * input_scale      [B, S, ple_dim]
 *   gate     = gelu(hidden @ gate_w)                   [B, S, ple_dim]
 *   out      = rms_norm(gate * ple_in @ down_proj_w)   [B, S, H]
 *   output   = hidden + out
 *
 * Inputs:
 *   input[0]: hidden_states   [B, S, hidden_size]
 *   input[1]: initial_embeds  [B, S, hidden_size]  (embedding0 output)
 *
 * Token IDs must be set before each forward pass via setCurrentTokenIds().
 *
 * Weights (in requestWeight order = weight-file byte order):
 *   [0] embed_slice      [vocab_per_layer, ple_dim]
 *   [1] model_proj_w     [hidden_size, ple_dim]
 *   [2] model_proj_norm  [ple_dim]            (GemmaRMSNorm: stored as w+1)
 *   [3] gate_w           [hidden_size, ple_dim]
 *   [4] down_proj_w      [ple_dim, hidden_size]
 *   [5] post_norm        [hidden_size]         (GemmaRMSNorm: stored as w+1)
 *   [6] layer_scalar     [1]                  (scalar multiplier on PLE output)
 */

#ifndef __GEMMA4_PLE_BLOCK_H__
#define __GEMMA4_PLE_BLOCK_H__
#ifdef __cplusplus

#pragma once
#ifdef _WIN32
#define WIN_EXPORT __declspec(dllexport)
#else
#define WIN_EXPORT
#endif

#include <common_properties.h>
#include <layer_impl.h>
#include <vector>

namespace causallm {

WIN_EXPORT class Gemma4PleBlock : public nntrainer::LayerImpl {
public:
  WIN_EXPORT Gemma4PleBlock();
  WIN_EXPORT ~Gemma4PleBlock() = default;
  WIN_EXPORT Gemma4PleBlock(Gemma4PleBlock &&rhs) noexcept = default;
  WIN_EXPORT Gemma4PleBlock &operator=(Gemma4PleBlock &&rhs) = default;

  WIN_EXPORT void finalize(nntrainer::InitLayerContext &context) override;
  WIN_EXPORT void forwarding(nntrainer::RunLayerContext &context,
                             bool training) override;
  WIN_EXPORT void
  incremental_forwarding(nntrainer::RunLayerContext &context, unsigned int from,
                         unsigned int to, bool training) override;

  WIN_EXPORT void calcDerivative(nntrainer::RunLayerContext &context) override;
  WIN_EXPORT void exportTo(nntrainer::Exporter &exporter,
                           const ml::train::ExportMethods &method) const override;
  WIN_EXPORT const std::string getType() const override { return type; }
  WIN_EXPORT bool supportBackwarding() const override { return false; }
  WIN_EXPORT void setProperty(const std::vector<std::string> &values) override;

  /** Set token IDs for the current forward pass (call before forwarding). */
  static void setCurrentTokenIds(const unsigned int *ids, int seq_len);

  inline static const std::string type = "gemma4_ple_block";

private:
  // Layer properties
  std::tuple<nntrainer::props::InDim, nntrainer::props::OutDim,
             nntrainer::props::Epsilon>
    ple_props; // InDim=vocab_per_layer, OutDim=ple_dim, Epsilon=norm_eps

  float proj_scale{1.0f};   // per_layer_model_projection_scale
  float input_scale{1.0f};  // per_layer_input_scale

  // Weight indices
  unsigned int embed_w_idx{0};
  unsigned int model_proj_w_idx{0};
  unsigned int model_proj_norm_idx{0};
  unsigned int gate_w_idx{0};
  unsigned int down_proj_w_idx{0};
  unsigned int post_norm_idx{0};
  unsigned int layer_scalar_idx{0};

  // Thread-local token IDs set before each forward pass
  static thread_local std::vector<unsigned int> s_token_ids;
  static thread_local int s_seq_len;
};

} // namespace causallm

#endif /* __cplusplus */
#endif /* __GEMMA4_PLE_BLOCK_H__ */
