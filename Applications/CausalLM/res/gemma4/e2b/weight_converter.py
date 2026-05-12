## SPDX-License-Identifier: Apache-2.0
## Copyright (C) 2025 Samsung Electronics Co., Ltd. All Rights Reserved.
##
## @file weight_converter.py
## @brief weight conversion script for Gemma4 (E2B) model
## @author Eunju Yang <ej.yang@samsung.com>
##
## NNTrainer weight file order per decoder block:
##   [PLE]         embed_slice[V x P], model_proj_w[H x P], model_proj_norm[P],
##                 gate_w[H x P], down_proj_w[P x H], post_norm[H]
##   [Attention]   attention_norm[H],
##                 v_proj[H x nkv*hd], k_proj[H x nkv*hd] (skipped for kv-shared),
##                 k_norm[hd], q_proj[H x nh*hd], q_norm[hd], o_proj[nh*hd x H]
##   [FFN]         post_attention_norm[H], pre_ffn_norm[H],
##                 gate_proj[H x I], up_proj[H x I], down_proj[I x H],
##                 post_ffn_norm[H]
## Final: model_norm[H], lm_head[V x H]

import argparse
import sys
import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM


def save_gemma4_for_nntrainer(params, config, dtype, file):
    """Convert and save weights as nntrainer binary format for Gemma4."""
    # Gemma4ForConditionalGeneration wraps the text model under text_config
    tcfg = getattr(config, "text_config", config)
    n_layers = tcfg.num_hidden_layers
    ple_dim = getattr(tcfg, "hidden_size_per_layer_input", 256)
    vocab_per_layer = getattr(tcfg, "vocab_size_per_layer_input", 262144)
    num_kv_shared_layers = getattr(tcfg, "num_kv_shared_layers", 0)

    first_shared_layer = n_layers - num_kv_shared_layers if num_kv_shared_layers > 0 else n_layers

    def save(tensor, is_rms=False):
        t = tensor
        if is_rms:
            t = t + 1.0
        np.array(t, dtype=dtype).tofile(file)

    def save_proj(key):
        save(params[key].permute(1, 0))

    # ── Global embedding (shared vocabulary) ──────────────────────────────
    save(params["model.embed_tokens.weight"])

    # ── Per-layer embedding tensor ─────────────────────────────────────────
    # HuggingFace stores embed_tokens_per_layer with one of:
    #   shape [vocab_per_layer, num_layers, ple_dim]  → slice [:, i, :]
    #   shape [num_layers, vocab_per_layer, ple_dim]  → slice [i, :, :]
    # We auto-detect based on the first dimension.
    epl_weight = params["model.embed_tokens_per_layer.weight"].float()
    if epl_weight.dim() == 2:
        # Flat [vocab_per_layer * num_layers, ple_dim] → reshape
        epl_weight = epl_weight.reshape(n_layers, vocab_per_layer, ple_dim)
        def embed_slice(i):
            return epl_weight[i]                           # [vocab_per_layer, ple_dim]
    elif epl_weight.shape[0] == n_layers:
        # [num_layers, vocab_per_layer, ple_dim]
        def embed_slice(i):
            return epl_weight[i]
    else:
        # [vocab_per_layer, num_layers, ple_dim]
        def embed_slice(i):
            return epl_weight[:, i, :]

    # ── Decoder layers ─────────────────────────────────────────────────────
    for i in range(n_layers):
        lp = f"model.layers.{i}."
        kv_shared = (i >= first_shared_layer)

        # ------ PLE block ------------------------------------------------
        save(embed_slice(i))
        # per_layer_model_projection: Linear(hidden → ple_dim), weight [P, H] → [H, P]
        save_proj(f"{lp}per_layer_model_projection.weight")
        save(params[f"{lp}per_layer_projection_norm.weight"], is_rms=True)
        # per_layer_input_gate: Linear(hidden → ple_dim), weight [P, H] → [H, P]
        save_proj(f"{lp}per_layer_input_gate.weight")
        # per_layer_projection: Linear(ple_dim → hidden), weight [H, P] → [P, H]
        save_proj(f"{lp}per_layer_projection.weight")
        save(params[f"{lp}post_per_layer_input_norm.weight"], is_rms=True)

        # ------ Attention ------------------------------------------------
        save(params[f"{lp}input_layernorm.weight"], is_rms=True)

        if not kv_shared:
            save_proj(f"{lp}self_attn.v_proj.weight")
            save_proj(f"{lp}self_attn.k_proj.weight")

        save(params[f"{lp}self_attn.k_norm.weight"], is_rms=True)
        save_proj(f"{lp}self_attn.q_proj.weight")
        save(params[f"{lp}self_attn.q_norm.weight"], is_rms=True)
        save_proj(f"{lp}self_attn.o_proj.weight")

        # ------ FFN -------------------------------------------------------
        save(params[f"{lp}post_attention_layernorm.weight"], is_rms=True)
        save(params[f"{lp}pre_feedforward_layernorm.weight"], is_rms=True)
        save_proj(f"{lp}mlp.gate_proj.weight")
        save_proj(f"{lp}mlp.up_proj.weight")
        save_proj(f"{lp}mlp.down_proj.weight")
        save(params[f"{lp}post_feedforward_layernorm.weight"], is_rms=True)

    # ── Final norm + LM head ───────────────────────────────────────────────
    save(params["model.norm.weight"], is_rms=True)

    # Gemma4 uses tied embeddings by default; lm_head may not be present.
    if "lm_head.weight" in params:
        save_proj("lm_head.weight")
    else:
        # tied: lm_head = embed_tokens^T  →  save transposed embed_tokens
        save(params["model.embed_tokens.weight"].permute(1, 0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Gemma4 weights to NNTrainer binary format")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to HuggingFace Gemma4 model directory")
    parser.add_argument("--output", type=str, default="nntr_gemma4_e2b_fp32.bin",
                        help="Output binary file path")
    parser.add_argument("--dtype", type=str, default="float32",
                        choices=["float32", "float16"],
                        help="Output weight dtype")
    args = parser.parse_args()

    print(f"Loading model from: {args.model_path}")
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float32, trust_remote_code=True
    )
    model.eval()

    tcfg = getattr(config, "text_config", config)
    print(f"Architecture: {config.architectures}")
    print(f"Layers: {tcfg.num_hidden_layers}")
    print(f"Hidden size: {tcfg.hidden_size}")
    print(f"PLE dim: {getattr(tcfg, 'hidden_size_per_layer_input', 'N/A')}")
    print(f"Vocab per layer: {getattr(tcfg, 'vocab_size_per_layer_input', 'N/A')}")
    print(f"KV shared layers: {getattr(tcfg, 'num_kv_shared_layers', 0)}")

    state_dict = model.state_dict()

    # Gemma4ForConditionalGeneration stores text weights under "language_model." prefix
    # Strip it so the converter can use standard "model.*" / "lm_head.*" keys
    prefixes = ["language_model.", "model.language_model."]
    for pfx in prefixes:
        if any(k.startswith(pfx) for k in state_dict):
            print(f"Stripping state_dict prefix: '{pfx}'")
            state_dict = {k[len(pfx):] if k.startswith(pfx) else k: v
                          for k, v in state_dict.items()}
            break

    # Print parameter names (for debugging shape issues)
    ple_keys = [k for k in state_dict.keys() if "per_layer" in k or "embed_tokens_per_layer" in k]
    print("\nPLE-related parameters:")
    for k in ple_keys[:10]:
        print(f"  {k}: {state_dict[k].shape}")
    if len(ple_keys) > 10:
        print(f"  ... and {len(ple_keys) - 10} more")

    print("\nFirst 10 keys in state_dict:")
    for k in list(state_dict.keys())[:10]:
        print(f"  {k}: {state_dict[k].shape}")

    with open(args.output, "wb") as f:
        save_gemma4_for_nntrainer(state_dict, config, args.dtype, f)

    file_size_mb = __import__("os").path.getsize(args.output) / (1024 * 1024)
    print(f"\nSaved: {args.output} ({file_size_mb:.1f} MB)")
