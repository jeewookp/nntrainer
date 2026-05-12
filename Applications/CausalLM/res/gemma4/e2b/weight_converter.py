## SPDX-License-Identifier: Apache-2.0
## Copyright (C) 2025 Samsung Electronics Co., Ltd. All Rights Reserved.
##
## @file weight_converter.py
## @brief weight conversion script for Gemma4 (E2B) model
## @author Eunju Yang <ej.yang@samsung.com>
##
## NNTrainer weight file order per decoder block:
##   [PLE]         embed_slice[V x P], model_proj_w[H x P], model_proj_norm[P],
##                 gate_w[H x P], down_proj_w[P x H], post_norm[H], layer_scalar[1]
##   [Attention]   attention_norm[H],
##                 v_proj[H x nkv*hd], k_proj[H x nkv*hd],
##                 k_norm[hd]  (non-KV-shared layers only),
##                 q_proj[H x nh*hd], q_norm[hd], o_proj[nh*hd x H]
##   [FFN]         post_attention_norm[H], pre_ffn_norm[H],
##                 gate_proj[H x I], up_proj[H x I], down_proj[I x H],
##                 post_ffn_norm[H]
## Final: model_norm[H], lm_head[V x H]
##
## KV shared layers (last num_kv_shared_layers): no v_proj, k_proj, k_norm

import argparse
import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM


def save_gemma4_for_nntrainer(params, config, dtype, file):
    """Convert and save weights as nntrainer binary format for Gemma4."""
    tcfg = getattr(config, "text_config", config)
    n_layers = tcfg.num_hidden_layers
    ple_dim = getattr(tcfg, "hidden_size_per_layer_input", 256)
    vocab_per_layer = getattr(tcfg, "vocab_size_per_layer_input", 262144)
    num_kv_shared_layers = getattr(tcfg, "num_kv_shared_layers", 0)
    first_shared_layer = n_layers - num_kv_shared_layers if num_kv_shared_layers > 0 else n_layers

    def save(tensor, is_rms=False):
        t = tensor.float() if hasattr(tensor, 'float') else tensor
        if is_rms:
            t = t + 1.0
        np.array(t, dtype=dtype).tofile(file)

    def save_proj(key):
        save(params[key].permute(1, 0))

    # ── Global embedding ───────────────────────────────────────────────────
    save(params["embed_tokens.weight"])

    # ── Per-layer embedding: [vocab_per_layer, n_layers * ple_dim] ─────────
    epl = params["embed_tokens_per_layer.weight"].float()   # [V, L*P]
    # Global model projection: [n_layers * ple_dim, hidden_size]
    mproj = params["per_layer_model_projection.weight"].float()  # [L*P, H]
    # Shared projection norm: [ple_dim]
    mproj_norm = params["per_layer_projection_norm.weight"]

    # ── Decoder layers ─────────────────────────────────────────────────────
    for i in range(n_layers):
        lp = f"layers.{i}."
        kv_shared = (i >= first_shared_layer)

        # ------ PLE block ------------------------------------------------
        # embed_slice: [V, P] from column slice of global epl
        save(epl[:, i * ple_dim:(i + 1) * ple_dim])
        # model_proj_w: [H, P] from row slice of global mproj, permuted [P,H]->[H,P]
        save(mproj[i * ple_dim:(i + 1) * ple_dim, :].permute(1, 0))
        # model_proj_norm: shared [P]
        save(mproj_norm, is_rms=True)
        # per_layer_input_gate: [P, H] -> permute -> [H, P]
        save_proj(f"{lp}per_layer_input_gate.weight")
        # per_layer_projection: [H, P] -> permute -> [P, H]
        save_proj(f"{lp}per_layer_projection.weight")
        save(params[f"{lp}post_per_layer_input_norm.weight"], is_rms=True)
        save(params[f"{lp}layer_scalar"])

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
    save(params["norm.weight"], is_rms=True)
    if "lm_head.weight" in params:
        save_proj("lm_head.weight")
    else:
        save(params["embed_tokens.weight"].permute(1, 0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Gemma4 weights to NNTrainer binary format")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output", type=str, default="nntr_gemma4_e2b_fp32.bin")
    parser.add_argument("--dtype", type=str, default="float32",
                        choices=["float32", "float16"])
    args = parser.parse_args()

    print(f"Loading model from: {args.model_path}")
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float32, trust_remote_code=True
    )
    model.eval()

    tcfg = getattr(config, "text_config", config)
    print(f"Architecture     : {config.architectures}")
    print(f"Layers           : {tcfg.num_hidden_layers}")
    print(f"Hidden size      : {tcfg.hidden_size}")
    print(f"PLE dim          : {getattr(tcfg, 'hidden_size_per_layer_input', 'N/A')}")
    print(f"Vocab per layer  : {getattr(tcfg, 'vocab_size_per_layer_input', 'N/A')}")
    print(f"KV shared layers : {getattr(tcfg, 'num_kv_shared_layers', 0)}")

    state_dict = model.state_dict()

    # Strip "model.language_model." or "language_model." prefix
    for pfx in ["model.language_model.", "language_model."]:
        if any(k.startswith(pfx) for k in state_dict):
            print(f"Stripping prefix : '{pfx}'")
            state_dict = {k[len(pfx):] if k.startswith(pfx) else k: v
                          for k, v in state_dict.items()}
            break

    with open(args.output, "wb") as f:
        save_gemma4_for_nntrainer(state_dict, config, args.dtype, f)

    size_mb = __import__("os").path.getsize(args.output) / (1024 * 1024)
    print(f"\nSaved: {args.output} ({size_mb:.1f} MB)")
