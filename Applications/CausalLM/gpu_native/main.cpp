// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   main.cpp
 * @date   29 May 2026
 * @brief  Entry point for the GPU-native Qwen3 forward binary
 *         (nntrainer_qwen3_gpu).
 *
 * First commit: just init, dump weight-file header, run SVM round-trip
 * smoke test. No forward yet. Next commits add layer-0 forward, then
 * full 36 layers + lm_head.
 *
 * Usage:
 *   nntrainer_qwen3_gpu <weight_file_path>
 *
 * The Qwen3-4B config is hardcoded for now (matches the production
 * QINT4 model on device). When more configs are needed we'll parse
 * config.json — not in scope for the skeleton commit.
 */

#include "qwen3_forward.h"

#include <cstdio>
#include <cstdlib>
#include <string>

// Bypass the production safety gate in two_conv_attention_prefill_f16_cl:
// the existing CausalLM defaults to CPU fallback when NNTR_MHA_GPU=1 (to
// avoid chain drift on Qwen3-0.6B). NNTR_MHA_VERIFY=1 is the documented
// opt-in to actually run the GPU kernel. The from-scratch runtime is the
// intended consumer (paper §3.6 same-numerics chain — we accept GPU
// baseline as the reference, not bit-equality to CPU).
static struct EnvSetup {
  EnvSetup() { setenv("NNTR_MHA_VERIFY", "1", 1); }
} _env_setup;

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fprintf(stderr,
                 "usage: %s <weight_file_path>\n"
                 "  e.g. %s /data/local/tmp/nntrainer/causallm/models/"
                 "qwen3-4b/nntr_qwen3-4b-q6_K-qint4-idx3-fp32-arm.bin\n",
                 argv[0], argv[0]);
    return 1;
  }
  const std::string weight_path = argv[1];

  // Qwen3-0.6B QINT4 hardcoded (from config.json on device). The 4B
  // file `nntr_qwen3-4b-...idx3-fp32-arm.bin` was preferred but it
  // currently fails the production load path with "QINT4 Dot on CPU
  // only supports PER_CHANNEL_AFFINE or KAI_QSI4CXP_4x4x32 scheme"
  // — invalid qscheme bytes. The 0.6B model is the verified production
  // QINT4 path and the same kernels target both, so the chain we build
  // here transfers to the 4B model once it's re-quantized properly.
  causallm_gpu::Qwen3Config cfg;
  cfg.hidden_size = 1024;
  cfg.intermediate_size = 3072;
  cfg.head_dim = 128;
  cfg.num_heads_Q = 16;
  cfg.num_heads_KV = 8;
  cfg.num_layers = 28;
  cfg.vocab_size = 151936;
  cfg.max_seq_len = 20480;
  cfg.rms_norm_eps = 1e-6f;
  cfg.rope_theta = 1e6f;

  causallm_gpu::Qwen3Forward fwd;
  if (!fwd.init(cfg, weight_path)) {
    std::fprintf(stderr, "[main] init failed\n");
    return 2;
  }
  fwd.dump_weight_header(64);

  // SVM round-trip: 256 KB, well within any sane SVM budget.
  if (!fwd.svm_smoke_test(256 * 1024)) {
    std::fprintf(stderr, "[main] svm_smoke_test failed\n");
    return 3;
  }

  // Step 2: load layer 0 attention_norm gamma into SVM + run rmsnorm.cl
  // on a deterministic input pattern + verify output is finite. Proves
  // the (weight-mmap -> SVM -> GPU kernel -> read back) data path works
  // end-to-end with a single op.
  if (!fwd.load_layer0_attention_norm_to_svm()) {
    std::fprintf(stderr, "[main] load_layer0_attention_norm failed\n");
    return 4;
  }
  if (!fwd.run_rmsnorm_layer0()) {
    std::fprintf(stderr, "[main] run_rmsnorm_layer0 failed\n");
    return 5;
  }

  // Step 4: load Q/K/V projection weights together + run all three FCs
  // against the SAME rmsnorm output (shared activation quant — single
  // quantize_act, three GEMMs). Outputs Q[hQ*d], K[hKV*d], V[hKV*d] —
  // ready inputs for q_norm/k_norm/RoPE/attention (next commits).
  if (!fwd.load_layer0_qkv_weights()) {
    std::fprintf(stderr, "[main] load_layer0_qkv_weights failed\n");
    return 6;
  }
  if (!fwd.load_layer0_wo()) {
    std::fprintf(stderr, "[main] load_layer0_wo failed\n");
    return 11;
  }
  if (!fwd.load_layer0_ffn_weights()) {
    std::fprintf(stderr, "[main] load_layer0_ffn_weights failed\n");
    return 12;
  }
  if (!fwd.load_layer0_qk_norm_gammas()) {
    std::fprintf(stderr, "[main] load_layer0_qk_norm_gammas failed\n");
    return 7;
  }
  // Step 5 attention check: position=0 makes N_kv=1 in the cache, which
  // is degenerate single-token attention (softmax of one element = 1.0).
  // The expected output per head_q is exactly V[head_q / gqa] — easy
  // bit-pattern check vs the post-projection V values.
  if (!fwd.precompute_rope_for_position(0)) {
    std::fprintf(stderr, "[main] precompute_rope_for_position failed\n");
    return 8;
  }
  if (!fwd.allocate_layer0_kv_cache_svm()) {
    std::fprintf(stderr, "[main] allocate_layer0_kv_cache_svm failed\n");
    return 9;
  }
  if (!fwd.run_layer0_qkv_projection()) {
    std::fprintf(stderr,
                 "[main] run_layer0_qkv_projection failed (attention path)\n");
    return 10;
  }

  if (!fwd.run_layer0_ffn()) {
    std::fprintf(stderr, "[main] run_layer0_ffn failed\n");
    return 13;
  }

  std::fprintf(stderr,
               "[main] step 6 OK (layer 0 full forward complete: rmsnorm -> "
               "QKV -> q_norm/k_norm -> RoPE -> KV cache -> attention -> "
               "wo + residual_1 -> ffn_norm -> ffn_up/gate -> swiglu -> "
               "ffn_down -> residual_2 -> layer 0 output).\n"
               "Next: chain 28 layers + lm_head + first token sampling.\n");
  return 0;
}
