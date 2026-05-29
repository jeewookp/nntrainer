// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   main.cpp
 * @date   29 May 2026
 * @brief  Entry point for the GPU-native Qwen3 forward binary
 *         (nntrainer_qwen3_gpu).
 *
 * Step 7b: end-to-end 28-layer chain via the generic load_layer +
 * forward_one_layer path. Old layer0_* methods are still in the .cpp
 * but unused — they'll go away when output_norm + lm_head land in
 * step 7c (which finishes the from-scratch inference pipeline up
 * through the first generated token).
 *
 * Usage:
 *   nntrainer_qwen3_gpu <weight_file_path>
 *
 * Qwen3-0.6B config is hardcoded (matches the verified production
 * QINT4 model on device).
 */

#include "qwen3_forward.h"

#include <cl_context.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <engine.h>
#include <limits>
#include <string>
#include <vector>

// Bypass the production safety gate in two_conv_attention_prefill_f16_cl.
// The from-scratch runtime explicitly accepts GPU baseline as reference
// (paper §3.6 same-numerics chain), so the existing "Using CPU mha"
// fallback in the production wrapper isn't useful here.
static struct EnvSetup {
  EnvSetup() { setenv("NNTR_MHA_VERIFY", "1", 1); }
} _env_setup;

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fprintf(stderr,
                 "usage: %s <weight_file_path>\n"
                 "  e.g. %s /data/local/tmp/nntrainer/causallm/models/"
                 "qwen3-0.6b-qint4-fresh/nntr_qwen3_0.6b_qint4.bin\n",
                 argv[0], argv[0]);
    return 1;
  }
  const std::string weight_path = argv[1];

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

  // RoPE freqs for position 0 (identity rotation; degenerate single-
  // token attention where softmax of one element = 1.0, attention
  // output = V per head). Precomputed once and reused across all 28
  // layers.
  if (!fwd.precompute_rope_for_position(0)) {
    std::fprintf(stderr, "[main] precompute_rope_for_position failed\n");
    return 3;
  }

  // Walk the weight file, loading all 28 layers. Per-layer KV cache
  // sized for max_seq_len_used = 8 (we only test position 0 today; 8
  // is round-up safety for future short prefills).
  const unsigned int max_seq_len_used = 8;
  size_t off = fwd.layers_start_offset();
  for (unsigned int L = 0; L < cfg.num_layers; ++L) {
    if (!fwd.load_layer(L, &off, max_seq_len_used)) {
      std::fprintf(stderr, "[main] load_layer(%u) failed\n", L);
      return 10 + (int)L;
    }
  }
  std::fprintf(stderr,
               "[main] all %u layers loaded; final offset=%zu MB "
               "(file=%zu MB)\n",
               cfg.num_layers, off / (1024 * 1024),
               fwd.weight_file_size() / (1024 * 1024));

  // Initial layer-0 input: real embedding lookup. Use Qwen3's BOS token
  // (151643 per HF config.json) so the first generated token has actual
  // semantic meaning (vs the synthetic ramp pattern). The lookup is CPU-
  // side Q6_K dequant + clCreateBuffer into a fresh cl_mem fp32 buffer.
  constexpr unsigned int BOS_TOKEN = 151643;
  auto *cl = static_cast<nntrainer::ClContext *>(
    nntrainer::Engine::Global().getRegisteredContext("gpu"));
  cl_command_queue q = cl->command_queue_inst_.GetCommandQueue();
  const unsigned int H = cfg.hidden_size;
  cl_mem cur = fwd.embedding_lookup_to_fp32_clmem(BOS_TOKEN);
  if (cur == nullptr) {
    std::fprintf(stderr, "[main] embedding_lookup failed\n");
    return 50;
  }

  // Chain through all 28 layers. Each forward_one_layer returns a fresh
  // cl_mem; release the previous one after the call.
  for (unsigned int L = 0; L < cfg.num_layers; ++L) {
    cl_mem nxt = fwd.forward_one_layer(L, cur, /*position=*/0);
    if (nxt == nullptr) {
      std::fprintf(stderr, "[main] forward_one_layer(%u) failed\n", L);
      clReleaseMemObject(cur);
      return 100 + (int)L;
    }
    clReleaseMemObject(cur);
    cur = nxt;
    // Periodic progress + sanity summary every 7 layers.
    if (L == 0 || L == 6 || L == 13 || L == 20 || L == cfg.num_layers - 1) {
      std::vector<float> r(H);
      clEnqueueReadBuffer(q, cur, CL_TRUE, 0, H * sizeof(float), r.data(),
                          0, nullptr, nullptr);
      bool finite = true;
      float mn = std::numeric_limits<float>::infinity();
      float mx = -mn;
      for (float v : r) {
        if (!std::isfinite(v)) finite = false;
        if (v < mn) mn = v;
        if (v > mx) mx = v;
      }
      std::fprintf(stderr,
                   "[chain] after layer %2u  first 4: %g %g %g %g  "
                   "min=%g max=%g finite=%d\n",
                   L, r[0], r[1], r[2], r[3], mn, mx, finite);
    }
  }

  // Step 7c: output_norm. Loads the final gamma at the tail of the
  // weight file (right after layer 27's last byte; `off` was advanced
  // by the load loop) and applies rmsnorm.cl in place. Post-norm the
  // residual stream magnitude collapses from ~10^4 back to ~[-2, +2]
  // (RMS-normalized) — that's the shape lm_head expects.
  if (!fwd.load_output_norm(off)) {
    std::fprintf(stderr, "[main] load_output_norm failed\n");
    clReleaseMemObject(cur);
    return 60;
  }
  if (!fwd.run_output_norm(cur)) {
    std::fprintf(stderr, "[main] run_output_norm failed\n");
    clReleaseMemObject(cur);
    return 61;
  }
  {
    std::vector<float> r(H);
    clEnqueueReadBuffer(q, cur, CL_TRUE, 0, H * sizeof(float), r.data(), 0,
                        nullptr, nullptr);
    bool finite = true;
    float mn = std::numeric_limits<float>::infinity();
    float mx = -mn;
    for (float v : r) {
      if (!std::isfinite(v)) finite = false;
      if (v < mn) mn = v;
      if (v > mx) mx = v;
    }
    std::fprintf(stderr,
                 "[qwen3-gpu] POST output_norm fp32 N=%u first 8:", H);
    for (int i = 0; i < 8; ++i) std::fprintf(stderr, " %g", r[i]);
    std::fprintf(stderr, "\n  last 4:");
    for (int i = 0; i < 4; ++i) std::fprintf(stderr, " %g", r[H - 4 + i]);
    std::fprintf(stderr,
                 "\n  min=%g max=%g all_finite=%d\n", mn, mx, finite);
  }

  // Step 8: lm_head (Q6_K tied embedding, CPU) + argmax → first token id.
  int next_token = fwd.run_lm_head_and_argmax_cpu(cur);
  clReleaseMemObject(cur);
  if (next_token < 0) {
    std::fprintf(stderr, "[main] lm_head failed\n");
    return 70;
  }

  std::fprintf(stderr,
               "[main] step 8 OK — END-TO-END INFERENCE COMPLETE.\n"
               "  input  token: %u (BOS)\n"
               "  output token: %d\n"
               "  pipeline: Q6_K embed lookup -> 28 GPU layers "
               "(rmsnorm/QKV/qknorm/RoPE/SVM KV cache/attention/wo/"
               "residual_1/ffn_norm/ffn_up/gate/swiglu/ffn_down/"
               "residual_2) -> output_norm -> CPU Q6_K lm_head -> "
               "argmax.\n",
               BOS_TOKEN, next_token);
  return 0;
}
