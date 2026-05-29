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

  // Qwen3-4B QINT4 hardcoded (from config.json on device).
  causallm_gpu::Qwen3Config cfg;
  cfg.hidden_size = 2560;
  cfg.intermediate_size = 9728;
  cfg.head_dim = 128;
  cfg.num_heads_Q = 32;
  cfg.num_heads_KV = 8;
  cfg.num_layers = 36;
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

  std::fprintf(stderr,
               "[main] skeleton OK. Next commit: layer-0 forward.\n");
  return 0;
}
