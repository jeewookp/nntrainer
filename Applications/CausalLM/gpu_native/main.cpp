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

#include <chrono>
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
  // Default: Qwen3-0.6B. NNTR_MODEL_4B=1 selects Qwen3-4B dims (hidden 2560,
  // 36 layers, 32 Q / 8 KV heads, inter 9728) — for the 8/4/4 coherence demo.
  const bool model_4b = []() {
    const char *e = std::getenv("NNTR_MODEL_4B");
    return e && std::atoi(e) != 0;
  }();
  if (model_4b) {
    cfg.hidden_size = 2560;
    cfg.intermediate_size = 9728;
    cfg.head_dim = 128;
    cfg.num_heads_Q = 32;
    cfg.num_heads_KV = 8;
    cfg.num_layers = 36;
  } else {
    cfg.hidden_size = 1024;
    cfg.intermediate_size = 3072;
    cfg.head_dim = 128;
    cfg.num_heads_Q = 16;
    cfg.num_heads_KV = 8;
    cfg.num_layers = 28;
  }
  cfg.vocab_size = 151936;
  cfg.max_seq_len = 20480;
  cfg.rms_norm_eps = 1e-6f;
  cfg.rope_theta = 1e6f;
  std::fprintf(stderr, "[main] model=%s hidden=%u L=%u hQ=%u\n",
               model_4b ? "Qwen3-4B" : "Qwen3-0.6B", cfg.hidden_size,
               cfg.num_layers, cfg.num_heads_Q);

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
  // sized for max_seq_len_used = 1024 so that the prefill measurement
  // at M=1024 doesn't overrun the cache (each layer writes M rows).
  // Per-layer cache memory at this size = 2 * 1024 * 8 * 128 * 2 =
  // 4 MB; 28 layers => 112 MB total SVM (fine on SD8 Elite).
  const unsigned int max_seq_len_used = 1024;
  auto NOW = []() { return std::chrono::steady_clock::now(); };
  auto MS = [](auto t1, auto t0) {
    return std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0)
             .count() / 1000.0;
  };

  auto t_load_start = NOW();
  size_t off = fwd.layers_start_offset();
  for (unsigned int L = 0; L < cfg.num_layers; ++L) {
    if (!fwd.load_layer(L, &off, max_seq_len_used)) {
      std::fprintf(stderr, "[main] load_layer(%u) failed\n", L);
      return 10 + (int)L;
    }
  }
  auto t_load_done = NOW();
  std::fprintf(stderr,
               "[main] all %u layers loaded in %.1f ms; final offset=%zu "
               "MB (file=%zu MB)\n",
               cfg.num_layers, MS(t_load_done, t_load_start),
               off / (1024 * 1024),
               fwd.weight_file_size() / (1024 * 1024));

  // Load output_norm gamma up front (sits at the file tail right after
  // layer 27). Doing it here makes the per-iteration timing below
  // exclude this one-time load.
  if (!fwd.load_output_norm(off)) {
    std::fprintf(stderr, "[main] load_output_norm failed\n");
    return 60;
  }

  constexpr unsigned int BOS_TOKEN = 151643;
  auto *cl = static_cast<nntrainer::ClContext *>(
    nntrainer::Engine::Global().getRegisteredContext("gpu"));
  cl_command_queue q = cl->command_queue_inst_.GetCommandQueue();
  const unsigned int H = cfg.hidden_size;

  // Step 9: run 3 iterations to measure timing breakdown + verify
  // determinism (same predicted token every run).
  constexpr int NUM_RUNS = 3;
  int prev_token = -1;
  bool deterministic = true;
  for (int run = 0; run < NUM_RUNS; ++run) {
    auto t_embed_start = NOW();
    cl_mem cur = fwd.embedding_lookup_to_fp32_clmem(BOS_TOKEN);
    if (cur == nullptr) {
      std::fprintf(stderr, "[main] embedding_lookup failed\n");
      return 50;
    }
    auto t_embed_done = NOW();

    // Ping-pong output buffers (persistent across the chain).
    // forward_one_layer_v2 takes caller-managed in/out — both
    // [hidden] fp32 cl_mems. Alternate which is "in" each layer.
    cl_int e;
    cl_context ctx2 = cl->context_inst_.GetContext();
    static cl_mem buf_a = nullptr, buf_b = nullptr;
    if (buf_a == nullptr) {
      buf_a = clCreateBuffer(ctx2, CL_MEM_READ_WRITE, H * sizeof(float),
                             nullptr, &e);
      buf_b = clCreateBuffer(ctx2, CL_MEM_READ_WRITE, H * sizeof(float),
                             nullptr, &e);
    }
    // Copy embedding-lookup `cur` into buf_a as the layer-0 input,
    // then release the per-iter `cur` (the lookup allocates fresh).
    clEnqueueCopyBuffer(q, cur, buf_a, 0, 0, H * sizeof(float), 0, nullptr,
                        nullptr);
    clReleaseMemObject(cur);
    cur = nullptr;

    auto t_chain_start = NOW();
    cl_mem layer_in = buf_a;
    cl_mem layer_out = buf_b;
    for (unsigned int L = 0; L < cfg.num_layers; ++L) {
      if (!fwd.forward_one_layer_v2(L, layer_in, layer_out, /*position=*/0)) {
        std::fprintf(stderr, "[main] forward_one_layer_v2(%u) failed\n", L);
        return 100 + (int)L;
      }
      std::swap(layer_in, layer_out);
    }
    // After the loop layer_in holds the final layer's output (swap ran 28x).
    cur = layer_in;
    auto t_chain_done = NOW();

    if (!fwd.run_output_norm(cur)) {
      std::fprintf(stderr, "[main] run_output_norm failed\n");
      return 61;
    }
    auto t_norm_done = NOW();

    int next_token = fwd.run_lm_head_and_argmax_cpu(cur);
    // cur points into the persistent buf_a/buf_b ping-pong pool —
    // intentionally NOT released here so the next iteration reuses it.
    if (next_token < 0) {
      std::fprintf(stderr, "[main] lm_head failed\n");
      return 70;
    }
    auto t_lm_done = NOW();

    const double t_embed   = MS(t_embed_done, t_embed_start);
    const double t_chain   = MS(t_chain_done, t_chain_start);
    const double t_norm    = MS(t_norm_done,  t_chain_done);
    const double t_lm      = MS(t_lm_done,    t_norm_done);
    const double t_total   = MS(t_lm_done,    t_embed_start);
    const double t_per_layer = t_chain / cfg.num_layers;
    const double tps = 1000.0 / t_total;

    std::fprintf(stderr,
                 "[run %d] embed=%.2f ms  chain=%.1f ms (%.2f ms/layer)  "
                 "out_norm=%.2f ms  lm_head=%.1f ms  TOTAL=%.1f ms  "
                 "(=> %.2f TPS effective)  -> token %d\n",
                 run, t_embed, t_chain, t_per_layer, t_norm, t_lm, t_total,
                 tps, next_token);

    if (run > 0 && next_token != prev_token) deterministic = false;
    prev_token = next_token;
  }

  std::fprintf(stderr,
               "\n[main] decode (M=1) summary:\n"
               "  predicted token over %d runs: %d (deterministic=%d)\n"
               "  baseline reference: CausalLM ~6.7 decode TPS "
               "(== ~150 ms/decode token) on the same SD8 Elite.\n",
               NUM_RUNS, prev_token, deterministic ? 1 : 0);

  // ===== Phase A #2: multi-token prefill timing (task #45) =====
  // Measure 28-layer chain at various M values. Initial input is the
  // BOS embedding replicated M times — semantically wrong for prefill
  // (real prefill needs M distinct tokens + per-position RoPE = task
  // #45b) but the kernel chain runs end-to-end so per-op wall time
  // is valid. Compare to baseline 1K prefill 460 TPS.
  std::fprintf(stderr,
               "\n[main] === Phase A #2: prefill timing (M>1) ===\n"
               "  NOTE: per-token RoPE not yet implemented (task #45b);\n"
               "  output token id is not meaningful for M>1 but per-op\n"
               "  timing is.\n");

  constexpr int PREFILL_MS[] = {2, 8, 64, 256, 512, 1024};
  // Allocate bigger ping-pong buffers + warm the scratch up to M=1024.
  const unsigned int M_max = 1024;
  if (!fwd.ensure_forward_scratch_allocated(M_max)) {
    std::fprintf(stderr, "[main] ensure_forward_scratch_allocated(M=%u) failed\n",
                 M_max);
    return 80;
  }
  cl_context ctx3 = cl->context_inst_.GetContext();
  cl_int e3 = CL_SUCCESS;
  cl_mem pf_in = clCreateBuffer(ctx3, CL_MEM_READ_WRITE,
                                (size_t)M_max * H * sizeof(float), nullptr,
                                &e3);
  cl_mem pf_out = clCreateBuffer(ctx3, CL_MEM_READ_WRITE,
                                 (size_t)M_max * H * sizeof(float), nullptr,
                                 &e3);
  if (e3 != CL_SUCCESS) {
    std::fprintf(stderr, "[main] prefill bufs alloc err=%d\n", e3);
    return 81;
  }
  // Load BOS embedding once on host.
  std::vector<float> bos_host(H);
  {
    cl_mem one = fwd.embedding_lookup_to_fp32_clmem(BOS_TOKEN);
    clEnqueueReadBuffer(q, one, CL_TRUE, 0, H * sizeof(float),
                        bos_host.data(), 0, nullptr, nullptr);
    clReleaseMemObject(one);
  }
  // Replicate it M_max times in host buffer then upload once.
  std::vector<float> rep_input((size_t)M_max * H);
  for (unsigned int m = 0; m < M_max; ++m)
    std::memcpy(rep_input.data() + (size_t)m * H, bos_host.data(),
                H * sizeof(float));

  for (int M_test : PREFILL_MS) {
    clEnqueueWriteBuffer(q, pf_in, CL_TRUE, 0,
                         (size_t)M_test * H * sizeof(float),
                         rep_input.data(), 0, nullptr, nullptr);
    // Enable per-stage profiling at M=256 (peak) and M=1024 (cliff)
    // so we can see WHERE the cliff time goes. Profiling adds a
    // clFinish per stage = overhead; total ms reported INCLUDES that
    // overhead but per-stage attribution is clean.
    const bool profile = (M_test == 256 || M_test == 1024);
    if (profile) {
      fwd.timings_.reset();
      fwd.profile_stages_ = true;
    } else {
      fwd.profile_stages_ = false;
    }
    auto t0 = NOW();
    cl_mem in_b = pf_in, out_b = pf_out;
    bool ok = true;
    double t_layer_0 = 0, t_layer_last = 0;
    for (unsigned int L = 0; L < cfg.num_layers; ++L) {
      auto tl0 = NOW();
      if (!fwd.forward_one_layer_v2(L, in_b, out_b, 0,
                                    (unsigned int)M_test)) {
        std::fprintf(stderr,
                     "[main] prefill M=%d layer %u failed\n", M_test, L);
        ok = false;
        break;
      }
      auto tl1 = NOW();
      if (L == 0) t_layer_0 = MS(tl1, tl0);
      if (L == cfg.num_layers - 1) t_layer_last = MS(tl1, tl0);
      std::swap(in_b, out_b);
    }
    auto t1 = NOW();
    if (!ok) continue;
    const double t_ms = MS(t1, t0);
    const double tps = (double)M_test * 1000.0 / t_ms;
    const double ms_per_token = t_ms / M_test;
    std::fprintf(stderr,
                 "[prefill M=%4d] chain=%7.1f ms  %.3f ms/token  "
                 "=> %7.1f TPS  (L0=%.1f ms, L27=%.1f ms)\n",
                 M_test, t_ms, ms_per_token, tps,
                 t_layer_0, t_layer_last);
    if (profile) {
      const auto &tt = fwd.timings_;
      const double sum = tt.pad_attn_norm_ms + tt.qkv_quant_image_ms +
                         tt.qkv_gemm_ms + tt.qk_norm_rope_ms +
                         tt.kv_write_ms + tt.attn_dispatch_ms +
                         tt.wo_ms + tt.ffn_ms;
      auto pct = [&](double v) { return sum > 0 ? 100.0 * v / sum : 0.0; };
      std::fprintf(stderr,
                   "  [stage timings, M=%d, %d layer-calls totaling %.0f ms]:\n"
                   "    (a) pad+attn_norm  %7.1f ms (%4.1f%%)\n"
                   "    (b) qkv quant+img  %7.1f ms (%4.1f%%)\n"
                   "    (c) Q/K/V GEMM     %7.1f ms (%4.1f%%)\n"
                   "    (d) qk_norm[+RoPE] %7.1f ms (%4.1f%%)\n"
                   "    (e) KV write SVM   %7.1f ms (%4.1f%%)\n"
                   "    (f) attention      %7.1f ms (%4.1f%%)\n"
                   "    (g) wo + resid_1   %7.1f ms (%4.1f%%)\n"
                   "    (h) ffn block      %7.1f ms (%4.1f%%)\n",
                   M_test, tt.calls, sum,
                   tt.pad_attn_norm_ms,   pct(tt.pad_attn_norm_ms),
                   tt.qkv_quant_image_ms, pct(tt.qkv_quant_image_ms),
                   tt.qkv_gemm_ms,        pct(tt.qkv_gemm_ms),
                   tt.qk_norm_rope_ms,    pct(tt.qk_norm_rope_ms),
                   tt.kv_write_ms,        pct(tt.kv_write_ms),
                   tt.attn_dispatch_ms,   pct(tt.attn_dispatch_ms),
                   tt.wo_ms,              pct(tt.wo_ms),
                   tt.ffn_ms,             pct(tt.ffn_ms));
    }
    // ALWAYS-ON host-bridge timing (env-gated print). The host_*_ms fields
    // accumulate the host wall-clock stalls of the SVM<->cl_mem bridges
    // across all layer calls of this forward. They are reset along with the
    // stage timings (timings_.reset() at the start of each profile M), so
    // the printed value is the clean per-forward total at this M.
    if (profile && std::getenv("NNTR_HOST_TIMING")) {
      const auto &tt = fwd.timings_;
      const double host_total =
        tt.host_kv_ms + tt.host_q_ms + tt.host_copy_svm_ms;
      std::fprintf(stderr,
                   "  [host-timing M=%d, %d layer-calls]: "
                   "kv_bridge=%.2f ms  q_bridge=%.2f ms  copy_svm=%.2f ms  "
                   "=> host_total=%.2f ms (%.1f%% of chain=%.1f ms)\n",
                   M_test, tt.calls, tt.host_kv_ms, tt.host_q_ms,
                   tt.host_copy_svm_ms, host_total,
                   t_ms > 0 ? 100.0 * host_total / t_ms : 0.0, t_ms);
    }
    // True on-device per-kernel GPU time (no-op unless NNTR_OPENCL_PROFILING
    // is set). Unlike the clFinish-bracketed stage timings above, this is
    // immune to out-of-order queue catch-up — it reads each kernel's own
    // CL_PROFILING_COMMAND_START/END. dumpProfile clears its event log each
    // call, so this captures exactly this M_test chain.
    {
      char ptag[32];
      std::snprintf(ptag, sizeof(ptag), "M=%d", M_test);
      cl->command_queue_inst_.dumpProfile(ptag);
    }
  }

  // ===== Prefill CORRECTNESS: greedy generation via repeated prefill =====
  // Real multi-token prefill (distinct tokens + per-position RoPE + causal
  // attention), re-prefilling the growing sequence each step and reading the
  // LAST position's logits. Validates that prefill output is now valid
  // (#47i fp32 swiglu fixed the last-layer fp16 overflow NaN).
  {
    std::fprintf(stderr,
                 "\n[main] === prefill correctness: greedy generation from BOS ===\n");
    cl_int eg = CL_SUCCESS;
    cl_mem last_row =
      clCreateBuffer(ctx3, CL_MEM_READ_WRITE, H * sizeof(float), nullptr, &eg);
    std::vector<int> seq{(int)BOS_TOKEN};
    const int GEN = 20;
    auto prefill_predict = [&](int read_row) -> int {
      const int M = (int)seq.size();
      for (int i = 0; i < M; ++i) {
        cl_mem em = fwd.embedding_lookup_to_fp32_clmem((unsigned int)seq[i]);
        if (!em) return -2;
        clEnqueueCopyBuffer(q, em, pf_in, 0, (size_t)i * H * sizeof(float),
                            H * sizeof(float), 0, nullptr, nullptr);
        clReleaseMemObject(em);
      }
      clFinish(q);
      cl_mem in_b = pf_in, out_b = pf_out;
      for (unsigned int L = 0; L < cfg.num_layers; ++L) {
        if (!fwd.forward_one_layer_v2(L, in_b, out_b, 0, (unsigned int)M))
          return -2;
        std::swap(in_b, out_b);
      }
      const int r = (read_row < 0) ? (M - 1) : read_row;
      clEnqueueCopyBuffer(q, in_b, last_row, (size_t)r * H * sizeof(float), 0,
                          H * sizeof(float), 0, nullptr, nullptr);
      clFinish(q);
      if (!fwd.run_output_norm(last_row)) return -2;
      return fwd.run_lm_head_and_argmax_cpu(last_row);
    };
    bool ok = true;
    for (int step = 0; step < GEN; ++step) {
      int nxt = prefill_predict(-1);
      if (nxt < 0) { std::fprintf(stderr, "  step %d: predict failed (%d)\n", step, nxt); ok = false; break; }
      if (step == 0)
        std::fprintf(stderr,
                     "  [self-consistency] prefill([BOS]) -> %d (decode=7212, match=%d)\n",
                     nxt, nxt == 7212);
      seq.push_back(nxt);
    }
    // Causal check: prefill([BOS,X]) row 0 must equal [BOS]-alone prediction.
    {
      std::vector<int> saved = seq; seq = {(int)BOS_TOKEN, 12345};
      int r0 = prefill_predict(0);
      std::fprintf(stderr, "  [causal] prefill([BOS,12345]) row0 -> %d (match=%d)\n",
                   r0, r0 == 7212);
      seq = saved;
    }
    std::fprintf(stderr, "  generated %zu token ids (greedy, no NaN=%d):\n   ",
                 seq.size(), ok);
    for (int t : seq) std::fprintf(stderr, " %d", t);
    std::fprintf(stderr, "\n");
    clReleaseMemObject(last_row);
  }

  clReleaseMemObject(pf_in);
  clReleaseMemObject(pf_out);

  std::fprintf(stderr,
               "\n[main] step #45 OK — multi-token prefill chain runs.\n");
  return 0;
}
