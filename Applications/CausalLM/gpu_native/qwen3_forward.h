// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   qwen3_forward.h
 * @date   29 May 2026
 * @brief  Paper-aligned GPU-native Qwen3 forward.
 *
 * Bypasses the nntrainer layer graph. Activations live in SVM cl_mem from
 * embedding through lm_head; no host round-trip per layer. Uses the
 * existing nntrainer kernels (v8c FC, rmsnorm.cl, rotary_emb.cl,
 * two_conv_attention.cl) as a "kernel library" — not the layer runtime.
 *
 * Scope of the first commit: skeleton + weight-file mmap + CL context
 * init + single SVM round-trip smoke test. Layer-0 forward is the next
 * commit; full 36-layer + lm_head is the one after.
 *
 * The acceptance criterion for "paper-level" is NOT bit-equal to the
 * CPU baseline. It is: the GPU chain produces coherent output (readable
 * text) under one consistent numerics regime (paper §3.2/§3.6).
 */

#ifndef __QWEN3_FORWARD_H__
#define __QWEN3_FORWARD_H__

#include <CL/cl.h>
#include <cstddef>
#include <cstdint>
#include <string>

namespace causallm_gpu {

struct Qwen3Config {
  unsigned int hidden_size;       // 2560 for 4B
  unsigned int intermediate_size; // 9728 for 4B
  unsigned int head_dim;          // 128
  unsigned int num_heads_Q;       // 32
  unsigned int num_heads_KV;      // 8 (GQA=4)
  unsigned int num_layers;        // 36
  unsigned int vocab_size;        // 151936
  unsigned int max_seq_len;       // 20480 (from nntr_config)
  float rms_norm_eps;             // 1e-6
  float rope_theta;               // 1e6
};

/// Per-layer weight handles (SVM cl_mem). All allocated once at init,
/// freed in the destructor. Layout: see the .cpp comments at the load
/// site for the exact tensor save order.
struct LayerWeights {
  cl_mem attention_norm_gamma = nullptr; // [hidden] fp32
  cl_mem wq_packed = nullptr;            // QINT4 [hidden, hQ*d]
  cl_mem wq_scales = nullptr;            // fp16 per-channel
  cl_mem wk_packed = nullptr;            // QINT4 [hidden, hKV*d]
  cl_mem wk_scales = nullptr;
  cl_mem wv_packed = nullptr;            // QINT4 [hidden, hKV*d]
  cl_mem wv_scales = nullptr;
  cl_mem wo_packed = nullptr;            // QINT4 [hQ*d, hidden]
  cl_mem wo_scales = nullptr;
  cl_mem ffn_norm_gamma = nullptr;       // [hidden] fp32
  cl_mem wgate_packed = nullptr;         // QINT4 [hidden, intermediate]
  cl_mem wgate_scales = nullptr;
  cl_mem wup_packed = nullptr;           // QINT4 [hidden, intermediate]
  cl_mem wup_scales = nullptr;
  cl_mem wdown_packed = nullptr;         // QINT4 [intermediate, hidden]
  cl_mem wdown_scales = nullptr;
};

class Qwen3Forward {
public:
  Qwen3Forward();
  ~Qwen3Forward();

  Qwen3Forward(const Qwen3Forward &) = delete;
  Qwen3Forward &operator=(const Qwen3Forward &) = delete;

  /// Initialize OpenCL context (via nntrainer's ClContext) + mmap weight
  /// file + read header. Does NOT load weights to GPU yet.
  /// Returns true on success.
  bool init(const Qwen3Config &cfg, const std::string &weight_path);

  /// Smoke test: allocate one SVM cl_mem of `bytes` size, write a
  /// known pattern, read it back, verify, free. Returns true if all
  /// CL calls succeed and the data round-trips correctly.
  bool svm_smoke_test(size_t bytes);

  /// Print the first `n` bytes of the mmap'd weight file as hex.
  void dump_weight_header(size_t n);

  /// Load layer 0's attention_norm gamma (fp32, [hidden_size]) into an
  /// SVM cl_mem. Computes the offset manually: embedding tensor is
  /// first in the weight file (Q6_K, [vocab, hidden] = vocab*hidden/256
  /// blocks * 210 bytes/block), then layer 0 attention_norm gamma. The
  /// allocated SVM pointer is owned by the class and freed in the
  /// destructor.
  bool load_layer0_attention_norm_to_svm();

  /// Dispatch rmsnorm.cl on a known input pattern using the previously
  /// loaded attention_norm gamma. Prints a summary of input + output
  /// (first/last few values) and a quick sanity check on the rms norm
  /// math. Returns true if the kernel ran and produced finite output.
  bool run_rmsnorm_layer0();

  const Qwen3Config &config() const { return cfg_; }
  size_t weight_file_size() const { return weight_bytes_; }

private:
  /// Byte size of the embedding tensor on disk (Q6_K).
  size_t embed_table_bytes() const;

  Qwen3Config cfg_{};
  std::string weight_path_;
  const uint8_t *weight_mmap_ = nullptr; // mmap'd, read-only
  size_t weight_bytes_ = 0;
  int weight_fd_ = -1;

  cl_context cl_ctx_ = nullptr;        // borrowed from ClContext
  cl_command_queue cl_q_ = nullptr;    // borrowed from ClContext
  cl_device_id cl_dev_ = nullptr;      // borrowed

  // Layer 0 attention_norm gamma (fp32, [hidden_size]) in SVM.
  void *layer0_attn_norm_gamma_svm_ = nullptr;
};

} // namespace causallm_gpu

#endif // __QWEN3_FORWARD_H__
