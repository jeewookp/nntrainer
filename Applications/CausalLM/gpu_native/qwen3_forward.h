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
#include <vector>

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

  /// Load layer 0's QKV projection weights (wq, wk, wv) — all QINT4 in
  /// KAI Section A format, sharing a common K=hidden input dim — from
  /// the mmap'd weight file. Each weight gets a v8c-ready backing
  /// (cl_mem buffer + image2d view + per-channel fp32 scales + per-
  /// channel int32 weight row-sums) via
  /// make_v8c_weight_backing_from_kai_section_a. Skips over the
  /// q_norm and k_norm gamma slabs interleaved between wq/wk/wv per
  /// the Qwen3 layer save order (load helpers for those come in the
  /// next commit).
  bool load_layer0_qkv_weights();

  /// Load layer 0's attention-output projection wo (QINT4, K=hQ*d,
  /// N=hidden). Sits right after wv in the save order. The wv slab
  /// has the same per-tensor format as wq/wk so the offset
  /// calculation just chains off load_layer0_qkv_weights's layout.
  bool load_layer0_wo();

  /// Load layer 0's FFN block weights: ffn_norm gamma (fp32 [hidden]
  /// pushed to SVM), ffn_up + ffn_gate (QINT4 K=hidden, N=intermediate),
  /// ffn_down (QINT4 K=intermediate, N=hidden). Save order (per
  /// transformer.cpp::createMlp): ffn_up, ffn_gate, ffn_down — and
  /// they sit right after the ffn_norm gamma which itself sits right
  /// after wo + decoder_add (no weights).
  bool load_layer0_ffn_weights();

  /// Run the FFN block on the post-attention residual:
  ///   ffn_norm.cl -> quantize_act -> v8c(ffn_up), v8c(ffn_gate)
  ///   -> swiglu -> quantize_act -> v8c(ffn_down) -> add residual_1
  ///   -> layer0_output.
  /// Requires layer0_residual1_fp32_ to be populated (run after
  /// run_layer0_qkv_projection in step 6a). Prints summary stats.
  bool run_layer0_ffn();

  /// Load weights for one decoder layer at the given file offset.
  /// Advances offset_inout past the layer's bytes. Populates
  /// layers_[layer_id]. Also allocates per-layer KV cache (SVM,
  /// sized for max_seq_len_used positions).
  bool load_layer(unsigned int layer_id, size_t *offset_inout,
                  unsigned int max_seq_len_used);

  /// Generic per-layer forward used for chaining. Takes a cl_mem
  /// fp32 [hidden] input buffer, returns a newly-allocated cl_mem
  /// fp32 [hidden] output buffer (the post-layer residual stream).
  /// Caller owns the output. Position is the RoPE / KV cache
  /// position for the single token being processed (single-token
  /// forward only for now — prefill comes later).
  cl_mem forward_one_layer(unsigned int layer_id, cl_mem in_fp32,
                           unsigned int position);

  /// Load the model's final output_norm gamma (fp32 [hidden]) from
  /// the tail of the weight file. Caller passes the file offset
  /// right after the last layer's bytes; this method reads the gamma
  /// and pushes it to SVM. Returns true on success.
  bool load_output_norm(size_t file_offset);

  /// Apply the output_norm (rmsnorm.cl) to a cl_mem fp32 [hidden]
  /// input, in place. Caller's buffer is modified.
  bool run_output_norm(cl_mem inout_fp32);

  /// Load layer 0's q_norm and k_norm gammas (fp32, [head_dim]) from
  /// the mmap'd weights. They sit between wq/wk and wk/wv in the
  /// save order. Pushed into SVM as fp16 (the rmsnorm_cl_fp16 kernel
  /// expects `half alpha`) — converted on the host side at load time.
  bool load_layer0_qk_norm_gammas();

  /// Precompute the RoPE cos/sin tables for one position into SVM
  /// (fp16, [head_dim] each) using the cfg's rope_theta. Position 0
  /// gives identity rotation (cos=1, sin=0); pass a non-zero position
  /// to actually exercise the rotation math. Reuses existing SVM
  /// allocations if called multiple times (always rewrites the table).
  bool precompute_rope_for_position(unsigned int position);

  /// Dispatch the inline `rope_fp16` kernel on the layer-0 Q and K
  /// fp16 cl_mem buffers in place. Must be called after RoPE freqs
  /// are precomputed for the same position. Hooked into the QKV
  /// pipeline by run_layer0_qkv_projection when rope position is
  /// preconfigured; otherwise skipped.
  bool run_layer0_rope_on_qk(cl_mem y_q, cl_mem y_k);

  /// Allocate layer 0's K and V caches as SVM cl_mem buffers, sized
  /// for the full max_seq_len. Both zeroed at allocation so any
  /// position we haven't written reads as 0 (causal-mask-friendly).
  /// Cache layout (concat / row-major): [max_seq_len, hKV * head_dim]
  /// fp16 — same as the existing CausalLM KVCacheManager for the
  /// non-OHWI path. SVM-allocated so the attention kernel can read
  /// them with svm_inputs=true (no upload).
  bool allocate_layer0_kv_cache_svm();

  /// Run the full QKV projection pipeline for layer 0 on the same
  /// deterministic input pattern as step 2/3: rmsnorm.cl ONCE against
  /// the SVM attention_norm gamma, then quantize_act_v8c_fp32_cl ONCE
  /// on the rmsnorm output (shared activation quant — paper §3.6
  /// insight, same pattern as dotCl_v8c's shared-quant cache hit on
  /// wq/wk/wv), then three gemm_int8_v8c_cl dispatches against the
  /// loaded Q/K/V weights. Then rmsnorm_cl_fp16 over each head row of
  /// Q (H=hQ, W=d) and K (H=hKV, W=d) using the SVM q_norm/k_norm
  /// gammas — paper §3.3 Qwen3-specific per-head normalization.
  /// Outputs three fp16 cl_mem buffers (Q[hQ*d], K[hKV*d], V[hKV*d])
  /// — Q/K are post-norm and ready for RoPE in the next commit;
  /// V is unchanged (Qwen3 has no v_norm).
  bool run_layer0_qkv_projection();

  const Qwen3Config &config() const { return cfg_; }
  size_t weight_file_size() const { return weight_bytes_; }

  /// Byte offset in the weight file where decoder layer 0 starts —
  /// i.e. right after the embedding (Q6_K) blob. Useful for chaining
  /// load_layer(0..L-1, &offset, ...).
  size_t layers_start_offset() const;

  /// Accessor for the layer-0 output cl_mem produced by the old path
  /// (run_layer0_ffn). Available so step-7a's new chain (which uses
  /// the generic load_layer + forward_one_layer path) can take layer
  /// 0's output as its starting input.
  cl_mem layer0_output_fp32() const { return layer0_output_fp32_; }

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
  // Layer 0 q_norm / k_norm gammas (fp16, [head_dim]) in SVM. Converted
  // from the fp32 on-disk gammas at load time so they can be passed
  // directly to rmsnorm_cl_fp16 (which takes `half alpha`).
  void *layer0_q_norm_gamma_svm_fp16_ = nullptr;
  void *layer0_k_norm_gamma_svm_fp16_ = nullptr;
  // RoPE cos/sin tables for the currently-configured position
  // (fp16, [head_dim]) in SVM. Repopulated by
  // precompute_rope_for_position(). The doubled-half layout
  // matches the CPU mha_core path: cos[k+half] = cos[k], sin[k+half] =
  // sin[k] so the kernel can index head_dim values directly.
  void *layer0_rope_cos_svm_fp16_ = nullptr;
  void *layer0_rope_sin_svm_fp16_ = nullptr;
  int   layer0_rope_position_ = -1; // -1 = no position configured

  // Layer 0 K and V caches (SVM cl_mem), [max_seq_len * hKV * d] fp16
  // each. Concat row-major layout: cache[pos * hKV*d + hkv*d + k]. Both
  // zero-initialized at allocation so unwritten positions read as 0
  // (clean under the causal mask).
  void *layer0_cache_k_svm_ = nullptr;
  void *layer0_cache_v_svm_ = nullptr;

  /// Bundled v8c-ready state for one int4 GEMM weight. Built once by
  /// load_qint4_weight_at() from a mmap'd KAI Section A payload via
  /// make_v8c_weight_backing_from_kai_section_a. The TensorBacking is
  /// held opaquely so this header doesn't depend on cl_tensor_view.h;
  /// the dispatch site casts back to access the cl_mem image view.
  struct V8cFcWeight {
    void *backing = nullptr;       // tv::TensorBacking* (owning)
    cl_mem weight_image = nullptr; // image2d view (owned by backing)
    cl_mem scale_buf = nullptr;    // fp32 [N]
    cl_mem row_sum_w_int4 = nullptr; // int32 [N]
    unsigned int K = 0, N = 0;
  };

  // Layer 0 Q/K/V projection weights + attention output projection wo.
  V8cFcWeight layer0_wq_{};
  V8cFcWeight layer0_wk_{};
  V8cFcWeight layer0_wv_{};
  V8cFcWeight layer0_wo_{};
  // Layer 0 FFN block weights.
  V8cFcWeight layer0_ffn_up_{};
  V8cFcWeight layer0_ffn_gate_{};
  V8cFcWeight layer0_ffn_down_{};
  // ffn_norm gamma (fp32, [hidden]) in SVM.
  void *layer0_ffn_norm_gamma_svm_ = nullptr;

  // Persistent buffers for the layer-0 residual stream. Allocated lazily
  // by the forward step that needs them; lives for the layer-0 forward
  // pass (subsequent steps in this layer read them; freed in destructor).
  cl_mem layer0_residual1_fp32_ = nullptr; // [hidden] = x + wo(O)
  cl_mem layer0_output_fp32_    = nullptr; // [hidden] = residual_1 + ffn(.)

  /// Bundled per-layer weight state for the generic forward path.
  /// Populated by load_layer(); consumed by forward_one_layer().
  struct LayerWeights {
    void *attn_norm_gamma_svm = nullptr;     // fp32 [hidden]
    void *q_norm_gamma_svm_fp16 = nullptr;   // fp16 [head_dim]
    void *k_norm_gamma_svm_fp16 = nullptr;   // fp16 [head_dim]
    void *ffn_norm_gamma_svm = nullptr;      // fp32 [hidden]
    V8cFcWeight wq, wk, wv, wo;
    V8cFcWeight ffn_up, ffn_gate, ffn_down;
    void *cache_k_svm = nullptr; // fp16 [max_seq_len_used * hKV * d]
    void *cache_v_svm = nullptr; // fp16 [max_seq_len_used * hKV * d]
  };
  std::vector<LayerWeights> layers_;

  // Final output_norm gamma (fp32, [hidden]) in SVM.
  void *output_norm_gamma_svm_ = nullptr;

  /// Generic loader: parses one Int4QTensor blob at the given file
  /// offset ([qscheme u16][packed K*N/2][scales N*u16]) and populates
  /// `out`. Returns false on shape / qscheme validation failure.
  bool load_qint4_weight_at(size_t file_offset, unsigned int K,
                            unsigned int N, V8cFcWeight *out,
                            const char *tag);

  /// Release a V8cFcWeight (used by destructor).
  void release_v8c_weight(V8cFcWeight *w);
};

} // namespace causallm_gpu

#endif // __QWEN3_FORWARD_H__
