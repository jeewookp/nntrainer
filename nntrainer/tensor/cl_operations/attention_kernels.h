// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Yash Singh <yash.singh@samsung.com>
 *
 * @file	attention_kernels.h
 * @date	28 August 2024
 * @brief	Common attention OpenCL kernels
 * @see		https://github.com/nntrainer/nntrainer
 * @author	Yash Singh <yash.singh@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#ifndef __ATTENTION_KERNELS_H__
#define __ATTENTION_KERNELS_H__

#include <cl_context.h>
#include <engine.h>
#include <opencl_buffer.h>
#include <opencl_kernel.h>

#include <string>

namespace nntrainer {

/**
 * @brief     Rotary Embedding process
 * @param[in] in _FP16 * input
 * @param[in] out _FP16 * output
 * @param[out] freqs_cos cosine of the frequencies
 * @param[out] freqs_sin sine of the frequencies
 * @param[in] cos_ vector of cos values
 * @param[in] sin_ vector of sin values
 * @param[in] batch size of batch
 * @param[in] channel channel of input
 * @param[in] height height of input
 * @param[in] width width of input
 * @param[in] dim hidden dim size
 * @param[in] from sequence order
 * @param[in] max_timestep max timestep
 * @param[in] in_size size of input
 * @param[in] out_size size of output
 */
void rotary_emb_cl(float *in, float *out,
                   const std::vector<std::vector<float>> &freqs_cos,
                   const std::vector<std::vector<float>> &freqs_sin,
                   const std::vector<float> &cos_,
                   const std::vector<float> &sin_, unsigned int batch,
                   unsigned int channel, unsigned int height,
                   unsigned int width, unsigned int dim, unsigned int from,
                   unsigned int max_timestamp, unsigned int in_size,
                   unsigned int out_size);

#ifdef ENABLE_FP16

/**
 * @brief     Rotary Embedding process
 * @param[in] in _FP16 * input
 * @param[in] out _FP16 * output
 * @param[out] freqs_cos cosine of the frequencies
 * @param[out] freqs_sin sine of the frequencies
 * @param[in] cos_ vector of cos values
 * @param[in] sin_ vector of sin values
 * @param[in] batch size of batch
 * @param[in] channel channel of input
 * @param[in] height height of input
 * @param[in] width width of input
 * @param[in] dim hidden dim size
 * @param[in] from sequence order
 * @param[in] max_timestep max timestep
 * @param[in] in_size size of input
 * @param[in] out_size size of output
 */
void rotary_emb_cl(_FP16 *in, _FP16 *out,
                   const std::vector<std::vector<float>> &freqs_cos,
                   const std::vector<std::vector<float>> &freqs_sin,
                   const std::vector<float> &cos_,
                   const std::vector<float> &sin_, unsigned int batch,
                   unsigned int channel, unsigned int height,
                   unsigned int width, unsigned int dim, unsigned int from,
                   unsigned int max_timestamp, unsigned int in_size,
                   unsigned int out_size);

#endif

/// GPU two-1x1-conv attention for prefill (ML Drift section 3.7
/// algorithm). Three-kernel pipeline:
///   K1: qk_matmul_f16    Q @ K^T  -> scores  (per head, full d reduce)
///   K2: softmax_row_f16  in-place row softmax over the N_kv axis
///   K3: sv_matmul_f16    scores @ V  -> O    (per head)
/// All three operate on FP16 storage (uint16 bits) with FP32 register
/// accumulators. GQA is handled inside the kernels via head_q / gqa.
/// `svm_inputs == true` passes Q/K/V/O directly via SVM pointers; the
/// `scores` buffer is always a grow-only cl_mem scratch.
/// Returns false if shape unsupported (caller falls back to CPU).
bool two_conv_attention_prefill_f16_cl(const uint16_t *Q_host,
                                       const uint16_t *K_host,
                                       const uint16_t *V_host,
                                       uint16_t *O_host, unsigned int M,
                                       unsigned int N_kv,
                                       unsigned int num_heads_Q,
                                       unsigned int num_heads_KV,
                                       unsigned int head_dim, bool causal,
                                       bool svm_inputs = false);

/// int8-KV variant of two_conv_attention_prefill_f16_cl. Same shapes,
/// but K/V are stored as signed int8 bytes with a per-(token, head)
/// FP16 amax scale (paper §3.7). The kernel dequantizes inline by
/// folding the scale into the QK^T and SV reductions.
///   K_i8_host, V_i8_host:  [N_kv, num_heads_KV * head_dim] int8
///   K_scale_host, V_scale_host: [N_kv, num_heads_KV] fp16-bit
bool two_conv_attention_prefill_f16_kvi8_cl(
  const uint16_t *Q_host, const int8_t *K_i8_host, const int8_t *V_i8_host,
  const uint16_t *K_scale_host, const uint16_t *V_scale_host,
  uint16_t *O_host, unsigned int M, unsigned int N_kv, unsigned int num_heads_Q,
  unsigned int num_heads_KV, unsigned int head_dim, bool causal,
  bool svm_inputs = false);

/// image2d_from_buffer variant: Q/K/V viewed as RGBA UINT32 image2d
/// (16 bytes = 8 halves per texel), 8x fewer memory transactions vs
/// scalar half loads. Requires head_dim, HD_Q, HD_KV all multiples of 8.
/// SVM inputs not supported in this variant (image2d_from_buffer needs
/// cl_mem; the wrapper copies host-host to scratch cl_mem first).
bool two_conv_attention_prefill_f16_img_cl(
  const uint16_t *Q_host, const uint16_t *K_host, const uint16_t *V_host,
  uint16_t *O_host, unsigned int M, unsigned int N_kv, unsigned int num_heads_Q,
  unsigned int num_heads_KV, unsigned int head_dim, bool causal);

/// §3.8 OHWI K-cache variant of two_conv_attention_prefill_f16_cl.
/// Same three-kernel pipeline but K is laid out as [H_kv, S_max, d]
/// (per-head contiguous, paper's "convolution weight" form) rather
/// than the default row-major [N_kv, H_kv * d]. V is still in concat
/// layout — only K1 (qk_matmul_f16_ohwi) is replaced; K2 (softmax)
/// and K3 (sv_matmul_f16) are reused unchanged.
///
///   K_host: per-batch base of the OHWI cache, i.e.
///           &cache_key[batch][0][0][0]; total size H_kv*S_max*d halves.
///   max_seq_len: the allocated S_max (head stride in the OHWI layout).
///                The kernel only reads N_kv rows.
///
/// Opt-in by setting BOTH NNTR_KV_OHWI=1 AND NNTR_MHA_GPU=1. Unlike
/// the concat _f16_cl wrapper, this path has no "force-broken" gate:
/// turning both env vars on is the explicit opt-in.
bool two_conv_attention_prefill_f16_ohwi_cl(
  const uint16_t *Q_host, const uint16_t *K_host, const uint16_t *V_host,
  uint16_t *O_host, unsigned int M, unsigned int N_kv, unsigned int num_heads_Q,
  unsigned int num_heads_KV, unsigned int head_dim, unsigned int max_seq_len,
  bool causal, bool svm_inputs = false);

/// §3.8 FULL-OHWI variant: K is OHWI [H_kv, S_max, d] AND V is OHWI-
/// reversed [H_kv, d, S_max]. Same three-kernel pipeline as the
/// half-OHWI variant; only K3 (sv_matmul) changes — uses
/// sv_matmul_f16_ohwi which reads V at stride-1 per n (cache-
/// friendly across the reduction). Caller must scatter both K and V
/// caches into their respective OHWI layouts. Same SVM input
/// semantics as _ohwi_cl.
bool two_conv_attention_prefill_f16_ohwi_full_cl(
  const uint16_t *Q_host, const uint16_t *K_host, const uint16_t *V_host,
  uint16_t *O_host, unsigned int M, unsigned int N_kv, unsigned int num_heads_Q,
  unsigned int num_heads_KV, unsigned int head_dim, unsigned int max_seq_len,
  bool causal, bool svm_inputs = false);

/// §3.8 + paper-style image2d_from_buffer V variant. Same pipeline as
/// _ohwi_full_cl but the SV kernel reads V via image2d_from_buffer
/// (sv_matmul_f16_ohwi_img), exploiting the same texture-cache pattern
/// that gives v8c FC 87% of Adreno 830 peak. V must already be in
/// OHWI-reversed [H_kv, d, S_max] layout in a regular cl_mem (NOT SVM
/// — image2d_from_buffer requires a cl_mem-backed buffer). Q and K
/// are still SVM (qk_matmul_f16_ohwi unchanged); only V uses image2d.
///
///   V_buf_ohwi: cl_mem holding the per-batch OHWI-reversed V slab,
///               H_kv * d * max_seq_len halves, row_pitch = max_seq_len.
bool two_conv_attention_prefill_f16_ohwi_img_cl(
  const uint16_t *Q_svm, const uint16_t *K_svm, cl_mem V_buf_ohwi,
  uint16_t *O_svm, unsigned int M, unsigned int N_kv, unsigned int num_heads_Q,
  unsigned int num_heads_KV, unsigned int head_dim, unsigned int max_seq_len,
  bool causal);

/// Variant of _ohwi_img_cl that takes a pre-built cl_mem image2d view
/// over the OHWI-reversed V buffer. Saves one clCreateImage per call
/// when the caller can cache the view (e.g. once per layer at load).
bool two_conv_attention_prefill_f16_ohwi_img_view_cl(
  const uint16_t *Q_svm, const uint16_t *K_svm, cl_mem V_image_ohwi,
  uint16_t *O_svm, unsigned int M, unsigned int N_kv, unsigned int num_heads_Q,
  unsigned int num_heads_KV, unsigned int head_dim, unsigned int max_seq_len,
  bool causal);

/// Single-kernel flash-attention prefill (paper §3.6 fusion +
/// Dao et al. 2022 online softmax). Replaces the three-kernel
/// two_conv_attention pipeline with one kernel that does QK · softmax
/// · V·S inline using local-memory K/V tiles + per-row online softmax
/// accumulators. Eliminates the global scores[H, M, N_kv] tensor
/// (~6.7 MB bandwidth per prefill on Qwen3-0.6B at S=282) and avoids
/// the VGPR spill that makes the 3-kernel path slower than CPU.
///
///   Q: [M, num_heads_Q * head_dim] fp16 row-major
///   K: [N_kv, num_heads_KV * head_dim] fp16
///   V: [N_kv, num_heads_KV * head_dim] fp16
///   O: [M, num_heads_Q * head_dim] fp16
/// GQA is resolved inside the kernel: head_kv = head_q / (num_heads_Q
/// / num_heads_KV).
///
/// Env-gated by NNTR_GPU_MHA=1. Returns false on shape mismatch or
/// when the env is unset; caller falls back to CPU mha. Step #1 of
/// the everything-on-GPU plan (see TENSOR_VIRTUALIZATION_PLAN.md).
///
/// Step status: SKELETON. Body is a zero-fill stub; the host wrapper
/// always returns false so the caller's existing CPU path runs. The
/// real flash-attention body lands once the dispatch path is
/// proven and the synthetic-input validation harness is in place.
bool flash_attention_prefill_f16_cl(const uint16_t *Q_host,
                                    const uint16_t *K_host,
                                    const uint16_t *V_host,
                                    uint16_t *O_host, unsigned int M,
                                    unsigned int N_kv,
                                    unsigned int num_heads_Q,
                                    unsigned int num_heads_KV,
                                    unsigned int head_dim, bool causal,
                                    bool svm_inputs = false);

} // namespace nntrainer
#endif /* __ATTENTION_KERNELS_H__ */
