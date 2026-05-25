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

} // namespace nntrainer
#endif /* __ATTENTION_KERNELS_H__ */
