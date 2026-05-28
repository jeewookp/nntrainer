// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Debadri Samaddar <s.debadri@samsung.com>
 *
 * @file	blas_kernel_interface.h
 * @date	5 June 2024
 * @brief	Interface for blas OpenCL kernels
 * @see		https://github.com/nntrainer/nntrainer
 * @author	Debadri Samaddar <s.debadri@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#ifndef __BLAS_KERNEL_INTERFACE_H__
#define __BLAS_KERNEL_INTERFACE_H__

#include <string>
#include <tensor.h>

namespace nntrainer {

/**
 * @brief Process data and dimensions for OpenCL dot operation
 * @param[in] input Tensor
 * @param[in] m Tensor
 * @param[in] RunLayerContext reference
 * @param[in] trans bool
 * @param[in] trans_m bool
 */
Tensor dotCl(Tensor const &input, Tensor const &m, bool trans = false,
             bool trans_m = false);

/**
 * @brief Process data and dimensions for OpenCL dot operation
 * @param[in] input Tensor
 * @param[in] m Tensor
 * @param[in] result Tensor
 * @param[in] RunLayerContext reference
 * @param[in] trans bool
 * @param[in] trans_m bool
 */
void dotCl(Tensor const &input, Tensor const &m, Tensor &result,
           bool trans = false, bool trans_m = false);

/**
 * @brief Process data and dimensions for OpenCL dot operation
 * @param[in] input Tensor
 * @param[in] m Tensor
 * @param[in] result Tensor
 * @param[in] RunLayerContext reference
 * @param[in] trans bool
 * @param[in] trans_m bool
 */
void dotBatchedCl(Tensor const &input, Tensor const &m, Tensor &result,
                  bool trans = false, bool trans_m = false);

/**
 * @brief Multiply value element by element immediately
 * @param[in] input Tensor
 * @param[in] value multiplier
 * @param[in] RunLayerContext reference
 */
void multiplyCl(Tensor &input, float const &value);

/**
 * @brief Process data and dimensions for add operation
 * @param[in] result Tensor
 * @param[in] input Tensor
 */
void add_i_cl(Tensor &result, Tensor const &input);

/**
 * @brief Process data and dimensions for transpose operation
 * @param[in] direction string
 * @param[in] input Tensor
 * @param[in] result Tensor
 */
void transposeCl(const std::string &direction, Tensor const &in,
                 Tensor &result);

/**
 * @brief Copy data from one tensor to another
 *
 * @param input Tensor
 * @param result Tensor
 */
void copyCl(const Tensor &input, Tensor &result);

/**
 * @brief nrm2 computation : Euclidean norm
 * @param input Tensor
 * @return Euclidean norm
 * @note This function is used to compute the Euclidean norm of a vector.
 */
float nrm2Cl(const Tensor &input);

/**
 * @brief Absolute sum computation
 *
 * @param input Tensor
 * @return float absolute sum of the elements
 */
float asumCl(const Tensor &input);

/**
 * @brief Absolute max computation
 *
 * @param input Tensor
 * @return int index of the maximum absolute value
 * @note Not necessarily the first if there are multiple maximums.
 */
int amaxCl(const Tensor &input);

/**
 * @brief Absolute min computation
 *
 * @param input Tensor
 * @return int index of the minimum absolute value
 * @note Not necessarily the first if there are multiple minimums.
 */
int aminCl(const Tensor &input);

/**
 * @brief v8c GPU path entry point — paper 8/4/4: int8 activation × channel-wise
 *        QINT4 weight, 87% of Adreno 830 dp4a peak (4499 GFLOP/s validated).
 *        Env-gated via NNTR_FC_INT8_GPU=1. Caller falls back to dotCl on false.
 * @param[in] input fp32 or fp16 activation tensor [M, K]
 * @param[in] weight Int4QTensor (channel-wise QINT4, osv32) [K, N]
 * @param[out] output fp32 or fp16 tensor [M, N] (preallocated)
 * @return true if v8c path executed; false if not applicable
 *         (env disabled, weight not QINT4, shape misaligned).
 */
bool dotCl_v8c(const Tensor &input, const Tensor &weight, Tensor &output);

/**
 * @brief Fused Q + K + V projection + RoPE + layout transform (paper §3.6 #1).
 *
 * Replaces three separate dotCl_v8c dispatches (Q, K, V FCs) plus the CPU
 * RoPE pass in MHACoreLayer with one OpenCL kernel. Reference paper:
 * "We crafted a custom kernel to combine rotary embedding with the layout
 *  transformations of query (Q), key (K), and value (V) projections,
 *  transforming the query projection from (B,1,S,hq·dh) to
 *  (B·hkv, S·hq/hkv, dh)." -- arXiv:2505.00232 §3.6
 *
 * Step 2a (this commit): skeleton — kernel body is a stub (zero-fills
 *   outputs); validates that the dispatch + env gate + build wiring is
 *   sound. Returns false unless NNTR_FUSED_QKV_GPU=1.
 *
 * Step 2b (next): replace stub with shared activation quant + 3-pass int4
 *   GEMM + per-Q/K RoPE + writeback in the OHWI-ready layout.
 *
 * @param[in]  input  activation `[B, 1, S, hidden]` FP16 (FP32 → reject in 2a)
 * @param[in]  wq     Q weight, QINT4 channel-wise, [hidden, hq*dh]
 * @param[in]  wk     K weight, QINT4 channel-wise, [hidden, hkv*dh]
 * @param[in]  wv     V weight, QINT4 channel-wise, [hidden, hkv*dh]
 * @param[in]  cos_table cos LUT `[max_pos, dh]` FP16
 * @param[in]  sin_table sin LUT `[max_pos, dh]` FP16
 * @param[in]  from_pos  RoPE position offset (cache index for the first
 *                       token of this dispatch)
 * @param[in]  hq, hkv, dh head geometry
 * @param[out] q_out  `[B, S, hq*dh]` FP16, RoPE applied
 * @param[out] k_out  `[B, S, hkv*dh]` FP16, RoPE applied
 * @param[out] v_out  `[B, S, hkv*dh]` FP16, no RoPE (paper convention)
 * @return true if the fused path executed; false if env not set, shapes
 *         unsupported, or any binding precondition failed. Caller MUST fall
 *         back to the existing 3-FC + CPU RoPE path on false.
 */
bool fused_qkv_rope_layout_gpu(
  const Tensor &input, const Tensor &wq, const Tensor &wk, const Tensor &wv,
  const Tensor &cos_table, const Tensor &sin_table,
  unsigned int from_pos, unsigned int hq, unsigned int hkv, unsigned int dh,
  Tensor &q_out, Tensor &k_out, Tensor &v_out);

/**
 * @brief Segment A: GPU RMSNorm with TensorBacking output residency.
 *
 *        Paper §3.2 cross-layer residency: the output cl_mem is owned by
 *        the process-global TensorBackingPool keyed by `output_name` and
 *        also assigned to `output.setBacking()`. Host data of `output` is
 *        left untouched — downstream consumers MUST read via
 *        `getBacking()` (or pool lookup by name).
 *
 *        If `input.getBacking()` exists with FP16 encoding, the backing
 *        cl_mem is used directly (zero host transfer). Otherwise the
 *        input is uploaded from host (one transfer, same as today).
 *        Gamma is uploaded once per (gamma name) and cached.
 *
 *        Env-gated via NNTR_RESIDENT_RMSNORM=1. Returns false if env not
 *        set or any precondition fails; caller falls back to CPU path.
 *
 * @param[in]  input  FP16 activation [B, C, H, W]
 * @param[in]  gamma  FP16 per-channel scale [W]
 * @param[in]  epsilon  RMS epsilon
 * @param[in]  B, C, H, W  shape constants matching `input`
 * @param[in]  output_name  stable Tensor name (used as pool key)
 * @param[out] output Tensor; setBacking() is called on success
 * @return true if the GPU path ran; false otherwise
 */
bool rmsnorm_resident_fp16(const Tensor &input, const Tensor &gamma,
                           float epsilon, unsigned int B, unsigned int C,
                           unsigned int H, unsigned int W,
                           const std::string &output_name, Tensor &output);

/**
 * @brief FP32 variant of rmsnorm_resident. Same contract as the FP16
 *        version but uses rmsnorm_cl (subgroup-reduced kernel) for the
 *        FP32 residual stream Qwen3 currently uses. Encoding of the
 *        resulting TensorBacking is Encoding::FP32.
 */
bool rmsnorm_resident_fp32(const Tensor &input, const Tensor &gamma,
                           float epsilon, unsigned int H, unsigned int W,
                           const std::string &output_name, Tensor &output);

} // namespace nntrainer
#endif /* __BLAS_KERNEL_INTERFACE_H__ */
