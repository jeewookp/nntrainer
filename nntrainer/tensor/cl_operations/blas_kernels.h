// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Debadri Samaddar <s.debadri@samsung.com>
 *
 * @file	blas_kernels.h
 * @date	14 May 2024
 * @brief	Common blas OpenCL kernels
 * @see		https://github.com/nntrainer/nntrainer
 * @author	Debadri Samaddar <s.debadri@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#ifndef __BLAS_KERNELS_H__
#define __BLAS_KERNELS_H__

#include <cl_context.h>
#include <engine.h>
#include <opencl_buffer.h>
#include <opencl_buffer_manager.h>
#include <opencl_kernel.h>

#include <string>
#include <vector>

namespace nntrainer {

/**
 * @brief     signed 4-bit integer gemv async computation : C = A*B
 * @param[in] weight std::vector<void *> for int4 quantized weight
 * @param[in] scale std::vector<uint16_t *> for scales
 * @param[in] input uint16_t * for input
 * @param[in] output std::vector<uint16_t *> for output
 * @param[in] K hidden dimension
 * @param[in] Ns output dimensions
 */
void gemv_int4_async_cl(std::vector<void *> weights,
                        std::vector<uint16_t *> scales, uint16_t *input,
                        std::vector<uint16_t *> outputs, unsigned int K,
                        std::vector<unsigned int> Ns,
                        unsigned int quantization_group_size);

/**
 * @brief     signed 4-bit integer gemv async computation : C = A*B
 * @param[in] weight std::vector<void *> for int4 quantized weight
 * @param[in] scale std::vector<uint16_t *> for scales
 * @param[in] input float * for input
 * @param[in] output std::vector<float *> for output
 * @param[in] K hidden dimension
 * @param[in] Ns output dimensions
 */
void gemv_int4_async_cl(std::vector<void *> weights,
                        std::vector<uint16_t *> scales, float *input,
                        std::vector<float *> outputs, unsigned int K,
                        std::vector<unsigned int> Ns,
                        unsigned int quantization_group_size);

/**
 * @brief     signed 4-bit integer gemv computation : C = A*B
 * @param[in] weight char * for int4 quantized weight
 * @param[in] scale uint16_t * for scales
 * @param[in] input uint16_t * for input
 * @param[in] output uint16_t * for output
 * @param[in] K hidden dimension
 * @param[in] N output dimension
 */
void gemv_int4_cl(char *weight, uint16_t *scale, uint16_t *input,
                  uint16_t *output, unsigned int K, unsigned int N,
                  unsigned int quantization_group_size);

/**
 * @brief     signed 4-bit integer gemv computation : C = A*B
 * @param[in] weight char * for int4 quantized weight
 * @param[in] scale uint16_t * for scales
 * @param[in] input float * for input
 * @param[in] output float * for output
 * @param[in] K hidden dimension
 * @param[in] N output dimension
 */
void gemv_int4_cl(char *weight, uint16_t *scale, float *input, float *output,
                  unsigned int K, unsigned int N,
                  unsigned int quantization_group_size);

/**
 * @brief     Q4_0 gemm async computation : C = A*B
 * @param[in] matAdata std::vector<void *> for Matrix A
 * @param[in] matBdata float * for Matrix B
 * @param[in] matCdata std::vector<float *> for Matrix C
 * @param[in] M input dimension
 * @param[in] N output dimensions of As
 * @param[in] K hidden dimension
 */
void gemm_q4_0_async_cl(std::vector<void *> matAdata, float *matBdata,
                        std::vector<float *> matCdata, unsigned int M,
                        std::vector<unsigned int> N, unsigned int K);

/**
 * @brief     Q4_0 gemm computation : C = A*B
 * @param[in] matAdata void * for Matrix A
 * @param[in] matBdata float * for Matrix B
 * @param[in] matCdata float * for Matrix C
 * @param[in] M input dimension
 * @param[in] K hidden dimension
 * @param[in] N output dimension
 */
void gemm_q4_0_cl(void *matAdata, float *matBdata, float *matCdata,
                  unsigned int M, unsigned int N, unsigned int K);

/**
 * @brief INT4 GEMM computation for float input / output
 */
void sgemm_int4_cl(float *input, char *weight, uint16_t *scale, float *output,
                   unsigned int M, unsigned int N, unsigned int K,
                   unsigned int quantization_group_size);
/**
 * @brief INT4 GEMM computation for fp16 input / output
 */
void gemm_int4_cl(void *input, void *weights, void *scales, void *output,
                  unsigned int M, unsigned int N, unsigned int K,
                  unsigned int quantization_group_size);

/**
 * @brief INT4 GEMM async computation
 */
void gemm_int4_async_cl(float *input, std::vector<void *> weights,
                        std::vector<uint16_t *> scales,
                        std::vector<float *> matCdata, unsigned int M,
                        std::vector<unsigned int> Ns, unsigned int K,
                        unsigned int quantization_group_size);

/**
 * @brief INT4 channel-wise GEMM for Adreno GPUs (gpu_int4_gemm_adreno kernel).
 *
 * Two-pass GPU pipeline (adopted from origin/main):
 *   1. input_transpose kernel transposes `input` ([M][alignK] fp16) into
 *      `input_transposed` ([alignK][align(M,4)] fp16) using
 *      image1d_buffer_t reads/writes -- much faster than the previous
 *      CPU transpose loop.
 *   2. gpu_int4_gemm_adreno reads the transposed input as a texture
 *      (image1d_buffer_t) and produces fp16 output of shape [M][N].
 *
 * Layout (matches Int4Utils::convertKaiToChannelwise):
 *   - weights : ushort[(K/4) * N], 4 unsigned bias-8 nibbles per ushort
 *               for one channel at K positions [k, k+1, k+2, k+3]
 *   - scales  : fp16[N], one scale per output channel (per-channel quant)
 *
 * @param[in] input            activation [M][alignK] fp16, host pre-converted
 *                             from fp32. Must be a SVM ptr (the wrapper
 *                             wraps it with CL_MEM_USE_HOST_PTR to expose
 *                             it as a cl_mem image).
 * @param[in,out] input_transposed scratch SVM buffer of size
 *                             align(M,4) * alignK fp16; gets overwritten
 *                             with the GPU-transposed input.
 * @param[in] weights          packed int4 weights (ushort SVM)
 * @param[in] scales           per-channel fp16 scales (fp16 SVM)
 * @param[out] output          output [M][N] fp16 (SVM)
 * @param[in] M batch / token count
 * @param[in] N output dimension
 * @param[in] K input dimension
 *
 * Assumes N % 4 == 0, K % 4 == 0, M >= 1. For Qwen3-4B all FC widths
 * satisfy the additional constraint that N is a multiple of 32 too,
 * which keeps the kernel's output guard happy.
 */
void gemm_int4_adreno_cl(uint16_t *input, uint16_t *input_transposed,
                         uint16_t *weights, uint16_t *scales, uint16_t *output,
                         unsigned int M, unsigned int N, unsigned int K);

/**
 * @brief Delegate-style fp16 GEMM using captured conv_wave_memory kernel.
 *
 * Accepts the same int4 weights + scales as gemm_int4_adreno_cl, but
 * internally dequantizes to fp16 and dispatches the delegate's
 * wave-memory kernel (delegate_conv_wave.cl). Weight conversion is
 * cached per (weights_ptr, N, K) so the dequant+repack cost is paid
 * only on the first call for each weight matrix.
 *
 * Achieves ~2.5 TFLOPS on Adreno 830 (vs ~1.5 for int4_gemm_adreno).
 */
void gemm_delegate_fp16_cl(uint16_t *input, uint16_t *input_transposed,
                           uint16_t *weights, uint16_t *scales,
                           uint16_t *output, unsigned int M, unsigned int N,
                           unsigned int K);

/**
 * @brief Batched delegate GEMM with shared input.
 *
 * Dispatches N separate gemm_delegate calls that all share the same
 * activation `input`. Unlike calling gemm_delegate_fp16_cl N times,
 * this path:
 *   1. Runs the SVM->image2d input reformat ONCE (saved across all
 *      weights), instead of once per weight.
 *   2. Skips the per-call blocking SVMMap; a single SVMMap barrier at
 *      the end of the batch drains the whole pipeline.
 *
 * All output tensors still receive the same fp16 [M][Ni] SVM data they
 * would have gotten from per-weight gemm_delegate_fp16_cl, so this is a
 * drop-in replacement for the loop in HalfTensor::dot(vector,vector).
 *
 * Constraint: every Ns[i] must be a multiple of 32 and K a multiple of
 * 8 (same as gemm_delegate_fp16_cl). Caller is responsible for falling
 * back to the int4 path if the constraint does not hold.
 *
 * @param[in]  input    shared fp16 [M][K] SVM activation
 * @param[in]  weights  per-output packed int4 weight SVM pointers
 * @param[in]  scales   per-output fp16 channel scale SVM pointers
 * @param[out] outputs  per-output fp16 [M][Ns[i]] SVM pointers
 * @param[in]  Ns       per-output N dimensions (width of each weight)
 * @param[in]  M        batch / token count
 * @param[in]  K        hidden dimension (shared across all weights)
 */
void gemm_delegate_fp16_cl_batched(uint16_t *input,
                                   const std::vector<uint16_t *> &weights,
                                   const std::vector<uint16_t *> &scales,
                                   const std::vector<uint16_t *> &outputs,
                                   const std::vector<unsigned int> &Ns,
                                   unsigned int M, unsigned int K);

/**
 * @brief Phase A helper — copy a fp16 SVM [M][K] buffer into an image2d
 * (width=M, height=K/4, CL_RGBA|CL_HALF_FLOAT) and register that image2d
 * in GpuImagePool keyed by `svm_ptr`, so a downstream image2d-aware
 * consumer (e.g. gemm_delegate_fp16_cl) can read it directly and skip
 * its own svm_to_image2d reformat. The image2d allocation is cached
 * per svm_ptr across calls; shape changes trigger a re-alloc.
 *
 * Enqueues one svm_to_image2d kernel dispatch on the global OpenCL
 * command queue; caller does not need to sync.
 */
void svm_to_image2d_publish(void *svm_ptr, unsigned int M, unsigned int K);

/**
 * @brief GPU RMSNorm via rmsnorm_image2d_v2 (cooperative workgroup reduction).
 *
 * Layout: image2d input of width=M, height=K/4, RGBA half. Same for output.
 *
 * @param in_svm   fp16 SVM pointer to (M x K) input.
 * @param out_svm  fp16 SVM pointer to (M x K) output (must be distinct).
 * @param gamma    fp32 per-channel scale of length K. Cached as a cl_mem
 *                 buffer keyed by pointer (built once per layer).
 * @param M        number of rows.
 * @param K        hidden / feature width. Must be a multiple of 4.
 * @param epsilon  variance epsilon.
 *
 * If @p in_svm is already in GpuImagePool the svm_to_image2d reformat
 * is skipped. After the kernel, the output image2d is both read back
 * into @p out_svm and registered in GpuImagePool under @p out_svm so
 * the next pool-aware layer (e.g. gemm_delegate_fp16_cl) can skip
 * its reformat.
 */
void rmsnorm_image2d_cl(void *in_svm, void *out_svm,
                         const float *gamma, unsigned int M, unsigned int K,
                         float epsilon);

/**
 * @brief Phase B helper — fp16 SwiGLU on an SVM in1/in2/out via a GPU
 * kernel dispatch. Enqueues `swiglu_cl_fp16` (a file-scope kernel
 * already bundled in the OpenCL build) with the three tensor SVM
 * pointers bound as kernel args. No host-side work, no per-call
 * SVMMap — the downstream consumer's own fence (e.g. the next gemm
 * on the same in-order queue, or the next CPU layer's entry
 * SVMMap) handles coherence.
 *
 * `total` is the flat element count (batch * channel * height *
 * width) — matches what swiglu_cl_fp16's global_id indexing expects.
 */
void swiglu_fp16_svm_cl(const void *in1, const void *in2, void *out,
                        size_t total);

/**
 * @brief Phase B helper — fp16 residual add `out[i] = a[i] + b[i]` on
 * SVM buffers, dispatched as add2_fp16_svm on the global OpenCL queue.
 * Used for Qwen3-style residual connections to replace NEON CPU add_i
 * so the residual stays on the GPU pipeline, avoids the CPU-layer
 * entry SVMMap drain, and the result can be published into the pool
 * for the next gemm to hit.
 */
void add2_fp16_svm_cl(const void *a, const void *b, void *out, size_t total);

/**
 * @brief INT4 channel-wise GEMM for Adreno GPUs, Phase 3c v2 (weight
 *        read through image texture).
 *
 * Same weight / scale layout, same M>1 prefill contract as
 * `gemm_int4_adreno_cl`. The ONLY functional difference is that the
 * packed int4 weights are bound to the kernel as an `image1d_buffer_t`
 * (CL_RGBA/CL_UNSIGNED_INT16) instead of a `__global const ushort *`,
 * so weight fetches go through the texture cache / fetch unit. This
 * mirrors LiteRT's int8 FC path, which on Adreno 830 reaches ~18.7
 * TFLOPS by keeping activation + weight reads on independent memory
 * pipes.
 *
 * All other args (input SVM, input_transposed scratch, scales, output,
 * M/N/K) behave exactly like `gemm_int4_adreno_cl`.
 *
 * Assumes N % 4 == 0, K % 4 == 0. No additional constraints vs v1.
 */
void gemm_int4_adreno_v2_cl(uint16_t *input, uint16_t *input_transposed,
                            uint16_t *weights, uint16_t *scales,
                            uint16_t *output, unsigned int M, unsigned int N,
                            unsigned int K);

/**
 * @brief INT4 channel-wise GEMM for Adreno GPUs, Phase 3c v3
 *        (weight texture + K-axis WG split + __local reduction).
 *
 * Builds on v2's weight texture binding but adds LiteRT's main
 * throughput lever: each output tile is computed by K_SPLIT=4
 * work-items collaborating along the K dimension. Each of the 4 WIs
 * does a quarter of the K reduction, writes its partial result to
 * __local memory, a barrier synchronizes, and the ly==0 lane sums the
 * 4 quarters and emits the output. Quadruples the number of
 * concurrent memory fetches per output element.
 *
 * Dispatch geometry: local=(32, 4, 1), global=(N/4, 4, ceilDiv(M, 8)).
 *
 * Shape constraints:
 *   - N % 4 == 0 (WI writes 4 cols)
 *   - K % 16 == 0 (K_SPLIT=4, each split processes multiples of 4)
 *
 * Same weight/input/scale/output layout contract as
 * gemm_int4_adreno_cl and gemm_int4_adreno_v2_cl.
 */
void gemm_int4_adreno_v3_cl(uint16_t *input, uint16_t *input_transposed,
                            uint16_t *weights, uint16_t *scales,
                            uint16_t *output, unsigned int M, unsigned int N,
                            unsigned int K);

/**
 * @brief INT4 channel-wise GEMV for Adreno GPUs (M = 1, gpu_int4_gemv_adreno
 *        kernel).
 *
 * Same weight / scale layout as gemm_int4_adreno_cl. Specialized for the
 * decode (single-token) path where the GEMM kernel's overhead per call
 * dominates.
 *
 * @param[in] input   activation buffer of K fp16 values (SVM)
 * @param[in] weights packed int4 weights, ushort[(K/4) * N] (SVM)
 * @param[in] scales  per-channel fp16 scales, length N (SVM)
 * @param[out] output  output of N fp16 values (SVM)
 * @param[in] K input dimension
 * @param[in] N output dimension
 */
void gemv_int4_adreno_cl(uint16_t *input, uint16_t *weights, uint16_t *scales,
                         uint16_t *output, unsigned int K, unsigned int N);

/**
 * @brief INT8 activation x INT4 weight GEMM for Adreno GPUs
 *        (gpu_int8_int4_gemm_adreno kernel).
 *
 * Activations are pre-quantized to signed int8 on the host; the Adreno
 * DP4A (ssad) path gives ~3.6x the fp16 FMA throughput on 7xx+ GPUs. The
 * weight layout is identical to `gemm_int4_adreno_cl` so callers can
 * reuse the same Int4Utils packed buffers.
 *
 * @param[in] x_q      int8 activation buffer of align(M,8) * K bytes.
 *                     Rows in [M, align(M,8)) must be zero-padded by the
 *                     caller so the kernel's vload4 is in-bounds. SVM ptr.
 * @param[in] x_scale  fp16 per-row activation scale, length M (SVM).
 * @param[in] w_scale  fp16 per-channel weight scale, length align(N,4)
 *                     (SVM). Tail padding entries are ignored.
 * @param[in] weights  packed int4 weights ushort[(K/4) * N] (SVM).
 * @param[out] output  output [M][N] fp16 (SVM).
 * @param[in] M batch / token count
 * @param[in] N output dimension
 * @param[in] K input dimension
 *
 * Assumes N % 4 == 0, K % 4 == 0.
 */
void gemm_int8_int4_adreno_cl(int8_t *x_q, uint16_t *x_scale,
                              uint16_t *w_scale, uint16_t *weights,
                              uint16_t *output, unsigned int M, unsigned int N,
                              unsigned int K);

/**
 * @brief     Q6_K sgemv computation : Y = A*X
 * @param[in] matAdata void * for Matrix A
 * @param[in] vecXdata float * for Vector X
 * @param[in] vecYdata float * for Vector Y
 * @param[in] M number of rows in matrix A
 * @param[in] N number of columns in matrix A
 */
void sgemv_q6_k_cl(void *matAdata, float *vecXdata, float *vecYdata,
                   unsigned int M, unsigned int N);

/**
 * @brief     sgemv computation : Y = A*X + Y
 * @param[in] matAdata float * for Matrix A
 * @param[in] vecXdata float * for Vector X
 * @param[in] vecYdata float * for Vector Y
 * @param[in] transA bool transpose
 * @param[in] dim1 number of A's columns
 * @param[in] dim2 number of A's rows
 * @param[in] lda number of X's columns
 * @param[in] context RunLayerContext reference
 */
void sgemv_cl(const float *matAdata, const float *vecXdata, float *vecYdata,
              bool TransA, unsigned int dim1, unsigned int dim2,
              unsigned int lda);

/**
 * @brief     dot computation : sum of all X * Y
 * @param[in] vecAdata float * for Vector A
 * @param[in] vecXdata float * for Vector X
 * @param[in] dim1 number of elements in both input vectors
 * @param[in] context RunLayerContext reference
 * @return    float dot product result
 */
float dot_cl(const float *vecAdata, const float *vecXdata, unsigned int dim1);

/**
 * @brief     sgemm computation : Y = op(A)*op(B) + C,
 * where op(X) is one of X or X**T
 * @param[in] transA bool transpose
 * @param[in] transB bool transpose
 * @param[in] A float * for Matrix A
 * @param[in] B float * for Matrix B
 * @param[in] C float * for Matrix C
 * @param[in] M number of op(A)'s and C's row
 * @param[in] N number of op(B)'s and C's columns
 * @param[in] K number of op(A)'s and columns and op(B)'s rows
 * @param[in] lda number of A's columns
 * @param[in] ldb number of B's columns
 * @param[in] ldc number of C's columns
 * @param[in] context RunLayerContext reference
 */
void sgemm_cl(bool TransA, bool TransB, const float *A, const float *B,
              float *C, unsigned int M, unsigned int N, unsigned int K,
              unsigned int lda, unsigned int ldb, unsigned int ldc);

/**
 * @brief     addition : sum of all input vectors
 * @param[in] input float * for input
 * @param[in] res float * for result/output
 * @param[in] size_input number of elements in input vector
 * @param[in] size_res number of elements in result vector
 */
void addition_cl(const float *input, float *res, unsigned int size_input,
                 unsigned int size_res);

/**
 * @brief rmsnorm each row of the tensor
 * @param[in] input float * for input
 * @param[in] gamma float * for gamma multiplier for each row
 * @param[in] result float * for result
 * @param[in] epsilon epsilon to add to each row sum to prevent division by zero
 * @param[in] height height of the tensor
 * @param[in] width width of the tensor
 * @param[in] use_svm whether to treat pointers as SVM
 */
void rmsnorm_cl(const float *input, const float *gamma, float *result,
                const float epsilon, unsigned int height, unsigned int width,
                const bool use_svm = true);

/**
 * @brief     sscal value element by element immediately
 * @param[in] X float * input
 * @param[in] N unsigned int number of elements
 * @param[in] alpha float multiplier
 * @param[in] context RunLayerContext reference
 */
void sscal_cl(float *X, const unsigned int N, const float alpha);

/**
 * @brief     transpose computation
 * @param[in] input float * for Input Tensor
 * @param[in] res float * for Output Tensor
 * @param[in] input_batch_size  represents the number of samples in the input
 * tensor
 * @param[in] input_channels   represents the channels of the input tensor
 * @param[in] input_height   represents the height of the input tensor
 * @param[in] input_width   represents the width of the input tensor
 * @param[in] axis   transpose about axis, 0-> channels & height, 1-> height &
 * width, 2-> channels & width
 */
void transpose_cl_axis(const float *in, float *res,
                       unsigned int input_batch_size,
                       unsigned int input_channels, unsigned int input_height,
                       unsigned int input_width, unsigned int axis);
/**
 * @brief  Separate the quantized bits and scale from block_q4_0
 *
 * @param src source pointer to the block_q4_0 data
 * @param dst_q destination pointer for the quantized bits
 * @param dst_d destination pointer for the scale
 * @param num_blocks number of blocks to process
 */
void flatten_block_q4_0_cl(const void *src, void *dst_q, void *dst_d,
                           unsigned int num_blocks);

/**
 * @brief Restore the original block_q4_0 from the quantized bits and scale
 *
 * @param src_q source pointer to the quantized bits
 * @param src_d source pointer to the scale
 * @param dst destination pointer for the restored block_q4_0
 * @param num_blocks number of blocks to process
 */
void restore_block_q4_0_cl(const void *src_q, const void *src_d, void *dst,
                           unsigned int num_blocks);

/**
 * @brief This kernel load & store a 4x4 tile of elements
 *
 * @param data Input FP32 matrix data
 * @param M width (row)
 * @param K height (col)
 *
 * @note This kernel is only used for activations
 * Activation is coverted to FP16 and adds zero padding for non multiple of 8
 * Output is not returned and instead saved to outBufferB
 */
void transpose_32_16(float *data, int M, int K);

/**
 * @brief This kernel transpose fp16 type
 *
 * @param data input fp16 matrix data
 * @param output output fp16 matrix data
 * @param width widh
 * @param height height
 * @param size_bytes data size in bytes
 *
 * @note Temporary disable transpose 16
 */
// void transpose_16(void *data, void *output, int width, int height,
//                   int size_bytes, bool isQuant = false);

#ifdef ENABLE_FP16

/**
 * @brief     fp16 sgemv computation : Y = A*X + Y
 * @param[in] matAdata fp16 * for Matrix A
 * @param[in] vecXdata fp16 * for Vector X
 * @param[in] vecYdata fp16 * for Vector Y
 * @param[in] transA bool transpose
 * @param[in] dim1 number of A's columns
 * @param[in] dim2 number of A's rows
 * @param[in] lda number of X's columns
 * @param[in] context RunLayerContext reference
 */
void sgemv_cl(const _FP16 *matAdata, const _FP16 *vecXdata, _FP16 *vecYdata,
              bool TransA, unsigned int dim1, unsigned int dim2,
              unsigned int lda);

/**
 * @brief     fp16 dot computation : sum of all X * Y
 * @param[in] vecAdata fp16 * for Vector A
 * @param[in] vecXdata fp16 * for Vector X
 * @param[in] dim1 number of elements in both input vectors
 * @param[in] context RunLayerContext reference
 * @return    fp16 dot product result
 */
_FP16 dot_cl(const _FP16 *vecAdata, const _FP16 *vecXdata, unsigned int dim1);

/**
 * @brief     fp16 sgemm computation : Y = op(A)*op(B) + C,
 * where op(X) is one of X or X**T
 * @param[in] transA bool transpose
 * @param[in] transB bool transpose
 * @param[in] A fp16 * for Matrix A
 * @param[in] B fp16 * for Matrix B
 * @param[in] C fp16 * for Matrix C
 * @param[in] M number of op(A)'s and C's row
 * @param[in] N number of op(B)'s and C's columns
 * @param[in] K number of op(A)'s and columns and op(B)'s rows
 * @param[in] lda number of A's columns
 * @param[in] ldb number of B's columns
 * @param[in] ldc number of C's columns
 * @param[in] context RunLayerContext reference
 */
void sgemm_cl(bool TransA, bool TransB, const _FP16 *A, const _FP16 *B,
              _FP16 *C, unsigned int M, unsigned int N, unsigned int K,
              unsigned int lda, unsigned int ldb, unsigned int ldc);

/**
 * @brief     fp16 addition : sum of all input vectors
 * @param[in] input fp16 * for input
 * @param[in] res fp16 * for result/output
 * @param[in] size_input number of elements in input vector
 * @param[in] size_res number of elements in result vector
 */
void addition_cl(const _FP16 *input, _FP16 *res, unsigned int size_input,
                 unsigned int size_res);

/**
 * @brief     fp16 sscal value element by element immediately
 * @param[in] X _FP16 * input
 * @param[in] N unsigned int number of elements
 * @param[in] alpha float multiplier
 * @param[in] context RunLayerContext reference
 */
void sscal_cl(_FP16 *X, const unsigned int N, const float alpha);

/**
 * @brief     transpose computation
 * @param[in] input fp16 * for Input Tensor
 * @param[in] res fp16 * for Output Tensor
 * @param[in] input_batch_size  represents the number of samples in the input
 * tensor
 * @param[in] input_channels   represents the channels of the input tensor
 * @param[in] input_height   represents the height of the input tensor
 * @param[in] input_width   represents the width of the input tensor
 * @param[in] axis   transpose about axis, 0-> channels & height, 1-> height &
 * width, 2-> channels and width
 */
void transpose_cl_axis(const _FP16 *in, _FP16 *res,
                       unsigned int input_batch_size,
                       unsigned int input_channels, unsigned int input_height,
                       unsigned int input_width, unsigned int axis);
#endif

} // namespace nntrainer
#endif /* __BLAS_KERNELS_H__ */
