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
/**
 * @brief Publish SVM→image2d when the SVM was last written by a GPU kernel.
 *
 * Same as svm_to_image2d_publish but skips the SVMUnmap fence: the caller
 * guarantees that svm_ptr was written by a GPU kernel in the same in-order
 * command queue, so no CPU-GPU coherence operation is required.
 */
void svm_to_image2d_publish(void *svm_ptr, unsigned int M, unsigned int K,
                             bool gpu_source = false);

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
 * @brief SVM-direct fp16 RMSNorm with cooperative WG=64 reduction.
 *
 * Same algorithm as rmsnorm_image2d_v2 but reads/writes SVM tensors
 * directly -- no svm_to_image2d, image2d_to_svm, or pool publish
 * round-trip. Built for the M==1 decode hot path where the four
 * image2d dispatches and the blocking SVMMap fence dominate the
 * actual compute by ~17x. For prefill (M >> 1) the image2d_cl path
 * stays preferable because the reformat amortises over many rows.
 *
 * Args:
 *   in_svm   : SVM half[H_rows * W] (row-major)
 *   out_svm  : SVM half[H_rows * W]
 *   gamma    : __global float[W] (cl_mem cached host-side, fp32)
 *   H_rows   : number of independent rows to normalise (= 1 for decode)
 *   W        : channel dim (must match gamma length)
 *   epsilon  : RMSNorm epsilon
 *
 * No exit SVMMap drain: same OpenCL queue ordering serialises with
 * downstream GPU consumers; Tensor::copy / NEON consumers must do
 * their own entry drain (none of the model's downstream consumers
 * of this output do that today).
 */
void rmsnorm_fp16_svm_cl(void *in_svm, void *out_svm,
                          const float *gamma, unsigned int H_rows,
                          unsigned int W, float epsilon);

#ifdef ENABLE_FP16
/**
 * @brief Fused FlashAttention dispatch on the GPU queue.
 *
 * Replaces the NEON compute_kcaches + softmax_triangle +
 * compute_fp16vcache_transposed triple with one kernel.  Uses an
 * online-softmax running max/sum so the attention matrix is never
 * materialised.
 *
 * All tensors fp16; Q and the KV cache are already on SVM
 * (FC output for Q; MHA's cache SVM allocation for K/V).  The output
 * attention tensor on the Qwen3 prefill path is SVM too, so the helper
 * writes directly into it.
 *
 * head_dim is fixed to 128 (the kernel hard-codes HD); the layer-side
 * gate enforces that before calling.
 */
void attention_fused_fp16_cl(void *q_svm, void *k_cache_svm,
                              void *v_cache_svm,
                              void *out_svm,
                              unsigned int M, unsigned int T,
                              unsigned int from,
                              unsigned int num_heads_Q,
                              unsigned int gqa_size,
                              unsigned int head_dim, int is_causal);

/**
 * @brief Decode-only GPU RoPE for FP16 (M=1).
 *
 * Dispatches the rope_decode_fp16 kernel which applies rotary
 * embedding in-kernel using on-the-fly cos/sin (libm-precise) and
 * native_exp for the per-dim theta.  In-place when out_svm == in_svm,
 * out-of-place when writing to a kv-cache slice.
 *
 * Returns false on shape constraint or driver error so the caller
 * can fall back to the CPU path.
 *
 * head_dim must be even and <= 128 (WG = head_dim/2 lanes <= 64).
 */
bool rope_decode_fp16_cl(void *in_svm, void *out_svm,
                          unsigned int num_heads, unsigned int head_dim,
                          unsigned int position, float theta_base);

/**
 * @brief Fused Q+K decode RoPE — one dispatch instead of two.
 * Applies rotary embedding to both Q (in-place) and K (-> kc_out)
 * in a single clEnqueueNDRangeKernel call, saving ~18 us host overhead
 * per decode step.  head_dim must be 128.
 */
bool rope_decode_fp16_qk_cl(void *q_in, void *q_out,
                              void *k_in, void *kc_out,
                              unsigned int num_heads_Q, unsigned int num_heads_K,
                              unsigned int head_dim,
                              unsigned int position, float theta_base);

/**
 * @brief GPU SVM-to-SVM memcpy via clEnqueueSVMMemcpy on blas_cc's
 * queue. Used to publish V-cache writes without paying a host-side
 * SVMMap drain.
 */
bool svm_memcpy_fp16_cl(const void *src_svm, void *dst_svm, size_t bytes);

/**
 * @brief Returns (lazy-allocating) the 4-byte SVM int buffer used by
 * gpu_argmax_fp16_cl to store the argmax result.  Stable for the
 * process lifetime.  nullptr on allocation failure.
 */
void *gpu_argmax_fp16_result_svm();

/**
 * @brief GPU argmax over FP16 logits.  Dispatches gpu_argmax_fp16 (single
 * 256-thread workgroup, strided loop + tree reduction) on the in-order
 * blas queue.  result_svm must be a 4-byte SVM int buffer; it is written
 * with the argmax index.  The result is coherent after the next blocking
 * enqueueSVMMap on the same queue (i.e. the lm_head output drain).
 */
bool gpu_argmax_fp16_cl(void *logits_svm, void *result_svm, int n);

/**
 * @brief GPU embedding lookup from INT4 chunked cache (async pipeline).
 *  token_id_svm: SVM int* written by a previous gpu_argmax_fp16_cl call.
 *  output_svm:   SVM half* for the K-element embedding row output.
 *  K:            embedding dimension.
 * Requires lmhead_int4_chunked_is_initialized() == true.
 * Returns false if cache not initialized or dispatch fails.
 */
bool gpu_embedding_int4chunked_cl(int *token_id_svm, uint16_t *output_svm,
                                   unsigned int K, float emb_scale = 1.0f);

/**
 * @brief Async pipeline context: set before each non-blocking
 * incremental_inference() call.  token_id_svm is the SVM int read by the
 * embedding GPU kernel; argmax_out_svm is the SVM int written by the
 * lm_head GPU argmax kernel.  Both must remain valid until the final
 * blocking drain.  Pass nullptrs to disable the async path.
 */
void set_async_pipeline_ctx(int *token_id_svm, int *argmax_out_svm);
int *get_async_pipeline_token_id_svm();
int *get_async_pipeline_argmax_out_svm();

/**
 * @brief Allocate n ints in coarse-grained SVM (CL_MEM_READ_WRITE).
 *        Returns nullptr on failure.  Wraps clSVMAlloc so callers
 *        outside libnntrainer.so do not need to link libOpenCL directly.
 */
int *svm_alloc_ints(int n);

/**
 * @brief Free a pointer previously returned by svm_alloc_ints.
 *        Wraps clSVMFree; safe to call with nullptr.
 */
void svm_free_ptr(void *p);

/**
 * @brief Blocking SVMMap(CL_MAP_READ) on the GPU command queue.
 *        Drains all in-flight GPU work that writes to [p, p+bytes).
 *        Wraps enqueueSVMMap(blocking=true) so callers outside
 *        libnntrainer.so do not need to link libOpenCL directly.
 */
void svm_blocking_read(void *p, size_t bytes);

#endif

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
 * @brief Phase M helper -- gate proj + up proj + SwiGLU fused into a
 * single int4 GEMV dispatch.
 *
 * Replaces the [fused_gemv_int4(gate, up) -> SwiGLU] pair with one
 * kernel that loads gate/up int4 weights via image2d (RGBA32UI v2
 * layout, same as fused_gemv_int4_image2d_v2), accumulates both
 * partials in registers, applies scales, and computes
 *   silu(scale_g * gate_lin) * (scale_u * up_lin)
 * before writing a single output. Eliminates the separate SwiGLU
 * dispatch and the gate / up intermediate SVM writes.
 *
 * input_svm  : layer input, K fp16 elements (post-rmsnorm hidden state)
 * gate_w/u   : int4 packed weights in v2 RGBA32UI layout
 * gate_s/u_s : per-N fp16 channel scales
 * out_svm    : N fp16 elements (the swiglu_out)
 * sync_output: when true, blocking SVMMap on out_svm before return
 */
/**
 * @brief Phase R0 PoC -- gate-up int4 GEMV that writes its outputs
 * to cl_mem buffers (instead of SVM). Same kernel as
 * fused_gemv_int4_image2d_v2, just different arg binding for the
 * output side.
 *
 * Why: Adreno's coarse-grained SVM races on inter-kernel data flow
 * (gate_up SVM write -> swiglu SVM read). cl_mem buffers don't
 * have this issue per OpenCL spec (in-order queue gives strict
 * ordering on cl_mem reads/writes). Goal: prove cl_mem fixes the
 * race so we can eliminate the structural drain.
 *
 * Caller allocates cl_mem buffers (one for gate, one for up) sized
 * to N_g * fp16 and N_u * fp16 respectively.
 */
// Lazy-allocated cl_mem buffers for the PoC. Returns nullptr on alloc
// failure. The buffer is sized to N elements of fp16; reused across
// layers if size is sufficient. Single-threaded decode -> no contention.
cl_mem clmem_poc_gate_buf(unsigned int N_q_fp16);
cl_mem clmem_poc_up_buf(unsigned int N_k_fp16);

bool fused_gemv_int4_image2d_clmem_out_cl(uint16_t *input_svm,
                                           uint16_t *q_weights,
                                           uint16_t *q_scales,
                                           cl_mem q_out_clmem,
                                           uint16_t *k_weights,
                                           uint16_t *k_scales,
                                           cl_mem k_out_clmem,
                                           unsigned int K,
                                           unsigned int N_q,
                                           unsigned int N_k);

/**
 * @brief Phase R0 PoC -- swiglu reading inputs from cl_mem buffers,
 * writing output to SVM. Pair with the cl_mem output of the
 * previous helper to test whether cl_mem auto-coheres across
 * kernels in the same blas_cc queue.
 */
bool swiglu_fp16_clmem_in_cl(cl_mem in1_clmem, cl_mem in2_clmem,
                              void *out_svm, size_t total);

bool gateup_swiglu_int4_image2d_cl(uint16_t *input_svm,
                                    uint16_t *gate_weights,
                                    uint16_t *gate_scales,
                                    uint16_t *up_weights,
                                    uint16_t *up_scales,
                                    uint16_t *out_svm,
                                    unsigned int M,
                                    unsigned int K,
                                    unsigned int N,
                                    bool sync_output);

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
// sync_output=true (default): helper issues a blocking enqueueSVMMap(output)
// before returning, so the caller can scalar-read output immediately.  Set
// false when the caller writes the kernel result to an SVM output tensor
// that the DOWNSTREAM GPU kernel (on the same queue) or a dedicated later
// CPU fence will consume -- saves the 200-400 us per-call sync on
// Adreno 830 coarse-grained SVM.
void gemv_int4_adreno_cl(uint16_t *input, uint16_t *weights, uint16_t *scales,
                         uint16_t *output, unsigned int K, unsigned int N,
                         bool sync_output = true);

// Same contract as gemv_int4_adreno_cl, but binds `input` through
// __constant memory.  Tests whether routing the activation vector
// through Adreno's per-CU constant cache (separate from L1) measurably
// reduces per-call wall vs baseline.  Modeled after LiteRT's
// program_002.cl (matmul_micro_benchmark int8 capture) which uses
// __constant for its xmem buffer.
void gemv_int4_adreno_v2_cl(uint16_t *input, uint16_t *weights,
                            uint16_t *scales, uint16_t *output,
                            unsigned int K, unsigned int N,
                            bool sync_output = true);

// v3 = v2 (__constant input) + LiteRT-style QCOM-specific hints.
// Adds `__attribute__((sub_group_uniform))` on the input pointer
// (cl_qcom_subgroup_uniform_load) so the driver can broadcast the
// uniform vload4(input + k) to all subgroup lanes from a single
// load, and `__attribute__((qcom_max_concurrent_subgroups(12)))` on
// the kernel for occupancy. Both attributes degrade to no-ops on
// non-Adreno via #ifdef cl_qcom_subgroup_uniform_load. Algorithm and
// dispatch geometry are otherwise byte-identical to v1/v2.
void gemv_int4_adreno_v3_cl(uint16_t *input, uint16_t *weights,
                            uint16_t *scales, uint16_t *output,
                            unsigned int K, unsigned int N,
                            bool sync_output = true);

// Phase A1 (image2d weight migration): same algorithm as v3 but
// reads the int4 packed weight matrix from a CL_RGBA16UI image2d
// view of the SVM weight bytes. Image2D wrapper is cached per
// weight pointer at first call. Returns false (caller falls back
// to SVM v3 path) when N/4 or K/4 exceeds the device's image2d
// max dimensions (e.g. lm_head N=152064 doesn't fit), or on
// allocation failure.
bool gemv_int4_weight_image2d_cl(uint16_t *input, uint16_t *weights,
                                  uint16_t *scales, uint16_t *output,
                                  unsigned int K, unsigned int N,
                                  bool sync_output = true);

// v2: same contract as gemv_int4_weight_image2d_cl but uses RGBA32UI
// weight image (8 K-positions per pixel, K/8 height) for 2x reduction
// in imageread count. Repacks the ushort SVM weight into a uint heap
// buffer on first call (cached). Memory cost: K*N/2 bytes per weight.
// Returns false on shape (K%8 || N%4), image dim limit, alloc fail.
bool gemv_int4_weight_image2d_v2_cl(uint16_t *input, uint16_t *weights,
                                     uint16_t *scales, uint16_t *output,
                                     unsigned int K, unsigned int N,
                                     bool sync_output = true);

// v4: same uint-packed memory layout as v2 image2d but read via
// __global const uint4 * (no image2d size limit, so handles N >
// CL_DEVICE_IMAGE2D_MAX_WIDTH * 4 -- e.g. lm_head's vocab projection).
// Same in-place ushort → uint repack as v2 (zero memory delta).
bool gemv_int4_adreno_v4_cl(uint16_t *input, uint16_t *weights,
                             uint16_t *scales, uint16_t *output,
                             unsigned int K, unsigned int N,
                             bool sync_output = true);

// Step 1..N of the incremental image2d M=1 gemv rewrite.  Each step
// loads more of the gemv work (step 1 = weight only, step 2 = +
// input, ...) but all write deterministic zero until correctness
// is restored in step 4.  Useful for per-kernel timing via
// cl_event; NOT correct by design for steps 1..3.  Env-gated via
// NNTRAINER_GEMV_IMAGE_STEP=N in HalfTensor::dotQInteger M=1.
void gemv_int4_image_v1_cl(uint16_t *input, uint16_t *weights,
                           uint16_t *scales, uint16_t *output,
                           unsigned int K, unsigned int N);
void gemv_int4_image_v2_cl(uint16_t *input, uint16_t *weights,
                           uint16_t *scales, uint16_t *output,
                           unsigned int K, unsigned int N);
void gemv_int4_image_v3_cl(uint16_t *input, uint16_t *weights,
                           uint16_t *scales, uint16_t *output,
                           unsigned int K, unsigned int N);
void gemv_int4_image_v4_cl(uint16_t *input, uint16_t *weights,
                           uint16_t *scales, uint16_t *output,
                           unsigned int K, unsigned int N);
void gemv_int4_image_v5_cl(uint16_t *input, uint16_t *weights,
                           uint16_t *scales, uint16_t *output,
                           unsigned int K, unsigned int N);

/**
 * @brief Phase 2 image2d gemv: image2d input + image2d output.
 *
 * Reads input via image2d (must already be registered in GpuImagePool
 * keyed by @p input_svm with shape M=1, height=K/4 -- typically
 * published by upstream RMSNorm via svm_to_image2d_publish).  Writes
 * output to a cached image2d (allocated/looked-up keyed by
 * @p output_svm) and registers it in GpuImagePool for the next
 * image2d-aware consumer.
 *
 * Avoids Adreno 830 coarse-grained SVM cross-kernel staleness entirely:
 * the FC -> consumer chain is image-cache only, no SVM read between
 * GPU kernels.
 *
 * @returns true if dispatched successfully, false on pool miss / shape
 *   mismatch / unsupported K,N.  Caller is expected to fall back to
 *   gemv_int4_adreno_cl (with sync_output=true) on miss so correctness
 *   is preserved during decode.
 */
bool gemv_int4_image2d_cl(uint16_t *input_svm, uint16_t *weights,
                          uint16_t *scales, uint16_t *output_svm,
                          unsigned int K, unsigned int N);

/**
 * @brief Phase 2 image-only consumer helpers.
 *
 * SwiGLU and elementwise addition that take BOTH inputs from
 * GpuImagePool (typically the previous FC's image2d output published
 * by gemv_int4_image2d_cl) and write a fresh image2d output keyed by
 * @p out_svm.  Letting the FC -> consumer chain stay entirely on
 * image cache means downstream layers never have to read the SVM
 * companion of an image2d FC, so Adreno's coarse-grained SVM
 * coherence never bites and the per-FC blocking SVMMap (DEBUG_SYNC)
 * can eventually be dropped.
 *
 * Both helpers return false on pool miss / shape mismatch so the
 * caller can fall back to its existing SVM path.  Output image2d
 * shape is (M, K/4); K must be a multiple of 4.
 */
bool swiglu_image2d_cl(void *gate_svm, void *up_svm, void *out_svm,
                       unsigned int M, unsigned int K);
bool addition_image2d_cl(void *a_svm, void *b_svm, void *out_svm,
                         unsigned int M, unsigned int K);

/**
 * @brief Phase 1A step 1: fused RMSNorm + Q/K/V projection.
 *
 * One GPU dispatch replaces the canonical 4 (rms_norm + 3x
 * fully_connected).  Saves 3 SVMMap drains + 3 dispatch overheads
 * + the SVM round-trip for the normalised hidden vector.  All
 * weights and scales are channel-wise int4 quantised, same layout
 * as gemv_int4_adreno_cl.
 *
 * @param input_svm  fp16 hidden vector, length K_in (SVM)
 * @param gamma_svm  fp32 rmsnorm gamma, length K_in (SVM)
 * @param q_weights  ushort[(K_in/4) * N_q]
 * @param q_scales   half[N_q]
 * @param q_out      half[N_q] (SVM) -- written
 * @param k_weights  ushort[(K_in/4) * N_k]
 * @param k_scales   half[N_k]
 * @param k_out      half[N_k] (SVM) -- written
 * @param v_weights  ushort[(K_in/4) * N_v]
 * @param v_scales   half[N_v]
 * @param v_out      half[N_v] (SVM) -- written
 * @param K_in       hidden dimension (Qwen3-4B: 2560)
 * @param N_q        Q output dim (num_heads_q * head_dim)
 * @param N_k        K output dim (num_heads_kv * head_dim)
 * @param N_v        V output dim (= N_k for GQA)
 * @param epsilon    rmsnorm epsilon
 *
 * @returns false on shape constraint violation (K_in > 2560 or
 *   N_{q,k,v} not divisible by 64); caller must fall back to the
 *   separate rmsnorm + 3 FCs path.
 */
bool fused_rmsnorm_qkv_cl(uint16_t *input_svm,
                          const float *gamma_svm,
                          uint16_t *q_weights, uint16_t *q_scales,
                          uint16_t *q_out,
                          uint16_t *k_weights, uint16_t *k_scales,
                          uint16_t *k_out,
                          uint16_t *v_weights, uint16_t *v_scales,
                          uint16_t *v_out,
                          unsigned int K_in,
                          unsigned int N_q,
                          unsigned int N_k,
                          unsigned int N_v,
                          float epsilon);

/**
 * @brief Fused post-attention RMSNorm + Gate/Up projection for M=1
 *   decode.  Replaces 3 dispatches (rmsnorm + gate_proj gemv + up_proj
 *   gemv) with a single dispatch: cooperative sum_sq, __local
 *   normalised input cache, then per-WI MAC over Gate/Up partitions.
 *   Saves 2 SVMMap drains per layer (the two intermediate FC outputs
 *   no longer cross host/device boundary in between).
 *
 *   For Qwen3-4B: K_in = 2560, N_gate = N_up = 9728.
 *
 * @returns false on shape constraint violation (K_in > 2560 or
 *   N_{gate,up} not divisible by 64).
 */
bool fused_rmsnorm_gate_up_cl(uint16_t *input_svm,
                              const float *gamma_svm,
                              uint16_t *gate_weights,
                              uint16_t *gate_scales,
                              uint16_t *gate_out,
                              uint16_t *up_weights,
                              uint16_t *up_scales,
                              uint16_t *up_out,
                              unsigned int K_in,
                              unsigned int N_gate,
                              unsigned int N_up,
                              float epsilon);

/**
 * @brief Multi-output int4 GEMV for M=1 decode -- fuses 2 or 3
 *   projection partitions that share the same activation into a
 *   single dispatch.  Use cases: QKV (3 partitions) and Gate/Up MLP
 *   (2 partitions; pass v_weights = q_weights, v_scales = q_scales,
 *   v_out = q_out, N_v = 0 so the V branch is never taken).
 *
 *   Saves (input.size() - 1) SVMMap drains per attention/MLP block
 *   relative to looping baseline gemv_int4_adreno_cl per weight.
 *
 * @returns false on shape constraint violation (K not div by 4 or
 *   any N not div by 4).
 */
bool fused_gemv_int4_cl(uint16_t *input_svm,
                        uint16_t *q_weights, uint16_t *q_scales,
                        uint16_t *q_out,
                        uint16_t *k_weights, uint16_t *k_scales,
                        uint16_t *k_out,
                        uint16_t *v_weights, uint16_t *v_scales,
                        uint16_t *v_out,
                        unsigned int K,
                        unsigned int N_q,
                        unsigned int N_k,
                        unsigned int N_v,
                        bool sync_output = true);

// v2: same as fused_gemv_int4_cl but binds `input_svm` through
// __constant memory.  Routes the activation vector through Adreno's
// per-CU constant cache (separate from L1).  Same trick that
// gpu_int4_gemv_adreno_v3 used to gain +1.1% TPS on the per-FC path;
// this helper applies it to the fused QKV/Gate-Up dispatch which
// represents a much larger share of decode weight reads.
bool fused_gemv_int4_v2_cl(uint16_t *input_svm,
                           uint16_t *q_weights, uint16_t *q_scales,
                           uint16_t *q_out,
                           uint16_t *k_weights, uint16_t *k_scales,
                           uint16_t *k_out,
                           uint16_t *v_weights, uint16_t *v_scales,
                           uint16_t *v_out,
                           unsigned int K,
                           unsigned int N_q,
                           unsigned int N_k,
                           unsigned int N_v,
                           bool sync_output = true);

// Phase A2/A3 (image2d migration extends to fused gemv): same as
// fused_gemv_int4_cl but reads each partition's int4 weights from a
// CL_RGBA16UI image2d view (zero-copy via cl_khr_image2d_from_buffer
// over the SVM weight bytes).  Builds on Phase B1's per-FC win.
// Returns false on shape constraint violation or image2d allocation
// failure; caller falls back to fused_gemv_int4_cl.
bool fused_gemv_int4_image2d_cl(uint16_t *input_svm,
                                 uint16_t *q_weights, uint16_t *q_scales,
                                 uint16_t *q_out,
                                 uint16_t *k_weights, uint16_t *k_scales,
                                 uint16_t *k_out,
                                 uint16_t *v_weights, uint16_t *v_scales,
                                 uint16_t *v_out,
                                 unsigned int K,
                                 unsigned int N_q,
                                 unsigned int N_k,
                                 unsigned int N_v,
                                 bool sync_output = true);

/**
 * @brief Diagnostic-only helper: read a published image2d from
 * GpuImagePool back into @p dst_svm via image2d_to_svm.  Lets a
 * unittest cross-check what a producer kernel wrote to image2d
 * vs the SVM companion it wrote in parallel.
 */
bool image2d_to_svm_for_test(void *src_svm_key, void *dst_svm,
                              unsigned int M, unsigned int N);

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
 * @brief fp16 IO variant of sgemv_q6_k_cl for the lm_head path.
 * @param matAdata SVM Q6_K weight
 * @param vecXdata SVM fp16 activation, K halfs
 * @param vecYdata SVM fp16 logits output, N halfs
 * @param M = K (inner accumulation dim, must be % QK_K==0)
 * @param N = N (output dim, must be % N_SIMDGROUP==0)
 */
void sgemv_q6_k_fp16_cl(void *matAdata, uint16_t *vecXdata, uint16_t *vecYdata,
                         unsigned int M, unsigned int N);

/**
 * @brief One-time conversion of Q6_K lm_head weight to channel-wise int4
 *        chunked uint layout (3 chunks fitting Adreno image2d max width).
 *        Allocates SVM buffers, runs CPU dequant + re-quant, and prepares
 *        image2d_from_buffer views per chunk. Optionally madvises the
 *        original Q6_K bytes (NNTRAINER_LMHEAD_INT4_MADVISE=1) to release
 *        physical pages back to the OS.
 * @param q6k_weight pointer to host-allocated Q6_K weight bytes
 * @param K inner dim (must be % 256 = 0, Q6_K block size)
 * @param N output dim (must be % 4 = 0)
 * @return true on success, false on shape / alloc / image dim failure
 */
bool lmhead_int4_chunked_init(void *q6k_weight, unsigned int K, unsigned int N);

/**
 * @brief Returns true once the chunked int4 cache has been built. Use this
 *        from OpenMP-parallel paths to gate dispatch without re-entering
 *        the init's mutex (which would also try to allocate).
 */
bool lmhead_int4_chunked_is_initialized();

/**
 * @brief Embedding-mode lookup that dequantizes a single row from the
 *        chunked int4 cache to fp16 halfs (raw uint16 bits). Caller must
 *        have called lmhead_int4_chunked_init() once before.
 */
void lmhead_int4_chunked_embedding_u16(unsigned int row, uint16_t *out,
                                        unsigned int K);

/**
 * @brief fp32 output variant of the embedding row dequant.
 */
void lmhead_int4_chunked_embedding_f32(unsigned int row, float *out,
                                        unsigned int K);

/**
 * @brief Lmhead GPU dispatch through the chunked int4 cache. Runs the v3
 *        image2d gemv kernel 3 times (one per chunk) and drains the queue
 *        on the full output via enqueueSVMMap.
 */
void lmhead_int4_chunked_dispatch_cl(uint16_t *input, uint16_t *output,
                                      unsigned int K, unsigned int N);

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
