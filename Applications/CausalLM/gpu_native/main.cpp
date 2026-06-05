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
#include <cstring>
#include <dlfcn.h>
#include <engine.h>
#include <limits>
#include <string>
#include <vector>

// Defined in libnntrainer (blas_kernels.cpp); used by the in-process LWS sweep.
namespace nntrainer {
void set_v8c_lws_override(int lx, int ly);
void set_v8c_prefetch_override(int mode);
void set_v8c_tile_override(int tm, int tn);
void set_v8c_nocompute_override(int on);
}

// Bypass the production safety gate in two_conv_attention_prefill_f16_cl.
// The from-scratch runtime explicitly accepts GPU baseline as reference
// (paper §3.6 same-numerics chain), so the existing "Using CPU mha"
// fallback in the production wrapper isn't useful here.
static struct EnvSetup {
  EnvSetup() { setenv("NNTR_MHA_VERIFY", "1", 1); }
} _env_setup;

// #69 On-device compute-peak microbench (NNTR_PEAK_BENCH=1). Register-resident
// loops (NO memory traffic inside the loop) measure the Adreno int8-dp4a peak
// — the ceiling our int8×int4 FC GEMM competes against — and the fp16-FMA peak.
// 16 independent dependency chains hide instruction latency so we measure
// THROUGHPUT, not latency. Counted as flops (dp4a = 4 MAC = 8 flop; fma = 2
// flop) to match the GEMM's "TOP/s" = 2×MAC convention (our v8c FC = 5.08).
static const char *kPeakKernels = R"CLC(
#pragma OPENCL EXTENSION cl_khr_integer_dot_product : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define C32(op) op(0) op(1) op(2) op(3) op(4) op(5) op(6) op(7) \
  op(8) op(9) op(10) op(11) op(12) op(13) op(14) op(15) \
  op(16) op(17) op(18) op(19) op(20) op(21) op(22) op(23) \
  op(24) op(25) op(26) op(27) op(28) op(29) op(30) op(31)
// 32 independent dp4a recurrence chains, no extra ALU per dp4a (just the
// accumulate, matching the GEMM's acc += dot). Lots of in-flight WIs + 32
// chains/WI saturate throughput.
__kernel void dp4a_peak(__global int *out, const int iters, const uint b) {
  int a[32];
  #pragma unroll
  for (int j=0;j<32;j++) a[j]=(int)get_global_id(0)+j+1;
  for (int i=0;i<iters;i++) {
    #define DP(j) a[j]=dot_4x8packed_su_int(as_uint(a[j]),b)+a[j];
    C32(DP)
    #undef DP
  }
  int s=0;
  #pragma unroll
  for (int j=0;j<32;j++) s+=a[j];
  out[get_global_id(0)]=s;
}
// fp16 FMA peak: 32 independent half2 chains (2 lanes × 1 fma). Measures the
// fp16-FMA throughput ceiling — the alternative path an int4->fp16 dequant GEMM
// would target. The recurrence a = a*b + c with b<1 CONVERGES (stays bounded),
// so values never overflow to inf (which would trigger a slow special-case
// path and mismeasure the peak). Call with b = 0.5h.
// Mirrors dp4a_peak exactly but with scalar half FMA (rules out half2/spill
// artifacts). a=a*b+c with b=0.5h converges to 2 (bounded, no inf).
__kernel void fma_peak(__global half *out, const int iters, const half bb) {
  half a[32]; half c = (half)1.0h;
  #pragma unroll
  for (int j=0;j<32;j++) a[j]=(half)((get_global_id(0)+j)&3);
  for (int i=0;i<iters;i++) {
    #define FM(j) a[j]=fma(a[j],bb,c);
    C32(FM)
    #undef FM
  }
  half s=(half)0;
  #pragma unroll
  for (int j=0;j<32;j++) s+=a[j];
  out[get_global_id(0)]=s;
}
)CLC";

static void run_peak_bench() {
  auto *cl = static_cast<nntrainer::ClContext *>(
    nntrainer::Engine::Global().getRegisteredContext("gpu"));
  if (!cl) { std::fprintf(stderr, "[peak] no gpu context\n"); return; }
  cl_command_queue q = cl->command_queue_inst_.GetCommandQueue();
  cl_context ctx = nullptr;
  clGetCommandQueueInfo(q, CL_QUEUE_CONTEXT, sizeof(ctx), &ctx, nullptr);
  // #69b: dump device HW facilities — which fast ops does THIS Adreno expose,
  // and are we using them? (dp4a=yes; ml_ops/command_buffer/recordable=?)
  {
    cl_device_id dev = nullptr;
    clGetCommandQueueInfo(q, CL_QUEUE_DEVICE, sizeof(dev), &dev, nullptr);
    static char buf[16384];
    size_t n = 0;
    clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof(buf), buf, &n);
    std::fprintf(stderr, "[hw] device: %s\n", buf);
    clGetDeviceInfo(dev, CL_DEVICE_VERSION, sizeof(buf), buf, &n);
    std::fprintf(stderr, "[hw] version: %s\n", buf);
    clGetDeviceInfo(dev, CL_DEVICE_EXTENSIONS, sizeof(buf), buf, &n);
    std::fprintf(stderr, "[hw] extensions: %s\n", buf);
    const char *probe[] = {"cl_qcom_ml_ops", "cl_khr_command_buffer",
                           "cl_qcom_recordable_queue", "cl_khr_integer_dot_product",
                           "cl_qcom_dot_product8", "cl_qcom_subgroup",
                           "cl_qcom_accelerated_image_ops", "cl_khr_subgroups",
                           "cl_qcom_extended_query_image_info", "matrix"};
    for (auto *p : probe)
      std::fprintf(stderr, "[hw]   %-34s : %s\n", p,
                   std::strstr(buf, p) ? "YES" : "no");
  }
  const size_t GS = 256 * 1024;   // work-items
  const size_t LWS = 64;
  cl_int err = 0;
  cl_mem out_i = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, GS * sizeof(int),
                                nullptr, &err);
  cl_mem out_h = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, GS * sizeof(uint16_t),
                                nullptr, &err);
  auto NOW = []() { return std::chrono::steady_clock::now(); };
  auto SEC = [](auto t1, auto t0) {
    return std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0)
             .count() / 1e6;
  };
  std::array<size_t, 1> gws = {GS}, lws = {LWS};

  // ---- int8 dp4a peak ----
  for (int iters : {4096, 16384}) {
    auto kp = cl->registerClKernel(kPeakKernels, "dp4a_peak");
    uint32_t b = 0x01010101u;
    kp->SetKernelArguments(0, &out_i, sizeof(cl_mem));
    kp->SetKernelArguments(1, &iters, sizeof(int));
    kp->SetKernelArguments(2, &b, sizeof(uint32_t));
    cl->command_queue_inst_.enqueueKernel(kp->GetKernel(), 1, gws.data(),
                                          lws.data(), 0, nullptr, nullptr);
    clFinish(q);  // warmup
    auto t0 = NOW();
    cl->command_queue_inst_.enqueueKernel(kp->GetKernel(), 1, gws.data(),
                                          lws.data(), 0, nullptr, nullptr);
    clFinish(q);
    double s = SEC(NOW(), t0);
    double flop = (double)GS * iters * 32.0 * 8.0;  // 32 chains × dp4a(8 flop)
    std::fprintf(stderr, "[peak] int8-dp4a iters=%d : %.3f ms => %.2f TOP/s\n",
                 iters, s * 1e3, flop / s / 1e12);
  }
  std::fprintf(stderr, "[peak] (our v8c int8xint4 FC in-forward = 5.08 TOP/s; "
                       "this register-only loop is the int8-dp4a ceiling)\n");

  // ---- fp16 half2 FMA peak ----
  for (int iters : {4096, 16384}) {
    auto kp = cl->registerClKernel(kPeakKernels, "fma_peak");
    cl_half bb = 0x3800; // 0.5h (bounded recurrence a=a*0.5+1 -> 2)
    kp->SetKernelArguments(0, &out_h, sizeof(cl_mem));
    kp->SetKernelArguments(1, &iters, sizeof(int));
    kp->SetKernelArguments(2, &bb, sizeof(cl_half));
    cl->command_queue_inst_.enqueueKernel(kp->GetKernel(), 1, gws.data(),
                                          lws.data(), 0, nullptr, nullptr);
    clFinish(q); // warmup
    auto t0 = NOW();
    cl->command_queue_inst_.enqueueKernel(kp->GetKernel(), 1, gws.data(),
                                          lws.data(), 0, nullptr, nullptr);
    clFinish(q);
    double s = SEC(NOW(), t0);
    // 32 scalar-half chains × fma(2 flop) = 64 flop per WI per iter.
    double flop = (double)GS * iters * 32.0 * 2.0;
    std::fprintf(stderr, "[peak] fp16-fma  iters=%d : %.3f ms => %.2f TOP/s\n",
                 iters, s * 1e3, flop / s / 1e12);
  }
  std::fprintf(stderr,
               "[peak] (fp16 peak >> int8-dp4a peak ? then int4->fp16 dequant "
               "GEMM can raise the compute ceiling)\n");

  clReleaseMemObject(out_i);
  clReleaseMemObject(out_h);
}

// Phase-0 gate #1 for the cl_qcom_ml_ops matrix-engine GEMM: does THIS device
// actually expose the ML-ops entrypoints? (Advertised extensions can be
// non-functional — e.g. recordable_queues here.) The QCOM ML API is reached
// through a versioned interface table obtained from clGetMLInterfaceVxQCOM /
// clQueryMLInterfaceVersionsQCOM, resolved via the platform extension-function
// query. We only check non-null here (no calls) — if these resolve, ml_ops is
// callable and the GEMM spike is worth building; if null, ml_ops is a dead end.
static void run_mlops_probe() {
  auto *cl = static_cast<nntrainer::ClContext *>(
    nntrainer::Engine::Global().getRegisteredContext("gpu"));
  if (!cl) { std::fprintf(stderr, "[mlops] no gpu context\n"); return; }
  cl_command_queue q = cl->command_queue_inst_.GetCommandQueue();
  cl_device_id dev = nullptr;
  clGetCommandQueueInfo(q, CL_QUEUE_DEVICE, sizeof(dev), &dev, nullptr);
  cl_platform_id plat = nullptr;
  clGetDeviceInfo(dev, CL_DEVICE_PLATFORM, sizeof(plat), &plat, nullptr);

  const char *names[] = {
    "clQueryMLInterfaceVersionsQCOM", "clGetMLInterfaceV1QCOM",
    "clGetMLInterfaceV2QCOM",         "clGetMLInterfaceV3QCOM",
    "clGetMLInterfaceV4QCOM",         "clGetMLInterfaceV5QCOM",
    "clGetMLInterfaceV6QCOM",         "clCreateMLTensorQCOM",
    "clCreateMLOpQCOM",               "clEnqueueMLOpQCOM",
    "clReleaseMLTensorQCOM",          "clReleaseMLOpQCOM",
    "clCreateMLTensorMemoryQCOM",     "clGetMLTensorMemoryRequirementsQCOM",
  };
  std::fprintf(stderr, "[mlops] probing cl_qcom_ml_ops entrypoints (platform=%p):\n",
               (void *)plat);
  int found = 0;
  for (auto *nm : names) {
    void *p = clGetExtensionFunctionAddressForPlatform(plat, nm);
    std::fprintf(stderr, "[mlops]   %-38s : %s\n", nm, p ? "RESOLVED" : "null");
    if (p) found++;
  }
  std::fprintf(stderr,
               "[mlops] via-loader: %d/%d entrypoints resolved\n", found,
               (int)(sizeof(names) / sizeof(names[0])));

  // Cross-check against the VENDOR driver directly — our pushed libOpenCL.so is
  // an ICD/loader and may not forward clGetExtensionFunctionAddressForPlatform.
  // dlopen the vendor lib, pull its own clGetPlatformIDs + extension query, and
  // re-probe. This distinguishes "vendor genuinely lacks ml_ops" from "our
  // loader masks it".
  int vfound = -1;
  const char *vpaths[] = {"/vendor/lib64/libOpenCL.so",
                          "/system/vendor/lib64/libOpenCL.so",
                          "/system/lib64/libOpenCL.so"};
  for (auto *vp : vpaths) {
    void *h = dlopen(vp, RTLD_NOW | RTLD_LOCAL);
    if (!h) continue;
    auto getplat = (cl_int(*)(cl_uint, cl_platform_id *, cl_uint *))dlsym(
      h, "clGetPlatformIDs");
    auto getext = (void *(*)(cl_platform_id, const char *))dlsym(
      h, "clGetExtensionFunctionAddressForPlatform");
    std::fprintf(stderr, "[mlops] vendor %s: getPlatformIDs=%p getExt=%p\n", vp,
                 (void *)getplat, (void *)getext);
    if (!getext) { dlclose(h); continue; }
    cl_platform_id vplat = nullptr;
    cl_uint np = 0;
    if (getplat) getplat(1, &vplat, &np);
    vfound = 0;
    for (auto *nm : names) {
      void *p = getext(vplat, nm);
      if (p) {
        vfound++;
        std::fprintf(stderr, "[mlops]   vendor %-30s : RESOLVED\n", nm);
      }
    }
    std::fprintf(stderr, "[mlops] via-vendor(%s): %d/%d entrypoints resolved\n",
                 vp, vfound, (int)(sizeof(names) / sizeof(names[0])));

    // Gate #2 step 1: actually CALL clQueryMLInterfaceVersionsQCOM to learn the
    // supported CLML interface version(s). Signature (Qualcomm CLML SDK):
    //   cl_int clQueryMLInterfaceVersionsQCOM(cl_int *major, cl_int *minor,
    //                                         cl_uint *numVersions, void *resv);
    // Count-first (NULL,NULL,&n,NULL) then fill — safest probe of a live call.
    using QueryVerFn = cl_int (*)(cl_int *, cl_int *, cl_uint *, void *);
    auto qver = (QueryVerFn)getext(vplat, "clQueryMLInterfaceVersionsQCOM");
    if (qver) {
      cl_uint nv = 0;
      cl_int rc = qver(nullptr, nullptr, &nv, nullptr);
      std::fprintf(stderr, "[mlops] clQueryMLInterfaceVersionsQCOM count: rc=%d nv=%u\n",
                   rc, nv);
      if (rc == CL_SUCCESS && nv > 0 && nv < 64) {
        std::vector<cl_int> maj(nv, 0), min(nv, 0);
        rc = qver(maj.data(), min.data(), &nv, nullptr);
        std::fprintf(stderr, "[mlops] supported CLML versions (rc=%d):", rc);
        for (cl_uint i = 0; i < nv; i++)
          std::fprintf(stderr, " v%d.%d", maj[i], min[i]);
        std::fprintf(stderr, "\n");
      }
    } else {
      std::fprintf(stderr, "[mlops] clQueryMLInterfaceVersionsQCOM did not resolve\n");
    }
    dlclose(h);
    break;
  }

  const int total = found + (vfound > 0 ? vfound : 0);
  std::fprintf(stderr, "[mlops] => ml_ops %s\n",
               total ? "CALLABLE — proceed to single-GEMM spike"
                     : "NOT callable on this device (advertised but no entry "
                       "points, loader+vendor — dead end like recordable_queues)");
}

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
  // #63: NNTR_MODEL_GEMMA2=1 selects Gemma2-2B (the ML Drift paper's model) —
  // a plain transformer but with head_dim 256, sandwich norm, GeGLU, attn/final
  // soft-cap, embed*sqrt(H), no q/k-norm. Proves gpu_native is model-agnostic.
  const bool model_gemma2 = []() {
    const char *e = std::getenv("NNTR_MODEL_GEMMA2");
    return e && std::atoi(e) != 0;
  }();
  const char *model_name = "Qwen3-0.6B";
  if (model_gemma2) {
    cfg.hidden_size = 2304;
    cfg.intermediate_size = 9216;
    cfg.head_dim = 256;
    cfg.num_heads_Q = 8;
    cfg.num_heads_KV = 4;
    cfg.num_layers = 26;
    cfg.vocab_size = 256000;
    cfg.rope_theta = 1e4f;            // Gemma2 sliding/default rope
    cfg.is_gemma2 = true;
    cfg.embed_scale = std::sqrt((float)cfg.hidden_size); // ~48.0
    cfg.attn_logit_softcap = 50.0f;
    cfg.final_logit_softcap = 30.0f;
    cfg.sliding_window = 4096;        // no-op for prefill M<=1024
    model_name = "Gemma2-2B";
  } else if (model_4b) {
    cfg.hidden_size = 2560;
    cfg.intermediate_size = 9728;
    cfg.head_dim = 128;
    cfg.num_heads_Q = 32;
    cfg.num_heads_KV = 8;
    cfg.num_layers = 36;
    cfg.vocab_size = 151936;
    cfg.rope_theta = 1e6f;
    model_name = "Qwen3-4B";
  } else {
    cfg.hidden_size = 1024;
    cfg.intermediate_size = 3072;
    cfg.head_dim = 128;
    cfg.num_heads_Q = 16;
    cfg.num_heads_KV = 8;
    cfg.num_layers = 28;
    cfg.vocab_size = 151936;
    cfg.rope_theta = 1e6f;
  }
  cfg.max_seq_len = 20480;
  cfg.rms_norm_eps = 1e-6f;
  std::fprintf(stderr,
               "[main] model=%s hidden=%u L=%u hQ=%u hKV=%u hd=%u V=%u gemma2=%d\n",
               model_name, cfg.hidden_size, cfg.num_layers, cfg.num_heads_Q,
               cfg.num_heads_KV, cfg.head_dim, cfg.vocab_size,
               (int)cfg.is_gemma2);

  causallm_gpu::Qwen3Forward fwd;
  if (!fwd.init(cfg, weight_path)) {
    std::fprintf(stderr, "[main] init failed\n");
    return 2;
  }

  // #69 compute-peak microbench (no model needed): measure the actual
  // Adreno int8-dp4a + fp16 peak on THIS device, then exit.
  if (std::getenv("NNTR_PEAK_BENCH")) {
    run_peak_bench();
    return 0;
  }
  // Phase-0 gate #1 for the ml_ops matrix-engine GEMM (NNTR_MLOPS_PROBE=1).
  if (std::getenv("NNTR_MLOPS_PROBE")) {
    run_mlops_probe();
    return 0;
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

  // Gemma2 BOS = 2 (<bos>); Qwen3 uses 151643. Model-aware so the seed
  // token is in-distribution for the loaded model (#63).
  const unsigned int BOS_TOKEN = cfg.is_gemma2 ? 2u : 151643u;
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

    int next_token = fwd.run_lm_head_and_argmax(cur);
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

  // In-process v8c GEMM LWS sweep (NNTR_LWS_SWEEP=1): the model is already
  // loaded, so retune the GEMM local size and re-time M=1024 prefill for each
  // candidate WITHOUT reloading — ~10x faster than re-launching per LWS. Exits
  // after the sweep.
  if (std::getenv("NNTR_LWS_SWEEP")) {
    const int cand[][2] = {{4, 16}, {16, 4}, {8, 8},  {2, 32},
                           {32, 2}, {8, 16}, {16, 8}, {64, 1}};
    const unsigned int Ms = 1024;
    clEnqueueWriteBuffer(q, pf_in, CL_TRUE, 0, (size_t)Ms * H * sizeof(float),
                         rep_input.data(), 0, nullptr, nullptr);
    std::fprintf(stderr, "[main] === in-process LWS sweep (M=1024) ===\n");
    for (auto &c : cand) {
      nntrainer::set_v8c_lws_override(c[0], c[1]);
      double best = 1e30;
      for (int rep = 0; rep < 3; ++rep) {  // rep 0 = warmup (JIT)
        cl_mem a = pf_in, b = pf_out;
        auto t0 = NOW();
        bool ok = true;
        for (unsigned int L = 0; L < cfg.num_layers && ok; ++L) {
          ok = fwd.forward_one_layer_v2(L, a, b, 0, Ms);
          std::swap(a, b);
        }
        clFinish(q);
        double ms = MS(NOW(), t0);
        if (rep > 0 && ms < best) best = ms;
        if (!ok) { best = -1; break; }
      }
      if (best < 0)
        std::fprintf(stderr, "[lws-sweep] LWS=%d,%d : FAILED\n", c[0], c[1]);
      else
        std::fprintf(stderr,
                     "[lws-sweep] LWS=%d,%d : chain=%.1f ms => %.1f TPS\n",
                     c[0], c[1], best, (double)Ms * 1000.0 / best);
    }
    nntrainer::set_v8c_lws_override(0, 0);
    return 0;
  }

  // In-process weight-prefetch A/B (NNTR_PF_AB=1): one model load, re-time
  // M=1024 prefill for 1-ahead vs 2-ahead prefetch (no profiler clFinish), best
  // of 3, then exit. Clean single-run comparison via a plain `sh execute.sh`.
  if (std::getenv("NNTR_PF_AB")) {
    const unsigned int Ms = 1024;
    clEnqueueWriteBuffer(q, pf_in, CL_TRUE, 0, (size_t)Ms * H * sizeof(float),
                         rep_input.data(), 0, nullptr, nullptr);
    std::fprintf(stderr, "[main] === in-process prefetch A/B (M=1024) ===\n");
    const int modes[] = {1, 2};
    for (int pf : modes) {
      nntrainer::set_v8c_prefetch_override(pf);
      double best = 1e30;
      for (int rep = 0; rep < 3; ++rep) { // rep 0 = warmup (kernel JIT)
        cl_mem a = pf_in, b = pf_out;
        auto t0 = NOW();
        bool ok = true;
        for (unsigned int L = 0; L < cfg.num_layers && ok; ++L) {
          ok = fwd.forward_one_layer_v2(L, a, b, 0, Ms);
          std::swap(a, b);
        }
        clFinish(q);
        double ms = MS(NOW(), t0);
        if (rep > 0 && ms < best) best = ms;
        if (!ok) { best = -1; break; }
      }
      if (best < 0)
        std::fprintf(stderr, "[pf-ab] prefetch=%d-ahead : FAILED\n", pf);
      else
        std::fprintf(stderr,
                     "[pf-ab] prefetch=%d-ahead : chain=%.1f ms => %.1f TPS\n",
                     pf, best, (double)Ms * 1000.0 / best);
    }
    nntrainer::set_v8c_prefetch_override(-1);
    return 0;
  }

  // In-process register-tile A/B (NNTR_TILE_AB=1): one model load, re-time
  // M=1024 prefill for the 4x8 default vs 8x4 (2x weight reuse) and a few other
  // 32-accumulator tiles, then exit. M=1024 is divisible by all TM tried.
  if (std::getenv("NNTR_TILE_AB")) {
    const unsigned int Ms = 1024;
    clEnqueueWriteBuffer(q, pf_in, CL_TRUE, 0, (size_t)Ms * H * sizeof(float),
                         rep_input.data(), 0, nullptr, nullptr);
    std::fprintf(stderr, "[main] === in-process tile A/B (M=1024) ===\n");
    const int tiles[][2] = {{4, 8}, {8, 4}, {2, 16}, {4, 4}}; // <=32 acc (8x8 spills->hang)
    for (auto &t : tiles) {
      if (Ms % (unsigned)t[0] != 0) continue;
      nntrainer::set_v8c_tile_override(t[0], t[1]);
      double best = 1e30;
      for (int rep = 0; rep < 3; ++rep) { // rep 0 = warmup (kernel JIT)
        cl_mem a = pf_in, b = pf_out;
        auto t0 = NOW();
        bool ok = true;
        for (unsigned int L = 0; L < cfg.num_layers && ok; ++L) {
          ok = fwd.forward_one_layer_v2(L, a, b, 0, Ms);
          std::swap(a, b);
        }
        clFinish(q);
        double ms = MS(NOW(), t0);
        if (rep > 0 && ms < best) best = ms;
        if (!ok) { best = -1; break; }
      }
      if (best < 0)
        std::fprintf(stderr, "[tile-ab] TMxTN=%dx%d : FAILED\n", t[0], t[1]);
      else
        std::fprintf(stderr,
                     "[tile-ab] TMxTN=%dx%d : chain=%.1f ms => %.1f TPS\n",
                     t[0], t[1], best, (double)Ms * 1000.0 / best);
    }
    nntrainer::set_v8c_tile_override(0, 0);
    return 0;
  }

  // In-process GEMM-compute attribution (NNTR_GEMM_ATTRIB=1): clean M=1024
  // forward wall for (a) full, (b) dp4a removed from every v8c GEMM. The diff
  // is the share of prefill the GEMM *compute* actually costs — tells us
  // whether a faster GEMM kernel can move the needle, or the gap is elsewhere.
  if (std::getenv("NNTR_GEMM_ATTRIB")) {
    const unsigned int Ms = 1024;
    clEnqueueWriteBuffer(q, pf_in, CL_TRUE, 0, (size_t)Ms * H * sizeof(float),
                         rep_input.data(), 0, nullptr, nullptr);
    std::fprintf(stderr, "[main] === GEMM compute attribution (M=1024) ===\n");
    // Bitmask modes decompose the prefill wall into orthogonal layers:
    //   0=full, 1=no-dp4a, 3=no-dp4a+no-wfetch, 7=no-dp4a+no-wfetch+no-afetch.
    const int modes[] = {0, 1, 3, 7};
    const char *labels[] = {"full", "no-dp4a", "no-dp4a,no-wfetch",
                            "no-dp4a,no-w/a-fetch (floor)"};
    double walls[4] = {0, 0, 0, 0};
    for (int mi = 0; mi < 4; ++mi) {
      nntrainer::set_v8c_nocompute_override(modes[mi]);
      double best = 1e30;
      for (int rep = 0; rep < 3; ++rep) {
        cl_mem a = pf_in, b = pf_out;
        auto t0 = NOW();
        bool ok = true;
        for (unsigned int L = 0; L < cfg.num_layers && ok; ++L) {
          ok = fwd.forward_one_layer_v2(L, a, b, 0, Ms);
          std::swap(a, b);
        }
        clFinish(q);
        double ms = MS(NOW(), t0);
        if (rep > 0 && ms < best) best = ms;
      }
      walls[mi] = best;
      std::fprintf(stderr, "[gemm-attrib] %-28s chain=%.1f ms => %.1f TPS\n",
                   labels[mi], best, (double)Ms * 1000.0 / best);
    }
    // Orthogonal cost decomposition (each = wall delta when that layer drops).
    // floor = pipeline/dispatch + attention + layernorm/quant/elementwise.
    std::fprintf(stderr, "[gemm-attrib] --- decomposition (of %.1f ms full) ---\n",
                 walls[0]);
    std::fprintf(stderr, "[gemm-attrib]   dp4a compute     = %6.1f ms (%4.1f%%)\n",
                 walls[0] - walls[1], 100.0 * (walls[0] - walls[1]) / walls[0]);
    std::fprintf(stderr, "[gemm-attrib]   v8c weight fetch = %6.1f ms (%4.1f%%)\n",
                 walls[1] - walls[2], 100.0 * (walls[1] - walls[2]) / walls[0]);
    std::fprintf(stderr, "[gemm-attrib]   v8c act fetch    = %6.1f ms (%4.1f%%)\n",
                 walls[2] - walls[3], 100.0 * (walls[2] - walls[3]) / walls[0]);
    std::fprintf(stderr, "[gemm-attrib]   floor (rest)     = %6.1f ms (%4.1f%%)\n",
                 walls[3], 100.0 * walls[3] / walls[0]);
    nntrainer::set_v8c_nocompute_override(0);

    // One profiled pass (per-stage clFinish) to break the floor down into the
    // actual ops, so we know WHICH non-GEMM kernels to fuse. Absolute ms here
    // are inflated by the per-stage drain, but the relative split is clean.
    fwd.timings_.reset();
    fwd.profile_stages_ = true;
    {
      cl_mem a = pf_in, b = pf_out;
      for (unsigned int L = 0; L < cfg.num_layers; ++L) {
        if (!fwd.forward_one_layer_v2(L, a, b, 0, Ms)) break;
        std::swap(a, b);
      }
      clFinish(q);
    }
    fwd.profile_stages_ = false;
    const auto &tt = fwd.timings_;
    const double psum = tt.pad_attn_norm_ms + tt.qkv_quant_image_ms +
                        tt.qkv_gemm_ms + tt.qk_norm_rope_ms + tt.kv_write_ms +
                        tt.attn_dispatch_ms + tt.wo_ms + tt.ffn_ms;
    auto ppct = [&](double v) { return psum > 0 ? 100.0 * v / psum : 0.0; };
    std::fprintf(stderr,
                 "[gemm-attrib] --- per-stage (profiled, %d calls, %.0f ms) ---\n"
                 "[gemm-attrib]   (a) pad+attn_norm   %7.1f ms (%4.1f%%)\n"
                 "[gemm-attrib]   (b) qkv quant+img   %7.1f ms (%4.1f%%)\n"
                 "[gemm-attrib]   (c) Q/K/V GEMM      %7.1f ms (%4.1f%%)\n"
                 "[gemm-attrib]   (d) qk_norm[+RoPE]  %7.1f ms (%4.1f%%)\n"
                 "[gemm-attrib]   (e) KV write SVM    %7.1f ms (%4.1f%%)\n"
                 "[gemm-attrib]   (f) attention       %7.1f ms (%4.1f%%)\n"
                 "[gemm-attrib]   (g) wo + resid_1    %7.1f ms (%4.1f%%)\n"
                 "[gemm-attrib]   (h) ffn block       %7.1f ms (%4.1f%%)\n"
                 "[gemm-attrib]       (h1) ffn norm+quant %7.1f ms (%4.1f%%)\n"
                 "[gemm-attrib]       (h2) gate+up GEMM   %7.1f ms (%4.1f%%)\n"
                 "[gemm-attrib]       (h3) cvt+glu+quant  %7.1f ms (%4.1f%%)\n"
                 "[gemm-attrib]       (h4) ffn_down GEMM  %7.1f ms (%4.1f%%)\n"
                 "[gemm-attrib]       (h5) post_norm+add  %7.1f ms (%4.1f%%)\n",
                 tt.calls, psum,
                 tt.pad_attn_norm_ms, ppct(tt.pad_attn_norm_ms),
                 tt.qkv_quant_image_ms, ppct(tt.qkv_quant_image_ms),
                 tt.qkv_gemm_ms, ppct(tt.qkv_gemm_ms),
                 tt.qk_norm_rope_ms, ppct(tt.qk_norm_rope_ms),
                 tt.kv_write_ms, ppct(tt.kv_write_ms),
                 tt.attn_dispatch_ms, ppct(tt.attn_dispatch_ms),
                 tt.wo_ms, ppct(tt.wo_ms),
                 tt.ffn_ms, ppct(tt.ffn_ms),
                 tt.ffn_norm_ms, ppct(tt.ffn_norm_ms),
                 tt.ffn_gateup_ms, ppct(tt.ffn_gateup_ms),
                 tt.ffn_swiglu_ms, ppct(tt.ffn_swiglu_ms),
                 tt.ffn_down_ms, ppct(tt.ffn_down_ms),
                 tt.ffn_post_ms, ppct(tt.ffn_post_ms));
    return 0;
  }

  for (int M_test : PREFILL_MS) {
    clEnqueueWriteBuffer(q, pf_in, CL_TRUE, 0,
                         (size_t)M_test * H * sizeof(float),
                         rep_input.data(), 0, nullptr, nullptr);
    // Enable per-stage profiling at M=256 (peak) and M=1024 (cliff)
    // so we can see WHERE the cliff time goes. Profiling adds a
    // clFinish per stage = overhead; total ms reported INCLUDES that
    // overhead but per-stage attribution is clean.
    // per-stage timing brackets each stage with clFinish(cl_q_), which DRAINS
    // the in-order queue and destroys host/GPU dispatch overlap — i.e. it
    // inflates the very M=256/M=1024 wall it is trying to measure (M=512 never
    // set it, which is why M=512 looked faster per-token). Gate it OFF by
    // default; NNTR_STAGE_PROFILE=1 restores per-stage attribution.
    static const bool want_stage_prof = []() {
      const char *e = std::getenv("NNTR_STAGE_PROFILE");
      return e && std::atoi(e) != 0;
    }();
    const bool profile = want_stage_prof && (M_test == 256 || M_test == 1024);
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
                   "    (h) ffn block      %7.1f ms (%4.1f%%)\n"
                   "        (h1) ffn norm+quant %7.1f ms (%4.1f%%)\n"
                   "        (h2) gate+up GEMM   %7.1f ms (%4.1f%%)\n"
                   "        (h3) cvt+glu+quant  %7.1f ms (%4.1f%%)\n"
                   "        (h4) ffn_down GEMM  %7.1f ms (%4.1f%%)\n"
                   "        (h5) post_norm+add  %7.1f ms (%4.1f%%)\n",
                   M_test, tt.calls, sum,
                   tt.pad_attn_norm_ms,   pct(tt.pad_attn_norm_ms),
                   tt.qkv_quant_image_ms, pct(tt.qkv_quant_image_ms),
                   tt.qkv_gemm_ms,        pct(tt.qkv_gemm_ms),
                   tt.qk_norm_rope_ms,    pct(tt.qk_norm_rope_ms),
                   tt.kv_write_ms,        pct(tt.kv_write_ms),
                   tt.attn_dispatch_ms,   pct(tt.attn_dispatch_ms),
                   tt.wo_ms,              pct(tt.wo_ms),
                   tt.ffn_ms,             pct(tt.ffn_ms),
                   tt.ffn_norm_ms,        pct(tt.ffn_norm_ms),
                   tt.ffn_gateup_ms,      pct(tt.ffn_gateup_ms),
                   tt.ffn_swiglu_ms,      pct(tt.ffn_swiglu_ms),
                   tt.ffn_down_ms,        pct(tt.ffn_down_ms),
                   tt.ffn_post_ms,        pct(tt.ffn_post_ms));
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
      return fwd.run_lm_head_and_argmax(last_row);
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
    // Stable single-line form for golden-reference capture/compare (execute.sh
    // greps this). nan=0 marks an invalid generation so the harness can flag it.
    std::fprintf(stderr, "[gen-tokens] nonan=%d", ok);
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
