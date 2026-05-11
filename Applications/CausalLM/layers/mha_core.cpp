// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   mha_core.cpp
 * @date   11 July 2025
 * @see    https://github.com/nntrainer/nntrainer
 *         https://arxiv.org/abs/1706.03762
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This code is based on custom_multi_head_attention_layer.cpp.
 *         This code is a part of the break down version of the mha layer.
 */
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <mutex>
#include <omp.h>
#include <thread>
#include <vector>

static std::mutex rope_init_mtx;

#include <engine.h>
#include <fp16.h>
#include <layer_context.h>
#include <mha_core.h>
#include <nntrainer_error.h>
#include <node_exporter.h>
#include <profile_gate.h>

#if defined(ENABLE_OPENCL) && ENABLE_OPENCL == 1
#include <blas_kernels.h>
#include <cl_context.h>
#endif

namespace {

struct MHACoreProfile {
  std::atomic<uint64_t> calls{0};
  std::atomic<uint64_t> ns{0};
  // Stage 0 sub-timers for one_batch_incremental_forwarding's six
  // NEON-CPU stages (FP16 path; the FP32 / sliding-window variants are
  // not measured because they are not on the Qwen3-4B prefill path).
  std::atomic<uint64_t> ns_rope_q{0};   // apply_rotary_emb_tensor_v2(query)
  std::atomic<uint64_t> ns_rope_k{0};   // apply_rotary_emb_tensor_v2(key->cache)
  std::atomic<uint64_t> ns_v_copy{0};   // b_cache_value_step.copyData(value)
  std::atomic<uint64_t> ns_qk{0};       // compute_kcaches (Q dot K^T)
  std::atomic<uint64_t> ns_softmax{0};  // softmax_triangle (masked softmax)
  std::atomic<uint64_t> ns_av{0};       // compute_fp16vcache_transposed (A * V)

  ~MHACoreProfile() {
    const uint64_t c = calls.load();
    if (c == 0)
      return;
    if (nntrainer::prefill_profile_suppressed())
      return;
    const uint64_t t = ns.load();
    const double T = t / 1.0e6;
    auto pct = [&](uint64_t v) {
      return t == 0 ? 0.0 : (v / 1.0e6) / T * 100.0;
    };
    std::fprintf(stderr,
                 "[PROFILE MHACoreLayer prefill (M>1)] total=%.2f ms "
                 "calls=%llu avg=%.3f ms\n",
                 T, (unsigned long long)c, T / static_cast<double>(c));
    std::fprintf(stderr,
                 "  rope_q   : %8.2f ms (%5.1f%%)  [apply_rotary_emb_v2 query]\n",
                 ns_rope_q / 1.0e6, pct(ns_rope_q));
    std::fprintf(stderr,
                 "  rope_k   : %8.2f ms (%5.1f%%)  [apply_rotary_emb_v2 key->cache]\n",
                 ns_rope_k / 1.0e6, pct(ns_rope_k));
    std::fprintf(stderr,
                 "  v_copy   : %8.2f ms (%5.1f%%)  [value -> cache copyData]\n",
                 ns_v_copy / 1.0e6, pct(ns_v_copy));
    std::fprintf(stderr,
                 "  qk       : %8.2f ms (%5.1f%%)  [compute_kcaches Q dot K^T]\n",
                 ns_qk / 1.0e6, pct(ns_qk));
    std::fprintf(stderr,
                 "  softmax  : %8.2f ms (%5.1f%%)  [softmax_triangle]\n",
                 ns_softmax / 1.0e6, pct(ns_softmax));
    std::fprintf(stderr,
                 "  av       : %8.2f ms (%5.1f%%)  [compute_fp16vcache_transposed A*V]\n",
                 ns_av / 1.0e6, pct(ns_av));
  }
};

MHACoreProfile g_mha_core_profile;

// Decode-path (M == 1) version of the profiler.  Parallel to
// g_mha_core_profile, but fires for M==1 calls so we can see per-
// token attention costs during generation.  The "rope_q", "rope_k",
// "v_copy", "qk+softmax+av" (or fused) stages each attribute to a
// separate bucket.  Totals divided by calls give avg per-token MHA.
struct MHACoreDecodeProfile {
  std::atomic<uint64_t> calls{0};
  std::atomic<uint64_t> ns{0};
  std::atomic<uint64_t> ns_rope_q{0};
  std::atomic<uint64_t> ns_rope_k{0};
  std::atomic<uint64_t> ns_v_copy{0};
  std::atomic<uint64_t> ns_qk{0};
  std::atomic<uint64_t> ns_softmax{0};
  std::atomic<uint64_t> ns_av{0};
  // Detailed buckets added to identify where the drain stall lives.
  std::atomic<uint64_t> ns_entry_drain{0};      // 4x SVMMap on q/k/v/output
  // Per-SVMMap split: tells us if first one carries the queue wait
  // and the rest are pure per-region cache flush, vs. all four
  // equally expensive.
  std::atomic<uint64_t> ns_drain_q{0};
  std::atomic<uint64_t> ns_drain_k{0};
  std::atomic<uint64_t> ns_drain_v{0};
  std::atomic<uint64_t> ns_drain_o{0};
  std::atomic<uint64_t> ns_attn_fused_call{0};  // attention_fused_fp16_cl wall
  std::atomic<uint64_t> ns_exit_unmap_publish{0}; // SVMUnmap(output) + publish
  std::atomic<uint64_t> ns_cpu{0};              // thread CPU time across calls

  ~MHACoreDecodeProfile() {
    const uint64_t c = calls.load();
    if (c == 0)
      return;
    const uint64_t t = ns.load();
    const double T = t / 1.0e6;
    auto pct = [&](uint64_t v) {
      return t == 0 ? 0.0 : (v / 1.0e6) / T * 100.0;
    };
    std::fprintf(stderr,
                 "[PROFILE MHACoreLayer decode (M==1)] total=%.2f ms "
                 "calls=%llu avg=%.3f ms\n",
                 T, (unsigned long long)c, T / static_cast<double>(c));
    std::fprintf(stderr,
                 "  entry_drain    : %8.2f ms (%5.1f%%)  "
                 "[4x SVMMap blocking q/k/v/output]\n",
                 ns_entry_drain / 1.0e6, pct(ns_entry_drain));
    std::fprintf(stderr,
                 "    drain_q  : %8.2f ms (%5.1f%%)\n",
                 ns_drain_q / 1.0e6, pct(ns_drain_q));
    std::fprintf(stderr,
                 "    drain_k  : %8.2f ms (%5.1f%%)\n",
                 ns_drain_k / 1.0e6, pct(ns_drain_k));
    std::fprintf(stderr,
                 "    drain_v  : %8.2f ms (%5.1f%%)\n",
                 ns_drain_v / 1.0e6, pct(ns_drain_v));
    std::fprintf(stderr,
                 "    drain_o  : %8.2f ms (%5.1f%%)\n",
                 ns_drain_o / 1.0e6, pct(ns_drain_o));
    std::fprintf(stderr,
                 "  attn_fused     : %8.2f ms (%5.1f%%)  "
                 "[attention_fused_fp16_cl wall (incl. exit drain when "
                 "ATTN_NO_DRAIN unset)]\n",
                 ns_attn_fused_call / 1.0e6, pct(ns_attn_fused_call));
    std::fprintf(stderr,
                 "  exit           : %8.2f ms (%5.1f%%)  "
                 "[SVMUnmap(output) + image2d publish]\n",
                 ns_exit_unmap_publish / 1.0e6,
                 pct(ns_exit_unmap_publish));
    std::fprintf(stderr,
                 "  rope_q   : %8.2f ms (%5.1f%%)\n",
                 ns_rope_q / 1.0e6, pct(ns_rope_q));
    std::fprintf(stderr,
                 "  rope_k   : %8.2f ms (%5.1f%%)\n",
                 ns_rope_k / 1.0e6, pct(ns_rope_k));
    std::fprintf(stderr,
                 "  v_copy   : %8.2f ms (%5.1f%%)\n",
                 ns_v_copy / 1.0e6, pct(ns_v_copy));
    std::fprintf(stderr,
                 "  qk       : %8.2f ms (%5.1f%%)\n",
                 ns_qk / 1.0e6, pct(ns_qk));
    std::fprintf(stderr,
                 "  softmax  : %8.2f ms (%5.1f%%)\n",
                 ns_softmax / 1.0e6, pct(ns_softmax));
    std::fprintf(stderr,
                 "  av       : %8.2f ms (%5.1f%%)\n",
                 ns_av / 1.0e6, pct(ns_av));
    const uint64_t cpu = ns_cpu.load();
    const double C = cpu / 1.0e6;
    const double IDLE = T - C;
    std::fprintf(stderr,
                 "  cpu_busy : %8.2f ms (%5.1f%%)  [thread CPU time]\n"
                 "  idle_wait: %8.2f ms (%5.1f%%)  [wall - cpu, host "
                 "blocked on GPU queue drains]\n",
                 C, t == 0 ? 0.0 : C / T * 100.0,
                 IDLE, t == 0 ? 0.0 : IDLE / T * 100.0);
  }
};

MHACoreDecodeProfile g_mha_core_decode_profile;

inline uint64_t mha_now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
           std::chrono::steady_clock::now().time_since_epoch())
    .count();
}

inline uint64_t mha_now_cpu_ns() {
  struct timespec ts;
  clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

} // namespace

#include <cstdint>

inline float convert_scalar(uint16_t h) {
  return nntrainer::compute_fp16_to_fp32(h);
}

namespace causallm {

#define tile_size 4

/************************************************************** */

/**
 * @brief constructor of MHACoreLayer
 */
MHACoreLayer::MHACoreLayer() :
  mha_core_props(
    nntrainer::props::NumHeads(), props::NumHeads_KV(),
    nntrainer::props::ProjectedKeyDim(), nntrainer::props::ProjectedValueDim(),
    nntrainer::props::OutputShape(), nntrainer::props::DropOutRate(),
    nntrainer::props::ReturnAttentionWeight(),
    nntrainer::props::AverageAttentionWeight(), nntrainer::props::MaxTimestep(),
    props::SlidingWindow(), props::MaxNewTokens(), props::RopeTheta(),
    props::MaxPositionEmbeddings(), props::UseSink(), props::RopeScalingType(),
    props::RopeScalingFactor(), props::RopeScalingMaxPositionEmbeddings(),
    props::AttnLogitSoftcapping(), props::IsCausal()),
  sm(nntrainer::ActivationType::ACT_SOFTMAX),
  epsilon(1e-3),
  cache_index(0),
  num_heads_Q(0),
  num_heads_KV(0),
  head_dim(0),
  cache_shift(false) {
  tensor_idx.fill(std::numeric_limits<unsigned>::max());
}

MHACoreLayer::~MHACoreLayer() {}

/************************************************************** */

void MHACoreLayer::finalize(nntrainer::InitLayerContext &context) {

  NNTR_THROW_IF(context.getNumInputs() < 3 || context.getNumInputs() > 4,
                std::invalid_argument)
    << "Multi head Attention layer needs 3 or 4 inputs. (query, key, value and "
       "mask is optional)";
  ml::train::TensorDim::TensorType activation_type = {
    context.getFormat(), context.getActivationDataType()};
  ml::train::TensorDim empty_dim(activation_type);

  const std::vector<ml::train::TensorDim> &input_dims =
    context.getInputDimensions();
  const ml::train::TensorDim &query_dim = input_dims[INOUT_INDEX::QUERY];
  const ml::train::TensorDim &key_dim = input_dims[INOUT_INDEX::KEY];

  /** max time step of this model */
  const unsigned int max_timestep =
    std::get<nntrainer::props::MaxTimestep>(mha_core_props).get();

  /** max position embeddings */
  max_position_embeddings =
    std::get<props::MaxPositionEmbeddings>(mha_core_props).get();

  /** local window size */
  local_window_size = std::get<props::SlidingWindow>(mha_core_props).get();

  /** attention scaling computation */
  rope_scaling_type = std::get<props::RopeScalingType>(mha_core_props).get();
  scale = std::get<props::RopeScalingFactor>(mha_core_props).get();
  if (rope_scaling_type == "yarn")
    original_max_position_embeddings =
      std::get<props::RopeScalingMaxPositionEmbeddings>(mha_core_props).get();

  /** query_dim = (B, 1, seq_len, H_Q * Head_Dim ) */
  const unsigned int batch_size = query_dim.batch();
  const unsigned int query_width = query_dim.width();
  /** key_dim = (B, 1, max_seq_len, H_KV * Head_Dim ) */
  const unsigned int key_width = key_dim.width();

  /**
   *  @note If NumHeads_KV is set, then use the value. Otherwise,
   *        we initialize num_heads_KV with num_heads_Q.
   */
  num_heads_Q = static_cast<size_t>(
    std::get<nntrainer::props::NumHeads>(mha_core_props).get());
  num_heads_KV =
    std::get<props::NumHeads_KV>(mha_core_props).empty()
      ? num_heads_Q
      : static_cast<size_t>(std::get<props::NumHeads_KV>(mha_core_props).get());

  // head_dim
  head_dim = static_cast<size_t>(query_width) / num_heads_Q;
  NNTR_THROW_IF(head_dim != key_width / num_heads_KV, std::invalid_argument)
    << "num_heads_Q and num_heads_KV are not properly given. Please check the "
       "num_heads_* are set correctly so that the `head_dim`s are all same for "
       "query / key / value";

  /** Weight for Sink */
  use_sink = std::get<props::UseSink>(mha_core_props).get();
  if (use_sink) {
#if ENABLE_FP16 && defined(__ANDROID__)
    nntrainer::TensorDim sink_dim(
      1, 1, 1, num_heads_Q,
      nntrainer::TensorDim::TensorType(context.getFormat(),
                                       ml::train::TensorDim::DataType::FP16));
#else
    nntrainer::TensorDim sink_dim(
      1, 1, 1, num_heads_Q,
      nntrainer::TensorDim::TensorType(context.getFormat(),
                                       context.getActivationDataType()));
#endif
    sink_idx = context.requestWeight(sink_dim, nntrainer::Initializer::ZEROS,
                                     nntrainer::WeightRegularizer::NONE, 0.0f,
                                     0.0f, "sink");
  }

  attn_logit_softcapping =
    std::get<props::AttnLogitSoftcapping>(mha_core_props).get();

  /** Is Causal */
  is_causal = std::get<props::IsCausal>(mha_core_props).get();

  /** Tensor for KV-Cache */
#ifdef ENABLE_FP16
  ml::train::TensorDim cache_key_dim(
    {batch_size, 1, max_timestep, num_heads_KV * head_dim},
    {context.getFormat(), ml::train::TensorDim::DataType::FP16});
  ml::train::TensorDim cache_value_dim(
    {batch_size, 1, max_timestep, num_heads_KV * head_dim},
    {context.getFormat(), ml::train::TensorDim::DataType::FP16});
#else
  ml::train::TensorDim cache_key_dim(
    {batch_size, 1, max_timestep, num_heads_KV * head_dim},
    {context.getFormat(), ml::train::TensorDim::DataType::UINT16});
  ml::train::TensorDim cache_value_dim(
    {batch_size, 1, max_timestep, num_heads_KV * head_dim},
    {context.getFormat(), ml::train::TensorDim::DataType::UINT16});
#endif

  tensor_idx[AttentionParams::cache_key] = context.requestTensor(
    cache_key_dim, "cache_key", nntrainer::Initializer::NONE, false,
    nntrainer::TensorLifespan::MAX_LIFESPAN);
  tensor_idx[AttentionParams::cache_value] = context.requestTensor(
    cache_value_dim, "cache_value", nntrainer::Initializer::NONE, false,
    nntrainer::TensorLifespan::MAX_LIFESPAN);

  theta = (float)std::get<props::RopeTheta>(mha_core_props).get();

  /** set Output dimension! - one output */
  std::vector<nntrainer::TensorDim> output_dims(1);
  output_dims[0] = input_dims[0];
  output_dims[0].width(head_dim * num_heads_Q);
  output_dims[0].setTensorType(
    {context.getFormat(), context.getActivationDataType()});
  context.setOutputDimensions(output_dims);
}

/************************************************************** */

/**
 * @note This forwarding function is used for training mode.
 *       This will be implemented ASAP.
 * @date 2024-09-02
 */
void MHACoreLayer::forwarding(nntrainer::RunLayerContext &context,
                              bool training) {}

/**
 * @note This incremental_forwarding method is invoked for inference mode.
 *       Please note that Transformer Decoder's MHA takes only one sequence at a
 * step. Incremental forwarding function is used for this.
 */
void MHACoreLayer::incremental_forwarding(nntrainer::RunLayerContext &context,
                                          unsigned int _from, unsigned int _to,
                                          bool training) {
  const bool profile_this_call = (_to - _from) > 1;
  const bool profile_this_decode = (_to - _from) == 1;
  const uint64_t t_layer_start =
    (profile_this_call || profile_this_decode) ? mha_now_ns() : 0;
  const uint64_t t_cpu_start =
    profile_this_decode ? mha_now_cpu_ns() : 0;

  /// @todo replace step_size into input height
  unsigned int step_size = _to - _from;

  unsigned int max_timestep =
    std::get<nntrainer::props::MaxTimestep>(mha_core_props).get();

  unsigned int from = _from;
  unsigned int to = _to;

  if (to >= max_timestep) {
    // initial forwarding
    if (!_from) {
      throw std::invalid_argument(
        "to shouldn't greater than max_timestep for initial forwarding");
    } else {
      throw std::runtime_error("NYI: cache shift is not available");
      // exceeds the kv_cache size
      // KV_cache is shifted!
      cache_shift = true;
      from = max_timestep - 1;
      to = max_timestep;
    }
  }

  // util fn to compute tensor dimension for one step.
  auto get_step_dim = [step_size](const ml::train::TensorDim &dim) {
    auto step_dim = dim;
    step_dim.batch(1);
    step_dim.height(step_size);
    return step_dim;
  };

  /** incremental forwarding for each batch */
  nntrainer::Tensor &query =
    context.getInput(INOUT_INDEX::QUERY); // projected query
  nntrainer::Tensor &key = context.getInput(INOUT_INDEX::KEY); // projected key
  nntrainer::Tensor &value =
    context.getInput(INOUT_INDEX::VALUE); // projected value
  nntrainer::Tensor &output =
    context.getOutput(INOUT_INDEX::OUTPUT); // output to be projected

#if defined(ENABLE_OPENCL) && ENABLE_OPENCL == 1
  // MHACore reads q/k/v via getData<>() (CPU NEON loops for RoPE,
  // softmax, KV-compute) and writes `output` via CPU too. Downstream
  // o_proj is a GPU gemm that consumes this output as an SVM kernel
  // arg. Upstream q_proj/k_proj/v_proj enqueued only a non-blocking
  // SVMMap, so their image2d_to_svm writes may still be in flight.
  //
  // Handshake: map(READ) q/k/v and map(WRITE) output with blocking=true
  // here, drains the queue AND announces the CPU access window. The
  // matching SVMUnmap(output) at the bottom of incremental_forwarding
  // commits the CPU writes back so the o_proj kernel sees them.
  //
  // Phase B.11 (REVERTED): when NNTRAINER_ATTN_GPU=1 routes the whole
  // attention through attention_fused_fp16_cl on the OpenCL queue,
  // q/k/v are read by the GPU kernel (not the host) and output is
  // written by the GPU kernel. The CPU-side handshake here looked
  // redundant for the GPU path and the [PROFILE NetworkGraph] mha_core
  // 2607 ms / 1152 calls (2.26 ms/call) was almost entirely those 4
  // blocking SVMMaps absorbing upstream stall.
  //
  // Empirical result (B.11): skipping the entry drains broke decode
  // output:
  //   "ting editionêu웜_RESETxl inconsist  ˻ഫÏ多功能☽è一刀..."
  // Same-queue dispatch ordering does NOT imply same-queue SVM cache
  // coherence on Adreno coarse-grained SVM. The drains were acting as
  // explicit flush triggers between successive GPU kernels (upstream
  // Q/K/V proj + reshaped_rms_norm SVM writes -> attention_fused
  // reads). Without them attention sees stale data.
  //
  // NNTRAINER_MHA_NO_ENTRY_DRAIN=1 keeps the (broken on this driver)
  // skip code reachable for future testing on a different driver
  // version, but it's off by default. ATTN_NO_DRAIN (exit drain skip
  // inside attention_fused_fp16_cl) stays separate and remains safe
  // because the next consumer (o_proj per-FC) reads via SVM kernel
  // arg + has its own consumer-side coherence handling.
  static const bool s_mha_no_entry_drain =
    std::getenv("NNTRAINER_MHA_NO_ENTRY_DRAIN") != nullptr;
  // The skip is only safe when downstream RoPE consumers also run
  // on GPU. Decode (M==1) flips to GPU RoPE under NNTRAINER_RoPE_GPU=1,
  // but PREFILL (M>1) still uses CPU RoPE -- if we skipped the drain
  // there, CPU RoPE would read stale Q/K and corrupt the kv-cache,
  // poisoning every subsequent decode step. Limit the skip to
  // (decode && rope-gpu).
  static const bool s_rope_gpu_for_drain_skip =
    std::getenv("NNTRAINER_RoPE_GPU") != nullptr;
  const bool skip_entry_drain =
    s_mha_no_entry_drain && profile_this_decode &&
    s_rope_gpu_for_drain_skip;
  // Phase B.13: skip ONLY the output (CPU-write) entry SVMMap when
  // attention runs on GPU. q/k/v entry maps are kept because they
  // act as Adreno coarse-SVM cache-flush triggers for upstream Q/K/V
  // projection writes -- B.11 confirmed that skipping them races.
  // The output map is different: it announces a CPU write window,
  // but in the GPU path attention_fused_fp16_cl writes output via
  // the GPU kernel and never via the host. So the announcement is
  // pure overhead. Only saving 1 of 4 drains, but each map carries
  // a per-region cache-flush cost on Adreno on top of the queue
  // drain.
  static const bool s_mha_no_output_entry_drain =
    std::getenv("NNTRAINER_MHA_NO_OUTPUT_ENTRY_DRAIN") != nullptr;
  static const bool s_attn_gpu_for_output_drain =
    std::getenv("NNTRAINER_ATTN_GPU") != nullptr;
  const bool skip_output_entry_drain =
    s_mha_no_output_entry_drain && s_attn_gpu_for_output_drain;
  auto *mha_sync_cl_ctx = static_cast<nntrainer::ClContext *>(
    nntrainer::Engine::Global().getRegisteredContext("gpu"));
  if (mha_sync_cl_ctx && !skip_entry_drain) {
    auto map_if_svm = [&](nntrainer::Tensor &t, bool ro) {
      if (t.getMemoryData() && t.getMemoryData()->isSVM()) {
        mha_sync_cl_ctx->command_queue_inst_.enqueueSVMMap(
          t.getData<char>(), t.bytes(), /*read_only=*/ro);
      }
    };
    const uint64_t t_drain0 =
      profile_this_decode ? mha_now_ns() : 0;
    // Phase profile-driven: drain_q carries 94% of entry_drain wall
    // (queue wait for ~10 upstream dispatches per layer). drain_k/v/o
    // are pure region cache flushes (~35-50us each). Try replacing
    // the three secondary maps with a single bounding-range map when
    // the four tensors lie within a reasonable contiguous span.
    // Saves 2 of the 3 region-flush costs (~50us * 2 / call).
    static const bool s_unified_drain =
      std::getenv("NNTRAINER_MHA_UNIFIED_ENTRY_DRAIN") != nullptr;
    if (s_unified_drain &&
        query.getMemoryData() && query.getMemoryData()->isSVM() &&
        key.getMemoryData() && key.getMemoryData()->isSVM() &&
        value.getMemoryData() && value.getMemoryData()->isSVM() &&
        (skip_output_entry_drain ||
         (output.getMemoryData() && output.getMemoryData()->isSVM()))) {
      const char *q_p = query.getData<char>();
      const char *k_p = key.getData<char>();
      const char *v_p = value.getData<char>();
      const char *o_p = skip_output_entry_drain ? q_p
                                                 : output.getData<char>();
      const size_t q_b = query.bytes();
      const size_t k_b = key.bytes();
      const size_t v_b = value.bytes();
      const size_t o_b = skip_output_entry_drain ? 0 : output.bytes();
      const char *min_p = q_p;
      if (k_p < min_p) min_p = k_p;
      if (v_p < min_p) min_p = v_p;
      if (!skip_output_entry_drain && o_p < min_p) min_p = o_p;
      const char *max_e = q_p + q_b;
      if (k_p + k_b > max_e) max_e = k_p + k_b;
      if (v_p + v_b > max_e) max_e = v_p + v_b;
      if (!skip_output_entry_drain && o_p + o_b > max_e)
        max_e = o_p + o_b;
      const size_t span = (size_t)(max_e - min_p);
      // Reject if the bounding range is unreasonably large -- tensor
      // pool may have placed them far apart with unrelated allocations
      // in between, which would force the driver to flush MB of cache
      // we don't need. 64 MB cap = ~16384 fp16 hidden across many
      // partitions; comfortable upper bound.
      if (span <= 64ull * 1024 * 1024) {
        mha_sync_cl_ctx->command_queue_inst_.enqueueSVMMap(
          const_cast<char *>(min_p), span, /*read_only=*/true);
        if (profile_this_decode) {
          g_mha_core_decode_profile.ns_entry_drain +=
            mha_now_ns() - t_drain0;
          g_mha_core_decode_profile.ns_drain_q +=
            mha_now_ns() - t_drain0;
          // k/v/o intentionally left at 0 to make the unified vs
          // four-map comparison obvious in the printed profile.
        }
        // Skip the four separate maps below.
        goto mha_entry_drain_done;
      }
    }
    {
      map_if_svm(query, /*read_only=*/true);
      const uint64_t t_drain_q =
        profile_this_decode ? mha_now_ns() : 0;
      map_if_svm(key, /*read_only=*/true);
      const uint64_t t_drain_k =
        profile_this_decode ? mha_now_ns() : 0;
      map_if_svm(value, /*read_only=*/true);
      const uint64_t t_drain_v =
        profile_this_decode ? mha_now_ns() : 0;
      if (!skip_output_entry_drain) {
        map_if_svm(output, /*read_only=*/false);  // CPU will write here
      }
      const uint64_t t_drain_o =
        profile_this_decode ? mha_now_ns() : 0;
      if (profile_this_decode) {
        g_mha_core_decode_profile.ns_entry_drain += t_drain_o - t_drain0;
        g_mha_core_decode_profile.ns_drain_q += t_drain_q - t_drain0;
        g_mha_core_decode_profile.ns_drain_k += t_drain_k - t_drain_q;
        g_mha_core_decode_profile.ns_drain_v += t_drain_v - t_drain_k;
        g_mha_core_decode_profile.ns_drain_o += t_drain_o - t_drain_v;
      }
    }
mha_entry_drain_done:;
  }
#endif

  nntrainer::Tensor &cache_key =
    context.getTensor(tensor_idx[AttentionParams::cache_key]);
  nntrainer::Tensor &cache_value =
    context.getTensor(tensor_idx[AttentionParams::cache_value]);

  nntrainer::Tensor sink;
  if (use_sink) {
    sink = context.getWeight(sink_idx);
  }

  ml::train::TensorDim query_dim =
    query.getDim(); // (B, 1, seq_len, n_heads_Q * head_dim)
  ml::train::TensorDim key_dim =
    key.getDim(); // (B, 1, seq_len, n_heads_KV * head_dim)
  ml::train::TensorDim value_dim =
    value.getDim(); // (B, 1, seq_len, n_heads_KV * head_dim)
  ml::train::TensorDim output_dim =
    output.getDim(); // (B, 1, seq_len, n_heads_Q * head_dim)
  ml::train::TensorDim cache_key_dim =
    cache_key.getDim(); // (B, 1, max_timestep, n_heads_KV * head_dim)
  ml::train::TensorDim cache_value_dim =
    cache_value.getDim(); // (B, 1, max_timestep, n_heads_KV * head_dim)

  ml::train::TensorDim query_step_dim =
    get_step_dim(query_dim); // (1, 1, step_size, n_heads_Q * head_dim)
  ml::train::TensorDim key_step_dim = get_step_dim(key_dim);
  ml::train::TensorDim value_step_dim = get_step_dim(value_dim);
  ml::train::TensorDim output_step_dim =
    get_step_dim(output_dim); // (1, 1, step_size, n_heads_Q * head_dim)
  ml::train::TensorDim cache_key_step_dim =
    get_step_dim(cache_key_dim); // (1, 1, step_size, n_heads_KV * head_dim)

  ml::train::TensorDim cache_value_step_dim =
    get_step_dim(cache_value_dim); // (1, 1, step_size, n_heads_KV * head_dim)

  unsigned int batch_size = query_dim.batch();
  // do the incremental forwarding
  for (unsigned int batch = 0; batch < batch_size; ++batch) {

    // preparing step tensors
    nntrainer::Tensor query_step = query.getSharedDataTensor(
      query_step_dim, batch * query_dim.getFeatureLen(), true);
    nntrainer::Tensor key_step = key.getSharedDataTensor(
      key_step_dim, batch * key_dim.getFeatureLen(), true);
    nntrainer::Tensor value_step = value.getSharedDataTensor(
      value_step_dim, batch * value_dim.getFeatureLen(), true);
    nntrainer::Tensor output_step = output.getSharedDataTensor(
      output_step_dim, batch * output_dim.getFeatureLen(), true);

    if (query_step.getDataType() == ml::train::TensorDim::DataType::FP32) {
#if ENABLE_FP16 && defined(__ANDROID__)
      nntrainer::TensorDim Q_step_dim = query_step_dim;
      nntrainer::TensorDim K_step_dim = key_step_dim;
      nntrainer::TensorDim V_step_dim = value_step_dim;
      nntrainer::TensorDim O_step_dim = output_step_dim;
      Q_step_dim.setDataType(ml::train::TensorDim::DataType::FP16);
      K_step_dim.setDataType(ml::train::TensorDim::DataType::FP16);
      V_step_dim.setDataType(ml::train::TensorDim::DataType::FP16);
      O_step_dim.setDataType(ml::train::TensorDim::DataType::FP16);

      nntrainer::Tensor Q_step = nntrainer::Tensor(Q_step_dim, true);
      nntrainer::Tensor K_step = nntrainer::Tensor(K_step_dim, true);
      nntrainer::Tensor V_step = nntrainer::Tensor(V_step_dim, true);
      nntrainer::Tensor O_step = nntrainer::Tensor(O_step_dim, true);

      Q_step.copyData(query_step);
      K_step.copyData(key_step);
      V_step.copyData(value_step);
      if (use_sink) {
        one_batch_incremental_forwarding(
          batch, _from, from, to, Q_step, K_step, V_step, O_step, cache_key,
          cache_value, cache_key_dim, cache_key_step_dim, cache_value_dim,
          cache_value_step_dim, sink);
      } else {
        one_batch_incremental_forwarding(batch, _from, from, to, Q_step, K_step,
                                         V_step, O_step, cache_key, cache_value,
                                         cache_key_dim, cache_key_step_dim,
                                         cache_value_dim, cache_value_step_dim);
      }
      output_step.copyData(O_step);
#else
      if (use_sink) {
        one_batch_incremental_forwarding(
          batch, _from, from, to, query_step, key_step, value_step, output_step,
          cache_key, cache_value, cache_key_dim, cache_key_step_dim,
          cache_value_dim, cache_value_step_dim, sink);
      } else {
        one_batch_incremental_forwarding(
          batch, _from, from, to, query_step, key_step, value_step, output_step,
          cache_key, cache_value, cache_key_dim, cache_key_step_dim,
          cache_value_dim, cache_value_step_dim);
      }
#endif
    } else {
      one_batch_incremental_forwarding(
        batch, _from, from, to, query_step, key_step, value_step, output_step,
        cache_key, cache_value, cache_key_dim, cache_key_step_dim,
        cache_value_dim, cache_value_step_dim);
    }
  }

  // increase cache size
  cache_index += step_size;

#if defined(ENABLE_OPENCL) && ENABLE_OPENCL == 1
  // Matching unmap for the map(WRITE) on `output` at the top — commits
  // the CPU writes in this layer back to SVM so the o_proj gemm that
  // follows (reading `output` as an SVM kernel arg) sees the right
  // data. Queue command; in-order queue guarantees the o_proj kernel
  // runs after this unmap.
  // Phase B.13: skip the unmap when we skipped the matching map at
  // the top (NNTRAINER_MHA_NO_OUTPUT_ENTRY_DRAIN=1 + ATTN_GPU=1).
  // The GPU path never CPU-wrote so there's nothing to commit back.
  //
  // Stage 2 fix history:
  //   Stage 2b (wrong): added `&& !skip_entry_drain` here, skipping the
  //   entire block. The old UB-unmap (unmap without prior map) was
  //   accidentally providing an Adreno SVM coherence fence for o_proj;
  //   removing it entirely caused wrong generation output ("Chinese zephyr"
  //   hallucination at first decode token).
  //
  //   Stage 2b correction (this code): restore the block for skip_entry_drain
  //   path but prefix a proper SVMMap(read_only) before the Unmap so the
  //   Map+Unmap pair is valid (not UB) while still providing the coherence
  //   fence. When skip_entry_drain=false the entry already did the Map, so
  //   we only need the Unmap (same as before Stage 2).
  if (mha_sync_cl_ctx && !skip_output_entry_drain &&
      output.getMemoryData() && output.getMemoryData()->isSVM()) {
    const uint64_t t_exit0 =
      profile_this_decode ? mha_now_ns() : 0;
    if (skip_entry_drain) {
      // Entry map was skipped; enqueue a non-blocking read-only map so the
      // subsequent Unmap is a valid (non-UB) pair and the Map+Unmap sequence
      // acts as an Adreno SVM coherence fence in the GPU timeline.
      // blocking=false: CPU returns immediately; GPU executes Map then Unmap
      // in-order before o_proj's kernel reads output SVM.
      // Avoids the ~1.2ms/call CPU stall that blocking=true caused.
      mha_sync_cl_ctx->command_queue_inst_.enqueueSVMMap(
        output.getData<char>(), output.bytes(), /*read_only=*/true,
        /*blocking=*/false);
    }
    mha_sync_cl_ctx->command_queue_inst_.enqueueSVMUnmap(
      output.getData<char>());
    // Phase B publish: put MHA output into GpuImagePool so o_proj
    // can pool-hit. Skips its svm_to_image2d and — crucially — the
    // blocking SVMMap the HalfTensor wrapper would otherwise need
    // to drain MHA's in-flight writes. Publish matches o_proj's
    // expected shape: M = step_size, K = output.width() (hidden_dim).
    //
    // Phase I: o_proj currently goes through HalfTensor::dotQInteger's
    // SVM path (not the gemv_int4_image2d pool-hit path), so this
    // publish runs an extra svm_to_image2d kernel dispatch that
    // nothing consumes -- pure overhead. Skip it when env-gated.
    static const bool s_mha_no_publish =
      std::getenv("NNTRAINER_MHA_NO_PUBLISH") != nullptr;
    const int pub_W = (int)output.width();
    if (!s_mha_no_publish && (pub_W % 4) == 0) {
      const int pub_M = (int)(output.batch() * output.channel() *
                                (int)step_size);
      nntrainer::svm_to_image2d_publish(output.getData<char>(),
                                        pub_M, pub_W);
    }
    if (profile_this_decode)
      g_mha_core_decode_profile.ns_exit_unmap_publish +=
        mha_now_ns() - t_exit0;
  }
#endif

  if (profile_this_call) {
    g_mha_core_profile.ns += mha_now_ns() - t_layer_start;
    g_mha_core_profile.calls++;
  } else if (profile_this_decode) {
    g_mha_core_decode_profile.ns += mha_now_ns() - t_layer_start;
    g_mha_core_decode_profile.ns_cpu +=
      mha_now_cpu_ns() - t_cpu_start;
    g_mha_core_decode_profile.calls++;
  }
}

/**
 * @brief Function to compute Attention Scores using Tensor inputs. Wrapper
 * around nntrainer::compute_kcaches with multi-threading support
 *
 * Expected Input Shapes:
 * @param in (Query): [Batch, 1, sequence_len, Num_Heads_Q * Head_Dim]
 * @param cache (Key Cache): [Batch, 1, Max_Timestep, Num_Heads_KV * Head_Dim]
 * @param out (Attention Score): [Batch, 1, 1, Num_Heads_Q * Context_Len]
 *            where Context_Len is usually the current timestep 'to'.
 *
 */
void MHACoreLayer::compute_kcaches(
  nntrainer::Tensor &in, nntrainer::Tensor &cache, nntrainer::Tensor &out,
  unsigned int from, size_t sequence_len, unsigned int num_head,
  unsigned int group_size, unsigned int head_dim, BS::thread_pool<> &pool) {

  // Dispatch based on data type (FP32 or FP16)
  if (in.getDataType() == ml::train::TensorDim::DataType::FP32) {
    if (sequence_len == 1) {
      // Single token processing (common during generation)
      // Parallelize over KV heads for decoding since Q direction is always 1
      int row_to_compute = is_causal ? from + 1 : from + sequence_len;
      unsigned int num_cache_head = num_head / group_size;

      // Use OpenMP for lower overhead parallelization during decoding
      const float *in_data = in.getData<float>();
      const uint16_t *cache_data = cache.getData<uint16_t>();
      float *out_data = out.getData<float>();

#pragma omp parallel for schedule(static)
      for (unsigned int head_kv = 0; head_kv < num_cache_head; ++head_kv) {
        nntrainer::compute_kcaches<uint16_t>(
          in_data, cache_data, out_data, row_to_compute, num_cache_head,
          head_dim, group_size, tile_size, local_window_size, head_kv,
          head_kv + 1);
      }

    } else {
      // Sequence processing (prefill or chunked)
      // Parallelize over the sequence length
      std::vector<std::future<void>> futures;
      int seq =
        sequence_len < local_window_size ? sequence_len : local_window_size;

      for (int i = 0; i < seq; ++i) {
        float *input_addr = in.getData<float>() + num_head * head_dim * i;
        uint16_t *cache_addr = cache.getData<uint16_t>();
        int row_to_compute = is_causal ? from + i + 1 : from + sequence_len;
        // Calculate dynamic offset for the output (triangle optimization)
        size_t out_start_row =
          is_causal ? calc_attn_index(from + i) - calc_attn_index(from)
                    : i * (from + sequence_len);
        float *output_addr = out.getData<float>() + out_start_row * num_head;

        futures.emplace_back(pool.submit_task([=]() {
          nntrainer::compute_kcaches<uint16_t>(
            input_addr, cache_addr, output_addr, row_to_compute,
            num_head / group_size, head_dim, group_size, tile_size,
            local_window_size);
        }));
      }
      for (auto &fut : futures)
        fut.get();
    }
  } else if (in.getDataType() == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    if (sequence_len == 1) {
      // Single token processing (common during generation)
      // Parallelize over KV heads for decoding since Q direction is always 1
      int num_rows = is_causal ? from + 1 : from + sequence_len;
      unsigned int num_cache_head = num_head / group_size;

      // Use OpenMP for lower overhead parallelization during decoding
      const _FP16 *in_data = in.getData<_FP16>();
      const _FP16 *cache_data = cache.getData<_FP16>();
      _FP16 *out_data = out.getData<_FP16>();

#pragma omp parallel for schedule(static)
      for (unsigned int head_kv = 0; head_kv < num_cache_head; ++head_kv) {
        nntrainer::compute_kcaches(
          in_data, cache_data, out_data, num_rows, num_cache_head, head_dim,
          group_size, tile_size, local_window_size, head_kv, head_kv + 1);
      }
    } else {
      {
        std::vector<std::future<void>> futures;
        unsigned int seq_start =
          sequence_len < local_window_size ? 0 : sequence_len - local_window_size;
        for (unsigned int i = seq_start; i < sequence_len; ++i) {
          _FP16 *input_addr = in.getData<_FP16>() + num_head * head_dim * i;
          _FP16 *cache_addr = cache.getData<_FP16>();
          int row_to_compute = is_causal ? from + i + 1 : from + sequence_len;
          size_t out_start_row =
            is_causal ? calc_attn_index(from + i) - calc_attn_index(from)
                      : i * (from + sequence_len);

          _FP16 *output_addr = out.getData<_FP16>() + out_start_row * num_head;

          futures.emplace_back(pool.submit_task([=]() {
            nntrainer::compute_kcaches(input_addr, cache_addr, output_addr,
                                       row_to_compute, num_head / group_size,
                                       head_dim, group_size, tile_size,
                                       local_window_size);
          }));
        }
        for (auto &fut : futures)
          fut.get();
      }
    }
#else
    NNTR_THROW_IF(true, std::invalid_argument) << "enable-fp16 is not set!";
#endif
  }
}

void MHACoreLayer::one_batch_incremental_forwarding(
  const unsigned int batch, const unsigned int _from, const unsigned int from,
  const unsigned int to, nntrainer::Tensor &query_step,
  nntrainer::Tensor &key_step, nntrainer::Tensor &value_step,
  nntrainer::Tensor &attention_output_step, nntrainer::Tensor &cache_key,
  nntrainer::Tensor &cache_value, ml::train::TensorDim &cache_key_dim,
  ml::train::TensorDim &cache_key_step_dim,
  ml::train::TensorDim &cache_value_dim,
  ml::train::TensorDim &cache_value_step_dim) {

  /**
   *
   *  cache_key
   *  +------------------------------------------+
   *  |<--cache_index-->|<--b_cache_value_step-->|
   *  +------------------------------------------+
   *                    |<-------key_step------->|
   *  |<-------------b_cached_key--------------->|
   */

  // Load Input Tensors of this batch : b_ denotes a Tensor for this batch
  auto &pool =
    nntrainer::Engine::Global().getThreadPoolManager()->getThreadPool();

  nntrainer::Tensor b_cache_key_step = cache_key.getSharedDataTensor(
    cache_key_step_dim,
    batch * cache_key_dim.getFeatureLen() + cache_index * cache_key_dim.width(),
    true);
  nntrainer::Tensor b_cache_value_step =
    cache_value.getSharedDataTensor(cache_value_step_dim,
                                    batch * cache_value_dim.getFeatureLen() +
                                      cache_index * cache_value_dim.width(),
                                    true);

  // Sub-stage profile routing: prefill (M>1) buckets into
  // g_mha_core_profile, decode (M==1) buckets into
  // g_mha_core_decode_profile.  Timestamps are cheap so we take them
  // unconditionally and just route at accumulate time.
  const bool profile_substage = (to - from) > 1;
  const bool profile_decode   = (to - from) == 1;
  auto acc_rope_q = [&](uint64_t dt) {
    if (profile_substage) g_mha_core_profile.ns_rope_q += dt;
    else if (profile_decode) g_mha_core_decode_profile.ns_rope_q += dt;
  };
  auto acc_rope_k = [&](uint64_t dt) {
    if (profile_substage) g_mha_core_profile.ns_rope_k += dt;
    else if (profile_decode) g_mha_core_decode_profile.ns_rope_k += dt;
  };
  auto acc_v_copy = [&](uint64_t dt) {
    if (profile_substage) g_mha_core_profile.ns_v_copy += dt;
    else if (profile_decode) g_mha_core_decode_profile.ns_v_copy += dt;
  };

  // GPU-RoPE path: when env-gated and decode-shape (M=1, fp16, SVM),
  // dispatch the rope_decode_fp16 kernel + clEnqueueSVMMemcpy on the
  // blas_cc queue so that NNTRAINER_MHA_NO_ENTRY_DRAIN can be flipped
  // on without racing CPU-side RoPE against GPU producer writes.
  bool rope_gpu_done = false;
#if defined(ENABLE_OPENCL) && ENABLE_OPENCL == 1 && defined(ENABLE_FP16)
  static const bool s_rope_gpu =
    std::getenv("NNTRAINER_RoPE_GPU") != nullptr;
  const bool q_svm = query_step.getMemoryData() &&
                     query_step.getMemoryData()->isSVM();
  const bool k_svm = key_step.getMemoryData() &&
                     key_step.getMemoryData()->isSVM();
  const bool v_svm = value_step.getMemoryData() &&
                     value_step.getMemoryData()->isSVM();
  const bool kc_svm = b_cache_key_step.getMemoryData() &&
                      b_cache_key_step.getMemoryData()->isSVM();
  const bool vc_svm = b_cache_value_step.getMemoryData() &&
                      b_cache_value_step.getMemoryData()->isSVM();
  const bool fp16_q =
    query_step.getDataType() == ml::train::TensorDim::DataType::FP16;
  const bool rope_gpu_eligible =
    s_rope_gpu && profile_decode && fp16_q && head_dim == 128 &&
    q_svm && k_svm && v_svm && kc_svm && vc_svm;
  static std::atomic<int> s_rope_gpu_diag{0};
  if (s_rope_gpu_diag.fetch_add(1, std::memory_order_relaxed) == 0) {
    std::fprintf(stderr,
                 "[ROPE_GPU_DIAG] env=%d decode=%d fp16=%d head_dim=%zu "
                 "svm{q=%d k=%d v=%d kc=%d vc=%d} -> eligible=%d\n",
                 (int)s_rope_gpu, (int)profile_decode, (int)fp16_q,
                 (size_t)head_dim,
                 (int)q_svm, (int)k_svm, (int)v_svm,
                 (int)kc_svm, (int)vc_svm,
                 (int)rope_gpu_eligible);
  }
  if (rope_gpu_eligible) {
    // query_step layout: (1, 1, 1, num_heads_Q * head_dim) for decode.
    // Use the layer's cached num_heads_{Q,KV} rather than tensor dims
    // so we don't depend on a particular shape convention.
    const uint64_t t_rope_q0 = mha_now_ns();
    const bool ok_q = nntrainer::rope_decode_fp16_cl(
      query_step.getData<char>(), query_step.getData<char>(),
      (unsigned int)num_heads_Q, (unsigned int)head_dim,
      cache_index, theta);
    acc_rope_q(mha_now_ns() - t_rope_q0);

    const uint64_t t_rope_k0 = mha_now_ns();
    const bool ok_k = nntrainer::rope_decode_fp16_cl(
      key_step.getData<char>(), b_cache_key_step.getData<char>(),
      (unsigned int)num_heads_KV, (unsigned int)head_dim,
      cache_index, theta);
    acc_rope_k(mha_now_ns() - t_rope_k0);

    const uint64_t t_v0 = mha_now_ns();
    const bool ok_v = nntrainer::svm_memcpy_fp16_cl(
      value_step.getData<char>(), b_cache_value_step.getData<char>(),
      value_step.bytes());
    acc_v_copy(mha_now_ns() - t_v0);

    rope_gpu_done = ok_q && ok_k && ok_v;
    static std::atomic<int> s_rope_helper_diag{0};
    if (s_rope_helper_diag.fetch_add(1, std::memory_order_relaxed) == 0) {
      std::fprintf(stderr,
                   "[ROPE_GPU_DIAG] helper ok_q=%d ok_k=%d ok_v=%d -> done=%d\n",
                   (int)ok_q, (int)ok_k, (int)ok_v, (int)rope_gpu_done);
    }
  }
#endif

  if (!rope_gpu_done) {
    // apply rotary embedding for query
    const uint64_t t_rope_q0 = mha_now_ns();
    apply_rotary_emb_tensor_v2(query_step, query_step, head_dim, cache_index,
                               false);
    acc_rope_q(mha_now_ns() - t_rope_q0);

    // append kcache with rotary embedding
    const uint64_t t_rope_k0 = mha_now_ns();
    apply_rotary_emb_tensor_v2(key_step, b_cache_key_step, head_dim,
                               cache_index, false);
    acc_rope_k(mha_now_ns() - t_rope_k0);

    // append vcache without rotary embedding
    if (query_step.getDataType() == ml::train::TensorDim::DataType::FP32) {
      const uint64_t t_v0 = mha_now_ns();
      apply_rotary_emb_tensor_v2(value_step, b_cache_value_step, head_dim,
                                 cache_index, true);
      acc_v_copy(mha_now_ns() - t_v0);
    } else if (query_step.getDataType() ==
               ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
      const uint64_t t_v0 = mha_now_ns();
      b_cache_value_step.copyData(value_step);
      acc_v_copy(mha_now_ns() - t_v0);
#else
      NNTR_THROW_IF(true, std::invalid_argument) << "enable-fp16 is not set!";
#endif
    }
  }

  /// @todo replace step_size into input height
  unsigned int step_size = to - from;
  unsigned int cache_from = cache_index;
  unsigned int cache_to = cache_from + step_size;

  ml::train::TensorDim cached_key_dim = cache_key_dim;
  ml::train::TensorDim cached_value_dim = cache_value_dim;
  cached_key_dim.height(cache_to);
  cached_value_dim.height(cache_to);

  nntrainer::Tensor b_cached_key = cache_key.getSharedDataTensor(
    cached_key_dim, batch * cache_key_dim.getFeatureLen(), true);
  nntrainer::Tensor b_cached_value = cache_value.getSharedDataTensor(
    cached_value_dim, batch * cache_value_dim.getFeatureLen(), true);

  unsigned int gqa_size = num_heads_Q / num_heads_KV;

#if defined(ENABLE_OPENCL) && ENABLE_OPENCL == 1
  // NNTRAINER_ATTN_GPU=1 replaces the NEON qk + softmax + av triple
  // with a single fused FlashAttention-style GPU dispatch.  Needs
  // SVM-backed Q / K / V / output, GQA shape, head_dim == 128.
  // Numerical result is the same attention output tensor the NEON
  // path would produce, so we can short-circuit the entire block
  // (no out_ scratch needed).
  static const bool s_attn_gpu =
    std::getenv("NNTRAINER_ATTN_GPU") != nullptr;
  static bool s_attn_gpu_logged = false;
  if (!s_attn_gpu_logged) {
    std::fprintf(stderr,
                 "[mha_core] NNTRAINER_ATTN_GPU=%s -> %s\n",
                 std::getenv("NNTRAINER_ATTN_GPU")
                   ? std::getenv("NNTRAINER_ATTN_GPU")
                   : "(unset)",
                 s_attn_gpu ? "GPU attention_fused_fp16"
                            : "NEON qk + softmax + av");
    s_attn_gpu_logged = true;
  }
  if (s_attn_gpu &&
      query_step.getMemoryData() && query_step.getMemoryData()->isSVM() &&
      b_cached_key.getMemoryData() && b_cached_key.getMemoryData()->isSVM() &&
      b_cached_value.getMemoryData() && b_cached_value.getMemoryData()->isSVM() &&
      attention_output_step.getMemoryData() &&
      attention_output_step.getMemoryData()->isSVM() &&
      query_step.getDataType() == ml::train::TensorDim::DataType::FP16 &&
      head_dim == 128 &&
      (num_heads_Q % gqa_size) == 0) {
    const uint64_t t_fused0 = mha_now_ns();
    nntrainer::attention_fused_fp16_cl(
      (void *)query_step.getData<_FP16>(),
      (void *)b_cached_key.getData<_FP16>(),
      (void *)b_cached_value.getData<_FP16>(),
      (void *)attention_output_step.getData<_FP16>(),
      (unsigned int)step_size, (unsigned int)cache_to,
      (unsigned int)cache_from,
      (unsigned int)num_heads_Q, (unsigned int)gqa_size,
      (unsigned int)head_dim, is_causal ? 1 : 0);
    {
      // Fused kernel covers all three sub-stages; record the combined
      // time under ns_qk for the moment (ns_softmax and ns_av stay at
      // zero so the profile makes the shift obvious at a glance).
      // Also record under ns_attn_fused_call for the new detailed
      // breakdown that distinguishes entry_drain / fused_call / exit.
      const uint64_t dt = mha_now_ns() - t_fused0;
      if (profile_substage)      g_mha_core_profile.ns_qk += dt;
      else if (profile_decode) {
        g_mha_core_decode_profile.ns_qk += dt;
        g_mha_core_decode_profile.ns_attn_fused_call += dt;
      }
    }
    return;
  }
#endif

  // out_ stores the output of Q * K
  nntrainer::Tensor out_(
    1, 1,
    is_causal ? (calc_attn_index(cache_to) - calc_attn_index(cache_from))
              : (step_size * cache_to),
    num_heads_Q, query_step.getTensorType());

  const uint64_t t_qk0 = mha_now_ns();
  compute_kcaches(query_step, b_cached_key, out_, cache_from,
                  cache_to - cache_from, num_heads_Q, gqa_size, head_dim, pool);
  {
    const uint64_t dt = mha_now_ns() - t_qk0;
    if (profile_substage)     g_mha_core_profile.ns_qk += dt;
    else if (profile_decode)  g_mha_core_decode_profile.ns_qk += dt;
  }

  const uint64_t t_sm0 = mha_now_ns();
  softmax_triangle(out_, step_size, num_heads_Q, cache_from, pool);
  {
    const uint64_t dt = mha_now_ns() - t_sm0;
    if (profile_substage)     g_mha_core_profile.ns_softmax += dt;
    else if (profile_decode)  g_mha_core_decode_profile.ns_softmax += dt;
  }

  const uint64_t t_av0 = mha_now_ns();
  compute_fp16vcache_transposed(out_, b_cached_value, attention_output_step,
                                cache_from, num_heads_KV, gqa_size, head_dim,
                                cache_to, pool);
  {
    const uint64_t dt = mha_now_ns() - t_av0;
    if (profile_substage)     g_mha_core_profile.ns_av += dt;
    else if (profile_decode)  g_mha_core_decode_profile.ns_av += dt;
  }
}

void MHACoreLayer::one_batch_incremental_forwarding(
  const unsigned int batch, const unsigned int _from, const unsigned int from,
  const unsigned int to, nntrainer::Tensor &query_step,
  nntrainer::Tensor &key_step, nntrainer::Tensor &value_step,
  nntrainer::Tensor &attention_output_step, nntrainer::Tensor &cache_key,
  nntrainer::Tensor &cache_value, ml::train::TensorDim &cache_key_dim,
  ml::train::TensorDim &cache_key_step_dim,
  ml::train::TensorDim &cache_value_dim,
  ml::train::TensorDim &cache_value_step_dim, nntrainer::Tensor &sink_step) {
  /// @todo replace from, to into cache_index, input height
  /// @note currently, only gpt-oss uses this method

  /**
   *  cache_key
   *  +--------+                        ->
   *  |        |                        ->
   *  |        |                        ->
   *  |........| from                   ->
   *  |........| to -> b_cache_key_step -> b_cached_key
   *  |        |
   *  +--------+
   *
   */

  /** 1. Load Input Tensors of this batch : b_ denotes a Tensor for this batch
   * **/
  auto &pool =
    nntrainer::Engine::Global().getThreadPoolManager()->getThreadPool();

  nntrainer::Tensor b_cache_key_step = cache_key.getSharedDataTensor(
    cache_key_step_dim,
    batch * cache_key_dim.getFeatureLen() + from * cache_key_dim.width(), true);
  nntrainer::Tensor b_cache_value_step = cache_value.getSharedDataTensor(
    cache_value_step_dim,
    batch * cache_value_dim.getFeatureLen() + from * cache_value_dim.width(),
    true);

  // Sub-stage profile gate: only count prefill calls (to-from > 1) so
  // decode noise doesn't dilute the breakdown.
  const bool profile_substage = (to - from) > 1;

  const uint64_t t_rope_q0 = profile_substage ? mha_now_ns() : 0;
  apply_rotary_emb_tensor_v2(query_step, query_step, head_dim, _from, false);
  if (profile_substage)
    g_mha_core_profile.ns_rope_q += mha_now_ns() - t_rope_q0;

  const uint64_t t_rope_k0 = profile_substage ? mha_now_ns() : 0;
  apply_rotary_emb_tensor_v2(key_step, b_cache_key_step, head_dim, _from,
                             false);
  if (profile_substage)
    g_mha_core_profile.ns_rope_k += mha_now_ns() - t_rope_k0;

  if (query_step.getDataType() == ml::train::TensorDim::DataType::FP32) {
    const uint64_t t_v0 = profile_substage ? mha_now_ns() : 0;
    apply_rotary_emb_tensor_v2(value_step, b_cache_value_step, head_dim, _from,
                               true);
    if (profile_substage)
      g_mha_core_profile.ns_v_copy += mha_now_ns() - t_v0;
  } else if (query_step.getDataType() == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    const uint64_t t_v0 = profile_substage ? mha_now_ns() : 0;
    b_cache_value_step.copyData(value_step);
    if (profile_substage)
      g_mha_core_profile.ns_v_copy += mha_now_ns() - t_v0;
#else
    NNTR_THROW_IF(true, std::invalid_argument) << "enable-fp16 is not set!";
#endif
  }

  ml::train::TensorDim cached_key_dim = cache_key_dim;
  ml::train::TensorDim cached_value_dim = cache_value_dim;
  cached_key_dim.height(to);
  cached_value_dim.height(to);

  nntrainer::Tensor b_cached_key = cache_key.getSharedDataTensor(
    cached_key_dim, batch * cache_key_dim.getFeatureLen(), true);
  nntrainer::Tensor b_cached_value = cache_value.getSharedDataTensor(
    cached_value_dim, batch * cache_value_dim.getFeatureLen(), true);

  nntrainer::Tensor out_(
    1, 1,
    is_causal
      ? (((to - from) == 1) ? to : calc_attn_index(to) - calc_attn_index(from))
      : ((to - from) * to),
    num_heads_Q, query_step.getTensorType());

  unsigned int gqa_size = num_heads_Q / num_heads_KV;

  const uint64_t t_qk0 = profile_substage ? mha_now_ns() : 0;
  compute_kcaches(query_step, b_cached_key, out_, _from, to - from, num_heads_Q,
                  gqa_size, head_dim, pool);
  if (profile_substage)
    g_mha_core_profile.ns_qk += mha_now_ns() - t_qk0;

  const uint64_t t_sm0 = profile_substage ? mha_now_ns() : 0;
  softmax_triangle(out_, to - from, num_heads_Q, from, pool, sink_step);
  if (profile_substage)
    g_mha_core_profile.ns_softmax += mha_now_ns() - t_sm0;

  const uint64_t t_av0 = profile_substage ? mha_now_ns() : 0;
  compute_fp16vcache_transposed(out_, b_cached_value, attention_output_step,
                                from, num_heads_KV, gqa_size, head_dim, to,
                                pool);
  if (profile_substage)
    g_mha_core_profile.ns_av += mha_now_ns() - t_av0;
}

/************************************************************** */

/**
 * @brief rotary embedding-related member function
 * @note seq_len -> max_position_embeddings
 */
void MHACoreLayer::precompute_freqs(int head_dim, unsigned int seq_len,
                                    float theta, bool is_fp16) {
  // compute the freqs only when it is the first time to call this function
#ifdef ENABLE_FP16
  if (freqs_cos_fp16 != nullptr && freqs_cos_fp16->size() == seq_len)
    return;
#else
  if (freqs_cos != nullptr && freqs_cos->size() == seq_len)
    return;
#endif

  if (thetas.empty()) {
    if (rope_scaling_type == "default")
      _compute_default_parameters(head_dim, theta);
    else if (rope_scaling_type == "yarn")
      _compute_yarn_parameters(head_dim, theta);
    else
      NNTR_THROW_IF(true, std::invalid_argument) << "Unsupported rope type!";
  }

  unsigned int half_ = head_dim / 2;

  if (!is_fp16) {
    // cos / sin
    auto cos = new std::vector<std::vector<float>>();
    cos->assign(seq_len, std::vector<float>(head_dim, 0));
    auto sin = new std::vector<std::vector<float>>();
    sin->assign(seq_len, std::vector<float>(head_dim, 0));

    // update cos / sin frequency
    for (unsigned int i = 0; i < seq_len; ++i) {

#ifdef USE_NEON
      nntrainer::calc_trigonometric_vals_dup(half_, thetas.data(),
                                             (*cos)[i].data(), (*sin)[i].data(),
                                             i, attention_scaling);
#else
      for (unsigned int j = 0; j < half_; ++j) {
        float angle = i * thetas[j];
        (*cos)[i][j] = std::cos(angle) * attention_scaling;
        (*cos)[i][j + half_] =
          std::cos(angle) * attention_scaling; // repeated 2 times

        (*sin)[i][j] = std::sin(angle) * attention_scaling;
        (*sin)[i][j + half_] =
          std::sin(angle) * attention_scaling; // repeated 2 times
      }
#endif
    }
    freqs_cos = cos;
    freqs_sin = sin;
  }

#ifdef ENABLE_FP16
  if (is_fp16) {
    // cos / sin for FP16
    auto cos_fp16 = new std::vector<std::vector<_FP16>>();
    cos_fp16->assign(seq_len, std::vector<_FP16>(head_dim, 0));
    auto sin_fp16 = new std::vector<std::vector<_FP16>>();
    sin_fp16->assign(seq_len, std::vector<_FP16>(head_dim, 0));

    std::vector<float> cos_tmp(head_dim);
    std::vector<float> sin_tmp(head_dim);

    for (unsigned int i = 0; i < seq_len; ++i) {
#ifdef USE_NEON
      nntrainer::calc_trigonometric_vals_dup(half_, thetas.data(),
                                             cos_tmp.data(), sin_tmp.data(), i,
                                             attention_scaling);
#else
      for (unsigned int j = 0; j < half_; ++j) {
        float angle = i * thetas[j];
        cos_tmp[j] = std::cos(angle) * attention_scaling;
        cos_tmp[j + half_] =
          std::cos(angle) * attention_scaling; // repeated 2 times

        sin_tmp[j] = std::sin(angle) * attention_scaling;
        sin_tmp[j + half_] =
          std::sin(angle) * attention_scaling; // repeated 2 times
      }
#endif
      for (unsigned int j = 0; j < head_dim; ++j) {
        (*cos_fp16)[i][j] = (_FP16)cos_tmp[j];
        (*sin_fp16)[i][j] = (_FP16)sin_tmp[j];
      }
    }
    freqs_cos_fp16 = cos_fp16;
    freqs_sin_fp16 = sin_fp16;
  }
#endif
};

void MHACoreLayer::_compute_default_parameters(int head_dim, float theta) {

  // no attention scaling
  attention_scaling = 1.0f;

  // theta_i = 10000^(-2(i-1)/dim) for i = [1, 2, ... , dim/2]
  // head_dim should be divisible by 2
  unsigned int half_ = head_dim / 2;
  for (unsigned int i = 0; i < half_; ++i) {
    thetas.push_back(1.0 /
                     (std::pow(theta, (2 * i) / static_cast<float>(head_dim))));
  }
}

void MHACoreLayer::_compute_yarn_parameters(int head_dim, float theta) {

  // Config parameters
  ///@todo partial_rotary_factor should be generalized to fully support
  /// transformers's implementation
  // const float partial_rotary_factor = has_partial_rotary_factor ?
  // config_partial_rotary_factor : 1.0f;
  const float partial_rotary_factor = 1.0f;
  const int dim = static_cast<int>(head_dim * partial_rotary_factor);
  const float base = theta;

  // Handle max position embeddings

  // Attention scaling calculation (simplified from Python version)
  auto get_mscale = [](float scale, float mscale = 1.0f) {
    return (scale <= 1.0f) ? 1.0f : (0.1f * mscale * std::log(scale) + 1.0f);
  };

  ///@todo attention_scaling should be generalized to fully support
  /// transformers's implementation
  // if (has_mscale && has_mscale_all_dim) {
  // attention_scaling = get_mscale(factor, mscale) / get_mscale(factor,
  // mscale_all_dim);
  // } else {
  // attention_scaling = get_mscale(factor);
  // }
  attention_scaling = get_mscale(scale);

  ///@todo attention_scaling should be generalized to fully support
  /// transformers's implementation
  // const float beta_fast = has_beta_fast ? config_beta_fast : 32.0f;
  // const float beta_slow = has_beta_slow ? config_beta_slow : 1.0f;
  // const bool truncate = has_truncate ? config_truncate : true;
  // Beta parameters
  const float beta_fast = 32.0f;
  const float beta_slow = 1.0f;
  const bool truncate = false;

  // Helper functions
  auto find_correction_dim = [&](float num_rotations) {
    return (dim * std::log(original_max_position_embeddings /
                           (num_rotations * 2 * M_PI))) /
           (2 * std::log(base));
  };

  auto [low, high] = [&]() {
    float low_val = find_correction_dim(beta_fast);
    float high_val = find_correction_dim(beta_slow);
    if (truncate) {
      low_val = std::floor(low_val);
      high_val = std::ceil(high_val);
    }
    return std::make_pair(low_val, high_val);
  }();

  // Compute position frequencies
  thetas.resize(dim / 2);

  // Compute interpolation and extrapolation frequencies
  std::vector<float> inv_freq_interpolation;
  std::vector<float> inv_freq_extrapolation;
  for (size_t i = 0; i < dim / 2; ++i) {
    inv_freq_extrapolation.push_back(
      1.0 / (std::pow(theta, (2 * i) / static_cast<float>(head_dim))));
    inv_freq_interpolation.push_back(
      1.0 / (scale * std::pow(theta, (2 * i) / static_cast<float>(head_dim))));
  }

  auto linear_ramp_factor = [](float min, float max, int size) {
    if (min == max) {
      max += 0.001f; // Prevent singularity
    }
    std::vector<float> ramp(size);
    for (int i = 0; i < size; ++i) {
      float val = (i - min) / (max - min);
      ramp[i] = std::clamp(val, 0.0f, 1.0f);
    }
    return ramp;
  };

  std::vector<float> inv_freq_extrapolation_factor =
    linear_ramp_factor(low, high, dim / 2);
  for (auto &val : inv_freq_extrapolation_factor) {
    val = 1.0f - val;
  }

  // Combine frequencies
  for (size_t i = 0; i < thetas.size(); ++i) {
    thetas[i] =
      inv_freq_extrapolation[i] * inv_freq_extrapolation_factor[i] +
      inv_freq_interpolation[i] * (1.0f - inv_freq_extrapolation_factor[i]);
  }
}

void MHACoreLayer::apply_rotary_emb_tensor_v2(nntrainer::Tensor &in,
                                              nntrainer::Tensor &out,
                                              unsigned int dim,
                                              unsigned int from,
                                              bool convert_only) {
  unsigned int half_ = dim / 2;
  unsigned int max_timestep =
    std::get<nntrainer::props::MaxTimestep>(mha_core_props).get();

  if (in.getDataType() == ml::train::TensorDim::DataType::FP32) {
    if (freqs_cos == nullptr) {
      const std::lock_guard<std::mutex> lock(rope_init_mtx);
      if (freqs_cos == nullptr) {
        precompute_freqs(head_dim, max_position_embeddings, theta, false);
      }
    }
    std::vector<float> *cos_ = nullptr;
    std::vector<float> *sin_ = nullptr;

    for (unsigned int b = 0; b < in.batch(); b++) {
      for (unsigned int c = 0; c < in.channel(); c++) {
        for (unsigned int h = 0; h < in.height(); h++) {
          if (from < max_timestep) {
            cos_ = &(*freqs_cos)[from + h];
            sin_ = &(*freqs_sin)[from + h];
          }
          float *in_ptr = in.getData<float>() +
                          b * in.channel() * in.height() * in.width() +
                          c * in.height() * in.width() + h * in.width();

          if (out.getDataType() == ml::train::TensorDim::DataType::FP32) {

            nntrainer::compute_rotary_emb_value(in.width(), dim, half_, in_ptr,
                                                nullptr, cos_->data(),
                                                sin_->data(), convert_only);
          } else if (out.getDataType() ==
                       ml::train::TensorDim::DataType::UINT16 ||
                     out.getDataType() ==
                       ml::train::TensorDim::DataType::FP16) {
            uint16_t *out_ptr = out.getData<uint16_t>() +
                                b * out.channel() * out.height() * out.width() +
                                c * out.height() * out.width() +
                                h * out.width();

            nntrainer::compute_rotary_emb_value(in.width(), dim, half_, in_ptr,
                                                out_ptr, cos_->data(),
                                                sin_->data(), convert_only);
          }
        }
      }
    }
  } else if (in.getDataType() == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    if (freqs_cos_fp16 == nullptr) {
      const std::lock_guard<std::mutex> lock(rope_init_mtx);
      if (freqs_cos_fp16 == nullptr) {
        precompute_freqs(head_dim, max_position_embeddings, theta, true);
      }
    }

    std::vector<_FP16> *cos_ = nullptr;
    std::vector<_FP16> *sin_ = nullptr;

    for (unsigned int b = 0; b < in.batch(); b++) {
      for (unsigned int c = 0; c < in.channel(); c++) {
        for (unsigned int h = 0; h < in.height(); h++) {
          if (from < max_timestep) {
            cos_ = &(*freqs_cos_fp16)[from + h];
            sin_ = &(*freqs_sin_fp16)[from + h];
          }
          _FP16 *in_ptr = in.getData<_FP16>() +
                          b * in.channel() * in.height() * in.width() +
                          c * in.height() * in.width() + h * in.width();
          _FP16 *out_ptr = out.getData<_FP16>() +
                           b * out.channel() * out.height() * out.width() +
                           c * out.height() * out.width() + h * out.width();

          nntrainer::compute_rotary_emb_value(in.width(), dim, half_, in_ptr,
                                              out_ptr, cos_->data(),
                                              sin_->data());
        }
      }
    }
#else
    NNTR_THROW_IF(true, std::invalid_argument) << "enable-fp16 is not set!";
#endif
  }
}

void MHACoreLayer::softmax_triangle(nntrainer::Tensor &qk_out, size_t row,
                                    size_t num_head, unsigned int from,
                                    BS::thread_pool<> &pool) {
  if (qk_out.getDataType() == ml::train::TensorDim::DataType::FP32) {
    float *qk_out_ = qk_out.getData<float>();

    if (attn_logit_softcapping > 0.0f) {
      size_t len =
        qk_out.batch() * qk_out.height() * qk_out.width() * qk_out.channel();
      float inv_softcapping = 1.0f / attn_logit_softcapping;
      for (size_t i = 0; i < len; ++i) {
        qk_out_[i] =
          std::tanh(qk_out_[i] * inv_softcapping) * attn_logit_softcapping;
      }
    }

    if (row == 1) {
      size_t start_row = 0;
      size_t end_row = 0;
      if (is_causal) {
        end_row = from < local_window_size ? from + 1 : local_window_size;
      } else {
        end_row = from + row; // end_row = to
      }
      nntrainer::softmax_row_inplace(qk_out_, start_row, end_row, num_head);
    } else {
      std::vector<std::future<void>> futures;
      int seq = row < local_window_size ? row : local_window_size;
      if (!is_causal)
        seq = row;

      for (int i = 0; i < seq; ++i) {
        size_t start_row, end_row;
        if (is_causal) {
          start_row = calc_attn_index(from + i) - calc_attn_index(from);
          end_row = calc_attn_index(from + i + 1) - calc_attn_index(from);
        } else {
          unsigned int to = from + row;
          start_row = i * to;
          end_row = (i + 1) * to;
        }
        futures.push_back(pool.submit_task([=]() {
          nntrainer::softmax_row(qk_out_, start_row, end_row, num_head);
        }));
      }
      for (auto &fut : futures) {
        fut.get();
      }
    }
  } else if (qk_out.getDataType() == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    _FP16 *qk_out_ = qk_out.getData<_FP16>();

    if (attn_logit_softcapping > 0.0f) {
      size_t len =
        qk_out.batch() * qk_out.height() * qk_out.width() * qk_out.channel();
      float inv_softcapping = 1.0f / attn_logit_softcapping;
      for (size_t i = 0; i < len; ++i) {
        qk_out_[i] = (_FP16)(std::tanh((float)qk_out_[i] * inv_softcapping) *
                             attn_logit_softcapping);
      }
    }

    if (row == 1) {
      size_t start_row = 0;
      size_t end_row = 0;
      if (is_causal) {
        end_row = from < local_window_size ? from + 1 : local_window_size;
      } else {
        end_row = from + row; // end_row = to
      }
      nntrainer::softmax_row_inplace(qk_out_, start_row, end_row, num_head);
    } else {
      std::vector<std::future<void>> futures;
      int seq = row < local_window_size ? row : local_window_size;
      if (!is_causal)
        seq = row;

      for (int i = 0; i < seq; ++i) {
        size_t start_row, end_row;
        if (is_causal) {
          start_row = calc_attn_index(from + i) - calc_attn_index(from);
          end_row = calc_attn_index(from + i + 1) - calc_attn_index(from);
        } else {
          unsigned int to = from + row;
          start_row = i * to;
          end_row = (i + 1) * to;
        }
        futures.push_back(pool.submit_task([=]() {
          nntrainer::softmax_row_inplace(qk_out_, start_row, end_row, num_head);
        }));
      }
      for (auto &fut : futures) {
        fut.get();
      }
    }
#else
    NNTR_THROW_IF(true, std::invalid_argument) << "enable-fp16 is not set!";
#endif
  }
}

void MHACoreLayer::softmax_triangle(nntrainer::Tensor &qk_out, size_t row,
                                    size_t num_head, unsigned int from,
                                    BS::thread_pool<> &pool,
                                    nntrainer::Tensor &sink_step) {
  if (qk_out.getDataType() == ml::train::TensorDim::DataType::FP32) {
    float *qk_out_ = qk_out.getData<float>();

    if (attn_logit_softcapping > 0.0f) {
      size_t len =
        qk_out.batch() * qk_out.height() * qk_out.width() * qk_out.channel();
      float inv_softcapping = 1.0f / attn_logit_softcapping;
      for (size_t i = 0; i < len; ++i) {
        qk_out_[i] =
          std::tanh(qk_out_[i] * inv_softcapping) * attn_logit_softcapping;
      }
    }

    if (row == 1) {
      size_t start_row = 0;
      size_t end_row = 0;
      if (is_causal) {
        end_row = from < local_window_size ? from + 1 : local_window_size;
      } else {
        unsigned int to = from + row;
        end_row = to;
      }
      nntrainer::softmax_row_inplace(qk_out_, start_row, end_row, num_head,
                                     sink_step.getData());
    } else {
      std::vector<std::future<void>> futures;

      int seq = row < local_window_size ? row : local_window_size;
      if (!is_causal)
        seq = row;

      for (int i = 0; i < seq; ++i) {
        size_t start_row, end_row;
        if (is_causal) {
          start_row = calc_attn_index(i + from) - calc_attn_index(from);
          end_row = calc_attn_index(from + i + 1) - calc_attn_index(from);
        } else {
          unsigned int to = from + row;
          start_row = i * to;
          end_row = (i + 1) * to;
        }
        futures.push_back(pool.submit_task([=]() {
          nntrainer::softmax_row(qk_out_, start_row, end_row, num_head,
                                 sink_step.getData());
        }));
      }
      for (auto &fut : futures) {
        fut.get();
      }
    }
  } else if (qk_out.getDataType() == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    _FP16 *qk_out_ = qk_out.getData<_FP16>();
    _FP16 *sink_step_ = sink_step.getData<_FP16>();

    if (attn_logit_softcapping > 0.0f) {
      size_t len =
        qk_out.batch() * qk_out.height() * qk_out.width() * qk_out.channel();
      float inv_softcapping = 1.0f / attn_logit_softcapping;
      for (size_t i = 0; i < len; ++i) {
        qk_out_[i] = (_FP16)(std::tanh((float)qk_out_[i] * inv_softcapping) *
                             attn_logit_softcapping);
      }
    }

    if (row == 1) {
      size_t start_row = 0;
      size_t end_row = 0;
      if (is_causal) {
        end_row = from < local_window_size ? from + 1 : local_window_size;
      } else {
        end_row = from + row; // end_row = to
      }
      nntrainer::softmax_row_inplace(qk_out_, start_row, end_row, num_head,
                                     sink_step_);
    } else {
      std::vector<std::future<void>> futures;
      int seq = row < local_window_size ? row : local_window_size;
      if (!is_causal)
        seq = row;

      for (int i = 0; i < seq; ++i) {
        size_t start_row = calc_attn_index(i + from) - calc_attn_index(from);
        size_t end_row = calc_attn_index(from + i + 1) - calc_attn_index(from);
        futures.push_back(pool.submit_task([=]() {
          nntrainer::softmax_row(qk_out_, start_row, end_row, num_head,
                                 sink_step_);
        }));
      }
      for (auto &fut : futures) {
        fut.get();
      }
    }
#else
    NNTR_THROW_IF(true, std::invalid_argument) << "enable-fp16 is not set!";
#endif
  }
}

void MHACoreLayer::compute_fp16vcache_transposed(
  nntrainer::Tensor &in, nntrainer::Tensor &vcache, nntrainer::Tensor &output,
  int from, int num_cache_head, int gqa_size, int head_dim, int to,
  BS::thread_pool<> &pool) {

  if (in.getDataType() == ml::train::TensorDim::DataType::FP32) {
    if ((to - from) != 1) {
      std::vector<std::future<void>> futures;

      int seq = (to - from) < local_window_size ? to - from : local_window_size;
      // if non-causal, seq is practically to - from.
      if (!is_causal)
        seq = to - from;
      futures.reserve(seq);

      for (int i = 0; i < seq; ++i) {
        futures.push_back(pool.submit_task([=]() {
          size_t start_idx;
          if (is_causal) {
            start_idx =
              calc_attn_index(to - seq + i) - calc_attn_index(to - seq);
          } else {
            start_idx = i * to; // linear index
          }
          const float *input =
            in.getData<float>() + start_idx * num_cache_head * gqa_size;
          float *out = output.getData<float>() +
                       i * (num_cache_head * gqa_size * head_dim);

          int row_num = is_causal ? (to - seq + i) : to - 1;
          nntrainer::compute_fp16vcache_fp32_transposed(
            row_num, input, vcache.getData<uint16_t>(), out, num_cache_head,
            gqa_size, head_dim, local_window_size);
        }));
      }
      for (auto &fut : futures)
        fut.get();
    } else {
      // Single token processing (common during generation)
      // Parallelize over KV heads for decoding since Q direction is always 1
      int row_num = to - 1;

      // Use OpenMP for lower overhead parallelization during decoding
      const float *in_data = in.getData<float>();
      const uint16_t *vcache_data = vcache.getData<uint16_t>();
      float *output_data = output.getData<float>();

#pragma omp parallel for schedule(static)
      for (int head_kv = 0; head_kv < num_cache_head; ++head_kv) {
        nntrainer::compute_fp16vcache_fp32_transposed(
          row_num, in_data, vcache_data, output_data, num_cache_head, gqa_size,
          head_dim, local_window_size, head_kv, head_kv + 1);
      }
    }
  } else if (in.getDataType() == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    if ((to - from) != 1) {
      std::vector<std::future<void>> futures;
      int seq = (to - from) < local_window_size ? to - from : local_window_size;
      if (!is_causal)
        seq = to - from;
      futures.reserve(seq);

      for (int i = 0; i < seq; ++i) {
        futures.push_back(pool.submit_task([=]() {
          size_t start_idx;
          if (is_causal) {
            start_idx =
              calc_attn_index(to - seq + i) - calc_attn_index(to - seq);
          } else {
            start_idx = i * to;
          }
          const _FP16 *input =
            in.getData<_FP16>() + start_idx * num_cache_head * gqa_size;
          _FP16 *out = output.getData<_FP16>() +
                       i * (num_cache_head * gqa_size * head_dim);
          int row_num = is_causal ? (to - seq + i) : to - 1;
          nntrainer::compute_fp16vcache_transposed(
            row_num, input, vcache.getData<_FP16>(), out, num_cache_head,
            gqa_size, head_dim, local_window_size);
        }));
      }
      for (auto &fut : futures)
        fut.get();
    } else {
      // Single token processing (common during generation)
      // Parallelize over KV heads for decoding since Q direction is always 1
      int row_num = to - 1;

      // Use OpenMP for lower overhead parallelization during decoding
      const _FP16 *in_data = in.getData<_FP16>();
      const _FP16 *vcache_data = vcache.getData<_FP16>();
      _FP16 *output_data = output.getData<_FP16>();

#pragma omp parallel for schedule(static)
      for (int head_kv = 0; head_kv < num_cache_head; ++head_kv) {
        nntrainer::compute_fp16vcache_transposed(
          row_num, in_data, vcache_data, output_data, num_cache_head, gqa_size,
          head_dim, local_window_size, head_kv, head_kv + 1);
      }
    }
#else
    NNTR_THROW_IF(true, std::invalid_argument) << "enable-fp16 is not set!";
#endif
  }
}

void MHACoreLayer::setBatch(nntrainer::RunLayerContext &context,
                            unsigned int batch) {

  const float dropout_rate =
    std::get<nntrainer::props::DropOutRate>(mha_core_props).get();
  context.updateTensor(tensor_idx[AttentionParams::cache_key], batch);
  context.updateTensor(tensor_idx[AttentionParams::cache_value], batch);
  // context.updateTensor(tensor_idx[AttentionParams::attention_weight], batch);
  if (dropout_rate > epsilon) {
    context.updateTensor(tensor_idx[AttentionParams::dropout_mask], batch);
  }
}

void MHACoreLayer::updateTensorsByInputDimensions(
  nntrainer::RunLayerContext &context,
  std::vector<nntrainer::TensorDim> input_dimensions) {
  unsigned int height = input_dimensions[0].height();
  unsigned int &max_timestep =
    std::get<nntrainer::props::MaxTimestep>(mha_core_props).get();
  unsigned int &max_new_tokens =
    std::get<props::MaxNewTokens>(mha_core_props).get();
  max_position_embeddings =
    std::get<props::MaxPositionEmbeddings>(mha_core_props).get();
  max_timestep = height + max_new_tokens;

  ml::train::TensorDim kv_dim = input_dimensions[0];
  kv_dim.width(kv_dim.width() / (num_heads_Q / num_heads_KV));

  ml::train::TensorDim kv_cache_dim = kv_dim;
#ifdef ENABLE_FP16
  kv_cache_dim.setDataType(ml::train::TensorDim::DataType::FP16);
#else
  kv_cache_dim.setDataType(ml::train::TensorDim::DataType::UINT16);
#endif
  kv_cache_dim.height(max_timestep);

  context.updateInput(INOUT_INDEX::QUERY, input_dimensions[0]);
  context.updateInput(INOUT_INDEX::KEY, kv_dim);
  context.updateInput(INOUT_INDEX::VALUE, kv_dim);
  context.updateOutput(0, input_dimensions[0]);

  context.updateTensor(tensor_idx[AttentionParams::cache_key], kv_cache_dim);
  context.updateTensor(tensor_idx[AttentionParams::cache_value], kv_cache_dim);
}

void MHACoreLayer::calcDerivative(nntrainer::RunLayerContext &context) {}

void MHACoreLayer::calcGradient(nntrainer::RunLayerContext &context) {}

void MHACoreLayer::exportTo(nntrainer::Exporter &exporter,
                            const ml::train::ExportMethods &method) const {
  LayerImpl::exportTo(exporter, method);
  exporter.saveResult(mha_core_props, method, this);
}

void MHACoreLayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, mha_core_props);
  LayerImpl::setProperty(remain_props);
}

size_t MHACoreLayer::calc_attn_index(size_t i) { return (i * (i + 1)) / 2; };

#ifdef PLUGGABLE

nntrainer::Layer *create_mha_core_layer() {
  auto layer = new MHACoreLayer();
  return layer;
}

void destroy_mha_core_layer(nntrainer::Layer *layer) { delete layer; }

extern "C" {
nntrainer::LayerPluggable ml_train_layer_pluggable{create_mha_core_layer,
                                                   destroy_mha_core_layer};
}

#endif
} // namespace causallm
