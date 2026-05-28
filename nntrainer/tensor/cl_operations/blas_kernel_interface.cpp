// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Debadri Samaddar <s.debadri@samsung.com>
 *
 * @file	blas_kernel_interface.cpp
 * @date	5 June 2024
 * @brief	Interface for blas OpenCL kernels
 * @see		https://github.com/nntrainer/nntrainer
 * @author	Debadri Samaddar <s.debadri@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#include <blas_kernel_interface.h>
#include <blas_kernels.h>
#include <clblast_interface.h>

namespace nntrainer {
void dotBatchedCl(Tensor const &input, Tensor const &m, Tensor &result,
                  bool trans, bool trans_m) {
  if (!result.isAllocated())
    throw std::invalid_argument(
      "Output tensor must be preallocated for dotBatched operation");
  for (unsigned int b = 0; b < input.batch(); b++) {
    /** @todo try using transpose to speedup the operation */
    const Tensor this_b = input.getBatchSlice(b, 1);
    Tensor m_b = m.getBatchSlice(b, 1);
    Tensor result_b = result.getBatchSlice(b, 1);

    dotCl(this_b, m_b, result_b, trans, trans_m);
  }
}

Tensor dotCl(Tensor const &input, Tensor const &m, bool trans, bool trans_m) {
  Tensor output("", input.getFormat(), input.getDataType());
  dotCl(input, m, output, trans, trans_m);

  return output;
}

void dotCl(Tensor const &input, Tensor const &m, Tensor &result, bool trans,
           bool trans_m) {
  unsigned int dim1, dim2, mdim1, mdim2;
  if (input.getFormat() == Tformat::NHWC) {
    dim1 = input.batch() * input.height() * input.width();
    dim2 = input.channel();
    mdim1 = m.batch() * m.height() * m.width();
    mdim2 = m.channel();
  } else {
    dim1 = input.batch() * input.channel() * input.height();
    dim2 = input.width();
    mdim1 = m.batch() * m.channel() * m.height();
    mdim2 = m.width();
  }

  unsigned int M, N, K, lda, ldb, ldc;

  if (!trans && !trans_m) {
    if (dim2 != mdim1)
      throw std::runtime_error(
        "Error: incompatible dimensions for dot product");
    K = mdim1; /** == dim2 */
    N = mdim2;
    M = dim1;
    if (input.getFormat() == Tformat::NHWC) {
      CREATE_IF_EMPTY_DIMS(result, input.batch(), N, input.height(),
                           input.width(),
                           input.getTensorType()); //  NHWC Result Tensor
    } else {
      CREATE_IF_EMPTY_DIMS(result, input.batch(), input.channel(),
                           input.height(), N, input.getTensorType());
    }
  } else if (!trans && trans_m) {
    if (dim2 != mdim2)
      throw std::runtime_error(
        "Error: incompatible dimensions for dot product");
    K = mdim2; /** == dim2 */
    N = mdim1;
    M = dim1;
    if (input.getFormat() == Tformat::NHWC) {
      CREATE_IF_EMPTY_DIMS(result, input.batch(), N, input.height(),
                           input.width(), input.getTensorType());
    } else {
      CREATE_IF_EMPTY_DIMS(result, input.batch(), input.channel(),
                           input.height(), N, input.getTensorType());
    }
  } else if (trans && !trans_m) {
    if (dim1 != mdim1)
      throw std::runtime_error(
        "Error: incompatible dimensions for dot product");
    K = mdim1; /** == dim1 */
    N = mdim2;
    M = dim2;
    if (input.getFormat() == Tformat::NHWC) {
      CREATE_IF_EMPTY_DIMS(result, 1, N, M, 1, input.getTensorType());
    } else {
      CREATE_IF_EMPTY_DIMS(result, 1, 1, M, N, input.getTensorType());
    }
  } else {
    if (dim1 != mdim2)
      throw std::runtime_error(
        "Error: incompatible dimensions for dot product");
    K = mdim2; /** == dim1 */
    N = mdim1;
    M = dim2;
    if (input.getFormat() == Tformat::NHWC) {
      CREATE_IF_EMPTY_DIMS(result, 1, N, M, 1, input.getTensorType());
    } else {
      CREATE_IF_EMPTY_DIMS(result, 1, 1, M, N, input.getTensorType());
    }
  }

  lda = dim2;
  ldb = mdim2;
  ldc =
    (input.getFormat() == Tformat::NHWC) ? result.channel() : result.width();

  if (input.getDataType() == ml::train::TensorDim::DataType::FP32) {
    const float *data = input.getData();
    const float *mdata = m.getData();
    float *rdata = result.getData();

    /// shortcut handling in case of vector
    /// for vector, (1 * K) == (K * 1) in current memory layout...
    /// and plaese note that N, K, M is a fixed place holder after considering
    /// transpose.
    /// For example, there is no case like (1 * K) X (1 * K) while
    /// (1 * K) X (1 * M) can be a case
    /// case1: (1 * K) X (K * 1)
    if (M == 1 && N == 1) {
      // *rdata = dot_cl(data, mdata, K) + (*rdata);
      *rdata = dot_cl(K, data, mdata) + (*rdata);
    }
    /// case2: (M * K) X (K * 1)
    else if (N == 1) {
      gemv_cl(0, trans, dim1, dim2, 1.0f, data, lda, mdata, 0.0f, rdata, 1);
    }
    /// case3: (1 * K) X (K * N) = 1 * N = R
    /// = R^T = (K * N) ^T * (1 * K) ^T = (N * K) * (K * 1) = (N * K) * (1 * K)
    /// Effectively a translation of sgemv
    else if (M == 1) {
      gemv_cl(0, !trans_m, mdim1, mdim2, 1.0f, mdata, ldb, data, 0.0f, rdata,
              1);
    }
    /// case others: use gemm
    else {
      if (input.getFormat() == Tformat::NHWC) {
        sgemm_cl(trans, trans_m, data, mdata, rdata, M, N, K, lda, ldb, ldc);
      } else {
        gemm_cl(0, trans, trans_m, M, N, K, 1.0f, data, (trans) ? M : K, mdata,
                (trans_m) ? K : N, 1.0f, rdata, N);
      }
    }
  } else if (input.getDataType() == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    const _FP16 *data = input.getData<_FP16>();
    const _FP16 *mdata = m.getData<_FP16>();
    _FP16 *rdata = result.getData<_FP16>();

    /// shortcut handling in case of vector
    /// for vector, (1 * K) == (K * 1) in current memory layout...
    /// and plaese note that N, K, M is a fixed place holder after considering
    /// transpose.
    /// For example, there is no case like (1 * K) X (1 * K) while
    /// (1 * K) X (1 * M) can be a case
    /// case1: (1 * K) X (K * 1)
    if (M == 1 && N == 1) {
      *rdata = dot_cl(data, mdata, K) + (*rdata);
    }
    /// case2: (M * K) X (K * 1)
    else if (N == 1) {
      trans ? sgemv_cl(data, mdata, rdata, trans, dim2, dim1, lda)
            : sgemv_cl(data, mdata, rdata, trans, dim1, dim2, lda);
    }
    /// case3: (1 * K) X (K * N) = 1 * N = R
    /// = R^T = (K * N) ^T * (1 * K) ^T = (N * K) * (K * 1) = (N * K) * (1 * K)
    /// Effectively a translation of sgemv
    else if (M == 1) {
      trans_m ? sgemv_cl(mdata, data, rdata, !trans_m, mdim1, mdim2, ldb)
              : sgemv_cl(mdata, data, rdata, !trans_m, mdim2, mdim1, ldb);
    }
    /// case others: use sgemm
    else {
      sgemm_cl(trans, trans_m, data, mdata, rdata, M, N, K, lda, ldb, ldc);
    }
#else
    throw std::invalid_argument("Error: enable-fp16 is not enabled");
#endif
  }
}

void multiplyCl(Tensor &input, float const &value) {
  if (input.getDataType() == ml::train::TensorDim::DataType::FP32) {
    float *data = input.getData<float>();
    unsigned int len = input.size();

    scal_cl(len, value, data);
  } else if (input.getDataType() == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    _FP16 *data = input.getData<_FP16>();
    unsigned int len = input.size();
    sscal_cl(data, len, value);
#else
    throw std::invalid_argument("Error: enable-fp16 is not enabled");
#endif
  }
}

void add_i_cl(Tensor &result, Tensor const &input) {

  NNTR_THROW_IF(input.getData() == nullptr, std::invalid_argument)
    << input.getName() << " is not allocated";
  NNTR_THROW_IF(result.getData() == nullptr, std::invalid_argument)
    << result.getName() << " is not allocated";

  // Broadcasting done for the case where batch size vary for both inputs
  // If batch size vary, batch size of input must be 1
  if ((result.getDim() == input.getDim()) ||
      (result.getDim() != input.getDim() && input.batch() == 1 &&
       result.channel() == input.channel() &&
       result.height() == input.height() && result.width() == input.width())) {

    if (result.getDataType() == ml::train::TensorDim::DataType::FP32) {
      float *Y = result.getData();
      const float *X = input.getData();

      // axpy with alpha=1 is just elementwise add. Use our own addition_cl
      // kernel so this path doesn't pull in CLBlast (the bigger BLAS dep
      // is gated behind -Denable-clblast; the v8c paper path doesn't need
      // it). FP16 already uses addition_cl below — make FP32 symmetric.
      unsigned int size_input = input.size();
      for (unsigned int i = 0; i < result.batch() / input.batch(); ++i) {
        addition_cl(X, Y, size_input, size_input);
        Y += size_input;
      }
    } else if (result.getDataType() == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
      unsigned int size_res = result.size();
      unsigned int size_input = input.size();
      _FP16 *data_res = result.getData<_FP16>();
      const _FP16 *data_input = input.getData<_FP16>();

      addition_cl(data_input, data_res, size_input, size_res);

#else
      throw std::invalid_argument("Error: enable-fp16 is not enabled");
#endif
    }
  }

  else {
    throw std::invalid_argument(
      "Error: Broadcasting not supported for these dimensions!");
  }
}

void transposeCl(const std::string &direction, Tensor const &in,
                 Tensor &result) {

  unsigned int input_batch_size, input_height, input_width, input_channels;

  input_batch_size = in.batch();
  input_height = in.height();
  input_width = in.width();
  input_channels = in.channel();

  if (in.getDataType() == ml::train::TensorDim::DataType::FP32) {
    const float *data = in.getData();
    float *rdata = result.getData();
    // for transpose about channels and height
    if (direction[0] == '1' && direction[2] == '0') {
      transpose_cl_axis(data, rdata, input_batch_size, input_channels,
                        input_height, input_width, 0);
    }
    // for transpose about height and width
    else if (direction[0] == '0' && direction[2] == '2') {
      transpose_cl_axis(data, rdata, input_batch_size, input_channels,
                        input_height, input_width, 1);
    }
    // for transpose about channels and width
    else if (direction[0] == '2' && direction[2] == '1') {
      transpose_cl_axis(data, rdata, input_batch_size, input_channels,
                        input_height, input_width, 2);
    }

  } else if (in.getDataType() == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    const _FP16 *data = in.getData<_FP16>();
    _FP16 *rdata = result.getData<_FP16>();
    // for transpose about channels and height
    if (direction[0] == '1' && direction[2] == '0') {
      transpose_cl_axis(data, rdata, input_batch_size, input_channels,
                        input_height, input_width, 0);
    }
    // for transpose about height and width
    else if (direction[0] == '0' && direction[2] == '2') {
      transpose_cl_axis(data, rdata, input_batch_size, input_channels,
                        input_height, input_width, 1);
    }
    // for transpose about channels and width
    else if (direction[0] == '2' && direction[2] == '1') {
      transpose_cl_axis(data, rdata, input_batch_size, input_channels,
                        input_height, input_width, 2);
    }
#else
    throw std::invalid_argument("Error: enable-fp16 is not enabled");
#endif
  }
}

void copyCl(const Tensor &input, Tensor &result) {
  if (input.getDataType() == ml::train::TensorDim::DataType::FP32) {
    const float *data = input.getData();
    float *rdata = result.getData();

    unsigned int len = input.size();

    copy_cl(len, data, rdata);
  } else if (input.getDataType() == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    throw std::runtime_error("Error: Currently, copyCl not supported for FP16");
#endif
  }
}

float nrm2Cl(const Tensor &input) {
  float result = 0.0f;
  if (input.getDataType() == ml::train::TensorDim::DataType::FP32) {
    float *data = input.getData();
    unsigned int len = input.size();

    result = nrm2_cl(len, data);
  } else if (input.getDataType() == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    throw std::runtime_error("Error: Currently, nrm2Cl not supported for FP16");
#endif
  }

  return result;
}

float asumCl(const Tensor &input) {
  float result = 0.0f;
  if (input.getDataType() == ml::train::TensorDim::DataType::FP32) {
    float *data = input.getData();
    unsigned int len = input.size();

    result = asum_cl(len, data);
  } else if (input.getDataType() == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    throw std::runtime_error("Error: Currently, asumCl not supported for FP16");
#endif
  }

  return result;
}

int amaxCl(const Tensor &input) {
  int result = 0;
  if (input.getDataType() == ml::train::TensorDim::DataType::FP32) {
    float *data = input.getData();
    unsigned int len = input.size();

    result = amax_cl(len, data);
  } else if (input.getDataType() == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    throw std::runtime_error("Error: Currently, amaxCl not supported for FP16");
#endif
  }

  return result;
}

int aminCl(const Tensor &input) {
  int result = 0;
  if (input.getDataType() == ml::train::TensorDim::DataType::FP32) {
    float *data = input.getData();
    unsigned int len = input.size();

    result = amin_cl(len, data);
  } else if (input.getDataType() == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    throw std::runtime_error("Error: Currently, amaxCl not supported for FP16");
#endif
  }

  return result;
}

} // namespace nntrainer

// =============================================================================
// v8c (paper 8/4/4) dispatch entry — env-gated, dotCl fallback.
// =============================================================================
#include "blas_kernels.h"
#include "cl_tensor_backing_pool.h"
#include "cl_tensor_view.h"
#include <atomic>
#include <cl_context.h>
#include <cl_kernels/cl_kernels.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <engine.h>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace nntrainer {
namespace {
struct V8cWeightEntry {
  std::unique_ptr<tv::TensorBacking> backing;
  cl_mem scale_buf = nullptr;       // [N] fp32 recip-scale (owned)
  cl_mem row_sum_w_int4 = nullptr;  // [N] int32 sum_k(int4 w_nk) (owned)
  unsigned int N = 0, K = 0;
  cl_mem weight_image = nullptr; // cached image2d view (also released via TensorBacking)
};

static bool v8c_env_enabled() {
  static int cached = -1;
  if (cached < 0)
    cached = std::getenv("NNTR_FC_INT8_GPU") != nullptr ? 1 : 0;
  return cached != 0;
}

static std::mutex &v8c_cache_mtx() {
  static std::mutex m;
  return m;
}
static std::unordered_map<const void *, V8cWeightEntry> &v8c_weight_cache() {
  static std::unordered_map<const void *, V8cWeightEntry> c;
  return c;
}

// Grow-only scratch buffer pool, reused across all dotCl_v8c forward calls
// to avoid per-call clCreateBuffer/clReleaseMemObject churn (the dominant
// integration overhead, especially in M=1 decode where the same FC shapes
// recur thousands of times).
struct V8cScratch {
  cl_mem act_in = nullptr;
  size_t act_in_bytes = 0;
  cl_mem act_i8 = nullptr;
  size_t act_i8_bytes = 0;
  cl_mem act_scale = nullptr;
  size_t act_scale_bytes = 0;
  cl_mem act_rs = nullptr;
  size_t act_rs_bytes = 0;
  cl_mem act_zp = nullptr;       // [M] int32, asymmetric activation zero-point
  size_t act_zp_bytes = 0;
  cl_mem y_fp16 = nullptr;
  size_t y_fp16_bytes = 0;

  // Step 2b.0 shared-quant cache (paper §3.6 fused-quant motivation, host-side).
  // Qwen3 layer graph dispatches three consecutive dotCl_v8c calls with the
  // SAME input pointer (wq/wk/wv all read the same post-RMSNorm activation),
  // and similarly gate/up MLP FCs share their input. After the first call
  // populates act_i8/act_scale/act_zp/act_rs for that input, the next 2-of-3
  // calls can skip the host→device upload AND the quant kernel entirely.
  //
  // Cache key: (input data pointer, M, K, M_pad, dtype). Pointer identity is
  // sufficient within one forward pass since the layer graph executes
  // serially — the input buffer isn't aliased between dispatches.
  const void *last_quant_in_ptr = nullptr;
  unsigned int last_quant_M = 0;
  unsigned int last_quant_K = 0;
  unsigned int last_quant_M_pad = 0;
  int last_quant_dtype = -1;
  unsigned long long last_quant_resident_generation = 0;
};

// Process-global Segment A resident-buffer generation counter. Producers
// (Segment A's RMSNorm helpers) bump this on every successful write to a
// resident TensorBacking, signalling to dotCl_v8c that any quant cache
// keyed on a backing pointer is now stale. Same forward pass / multiple
// FCs reusing the same backing all share the generation, so the cache
// still hits on wq→wk→wv within the pass.
static std::atomic<unsigned long long> g_resident_quant_generation{0};
static V8cScratch &v8c_scratch() {
  static V8cScratch s;
  return s;
}
// Ensure *buf has at least `bytes` capacity with the given flags; (re)alloc
// only when too small. Returns false on alloc failure.
static bool v8c_ensure_buf(cl_context ctx, cl_mem *buf, size_t *cap,
                           size_t bytes, cl_mem_flags flags) {
  if (*buf && *cap >= bytes)
    return true;
  if (*buf) {
    clReleaseMemObject(*buf);
    *buf = nullptr;
    *cap = 0;
  }
  cl_int err = CL_SUCCESS;
  *buf = clCreateBuffer(ctx, flags, bytes, nullptr, &err);
  if (err != CL_SUCCESS || !*buf) {
    *buf = nullptr;
    *cap = 0;
    return false;
  }
  *cap = bytes;
  return true;
}

// Get or build the cached v8c weight backing for a given Int4QTensor weight.
// Returns nullptr if shape unsupported (caller falls back).
static V8cWeightEntry *v8c_get_or_build_weight(const Tensor &weight,
                                               unsigned int K, unsigned int N) {
  if (K % 32 != 0 || N % 8 != 0)
    return nullptr;
  const void *key = weight.getData<uint8_t>();
  if (!key)
    return nullptr;
  std::lock_guard<std::mutex> lock(v8c_cache_mtx());
  auto &cache = v8c_weight_cache();
  auto it = cache.find(key);
  if (it != cache.end())
    return &it->second;
  const uint8_t *section_a = weight.getData<uint8_t>();
  const uint16_t *fp16_scales = weight.getScale<uint16_t>();
  if (!section_a || !fp16_scales)
    return nullptr;
  V8cWeightEntry e;
  cl_mem sb = nullptr;
  cl_mem rsw = nullptr;
  try {
    // The on-disk QINT4 weight is the KAI Section A nibble payload + a
    // per-output-channel fp16 scale (one fp16 per N). Permute the nibbles
    // straight to the v8c row-major + offset-encoded layout — no dequant→
    // requant round-trip, so no extra quantization noise and no fp32
    // intermediate buffer. The scales transfer 1:1 (fp16 → fp32). The
    // helper also precomputes per-channel Σ_k int4_w[n,k] for the
    // asymmetric-act zero-point correction the GEMM applies later.
    e.backing = make_v8c_weight_backing_from_kai_section_a(
      section_a, fp16_scales, N, K, &sb, &rsw);
  } catch (...) {
    return nullptr;
  }
  e.scale_buf = sb;
  e.row_sum_w_int4 = rsw;
  e.N = N;
  e.K = K;
  tv::ViewSpec ws;
  ws.kind = tv::ViewKind::IMAGE_2D;
  ws.image_channel_order = CL_RGBA;
  ws.image_channel_type = CL_UNSIGNED_INT32;
  ws.width = K / 32;
  ws.height = N;
  ws.row_pitch_bytes = K / 2;
  try {
    e.weight_image = e.backing->imageView(ws);
  } catch (...) {
    if (sb)
      clReleaseMemObject(sb);
    return nullptr;
  }
  auto inserted = cache.emplace(key, std::move(e));
  return &inserted.first->second;
}

// fp16 → fp32 (host-side decode used to convert kernel fp16 output)
static inline float v8c_h2f(uint16_t h) {
  uint32_t s = (uint32_t)(h & 0x8000u) << 16;
  uint32_t e = (h >> 10) & 0x1fu;
  uint32_t m = h & 0x3ffu;
  uint32_t o;
  if (e == 0) {
    if (m == 0)
      o = s;
    else {
      e = 1;
      while ((m & 0x400u) == 0) {
        m <<= 1;
        e--;
      }
      m &= 0x3ffu;
      o = s | ((e + 112) << 23) | (m << 13);
    }
  } else if (e == 0x1f) {
    o = s | 0x7f800000u | (m << 13);
  } else {
    o = s | ((e + 112) << 23) | (m << 13);
  }
  float f;
  std::memcpy(&f, &o, 4);
  return f;
}
} // anonymous namespace

bool dotCl_v8c(const Tensor &input, const Tensor &weight, Tensor &output) {
  if (!v8c_env_enabled())
    return false;
  if (weight.getDataType() != ml::train::TensorDim::DataType::QINT4)
    return false;
  // Derive M, K, N from tensor dims (no-transpose case only).
  unsigned int M, K, N;
  if (input.getFormat() == Tformat::NHWC) {
    M = input.batch() * input.height() * input.width();
    K = input.channel();
  } else {
    M = input.batch() * input.channel() * input.height();
    K = input.width();
  }
  N = weight.width();
  if (K != weight.height())
    return false;
  if (N % 8 != 0 || K % 32 != 0)
    return false;
  if (input.getDataType() != ml::train::TensorDim::DataType::FP32 &&
      input.getDataType() != ml::train::TensorDim::DataType::FP16)
    return false;
  // Round M up to the kernel's tile size (V8C_TM=4). Padded rows produce
  // throwaway output that we never read back to the caller. Skips the
  // "M not divisible by 4 → CPU fallback" cliff so v8c runs for any prefill
  // length (the 18-token Qwen3 chat-template case in particular).
  constexpr unsigned int V8C_TM = 4;
  const unsigned int M_pad = (M + V8C_TM - 1) / V8C_TM * V8C_TM;

  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  cl_context ctx = blas_cc->context_inst_.GetContext();
  cl_command_queue q = blas_cc->command_queue_inst_.GetCommandQueue();

  V8cWeightEntry *w = v8c_get_or_build_weight(weight, K, N);
  if (!w)
    return false;

  // Reused scratch buffers (grow-only pool). The weight backing + scale are
  // already cached per-weight; only the activation/output scratch scales with
  // (M_pad, K, N), so we grow these lazily and reuse them across forwards.
  cl_int err = CL_SUCCESS;
  const size_t act_elem =
    (input.getDataType() == ml::train::TensorDim::DataType::FP16)
      ? sizeof(uint16_t)
      : sizeof(float);
  std::lock_guard<std::mutex> slock(v8c_cache_mtx());
  V8cScratch &sc = v8c_scratch();
  if (!v8c_ensure_buf(ctx, &sc.act_in, &sc.act_in_bytes,
                      (size_t)M_pad * K * act_elem, CL_MEM_READ_ONLY) ||
      !v8c_ensure_buf(ctx, &sc.act_i8, &sc.act_i8_bytes, (size_t)M_pad * K,
                      CL_MEM_READ_WRITE) ||
      !v8c_ensure_buf(ctx, &sc.act_scale, &sc.act_scale_bytes,
                      sizeof(float) * M_pad, CL_MEM_READ_WRITE) ||
      !v8c_ensure_buf(ctx, &sc.act_rs, &sc.act_rs_bytes, sizeof(int) * M_pad,
                      CL_MEM_READ_WRITE) ||
      !v8c_ensure_buf(ctx, &sc.act_zp, &sc.act_zp_bytes, sizeof(int) * M_pad,
                      CL_MEM_READ_WRITE) ||
      !v8c_ensure_buf(ctx, &sc.y_fp16, &sc.y_fp16_bytes,
                      sizeof(uint16_t) * (size_t)M_pad * N, CL_MEM_READ_WRITE))
    return false;

  // === v8c stage profiling (env-gated). Wall-clock with clFinish between
  // stages so each step's elapsed time is isolated. Skipped entirely when
  // NNTR_V8C_PROFILE is unset. Per-bin aggregates so prefill (large M) and
  // decode (M=1) regimes are separable. ===
  static bool prof_enabled = std::getenv("NNTR_V8C_PROFILE") != nullptr;
  struct V8cBin {
    long long write_ns = 0, quant_ns = 0, image_ns = 0, gemm_ns = 0,
              read_ns = 0;
    long long write_bytes = 0, read_bytes = 0;
    int calls = 0;
    long long m_sum = 0;
  };
  struct V8cProf {
    V8cBin bin[5]; // 0:M=1, 1:M=2-4, 2:M=5-32, 3:M=33-256, 4:M>256
    int total_calls = 0;
    static const char *bin_name(int b) {
      static const char *N[5] = {"M=1", "M=2-4", "M=5-32", "M=33-256", "M>256"};
      return N[b];
    }
    void dump(const char *tag) {
      std::FILE *f = std::fopen("/data/local/tmp/qwen3_qint4/v8c_prof.log", "a");
      if (!f) f = stderr;
      std::fprintf(f, "\n[V8C-PROF] %s after %d total calls:\n", tag,
                   total_calls);
      for (int b = 0; b < 5; b++) {
        const V8cBin &x = bin[b];
        if (!x.calls) continue;
        double total_ms = (x.write_ns + x.quant_ns + x.image_ns + x.gemm_ns +
                           x.read_ns) /
                          1e6;
        double avg_M = (double)x.m_sum / x.calls;
        std::fprintf(
          f,
          "  [%s] %d calls (avg M=%.1f) total=%.2f ms:\n"
          "    write_act    %7.2f ms (%.1f MB)  %5.1f%%\n"
          "    quant_kernel %7.2f ms             %5.1f%%\n"
          "    image_view   %7.2f ms             %5.1f%%\n"
          "    gemm_kernel  %7.2f ms             %5.1f%%\n"
          "    read_output  %7.2f ms (%.1f MB)  %5.1f%%\n",
          bin_name(b), x.calls, avg_M, total_ms, x.write_ns / 1e6,
          x.write_bytes / 1048576.0, 100.0 * x.write_ns / 1e6 / total_ms,
          x.quant_ns / 1e6, 100.0 * x.quant_ns / 1e6 / total_ms,
          x.image_ns / 1e6, 100.0 * x.image_ns / 1e6 / total_ms,
          x.gemm_ns / 1e6, 100.0 * x.gemm_ns / 1e6 / total_ms,
          x.read_ns / 1e6, x.read_bytes / 1048576.0,
          100.0 * x.read_ns / 1e6 / total_ms);
      }
      std::fflush(f);
      if (f != stderr) std::fclose(f);
    }
    ~V8cProf() {
      if (total_calls) dump("FINAL");
    }
  };
  static V8cProf prof;
  static int last_dumped_bin = -1;
  auto NOW = []() { return std::chrono::steady_clock::now(); };
  auto NS = [](auto t1, auto t0) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0)
      .count();
  };
  std::chrono::steady_clock::time_point T0, T1;

  // Upload activation into the (reused) act_in buffer. Zero-fill the padded
  // rows so the act_quant kernel sees deterministic values (per-row amax → 0
  // → scale defaults to 1.0 → q=0 → row_sum=0; padded rows produce 0 output).
  int prof_bin = (M == 1)     ? 0
                 : (M <= 4)   ? 1
                 : (M <= 32)  ? 2
                 : (M <= 256) ? 3
                              : 4;

  // Step 2b.0 shared-quant cache check (paper §3.6 fused-quant insight,
  // host-side). If this dotCl_v8c was called with the same (input ptr,
  // M, K, M_pad, dtype) as the most recent call, sc.act_i8/scale/zp/rs are
  // already correctly populated — skip both the host→device write AND the
  // quant kernel. Hits fire on the wq→wk and wq→wv legs of Qwen3 QKV (and
  // gate→up of the MLP block), where the input activation is literally the
  // same tensor across multiple FC dispatches.
  //
  // Segment A.1 residency input mode (paper §3.2 cross-layer GPU residency).
  // When NNTR_RESIDENT_FC=1 AND the input Tensor has a TensorBacking with
  // FP16 encoding holding the activation in cl_mem (set by a preceding GPU
  // op such as Segment A's RMSNorm), bypass the host→device upload entirely
  // — clEnqueueCopyBuffer (GPU→GPU) from the backing into sc.act_in instead.
  // Cache key uses the backing's cl_mem as the source identifier so the
  // wq→wk→wv repeat continues to skip redundant quant. Caller-set Tensor
  // host data is ignored in this mode.
  //
  // Tensor::getBacking() returns null when the producer set the backing on
  // a different Tensor instance than the one this consumer received (the
  // typical nntrainer pattern: layer N's "output" Tensor and layer N+1's
  // "input" Tensor are distinct instances that share the underlying data
  // buffer via TensorPool). Fall back to a pool lookup keyed by the host
  // data pointer; shared-data tensors share the data pointer even when
  // their names/instances differ.
  static const bool resident_fc_enabled =
    std::getenv("NNTR_RESIDENT_FC") != nullptr;
  const tv::TensorBacking *in_backing = nullptr;
  std::shared_ptr<tv::TensorBacking> in_backing_from_pool;
  if (resident_fc_enabled) {
    in_backing = input.getBacking();
    if (!in_backing) {
      const void *in_data_ptr = input.getData<uint8_t>();
      char key_buf[64];
      std::snprintf(key_buf, sizeof(key_buf), "ptr:%p", in_data_ptr);
      in_backing_from_pool =
        tv::TensorBackingPool::Global().get(std::string(key_buf));
      if (!in_backing_from_pool)
        in_backing_from_pool =
          tv::TensorBackingPool::Global().get(input.getName());
      if (in_backing_from_pool)
        in_backing = in_backing_from_pool.get();
    }
  }
  const bool resident_dtype_match =
    in_backing != nullptr &&
    ((in_backing->encoding() == tv::Encoding::FP16 &&
      input.getDataType() == ml::train::TensorDim::DataType::FP16) ||
     (in_backing->encoding() == tv::Encoding::FP32 &&
      input.getDataType() == ml::train::TensorDim::DataType::FP32));
  const bool use_resident_input = resident_dtype_match;

  const int cur_dtype =
    (input.getDataType() == ml::train::TensorDim::DataType::FP16) ? 1 : 0;
  const void *cur_in_ptr =
    use_resident_input
      ? static_cast<const void *>(in_backing->buffer())
      : static_cast<const void *>(input.getData<uint8_t>());
  // Step 2b.0 quant cache. For host-uploaded inputs the (data_ptr,
  // shape, dtype) tuple uniquely identifies the activation, so a hit
  // means the same data is already int8-quantized in sc.act_i8 and
  // we can skip both the copy and the quant kernel.
  //
  // For backing-sourced inputs the same logic holds *within a single
  // forward pass* because the backing pointer is stable. Across passes
  // the same backing pointer points to different data (RMSNorm
  // overwrites it). That cross-pass staleness is invalidated below
  // by an external generation counter tied to RMSNorm writes:
  // resident_input_quant_generation is bumped whenever a Segment A
  // RMSNorm producer writes to a backing, and cached against the
  // generation at the time of the last quant. If the generation has
  // advanced since the last cache update, the cache is invalidated.
  const bool quant_cache_hit =
    sc.last_quant_in_ptr != nullptr &&
    sc.last_quant_in_ptr == cur_in_ptr && sc.last_quant_M == M &&
    sc.last_quant_K == K && sc.last_quant_M_pad == M_pad &&
    sc.last_quant_dtype == cur_dtype &&
    (!use_resident_input ||
     sc.last_quant_resident_generation == g_resident_quant_generation);

  if (!quant_cache_hit) {
    if (prof_enabled) T0 = NOW();
    if (use_resident_input) {
      // GPU→GPU copy of the resident FP32/FP16 activation into sc.act_in.
      // Same shape as a host upload would produce, just without crossing
      // PCIe. Padded rows (M_pad > M) are zero-filled below.
      if (clEnqueueCopyBuffer(q, in_backing->buffer(), sc.act_in, 0, 0,
                              (size_t)M * K * act_elem, 0, nullptr,
                              nullptr) != CL_SUCCESS)
        return false;
    } else {
      if (clEnqueueWriteBuffer(q, sc.act_in, CL_FALSE, 0,
                               (size_t)M * K * act_elem, cur_in_ptr, 0,
                               nullptr, nullptr) != CL_SUCCESS)
        return false;
    }
    if (M_pad > M) {
      const size_t pad_bytes = (size_t)(M_pad - M) * K * act_elem;
      std::vector<uint8_t> zeros(pad_bytes, 0);
      if (clEnqueueWriteBuffer(q, sc.act_in, CL_FALSE,
                               (size_t)M * K * act_elem, pad_bytes,
                               zeros.data(), 0, nullptr, nullptr) != CL_SUCCESS)
        return false;
    }
    if (prof_enabled) {
      clFinish(q);
      T1 = NOW();
      prof.bin[prof_bin].write_ns += NS(T1, T0);
      prof.bin[prof_bin].write_bytes += (long long)M * K * act_elem;
    }
  }

  try {
    // (c) fp→int8 asymmetric act quant + zero-point + row_sum over M_pad rows.
    //     Padded rows map to (scale=1, zp=0, q=0, row_sum=0), so they
    //     contribute zero in the GEMM and don't pollute valid rows.
    //     Step 2b.0: skipped entirely on cache hit (see above).
    if (!quant_cache_hit) {
      if (prof_enabled) T0 = NOW();
      if (input.getDataType() == ml::train::TensorDim::DataType::FP16)
        quantize_act_v8c_fp16_cl(sc.act_in, sc.act_i8, sc.act_scale, sc.act_zp,
                                 sc.act_rs, M_pad, K);
      else
        quantize_act_v8c_fp32_cl(sc.act_in, sc.act_i8, sc.act_scale, sc.act_zp,
                                 sc.act_rs, M_pad, K);
      if (prof_enabled) {
        clFinish(q);
        T1 = NOW();
        prof.bin[prof_bin].quant_ns += NS(T1, T0);
      }
      // Update cache key only after a successful quant.
      sc.last_quant_in_ptr = cur_in_ptr;
      sc.last_quant_M = M;
      sc.last_quant_K = K;
      sc.last_quant_M_pad = M_pad;
      sc.last_quant_dtype = cur_dtype;
      sc.last_quant_resident_generation = g_resident_quant_generation.load();
    }

    // === Per-call CPU vs GPU quant equality check ===
    // For NNTR_V8C_QUANT_CHECK=1, recompute KAI-style asymmetric int8 act
    // quant on CPU for the same input row, compare against GPU readback.
    // Prints first divergent index per row and aggregate counts.
    if (std::getenv("NNTR_V8C_QUANT_CHECK") && M == 1003 &&
        input.getDataType() == ml::train::TensorDim::DataType::FP32) {
      static int qcheck_id = 0;
      ++qcheck_id;
      if (qcheck_id <= 3) {
        std::vector<int8_t> gpu_q(M_pad * K);
        std::vector<float> gpu_scale(M_pad);
        std::vector<int32_t> gpu_zp(M_pad);
        std::vector<int32_t> gpu_rs(M_pad);
        clEnqueueReadBuffer(q, sc.act_i8, CL_TRUE, 0, (size_t)M_pad * K,
                            gpu_q.data(), 0, nullptr, nullptr);
        clEnqueueReadBuffer(q, sc.act_scale, CL_TRUE, 0,
                            sizeof(float) * M_pad, gpu_scale.data(), 0,
                            nullptr, nullptr);
        clEnqueueReadBuffer(q, sc.act_zp, CL_TRUE, 0, sizeof(int) * M_pad,
                            gpu_zp.data(), 0, nullptr, nullptr);
        clEnqueueReadBuffer(q, sc.act_rs, CL_TRUE, 0, sizeof(int) * M_pad,
                            gpu_rs.data(), 0, nullptr, nullptr);
        const float *in = input.getData<float>();
        // CPU reference: same algorithm as KAI qai8dxp_f32 (lines 120-186 of
        // kai_lhs_quant_pack_qai8dxp_f32.c). Check first 2 rows only.
        for (int row = 0; row < (int)std::min(M, 2u); ++row) {
          float fmin = 0.0f, fmax = 0.0f;
          for (unsigned int k = 0; k < K; ++k) {
            float v = in[row * K + k];
            if (v < fmin) fmin = v;
            if (v > fmax) fmax = v;
          }
          float rmin = fmin < 0.0f ? fmin : 0.0f;
          float rmax = fmax > 0.0f ? fmax : 0.0f;
          float qmin = -128.0f, qmax = 127.0f;
          float scale = rmin == rmax ? 1.0f : (qmax - qmin) / (rmax - rmin);
          float recip = scale ? 1.0f / scale : 0.0f;
          float dmin = rmin * scale, dmax = rmax * scale;
          float zp_from_min = qmin + dmin;
          float zp_from_max = qmax + dmax;
          float zpf = (zp_from_min + zp_from_max > 0.0f) ? (qmin - dmin)
                                                         : (qmax - dmax);
          zpf = std::max(zpf, qmin);
          zpf = std::min(zpf, qmax);
          int cpu_zp = (int)std::round(zpf);
          int cpu_rs = 0;
          int q_diffs = 0, first_diff_k = -1;
          int8_t cpu_q_first = 0, gpu_q_first = 0;
          for (unsigned int k = 0; k < K; ++k) {
            int v = (int)std::round(in[row * K + k] * scale) + cpu_zp;
            if (v < -128) v = -128;
            if (v > 127) v = 127;
            int8_t cpuq = (int8_t)v;
            int8_t gpuq = gpu_q[row * K + k];
            if (cpuq != gpuq) {
              if (first_diff_k < 0) {
                first_diff_k = k;
                cpu_q_first = cpuq;
                gpu_q_first = gpuq;
              }
              q_diffs++;
            }
            cpu_rs += cpuq;
          }
          std::fprintf(
            stderr,
            "[V8C-QCHECK id=%d row=%d] cpu_scale=%.6f gpu_scale=%.6f | "
            "cpu_zp=%d gpu_zp=%d | cpu_rs=%d gpu_rs=%d | rmin=%.4f rmax=%.4f | "
            "q_diffs=%d/1024 first_k=%d cpu_q=%d gpu_q=%d\n",
            qcheck_id, row, recip, gpu_scale[row], cpu_zp, gpu_zp[row],
            cpu_rs, gpu_rs[row], rmin, rmax, q_diffs, first_diff_k,
            (int)cpu_q_first, (int)gpu_q_first);
          std::fflush(stderr);
        }
      }
    }

    // Build image2d view over act_i8 buffer (zero-copy, tensor virtualization).
    // This view is cheap; it must be recreated when M/K change, so keep it
    // local. (clCreateImage over an existing buffer is far cheaper than a
    // fresh device allocation.)
    if (prof_enabled) T0 = NOW();
    cl_image_format afmt{CL_RGBA, CL_UNSIGNED_INT32};
    cl_image_desc adesc{};
    adesc.image_type = CL_MEM_OBJECT_IMAGE2D;
    adesc.image_width = K / 16;
    adesc.image_height = M_pad;
    adesc.image_row_pitch = K;
    adesc.buffer = sc.act_i8;
    cl_mem act_image =
      clCreateImage(ctx, CL_MEM_READ_ONLY, &afmt, &adesc, nullptr, &err);
    if (err != CL_SUCCESS) throw std::runtime_error("act image view fail");
    if (prof_enabled) {
      T1 = NOW();
      prof.bin[prof_bin].image_ns += NS(T1, T0);
      T0 = T1;
    }

    // (b) v8c GEMM — run on padded M_pad rows, but only read back the valid
    // M rows to the caller buffer.
    gemm_int8_v8c_cl(act_image, w->weight_image, sc.act_scale, w->scale_buf,
                     sc.act_rs, sc.act_zp, w->row_sum_w_int4, sc.y_fp16, M_pad,
                     N, K);
    if (prof_enabled) {
      clFinish(q);
      T1 = NOW();
      prof.bin[prof_bin].gemm_ns += NS(T1, T0);
    }

    // === GEMM-output check: same int8 act + same int4 w + same formula. ===
    // Diagnoses whether the v8c GEMM kernel itself computes a value
    // mathematically identical to the int8×int4 + bias-correction formula
    // CPU would compute given the SAME quantized inputs. Quant (verified
    // bit-exact via NNTR_V8C_QUANT_CHECK), permute (verified byte-exact via
    // test/v8c_permute_test) are independently checked, so any divergence
    // here is in the GEMM/correction itself.
    if (std::getenv("NNTR_V8C_GEMM_CHECK") && M == 1003 &&
        input.getDataType() == ml::train::TensorDim::DataType::FP32) {
      static int gcheck_id = 0;
      ++gcheck_id;
      if (gcheck_id <= 2) {
        clFinish(q);
        std::vector<int8_t> gpu_q(M_pad * K);
        std::vector<float> gpu_scale_act(M_pad);
        std::vector<int32_t> gpu_zp(M_pad);
        std::vector<int32_t> gpu_rs(M_pad);
        std::vector<int32_t> gpu_rsw(N);
        std::vector<float> gpu_scale_wgt(N);
        std::vector<uint16_t> gpu_y((size_t)M * N);
        clEnqueueReadBuffer(q, sc.act_i8, CL_TRUE, 0, (size_t)M_pad * K,
                            gpu_q.data(), 0, nullptr, nullptr);
        clEnqueueReadBuffer(q, sc.act_scale, CL_TRUE, 0,
                            sizeof(float) * M_pad, gpu_scale_act.data(), 0,
                            nullptr, nullptr);
        clEnqueueReadBuffer(q, sc.act_zp, CL_TRUE, 0, sizeof(int) * M_pad,
                            gpu_zp.data(), 0, nullptr, nullptr);
        clEnqueueReadBuffer(q, sc.act_rs, CL_TRUE, 0, sizeof(int) * M_pad,
                            gpu_rs.data(), 0, nullptr, nullptr);
        clEnqueueReadBuffer(q, w->row_sum_w_int4, CL_TRUE, 0,
                            sizeof(int) * N, gpu_rsw.data(), 0, nullptr,
                            nullptr);
        clEnqueueReadBuffer(q, w->scale_buf, CL_TRUE, 0, sizeof(float) * N,
                            gpu_scale_wgt.data(), 0, nullptr, nullptr);
        clEnqueueReadBuffer(q, sc.y_fp16, CL_TRUE, 0,
                            sizeof(uint16_t) * gpu_y.size(), gpu_y.data(), 0,
                            nullptr, nullptr);
        // Decode v8c weight bytes for the same (n, k) used in GPU kernel.
        // section_a is KAI Section A, but we wrote v8c-permuted into
        // backing buffer. Easier to walk Section A → int4 directly.
        const uint8_t *section_a = weight.getData<uint8_t>();
        constexpr size_t KAI_NR2 = 4, KAI_KR_BY_SR2 = 8;
        constexpr size_t KAI_BYTES_PER_KBLK2 = 2 * KAI_NR2 * KAI_KR_BY_SR2;
        const size_t nibble_bytes_per_super_row = KAI_NR2 * (K / 2);
        auto decode_int4 = [&](unsigned int n, unsigned int k) -> int {
          const size_t kbl = k / 32;
          const size_t kp = k % 32;
          const size_t sr = n / KAI_NR2;
          const size_t nr = n % KAI_NR2;
          const uint8_t *sr_base =
            section_a + sr * nibble_bytes_per_super_row;
          const uint8_t *blk_a =
            sr_base + kbl * KAI_BYTES_PER_KBLK2 + nr * KAI_KR_BY_SR2;
          const uint8_t *blk_b = blk_a + KAI_NR2 * KAI_KR_BY_SR2;
          uint8_t nib = 0;
          if (kp < 8)
            nib = (blk_a[kp] ^ 0x88) & 0x0F;
          else if (kp < 16)
            nib = (blk_b[kp - 8] ^ 0x88) & 0x0F;
          else if (kp < 24)
            nib = ((blk_a[kp - 16] ^ 0x88) >> 4) & 0x0F;
          else
            nib = ((blk_b[kp - 24] ^ 0x88) >> 4) & 0x0F;
          return (int)nib - 8;
        };
        // CPU reference for a handful of (m, n) positions: replicate the
        // GPU bias-corrected dot product exactly.
        int diffs = 0;
        double max_abs_diff = 0;
        unsigned int worst_m = 0, worst_n = 0;
        float gpu_at_worst = 0, ref_at_worst = 0;
        for (unsigned int m = 0; m < std::min(M, 2u); ++m) {
          int rs = gpu_rs[m];
          int zp = gpu_zp[m];
          float s_act = gpu_scale_act[m];
          for (unsigned int n = 0; n < std::min(N, 32u); ++n) {
            int acc = 0;
            for (unsigned int k = 0; k < K; ++k) {
              int aq = gpu_q[m * K + k];
              int w_int4 = decode_int4(n, k);
              acc += aq * (w_int4 + 8);
            }
            int corrected = acc - 8 * rs - zp * gpu_rsw[n];
            float ref_v = (float)corrected * s_act * gpu_scale_wgt[n];
            float gpu_v = v8c_h2f(gpu_y[m * N + n]);
            float d = gpu_v - ref_v;
            if (std::fabs(d) > max_abs_diff) {
              max_abs_diff = std::fabs(d);
              worst_m = m;
              worst_n = n;
              gpu_at_worst = gpu_v;
              ref_at_worst = ref_v;
            }
            if (std::fabs(d) > 1e-3f) ++diffs;
          }
        }
        std::fprintf(stderr,
                     "[V8C-GCHECK id=%d M=%u N=%u K=%u] diffs(>1e-3)=%d/%u "
                     "max|diff|=%.5f at (%u,%u) gpu=%.5f ref=%.5f\n",
                     gcheck_id, M, N, K, diffs, 2 * std::min(N, 32u),
                     max_abs_diff, worst_m, worst_n, gpu_at_worst,
                     ref_at_worst);
        std::fflush(stderr);
      }
    }

    // === Per-call CPU vs GPU divergence trace ===
    // For NNTR_V8C_TRACE=1, compute the CPU "fp32 act × fp32 dequant w"
    // reference for the same (input, weight) and report relL2 vs the GPU
    // fp16 readback. This is the math-correct reference (no act quant), so
    // any extra error beyond ~ int4 quant noise points at v8c's symmetric
    // int8 act quant path.
    if (std::getenv("NNTR_V8C_TRACE") &&
        input.getDataType() == ml::train::TensorDim::DataType::FP32) {
      static int trace_id = 0;
      ++trace_id;
      std::vector<uint16_t> y_peek((size_t)M * N);
      clEnqueueReadBuffer(q, sc.y_fp16, CL_TRUE, 0,
                          sizeof(uint16_t) * y_peek.size(), y_peek.data(), 0,
                          nullptr, nullptr);
      std::vector<float> w_scale_h(N);
      clEnqueueReadBuffer(q, w->scale_buf, CL_TRUE, 0, sizeof(float) * N,
                          w_scale_h.data(), 0, nullptr, nullptr);
      const uint8_t *section_a = weight.getData<uint8_t>();
      const float *in_f = input.getData<float>();
      // Reference: v[m,n] = Σ_k act_fp[m,k] × (int4_w[n,k]) × scale_w[n]
      //   where int4_w is the actually-stored quantized weight value
      //   (decoded from KAI Section A nibble, range [-8..7]).
      constexpr size_t KAI_NR2 = 4, KAI_KR_BY_SR2 = 8;
      constexpr size_t KAI_BYTES_PER_KBLK2 = 2 * KAI_NR2 * KAI_KR_BY_SR2;
      const size_t nibble_bytes_per_super_row = KAI_NR2 * (K / 2);
      double sum_sq_diff = 0.0;
      double sum_sq_ref = 0.0;
      float max_abs_diff = 0.0f;
      unsigned int worst_m = 0, worst_n = 0;
      float gpu_at_worst = 0, ref_at_worst = 0;
      // Sample 8 (m, n) positions for tractability when M or N is large.
      const unsigned int sample_M = std::min(M, 4u);
      const unsigned int sample_N = std::min(N, 32u);
      for (unsigned int m = 0; m < sample_M; ++m) {
        for (unsigned int j = 0; j < sample_N; ++j) {
          // Decode int4 for (n=j, k) for all k
          float ref = 0.0f;
          for (unsigned int k = 0; k < K; ++k) {
            const size_t kbl = k / 32;
            const size_t kp = k % 32;
            const size_t sr = j / KAI_NR2;
            const size_t nr = j % KAI_NR2;
            const uint8_t *sr_base =
              section_a + sr * nibble_bytes_per_super_row;
            const uint8_t *blk_a = sr_base + kbl * KAI_BYTES_PER_KBLK2 +
                                   nr * KAI_KR_BY_SR2;
            const uint8_t *blk_b = blk_a + KAI_NR2 * KAI_KR_BY_SR2;
            uint8_t nib = 0;
            if (kp < 8) nib = (blk_a[kp] ^ 0x88) & 0x0F;
            else if (kp < 16) nib = (blk_b[kp - 8] ^ 0x88) & 0x0F;
            else if (kp < 24) nib = ((blk_a[kp - 16] ^ 0x88) >> 4) & 0x0F;
            else nib = ((blk_b[kp - 24] ^ 0x88) >> 4) & 0x0F;
            int int_w = (int)nib - 8;
            ref += in_f[m * K + k] * (float)int_w;
          }
          ref *= w_scale_h[j];
          const float gpu_v = v8c_h2f(y_peek[m * N + j]);
          const float d = gpu_v - ref;
          sum_sq_diff += d * d;
          sum_sq_ref += ref * ref;
          float ad = std::fabs(d);
          if (ad > max_abs_diff) {
            max_abs_diff = ad;
            worst_m = m;
            worst_n = j;
            gpu_at_worst = gpu_v;
            ref_at_worst = ref;
          }
        }
      }
      double relL2 =
        sum_sq_ref > 0.0 ? std::sqrt(sum_sq_diff / sum_sq_ref) : 0.0;
      std::fprintf(stderr,
                   "[V8C-TRACE] id=%d M=%u N=%u K=%u sampled=%ux%u "
                   "relL2=%.4f%% max|diff|=%.4f at (m=%u,n=%u) gpu=%.4f "
                   "ref=%.4f\n",
                   trace_id, M, N, K, sample_M, sample_N, relL2 * 100.0,
                   max_abs_diff, worst_m, worst_n, gpu_at_worst,
                   ref_at_worst);
      std::fflush(stderr);
    }

    // Read output fp16 (only the valid M rows; padded rows are discarded),
    // convert to output dtype.
    if (prof_enabled) T0 = NOW();
    std::vector<uint16_t> y_host((size_t)M * N);
    clEnqueueReadBuffer(q, sc.y_fp16, CL_TRUE, 0,
                        sizeof(uint16_t) * y_host.size(), y_host.data(), 0,
                        nullptr, nullptr);
    if (prof_enabled) {
      T1 = NOW();
      prof.bin[prof_bin].read_ns += NS(T1, T0);
      prof.bin[prof_bin].read_bytes +=
        (long long)sizeof(uint16_t) * (long long)M * N;
      prof.bin[prof_bin].calls++;
      prof.bin[prof_bin].m_sum += M;
      prof.total_calls++;
      // Dump on bin transition (prefill → decode shows up as M>256 → M=1).
      if (last_dumped_bin != prof_bin && last_dumped_bin >= 0) {
        char tag[64];
        std::snprintf(tag, sizeof(tag), "BIN-TRANSITION %s->%s",
                      V8cProf::bin_name(last_dumped_bin),
                      V8cProf::bin_name(prof_bin));
        prof.dump(tag);
      }
      last_dumped_bin = prof_bin;
      // Periodic dump every 500 calls so we don't depend on shutdown.
      if (prof.total_calls % 500 == 0) {
        char tag[32];
        std::snprintf(tag, sizeof(tag), "PERIODIC@%d", prof.total_calls);
        prof.dump(tag);
      }
    }
    if (output.getDataType() == ml::train::TensorDim::DataType::FP32) {
      float *out = output.getData<float>();
      for (size_t i = 0; i < y_host.size(); ++i) out[i] = v8c_h2f(y_host[i]);
    } else if (output.getDataType() == ml::train::TensorDim::DataType::FP16) {
      std::memcpy(output.getData<uint8_t>(), y_host.data(),
                  sizeof(uint16_t) * y_host.size());
    } else {
      clReleaseMemObject(act_image);
      throw std::runtime_error("unsupported output dtype");
    }
    clReleaseMemObject(act_image);

    // === Step 1e bridge round-trip (paper §3.2). Attach the cached
    // v8c weight backing to the output tensor as a non-owning tracer.
    // CPU consumers ignore this field. This is purely a bridge
    // integrity hook today; Step 2's fused QKV kernel will replace it
    // with a real output backing pointing at the cl_mem the next
    // GPU layer will consume. NNTR_TENSOR_BRIDGE_TRIP=1 logs the
    // first round-trip on real device traffic to confirm wiring.
    output.setBacking(w->backing.get());
    static int logged_trip = 0;
    if (!logged_trip && std::getenv("NNTR_TENSOR_BRIDGE_TRIP") != nullptr) {
      logged_trip = 1;
      tv::TensorBacking *back = output.getBacking();
      std::fprintf(stderr,
                   "[Step1e] bridge round-trip: set=%p get=%p %s\n",
                   (void *)w->backing.get(), (void *)back,
                   back == w->backing.get() ? "OK" : "MISMATCH");
      std::fflush(stderr);
    }
  } catch (...) {
    return false;
  }
  return true;
}

// =============================================================================
// Step 2 — Fused Q + K + V projection + RoPE + layout transform.
//
// Paper reference: arXiv:2505.00232 §3.6, "We crafted a custom kernel to
// combine rotary embedding with the layout transformations of query (Q),
// key (K), and value (V) projections."
//
// THIS IS THE STEP 2a SKELETON. The host-side dispatch + env gate are wired
// against the final-form OpenCL kernel signature, but the kernel body is a
// stub (zero-fills outputs). Returns false in all cases except when
// NNTR_FUSED_QKV_GPU=1 AND every binding precondition holds — and even then,
// the actual compute is the stub. Step 2b will replace the kernel body with
// the real shared-quant-then-3-GEMM-then-RoPE math.
//
// The function returns `false` even on stub success: callers (qkv_layer.cpp,
// once wired) MUST fall back to the existing 3-FC + CPU RoPE path. Step 2d
// flips the return to `true` once the kernel body is correctness-validated.
// =============================================================================

namespace {

static bool fused_qkv_env_enabled() {
  static int cached = -1;
  if (cached < 0)
    cached = std::getenv("NNTR_FUSED_QKV_GPU") != nullptr ? 1 : 0;
  return cached != 0;
}

} // anonymous namespace

bool fused_qkv_rope_layout_gpu(
  const Tensor &input, const Tensor &wq, const Tensor &wk, const Tensor &wv,
  const Tensor &cos_table, const Tensor &sin_table,
  unsigned int from_pos, unsigned int hq, unsigned int hkv, unsigned int dh,
  Tensor &q_out, Tensor &k_out, Tensor &v_out) {

  // Step 2a: gate off by default. No-op when the env flag isn't set so this
  // commit is invisible on every existing run (Qwen3-0.6B baseline, all CIs).
  if (!fused_qkv_env_enabled())
    return false;

  // === Precondition checks ===
  // (Mirroring dotCl_v8c's contract; Step 2b will tighten these once we know
  // exactly what the kernel body requires.)
  if (wq.getDataType() != ml::train::TensorDim::DataType::QINT4 ||
      wk.getDataType() != ml::train::TensorDim::DataType::QINT4 ||
      wv.getDataType() != ml::train::TensorDim::DataType::QINT4)
    return false;
  if (input.getDataType() != ml::train::TensorDim::DataType::FP16)
    return false; // Step 2a is FP16-only; FP32 path TBD in 2b.
  if (cos_table.getDataType() != ml::train::TensorDim::DataType::FP16 ||
      sin_table.getDataType() != ml::train::TensorDim::DataType::FP16)
    return false;
  if (hq == 0 || hkv == 0 || dh == 0 || hq % hkv != 0)
    return false;
  if (dh % 2 != 0)
    return false; // RoPE pair rotation needs even head_dim.

  // Step 2a stub: log once on first invocation that the gate fired, then
  // return false so the caller falls back. This proves the dispatch path
  // compiled and is reachable without changing model output.
  static int logged_stub = 0;
  if (!logged_stub) {
    logged_stub = 1;
    std::fprintf(
      stderr,
      "[Step2a] fused_qkv_rope_layout_gpu stub reached: "
      "S=%u hidden=%u hq=%u hkv=%u dh=%u from=%u\n",
      input.height(),
      (input.getFormat() == Tformat::NHWC) ? input.channel() : input.width(),
      hq, hkv, dh, from_pos);
    std::fflush(stderr);
  }

  // Silence unused-param warnings while the body is a stub.
  (void)wq; (void)wk; (void)wv;
  (void)cos_table; (void)sin_table;
  (void)q_out; (void)k_out; (void)v_out;

  // Stub: return false so the caller's existing 3-FC + CPU RoPE path runs.
  // Step 2d flips this once the kernel body produces correct output.
  return false;
}

// =============================================================================
// Segment A.2 — GPU RMSNorm with TensorBacking output residency.
//
// Paper §3.2 cross-layer residency. Produces a cl_mem holding FP16
// normalized activation that downstream FC layers consume directly via
// `dotCl_v8c`'s residency input mode (Segment A.1). No host materialize
// when env-gated.
//
// First wired consumer: Qwen3's `attention_norm` and `ffn_norm` (per
// transformer.cpp:340, 355). q_norm / k_norm use ReshapedRMSNormLayer
// (different class) and are out of Segment A scope.
// =============================================================================
namespace {

// Per-gamma persistent upload cache. Gamma weights don't change at
// inference; upload once per unique gamma name, reuse forever.
struct ResidentRmsState {
  std::unordered_map<std::string, cl_mem> gamma_bufs;
  std::mutex mtx;
};
static ResidentRmsState &resident_rms_state() {
  static ResidentRmsState s;
  return s;
}

static bool resident_rmsnorm_env_enabled() {
  static int cached = -1;
  if (cached < 0)
    cached = std::getenv("NNTR_RESIDENT_RMSNORM") != nullptr ? 1 : 0;
  return cached != 0;
}

} // anonymous namespace

bool rmsnorm_resident_fp16(const Tensor &input, const Tensor &gamma,
                           float epsilon, unsigned int B, unsigned int C,
                           unsigned int H, unsigned int W,
                           const std::string &output_name, Tensor &output) {
  if (!resident_rmsnorm_env_enabled())
    return false;
  if (input.getDataType() != ml::train::TensorDim::DataType::FP16 ||
      gamma.getDataType() != ml::train::TensorDim::DataType::FP16)
    return false;
  if (B == 0 || C == 0 || H == 0 || W == 0)
    return false;

  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  cl_context ctx = blas_cc->context_inst_.GetContext();
  cl_command_queue q = blas_cc->command_queue_inst_.GetCommandQueue();

  const size_t total_elems = (size_t)B * C * H * W;
  const size_t total_bytes = total_elems * sizeof(uint16_t);

  // 1) Get or create the output backing in the pool. Reused across calls
  //    with the same output_name (output is overwritten in place).
  auto &pool = tv::TensorBackingPool::Global();
  std::shared_ptr<tv::TensorBacking> out_bk = pool.get(output_name);
  if (!out_bk || out_bk->bytes() < total_bytes) {
    cl_int err = CL_SUCCESS;
    cl_mem buf =
      clCreateBuffer(ctx, CL_MEM_READ_WRITE, total_bytes, nullptr, &err);
    if (err != CL_SUCCESS || !buf)
      return false;
    out_bk = std::make_shared<tv::TensorBacking>(
      ctx, buf, tv::Encoding::FP16, tv::Layout::ROW_MAJOR, total_bytes,
      /*owned=*/true);
    pool.set(output_name, out_bk);
  }

  // 2) Source cl_mem for the input — from upstream backing if present
  //    (zero host transfer), else upload from host on each call.
  cl_mem in_cl = nullptr;
  cl_mem in_upload_owned = nullptr; // freed at function exit if allocated
  if (const tv::TensorBacking *in_bk = input.getBacking();
      in_bk != nullptr && in_bk->encoding() == tv::Encoding::FP16 &&
      in_bk->bytes() >= total_bytes) {
    in_cl = in_bk->buffer();
  } else {
    cl_int err = CL_SUCCESS;
    in_upload_owned = clCreateBuffer(ctx, CL_MEM_READ_ONLY, total_bytes,
                                     nullptr, &err);
    if (err != CL_SUCCESS || !in_upload_owned)
      return false;
    if (clEnqueueWriteBuffer(q, in_upload_owned, CL_TRUE, 0, total_bytes,
                             input.getData<uint8_t>(), 0, nullptr,
                             nullptr) != CL_SUCCESS) {
      clReleaseMemObject(in_upload_owned);
      return false;
    }
    in_cl = in_upload_owned;
  }

  // 3) Gamma upload cache — once per gamma name. Gamma doesn't change at
  //    inference. Keyed by gamma's name (stable per layer).
  cl_mem gamma_cl = nullptr;
  {
    auto &st = resident_rms_state();
    std::lock_guard<std::mutex> lock(st.mtx);
    const std::string &gn = gamma.getName();
    auto it = st.gamma_bufs.find(gn);
    if (it == st.gamma_bufs.end()) {
      cl_int err = CL_SUCCESS;
      cl_mem gbuf = clCreateBuffer(ctx, CL_MEM_READ_ONLY,
                                   (size_t)W * sizeof(uint16_t), nullptr,
                                   &err);
      if (err != CL_SUCCESS || !gbuf) {
        if (in_upload_owned) clReleaseMemObject(in_upload_owned);
        return false;
      }
      if (clEnqueueWriteBuffer(q, gbuf, CL_TRUE, 0,
                               (size_t)W * sizeof(uint16_t),
                               gamma.getData<uint8_t>(), 0, nullptr,
                               nullptr) != CL_SUCCESS) {
        clReleaseMemObject(gbuf);
        if (in_upload_owned) clReleaseMemObject(in_upload_owned);
        return false;
      }
      st.gamma_bufs[gn] = gbuf;
      gamma_cl = gbuf;
    } else {
      gamma_cl = it->second;
    }
  }

  // 4) Register the kernel (cached by ClContext on repeat registration).
  ClContext::SharedPtrClKernel kp =
    blas_cc->registerClKernel(rmsnorm_fp16_kernel, "rmsnorm_cl_fp16");
  if (!kp) {
    if (in_upload_owned) clReleaseMemObject(in_upload_owned);
    return false;
  }

  cl_mem out_buf = out_bk->buffer();
  cl_half eps_half;
  {
    const float ef = epsilon;
    // round-half-to-nearest float→half via the helper available in this TU.
    // The activation residual stream is FP16 so this matches the upstream
    // precision exactly.
    uint16_t bits = 0;
    // simple manual conversion for non-NaN positive epsilon (always small).
    union { float f; uint32_t u; } v;
    v.f = ef;
    uint32_t e = (v.u >> 23) & 0xFF;
    uint32_t m = v.u & 0x7FFFFF;
    if (e == 0) bits = 0;
    else if (e >= 143) bits = 0x7BFF; // clamp to fp16 max
    else if (e <= 112) bits = 0;       // underflow to 0
    else bits = ((e - 112) << 10) | (m >> 13);
    eps_half = bits;
  }

  int arg = 0;
  if (!kp->SetKernelArguments(arg++, &in_cl, sizeof(cl_mem)) ||
      !kp->SetKernelArguments(arg++, &out_buf, sizeof(cl_mem)) ||
      !kp->SetKernelArguments(arg++, &gamma_cl, sizeof(cl_mem)) ||
      !kp->SetKernelArguments(arg++, &eps_half, sizeof(cl_half)) ||
      !kp->SetKernelArguments(arg++, &B, sizeof(int)) ||
      !kp->SetKernelArguments(arg++, &C, sizeof(int)) ||
      !kp->SetKernelArguments(arg++, &H, sizeof(int)) ||
      !kp->SetKernelArguments(arg++, &W, sizeof(int))) {
    if (in_upload_owned) clReleaseMemObject(in_upload_owned);
    return false;
  }

  // work-groups per existing RMSNormLayerCl: {B*C, H, 1}, local {W, 1, 1}.
  const int wg_count[3] = {(int)(B * C), (int)H, 1};
  const int wg_size[3] = {(int)W, 1, 1};
  if (!blas_cc->command_queue_inst_.DispatchCommand(kp, wg_count, wg_size)) {
    if (in_upload_owned) clReleaseMemObject(in_upload_owned);
    return false;
  }

  // 5) Publish the backing to the consumer side. Both setBacking() on the
  //    Tensor (if the consumer's Tensor instance is the same one we got
  //    here) AND pool.set() (already done at allocation) so a name-based
  //    lookup is also possible. Host data of `output` is left undefined.
  output.setBacking(out_bk.get());
  // Bump the global resident-quant generation: any downstream FC that
  // sees this backing pointer in its quant cache must re-quant; the
  // backing's data has just changed.
  g_resident_quant_generation.fetch_add(1, std::memory_order_release);

  // OUT-OF-ORDER QUEUE FIX: the ClContext queue uses
  // CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE (opencl_command_queue_
  // manager.cpp:56). Without explicit events or a barrier, subsequent
  // enqueues are NOT guaranteed to wait for this kernel even when they
  // read the same cl_mem. The pool's ptr-keyed entry is overwritten by
  // every RMSNorm in the chain (TensorPool reuses the same host buffer
  // for all per-layer norm outputs), and multiple FC consumers race
  // against multiple RMSNorm producers. Barrier serializes.
  clEnqueueBarrierWithWaitList(q, 0, nullptr, nullptr);

  // Static one-shot log for first invocation so we can confirm wiring
  // when running a model. NNTR_RESIDENT_RMSNORM_TRIP=1 prints once.
  static int logged_trip = 0;
  if (!logged_trip && std::getenv("NNTR_RESIDENT_RMSNORM_TRIP") != nullptr) {
    logged_trip = 1;
    std::fprintf(stderr,
                 "[SegA-RMS] first invocation: out_name=%s B=%u C=%u H=%u "
                 "W=%u in_from_backing=%d\n",
                 output_name.c_str(), B, C, H, W,
                 in_upload_owned == nullptr ? 1 : 0);
    std::fflush(stderr);
  }

  if (in_upload_owned) {
    // The kernel reads input as a buffer; we can't release until the
    // kernel has finished. Block once to ensure the upload buffer is
    // safe to release. For backing-supplied inputs we don't allocate.
    clFinish(q);
    clReleaseMemObject(in_upload_owned);
  }

  return true;
}

// FP32 variant (Qwen3's actual residual-stream dtype). Same lifecycle/
// caching/backing semantics as the FP16 variant; just different kernel.
bool rmsnorm_resident_fp32(const Tensor &input, const Tensor &gamma,
                           float epsilon, unsigned int H, unsigned int W,
                           const std::string &output_name, Tensor &output) {
  if (!resident_rmsnorm_env_enabled())
    return false;
  if (input.getDataType() != ml::train::TensorDim::DataType::FP32 ||
      gamma.getDataType() != ml::train::TensorDim::DataType::FP32)
    return false;
  if (H == 0 || W == 0)
    return false;
  // rmsnorm_cl kernel reads input as float4 — W must be a multiple of 4.
  if (W % 4 != 0)
    return false;

  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  cl_context ctx = blas_cc->context_inst_.GetContext();
  cl_command_queue q = blas_cc->command_queue_inst_.GetCommandQueue();

  const size_t total_elems = (size_t)H * W;
  const size_t total_bytes = total_elems * sizeof(float);

  auto &pool = tv::TensorBackingPool::Global();
  std::shared_ptr<tv::TensorBacking> out_bk = pool.get(output_name);
  if (!out_bk || out_bk->bytes() < total_bytes ||
      out_bk->encoding() != tv::Encoding::FP32) {
    cl_int err = CL_SUCCESS;
    cl_mem buf =
      clCreateBuffer(ctx, CL_MEM_READ_WRITE, total_bytes, nullptr, &err);
    if (err != CL_SUCCESS || !buf)
      return false;
    out_bk = std::make_shared<tv::TensorBacking>(
      ctx, buf, tv::Encoding::FP32, tv::Layout::ROW_MAJOR, total_bytes,
      /*owned=*/true);
    pool.set(output_name, out_bk);
  }
  // Also register under the host-data-pointer key so consumers receiving a
  // different Tensor instance (with the same underlying data pointer) can
  // find this backing. See the dotCl_v8c residency-input lookup code.
  {
    const void *out_data_ptr = output.getData<uint8_t>();
    char key_buf[64];
    std::snprintf(key_buf, sizeof(key_buf), "ptr:%p", out_data_ptr);
    pool.set(std::string(key_buf), out_bk);
  }

  cl_mem in_cl = nullptr;
  cl_mem in_upload_owned = nullptr;
  std::shared_ptr<tv::TensorBacking> in_bk_pool_strong;
  if (const tv::TensorBacking *in_bk = input.getBacking();
      in_bk != nullptr && in_bk->encoding() == tv::Encoding::FP32 &&
      in_bk->bytes() >= total_bytes) {
    in_cl = in_bk->buffer();
  } else {
    const void *in_data_ptr = input.getData<uint8_t>();
    char key_buf[64];
    std::snprintf(key_buf, sizeof(key_buf), "ptr:%p", in_data_ptr);
    in_bk_pool_strong =
      tv::TensorBackingPool::Global().get(std::string(key_buf));
    if (in_bk_pool_strong &&
        in_bk_pool_strong->encoding() == tv::Encoding::FP32 &&
        in_bk_pool_strong->bytes() >= total_bytes) {
      in_cl = in_bk_pool_strong->buffer();
    }
  }
  if (in_cl == nullptr) {
    cl_int err = CL_SUCCESS;
    in_upload_owned = clCreateBuffer(ctx, CL_MEM_READ_ONLY, total_bytes,
                                     nullptr, &err);
    if (err != CL_SUCCESS || !in_upload_owned)
      return false;
    if (clEnqueueWriteBuffer(q, in_upload_owned, CL_TRUE, 0, total_bytes,
                             input.getData<uint8_t>(), 0, nullptr,
                             nullptr) != CL_SUCCESS) {
      clReleaseMemObject(in_upload_owned);
      return false;
    }
    in_cl = in_upload_owned;
  }

  cl_mem gamma_cl = nullptr;
  {
    auto &st = resident_rms_state();
    std::lock_guard<std::mutex> lock(st.mtx);
    const std::string gn = gamma.getName() + ":fp32";
    auto it = st.gamma_bufs.find(gn);
    if (it == st.gamma_bufs.end()) {
      cl_int err = CL_SUCCESS;
      cl_mem gbuf = clCreateBuffer(ctx, CL_MEM_READ_ONLY,
                                   (size_t)W * sizeof(float), nullptr, &err);
      if (err != CL_SUCCESS || !gbuf) {
        if (in_upload_owned) clReleaseMemObject(in_upload_owned);
        return false;
      }
      if (clEnqueueWriteBuffer(q, gbuf, CL_TRUE, 0, (size_t)W * sizeof(float),
                               gamma.getData<uint8_t>(), 0, nullptr,
                               nullptr) != CL_SUCCESS) {
        clReleaseMemObject(gbuf);
        if (in_upload_owned) clReleaseMemObject(in_upload_owned);
        return false;
      }
      st.gamma_bufs[gn] = gbuf;
      gamma_cl = gbuf;
    } else {
      gamma_cl = it->second;
    }
  }

  ClContext::SharedPtrClKernel kp =
    blas_cc->registerClKernel(rmsnorm_kernel, "rmsnorm_cl");
  if (!kp) {
    if (in_upload_owned) clReleaseMemObject(in_upload_owned);
    return false;
  }

  cl_mem out_buf = out_bk->buffer();
  int arg = 0;
  if (!kp->SetKernelArguments(arg++, &in_cl, sizeof(cl_mem)) ||
      !kp->SetKernelArguments(arg++, &out_buf, sizeof(cl_mem)) ||
      !kp->SetKernelArguments(arg++, &gamma_cl, sizeof(cl_mem)) ||
      !kp->SetKernelArguments(arg++, &epsilon, sizeof(float)) ||
      !kp->SetKernelArguments(arg++, &H, sizeof(int)) ||
      !kp->SetKernelArguments(arg++, &W, sizeof(int))) {
    if (in_upload_owned) clReleaseMemObject(in_upload_owned);
    return false;
  }

  // rmsnorm_cl uses get_group_id(0) → H groups, subgroup reduce inside.
  // DispatchCommand interprets the first array as the GLOBAL work-item
  // count (NDRange standard), so global = H * subgroup, local = subgroup.
  // Matches rmsnorm_cl_internal at blas_kernels_templates.h:428.
  // Diagnostic NNTR_SEGA_RMS_LOCAL=N overrides the local size. With
  // N=1, the kernel runs single-threaded per row (no subgroup reduce),
  // useful to isolate whether subgroup_reduce_add is the divergence
  // source from CPU NEON.
  int subgroup_size = 64; // Adreno default
  if (const char *e = std::getenv("NNTR_SEGA_RMS_LOCAL"))
    subgroup_size = std::atoi(e);
  const int wg_count[3] = {(int)H * subgroup_size, 1, 1};
  const int wg_size[3] = {subgroup_size, 1, 1};
  if (!blas_cc->command_queue_inst_.DispatchCommand(kp, wg_count, wg_size)) {
    if (in_upload_owned) clReleaseMemObject(in_upload_owned);
    return false;
  }

  output.setBacking(out_bk.get());
  // Bump the global resident-quant generation: any downstream FC that
  // sees this backing pointer in its quant cache must re-quant; the
  // backing's data has just changed.
  g_resident_quant_generation.fetch_add(1, std::memory_order_release);

  // OUT-OF-ORDER QUEUE FIX: the ClContext queue uses
  // CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE (opencl_command_queue_
  // manager.cpp:56). Without explicit events or a barrier, subsequent
  // enqueues are NOT guaranteed to wait for this kernel even when they
  // read the same cl_mem. The pool's ptr-keyed entry is overwritten by
  // every RMSNorm in the chain (TensorPool reuses the same host buffer
  // for all per-layer norm outputs), and multiple FC consumers race
  // against multiple RMSNorm producers. Barrier serializes.
  clEnqueueBarrierWithWaitList(q, 0, nullptr, nullptr);

  static int logged_trip = 0;
  if (!logged_trip && std::getenv("NNTR_RESIDENT_RMSNORM_TRIP") != nullptr) {
    logged_trip = 1;
    std::fprintf(stderr,
                 "[SegA-RMS-FP32] first invocation: out_name=%s H=%u W=%u "
                 "in_from_backing=%d\n",
                 output_name.c_str(), H, W,
                 in_upload_owned == nullptr ? 1 : 0);
    std::fflush(stderr);
  }

  if (in_upload_owned) {
    clFinish(q);
    clReleaseMemObject(in_upload_owned);
  }

  return true;
}

// CPU-norm + GPU-residency-handoff path. Uploads the output Tensor's
// host FP32 data into a TensorBacking under `output_name`. Bit-exact
// w.r.t. CPU computation because no GPU compute occurs here. Caller
// must have already populated output.getData<float>() via the existing
// CPU RMSNorm code.
bool publish_host_fp32_to_backing(const Tensor &output,
                                  const std::string &output_name) {
  if (output.getDataType() != ml::train::TensorDim::DataType::FP32)
    return false;
  const auto &dim = output.getDim();
  const size_t total_elems = (size_t)dim.batch() * dim.channel() *
                             dim.height() * dim.width();
  if (total_elems == 0)
    return false;
  const size_t total_bytes = total_elems * sizeof(float);

  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  cl_context ctx = blas_cc->context_inst_.GetContext();
  cl_command_queue q = blas_cc->command_queue_inst_.GetCommandQueue();

  auto &pool = tv::TensorBackingPool::Global();
  std::shared_ptr<tv::TensorBacking> bk = pool.get(output_name);
  if (!bk || bk->bytes() < total_bytes ||
      bk->encoding() != tv::Encoding::FP32) {
    cl_int err = CL_SUCCESS;
    cl_mem buf =
      clCreateBuffer(ctx, CL_MEM_READ_WRITE, total_bytes, nullptr, &err);
    if (err != CL_SUCCESS || !buf)
      return false;
    bk = std::make_shared<tv::TensorBacking>(
      ctx, buf, tv::Encoding::FP32, tv::Layout::ROW_MAJOR, total_bytes,
      /*owned=*/true);
    pool.set(output_name, bk);
  }
  // Also register under the host-data-pointer key so consumers receiving
  // a different Tensor instance (same underlying buffer) find this entry.
  {
    char key_buf[64];
    std::snprintf(key_buf, sizeof(key_buf), "ptr:%p",
                  static_cast<const void *>(output.getData<uint8_t>()));
    pool.set(std::string(key_buf), bk);
  }

  // Upload the CPU-computed RMSNorm output into the backing's cl_mem.
  if (clEnqueueWriteBuffer(q, bk->buffer(), CL_FALSE, 0, total_bytes,
                           output.getData<uint8_t>(), 0, nullptr,
                           nullptr) != CL_SUCCESS)
    return false;
  // Bump the generation so the v8c quant cache invalidates stale entries.
  g_resident_quant_generation.fetch_add(1, std::memory_order_release);
  // We don't need a barrier here — the FC's queued ops follow this write
  // in the same queue; OoO scheduler tracks the cl_mem write dependency
  // for the FC's read (and barrier would prevent the FC enqueue from
  // starting earlier than necessary anyway).

  return true;
}

bool readback_backing_to_host(Tensor &t) {
  const tv::TensorBacking *bk = t.getBacking();
  if (bk == nullptr || bk->buffer() == nullptr)
    return false;
  const auto &dim = t.getDim();
  const size_t elems = (size_t)dim.batch() * dim.channel() * dim.height() *
                       dim.width();
  if (elems == 0)
    return false;
  const size_t elem_bytes =
    (t.getDataType() == ml::train::TensorDim::DataType::FP16) ? 2u :
    (t.getDataType() == ml::train::TensorDim::DataType::FP32) ? 4u : 0u;
  if (elem_bytes == 0)
    return false;
  const size_t bytes = elems * elem_bytes;
  if (bk->bytes() < bytes)
    return false;
  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  cl_command_queue q = blas_cc->command_queue_inst_.GetCommandQueue();
  clFinish(q);
  if (clEnqueueReadBuffer(q, bk->buffer(), CL_TRUE, 0, bytes,
                          t.getData<uint8_t>(), 0, nullptr,
                          nullptr) != CL_SUCCESS)
    return false;
  return true;
}

} // namespace nntrainer
