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
#include "cl_tensor_view.h"
#include <cl_context.h>
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
};
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
  if (prof_enabled) T0 = NOW();
  if (clEnqueueWriteBuffer(q, sc.act_in, CL_FALSE, 0, (size_t)M * K * act_elem,
                           input.getData<uint8_t>(), 0, nullptr,
                           nullptr) != CL_SUCCESS)
    return false;
  if (M_pad > M) {
    const size_t pad_bytes = (size_t)(M_pad - M) * K * act_elem;
    std::vector<uint8_t> zeros(pad_bytes, 0);
    if (clEnqueueWriteBuffer(q, sc.act_in, CL_FALSE,
                             (size_t)M * K * act_elem, pad_bytes, zeros.data(),
                             0, nullptr, nullptr) != CL_SUCCESS)
      return false;
  }
  if (prof_enabled) {
    clFinish(q);
    T1 = NOW();
    prof.bin[prof_bin].write_ns += NS(T1, T0);
    prof.bin[prof_bin].write_bytes += (long long)M * K * act_elem;
  }

  try {
    // (c) fp→int8 asymmetric act quant + zero-point + row_sum over M_pad rows.
    //     Padded rows map to (scale=1, zp=0, q=0, row_sum=0), so they
    //     contribute zero in the GEMM and don't pollute valid rows.
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
  } catch (...) {
    return false;
  }
  return true;
}

} // namespace nntrainer
