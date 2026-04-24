// SPDX-License-Identifier: Apache-2.0
/**
 * @file	half_tensor.cpp
 * @date	01 December 2023
 * @brief	This is a HalfTensor class for 16-bit floating point calculation
 * @see		https://github.com/nntrainer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @author	Donghyeon Jeong <dhyeon.jeong@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>

#include <cpu_backend.h>
#include <half_tensor.h>
#include <profile_gate.h>
#include <tensor.h>
#include <util_func.h>

#if defined(ENABLE_OPENCL) && ENABLE_OPENCL == 1
#include <CL/cl.h>
#include "blas_kernels.h"
#include <cl_context.h>
#include <gpu_image_pool.h>
#endif

namespace nntrainer {

#if defined(ENABLE_OPENCL) && ENABLE_OPENCL == 1
namespace {

// Per-stage cumulative profiler for HalfTensor::dotQInteger PREFILL
// path (M > 1). Decode (M = 1) is intentionally NOT measured -- the
// per-call CPU stages are tiny on M=1 and we only want the prefill
// bottleneck breakdown.
//
// Mirror of FloatTensor's profiler but with different stage semantics:
// HalfTensor doesn't do any fp32<->fp16 CPU conversion because `*this`
// and `output` are already fp16, so `in_stage` / `out_stage` just
// measure the memcpy-equivalent staging into/out of page-aligned
// scratch SVM.
//
// Prints a one-shot summary at process exit.
struct HalfDotQInt4Profile {
  std::atomic<uint64_t> calls{0};        // M>1 (prefill) calls only
  std::atomic<uint64_t> ns_in_stage{0};  // fp16 -> page-aligned svm_in
  std::atomic<uint64_t> ns_in_fence{0};  // blocking SVMMap before staging copy
  std::atomic<uint64_t> ns_gpu_call{0};  // gemm wrapper (incl. sync)
  std::atomic<uint64_t> ns_out_stage{0}; // svm_out -> fp16 output tensor
  std::atomic<uint64_t> ns_misc{0};      // anything outside the 3 stages

  ~HalfDotQInt4Profile() {
    const uint64_t c = calls.load();
    if (c == 0)
      return;
    if (nntrainer::prefill_profile_suppressed())
      return;

    const uint64_t in = ns_in_stage.load();
    const uint64_t fence = ns_in_fence.load();
    const uint64_t gpu = ns_gpu_call.load();
    const uint64_t out = ns_out_stage.load();
    const uint64_t misc = ns_misc.load();
    const uint64_t total = in + fence + gpu + out + misc;
    const double total_ms = total / 1.0e6;

    auto pct = [&](uint64_t v) -> double {
      return total == 0 ? 0.0 : (double)v / (double)total * 100.0;
    };
    auto ms = [](uint64_t v) -> double { return v / 1.0e6; };

    std::fprintf(stderr,
                 "\n[PROFILE HalfTensor::dotQInteger prefill (M>1)] "
                 "total=%.2f ms calls=%llu\n",
                 total_ms, (unsigned long long)c);
    std::fprintf(stderr,
                 "  in_fence    : %8.2f ms (%5.1f%%)  [blocking SVMMap "
                 "on pool-miss inputs]\n",
                 ms(fence), pct(fence));
    std::fprintf(stderr,
                 "  in_stage    : %8.2f ms (%5.1f%%)  [fp16 -> svm_in "
                 "scalar loop]\n",
                 ms(in), pct(in));
    std::fprintf(stderr,
                 "  gpu_call    : %8.2f ms (%5.1f%%)  [wrapper + clEnqueue + "
                 "SVMMap sync]\n",
                 ms(gpu), pct(gpu));
    std::fprintf(stderr,
                 "  out_stage   : %8.2f ms (%5.1f%%)  [svm_out -> fp16 "
                 "scalar loop]\n",
                 ms(out), pct(out));
    std::fprintf(stderr,
                 "  misc        : %8.2f ms (%5.1f%%)  [assertions, ptr "
                 "setup]\n",
                 ms(misc), pct(misc));
  }
};

HalfDotQInt4Profile g_half_dotq_profile;

// Parallel profiler for the M == 1 decode path.  In generation mode
// dotQInteger is called once per FC per token (~252 calls / token for
// Qwen3-4B at 36 layers).  Each call has three wall-clock stages:
//   in_copy   : scalar in_u16 -> svm_in (K elements)
//   gemv_call : gpu_int4_gemv_adreno (incl. SVM sync)
//   out_copy  : scalar svm_out -> out_u16 (N elements)
// The prefill profiler above intentionally skips this path; we add a
// separate one here so a single run produces two independent
// breakdowns, one for prefill and one for decode.
struct HalfDotQInt4DecodeProfile {
  std::atomic<uint64_t> calls{0};
  std::atomic<uint64_t> ns_in_copy{0};
  std::atomic<uint64_t> ns_gemv_call{0};
  std::atomic<uint64_t> ns_out_copy{0};

  ~HalfDotQInt4DecodeProfile() {
    const uint64_t c = calls.load();
    if (c == 0)
      return;

    const uint64_t in_copy = ns_in_copy.load();
    const uint64_t gemv = ns_gemv_call.load();
    const uint64_t out_copy = ns_out_copy.load();
    const uint64_t total = in_copy + gemv + out_copy;
    const double total_ms = total / 1.0e6;

    auto pct = [&](uint64_t v) -> double {
      return total == 0 ? 0.0 : (double)v / (double)total * 100.0;
    };
    auto ms = [](uint64_t v) -> double { return v / 1.0e6; };

    std::fprintf(stderr,
                 "\n[PROFILE HalfTensor::dotQInteger decode (M==1)] "
                 "total=%.2f ms calls=%llu\n",
                 total_ms, (unsigned long long)c);
    std::fprintf(stderr,
                 "  in_copy   : %8.2f ms (%5.1f%%)  [in_u16 -> svm_in "
                 "scalar loop, K elems]\n",
                 ms(in_copy), pct(in_copy));
    std::fprintf(stderr,
                 "  gemv_call : %8.2f ms (%5.1f%%)  [gpu_int4_gemv_adreno + "
                 "SVM sync]\n",
                 ms(gemv), pct(gemv));
    std::fprintf(stderr,
                 "  out_copy  : %8.2f ms (%5.1f%%)  [svm_out -> out_u16 "
                 "scalar loop, N elems]\n",
                 ms(out_copy), pct(out_copy));
  }
};

HalfDotQInt4DecodeProfile g_half_dotq_decode_profile;

inline uint64_t now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
           std::chrono::steady_clock::now().time_since_epoch())
    .count();
}

} // namespace
#endif

HalfTensor::HalfTensor(std::string name_, Tformat fm) :
  TensorBase(name_, fm, Tdatatype::FP16) {}

HalfTensor::HalfTensor(const TensorDim &d, bool alloc_now, Initializer init,
                       std::string name) :
  TensorBase(d, alloc_now, init, name) {
  if (alloc_now)
    allocate();
}

HalfTensor::HalfTensor(const TensorDim &d, const void *buf) :
  HalfTensor(d, true) {
  if (d.getDataLen() != 0) {
    if (buf != nullptr)
      copy(buf);
  }
}

bool HalfTensor::operator==(const HalfTensor &rhs) const {
  const _FP16 *_data = (_FP16 *)getData();
  const _FP16 *_rdata = (_FP16 *)rhs.getData();
  for (size_t i = 0; i < size(); ++i) {
    if (std::isnan((float)_data[i]) || std::isnan((float)_rdata[i]) ||
        std::fabs((float)(_data[i] - _rdata[i])) > epsilon)
      return false;
  }

  return true;
}

/// @todo support allocation by src_tensor
void HalfTensor::allocate() {
  if (empty() || data)
    /// already allocated
    return;

  if (src_tensor) {
    /// allocate data based on the source tensor
    allocateSrcTensor();
    /** as this memory is shared, do NOT initialize */
  } else {
    /// allocate new memory for the tensor data
    MemoryData *mem_data;

    mem_data = new MemoryData((void *)(new _FP16[dim.getDataLen()]{}));
    data = std::shared_ptr<MemoryData>(mem_data, [](auto *mem_data) {
      delete[] mem_data->template getAddr<_FP16>();
      delete mem_data;
    });

    offset = 0;
    initialize();
  }
}

void HalfTensor::deallocate() {
  data = nullptr;
  offset = 0;
}

void *HalfTensor::getData() const {
  if (!data)
    return nullptr;

  data->validate();
  return data->getAddr<_FP16>() + offset;
}

void *HalfTensor::getData(size_t idx) const {
  if (!data)
    return nullptr;

  data->validate();
  return data->getAddr<_FP16>() + offset + idx;
}

void *HalfTensor::getAddress(unsigned int i) {
  size_t index = getIndex(batch(), channel(), height(), width());
  if (i > index) {
    return nullptr;
  }
  return &((_FP16 *)getData())[i];
}

const void *HalfTensor::getAddress(unsigned int i) const {
  size_t index = getIndex(batch(), channel(), height(), width());
  if (i > index) {
    return nullptr;
  }
  return &((_FP16 *)getData())[i];
}

const _FP16 &HalfTensor::getValue(unsigned int i) const {
  return ((_FP16 *)getData())[i];
}

_FP16 &HalfTensor::getValue(unsigned int i) { return ((_FP16 *)getData())[i]; }

const _FP16 &HalfTensor::getValue(unsigned int b, unsigned int c,
                                  unsigned int h, unsigned int w) const {
  return getValue(getIndex(b, c, h, w));
}

_FP16 &HalfTensor::getValue(unsigned int b, unsigned int c, unsigned int h,
                            unsigned int w) {
  return getValue(getIndex(b, c, h, w));
}

void HalfTensor::setValue(float value) {
  _FP16 *data = (_FP16 *)getData();
  std::fill(data, data + size(), static_cast<_FP16>(value));
}

void HalfTensor::setValue(unsigned int b, unsigned int c, unsigned int h,
                          unsigned int w, float value) {
  ((_FP16 *)getData())[getIndex(b, c, h, w)] = static_cast<_FP16>(value);
}

void HalfTensor::addValue(unsigned int b, unsigned int c, unsigned int h,
                          unsigned int w, float value, float beta) {
  auto const &idx = getIndex(b, c, h, w);
  ((_FP16 *)getData())[idx] *= static_cast<_FP16>(beta);
  ((_FP16 *)getData())[idx] += static_cast<_FP16>(value);
}

void HalfTensor::setZero() {
  if (contiguous) {
    //  sscal(size(), 0, (_FP16 *)getData(), 1);
    /// @note we cannot use sscal, when we set zero. if the data is inf or
    /// NaN, then the inf or NaN still remain.
    memset((_FP16 *)getData(), 0, sizeof(_FP16) * size());
  } else {
    /// @todo implement apply_i
    // apply_i<_FP16>([](_FP16 val) -> _FP16 { return 0; });
    setValue(0);
  }
}

void HalfTensor::setRandNormal(float mean, float stddev) {
  setDist<std::normal_distribution<float>>(
    std::normal_distribution<float>(mean, stddev));
}

void HalfTensor::setRandUniform(float min, float max) {
  setDist<std::uniform_real_distribution<float>>(
    std::uniform_real_distribution<float>(min, max));
}

void HalfTensor::setRandBernoulli(float probability) {
  setDist<std::bernoulli_distribution>(
    std::bernoulli_distribution(probability));
}

void HalfTensor::initialize() {
  if (empty() || !isAllocated())
    return;

  unsigned int fan_in, fan_out;

  /// @fixme: when unit is equal to one, this does not work, we need to rely on
  /// effective dimension then actual numbers here. For now, some heuristics
  /// added to infer what would be fan_in/fan_out
  if (dim.batch() * dim.channel() * dim.height() == 1) {
    fan_out = fan_in = dim.width();
  } else if (dim.batch() * dim.channel() == 1) { /// fc layer - 2-D tensor
    fan_in = dim.height();
    fan_out = dim.width();
  } else { /// conv2d filters - 4d tensor, @todo extend this to > 4
    auto field_size = dim.height() * dim.width();

    // this also handles below cases.
    // 1. fan_in = fan_out = 1 as well.
    // 2. batch == 1, channel == 1 and height == 1, theoretical rank of 1
    fan_in = dim.channel() * field_size;
    fan_out = dim.batch() * field_size;
  }

  switch (initializer) {
  case Initializer::ZEROS:
    setZero();
    break;
  case Initializer::ONES:
    setValue(1.0f);
    break;
  case Initializer::LECUN_NORMAL:
    setRandNormal(0.0f, sqrtFloat(1.0f / fan_in));
    break;
  case Initializer::XAVIER_NORMAL:
    setRandNormal(0.0f, sqrtFloat(2.0f / (fan_in + fan_out)));
    break;
  case Initializer::HE_NORMAL:
    setRandNormal(0.0f, sqrtFloat(2.0f / (fan_in)));
    break;
  case Initializer::LECUN_UNIFORM:
    setRandUniform(-1.0f * sqrtFloat(1.0f / fan_in), sqrtFloat(1.0f / fan_in));
    break;
  case Initializer::XAVIER_UNIFORM:
    setRandUniform(-1.0f * sqrtFloat(6.0f / (fan_in + fan_out)),
                   sqrtFloat(6.0 / (fan_in + fan_out)));
    break;
  case Initializer::HE_UNIFORM:
    setRandUniform(-1.0f * sqrtFloat(6.0f / (fan_in)),
                   sqrtFloat(6.0 / (fan_in)));
  default:
    break;
  }

  putData();
}

void HalfTensor::initialize(Initializer init) {
  initializer = init;
  initialize();
}

Tensor &HalfTensor::apply(std::function<_FP16(_FP16)> f, Tensor &output) const {
  CREATE_IF_EMPTY_DIMS(output, dim, nullptr);

  if (contiguous && output.getContiguous()) {
    const _FP16 *data = (_FP16 *)getData();
    _FP16 *rdata = output.getData<_FP16>();

    std::transform(data, data + size(), rdata, f);
  } else if (strides[3] == 1 && output.getStrides()[3] == 1) {
    /** @todo optimize this with combining these loops where stride is 1 */
    for (unsigned int b = 0; b < batch(); ++b) {
      for (unsigned int c = 0; c < channel(); ++c) {
        for (unsigned int h = 0; h < height(); ++h) {
          _FP16 *out_data = output.getAddress<_FP16>(b, c, h, 0);
          const _FP16 *in_data = (_FP16 *)getAddress(getIndex(b, c, h, 0));
          std::transform(in_data, in_data + width(), out_data, f);
        }
      }
    }
  } else {
    for (unsigned int b = 0; b < batch(); ++b) {
      for (unsigned int c = 0; c < channel(); ++c) {
        for (unsigned int h = 0; h < height(); ++h) {
          for (unsigned int w = 0; w < width(); ++w) {
            output.setValue(b, c, h, w, f(getValue(b, c, h, w)));
          }
        }
      }
    }
  }

  return output;
}

Tensor HalfTensor::multiply_strided(Tensor const &m, Tensor &output,
                                    const float beta) const {
  CREATE_IF_EMPTY_DIMS(output, dim, nullptr);

  if (size() != m.size() || size() != output.size())
    throw std::invalid_argument(
      "Strided multiplication does not support broadcasting");

  NNTR_THROW_IF(getData() == nullptr, std::invalid_argument)
    << getName() << " is not allocated";
  NNTR_THROW_IF(m.getData<_FP16>() == nullptr, std::invalid_argument)
    << m.getName() << " is not allocated";
  NNTR_THROW_IF(output.getData<_FP16>() == nullptr, std::invalid_argument)
    << output.getName() << " is not allocated";

  if (strides[3] != 1 || m.getStrides()[3] != 1 ||
      output.getStrides()[3] != 1 || std::fpclassify(beta) != FP_ZERO) {
    for (unsigned int b = 0; b < batch(); ++b) {
      for (unsigned int c = 0; c < channel(); ++c) {
        for (unsigned int h = 0; h < height(); ++h) {
          for (unsigned int w = 0; w < width(); ++w) {
            output.addValue(
              b, c, h, w, getValue(b, c, h, w) * m.getValue<_FP16>(b, c, h, w),
              beta);
          }
        }
      }
    }
  } else {
    /** @todo optimize by combining these loops where stride is 1 */
    if (this->getFormat() == Tformat::NCHW) {
      for (unsigned int b = 0; b < batch(); ++b) {
        for (unsigned int c = 0; c < channel(); ++c) {
          for (unsigned int h = 0; h < height(); ++h) {
            _FP16 *out_data = output.getAddress<_FP16>(b, c, h, 0);
            const _FP16 *m_data = m.getAddress<_FP16>(b, c, h, 0);
            const _FP16 *in_data = (_FP16 *)getAddress(getIndex(b, c, h, 0));
            std::transform(in_data, in_data + width(), m_data, out_data,
                           std::multiplies<_FP16>());
          }
        }
      }
    } else {
      for (unsigned int b = 0; b < batch(); ++b) {
        for (unsigned int h = 0; h < height(); ++h) {
          for (unsigned int w = 0; w < width(); ++w) {
            _FP16 *out_data = output.getAddress<_FP16>(b, 0, h, w);
            const _FP16 *m_data = m.getAddress<_FP16>(b, 0, h, w);
            const _FP16 *in_data = (_FP16 *)getAddress(getIndex(b, 0, h, w));
            std::transform(in_data, in_data + channel(), m_data, out_data,
                           std::multiplies<_FP16>());
          }
        }
      }
    }
  }

  return output;
}

int HalfTensor::multiply_i(float const &value) {
  _FP16 *data = (_FP16 *)getData();
  unsigned int len = size();
  sscal(len, value, data, 1);

  return ML_ERROR_NONE;
}

Tensor &HalfTensor::multiply(float const &value, Tensor &out) const {
  auto f = std::bind(std::multiplies<_FP16>(), std::placeholders::_1,
                     static_cast<_FP16>(value));
  apply(f, out);
  return out;
}

Tensor &HalfTensor::multiply(Tensor const &m, Tensor &output,
                             const float beta) const {
  auto f = [&](const BroadcastInfo &e, const _FP16 *buf, const _FP16 *m_buf,
               _FP16 *out_buf) {
    ele_mul(e.buffer_size, buf, m_buf, out_buf, 1, beta, e.strides[3],
            strides[3]);
  };

  NNTR_THROW_IF(m.getFormat() != this->getFormat(), std::invalid_argument)
    << "Tensor Format of " << getName() << ":"
    << ((bool)(this->getFormat()) ? "NHWC" : "NCHW") << " is not match. ("
    << ((bool)(m.getFormat()) ? "NHWC" : "NCHW") << ")";

  NNTR_THROW_IF(!contiguous || !m.getContiguous() || !output.getContiguous(),
                std::invalid_argument)
    << getName() << " is not contiguous, cannot multiply";

  apply_broadcast(m, f, output);
  return output;
}

Tensor &HalfTensor::add_strided(Tensor const &input, Tensor &output,
                                const float beta) const {
  if (size() != input.size() || size() != output.size())
    throw std::invalid_argument(
      "Strided multiplication does not support broadcasting");

  NNTR_THROW_IF(getData() == nullptr, std::invalid_argument)
    << getName() << " is not allocated";
  NNTR_THROW_IF(input.getData<_FP16>() == nullptr, std::invalid_argument)
    << input.getName() << " is not allocated";
  NNTR_THROW_IF(output.getData<_FP16>() == nullptr, std::invalid_argument)
    << output.getName() << " is not allocated";

  if (strides[3] != 1 || input.getStrides()[3] != 1 ||
      output.getStrides()[3] != 1 || std::fpclassify(beta) != FP_ZERO) {
    for (unsigned int b = 0; b < batch(); ++b) {
      for (unsigned int c = 0; c < channel(); ++c) {
        for (unsigned int h = 0; h < height(); ++h) {
          for (unsigned int w = 0; w < width(); ++w) {
            output.setValue(b, c, h, w,
                            getValue(b, c, h, w) +
                              input.getValue<_FP16>(b, c, h, w) * beta);
          }
        }
      }
    }
  } else {
    /** @todo optimize by combining these loops where stride is 1 */
    if (this->getFormat() == Tformat::NCHW) {
      for (unsigned int b = 0; b < batch(); ++b) {
        for (unsigned int c = 0; c < channel(); ++c) {
          for (unsigned int h = 0; h < height(); ++h) {
            _FP16 *out_data = output.getAddress<_FP16>(b, c, h, 0);
            const _FP16 *in_data = input.getAddress<_FP16>(b, c, h, 0);
            const _FP16 *_data = (_FP16 *)getAddress(getIndex(b, c, h, 0));
            std::transform(_data, _data + width(), in_data, out_data,
                           std::plus<_FP16>());
          }
        }
      }
    } else {
      for (unsigned int b = 0; b < batch(); ++b) {
        for (unsigned int h = 0; h < height(); ++h) {
          for (unsigned int w = 0; w < width(); ++w) {
            _FP16 *out_data = output.getAddress<_FP16>(b, 0, h, w);
            const _FP16 *in_data = input.getAddress<_FP16>(b, 0, h, w);
            const _FP16 *_data = (_FP16 *)getAddress(getIndex(b, 0, h, w));
            std::transform(_data, _data + channel(), in_data, out_data,
                           std::plus<_FP16>());
          }
        }
      }
    }
  }

  return output;
}

int HalfTensor::add_i_partial(unsigned int len, unsigned int addr_idx,
                              Tensor &m, unsigned int incX, unsigned int incY,
                              const Tensor alphas, unsigned int alpha_idx) {
  saxpy(len, alphas.getValue<_FP16>(alpha_idx), m.getData<_FP16>(), incX,
        (_FP16 *)getAddress(addr_idx), incY);

  return ML_ERROR_NONE;
}

Tensor &HalfTensor::add(float const &value, Tensor &output) const {
  auto f = std::bind(std::plus<_FP16>(), std::placeholders::_1,
                     static_cast<_FP16>(value));
  apply(f, output);
  return output;
}

Tensor &HalfTensor::add(Tensor const &m, Tensor &output,
                        float const alpha) const {
  auto f = [&](const BroadcastInfo &e, const _FP16 *buf, const _FP16 *m_buf,
               _FP16 *out_buf) {
    ele_add(e.buffer_size, buf, m_buf, out_buf, alpha, 0, e.strides[3],
            strides[3]);
  };
  apply_broadcast(m, f, output);
  return output;
}

Tensor &HalfTensor::subtract(float const &value, Tensor &output) const {
  auto f = std::bind(std::minus<_FP16>(), std::placeholders::_1,
                     static_cast<_FP16>(value));
  apply(f, output);
  return output;
}

void HalfTensor::sum_by_batch(Tensor &output) const {
  size_t feat_len = dim.getFeatureLen();
  size_t batch = dim.batch();

  const _FP16 *data = (_FP16 *)getData();
  _FP16 *out_data = output.getData<_FP16>();

  Tensor ones(1, 1, 1, feat_len, this->getTensorType());
  ones.setValue((_FP16)1.0);
  sgemv((unsigned int)dim.getStorageOrder(), false, batch, feat_len, 1, data,
        feat_len, ones.getData<_FP16>(), 1, 0.0, out_data, 1);
}

Tensor &HalfTensor::sum(unsigned int axis, Tensor &output, float alpha,
                        float beta) const {

  const _FP16 *data = (_FP16 *)getData();

  NNTR_THROW_IF(!contiguous, std::invalid_argument)
    << getName() << " is not contiguous, cannot sum";

  if (axis >= 4)
    throw std::out_of_range("Error: axis is invalid");

  if (dim.getDim()[axis] == 1 and alpha == 1.0 and !beta) {
    CREATE_IF_EMPTY_DIMS(output, dim);
    scopy(size(), (_FP16 *)getData(), 1, output.getData<_FP16>(), 1);
    return output;
  }

  switch (axis) {
  case 0: {
    CREATE_IF_EMPTY_DIMS(output, 1, dim.channel(), dim.height(), dim.width(),
                         this->getTensorType());
    size_t feat_len = dim.getFeatureLen();
    size_t batch = dim.batch();
    Tensor ones(1, 1, 1, batch, this->getTensorType());
    ones.setValue(alpha);
    sgemv((unsigned int)dim.getStorageOrder(), true, batch, feat_len, 1, data,
          feat_len, ones.getData<_FP16>(), 1, beta, output.getData<_FP16>(), 1);
  } break;
  case 1: {
    CREATE_IF_EMPTY_DIMS(output, dim[0], 1, dim[2], dim[3], getTensorType());
    if (this->getFormat() == Tformat::NHWC) {
      unsigned int feat_len = output.getDim().getDataLen();
      unsigned int t_axis = dim[1];
      Tensor ones(1, 1, 1, t_axis, this->getTensorType());
      ones.setValue(alpha);
      sgemv((unsigned int)dim.getStorageOrder(), false, feat_len, t_axis, 1,
            data, t_axis, ones.getData<_FP16>(), 1, beta,
            output.getData<_FP16>(), 1);
    } else {
      unsigned int feat_len = dim[2] * dim[3];
      unsigned int t_axis = dim[1];
      Tensor ones(1, 1, 1, t_axis, getTensorType());
      ones.setValue(alpha);
      _FP16 *rdata = output.getData<_FP16>();
      for (unsigned int k = 0; k < dim[0]; ++k) {
        sgemv((unsigned int)dim.getStorageOrder(), true, t_axis, feat_len, 1,
              &data[k * dim.getFeatureLen()], feat_len, ones.getData<_FP16>(),
              1, beta, &rdata[k * feat_len], 1);
      }
    }
  } break;
  case 2: {
    CREATE_IF_EMPTY_DIMS(output, dim[0], dim[1], 1, dim[3], getTensorType());

    if (this->getFormat() == Tformat::NHWC) {
      unsigned int feat_len = dim[1] * dim[3];
      unsigned int t_axis = dim[2];
      Tensor ones(1, 1, 1, t_axis, getTensorType());
      ones.setValue(alpha);
      _FP16 *rdata = output.getData<_FP16>();
      for (unsigned int k = 0; k < dim[0]; ++k) {
        sgemv((unsigned int)dim.getStorageOrder(), true, t_axis, feat_len, 1,
              &data[k * dim.getFeatureLen()], feat_len, ones.getData<_FP16>(),
              1, beta, &rdata[k * feat_len], 1);
      }
    } else {
      unsigned int t_3 = dim[3];
      unsigned int t_axis = dim[2];
      Tensor ones(1, 1, 1, t_axis, getTensorType());
      ones.setValue(alpha);
      _FP16 *rdata = output.getData<_FP16>();
      for (unsigned int k = 0; k < dim[0]; ++k) {
        for (unsigned int c = 0; c < dim[1]; ++c) {
          unsigned int idx = k * dim.getFeatureLen() + c * dim[3] * dim[2];
          unsigned int ridx = k * output.getDim().getFeatureLen() + c * dim[3];
          sgemv((unsigned int)dim.getStorageOrder(), true, t_axis, t_3, 1,
                &data[idx], t_3, ones.getData<_FP16>(), 1, beta, &rdata[ridx],
                1);
        }
      }
    }
  } break;
  case 3: {
    CREATE_IF_EMPTY_DIMS(output, dim[0], dim[1], dim[2], 1, getTensorType());
    if (this->getFormat() == Tformat::NHWC) {
      unsigned int t_3 = dim[1];
      unsigned int t_axis = dim[3];
      Tensor ones(1, 1, 1, t_axis, getTensorType());
      ones.setValue(alpha);
      _FP16 *rdata = output.getData<_FP16>();
      for (unsigned int k = 0; k < dim[0]; ++k) {
        for (unsigned int c = 0; c < dim[2]; ++c) {
          unsigned int idx = k * dim.getFeatureLen() + c * dim[3] * dim[1];
          unsigned int ridx = k * output.getDim().getFeatureLen() + c * dim[1];
          sgemv((unsigned int)dim.getStorageOrder(), true, t_axis, t_3, 1,
                &data[idx], t_3, ones.getData<_FP16>(), 1, beta, &rdata[ridx],
                1);
        }
      }
    } else {
      unsigned int m = output.getDim().getDataLen();
      unsigned int n = dim[3];
      Tensor ones(1, 1, 1, n, getTensorType());
      ones.setValue(alpha);
      sgemv((unsigned int)dim.getStorageOrder(), false, m, n, 1, data, n,
            ones.getData<_FP16>(), 1, beta, output.getData<_FP16>(), 1);
    }
  } break;
  default:
    throw std::out_of_range("Error: Dimension cannot exceed 3");
  }

  return output;
}

float HalfTensor::l2norm() const {
  return snrm2(size(), (_FP16 *)getData(), 1);
}

Tensor &HalfTensor::abs(Tensor &output) const {
  auto f = [](_FP16 in) {
    return static_cast<_FP16>(std::abs(static_cast<float>(in)));
  };
  apply(f, output);
  return output;
}

Tensor &HalfTensor::pow(float exponent, Tensor &output) const {
  auto f = [exponent](float in) {
    return static_cast<_FP16>(powf(in, exponent));
  };
  apply(f, output);
  return output;
}

Tensor &HalfTensor::sqrt(Tensor &output) const {
  auto f = [](_FP16 in) {
    return static_cast<_FP16>(std::sqrt(static_cast<float>(in)));
  };
  apply(f, output);
  return output;
}

Tensor &HalfTensor::erf(Tensor &output) const {
  auto f = [](_FP16 in) {
    return static_cast<_FP16>(std::erf(static_cast<float>(in)));
  };
  apply(f, output);
  return output;
}

void HalfTensor::tan(Tensor &output, float alpha) {
  auto f = [alpha](_FP16 in) -> _FP16 {
    return static_cast<_FP16>(std::tan(static_cast<float>(alpha * in)));
  };
  apply(f, output);
}

void HalfTensor::inv_sqrt(Tensor &out) {
  if (!contiguous) {
    apply(
      [](_FP16 val) -> _FP16 {
        return static_cast<_FP16>(1 / std::sqrt(static_cast<float>(val)));
      },
      out);
  } else {
    inv_sqrt_inplace(out.size(), out.getData<_FP16>());
  }
}

Tensor &HalfTensor::dotHalf(Tensor const &input, Tensor &output, bool trans,
                            bool trans_in, float beta) const {
  // Comment out with intension to support the calculation wrt. batch and height
  // direction. It supposes to have this->dim as [ BxCxH,W ] and input.dim is
  // [BxCxH,W] as well if (input.dim.rank() > 2) {
  //   throw exception::not_supported("Error: support only for rank of dot "
  //                                  "matrix <= 2");
  // }

  // Comment out with intension to support the calculation wrt. batch and height
  // direction of this tensor. It is OK as long as input is 2D
  if (trans && dim.rank() > 2) {
    ml_logw("Warning: support only for rank of dot matrix <= 2 with trans");
  }
  unsigned int first_three_flat, last_axis, input_first_three_flat,
    input_last_axis, M, N, K, lda, ldb, ldc;

  calculateFlattenDot(input, output, trans, trans_in, first_three_flat,
                      last_axis, input_first_three_flat, input_last_axis, M, N,
                      K, lda, ldb, ldc);

  const _FP16 *data = (_FP16 *)getData();
  const _FP16 *mdata = input.getData<_FP16>();
  _FP16 *rdata = output.getData<_FP16>();
  const float alpha = 1.0f;

  /// shortcut handling in case of vector
  /// for vector, (1 * K) == (K * 1) in current memory layout...
  /// and plaese note that N, K, M is a fixed place holder after considering
  /// transpose.
  /// For example, there is no case like (1 * K) X (1 * K) while
  /// (1 * K) X (1 * M) can be a case
  /// case1: (1 * K) X (K * 1)
  if (M == 1 && N == 1) {
    *rdata = sdot(K, data, 1, mdata, 1) +
             ((0.0f == beta) ? static_cast<_FP16>(0.0f)
                             : static_cast<_FP16>(beta) * *rdata);
  }
  /// case2: (M * K) X (K * 1)
  else if (N == 1) {
    sgemv((unsigned int)dim.getStorageOrder(), trans, first_three_flat,
          last_axis, alpha, data, lda, mdata, 1, beta, rdata, 1);
  }
  /// case3: (1 * K) X (K * N) = 1 * N = R
  /// = R^T = (K * N) ^T * (1 * K) ^T = (N * K) * (K * 1) = (N * K) * (1 * K)
  /// Effectively a translation of sgemv
  else if (M == 1) {
    sgemv((unsigned int)dim.getStorageOrder(), !trans_in,
          input_first_three_flat, input_last_axis, alpha, mdata, ldb, data, 1,
          beta, rdata, 1);
  }
  /// case others: use sgemm
  else {
    sgemm((unsigned int)dim.getStorageOrder(), trans, trans_in, M, N, K, alpha,
          data, lda, mdata, ldb, beta, rdata, ldc);
  }

  return output;
}

Tensor &HalfTensor::dot(Tensor const &input, Tensor &output, bool trans,
                        bool trans_in, float beta) const {
  switch (input.getDataType()) {
  case Tdatatype::FP16:
    dotHalf(input, output, trans, trans_in, beta);
    break;
  case Tdatatype::Q6_K:
  case Tdatatype::Q4_0:
    dotQnK(input, output, trans, trans_in, beta, input.getDataType());
    break;
  case Tdatatype::QINT4:
    dotQInteger(input, output, trans, trans_in, beta, input.getDataType());
    break;
  default:
    throw std::invalid_argument("Error: unsupported datatype");
  }
  return output;
}

Tensor &HalfTensor::dotQnK(Tensor const &input, Tensor &output, bool trans,
                           bool trans_in, float beta, Tdatatype dtype) const {
  ///@note Qn_K does not support transpose in principle. The `trans` flag
  /// here only carries dimension metadata, not a data transform.
  NNTR_THROW_IF(trans, std::invalid_argument)
    << "HalfTensor::dotQnK does not support trans";

  _FP16 *data = (_FP16 *)getData();
  uint8_t *mdata = input.getData<uint8_t>();
  _FP16 *rdata = output.getData<_FP16>();

  unsigned int M, N, K;
  M = getDim().height();
  K = getDim().width();
  N = trans_in ? input.getDim().height() : input.getDim().width();
  switch (dtype) {
  case Tdatatype::Q6_K:
    // gemm_q6_K has a `_FP16` template specialization in
    // arm_compute_backend_fp16.cpp / fallback_fp16.cpp, so this works
    // with a fp16 activation and fp16 output tensor.
    gemm_q6_K(M, N, K, data, K, (void *)mdata, N, rdata, N);
    break;
  case Tdatatype::Q4_0:
    gemm_q4_0(M, N, K, data, K, (void *)mdata, N, rdata, N);
    break;
  default:
    // Q4_K has no _FP16 specialization in cpu_backend yet. If Q4_K
    // weights ever need an fp16 activation path, we'd have to either
    // add that specialization or convert the activation to fp32
    // around the call.
    throw std::invalid_argument(
      "HalfTensor::dotQnK: unsupported weight datatype (Q4_K has no "
      "fp16 path)");
  }
  return output;
}

Tensor &HalfTensor::dotQInteger(Tensor const &input, Tensor &output, bool trans,
                                bool trans_in, float beta, Tdatatype dtype) const {
  // fp16 activation x QINT4 (channel-wise) weight.
  //
  // Mirror of FloatTensor::dotQInteger but without the fp32<->fp16 CPU
  // convert/transpose loops: `this` and `output` are already fp16, so
  // the kernel can consume the tensors directly via SVM. The only
  // per-call CPU work on this path is wrapping the SVM pointers as
  // cl_mem image objects inside gemm_int4_adreno_cl, which is what
  // the main branch's pipeline does already.
  //
  // The weight (`input`) comes from Int4Utils::attach_kai_buffer and
  // is guaranteed SVM. `this` and `output` are SVM only when the
  // Manager's tensor_pool SVM allocation succeeded (see memory_pool.cpp
  // -- on Adreno 830 this requires init_seq_len/max_seq_len to keep
  // the pool below the 1 GB per-allocation limit).
  _FP16 *data = (_FP16 *)getData();
  char *mdata = input.getData<char>();
  _FP16 *rdata = output.getData<_FP16>();

  const unsigned int M = getDim().height();
  const unsigned int K = getDim().width();
  const unsigned int N = output.getDim().width();

  // DIAG: assert output is actually an fp16 tensor. If a layer forgot to
  // propagate activation dtype to its output dim during finalize(), output
  // could secretly be an FP32 tensor and `getData<_FP16>()` would reinterpret
  // fp32 storage as fp16 -- we'd write into the wrong byte range (half the
  // buffer) and silently corrupt downstream layer inputs.
  NNTR_THROW_IF(output.getDataType() != Tdatatype::FP16, std::invalid_argument)
    << "[HalfTensor::dotQInteger] output tensor dtype mismatch: expected FP16,"
    << " got "
    << static_cast<int>(output.getDataType())
    << ". A layer's finalize() probably did not set output dim dtype to"
    << " the activation dtype. (M=" << M << " K=" << K << " N=" << N << ")";
  NNTR_THROW_IF(getDataType() != Tdatatype::FP16, std::invalid_argument)
    << "[HalfTensor::dotQInteger] *this dtype mismatch: expected FP16,"
    << " got " << static_cast<int>(getDataType());

#if defined(ENABLE_OPENCL) && ENABLE_OPENCL == 1
  const bool all_svm = input.getMemoryData()->isSVM() &&
                       output.getMemoryData()->isSVM() &&
                       getMemoryData()->isSVM();

  if (!all_svm) {
    throw std::invalid_argument(
      "HalfTensor::dotQInteger requires activation, output, and weight "
      "tensors to all live in SVM. Check that tensor_pool's "
      "createSVMRegion succeeded (see memory_pool.cpp diagnostics) and "
      "that init_seq_len/max_seq_len fit under the Adreno per-alloc "
      "limit.");
  }

  auto *weight_u16 = reinterpret_cast<uint16_t *>(mdata);
  auto *scale_u16 = input.getScale<uint16_t>();
  auto *in_u16 = reinterpret_cast<uint16_t *>(data);
  auto *out_u16 = reinterpret_cast<uint16_t *>(rdata);

  // gemm_int4_adreno_cl wraps `input` in a cl_mem via
  //   clCreateBuffer(CL_MEM_USE_HOST_PTR, size, host_ptr)
  // which requires `host_ptr` to be aligned to CL_DEVICE_MEM_BASE_ADDR_ALIGN
  // (typically a page on Adreno). The activation tensor lives at an
  // arbitrary offset inside the tensor_pool's single big SVM region
  // and is NOT aligned, so passing `data` directly fails with
  //   [!] FATAL ERROR: Failed to create input cl_mem buffer
  //
  // Stage through ClBufferManager::getSVMInput() / getSVMOutput(0),
  // which were each allocated by their own createSVMRegion call and
  // are therefore page-aligned. The fp16 -> fp16 memcpy here replaces
  // the much heavier fp32 -> fp16 + transpose loop that the FloatTensor
  // path used to do, so this is still a big net win on prefill.
  auto &clbuffInstance = ClBufferManager::Global();
  uint16_t *svm_in =
    reinterpret_cast<uint16_t *>(clbuffInstance.getSVMInput());
  uint16_t *svm_out =
    reinterpret_cast<uint16_t *>(clbuffInstance.getSVMOutput(0));

  // Element-wise scalar stage-in/out (see earlier SVM-coherency bisection
  // comment in git history -- std::memcpy and scalar loops are functionally
  // equivalent for the GPU visibility, but the scalar form keeps the code
  // symmetric with FloatTensor::dotQInteger and plays cleanly with the
  // profiler below).
  //
  // M>1 prefill path is instrumented with the per-stage profiler
  // (g_half_dotq_profile); M=1 decode path is deliberately NOT profiled
  // since per-call stage costs are tiny on decode and we want the prefill
  // bottleneck breakdown.
  if (M == 1) {
    // M=1 (decode/gen): dispatch via gpu_int4_gemv_adreno.  Three
    // stages instrumented into g_half_dotq_decode_profile: in_copy,
    // gemv_call, out_copy.
    //
    // A zero-copy variant (kernel writes out_u16 directly,
    // sync_output=false) was tried and produced garbage generations
    // despite MHACore/RMSNorm/Addition downstreams having their own
    // enqueueSVMMap fences -- the exact missing consumer fence is
    // still unidentified.  The potential 150 ms/token win is parked
    // here until the correctness regression is diagnosed; for now we
    // keep the per-call blocking sync (sync_output default = true).
    const uint64_t t_d0 = now_ns();
    for (unsigned int k = 0; k < K; ++k) {
      svm_in[k] = in_u16[k];
    }
    const uint64_t t_d1 = now_ns();
    gemv_int4_adreno_cl(svm_in, weight_u16, scale_u16, svm_out, K, N);
    const uint64_t t_d2 = now_ns();
    for (unsigned int n = 0; n < N; ++n) {
      out_u16[n] = svm_out[n];
    }
    const uint64_t t_d3 = now_ns();
    g_half_dotq_decode_profile.ns_in_copy   += t_d1 - t_d0;
    g_half_dotq_decode_profile.ns_gemv_call += t_d2 - t_d1;
    g_half_dotq_decode_profile.ns_out_copy  += t_d3 - t_d2;
    g_half_dotq_decode_profile.calls++;
  } else {
    // M>1 (prefill). Two paths:
    //   1) Delegate (preferred when N%32==0, K%8==0): zero-copy. Tensor
    //      data is SVM-backed (all_svm invariant above), and the
    //      delegate kernels use SetKernelSVMArguments which imposes no
    //      alignment requirement. Pass the tensor pointers directly —
    //      no staging, no scalar copies.
    //   2) int4: needs page-aligned staging because
    //      gemm_int4_adreno_cl wraps its input/output as cl_mem via
    //      CL_MEM_USE_HOST_PTR.
    static const bool s_disable_delegate =
      std::getenv("NNTRAINER_DISABLE_DELEGATE_GEMM") != nullptr;
    const bool use_delegate =
      !s_disable_delegate && (N % 32) == 0 && (K % 8) == 0;

    // Phase A: if an upstream image2d-aware layer (rmsnorm_image2d)
    // registered an image2d for this tensor pointer in the
    // GpuImagePool, gemm_delegate_fp16_cl will pick it up and skip
    // svm_to_image2d entirely — provided we pass the TENSOR pointer,
    // not the clbuffInstance staging buffer. Pool hit: no scalar
    // staging needed for the input side either. Pool miss: fall back
    // to the staging copy (which doubles as an Adreno SVM-coherence
    // trick — the scalar read of in_u16 triggers the cache invalidate
    // that makes subsequent GPU reads see coherent data).
    const bool pool_has_input =
      use_delegate && GpuImagePool::Global().get(in_u16) != nullptr;

    uint16_t *svm_in_T =
      reinterpret_cast<uint16_t *>(clbuffInstance.getSVMQuant(0));
    const size_t in_total = (size_t)M * K;
    const size_t out_total = (size_t)M * N;

    const uint64_t t0 = now_ns();
    uint64_t t_fence_end = t0;
    if (!pool_has_input) {
      // Upstream may be a GPU layer (e.g. swiglu_cl_fp16 when Phase B
      // GPU-SwiGLU is active). Its writes to in_u16 are still in
      // flight on the OpenCL queue, so the scalar copy below would
      // read stale host memory without a fence. Drain the queue here
      // — same pattern as the CPU-layer entry fences in
      // ReshapedRMSNorm / MHACore / SwiGLU.
      auto *cl_ctx = static_cast<ClContext *>(
        Engine::Global().getRegisteredContext("gpu"));
      if (cl_ctx) {
        cl_ctx->command_queue_inst_.enqueueSVMMap(
          in_u16, (size_t)M * K * sizeof(uint16_t), /*read_only=*/true);
      }
      t_fence_end = now_ns();
      for (size_t i = 0; i < in_total; ++i) {
        svm_in[i] = in_u16[i];
      }
    }
    const uint64_t t1 = now_ns();
    // DEBUG toggle: NNTRAINER_DEBUG_STAGE_OUT=1 forces the old staging
    // output path (svm_out scratch + blocking SVMMap + scalar copy) so
    // we can bisect whether zero-copy output is the cause of garbage.
    static const bool s_debug_stage_out =
      std::getenv("NNTRAINER_DEBUG_STAGE_OUT") != nullptr;
    if (use_delegate) {
      uint16_t *delegate_in = pool_has_input ? in_u16 : svm_in;
      uint16_t *delegate_out = s_debug_stage_out ? svm_out : out_u16;
      gemm_delegate_fp16_cl(delegate_in, svm_in_T, weight_u16, scale_u16,
                            delegate_out, M, N, K);
      if (s_debug_stage_out) {
        auto *cl_ctx = static_cast<ClContext *>(
          Engine::Global().getRegisteredContext("gpu"));
        if (cl_ctx) {
          cl_ctx->command_queue_inst_.enqueueSVMMap(
            svm_out, out_total * sizeof(uint16_t), /*read_only=*/true);
        }
      }
    } else {
      gemm_int4_adreno_cl(svm_in, svm_in_T, weight_u16, scale_u16, svm_out, M,
                          N, K);
    }
    const uint64_t t2 = now_ns();
    if (!use_delegate || s_debug_stage_out) {
      for (size_t i = 0; i < out_total; ++i) {
        out_u16[i] = svm_out[i];
      }
    }
    const uint64_t t3 = now_ns();

    g_half_dotq_profile.ns_in_fence += t_fence_end - t0;
    g_half_dotq_profile.ns_in_stage += t1 - t_fence_end;
    g_half_dotq_profile.ns_gpu_call += t2 - t1;
    g_half_dotq_profile.ns_out_stage += t3 - t2;
    g_half_dotq_profile.calls++;
  }
#else
  throw std::invalid_argument(
    "HalfTensor::dotQInteger is only implemented on the OpenCL build");
#endif

  return output;
}

void HalfTensor::dot(std::vector<Tensor *> input, std::vector<Tensor *> output,
                     bool trans, bool trans_in, float beta) const {
  // Batched FC dispatch -- used by CausalLM when Q/K/V projections (or
  // MLP up/gate) share the same activation.
  //
  // Only supports the QINT4 channel-wise path (the whole reason we're
  // on HalfTensor in the first place). For mixed / other dtype mixes
  // we fall back to the per-weight single dot.
  if (input.empty())
    return;

  Tdatatype input_dtype = input[0]->getDataType();

  // Non-QINT4 or mixed-dtype path: fall back to per-weight dot().
  if (input_dtype != Tdatatype::QINT4) {
    for (unsigned int i = 0; i < input.size(); ++i) {
      dot(*input[i], *output[i], trans, trans_in, beta);
    }
    return;
  }

#if defined(ENABLE_OPENCL) && ENABLE_OPENCL == 1
  const bool all_svm = input[0]->getMemoryData()->isSVM() &&
                       output[0]->getMemoryData()->isSVM() &&
                       getMemoryData()->isSVM();

  if (!all_svm) {
    // Fall back to per-weight dot() which throws the descriptive SVM
    // error message from HalfTensor::dotQInteger.
    for (unsigned int i = 0; i < input.size(); ++i) {
      dot(*input[i], *output[i], trans, trans_in, beta);
    }
    return;
  }

  _FP16 *data = (_FP16 *)getData();
  const unsigned int M = getDim().height();
  const unsigned int K = getDim().width();

  // DIAG: assert every output tensor is actually FP16 (see dotQInteger).
  for (unsigned int i = 0; i < output.size(); ++i) {
    NNTR_THROW_IF(output[i]->getDataType() != Tdatatype::FP16,
                  std::invalid_argument)
      << "[HalfTensor::dot(batched)] output[" << i
      << "] dtype mismatch: expected FP16, got "
      << static_cast<int>(output[i]->getDataType())
      << ". A layer's finalize() probably did not set output dim dtype to"
      << " the activation dtype.";
  }

  // Stage activation through the page-aligned scratch SVM buffer
  // (see comment in HalfTensor::dotQInteger) -- tensor_pool
  // sub-allocations are NOT aligned enough for cl_mem +
  // CL_MEM_USE_HOST_PTR. Per-weight outputs go to getSVMOutput(i).
  auto &clbuffInstance = ClBufferManager::Global();
  uint16_t *svm_in =
    reinterpret_cast<uint16_t *>(clbuffInstance.getSVMInput());
  auto *in_u16 = reinterpret_cast<uint16_t *>(data);

  // Shared activation stage-in. Only profiled on the M>1 prefill path to
  // stay consistent with the single-weight dotQInteger profiler.
  const size_t in_total = static_cast<size_t>(M) * static_cast<size_t>(K);
  const uint64_t t_in_start = (M > 1) ? now_ns() : 0;
  for (size_t i = 0; i < in_total; ++i) {
    svm_in[i] = in_u16[i];
  }
  const uint64_t t_in_end = (M > 1) ? now_ns() : 0;
  if (M > 1) {
    g_half_dotq_profile.ns_in_stage += t_in_end - t_in_start;
  }

  if (M == 1) {
    // M=1 (decode): dispatch gpu_int4_gemv_adreno per weight; the
    // shared activation has already been staged into svm_in above.
    for (unsigned int i = 0; i < input.size(); ++i) {
      auto *weight_u16 =
        reinterpret_cast<uint16_t *>(input[i]->getData<char>());
      auto *scale_u16 = input[i]->getScale<uint16_t>();
      const unsigned int Ni = output[i]->getDim().width();
      uint16_t *svm_out_i =
        reinterpret_cast<uint16_t *>(clbuffInstance.getSVMOutput(i));

      gemv_int4_adreno_cl(svm_in, weight_u16, scale_u16, svm_out_i, K, Ni);

      auto *out_u16 =
        reinterpret_cast<uint16_t *>(output[i]->getData<_FP16>());
      for (unsigned int n = 0; n < Ni; ++n) {
        out_u16[n] = svm_out_i[n];
      }
    }
  } else {
    // M>1 (prefill): one shared activation feeds every weight. Try the
    // batched delegate path first — it reformats the activation to
    // image2d ONCE (shared across all weights) and replaces N blocking
    // SVMMaps with a single end-of-batch drain. Falls back to the
    // per-weight int4 dispatch when any Ni isn't aligned for delegate.
    static const bool s_disable_delegate =
      std::getenv("NNTRAINER_DISABLE_DELEGATE_GEMM") != nullptr;
    bool all_delegate_ok = !s_disable_delegate && (K % 8) == 0;
    if (all_delegate_ok) {
      for (unsigned int i = 0; i < input.size(); ++i) {
        if ((output[i]->getDim().width() % 32) != 0) {
          all_delegate_ok = false;
          break;
        }
      }
    }

    if (all_delegate_ok) {
      std::vector<uint16_t *> weights_vec, scales_vec, outputs_vec;
      std::vector<unsigned int> Ns_vec;
      weights_vec.reserve(input.size());
      scales_vec.reserve(input.size());
      outputs_vec.reserve(input.size());
      Ns_vec.reserve(input.size());
      for (unsigned int i = 0; i < input.size(); ++i) {
        weights_vec.push_back(
          reinterpret_cast<uint16_t *>(input[i]->getData<char>()));
        scales_vec.push_back(input[i]->getScale<uint16_t>());
        outputs_vec.push_back(
          reinterpret_cast<uint16_t *>(clbuffInstance.getSVMOutput(i)));
        Ns_vec.push_back(output[i]->getDim().width());
      }

      const uint64_t t_gpu_start = now_ns();
      gemm_delegate_fp16_cl_batched(svm_in, weights_vec, scales_vec,
                                    outputs_vec, Ns_vec, M, K);
      const uint64_t t_gpu_end = now_ns();

      for (unsigned int i = 0; i < input.size(); ++i) {
        auto *out_u16 =
          reinterpret_cast<uint16_t *>(output[i]->getData<_FP16>());
        const unsigned int Ni = Ns_vec[i];
        const size_t out_total = (size_t)M * Ni;
        for (size_t j = 0; j < out_total; ++j) {
          out_u16[j] = outputs_vec[i][j];
        }
      }
      const uint64_t t_out_end = now_ns();

      g_half_dotq_profile.ns_gpu_call += t_gpu_end - t_gpu_start;
      g_half_dotq_profile.ns_out_stage += t_out_end - t_gpu_end;
      g_half_dotq_profile.calls += input.size();
    } else {
      // Fallback: per-weight int4 dispatch. GPU transpose reruns for
      // every weight because svm_in_T is overwritten.
      uint16_t *svm_in_T =
        reinterpret_cast<uint16_t *>(clbuffInstance.getSVMQuant(0));
      for (unsigned int i = 0; i < input.size(); ++i) {
        auto *weight_u16 =
          reinterpret_cast<uint16_t *>(input[i]->getData<char>());
        auto *scale_u16 = input[i]->getScale<uint16_t>();
        const unsigned int Ni = output[i]->getDim().width();
        uint16_t *svm_out_i =
          reinterpret_cast<uint16_t *>(clbuffInstance.getSVMOutput(i));

        const uint64_t t_gpu_start = now_ns();
        gemm_int4_adreno_cl(svm_in, svm_in_T, weight_u16, scale_u16,
                            svm_out_i, M, Ni, K);
        const uint64_t t_gpu_end = now_ns();

        auto *out_u16 =
          reinterpret_cast<uint16_t *>(output[i]->getData<_FP16>());
        const size_t out_total =
          static_cast<size_t>(M) * static_cast<size_t>(Ni);
        for (size_t j = 0; j < out_total; ++j) {
          out_u16[j] = svm_out_i[j];
        }
        const uint64_t t_out_end = now_ns();

        g_half_dotq_profile.ns_gpu_call += t_gpu_end - t_gpu_start;
        g_half_dotq_profile.ns_out_stage += t_out_end - t_gpu_end;
        g_half_dotq_profile.calls++;
      }
    }
  }
#else
  throw std::invalid_argument(
    "HalfTensor::dot(vector, vector) is only implemented on the OpenCL "
    "build");
#endif
}

void HalfTensor::dropout_mask(float dropout) {
  _FP16 scale = static_cast<_FP16>(1.0 / (1 - dropout));
  _FP16 *data_ = (_FP16 *)getData();
  for (unsigned int i = 0; i < size(); ++i) {
    if (data_[i] >= dropout)
      data_[i] = scale;
    else
      data_[i] = 0;
  }
}

void HalfTensor::filter_mask(const Tensor &mask_len, bool reverse) {
  float fill_mask_val = 0.0;
  float en_mask_val = 1.0 - fill_mask_val;

  if (reverse) {
    fill_mask_val = 1.0;
    en_mask_val = 1.0 - fill_mask_val;
  }

  setValue(fill_mask_val);

  NNTR_THROW_IF(mask_len.batch() != batch(), std::invalid_argument)
    << "Number of filter masks mismatched";

  for (unsigned int b = 0; b < batch(); b++) {
    _FP16 *addr = (_FP16 *)getAddress(getIndex(b, 0, 0, 0));
    const unsigned int *mask_len_val =
      mask_len.getAddress<unsigned int>(b, 0, 0, 0);
    std::fill(addr, addr + (*mask_len_val), (_FP16)en_mask_val);
  }
}

void HalfTensor::zoneout_mask(Tensor &opposite, float zoneout) {
  _FP16 zoneout_fp16 = (_FP16)zoneout;
  opposite.setRandBernoulli(zoneout_fp16);

  _FP16 *data = (_FP16 *)getData();
  _FP16 *opposite_data = opposite.getData<_FP16>();

  for (unsigned int i = 0; i < size(); ++i) {
    if (opposite_data[i] > epsilon) {
      data[i] = (_FP16)0.0;
    } else {
      data[i] = (_FP16)1.0;
    }
  }
}

std::vector<Tensor> HalfTensor::split(std::vector<size_t> sizes, int axis) {
  size_t num_size = sizes.size();

  if (axis == -1) {
    axis = 3;
  }

  size_t total_size = std::accumulate(sizes.begin(), sizes.end(), 0);
  NNTR_THROW_IF(dim.getTensorDim(axis) != total_size, std::invalid_argument)
    << "given sum of sizes did not match with origin tensor dim, tensor dim: "
    << dim.getTensorDim(axis) << " total size: " << total_size;

  std::vector<TensorDim> ret_dims(num_size, dim);
  for (unsigned int i = 0; i < num_size; ++i) {
    ret_dims[i].setTensorDim(axis, sizes[i]);
  }

  bool is_format_nchw = (dim.getFormat() == Tformat::NCHW) ? true : false;
  std::vector<Tensor> ret;

  auto iter_value = [this, is_format_nchw](
                      std::array<size_t, 4> &loc,
                      const std::array<size_t, 4> &end_loc,
                      const std::array<size_t, 4> &reset_dim_arr) -> _FP16 & {
    auto &value = (is_format_nchw) ? getValue(loc[0], loc[1], loc[2], loc[3])
                                   : getValue(loc[0], loc[3], loc[1], loc[2]);
    for (int i = 3; i >= 0; --i) {
      loc[i]++;
      if (loc[i] == end_loc[i]) {
        loc[i] -= reset_dim_arr[i];
        continue;
      }
      break;
    }
    return value;
  };

  unsigned int accumulated_size = 0;
  for (unsigned int i = 0; i < num_size; ++i) {
    std::array<size_t, 4> loc = {0, 0, 0, 0};

    if (is_format_nchw) {
      loc[axis] += accumulated_size;
    } else {
      if (axis == 0) {
        loc[0] += accumulated_size;
      } else if (axis == 1) {
        loc[3] += accumulated_size;
      } else if (axis == 2 || axis == 3) {
        loc[axis - 1] += accumulated_size;
      }
    }

    ret.push_back(Tensor(ret_dims[i]));
    auto &ret_t = ret.back();

    std::array<size_t, 4> end_loc;

    if (is_format_nchw) {
      end_loc = {ret_dims[i].batch(), ret_dims[i].channel(),
                 ret_dims[i].height(), ret_dims[i].width()};
    } else {
      end_loc = {ret_dims[i].batch(), ret_dims[i].height(), ret_dims[i].width(),
                 ret_dims[i].channel()};
    }

    accumulated_size += sizes[i];

    if (is_format_nchw) {
      end_loc[axis] = accumulated_size;
    } else {
      if (axis == 0) {
        end_loc[0] = accumulated_size;
      } else if (axis == 1) {
        end_loc[3] = accumulated_size;
      } else if (axis == 2 || axis == 3) {
        end_loc[axis - 1] = accumulated_size;
      }
    }

    std::array<size_t, 4> reset_dim_arr;
    if (is_format_nchw) {
      reset_dim_arr = {ret_dims[i].batch(), ret_dims[i].channel(),
                       ret_dims[i].height(), ret_dims[i].width()};
    } else {
      reset_dim_arr = {ret_dims[i].batch(), ret_dims[i].height(),
                       ret_dims[i].width(), ret_dims[i].channel()};
    }

    ret_t.apply_i<_FP16>(
      [&iter_value, &loc, &end_loc, &reset_dim_arr](_FP16 _) {
        return iter_value(loc, end_loc, reset_dim_arr);
      });
  }

  return ret;
}

Tensor HalfTensor::concat(const std::vector<Tensor> &tensors, int axis,
                          Tensor &output) {
  bool is_format_nchw = (tensors.front().getDim().getFormat() == Tformat::NCHW);

  auto iter_value =
    [is_format_nchw](std::array<unsigned, 4> &loc,
                     const std::array<unsigned, 4> &start_loc, Tensor &t,
                     const std::array<unsigned, 4> &ref_dim_arr) -> _FP16 & {
    auto &value = is_format_nchw
                    ? t.getValue<_FP16>(loc[0], loc[1], loc[2], loc[3])
                    : t.getValue<_FP16>(loc[0], loc[3], loc[1], loc[2]);

    for (int i = 3; i >= 0; --i) {
      loc[i]++;
      if (loc[i] - start_loc[i] == ref_dim_arr[i]) {
        loc[i] = start_loc[i];
        continue;
      }
      break;
    }
    return value;
  };

  std::array<unsigned, 4> loc = {0, 0, 0, 0};
  for (auto &t : tensors) {
    std::array<unsigned, 4> start_loc = loc;
    std::array<unsigned, 4> tensor_dim_arr;
    TensorDim curr_dim = t.getDim();

    tensor_dim_arr[0] = curr_dim.getTensorDim(0);
    tensor_dim_arr[1] =
      is_format_nchw ? curr_dim.getTensorDim(1) : curr_dim.getTensorDim(2);
    tensor_dim_arr[2] =
      is_format_nchw ? curr_dim.getTensorDim(2) : curr_dim.getTensorDim(3);
    tensor_dim_arr[3] =
      is_format_nchw ? curr_dim.getTensorDim(3) : curr_dim.getTensorDim(1);

    for (size_t i = 0u, sz = t.size(); i < sz; ++i) {
      iter_value(loc, start_loc, output, tensor_dim_arr) = t.getValue<_FP16>(i);
    }

    if (is_format_nchw) {
      loc[axis] += curr_dim.getTensorDim(axis);
    } else {
      if (axis == 0) {
        loc[0] += curr_dim.getTensorDim(axis);
      } else if (axis == 1) {
        loc[3] += curr_dim.getTensorDim(axis);
      } else if (axis == 2 || axis == 3) {
        loc[axis - 1] += curr_dim.getTensorDim(axis);
      }
    }
  }
  return output;
}

void HalfTensor::print(std::ostream &out) const {
  const _FP16 *data = (_FP16 *)getData();
  unsigned int len = size();
  out << "data addr: " << data << '\n';
  out << dim;

  if (len > 100) {
    out << '[' << (float)data[0] << ' ' << (float)data[1] << ' '
        << (float)data[2] << " ... " << (float)data[len - 3] << ' '
        << (float)data[len - 2] << ' ' << (float)data[len - 1] << ']'
        << std::endl;
    return;
  }

  std::ios init(NULL);
  init.copyfmt(out);

  if (getFormat() == Tformat::NCHW) {
    for (unsigned int k = 0; k < batch(); k++) {
      for (unsigned int l = 0; l < channel(); l++) {
        for (unsigned int i = 0; i < height(); i++) {
          for (unsigned int j = 0; j < width(); j++) {
            out << std::setw(10) << std::setprecision(10)
                << (float)data[getIndex(k, l, i, j)] << " ";
          }
          out << std::endl;
        }
        out << std::endl;
      }
      out << "-------" << std::endl;
    }
  } else {
    for (unsigned int k = 0; k < batch(); k++) {
      for (unsigned int i = 0; i < height(); i++) {
        for (unsigned int j = 0; j < width(); j++) {
          for (unsigned int l = 0; l < channel(); l++) {
            out << std::setw(10) << std::setprecision(10)
                << (float)data[getIndex(k, l, i, j)] << " ";
          }
          out << std::endl;
        }
        out << std::endl;
      }
      out << "-------" << std::endl;
    }
  }
  out.copyfmt(init);
}

Tensor &HalfTensor::divide(float const &value, Tensor &output) const {
  auto f = std::bind(std::divides<_FP16>(), std::placeholders::_1,
                     static_cast<_FP16>(value));
  apply(f, output);
  return output;
}

Tensor &HalfTensor::divide(Tensor const &m, Tensor &output) const {
  auto f = [&](const BroadcastInfo &e, const _FP16 *buf, const _FP16 *m_buf,
               _FP16 *out_buf) {
    ele_div(e.buffer_size, buf, m_buf, out_buf, 1, 0, e.strides[3], strides[3]);
  };

  apply_broadcast(m, f, output);
  return output;
}

void HalfTensor::copy(const Tensor &from) {
  reshape(from.getDim());
  copy(from.getData<_FP16>());
}

void HalfTensor::copyData(const Tensor &from) {
  if (!contiguous) {
    throw std::runtime_error("Cannot copy non-contiguous tensor");
  }

  if (size() != from.size())
    throw std::invalid_argument("Size of tensor to copy must match");

  switch (from.getDataType()) {
  case ml::train::TensorDim::DataType::FP32:
    scopy(size(), from.getData<float>(), 1, (_FP16 *)getData(), 1);
    break;
  case ml::train::TensorDim::DataType::FP16:
    copy(from.getData<_FP16>());
    break;
  case ml::train::TensorDim::DataType::QINT8:
    scopy_int8_to_float16(from.size(), from.getData<int8_t>(), 1,
                          (_FP16 *)getData(), 1);
    break;
  default:
    throw std::invalid_argument("Error: Unsupported data type");
    break;
  }
}

void HalfTensor::copy_with_stride(const Tensor &input, Tensor &output) {
  for (unsigned int b = 0; b < output.batch(); ++b) {
    for (unsigned int c = 0; c < output.channel(); ++c) {
      for (unsigned int h = 0; h < output.height(); ++h) {
        for (unsigned int w = 0; w < output.width(); ++w) {
          output.setValue(b, c, h, w, input.getValue<_FP16>(b, c, h, w));
        }
      }
    }
  }
}

std::vector<unsigned int> HalfTensor::argmax() const {
  std::vector<unsigned int> result;
  const _FP16 *data = (_FP16 *)getData();
  size_t batch_size = batch();
  size_t feature_len = dim.getFeatureLen();

  result.resize(batch_size);

  for (unsigned int b = 0; b < batch_size; b++) {
    auto max_iter =
      std::max_element(data + b * feature_len, data + (b + 1) * feature_len);
    result[b] = std::distance(data, max_iter) - (b * feature_len);
  }

  return result;
}

std::vector<unsigned int> HalfTensor::argmin() const {
  std::vector<unsigned int> result;
  const _FP16 *data = (_FP16 *)getData();
  size_t batch_size = batch();
  size_t feature_len = dim.getFeatureLen();

  result.resize(batch_size);

  for (unsigned int b = 0; b < batch_size; b++) {
    auto min_iter =
      std::min_element(data + b * feature_len, data + (b + 1) * feature_len);
    result[b] = std::distance(data, min_iter) - (b * feature_len);
  }
  return result;
}

float HalfTensor::max_abs() const {
  const _FP16 *data = (_FP16 *)getData();
  unsigned int idx = isamax(size(), data, 1);
  return (float)(*(data + idx));
}

float HalfTensor::maxValue() const {
  const _FP16 *data = (_FP16 *)getData();
  return (float)*std::max_element(data, data + size());
}

float HalfTensor::minValue() const {
  const _FP16 *data = (_FP16 *)getData();
  return (float)*std::min_element(data, data + size());
}

Tensor &HalfTensor::transpose(const std::string &direction,
                              Tensor &output) const {
  unsigned int SL, SI, SJ, SK;

  output.reshape(dim.transpose(direction));

  int indexI = direction[0] - '0';
  int indexJ = direction[2] - '0';

  SL = dim.batch(), SI = dim.channel(), SJ = dim.height(), SK = dim.width();

  bool is_format_nchw = (getFormat() == Tformat::NCHW);

  const _FP16 *inptr = (_FP16 *)getData();
  _FP16 *outptr = output.getData<_FP16>();
  switch (indexI) {
  case 0:
    if (indexJ == 1) {
      if (is_format_nchw) {
        transposeloop(l, i, j, k, SL, SI, SJ, SK);
      } else {
        transposeloop_nhwc(l, j, k, i, SL, SJ, SK, SI);
      }
    } else {
      if (is_format_nchw) {
        for (unsigned int b = 0; b < batch(); ++b) {
          for (unsigned int c = 0; c < channel(); ++c) {
            transpose_matrix(
              height(), width(), (_FP16 *)getData() + getIndex(b, c, 0, 0),
              width(), (_FP16 *)output.getData() + output.getIndex(b, c, 0, 0),
              output.width());
          }
        }
      } else {
        transposeloop_nhwc(l, k, j, i, SL, SK, SJ, SI);
      }
    }
    break;
  case 1:
    if (indexJ == 0) {
      if (is_format_nchw) {
        transposeloop(l, j, i, k, SL, SJ, SI, SK);
      } else {
        transposeloop_nhwc(l, i, k, j, SL, SI, SK, SJ);
      }
    } else {
      if (is_format_nchw) {
        transposeloop(l, j, k, i, SL, SJ, SK, SI);
      } else {
        transposeloop_nhwc(l, k, i, j, SL, SK, SI, SJ);
      }
    }
    break;
  case 2:
    if (indexJ == 0) {
      if (is_format_nchw) {
        transposeloop(l, k, i, j, SL, SK, SI, SJ);
      } else {
        transposeloop_nhwc(l, i, j, k, SL, SI, SJ, SK);
      }
    } else {
      if (is_format_nchw) {
        transposeloop(l, k, j, i, SL, SK, SJ, SI);
      } else {
        transposeloop_nhwc(l, j, i, k, SL, SJ, SI, SK);
      }
    }
    break;
  }

  return output;
}

void HalfTensor::copy(const void *buf) {
  NNTR_THROW_IF(!contiguous, std::invalid_argument)
    << getName() << " is not contiguous, cannot copy.";

  if (buf == getData()) {
    return;
  }

  scopy(size(), (_FP16 *)buf, 1, (_FP16 *)getData(), 1);
}

void HalfTensor::apply_broadcast(
  Tensor const &m,
  std::function<void(const BroadcastInfo &e, const _FP16 *, const _FP16 *,
                     _FP16 *)>
    v_func,
  Tensor &output) const {
  CREATE_IF_EMPTY_DIMS(output, dim, nullptr);

  NNTR_THROW_IF(getData() == nullptr, std::invalid_argument)
    << getName() << " is not allocated";
  NNTR_THROW_IF(m.getData<_FP16>() == nullptr, std::invalid_argument)
    << m.getName() << " is not allocated";
  NNTR_THROW_IF(output.getData<_FP16>() == nullptr, std::invalid_argument)
    << output.getName() << " is not allocated";

  /// shortcut to cover when dimension matches
  /// note that buffer_size, the last stride is only used in v_func but it
  /// might be changed
  if (dim == m.getDim()) {
    BroadcastInfo e;
    e.buffer_size = size();
    e.strides[3] = 1;
    v_func(e, (_FP16 *)getData(), m.getData<_FP16>(), output.getData<_FP16>());
    return;
  }

  return apply_broadcast_util(m, v_func, output, this->computeBroadcastInfo(m));
}

void HalfTensor::apply_broadcast_util(
  Tensor const &m,
  std::function<void(const BroadcastInfo &e, const _FP16 *, const _FP16 *,
                     _FP16 *)>
    v_func,
  Tensor &output, const BroadcastInfo &e, int cur_axis, size_t offset,
  size_t m_offset) const {

  const _FP16 *buf = (_FP16 *)this->getData();
  const _FP16 *m_buf = m.getData<_FP16>();
  _FP16 *out_buf = output.getData<_FP16>();

  if (e.buffer_axis == cur_axis) {
    v_func(e, buf + offset, m_buf + m_offset, out_buf + offset);
    return;
  }

  cur_axis++;
  for (unsigned int i = 0; i < dim.getTensorDim(cur_axis); ++i) {
    size_t next_offset = offset + i * strides[cur_axis];
    size_t next_m_offset = m_offset + i * e.strides[cur_axis];
    apply_broadcast_util(m, v_func, output, e, cur_axis, next_offset,
                         next_m_offset);
  }
}

bool HalfTensor::isValid() const {
  return is_valid(dim.getDataLen(), (_FP16 *)getData());
}

} // namespace nntrainer
