// SPDX-License-Identifier: Apache-2.0
/**
 * @file	int4_utils.cpp
 * @date	15 October 2025
 * @brief	This is Int4Utils class for utils for INT4 quantization format.
 * @see		https://github.com/nntrainer/nntrainer
 * @author	Grzegorz Kisala <gkisala@gmail.com>
 * @bug		No known bugs
 */

#include "int4_utils.h"

#include <cassert>
#include <cmath>
#include <cstring>

#include "cpu_backend.h"
#include "fp16.h"
#include "memory_data.h"
#include "nntrainer_error.h"
#include "tensor.h"
#include "util_func.h"

#include <cstdint>
#include <vector>

#if defined(ENABLE_OPENCL) && ENABLE_OPENCL == 1
#include <cl_context.h>
#include <engine.h>
#endif

namespace nntrainer {

namespace {

/// @brief Default packing parameters for the Kai qsi4cxp pipeline used by
/// the arm Q4_0 path. Matches Kai4Tensor defaults on the claude branch.
constexpr uint32_t KAI_DEFAULT_IDX_VARIANT = 3;
constexpr bool KAI_DEFAULT_TRANS_B = true;

/// @brief Kai pack parameters for idx_variant=3
/// (kai_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_8x4x32_neon_i8mm).
/// These are the values returned by ukernel.get_nr() / get_kr() / get_sr()
/// at runtime; we hardcode them here so int4_utils.cpp can stay free of
/// kai library headers (kai is ARM-only). If KAI_DEFAULT_IDX_VARIANT ever
/// changes, update these constants too.
constexpr size_t KAI_NR = 4;
constexpr size_t KAI_KR = 16;
constexpr size_t KAI_SR = 2;
constexpr size_t KAI_K_INTERLEAVED_V = 16;

/// Number of bytes per channel in the kai per-row "extras" area:
///   int32 sum + fp32 scale + fp32 bias = 12 bytes
constexpr size_t KAI_BYTES_PER_CHANNEL_EXTRAS =
  sizeof(int32_t) + sizeof(float) + sizeof(float);

inline size_t round_up(size_t v, size_t multiple) {
  return ((v + multiple - 1) / multiple) * multiple;
}

} // namespace

size_t Int4Utils::channelwise_layout_size(size_t n, size_t k) {
  NNTR_THROW_IF(k == 0 || n == 0, std::invalid_argument)
    << "channelwise_layout_size: n and k must be > 0";
  NNTR_THROW_IF(k % 4 != 0, std::invalid_argument)
    << "channelwise_layout_size: K must be a multiple of 4 (got K=" << k << ")";
  NNTR_THROW_IF(n % 32 != 0, std::invalid_argument)
    << "channelwise_layout_size: N must be a multiple of 32 (got N=" << n << ")";

  // data : (K/4) * N ushorts (kernel uses stride N; require N%32==0 so that
  //        N == align(N, 32) and the kernel's vload4 over align_N stays
  //        within the buffer).
  // scales: N fp16 values (per-channel, sequential by output channel).
  const size_t data_bytes = (k / 4) * n * sizeof(uint16_t);
  const size_t scale_bytes = n * sizeof(uint16_t);
  return data_bytes + scale_bytes;
}

void Int4Utils::convertKaiToChannelwise(const uint8_t *kai_packed, size_t n,
                                        size_t k, uint32_t idx_variant,
                                        uint16_t *out_data,
                                        uint16_t *out_scales) {
  NNTR_THROW_IF(kai_packed == nullptr, std::invalid_argument)
    << "convertKaiToChannelwise: kai_packed is null";
  NNTR_THROW_IF(out_data == nullptr || out_scales == nullptr,
                std::invalid_argument)
    << "convertKaiToChannelwise: output pointers must be non-null";
  NNTR_THROW_IF(idx_variant != KAI_DEFAULT_IDX_VARIANT, std::invalid_argument)
    << "convertKaiToChannelwise: only idx_variant="
    << KAI_DEFAULT_IDX_VARIANT << " (nr=" << KAI_NR << ", kr=" << KAI_KR
    << ", sr=" << KAI_SR << ") is supported, got " << idx_variant;
  NNTR_THROW_IF(k % 4 != 0, std::invalid_argument)
    << "convertKaiToChannelwise: K must be a multiple of 4 (got K=" << k << ")";
  NNTR_THROW_IF(n % 32 != 0, std::invalid_argument)
    << "convertKaiToChannelwise: N must be a multiple of 32 (got N=" << n
    << ")";

  // The kai packing routine (kai_run_rhs_pack_nxk_qsi4cxp_qs4cxs1s0)
  // processes weights as (n rows, k cols), groups n into nr-blocks, rounds
  // k internally up to 32, and stores per-row extras (int32 sum + fp32 scale
  // + fp32 bias) after the data section.
  const size_t k_internal = round_up(k, 32);
  const size_t dst_num_rows = round_up(n, KAI_NR) / KAI_NR;
  const size_t bytes_data_per_row = KAI_NR * (k_internal / 2);
  const size_t row_stride =
    bytes_data_per_row + KAI_NR * KAI_BYTES_PER_CHANNEL_EXTRAS;
  const size_t block_length_in_bytes = KAI_KR / KAI_SR; // = 8

  // Step 1: per-channel scales (kai stores fp32 * 0.0625; multiply by 16 to
  // recover the original scale, then convert to fp16).
  for (size_t dst_row_idx = 0; dst_row_idx < dst_num_rows; ++dst_row_idx) {
    const uint8_t *row_base = kai_packed + dst_row_idx * row_stride;
    const float *scales_in_row = reinterpret_cast<const float *>(
      row_base + bytes_data_per_row + KAI_NR * sizeof(int32_t));

    for (size_t i = 0; i < KAI_NR; ++i) {
      const size_t n_global = dst_row_idx * KAI_NR + i;
      if (n_global >= n)
        break;
      const float scale_f32 = scales_in_row[i] * 16.0f;
      out_scales[n_global] = compute_fp32_to_fp16(scale_f32);
    }
  }

  // Step 2: int4 data nibbles. Iterate kai dst bytes, decode (n0_idx,
  // k0_idx, k1_idx) per the kai pack formula, recover the original unsigned
  // bias-8 nibbles by undoing the XOR 0x88, and place each nibble at the
  // destination position used by gpu_int4_gemm_adreno:
  //
  //   out_data[(k/4) * N + n_global], bit position 4 * (k % 4)
  //
  for (size_t dst_row_idx = 0; dst_row_idx < dst_num_rows; ++dst_row_idx) {
    const uint8_t *data_section = kai_packed + dst_row_idx * row_stride;

    for (size_t dst_byte_idx = 0; dst_byte_idx < bytes_data_per_row;
         ++dst_byte_idx) {
      const size_t block_idx = dst_byte_idx / block_length_in_bytes;
      const size_t block_byte_idx = dst_byte_idx % block_length_in_bytes;
      const size_t super_block_idx = block_idx / KAI_NR;
      const size_t nr_idx = block_idx % KAI_NR;
      const size_t pos_in_super =
        super_block_idx * block_length_in_bytes + block_byte_idx;
      const size_t k_adjustment =
        (pos_in_super / KAI_K_INTERLEAVED_V) * KAI_K_INTERLEAVED_V;
      const size_t k0_idx = pos_in_super + k_adjustment;
      const size_t k1_idx = k0_idx + KAI_K_INTERLEAVED_V;
      const size_t n_global = dst_row_idx * KAI_NR + nr_idx;

      if (n_global >= n)
        continue;

      // kai on-disk byte = (orig_unsigned_bias8) XOR 0x88;
      // undo XOR to get the original nibble pair.
      const uint8_t kai_byte = data_section[dst_byte_idx];
      const uint8_t unxored = static_cast<uint8_t>(kai_byte ^ 0x88);
      const uint8_t lo_nibble = static_cast<uint8_t>(unxored & 0x0F);
      const uint8_t hi_nibble = static_cast<uint8_t>((unxored >> 4) & 0x0F);

      if (k0_idx < k) {
        const size_t dst_idx = (k0_idx / 4) * n + n_global;
        out_data[dst_idx] |=
          static_cast<uint16_t>(lo_nibble) << (4 * (k0_idx % 4));
      }
      if (k1_idx < k) {
        const size_t dst_idx = (k1_idx / 4) * n + n_global;
        out_data[dst_idx] |=
          static_cast<uint16_t>(hi_nibble) << (4 * (k1_idx % 4));
      }
    }
  }
}

size_t Int4Utils::attach_kai_buffer(Tensor &weight) {
  // Allocates the **destination** SVM buffer in the channel-wise layout
  // expected by gpu_int4_gemm_adreno (NOT the kai-packed layout). The kai
  // bytes are read into a temporary heap buffer in kai_to_int4(), converted
  // via convertKaiToChannelwise(), and discarded.
  const size_t n = weight.width();
  const size_t k = weight.height();
  const size_t dst_size = channelwise_layout_size(n, k);

  NNTR_THROW_IF(dst_size == 0, std::invalid_argument)
    << "[Int4Utils::attach_kai_buffer] channel-wise layout size is 0 for "
       "weight of dim "
    << weight.getDim();

  void *raw = nullptr;
  bool is_svm = false;

#if defined(ENABLE_OPENCL) && ENABLE_OPENCL == 1
  auto *cl_ctx =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  if (cl_ctx != nullptr) {
    raw = cl_ctx->context_inst_.createSVMRegion(dst_size);
    if (raw != nullptr) {
      std::memset(raw, 0, dst_size);
      is_svm = true;
    }
  }
#endif

  if (raw == nullptr) {
    raw = static_cast<void *>(new uint8_t[dst_size]());
  }

  auto *mem = new MemoryData(raw);
  if (is_svm) {
    // friend access -- Int4Utils is a friend of MemoryData (memory_data.h)
    mem->setSVM(true);
    std::shared_ptr<MemoryData> mem_data(mem, [](MemoryData *m) {
#if defined(ENABLE_OPENCL) && ENABLE_OPENCL == 1
      auto *cc =
        static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
      if (cc != nullptr) {
        cc->context_inst_.releaseSVMRegion(
          static_cast<void *>(m->getAddr<uint8_t>()));
      }
#endif
      delete m;
    });
    weight.setData(mem_data, /*off=*/0, /*init=*/false);
  } else {
    std::shared_ptr<MemoryData> mem_data(mem, [](MemoryData *m) {
      delete[] m->getAddr<uint8_t>();
      delete m;
    });
    weight.setData(mem_data, /*off=*/0, /*init=*/false);
  }

  return dst_size;
}

namespace {

/// Read the kai-packed bytes for one weight from `file` into `kai_buf` and
/// then convert them into the channel-wise int4 layout in the tensor's
/// already-attached SVM destination buffer. Used by both kai_to_int4
/// overloads to share the conversion logic.
#ifdef ENABLE_FP16
void read_and_convert_kai_to_channelwise(Tensor &weight, const uint8_t *kai_buf,
                                         size_t n, size_t k) {
  uint8_t *dst_bytes = weight.getData<uint8_t>();
  NNTR_THROW_IF(dst_bytes == nullptr, std::invalid_argument)
    << "[Int4Utils::kai_to_int4] destination buffer not attached";

  // Layout in the SVM destination buffer:
  //   [0 .. data_bytes)             : ushort-packed int4 nibbles
  //   [data_bytes .. data+scales)   : fp16 per-channel scales
  // Matches Int4QTensor::getScale() == data + (size+1)/2 for size = K*N
  // (since K*N is even when K%4==0).
  const size_t data_bytes = (k / 4) * n * sizeof(uint16_t);
  uint16_t *out_data = reinterpret_cast<uint16_t *>(dst_bytes);
  uint16_t *out_scales = reinterpret_cast<uint16_t *>(dst_bytes + data_bytes);

  Int4Utils::convertKaiToChannelwise(kai_buf, n, k, KAI_DEFAULT_IDX_VARIANT,
                                     out_data, out_scales);
}
#endif // ENABLE_FP16

} // namespace

void Int4Utils::kai_to_int4(Tensor &weight, std::ifstream &file,
                            size_t start_offset, bool read_from_offset) {
#ifndef ENABLE_FP16
  NNTR_THROW_IF(true, std::runtime_error)
    << "kai_to_int4 requires ENABLE_FP16";
#else
  attach_kai_buffer(weight);

  const size_t n = weight.width();
  const size_t k = weight.height();
  const size_t kai_size = nntr_get_rhs_packed_size_qsi4cxp_qs4cxs1s0(
    n, k, KAI_DEFAULT_IDX_VARIANT, KAI_DEFAULT_TRANS_B);
  NNTR_THROW_IF(kai_size == 0, std::invalid_argument)
    << "[Int4Utils::kai_to_int4] kai-packed size is 0 for weight of dim "
    << weight.getDim();

  // Temporary heap buffer for the on-disk kai-packed bytes.
  std::vector<uint8_t> kai_buf(kai_size);

  const std::streamsize sz = static_cast<std::streamsize>(kai_size);
  NNTR_THROW_IF(sz < 0, std::invalid_argument)
    << "[Int4Utils::kai_to_int4] read size: " << kai_size
    << " is too big. It cannot be represented by std::streamsize";

  checkedRead(file, reinterpret_cast<char *>(kai_buf.data()), sz,
              "[Int4Utils::kai_to_int4] failed to read Kai-format QINT4 "
              "weight",
              start_offset, read_from_offset);

  read_and_convert_kai_to_channelwise(weight, kai_buf.data(), n, k);
#endif // ENABLE_FP16
}

void Int4Utils::kai_to_int4(Tensor &weight, ReadSource src, size_t start_offset,
                            bool read_from_offset) {
#ifndef ENABLE_FP16
  NNTR_THROW_IF(true, std::runtime_error)
    << "kai_to_int4 requires ENABLE_FP16";
#else
  attach_kai_buffer(weight);

  const size_t n = weight.width();
  const size_t k = weight.height();
  const size_t kai_size = nntr_get_rhs_packed_size_qsi4cxp_qs4cxs1s0(
    n, k, KAI_DEFAULT_IDX_VARIANT, KAI_DEFAULT_TRANS_B);
  NNTR_THROW_IF(kai_size == 0, std::invalid_argument)
    << "[Int4Utils::kai_to_int4] kai-packed size is 0 for weight of dim "
    << weight.getDim();

  std::vector<uint8_t> kai_buf(kai_size);

  const std::streamsize sz = static_cast<std::streamsize>(kai_size);
  NNTR_THROW_IF(sz < 0, std::invalid_argument)
    << "[Int4Utils::kai_to_int4] read size: " << kai_size
    << " is too big. It cannot be represented by std::streamsize";

  checkedRead(src, reinterpret_cast<char *>(kai_buf.data()), sz,
              "[Int4Utils::kai_to_int4] failed to read Kai-format QINT4 "
              "weight",
              start_offset, read_from_offset);

  read_and_convert_kai_to_channelwise(weight, kai_buf.data(), n, k);
#endif // ENABLE_FP16
}

float Int4Utils::computeScaleForGroup(const float *group_weights,
                                      const size_t group_size) {
  auto max_absolute_weight = 0.0f;

  for (size_t i = 0; i < group_size; ++i) {
    auto weight = group_weights[i];

    NNTR_THROW_IF(!std::isfinite(weight), std::invalid_argument)
      << "Weight is not finite value";

    const auto absolute_weight = std::abs(weight);

    if (absolute_weight > max_absolute_weight) {
      max_absolute_weight = absolute_weight;
    }
  }

  auto group_scale =
    (max_absolute_weight == 0.0f) ? 1.0f : (max_absolute_weight / 7.0f);

  NNTR_THROW_IF(!std::isfinite(group_scale), std::invalid_argument)
    << "Scale is not finite value";

  return group_scale;
}

void Int4Utils::computeScales(const float *weights, const size_t rows_count,
                              const size_t columns_count,
                              const size_t group_size,
                              std::vector<float> &scales) {
  // NNTR_THROW_IF(columns_count % group_size, std::invalid_argument)
  //   << "Columns size not divisible by group size";
  NNTR_THROW_IF(columns_count % 4, std::invalid_argument)
    << "Columns size not divisible by 4";

  const auto full_groups_per_row = columns_count / group_size;
  const auto last_group_size = columns_count % group_size;
  const auto padded_groups_per_row = ceilDiv(columns_count, group_size);
  const auto rows_count_pad = align(rows_count, ROW_BLOCK_SIZE);
  scales.resize(rows_count_pad * padded_groups_per_row, 1.0f);

  for (size_t row_id = 0; row_id < rows_count; ++row_id) {
    const auto *weights_row = weights + (row_id * columns_count);

    for (size_t group_id = 0; group_id < full_groups_per_row; ++group_id) {
      const auto *weights_group = weights_row + (group_id * group_size);
      scales[(group_id * rows_count_pad) + row_id] =
        computeScaleForGroup(weights_group, group_size);
    }

    // Compute scale for the last padded group
    if (last_group_size > 0) {
      const auto *weights_group =
        weights_row + (full_groups_per_row * group_size);
      scales[(full_groups_per_row * rows_count_pad) + row_id] =
        computeScaleForGroup(weights_group, last_group_size);
    }
  }
}

uint8_t Int4Utils::pack(const float *weights, const float *scales,
                        const size_t row_id, const size_t column_id,
                        const size_t groups_per_row, const size_t group_size,
                        const size_t rows_count, const size_t columns_count,
                        const bool convert_with_add8) {
  {
    const auto rows_count_pad = align(rows_count, ROW_BLOCK_SIZE);
    const float scale =
      scales[row_id + ((column_id / group_size) * rows_count_pad)];
    const float weight = weights[(row_id * columns_count) + column_id];
    return quantizeToInt4(weight, scale, convert_with_add8);
  }
}

void Int4Utils::quantizeAndRepack(const float *weights, const size_t rows_count,
                                  const size_t columns_count,
                                  const size_t group_size,
                                  std::vector<uint8_t> &out_weights,
                                  std::vector<uint16_t> &out_scales) {
  NNTR_THROW_IF(!weights, std::invalid_argument) << "Weight cannot be null";

  NNTR_THROW_IF((rows_count <= 0), std::invalid_argument)
    << "Rows count needs to be greater than 0";

  NNTR_THROW_IF((columns_count <= 0), std::invalid_argument)
    << "Columns count needs to be greater than 0";

  NNTR_THROW_IF((!(group_size == 32 || group_size == 64 || group_size == 128)),
                std::invalid_argument)
    << "Group size must be 32/64/128";

  std::vector<float> scales_fp32;
  computeScales(weights, rows_count, columns_count, group_size, scales_fp32);

  out_scales.resize(scales_fp32.size());
  for (size_t scale_id = 0; scale_id < scales_fp32.size(); ++scale_id) {
    out_scales[scale_id] = compute_fp32_to_fp16(scales_fp32[scale_id]);
  }

  NNTR_THROW_IF(columns_count % COLUMN_BLOCK_SIZE, std::invalid_argument)
    << "Columns size not divisible by column block size";

  // Prepare output buffer in OS_IS_YX_OSV32_ISV2 layout
  const auto groups_per_row = ceilDiv(columns_count, group_size);
  const auto row_blocks_count = ceilDiv(rows_count, ROW_BLOCK_SIZE);
  const auto columns_count_pad = align(columns_count, group_size);
  const auto column_blocks_count =
    ceilDiv(columns_count_pad, COLUMN_BLOCK_SIZE);
  const auto rows_count_pad = row_blocks_count * ROW_BLOCK_SIZE;

  out_weights.resize((rows_count_pad * columns_count_pad) / 2, 0);

  size_t out_idx = 0;

  for (size_t row_block_id = 0; row_block_id < row_blocks_count;
       ++row_block_id) {
    for (size_t column_block_id = 0; column_block_id < column_blocks_count;
         ++column_block_id) {
      for (size_t i = 0; i < ROW_BLOCK_SIZE; ++i) {
        uint8_t lo = 0, hi = 0;
        const auto row_id_absolute = (row_block_id * ROW_BLOCK_SIZE) + i;
        if (row_id_absolute < rows_count) {
          const auto column_id_absolute_lo =
            (column_block_id * COLUMN_BLOCK_SIZE);
          if (column_id_absolute_lo < columns_count) {
            lo = pack(weights, scales_fp32.data(), row_id_absolute,
                      column_id_absolute_lo, groups_per_row, group_size,
                      rows_count, columns_count);

            const auto column_id_absolute_hi = column_id_absolute_lo + 1;
            if (column_id_absolute_hi < columns_count) {
              hi = pack(weights, scales_fp32.data(), row_id_absolute,
                        column_id_absolute_hi, groups_per_row, group_size,
                        rows_count, columns_count);
            }
          }
        }

        out_weights[out_idx++] = uint8_t((hi << 4) | lo);
      }
    }
  }
}

void Int4Utils::quantizeAndRepackSimpleLayout(
  const float *weights, const size_t N, const size_t K,
  const size_t group_size, std::vector<uint16_t> &out_weights,
  std::vector<uint16_t> &out_scales) {
  // 4 nibbles per ushort, laid out as [ceilDiv(K, 4)][N] for the
  // gpu_int4_gemm_adreno kernel. See Int4Utils::convertKaiToChannelwise
  // and runtime/util/int4_gemm_adreno.cl for the reader side.
  constexpr size_t COLUMN_BLOCK_SIZE_ADRENO = 4;

  NNTR_THROW_IF(!weights, std::invalid_argument) << "Weight cannot be null";

  NNTR_THROW_IF((N <= 0), std::invalid_argument)
    << "Rows count needs to be greater than 0";

  NNTR_THROW_IF((K <= 0), std::invalid_argument)
    << "Columns count needs to be greater than 0";

  NNTR_THROW_IF((!(group_size == 32 || group_size == 64 || group_size == 128)),
                std::invalid_argument)
    << "Group size must be 32/64/128";

  std::vector<float> scales_fp32;
  computeScales(weights, N, K, group_size, scales_fp32);

  out_scales.resize(scales_fp32.size());
  for (size_t scale_id = 0; scale_id < scales_fp32.size(); ++scale_id) {
    out_scales[scale_id] = compute_fp32_to_fp16(scales_fp32[scale_id]);
  }

  const auto K_pad = align(K, group_size);
  const auto K_pack_count = ceilDiv(K_pad, COLUMN_BLOCK_SIZE_ADRENO);

  out_weights.assign(N * K_pack_count, 0);

  for (size_t k = 0; k < K_pack_count; ++k) {
    for (size_t n = 0; n < N; ++n) {
      uint16_t bits0 = 0, bits1 = 0, bits2 = 0, bits3 = 0;
      const auto k_id = k * COLUMN_BLOCK_SIZE_ADRENO;
      if (k_id < K) {
        bits0 = pack(weights, scales_fp32.data(), n, k_id, 0, group_size, N, K,
                     /*convert_with_add8=*/true);
      }
      if (k_id + 1 < K) {
        bits1 = pack(weights, scales_fp32.data(), n, k_id + 1, 0, group_size,
                     N, K, /*convert_with_add8=*/true);
      }
      if (k_id + 2 < K) {
        bits2 = pack(weights, scales_fp32.data(), n, k_id + 2, 0, group_size,
                     N, K, /*convert_with_add8=*/true);
      }
      if (k_id + 3 < K) {
        bits3 = pack(weights, scales_fp32.data(), n, k_id + 3, 0, group_size,
                     N, K, /*convert_with_add8=*/true);
      }

      out_weights[k * N + n] =
        uint16_t((bits3 << 12) | (bits2 << 8) | (bits1 << 4) | bits0);
    }
  }
}

uint8_t Int4Utils::quantizeToInt4(const float weight, const float scale,
                                   const bool convert_with_add8) {
  // convert_with_add8 = true  : signed [-8, 7] -> unsigned bias-8 [0, 15]
  //                             (encoding consumed by gpu_int4_gemm_adreno)
  // convert_with_add8 = false : two's-complement nibble [0..7, 8..15]
  //                             (encoding consumed by osv32_isv2)
  auto div = std::nearbyintf(weight / scale);

  if (std::isnan(div)) {
    div = 0.0f;
  }

  div = std::clamp(div, -8.0f, 7.0f);
  int quantized = (int)div;
  if (convert_with_add8) {
    return uint8_t(quantized + 8);
  }
  return uint8_t(quantized & 0xF);
}

int Int4Utils::convertInt4ToInt(const uint8_t int4_value) {
  static int lookup[] = {0,  1,  2,  3,  4,  5,  6,  7,
                         -8, -7, -6, -5, -4, -3, -2, -1};

  return lookup[int4_value];
}

void Int4Utils::dequantizePacked(const std::vector<uint8_t> &weights,
                                 const std::vector<uint16_t> &scales,
                                 const size_t rows_count,
                                 const size_t columns_count,
                                 const size_t group_size,
                                 std::vector<float> &dequantized_weights) {
  const auto groups_per_row = ceilDiv(columns_count, group_size);
  const auto rows_count_pad = align(rows_count, ROW_BLOCK_SIZE);
  const auto row_blocks_count = ceilDiv(rows_count, ROW_BLOCK_SIZE);
  const auto columns_count_pad = align(columns_count, group_size);
  const auto column_blocks_count =
    ceilDiv(columns_count_pad, COLUMN_BLOCK_SIZE);

  dequantized_weights.resize(rows_count * columns_count);

  size_t weights_idx = 0;

  for (size_t row_block_id = 0; row_block_id < row_blocks_count;
       ++row_block_id) {
    for (size_t column_block_id = 0; column_block_id < column_blocks_count;
         ++column_block_id) {
      for (size_t i = 0; i < ROW_BLOCK_SIZE; ++i) {
        uint8_t lo = 0, hi = 0;
        const auto row_id_absolute = (row_block_id * ROW_BLOCK_SIZE) + i;
        if (row_id_absolute < rows_count) {
          const auto column_id_absolute_lo =
            (column_block_id * COLUMN_BLOCK_SIZE);
          if (column_id_absolute_lo < columns_count) {
            const auto column_id_absolute_hi = column_id_absolute_lo + 1;

            const auto scale_lo =
              scales[row_id_absolute +
                     ((column_id_absolute_lo / group_size) * rows_count_pad)];

            const auto scale_hi =
              scales[row_id_absolute +
                     ((column_id_absolute_hi / group_size) * rows_count_pad)];

            const auto weight = weights[weights_idx];
            const auto weight_lo = weight & 0xF;
            const auto weight_hi = (weight >> 4) & 0xF;

            dequantized_weights[(row_id_absolute * columns_count) +
                                column_id_absolute_lo] =
              Int4Utils::convertInt4ToInt(weight_lo) *
              nntrainer::compute_fp16_to_fp32(scale_lo);

            if (column_id_absolute_hi < columns_count) {
              dequantized_weights[(row_id_absolute * columns_count) +
                                  column_id_absolute_hi] =
                Int4Utils::convertInt4ToInt(weight_hi) *
                nntrainer::compute_fp16_to_fp32(scale_hi);
            }
          }
        }
        weights_idx++;
      }
    }
  }
}

void Int4Utils::dequantizePackedRow(uint8_t *weights, uint16_t *scales,
                                    const size_t rows_count,
                                    const size_t columns_count,
                                    const size_t group_size,
                                    const size_t row_index,
                                    float *dequantized_row) {
  // --- Validate ---
  NNTR_THROW_IF(rows_count == 0 || columns_count == 0, std::invalid_argument)
    << "rows_count and columns_count must be > 0";
  NNTR_THROW_IF(row_index >= rows_count, std::out_of_range)
    << "row_index out of range";
  NNTR_THROW_IF(!(group_size == 32 || group_size == 64 || group_size == 128),
                std::invalid_argument)
    << "group_size must be 32/64/128";

  // --- Layout ---
  const size_t rows_count_pad = align(rows_count, ROW_BLOCK_SIZE);
  const size_t columns_count_pad = align(columns_count, group_size);
  const size_t column_blocks_count =
    ceilDiv(columns_count_pad, COLUMN_BLOCK_SIZE); // COLUMN_BLOCK_SIZE == 2
  const size_t padded_groups_per_row = ceilDiv(columns_count, group_size);

  // Address the bytes for this row
  const size_t row_block_id = row_index / ROW_BLOCK_SIZE;
  const size_t i_in_block = row_index % ROW_BLOCK_SIZE;
  const size_t bytes_per_row_block_span = column_blocks_count * ROW_BLOCK_SIZE;
  const size_t row_block_base =
    row_block_id * bytes_per_row_block_span + i_in_block;

  for (size_t column_block_id = 0; column_block_id < column_blocks_count;
       ++column_block_id) {
    const size_t weights_idx =
      row_block_base + column_block_id * ROW_BLOCK_SIZE;
    const uint8_t packed_byte = weights[weights_idx];

    const size_t col_lo = column_block_id * COLUMN_BLOCK_SIZE;
    const size_t col_hi = col_lo + 1;

    const int q_lo = Int4Utils::convertInt4ToInt(packed_byte & 0xF);
    const int q_hi = Int4Utils::convertInt4ToInt((packed_byte >> 4) & 0xF);

    if (col_lo < columns_count) {
      const size_t g_lo = col_lo / group_size;
      const float s_lo = nntrainer::compute_fp16_to_fp32(
        scales[row_index + g_lo * rows_count_pad]);
      dequantized_row[col_lo] = static_cast<float>(q_lo) * s_lo;
    }
    if (col_hi < columns_count) {
      const size_t g_hi = col_hi / group_size;
      const float s_hi = nntrainer::compute_fp16_to_fp32(
        scales[row_index + g_hi * rows_count_pad]);
      dequantized_row[col_hi] = static_cast<float>(q_hi) * s_hi;
    }
  }
}

void Int4Utils::dequantizePackedRow32ToInt4Scale(
  const uint8_t *weights, const uint16_t *scales, const size_t rows_count,
  const size_t columns_count, const size_t group_size, const size_t row_index,
  const size_t column_index, uint8_t *weight_int4_row32, uint16_t *scale) {
  // --- Validate ---
  NNTR_THROW_IF(rows_count == 0 || columns_count == 0, std::invalid_argument)
    << "rows_count and columns_count must be > 0";
  NNTR_THROW_IF(row_index >= rows_count, std::out_of_range)
    << "row_index out of range";
  NNTR_THROW_IF(!(group_size == 32 || group_size == 64 || group_size == 128),
                std::invalid_argument)
    << "group_size must be 32/64/128";
  NNTR_THROW_IF(columns_count % 32 != 0, std::invalid_argument)
    << "columns_count must be divisible by 32";

  // --- Layout ---
  const size_t rows_count_pad = align(rows_count, ROW_BLOCK_SIZE);
  const size_t columns_count_pad = align(columns_count, group_size);
  const size_t column_blocks_count =
    ceilDiv(columns_count_pad, COLUMN_BLOCK_SIZE); // COLUMN_BLOCK_SIZE == 2
  const size_t padded_groups_per_row = ceilDiv(columns_count, group_size);

  // Address the bytes for this row
  const size_t row_block_id = row_index / ROW_BLOCK_SIZE;
  const size_t i_in_block = row_index % ROW_BLOCK_SIZE;
  const size_t bytes_per_row_block_span = column_blocks_count * ROW_BLOCK_SIZE;
  const size_t row_block_base =
    row_block_id * bytes_per_row_block_span + i_in_block;

  for (size_t column_block_id = 0; column_block_id < 16; ++column_block_id) {
    const size_t weights_idx =
      row_block_base + (column_index / 2 + column_block_id) * ROW_BLOCK_SIZE;
    const uint8_t packed_byte = weights[weights_idx];

    weight_int4_row32[column_block_id] = packed_byte;
  }

  *scale = scales[row_index + (column_index / group_size) * rows_count_pad];
}
} // namespace nntrainer
