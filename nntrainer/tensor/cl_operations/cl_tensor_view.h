// SPDX-License-Identifier: Apache-2.0
/**
 * @file    cl_tensor_view.h
 * @date    20 May 2026
 * @brief   Paper-style tensor virtualization (minimal): a single cl_mem
 *          backing + multiple zero-copy views (buffer + image2d_from_buffer).
 *          Designed to grow: more Encoding/Layout values added as new ops
 *          come online (attention/RMSNorm/SwiGLU/etc.).
 *
 * Background: arXiv:2505.00232 (ML Drift) §3.1 — tensor virtualization
 * decouples the physical allocation from the per-op view (image1D / image2D /
 * image3D / buffer). We start with image2d_from_buffer over a single cl_mem
 * for v8c FC; extend as needed.
 */
#ifndef __NNTRAINER_CL_TENSOR_VIEW_H__
#define __NNTRAINER_CL_TENSOR_VIEW_H__

#include <CL/cl.h>
#include <cstddef>
#include <cstdint>
#include <unordered_map>

namespace nntrainer::tv {

/**
 * @brief In-memory data encoding. Bits at rest.
 */
enum class Encoding : uint8_t {
  FP32,
  FP16,
  INT8,
  INT4_OFFSET, ///< 4-bit values stored as (value+8) in low 4 bits of each byte
  INT4_2COMP,  ///< 4-bit values in 2's complement (low 4 bits of each byte)
  // grow as needed
};

/**
 * @brief Element axis order / packing convention. Selecting the right
 *        layout for a tensor determines how kernels see its memory.
 */
enum class Layout : uint8_t {
  ROW_MAJOR,   ///< [outer][inner] linear (e.g. [N][K] for weight, [M][K] for act)
  OSV32_ISV2,  ///< Int4QTensor disk layout (osv32 + 2's-comp nibbles)
  PHWC4,       ///< paper §3.1 activation layout: [P, H, W, C/4] image2d-friendly
  OHWI,        ///< paper §3.8 K-cache layout: [cache_size, 1, 1, dh] weight-form
  OHWI_T,      ///< paper §3.8 V-cache layout: [dh, 1, 1, cache_size] reversed
};

/**
 * @brief Physical view kind. Paper §3.2 lists buffer + image1d + image2d
 *        + image3d + texture array. We currently implement BUFFER and
 *        IMAGE_2D; IMAGE_1D / IMAGE_3D enumerated for future use.
 */
enum class ViewKind : uint8_t {
  BUFFER,    ///< raw cl_mem buffer view (zero-copy, always available)
  IMAGE_1D,  ///< cl_image1d (future — paper "1D image buffers")
  IMAGE_2D,  ///< cl_image2d_from_buffer (implemented)
  IMAGE_3D,  ///< cl_image3d (future)
};

/**
 * @brief Image view spec. kind=BUFFER means raw buffer (no image params).
 *        For IMAGE_2D: width × height texels, row_pitch in bytes.
 *        For IMAGE_3D: width × height × depth (depth used only when
 *        kind == IMAGE_3D).
 */
struct ViewSpec {
  ViewKind kind = ViewKind::BUFFER;
  cl_channel_order image_channel_order = CL_RGBA;
  cl_channel_type image_channel_type = CL_UNSIGNED_INT32;
  size_t width = 0;
  size_t height = 0;
  size_t depth = 0;
  size_t row_pitch_bytes = 0;
  size_t slice_pitch_bytes = 0;

  bool operator==(const ViewSpec &o) const {
    return kind == o.kind &&
           image_channel_order == o.image_channel_order &&
           image_channel_type == o.image_channel_type &&
           width == o.width && height == o.height && depth == o.depth &&
           row_pitch_bytes == o.row_pitch_bytes &&
           slice_pitch_bytes == o.slice_pitch_bytes;
  }
};
struct ViewSpecHash {
  size_t operator()(const ViewSpec &s) const noexcept {
    size_t h = static_cast<size_t>(s.kind);
    h = h * 131 + s.image_channel_order;
    h = h * 131 + s.image_channel_type;
    h = h * 131 + s.width;
    h = h * 131 + s.height;
    h = h * 131 + s.depth;
    h = h * 131 + s.row_pitch_bytes;
    h = h * 131 + s.slice_pitch_bytes;
    return h;
  }
};

/**
 * @brief ViewSpec factory helpers for the common cases. Each one returns
 *        a ViewSpec ready to pass to TensorBacking::imageView. Encoding
 *        is read from the backing's metadata to pick the right channel type.
 *
 * make_image2d_rgba_uint32: packed view for v8c-style 16-byte texels
 *   (8 halves or 16 int8 per texel). Used by the v8c FC weight image.
 *   - byte_width and byte_height count BYTES in the source buffer; the
 *     image width is byte_width / 16.
 */
ViewSpec make_image2d_rgba_uint32(size_t byte_width, size_t byte_height);

/**
 * @brief Image2d view of a PHWC4 activation tensor (paper §3.1).
 *
 * The source buffer is `[B*S, hidden]` fp16 / int8 etc. We tile the
 * hidden axis into 4-channel texel chunks (RGBA), with one texel row per
 * (B, S) position.
 *   - For fp16 (2 bytes per element):
 *     channel_type = CL_HALF_FLOAT, RGBA = 8 bytes per texel = 4 halves.
 *     width = hidden / 4, height = B * S, row_pitch = hidden * 2.
 *   - For int8: channel_type = CL_SIGNED_INT8, width = hidden / 4.
 */
ViewSpec make_image2d_phwc4_fp16(size_t hidden, size_t batch_seq);
ViewSpec make_image2d_phwc4_int8(size_t hidden, size_t batch_seq);

/**
 * @brief Image2d view of an OHWI K-cache (paper §3.8). The K cache is
 *        laid out as [cache_size, dh] for one head; we view dh as 4 RGBA
 *        channels per texel.
 */
ViewSpec make_image2d_ohwi_kcache_fp16(size_t cache_size, size_t dh);

/**
 * @brief Image2d view of an OHWI-reversed V-cache. V is [dh, cache_size];
 *        we view cache_size as the packed dim (RGBA).
 */
ViewSpec make_image2d_ohwi_vcache_fp16(size_t dh, size_t cache_size);

/**
 * @brief One physical cl_mem allocation + metadata + cache of zero-copy views.
 *        Buffer view is always available; image views are created on first
 *        request via image2d_from_buffer and cached.
 */
class TensorBacking {
public:
  /**
   * @brief Construct with an existing cl_mem buffer (TensorBacking takes
   *        ownership unless owned=false).
   */
  TensorBacking(cl_context ctx, cl_mem buf, Encoding enc, Layout lay,
                size_t bytes, bool owned = true);
  ~TensorBacking();

  TensorBacking(const TensorBacking &) = delete;
  TensorBacking &operator=(const TensorBacking &) = delete;

  /// raw buffer view (zero-copy, always available)
  cl_mem buffer() const { return buf_; }
  /// image2d view over the same backing (zero-copy via image2d_from_buffer)
  cl_mem imageView(const ViewSpec &spec);

  Encoding encoding() const { return enc_; }
  Layout layout() const { return lay_; }
  size_t bytes() const { return bytes_; }

private:
  cl_context ctx_;
  cl_mem buf_;
  Encoding enc_;
  Layout lay_;
  size_t bytes_;
  bool owned_;
  std::unordered_map<ViewSpec, cl_mem, ViewSpecHash> image_cache_;
};

} // namespace nntrainer::tv

#endif // __NNTRAINER_CL_TENSOR_VIEW_H__
