// SPDX-License-Identifier: Apache-2.0
/**
 * @file    cl_tensor_view.cpp
 * @date    20 May 2026
 * @brief   TensorBacking implementation. Single cl_mem + cached zero-copy
 *          image2d_from_buffer views.
 */
#include "cl_tensor_view.h"

#include <stdexcept>
#include <string>

namespace nntrainer::tv {

TensorBacking::TensorBacking(cl_context ctx, cl_mem buf, Encoding enc,
                             Layout lay, size_t bytes, bool owned)
  : ctx_(ctx), buf_(buf), enc_(enc), lay_(lay), bytes_(bytes), owned_(owned) {}

TensorBacking::~TensorBacking() {
  for (auto &kv : image_cache_) {
    if (kv.second)
      clReleaseMemObject(kv.second);
  }
  if (owned_ && buf_)
    clReleaseMemObject(buf_);
}

cl_mem TensorBacking::imageView(const ViewSpec &spec) {
  if (spec.kind == ViewKind::BUFFER)
    return buf_;

  auto it = image_cache_.find(spec);
  if (it != image_cache_.end())
    return it->second;

  cl_image_format fmt{};
  fmt.image_channel_order = spec.image_channel_order;
  fmt.image_channel_data_type = spec.image_channel_type;

  cl_image_desc desc{};
  switch (spec.kind) {
  case ViewKind::IMAGE_2D:
    desc.image_type = CL_MEM_OBJECT_IMAGE2D;
    desc.image_width = spec.width;
    desc.image_height = spec.height;
    desc.image_row_pitch = spec.row_pitch_bytes;
    break;
  case ViewKind::IMAGE_1D:
    desc.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
    desc.image_width = spec.width;
    break;
  case ViewKind::IMAGE_3D:
    desc.image_type = CL_MEM_OBJECT_IMAGE3D;
    desc.image_width = spec.width;
    desc.image_height = spec.height;
    desc.image_depth = spec.depth;
    desc.image_row_pitch = spec.row_pitch_bytes;
    desc.image_slice_pitch = spec.slice_pitch_bytes;
    break;
  case ViewKind::BUFFER:
    break; // handled above
  }
  desc.buffer = buf_; // zero-copy view over the same cl_mem

  cl_int err = CL_SUCCESS;
  cl_mem img = clCreateImage(ctx_, CL_MEM_READ_ONLY, &fmt, &desc, nullptr, &err);
  if (err != CL_SUCCESS || img == nullptr) {
    throw std::runtime_error("TensorBacking::imageView clCreateImage failed: " +
                             std::to_string(err));
  }
  image_cache_.emplace(spec, img);
  return img;
}

// =============================================================================
// ViewSpec factories. These compute the right (width, height, row_pitch,
// channel_order, channel_type) for the named layout patterns so call sites
// don't repeat the bookkeeping.
// =============================================================================

ViewSpec make_image2d_rgba_uint32(size_t byte_width, size_t byte_height) {
  ViewSpec s;
  s.kind = ViewKind::IMAGE_2D;
  s.image_channel_order = CL_RGBA;
  s.image_channel_type = CL_UNSIGNED_INT32;
  s.width = byte_width / 16; // 16 bytes per RGBA UINT32 texel
  s.height = byte_height;
  s.row_pitch_bytes = byte_width;
  return s;
}

ViewSpec make_image2d_phwc4_fp16(size_t hidden, size_t batch_seq) {
  // 4 halves (= 8 bytes) per RGBA HALF_FLOAT texel.
  ViewSpec s;
  s.kind = ViewKind::IMAGE_2D;
  s.image_channel_order = CL_RGBA;
  s.image_channel_type = CL_HALF_FLOAT;
  s.width = hidden / 4;
  s.height = batch_seq;
  s.row_pitch_bytes = hidden * sizeof(uint16_t);
  return s;
}

ViewSpec make_image2d_phwc4_int8(size_t hidden, size_t batch_seq) {
  ViewSpec s;
  s.kind = ViewKind::IMAGE_2D;
  s.image_channel_order = CL_RGBA;
  s.image_channel_type = CL_SIGNED_INT8;
  s.width = hidden / 4;
  s.height = batch_seq;
  s.row_pitch_bytes = hidden;
  return s;
}

ViewSpec make_image2d_ohwi_kcache_fp16(size_t cache_size, size_t dh) {
  // K cache rows are (one per cached token) dh fp16 values; view dh as
  // packed RGBA halves so each kernel WI fetches 4 dh-values per read.
  ViewSpec s;
  s.kind = ViewKind::IMAGE_2D;
  s.image_channel_order = CL_RGBA;
  s.image_channel_type = CL_HALF_FLOAT;
  s.width = dh / 4;
  s.height = cache_size;
  s.row_pitch_bytes = dh * sizeof(uint16_t);
  return s;
}

ViewSpec make_image2d_ohwi_vcache_fp16(size_t dh, size_t cache_size) {
  // V cache is the reversed-dim version: rows are (one per dh) the
  // cache_size axis. Same packing pattern but with cache_size as the
  // packed dim.
  ViewSpec s;
  s.kind = ViewKind::IMAGE_2D;
  s.image_channel_order = CL_RGBA;
  s.image_channel_type = CL_HALF_FLOAT;
  s.width = cache_size / 4;
  s.height = dh;
  s.row_pitch_bytes = cache_size * sizeof(uint16_t);
  return s;
}

} // namespace nntrainer::tv
