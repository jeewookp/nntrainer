// SPDX-License-Identifier: Apache-2.0
// GPU Image2D pool for zero-copy layer-to-layer data passing.
// Layers write output to a pool image2d, next layer reads from it directly.

#ifndef __GPU_IMAGE_POOL_H__
#define __GPU_IMAGE_POOL_H__

#include <CL/cl.h>
#include <unordered_map>
#include <cstdint>

namespace nntrainer {

class GpuImagePool {
public:
  static GpuImagePool &Global() {
    static GpuImagePool instance;
    return instance;
  }

  // Store an image2d for a tensor (keyed by data pointer).
  // The pool does NOT own the image — caller manages lifetime.
  void set(void *tensor_data_ptr, cl_mem image, int width, int height) {
    entries_[reinterpret_cast<uintptr_t>(tensor_data_ptr)] = {image, width, height};
  }

  // Get the image2d for a tensor if available.
  // Returns nullptr if not in pool.
  cl_mem get(const void *tensor_data_ptr, int *width = nullptr, int *height = nullptr) const {
    auto it = entries_.find(reinterpret_cast<uintptr_t>(tensor_data_ptr));
    if (it == entries_.end()) return nullptr;
    if (width) *width = it->second.width;
    if (height) *height = it->second.height;
    return it->second.image;
  }

  // Remove entry (when tensor is reused for different data).
  void remove(const void *tensor_data_ptr) {
    entries_.erase(reinterpret_cast<uintptr_t>(tensor_data_ptr));
  }

  // Clear all entries (between inferences).
  void clear() { entries_.clear(); }

private:
  struct Entry {
    cl_mem image;
    int width, height;
  };
  std::unordered_map<uintptr_t, Entry> entries_;

  GpuImagePool() = default;
};

} // namespace nntrainer
#endif
