// SPDX-License-Identifier: Apache-2.0
/**
 * @file	kai4_tensor.cpp
 * @date	12 January 2026
 * @brief	This is Kai4Tensor class for Kai library integration.
 * @see		https://github.com/nnstreamer/nntrainer
 * @author h0g1 <h0g1.hong@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#include <cpu_backend.h>
#include <kai4_tensor.h>
#include <tensor.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <util_func.h>

#if defined(ENABLE_FP16) && defined(__aarch64__)
#include <cpu_backend/arm/kleidiai_interface.h>
#endif



#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>
#include <iostream>

namespace nntrainer {

Kai4Tensor::Kai4Tensor(std::string name_, Tformat fm, QScheme qscheme_) : 
  TensorBase(name_, fm, Tdatatype::QINT4), qscheme(qscheme_) {
  offset = 0;
  _idx_variant = 3; //Default idx
  transB = true;
}

Kai4Tensor::Kai4Tensor(const TensorDim &d, bool alloc_now, Initializer init,
                         std::string name, QScheme qscheme_, unsigned int idx_variant) :
  TensorBase(d, false, init, name), qscheme(qscheme_) {
  // Kai4Tensor expects 2D semantics (NxK) for packing
  // Adjust based on strict requirement if needed, essentially we want 2D
  NNTR_THROW_IF(d.batch() != 1 || d.channel() != 1, std::invalid_argument)
    << "Kai4Tensor must be 2 dimensional tensor (NxK) with batch size 1 and "
       "channel 1";
  

  if (alloc_now) {
    allocate();
  }
  offset = 0;
  _idx_variant = idx_variant; // Set idx
  transB = true;
}

Kai4Tensor::Kai4Tensor(const TensorDim &d, const void *buf) :
  Kai4Tensor(d, true, Initializer::NONE, "") {
  if (d.getDataLen() != 0) {
    if (buf != nullptr) {
      // If buf is provided, we assume it is ALREADY PACKED compatible data
      // For raw packing, use pack() explicitly
      copy_kai(buf);
    }
  }
}

void Kai4Tensor::allocate() {
  if (empty() || data)
    return;

  if (src_tensor) {
    /// allocate data based on the source tensor
    allocateSrcTensor();
 
    /** as this memory is shared, do NOT initialize */
  } else {
    /// allocate new memory for the tensor data
    MemoryData *mem_data;

    // Use calculateSize() to determine allocation size
    size_t alloc_size = size();
    
    mem_data = new MemoryData((void *)(new uint8_t[alloc_size]{}));
    data = std::shared_ptr<MemoryData>(mem_data, [](auto *mem_data) {
      delete[] mem_data->template getAddr<uint8_t>();
      delete mem_data;
    });
    offset = 0;
    initialize();
  }
}

void *Kai4Tensor::getData() const {
  if (!data)
    return nullptr;

  data->validate();
  return data->getAddr<uint8_t>() + offset;
}

void Kai4Tensor::setZero() {
  if (!data)
    return;
  uint8_t *ptr = (uint8_t *)getData();
  std::fill(ptr, ptr + size(), 0);
}

void Kai4Tensor::initialize() {
  if (empty() || !isAllocated())
    return;
  setZero();
}

size_t Kai4Tensor::size() const {
#if defined(ENABLE_FP16) && defined(__aarch64__)
  // Compute the packed size using Kai library function
  // This requires knowing packing params: nr, kr, bl
  // If not set, we can't compute size accurately
  
  // 
  return nntr_kai_get_rhs_packed_size_qsi4cxp_qs4cxs1s0(width(), height(), _idx_variant, transB);
#else
  return getDim().getDataLen();
#endif
}

size_t Kai4Tensor::getMemoryBytes() const {
  return size();
}

QScheme Kai4Tensor::q_scheme() const {
  return qscheme;  // Return the actual qscheme member variable
}



void Kai4Tensor::setKernelVariant(uint32_t variant_idx) {
  NNTR_THROW_IF(variant_idx > 7, std::invalid_argument)
    << "Kai4Tensor::setKernelVariant: variant_idx must be between 0 and 7, got " << variant_idx;
  
  _idx_variant = variant_idx;
  
  // Note: Packing parameters (nr, kr) will be retrieved from the kai interface
  // when needed via the ukernel_variants array, not stored here explicitly
}

void Kai4Tensor::copy_kai(const void *buf) {
  NNTR_THROW_IF(!contiguous, std::invalid_argument)
    << getName() << " is not contiguous, cannot copy.";

  if (buf == getData()) {
    return;
  }
  std::memcpy(getData(), buf, size());
}



} // namespace nntrainer
