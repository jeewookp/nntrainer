// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Debadri Samaddar <s.debadri@samsung.com>
 *
 * @file    opencl_context_manager.cpp
 * @date    06 Feb 2024
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Debadri Samaddar <s.debadri@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   OpenCL wrapper for context management
 *
 */

#include "opencl_context_manager.h"

#include <cstdio>
#include <cstring>
#include <iostream>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "CL/cl.h"
#include "opencl_loader.h"

#include <nntrainer_log.h>

namespace nntrainer::opencl {

/**
 * @brief Get the OpenCL context object
 *
 * @return const cl_context
 */
const cl_context &ContextManager::GetContext() {
  // loading the OpenCL library and required functions
  bool result = LoadOpenCL();

  if (!result) {
    context_ = nullptr;
    return context_;
  }

  // context_ is created + retained exactly once inside CreateCLContext()
  // (called from the branch below on first access) and kept alive until
  // ~ContextManager() runs at process exit. Every callsite of GetContext()
  // just borrows the pointer — they never balance it with clReleaseContext,
  // so the old per-call clRetainContext here was a ref-count leak AND a
  // per-call driver-lock cost (~290us on Adreno 830). Skip it.
  if (context_) {
    return context_;
  }

  do {
    result = CreateDefaultGPUDevice();
    if (!result) {
      break;
    }

    result = CreateCLContext();
    if (!result) {
      break;
    }

    // increments the context reference count
    clRetainContext(context_);

  } while (false);

  if (!result) {
    ml_loge("Failed to create OpenCL Context");
    context_ = nullptr;
  }

  return context_;
}

void ContextManager::ReleaseContext() {
  if (context_) {
    // decrements the context reference count
    clReleaseContext(context_);
  }
}

/**
 * @brief Get the Device Id object
 *
 * @return const cl_device_id
 */
const cl_device_id ContextManager::GetDeviceId() { return device_id_; }

// Tracks raw (unaligned) SVM pointers by their page-aligned counterpart.
// Qualcomm Adreno drivers ignore the alignment hint in clSVMAlloc and may
// return addresses that are not page-aligned.  We over-allocate by PAGE_SZ,
// manually advance to the next page boundary, and store the raw pointer here
// so releaseSVMRegion can pass the correct address to clSVMFree.
static std::unordered_map<void *, void *> s_svm_aligned_to_raw;
static std::mutex s_svm_map_mutex;

void *ContextManager::createSVMRegion(size_t size) {
  if (!context_) return nullptr;

  static const bool s_svm_fine_grain =
    std::getenv("NNTRAINER_SVM_FINE_GRAIN") != nullptr;
  cl_svm_mem_flags flags = CL_MEM_READ_WRITE;
  if (s_svm_fine_grain) {
    flags |= CL_MEM_SVM_FINE_GRAIN_BUFFER;
  }

  // Over-allocate by PAGE_SZ so we can align the returned pointer upward.
  // transformer.cpp's MAP_FIXED path requires pool_ptr % PAGE_SZ == file_pos
  // % PAGE_SZ; a page-aligned pool base guarantees this for any tensor offset
  // that itself matches the file layout.
  static const cl_uint PAGE_SZ = 4096u;
  const size_t alloc_size = size + PAGE_SZ;

  void *raw = clSVMAlloc(context_, flags, alloc_size, PAGE_SZ);
  if (!raw) {
    raw = clSVMAlloc(context_, flags, alloc_size, 0);
  }
  if (!raw && s_svm_fine_grain) {
    raw = clSVMAlloc(context_, CL_MEM_READ_WRITE, alloc_size, PAGE_SZ);
    if (!raw) raw = clSVMAlloc(context_, CL_MEM_READ_WRITE, alloc_size, 0);
  }
  if (!raw) return nullptr;

  // Align upward to the next page boundary.
  uintptr_t raw_addr     = reinterpret_cast<uintptr_t>(raw);
  uintptr_t aligned_addr = (raw_addr + PAGE_SZ - 1) & ~(uintptr_t)(PAGE_SZ - 1);
  void *aligned          = reinterpret_cast<void *>(aligned_addr);

  if (aligned != raw) {
    std::lock_guard<std::mutex> lk(s_svm_map_mutex);
    s_svm_aligned_to_raw[aligned] = raw;
  }

  ml_logi("createSVMRegion: size=%zu raw=%p aligned=%p inpage=%zu",
          size, raw, aligned, aligned_addr % PAGE_SZ);

  return aligned;
}

void ContextManager::releaseSVMRegion(void *svm_ptr) {
  if (!svm_ptr) {
    ml_logw("Attempted to deallocate a null pointer");
    return;
  }

  // If we aligned the pointer in createSVMRegion, free the original raw ptr.
  void *to_free = svm_ptr;
  {
    std::lock_guard<std::mutex> lk(s_svm_map_mutex);
    auto it = s_svm_aligned_to_raw.find(svm_ptr);
    if (it != s_svm_aligned_to_raw.end()) {
      to_free = it->second;
      s_svm_aligned_to_raw.erase(it);
    }
  }
  clSVMFree(context_, to_free);
}

/**
 * @brief Destroy the Context Manager object
 *
 */
ContextManager::~ContextManager() {
  if (context_) {
    // decrements the context reference count
    clReleaseContext(context_);
    context_ = nullptr;
  }
}

/**
 * @brief Create a Default GPU Device object
 *
 * @return true if successful or false otherwise
 */
bool ContextManager::CreateDefaultGPUDevice() {
  std::vector<std::pair<cl_platform_id, cl_device_id>> platform_device_pairs;

  platform_id_ = nullptr;
  device_id_ = nullptr;

  static constexpr cl_device_type kDefaultQueryDeviceType = CL_DEVICE_TYPE_GPU;

  ml_logi("Collecting OpenCL platforms ...");

  cl_uint num_platforms = 0;
  cl_int status = clGetPlatformIDs(0, nullptr, &num_platforms);
  if (status != CL_SUCCESS) {
    ml_loge("clGetPlatformIDs returned %d : %s", status,
            OpenCLErrorCodeToString(status));
    return false;
  }
  if (num_platforms == 0) {
    ml_loge("No supported OpenCL platform.");
    return false;
  }

  std::vector<cl_platform_id> platforms(num_platforms);
  status = clGetPlatformIDs(num_platforms, platforms.data(), nullptr);
  if (status != CL_SUCCESS) {
    ml_loge("clGetPlatformIDs returned %d : %s", status,
            OpenCLErrorCodeToString(status));
    return false;
  }

  for (size_t i = 0; i < static_cast<size_t>(num_platforms); i++) {
    ml_logi("Collecting OpenCL devices for platform %d / %d ...", (int32_t)i,
            (int32_t)num_platforms);

    cl_uint num_devices = 0;
    status = clGetDeviceIDs(platforms[i], kDefaultQueryDeviceType, 0, nullptr,
                            &num_devices);
    if (status != CL_SUCCESS) {
      ml_loge("clGetDeviceIDs returned %d : %s", status,
              OpenCLErrorCodeToString(status));
      continue;
    }
    if (num_devices == 0) {
      ml_loge("No GPU on current platform.");
      continue;
    }

    std::vector<cl_device_id> devices(num_devices);
    status = clGetDeviceIDs(platforms[i], kDefaultQueryDeviceType, num_devices,
                            devices.data(), nullptr);
    if (status != CL_SUCCESS) {
      ml_loge("clGetDeviceIDs returned %d : %s", status,
              OpenCLErrorCodeToString(status));
      continue;
    }

    for (size_t j = 0; j < static_cast<size_t>(num_devices); j++) {
      platform_device_pairs.push_back(
        std::make_pair<>(platforms[i], devices[j]));
    }
  }

  ml_logi("Looking for suitable OpenCL platform and device ...");

  // Vendor ID of Intel : 0x8086
  // Vendor ID of NVidia : 0x10DE / 0x13B5
  constexpr static cl_uint intel_igpu_vendor_id = 0x8086;
  constexpr static cl_device_type intel_igpu_device_type = CL_DEVICE_TYPE_GPU;
  constexpr static const char *const intel_igpu_device_name_pfx = "Intel";

#define SEARCH_BY_NAME 1

  for (const std::pair<cl_platform_id, cl_device_id> &platform_device :
       platform_device_pairs) {
    cl_platform_id platform = platform_device.first;
    cl_device_id device = platform_device.second;

    auto device_info = std::make_unique<DeviceInfo>();
    if (!device_info->read(device)) {
      ml_loge("Failed to read device info");
      return false;
    }

    const bool type_check =
      (device_info->getDeviceType() == intel_igpu_device_type);

#if SEARCH_BY_NAME
    std::string device_name_query(intel_igpu_device_name_pfx);

    const bool vendor_check = (device_info->getDeviceName().find(
                                 device_name_query) != std::string::npos);
#else
    const bool vendor_check =
      (device_info->getDeviceVendorID() == intel_igpu_vendor_id);
#endif

#undef SEARCH_BY_NAME

    if (vendor_check && type_check) {
      platform_id_ = platform;
      device_id_ = device;
      device_info_ = std::move(device_info);
      break;
    }
  }

  if ((nullptr == platform_id_) || (nullptr == device_id_)) {
    if (platform_device_pairs.empty()) {
      ml_loge("No suitable platforms / device found - aborting OpenCL context "
              "creation.");
      return false;
    }
    ml_loge("No suitable platform / device found - using default (first)");
    platform_id_ = platform_device_pairs[0].first;
    device_id_ = platform_device_pairs[0].second;
    device_info_ = std::make_unique<DeviceInfo>();
    if (!device_info_->read(device_id_)) {
      ml_loge("Failed to read device info");
      return false;
    }
  }

  // Raport device name
  ml_logi("Using device %s", device_info_->getDeviceName().data());
  device_info_->print();

  // Stage E (cl_khr_command_buffer + cl_qcom_recordable_queues)
  // probes ran several iterations and concluded both record/replay
  // paths are blocked on this driver:
  //   - cl_khr_command_buffer: extension absent, entrypoints NULL.
  //   - cl_qcom_recordable_queues: extension advertised, entrypoints
  //     resolved, but ALL clCreateCommandQueueWithProperties variants
  //     (16k+ tested across the full QCOM hex range and legacy bit-
  //     mask) reject every layout, AND libCB.so internally carries a
  //     "Recordable queues not supported by device" error string --
  //     driver-level gate.
  // Pivoted to LiteRT-style kernel-level vendor optimizations from
  // captured program_002.cl. Probes removed; loader keeps the
  // entrypoints in case a future driver update unlocks them.

  return true;
}

/**
 * @brief Create OpenCL context
 *
 * @return true if successful or false otherwise
 */
bool ContextManager::CreateCLContext() {
  int error_code;
  cl_context_properties properties[] = {
    CL_CONTEXT_PLATFORM, (cl_context_properties)platform_id_, 0};

  // creating valid ARM GPU OpenCL context, will return NULL with error code if
  // fails
  context_ =
    clCreateContext(properties, 1, &device_id_, nullptr, nullptr, &error_code);
  if (!context_) {
    ml_loge("Failed to create a compute context. OpenCL error code: %d : %s",
            error_code, OpenCLErrorCodeToString(error_code));
    return false;
  }

  return true;
}
} // namespace nntrainer::opencl
