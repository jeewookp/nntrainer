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
#include <string>
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

void *ContextManager::createSVMRegion(size_t size) {
  if (context_)
    return clSVMAlloc(context_, CL_MEM_READ_WRITE, size, 0);
  else
    return nullptr;
}

void ContextManager::releaseSVMRegion(void *svm_ptr) {
  if (svm_ptr) {
    // deallocates the SVM memory
    clSVMFree(context_, svm_ptr);
  } else {
    ml_logw("Attempted to deallocate a null pointer");
  }
  svm_ptr = nullptr;
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

  // Stage E (cl_khr_command_buffer) entrypoint probe. Even when the
  // extension is advertised in CL_DEVICE_EXTENSIONS, some drivers stub
  // the symbols; resolving via clGetExtensionFunctionAddressForPlatform
  // is the only reliable way to know the runtime actually exposes them.
  if (clGetExtensionFunctionAddressForPlatform) {
    void *p_create =
      clGetExtensionFunctionAddressForPlatform(platform_id_,
                                               "clCreateCommandBufferKHR");
    void *p_finalize = clGetExtensionFunctionAddressForPlatform(
      platform_id_, "clFinalizeCommandBufferKHR");
    void *p_enqueue = clGetExtensionFunctionAddressForPlatform(
      platform_id_, "clEnqueueCommandBufferKHR");
    void *p_ndrange = clGetExtensionFunctionAddressForPlatform(
      platform_id_, "clCommandNDRangeKernelKHR");
    void *p_release = clGetExtensionFunctionAddressForPlatform(
      platform_id_, "clReleaseCommandBufferKHR");
    void *p_update = clGetExtensionFunctionAddressForPlatform(
      platform_id_, "clUpdateMutableCommandsKHR");
    std::fprintf(stderr,
                 "[CMDBUF_PROBE] entrypoints: create=%p finalize=%p "
                 "enqueue=%p ndrange=%p release=%p update_mutable=%p\n",
                 p_create, p_finalize, p_enqueue, p_ndrange, p_release,
                 p_update);
    std::fflush(stderr);
  }

  // Stage E phase 2: cl_qcom_recordable_queues smoke probe.
  // The Adreno 830 driver advertises this vendor extension in lieu of
  // cl_khr_command_buffer. Verify (a) dlsym actually resolved the four
  // recordable-QCOM entrypoints, and (b) we can create a recordable
  // command queue + open/close an empty recording on it. Litert's
  // delegate_kernel_bench tries 3 different property layouts; we do
  // the same here and report which (if any) the driver accepted.
  std::fprintf(
    stderr,
    "[QCOM_REC_PROBE] entrypoints: NewRecordingQCOM=%p EndRecordingQCOM=%p "
    "EnqueueRecordingQCOM=%p ReleaseRecordingQCOM=%p "
    "CreateCommandQueueWithProperties=%p\n",
    (void *)clNewRecordingQCOM, (void *)clEndRecordingQCOM,
    (void *)clEnqueueRecordingQCOM, (void *)clReleaseRecordingQCOM,
    (void *)clCreateCommandQueueWithProperties);

  if (clNewRecordingQCOM && clEndRecordingQCOM && clEnqueueRecordingQCOM &&
      clReleaseRecordingQCOM && clCreateCommandQueueWithProperties) {
    // We need a context to test queue creation. Build a minimal one
    // bound to platform_id_ + device_id_; release at end of probe.
    cl_context_properties ctx_props[] = {
      CL_CONTEXT_PLATFORM, (cl_context_properties)platform_id_, 0};
    cl_int probe_err = 0;
    cl_context probe_ctx = clCreateContext(ctx_props, 1, &device_id_, nullptr,
                                           nullptr, &probe_err);
    if (!probe_ctx) {
      std::fprintf(stderr,
                   "[QCOM_REC_PROBE] minimal context creation failed err=%d\n",
                   probe_err);
    } else {
      struct Variant {
        const char *name;
        cl_queue_properties props[6];
      };
      const Variant variants[] = {
        // Litert "Attempt 1": just the QCOM property, no other flags.
        {"v1: { RECORDABLE_QCOM=1 }",
         {CL_QUEUE_RECORDABLE_QCOM, 1, 0, 0, 0, 0}},
        // "Attempt 2": as a CL_QUEUE_PROPERTIES bitmask (likely wrong --
        // 0x40E6 doesn't fit in cl_command_queue_properties bits).
        {"v2: CL_QUEUE_PROPERTIES=RECORDABLE_QCOM",
         {CL_QUEUE_PROPERTIES, CL_QUEUE_RECORDABLE_QCOM, 0, 0, 0, 0}},
        // Likely-correct: PROFILING + RECORDABLE as separate property
        // pairs. Profiling is enabled because our normal queue uses it
        // and we want the recordable replays to be timeable too.
        {"v3: PROFILING + RECORDABLE_QCOM=1",
         {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE,
          CL_QUEUE_RECORDABLE_QCOM, 1, 0, 0}},
        // v4-v6 variants tried after v1-v3 failed with err=-30 on Adreno
        // 830 / Compiler E031.47.18.13. The issue may be queue-properties
        // gating (some drivers need OoO + RECORDABLE), or non-1 enable
        // value (CL_TRUE numerically is 1 so equivalent), or it may be
        // that the literal hex 0x40E6 is wrong on this driver vintage.
        {"v4: OoO + RECORDABLE_QCOM=1",
         {CL_QUEUE_PROPERTIES, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE,
          CL_QUEUE_RECORDABLE_QCOM, 1, 0, 0}},
        {"v5: OoO + PROFILING + RECORDABLE_QCOM=1",
         {CL_QUEUE_PROPERTIES,
          CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_PROFILING_ENABLE,
          CL_QUEUE_RECORDABLE_QCOM, 1, 0}},
        {"v6: PROPERTIES=0 + RECORDABLE_QCOM=1",
         {CL_QUEUE_PROPERTIES, 0, CL_QUEUE_RECORDABLE_QCOM, 1, 0, 0}},
      };
      cl_command_queue rec_queue = nullptr;
      const char *winning_variant = nullptr;
      for (const auto &v : variants) {
        cl_int err = 0;
        cl_command_queue q = clCreateCommandQueueWithProperties(
          probe_ctx, device_id_, v.props, &err);
        std::fprintf(stderr, "[QCOM_REC_PROBE] %s -> queue=%p err=%d\n",
                     v.name, (void *)q, err);
        if (q && err == CL_SUCCESS && !rec_queue) {
          rec_queue = q;
          winning_variant = v.name;
        } else if (q) {
          // Only the first successful variant is kept; release the rest.
          clReleaseCommandQueue(q);
        }
      }
      // If no symbolic-name variant worked, brute-force the QCOM
      // property hex range 0x40D0..0x40EF. Some drivers reuse the
      // 0x40E0 block for different queue properties across releases;
      // the litert comment says 0x40E6 is the right value but this
      // driver clearly disagrees. 32 calls is cheap (~3 ms total).
      if (!rec_queue) {
        for (cl_queue_properties name = 0x40D0; name <= 0x40EF; ++name) {
          if (name == CL_QUEUE_RECORDABLE_QCOM) {
            continue; // already tested via v1
          }
          cl_queue_properties props[3] = {name, 1, 0};
          cl_int err = 0;
          cl_command_queue q = clCreateCommandQueueWithProperties(
            probe_ctx, device_id_, props, &err);
          if (q && err == CL_SUCCESS) {
            std::fprintf(stderr,
                         "[QCOM_REC_PROBE] HEX_SCAN name=0x%04lx -> queue=%p "
                         "err=%d (CANDIDATE)\n",
                         (unsigned long)name, (void *)q, err);
            if (!rec_queue) {
              rec_queue = q;
              static char buf[64];
              std::snprintf(buf, sizeof(buf), "hex_scan: 0x%04lx",
                            (unsigned long)name);
              winning_variant = buf;
            } else {
              clReleaseCommandQueue(q);
            }
          } else if (err != CL_INVALID_VALUE && err != -30) {
            // -30 (CL_INVALID_VALUE) is the spammy "wrong property"
            // case; everything else is interesting and worth logging.
            std::fprintf(stderr,
                         "[QCOM_REC_PROBE] HEX_SCAN name=0x%04lx -> "
                         "err=%d (unusual)\n",
                         (unsigned long)name, err);
          }
        }
      }
      if (rec_queue) {
        cl_int rec_err = 0;
        void *recording = clNewRecordingQCOM(rec_queue, &rec_err);
        std::fprintf(stderr,
                     "[QCOM_REC_PROBE] winning=\"%s\" "
                     "clNewRecordingQCOM=%p err=%d\n",
                     winning_variant, recording, rec_err);
        if (recording && rec_err == CL_SUCCESS) {
          // No NDRange enqueued -- empty recording. End it to verify
          // the full open/close path works without a captured kernel.
          cl_int end_err = clEndRecordingQCOM(recording);
          std::fprintf(stderr,
                       "[QCOM_REC_PROBE] clEndRecordingQCOM (empty) err=%d\n",
                       end_err);
          clReleaseRecordingQCOM(recording);
        }
        clReleaseCommandQueue(rec_queue);
      } else {
        std::fprintf(stderr,
                     "[QCOM_REC_PROBE] no property variant produced a "
                     "recordable queue -- Stage E blocked on this driver\n");
      }
      clReleaseContext(probe_ctx);
    }
    std::fflush(stderr);
  }

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
