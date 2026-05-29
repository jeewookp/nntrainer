// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   qwen3_forward.cpp
 * @date   29 May 2026
 * @brief  Paper-aligned GPU-native Qwen3 forward (skeleton commit).
 */

#include "qwen3_forward.h"

#include <cl_context.h>
#include <engine.h>

#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace causallm_gpu {

Qwen3Forward::Qwen3Forward() = default;

Qwen3Forward::~Qwen3Forward() {
  if (weight_mmap_ != nullptr && weight_bytes_ > 0) {
    munmap(const_cast<uint8_t *>(weight_mmap_), weight_bytes_);
  }
  if (weight_fd_ >= 0) {
    close(weight_fd_);
  }
}

bool Qwen3Forward::init(const Qwen3Config &cfg, const std::string &weight_path) {
  cfg_ = cfg;
  weight_path_ = weight_path;

  weight_fd_ = open(weight_path.c_str(), O_RDONLY);
  if (weight_fd_ < 0) {
    std::fprintf(stderr, "[qwen3-gpu] open(%s) failed: %s\n",
                 weight_path.c_str(), std::strerror(errno));
    return false;
  }
  struct stat st;
  if (fstat(weight_fd_, &st) != 0) {
    std::fprintf(stderr, "[qwen3-gpu] fstat failed: %s\n", std::strerror(errno));
    return false;
  }
  weight_bytes_ = static_cast<size_t>(st.st_size);
  void *m = mmap(nullptr, weight_bytes_, PROT_READ, MAP_PRIVATE, weight_fd_, 0);
  if (m == MAP_FAILED) {
    std::fprintf(stderr, "[qwen3-gpu] mmap failed: %s\n", std::strerror(errno));
    weight_bytes_ = 0;
    return false;
  }
  weight_mmap_ = static_cast<const uint8_t *>(m);

  auto *cl =
    static_cast<nntrainer::ClContext *>(
      nntrainer::Engine::Global().getRegisteredContext("gpu"));
  if (cl == nullptr) {
    std::fprintf(stderr, "[qwen3-gpu] no gpu context registered\n");
    return false;
  }
  cl_ctx_ = cl->context_inst_.GetContext();
  cl_q_ = cl->command_queue_inst_.GetCommandQueue();
  cl_dev_ = cl->context_inst_.GetDeviceId();
  if (cl_ctx_ == nullptr || cl_q_ == nullptr || cl_dev_ == nullptr) {
    std::fprintf(stderr,
                 "[qwen3-gpu] ClContext handles null: ctx=%p q=%p dev=%p\n",
                 cl_ctx_, cl_q_, cl_dev_);
    return false;
  }

  std::fprintf(stderr,
               "[qwen3-gpu] init OK: weights=%s size=%zu MB cl_ctx=%p\n",
               weight_path.c_str(), weight_bytes_ / (1024 * 1024), cl_ctx_);
  std::fprintf(stderr,
               "[qwen3-gpu] cfg: hidden=%u inter=%u d=%u hQ=%u hKV=%u "
               "L=%u vocab=%u S_max=%u\n",
               cfg_.hidden_size, cfg_.intermediate_size, cfg_.head_dim,
               cfg_.num_heads_Q, cfg_.num_heads_KV, cfg_.num_layers,
               cfg_.vocab_size, cfg_.max_seq_len);
  return true;
}

void Qwen3Forward::dump_weight_header(size_t n) {
  if (weight_mmap_ == nullptr) {
    std::fprintf(stderr, "[qwen3-gpu] dump_weight_header: not mmap'd\n");
    return;
  }
  const size_t lim = (n < weight_bytes_) ? n : weight_bytes_;
  std::fprintf(stderr, "[qwen3-gpu] first %zu bytes of %s:\n", lim,
               weight_path_.c_str());
  for (size_t i = 0; i < lim; ++i) {
    std::fprintf(stderr, "%02x ", weight_mmap_[i]);
    if ((i + 1) % 16 == 0) std::fprintf(stderr, "\n");
  }
  if (lim % 16 != 0) std::fprintf(stderr, "\n");
}

bool Qwen3Forward::svm_smoke_test(size_t bytes) {
  if (cl_ctx_ == nullptr) {
    std::fprintf(stderr, "[qwen3-gpu] svm_smoke_test: no cl_ctx\n");
    return false;
  }
  void *svm = clSVMAlloc(cl_ctx_, CL_MEM_READ_WRITE, bytes, /*alignment*/ 0);
  if (svm == nullptr) {
    std::fprintf(stderr,
                 "[qwen3-gpu] clSVMAlloc(%zu) returned null — SVM may be "
                 "unsupported on this device\n", bytes);
    return false;
  }
  // Map for host write (CL_MAP_WRITE) — coarse-grained SVM requires
  // explicit map/unmap; fine-grained also accepts it as a no-op.
  cl_int err = clEnqueueSVMMap(cl_q_, CL_TRUE, CL_MAP_WRITE, svm, bytes, 0,
                               nullptr, nullptr);
  if (err != CL_SUCCESS) {
    std::fprintf(stderr, "[qwen3-gpu] clEnqueueSVMMap(WRITE) err=%d\n", err);
    clSVMFree(cl_ctx_, svm);
    return false;
  }
  uint8_t *p = static_cast<uint8_t *>(svm);
  for (size_t i = 0; i < bytes; ++i) p[i] = static_cast<uint8_t>(i & 0xFF);
  err = clEnqueueSVMUnmap(cl_q_, svm, 0, nullptr, nullptr);
  if (err != CL_SUCCESS) {
    std::fprintf(stderr, "[qwen3-gpu] clEnqueueSVMUnmap(write) err=%d\n", err);
    clSVMFree(cl_ctx_, svm);
    return false;
  }
  clFinish(cl_q_);

  err = clEnqueueSVMMap(cl_q_, CL_TRUE, CL_MAP_READ, svm, bytes, 0, nullptr,
                        nullptr);
  if (err != CL_SUCCESS) {
    std::fprintf(stderr, "[qwen3-gpu] clEnqueueSVMMap(READ) err=%d\n", err);
    clSVMFree(cl_ctx_, svm);
    return false;
  }
  bool ok = true;
  for (size_t i = 0; i < bytes; ++i) {
    if (p[i] != static_cast<uint8_t>(i & 0xFF)) {
      std::fprintf(stderr,
                   "[qwen3-gpu] svm round-trip mismatch at %zu: got 0x%02x\n",
                   i, p[i]);
      ok = false;
      break;
    }
  }
  err = clEnqueueSVMUnmap(cl_q_, svm, 0, nullptr, nullptr);
  if (err != CL_SUCCESS) {
    std::fprintf(stderr, "[qwen3-gpu] clEnqueueSVMUnmap(read) err=%d\n", err);
    ok = false;
  }
  clFinish(cl_q_);
  clSVMFree(cl_ctx_, svm);
  if (ok) {
    std::fprintf(stderr,
                 "[qwen3-gpu] SVM smoke test PASS: %zu bytes round-trip\n",
                 bytes);
  }
  return ok;
}

} // namespace causallm_gpu
