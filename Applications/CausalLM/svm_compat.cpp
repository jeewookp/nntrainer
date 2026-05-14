// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * Weak-symbol fallbacks for nntrainer::svm_alloc_ints, svm_free_ptr, and
 * svm_blocking_read.  These stubs are compiled into libcausallm_core.so so
 * that the binary links and loads even when the prebuilt libnntrainer.so was
 * built before these symbols were added.
 *
 * __attribute__((weak)) lets a strong definition from libnntrainer.so
 * override the stub at runtime; if no override is present the fallback
 * returns nullptr, which causal_lm.cpp already handles gracefully by
 * falling back to the synchronous decode loop.
 */

#if defined(ENABLE_OPENCL) && ENABLE_OPENCL == 1
#include <cstddef>

namespace nntrainer {

__attribute__((weak)) int *svm_alloc_ints(int /*n*/) { return nullptr; }

__attribute__((weak)) void svm_free_ptr(void * /*p*/) {}

__attribute__((weak)) void svm_blocking_read(void * /*p*/, std::size_t /*bytes*/) {}

} // namespace nntrainer
#endif // ENABLE_OPENCL
