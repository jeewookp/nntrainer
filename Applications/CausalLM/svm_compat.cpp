// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * Fallback stubs for nntrainer::svm_alloc_ints, svm_free_ptr, and
 * svm_blocking_read.  Compiled into libcausallm_core.so via
 * LOCAL_WHOLE_STATIC_LIBRARIES so the binary loads even when the prebuilt
 * libnntrainer.so was built before these symbols were added.
 *
 * Definitions are STRONG (no __attribute__((weak))) so the NDK linker cannot
 * silently omit them.  If libnntrainer.so provides a strong definition it
 * wins at runtime (first-loaded DSO wins in ELF lookup order); if not, the
 * stub returns nullptr and causal_lm.cpp's if(argmax_svm) guard falls through
 * to the synchronous decode loop.
 */

#if defined(ENABLE_OPENCL) && ENABLE_OPENCL == 1
#include <cstddef>

namespace nntrainer {

// Weak stubs: strong definitions in libnntrainer.so always win via
// dlsym(RTLD_DEFAULT, ...) regardless of DSO load order.
__attribute__((weak)) int *svm_alloc_ints(int /*n*/) { return nullptr; }

__attribute__((weak)) void svm_free_ptr(void * /*p*/) {}

__attribute__((weak)) void svm_blocking_read(void * /*p*/, std::size_t /*bytes*/) {}

} // namespace nntrainer
#endif // ENABLE_OPENCL
