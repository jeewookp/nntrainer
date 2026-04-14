# ==============================================================================
# LiteRT-LM Android Orchestrator Script
# Executes ONCE in the root to prepare Phase 2 variables
# ==============================================================================

if(CMAKE_HOST_SYSTEM_NAME STREQUAL "Linux")
    set(NDK_HOST_TAG "linux-x86_64")
elseif(CMAKE_HOST_SYSTEM_NAME STREQUAL "Darwin")
    set(NDK_HOST_TAG "darwin-x86_64")
else()
    message(FATAL_ERROR "[LiteRTLM] Unsupported host OS for Android cross-compilation.")
endif()

string(REPLACE "android-" "" API_LEVEL "${ANDROID_PLATFORM}")

if(ANDROID_ABI STREQUAL "arm64-v8a")
    set(RUST_TARGET "aarch64-linux-android")
    set(CARGO_ENV "CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER")
elseif(ANDROID_ABI STREQUAL "x86_64")
    set(RUST_TARGET "x86_64-linux-android")
    set(CARGO_ENV "CARGO_TARGET_X86_64_LINUX_ANDROID_LINKER")
else()
    message(WARNING "LiteRT-LM: Unmapped Rust target for ABI: ${ANDROID_ABI}")
endif()

set(RUST_LINKER_PATH "${ANDROID_NDK_ROOT}/toolchains/llvm/prebuilt/${NDK_HOST_TAG}/bin/${RUST_TARGET}${API_LEVEL}-clang")

list(APPEND LITERTLM_TOOLCHAIN_ARGS "-DLITERTLM_RUST_LINKER_OVERRIDE=${RUST_LINKER_PATH}")
list(APPEND LITERTLM_TOOLCHAIN_ARGS "-DLITERTLM_RUST_CARGO_ENV_VAR=${CARGO_ENV}")