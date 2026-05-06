#!/usr/bin/env bash
# temp_litert_llm_kernel_capture.sh
#
# Captures the ACTUAL CL kernel sources that LiteRT-LM dispatches during
# real LLM prefill+decode (Gemma 4 E2B). Companion to
# temp_litert_cl_intercept.sh which runs the synthetic matmul_micro_-
# benchmark instead -- and as a result captures only GEMM-tile kernels
# (program_002.cl is 32-output tile GEMM), not the M=1 GEMV that the
# decode loop actually dispatches.
#
# Output goes to runtime/cl_bench/intercepted_llm/program_*.cl so it
# does NOT clobber the matmul_micro capture. Look for kernels that
# use 1D global size + activation broadcast across N work-items --
# those are the M=1 GEMV ones.
#
# Usage (from litert_lm_worktree/):
#   ./temp_litert_llm_kernel_capture.sh
#
# Prereqs are identical to temp_litert.sh (NDK r28b+, Bazel, model,
# adb).

set -euo pipefail
export ANDROID_NDK_HOME="${ANDROID_NDK_HOME:-$HOME/neo/android-ndk-r28b}"

DEVICE_FOLDER="${DEVICE_FOLDER:-/data/local/tmp/litert_lm}"
MODEL_PATH_HOST="${MODEL_PATH:-$HOME/.cache/litert_lm_models/gemma-4-E2B-it.litertlm}"
# Tiny prefill + decode: we only need to TRIGGER kernel compilation,
# not measure timing. Each shape = 1 program_NNN.cl emitted.
PREFILL_TOKENS="${PREFILL_TOKENS:-8}"
DECODE_TOKENS="${DECODE_TOKENS:-4}"
ASYNC="${ASYNC:-false}"
TASKSET_MASK="${TASKSET_MASK:-f0}"
HOST_OUT_DIR="runtime/cl_bench/intercepted_llm"
DEVICE_CL_DIR="${DEVICE_FOLDER}/cl_intercept_llm"

NDK_FALLBACK_R28B="${HOME}/neo/android-ndk-r28b"
case "${ANDROID_NDK_HOME}" in
  *android-ndk-r2[0-7]*)
    if [ -f "${NDK_FALLBACK_R28B}/source.properties" ]; then
      export ANDROID_NDK_HOME="${NDK_FALLBACK_R28B}"
    fi ;;
esac

BAZEL=$(command -v bazelisk || command -v bazel)
[ -n "${BAZEL}" ] || { echo "[capture] bazel not found"; exit 1; }

if [ ! -f "${MODEL_PATH_HOST}" ]; then
  echo "[capture] Model not found: ${MODEL_PATH_HOST}"
  echo "[capture] Run ./download_gemma.sh first (or set MODEL_PATH=...)."
  exit 1
fi

echo "[capture] === LiteRT-LM CL Kernel Source Interceptor (REAL LLM) ==="
echo "[capture] PREFILL=${PREFILL_TOKENS} DECODE=${DECODE_TOKENS}"
echo "[capture] (small numbers -- we only need kernel compilation triggered, not real timing)"
echo ""

# ---------------------------------------------------------------------
# Build interceptor + LLM main + libLiteRt.so
# ---------------------------------------------------------------------
echo "[capture] Building cl_intercept + litert_lm_main ..."
"${BAZEL}" build \
  --config=android_arm64 \
  --define=litert_link_capi_so=true \
  --define=resolve_symbols_in_exec=false \
  //runtime/cl_bench:cl_intercept \
  //runtime/engine:litert_lm_main

INTERCEPT_SO=$(find -L bazel-bin/runtime/cl_bench -name "libcl_intercept.so" 2>/dev/null | head -1)
[ -n "${INTERCEPT_SO}" ] || { echo "[capture] libcl_intercept.so not found"; exit 1; }
LLM_BIN=bazel-bin/runtime/engine/litert_lm_main
[ -f "${LLM_BIN}" ] || { echo "[capture] litert_lm_main not built"; exit 1; }
LIBLITERT_SO=$(find -L bazel-bin -maxdepth 8 -type f -name "libLiteRt.so" \
                  ! -path "*solib*" 2>/dev/null | head -1)
[ -n "${LIBLITERT_SO}" ] || \
  LIBLITERT_SO=$(find -L bazel-bin -type f -name "libLiteRt.so" 2>/dev/null | head -1)
[ -n "${LIBLITERT_SO}" ] || { echo "[capture] libLiteRt.so not found"; exit 1; }

# ---------------------------------------------------------------------
# Push everything to device
# ---------------------------------------------------------------------
echo ""
echo "[capture] Pushing artifacts to ${DEVICE_FOLDER} ..."
adb shell "mkdir -p ${DEVICE_FOLDER} ${DEVICE_CL_DIR}"
adb shell "rm -f ${DEVICE_CL_DIR}/*.cl ${DEVICE_CL_DIR}/*.spv"

adb push "${INTERCEPT_SO}" "${DEVICE_FOLDER}/libcl_intercept.so" >/dev/null
# Save the original libOpenCL.so if the previous matmul_micro intercept
# left a renamed copy; we use LD_PRELOAD so the libOpenCL.so trick is
# not strictly needed, but mirror temp_litert_cl_intercept.sh in case
# the delegate dlopen's libOpenCL.so by name.
adb shell "if [ -f ${DEVICE_FOLDER}/libOpenCL.so ] && \
            [ ! -f ${DEVICE_FOLDER}/libOpenCL.so.bak ]; then \
            mv ${DEVICE_FOLDER}/libOpenCL.so ${DEVICE_FOLDER}/libOpenCL.so.bak; \
          fi"
adb push "${INTERCEPT_SO}" "${DEVICE_FOLDER}/libOpenCL.so" >/dev/null

adb push "${LLM_BIN}" "${DEVICE_FOLDER}/litert_lm_main" >/dev/null
adb push "${LIBLITERT_SO}" "${DEVICE_FOLDER}/libLiteRt.so" >/dev/null
adb shell "chmod +x ${DEVICE_FOLDER}/litert_lm_main"

# Required GPU prebuilts (same set as temp_litert.sh).
PREBUILT_DIR=prebuilt/android_arm64
for f in libLiteRtGpuAccelerator.so libLiteRtOpenClAccelerator.so \
         libLiteRtTopKOpenClSampler.so libLiteRtTopKWebGpuSampler.so \
         libLiteRtWebGpuAccelerator.so libGemmaModelConstraintProvider.so; do
  if [ -f "${PREBUILT_DIR}/${f}" ]; then
    adb shell "test -f ${DEVICE_FOLDER}/${f}" || \
      adb push "${PREBUILT_DIR}/${f}" "${DEVICE_FOLDER}/${f}" >/dev/null
  fi
done

# Push model only if size mismatches.
MODEL_BASENAME=$(basename "${MODEL_PATH_HOST}")
HOST_SIZE=$(stat -c '%s' "${MODEL_PATH_HOST}" 2>/dev/null || stat -f '%z' "${MODEL_PATH_HOST}")
DEV_SIZE=$(adb shell "stat -c '%s' ${DEVICE_FOLDER}/${MODEL_BASENAME} 2>/dev/null || echo 0" | tr -d '\r')
if [ "${DEV_SIZE}" = "${HOST_SIZE}" ]; then
  echo "[capture] Model already on device (${HOST_SIZE} bytes). Skipping push."
else
  echo "[capture] Pushing model (${HOST_SIZE} bytes) ..."
  adb push "${MODEL_PATH_HOST}" "${DEVICE_FOLDER}/${MODEL_BASENAME}"
fi

# Clear CL program caches so the delegate is forced to compile from
# source. Without this we'd hit clCreateProgramWithBinary instead of
# clCreateProgramWithSource and capture nothing.
echo "[capture] Clearing CL program caches (force source compilation) ..."
adb shell "rm -rf ${DEVICE_FOLDER}/cache/* 2>/dev/null; \
           rm -rf /data/local/tmp/cl_cache/* 2>/dev/null; \
           rm -rf /data/data/*/cache/cl_cache/* 2>/dev/null" 2>/dev/null || true

# ---------------------------------------------------------------------
# Run litert_lm_main with interceptor preloaded. Tiny benchmark so only
# kernel compilation runs (we don't need timing).
# Disable op profiling -- it perturbs the delegate (see temp_litert.sh
# header) and we just want raw kernel sources.
# ---------------------------------------------------------------------
echo ""
echo "[capture] Running litert_lm_main with cl_intercept (this triggers kernel compile) ..."
RUN_LOG=temp_litert_llm_kernel_capture.log
adb shell "cd ${DEVICE_FOLDER}; \
  LD_PRELOAD=./libcl_intercept.so \
  LD_LIBRARY_PATH=. \
  CL_INTERCEPT_DUMP_DIR=${DEVICE_CL_DIR} \
  taskset ${TASKSET_MASK} ./litert_lm_main \
    --backend=gpu \
    --model_path=${DEVICE_FOLDER}/${MODEL_BASENAME} \
    --benchmark=true \
    --benchmark_prefill_tokens=${PREFILL_TOKENS} \
    --benchmark_decode_tokens=${DECODE_TOKENS} \
    --async=${ASYNC} \
    --enable_op_profiling=false" \
  2>&1 | tee "${RUN_LOG}" | head -200

# ---------------------------------------------------------------------
# Restore the real libOpenCL.so so subsequent runs work.
# ---------------------------------------------------------------------
echo ""
echo "[capture] Restoring original libOpenCL.so ..."
adb shell "rm -f ${DEVICE_FOLDER}/libOpenCL.so"
adb shell "if [ -f ${DEVICE_FOLDER}/libOpenCL.so.bak ]; then \
            mv ${DEVICE_FOLDER}/libOpenCL.so.bak ${DEVICE_FOLDER}/libOpenCL.so; \
          fi"

# ---------------------------------------------------------------------
# Pull captured .cl files back.
# ---------------------------------------------------------------------
echo "[capture] Pulling captured CL sources to ${HOST_OUT_DIR}/ ..."
mkdir -p "${HOST_OUT_DIR}"
CL_LIST=$(adb shell "ls ${DEVICE_CL_DIR}/*.cl 2>/dev/null" | tr -d '\r')
SPV_LIST=$(adb shell "ls ${DEVICE_CL_DIR}/*.spv 2>/dev/null" | tr -d '\r')
TOTAL=0
for f in ${CL_LIST} ${SPV_LIST}; do
  base=$(basename "$f")
  adb pull "$f" "${HOST_OUT_DIR}/${base}" 2>/dev/null
  size=$(wc -c < "${HOST_OUT_DIR}/${base}" 2>/dev/null || echo "?")
  echo "  ${HOST_OUT_DIR}/${base} (${size} bytes)"
  TOTAL=$((TOTAL + 1))
done

echo ""
echo "=========================================="
echo " Captured ${TOTAL} CL kernel source(s) from REAL LLM run"
echo "=========================================="
echo " Location: ${HOST_OUT_DIR}/"
echo ""
echo " Look for kernels with:"
echo "   * 1D global size (M=1 GEMV pattern), or"
echo "   * 'matmul', 'gemv', 'fc', 'fully_connected' substrings"
echo "   * patterns that broadcast 1 input across many outputs"
echo ""
echo " Compare these to nntrainer's gpu_int4_gemv_adreno (current"
echo " 4.43 TPS path) to see what LiteRT does that we don't."
echo ""

# Quick triage: print the first 30 lines of each captured .cl
echo "--- triage: first 20 lines of each captured kernel ---"
for f in "${HOST_OUT_DIR}"/*.cl; do
  [ -f "$f" ] || continue
  echo ""
  echo "=== $(basename "$f") ($(wc -c < "$f") bytes) ==="
  head -20 "$f"
done
