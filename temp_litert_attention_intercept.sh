#!/usr/bin/env bash
# temp_litert_attention_intercept.sh
#
# Captures the CL kernel sources LiteRT's GPU accelerator compiles
# during a REAL LLM prefill — including (hopefully) the attention
# kernel that libLiteRtGpuAccelerator.so uses for scaled-dot-product
# / flash-attention.
#
# The existing temp_litert_cl_intercept.sh only exercises matmul via
# matmul_micro_benchmark, so no softmax/exp/reduce-style kernels ever
# get compiled. This script substitutes litert_lm_main with
# --benchmark --benchmark_prefill_tokens=<N> so the full transformer
# graph (including attention) runs once and every
# clCreateProgramWithSource() lands in $DEVICE_CL_DIR.
#
# Prereqs:
#   * adb + device connected (Adreno target)
#   * Android NDK r28b (or compatible) at $ANDROID_NDK_HOME
#   * bazel / bazelisk on PATH
#   * A LiteRT-format model (.task file) already pushed to the device
#     at $MODEL_DEV  (defaults to /data/local/tmp/litert_lm/qwen3-4b.task)
#   * libLiteRtGpuAccelerator.so + libLiteRtOpenClAccelerator.so either
#     already on the device at $DEVICE_FOLDER, or under prebuilt/
#
# Output:
#   runtime/cl_bench/intercepted_attn/program_NNN.cl  — raw captures
#   Printed summary of which captures contain attention-y tokens.
#
# Usage:
#   ./temp_litert_attention_intercept.sh
#   MODEL_DEV=/data/local/tmp/litert_lm/qwen3-4b.task \
#     PREFILL_TOKENS=437 ./temp_litert_attention_intercept.sh

export ANDROID_NDK_HOME="${ANDROID_NDK_HOME:-$HOME/neo/android-ndk-r28b}"
set -euo pipefail

DEVICE_FOLDER="${DEVICE_FOLDER:-/data/local/tmp/litert_lm}"
DEVICE_CL_DIR="${DEVICE_FOLDER}/cl_intercept"
HOST_OUT_DIR="${HOST_OUT_DIR:-runtime/cl_bench/intercepted_attn}"
MODEL_DEV="${MODEL_DEV:-${DEVICE_FOLDER}/qwen3-4b.task}"
PREFILL_TOKENS="${PREFILL_TOKENS:-437}"
DECODE_TOKENS="${DECODE_TOKENS:-1}"
TASKSET_MASK="${TASKSET_MASK:-f0}"
PREBUILT_DIR="prebuilt/android_arm64"

# Auto-bump NDK to r28b if something older was set (same pattern as
# temp_litert_cl_intercept.sh)
NDK_FALLBACK_R28B="${HOME}/neo/android-ndk-r28b"
case "${ANDROID_NDK_HOME}" in
  *android-ndk-r2[0-7]*)
    if [ -f "${NDK_FALLBACK_R28B}/source.properties" ]; then
      echo "[attn-intercept.sh] WARN: auto-upgrading NDK -> ${NDK_FALLBACK_R28B}"
      export ANDROID_NDK_HOME="${NDK_FALLBACK_R28B}"
    fi ;;
esac

BAZEL=$(command -v bazelisk || command -v bazel)
[ -n "${BAZEL}" ] || { echo "[attn-intercept.sh] bazel not found"; exit 1; }

echo "[attn-intercept.sh] === Attention CL Kernel Interceptor ==="
echo "[attn-intercept.sh] MODEL_DEV        = ${MODEL_DEV}"
echo "[attn-intercept.sh] PREFILL_TOKENS   = ${PREFILL_TOKENS}"
echo "[attn-intercept.sh] DECODE_TOKENS    = ${DECODE_TOKENS}"
echo ""

# Step 1: Build interceptor + full LLM main.
echo "[attn-intercept.sh] Building cl_intercept + litert_lm_main ..."
"${BAZEL}" build \
  --config=android_arm64 \
  --define=litert_link_capi_so=true \
  --define=resolve_symbols_in_exec=false \
  //runtime/cl_bench:cl_intercept \
  //runtime/engine:litert_lm_main

INTERCEPT_SO=$(find -L bazel-bin/runtime/cl_bench -name "libcl_intercept.so" 2>/dev/null | head -1)
[ -n "${INTERCEPT_SO}" ] \
  || { echo "[attn-intercept.sh] ERROR: libcl_intercept.so not found"; exit 1; }

MAIN_BIN=bazel-bin/runtime/engine/litert_lm_main
[ -f "${MAIN_BIN}" ] \
  || { echo "[attn-intercept.sh] ERROR: litert_lm_main not built"; exit 1; }

LIBLITERT_SO=$(find -L bazel-bin -maxdepth 8 -type f -name "libLiteRt.so" \
                  ! -path "*solib*" 2>/dev/null | head -1)
if [ -z "${LIBLITERT_SO}" ]; then
  LIBLITERT_SO=$(find -L bazel-bin -type f -name "libLiteRt.so" 2>/dev/null | head -1)
fi
[ -n "${LIBLITERT_SO}" ] \
  || { echo "[attn-intercept.sh] ERROR: libLiteRt.so not found"; exit 1; }

# Step 2: Push to device.
echo ""
echo "[attn-intercept.sh] Pushing to ${DEVICE_FOLDER} ..."
adb shell "mkdir -p ${DEVICE_FOLDER} ${DEVICE_CL_DIR}"
adb shell "rm -f ${DEVICE_CL_DIR}/*.cl"

adb push "${INTERCEPT_SO}" "${DEVICE_FOLDER}/libcl_intercept.so" >/dev/null
adb push "${MAIN_BIN}"      "${DEVICE_FOLDER}/litert_lm_main"    >/dev/null
adb push "${LIBLITERT_SO}"  "${DEVICE_FOLDER}/libLiteRt.so"      >/dev/null
adb shell "chmod +x ${DEVICE_FOLDER}/litert_lm_main"

for f in libLiteRtGpuAccelerator.so libLiteRtOpenClAccelerator.so; do
  if ! adb shell "test -f ${DEVICE_FOLDER}/${f}"; then
    if [ -f "${PREBUILT_DIR}/${f}" ]; then
      echo "[attn-intercept.sh] Pushing ${f} ..."
      adb push "${PREBUILT_DIR}/${f}" "${DEVICE_FOLDER}/${f}" >/dev/null
    else
      echo "[attn-intercept.sh] WARN: ${f} not on device nor in ${PREBUILT_DIR}"
    fi
  fi
done

if ! adb shell "test -f ${MODEL_DEV}"; then
  echo "[attn-intercept.sh] ERROR: model not found at ${MODEL_DEV}"
  echo "  Push a .task file first, e.g.:"
  echo "    adb push qwen3-4b.task ${MODEL_DEV}"
  exit 1
fi

# Clear CL program caches so the delegate is forced to compile from
# source (otherwise clCreateProgramWithBinary uses a cached blob and we
# get nothing to intercept).
echo "[attn-intercept.sh] Clearing on-device CL caches ..."
adb shell "rm -rf ${DEVICE_FOLDER}/cache/* 2>/dev/null; \
           rm -rf /data/local/tmp/cl_cache/* 2>/dev/null; \
           rm -rf /data/data/*/cache/cl_cache/* 2>/dev/null" 2>/dev/null || true

# Step 3: Run one prefill under the interceptor.
echo ""
echo "[attn-intercept.sh] Running prefill with intercept ..."
adb shell "cd ${DEVICE_FOLDER}; \
  LD_PRELOAD=./libcl_intercept.so \
  LD_LIBRARY_PATH=. \
  taskset ${TASKSET_MASK} ./litert_lm_main \
    --backend=gpu \
    --model_path=${MODEL_DEV} \
    --benchmark=true \
    --benchmark_prefill_tokens=${PREFILL_TOKENS} \
    --benchmark_decode_tokens=${DECODE_TOKENS} \
    --async=false" 2>&1 | tail -60

# Step 4: Pull captured CL files.
echo ""
echo "[attn-intercept.sh] Pulling captures ..."
mkdir -p "${HOST_OUT_DIR}"
CL_COUNT=$(adb shell "ls ${DEVICE_CL_DIR}/*.cl 2>/dev/null | wc -l" | tr -d '\r ')

if [ "${CL_COUNT}" -gt 0 ]; then
  adb shell "ls ${DEVICE_CL_DIR}/*.cl 2>/dev/null" | tr -d '\r' | while read -r f; do
    [ -z "$f" ] && continue
    base=$(basename "$f")
    adb pull "$f" "${HOST_OUT_DIR}/${base}" 2>/dev/null
  done
  echo "=========================================="
  echo " Captured ${CL_COUNT} CL program(s) -> ${HOST_OUT_DIR}/"
  echo "=========================================="
  echo ""
  echo "-- Attention-ish matches (softmax / exp( / fmax / work_group_reduce): --"
  matched=0
  for f in "${HOST_OUT_DIR}"/*.cl; do
    [ -f "$f" ] || continue
    if grep -qE 'softmax|exp\(|fmax|work_group_reduce|running_max|flash' "$f"; then
      lines=$(wc -l < "$f")
      hits=$(grep -cE 'softmax|exp\(|fmax|work_group_reduce|running_max|flash' "$f")
      echo "  $(basename "$f"): lines=${lines}, attention-hits=${hits}"
      matched=$((matched+1))
    fi
  done
  if [ "${matched}" -eq 0 ]; then
    echo "  (none — LiteRT likely runs softmax on CPU, uses XNNPACK for"
    echo "   attention, or ships a pre-compiled binary blob. Inspect"
    echo "   ${HOST_OUT_DIR}/ manually to confirm.)"
  fi
else
  echo "[attn-intercept.sh] WARN: no .cl captured — delegate may be using"
  echo "  clCreateProgramWithBinary (cached). Verify by checking"
  echo "  ${DEVICE_FOLDER}/cache and re-running after removing it."
fi
