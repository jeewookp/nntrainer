#!/usr/bin/env bash
# Run the captured delegate kernel directly.
set -euo pipefail

export ANDROID_NDK_HOME="${ANDROID_NDK_HOME:-$HOME/neo/android-ndk-r28b}"
DEVICE_FOLDER="${DEVICE_FOLDER:-/data/local/tmp/litert_lm}"
DK_SHAPE="${DK_SHAPE:-1024x6144x1536}"
DK_WARMUP="${DK_WARMUP:-5}"
DK_ITERS="${DK_ITERS:-50}"
DK_CL_FILE="${DK_CL_FILE:-cl_intercept/program_002.cl}"
TASKSET_MASK="${TASKSET_MASK:-f0}"

NDK_FALLBACK_R28B="${HOME}/neo/android-ndk-r28b"
case "${ANDROID_NDK_HOME}" in
  *android-ndk-r2[0-7]*)
    if [ -f "${NDK_FALLBACK_R28B}/source.properties" ]; then
      export ANDROID_NDK_HOME="${NDK_FALLBACK_R28B}"
    fi ;;
esac

BAZEL=$(command -v bazelisk || command -v bazel)
[ -n "${BAZEL}" ] || { echo "bazel not found"; exit 1; }

echo "[dk_bench.sh] === Delegate Kernel Direct Benchmark ==="
echo "[dk_bench.sh] SHAPE=${DK_SHAPE} CL_FILE=${DK_CL_FILE}"
echo ""

"${BAZEL}" build --config=android_arm64 //runtime/cl_bench:delegate_kernel_bench

BIN=bazel-bin/runtime/cl_bench/delegate_kernel_bench
[ -f "${BIN}" ] || { echo "Binary not found"; exit 1; }

echo ""
adb shell "mkdir -p ${DEVICE_FOLDER}"
adb push "${BIN}" "${DEVICE_FOLDER}/delegate_kernel_bench" >/dev/null
adb shell "chmod +x ${DEVICE_FOLDER}/delegate_kernel_bench"

# Ensure the captured CL file is on device
if ! adb shell "test -f ${DEVICE_FOLDER}/${DK_CL_FILE}"; then
  LOCAL_CL="runtime/cl_bench/intercepted/$(basename ${DK_CL_FILE})"
  if [ -f "${LOCAL_CL}" ]; then
    adb shell "mkdir -p ${DEVICE_FOLDER}/$(dirname ${DK_CL_FILE})"
    adb push "${LOCAL_CL}" "${DEVICE_FOLDER}/${DK_CL_FILE}" >/dev/null
    echo "[dk_bench.sh] Pushed ${LOCAL_CL} → ${DK_CL_FILE}"
  else
    echo "[dk_bench.sh] ERROR: CL file not found: ${LOCAL_CL}"
    echo "  Run ./temp_litert_cl_patch_intercept.sh first to capture kernel sources."
    exit 1
  fi
fi

# Try to lock GPU to max frequency (may need root).
echo ""
echo "[dk_bench.sh] Attempting GPU freq boost ..."
adb shell "echo performance > /sys/class/kgsl/kgsl-3d0/devfreq/governor 2>/dev/null" 2>/dev/null || true
adb shell "cat /sys/class/kgsl/kgsl-3d0/devfreq/cur_freq 2>/dev/null" 2>/dev/null || true

echo ""
echo "[dk_bench.sh] Running (taskset ${TASKSET_MASK}) ..."
adb shell "cd ${DEVICE_FOLDER}; \
  LD_LIBRARY_PATH=/system/vendor/lib64:. \
  taskset ${TASKSET_MASK} ./delegate_kernel_bench \
    --shape=${DK_SHAPE} \
    --warmup=${DK_WARMUP} \
    --iters=${DK_ITERS} \
    --cl_file=${DK_CL_FILE}" 2>&1
echo ""
