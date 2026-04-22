#!/usr/bin/env bash
# Build and run the delegate conv_wave kernel unit test on Android device.
#
# Usage:
#   sh unittest_delegate_conv_wave.sh
#   sh unittest_delegate_conv_wave.sh --build-only
#   sh unittest_delegate_conv_wave.sh --run-only
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

BUILD_ONLY=0
RUN_ONLY=0
DEVICE_DIR="${DEVICE_DIR:-/data/local/tmp/nntr_android_test}"
TEST_BIN_NAME="unittest_delegate_conv_wave"
KERNEL_CL="nntrainer/tensor/cl_operations/cl_kernels/delegate_conv_wave.cl"

while [ $# -gt 0 ]; do
  case "$1" in
    --build-only) BUILD_ONLY=1; shift ;;
    --run-only)   RUN_ONLY=1;   shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

ADB_CMD="${ADB_CMD:-adb}"
if [ -n "$ADB_IP" ]; then ADB_CMD="adb -H ${ADB_IP}"; fi

if [ $RUN_ONLY -eq 0 ]; then
  echo "=========================================="
  echo "[Step 1] Building ${TEST_BIN_NAME} (ndk-build)"
  echo "=========================================="
  cd test/jni
  ndk-build clean 2>/dev/null
  ndk-build -j$(nproc) MESON_ENABLE_OPENCL=1 APP_MODULES="${TEST_BIN_NAME} googletest_main" 2>&1
  cd "$SCRIPT_DIR"

  echo ""
  echo "=========================================="
  echo "[Step 2] Pushing to device"
  echo "=========================================="
  $ADB_CMD shell mkdir -p ${DEVICE_DIR}

  if [ -f test/libs/arm64-v8a/${TEST_BIN_NAME} ]; then
    UNITTEST_BIN=test/libs/arm64-v8a/${TEST_BIN_NAME}
  elif [ -f test/obj/local/arm64-v8a/${TEST_BIN_NAME} ]; then
    UNITTEST_BIN=test/obj/local/arm64-v8a/${TEST_BIN_NAME}
  else
    echo "[ERROR] ${TEST_BIN_NAME} not found"; exit 1
  fi
  $ADB_CMD push "${UNITTEST_BIN}" "${DEVICE_DIR}/"
  $ADB_CMD push "${KERNEL_CL}" "${DEVICE_DIR}/"
  $ADB_CMD push "nntrainer/tensor/cl_operations/cl_kernels/int4_gemm_wave.cl" "${DEVICE_DIR}/"
  $ADB_CMD push "nntrainer/tensor/cl_operations/cl_kernels/int4_gemm_adreno.cl" "${DEVICE_DIR}/"
  $ADB_CMD push "nntrainer/tensor/cl_operations/cl_kernels/dequant_int4_to_fp16.cl" "${DEVICE_DIR}/"
  $ADB_CMD push "nntrainer/tensor/cl_operations/cl_kernels/fused_conv_int4_fp16.cl" "${DEVICE_DIR}/"

  # Push libc++_shared if available
  for so in libc++_shared.so; do
    if [ -f test/libs/arm64-v8a/${so} ]; then
      $ADB_CMD push "test/libs/arm64-v8a/${so}" "${DEVICE_DIR}/"
    fi
  done
fi

if [ $BUILD_ONLY -eq 1 ]; then
  echo "Build done (--build-only). Skipping execution."
  exit 0
fi

echo ""
echo "=========================================="
echo "[Step 3] Running on device"
echo "=========================================="
$ADB_CMD shell "cd ${DEVICE_DIR} && \
  LD_LIBRARY_PATH=/system/vendor/lib64:. \
  taskset f0 ./${TEST_BIN_NAME}" 2>&1 | tee unittest_delegate_conv_wave.log

echo ""
echo "Log: unittest_delegate_conv_wave.log"
