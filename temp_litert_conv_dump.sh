#!/usr/bin/env bash
# temp_litert_conv_dump.sh
#
# Step 1: Generate CL kernel source files from ConvGeneric.
# Runs conv_cl_dump on the device, which initializes OpenCL, creates
# ConvGeneric for each shape, and saves the generated CL source to files.
# The .cl files are then pulled back to the host project.
#
# Usage:
#   CONV_DUMP_SHAPES="1024x6144x1536,1024x1536x4096" ./temp_litert_conv_dump.sh
#
# Output:
#   runtime/cl_bench/generated/conv_kernel_1024x6144x1536.cl
#   runtime/cl_bench/generated/conv_kernel_1024x6144x1536.meta
#   ...
#
# Env vars:
#   CONV_DUMP_SHAPES      Required. MxNxK,... shapes to generate
#   CONV_DUMP_PRECISION   fp16 (default), fp32, fp16acc32
#   DEVICE_FOLDER         /data/local/tmp/litert_lm (default)

export ANDROID_NDK_HOME="${ANDROID_NDK_HOME:-$HOME/neo/android-ndk-r28b}"
set -euo pipefail

DEVICE_FOLDER="${DEVICE_FOLDER:-/data/local/tmp/litert_lm}"
CONV_DUMP_SHAPES="${CONV_DUMP_SHAPES:-}"
CONV_DUMP_PRECISION="${CONV_DUMP_PRECISION:-fp16}"
HOST_OUT_DIR="runtime/cl_bench/generated"
DEVICE_OUT_DIR="${DEVICE_FOLDER}/cl_dump"

if [ -z "${CONV_DUMP_SHAPES}" ]; then
  echo "[dump.sh] ERROR: set CONV_DUMP_SHAPES='MxNxK,...'"
  echo "  Example: CONV_DUMP_SHAPES='1024x6144x1536,1024x1536x4096' $0"
  exit 1
fi

NDK_FALLBACK_R28B="${HOME}/neo/android-ndk-r28b"
case "${ANDROID_NDK_HOME}" in
  *android-ndk-r2[0-7]*)
    if [ -f "${NDK_FALLBACK_R28B}/source.properties" ]; then
      echo "[dump.sh] WARNING: auto-upgrading NDK to ${NDK_FALLBACK_R28B}"
      export ANDROID_NDK_HOME="${NDK_FALLBACK_R28B}"
    fi ;;
esac

BAZEL=$(command -v bazelisk || command -v bazel)
[ -n "${BAZEL}" ] || { echo "[dump.sh] bazel not found"; exit 1; }

echo "[dump.sh] === ConvGeneric CL Kernel Dump ==="
echo "[dump.sh] SHAPES=${CONV_DUMP_SHAPES}"
echo "[dump.sh] PRECISION=${CONV_DUMP_PRECISION}"
echo ""

# Build.
echo "[dump.sh] Building //runtime/cl_bench:conv_cl_dump ..."
"${BAZEL}" build \
  --config=android_arm64 \
  //runtime/cl_bench:conv_cl_dump

DUMP_BIN=bazel-bin/runtime/cl_bench/conv_cl_dump
[ -f "${DUMP_BIN}" ] || { echo "[dump.sh] Binary not found"; exit 1; }

# Push & run on device.
echo ""
echo "[dump.sh] adb push + run ..."
adb shell "mkdir -p ${DEVICE_OUT_DIR}"
adb push "${DUMP_BIN}" "${DEVICE_FOLDER}/conv_cl_dump" >/dev/null
adb shell "chmod +x ${DEVICE_FOLDER}/conv_cl_dump"

adb shell "cd ${DEVICE_FOLDER}; \
  LD_LIBRARY_PATH=/system/vendor/lib64:. \
  ./conv_cl_dump \
    --shapes=${CONV_DUMP_SHAPES} \
    --precision=${CONV_DUMP_PRECISION} \
    --out_dir=${DEVICE_OUT_DIR}" 2>&1

# Pull .cl and .meta files to host.
echo ""
echo "[dump.sh] Pulling generated files ..."
mkdir -p "${HOST_OUT_DIR}"
adb shell "ls ${DEVICE_OUT_DIR}/*.cl ${DEVICE_OUT_DIR}/*.meta 2>/dev/null" | while read -r f; do
  f=$(echo "$f" | tr -d '\r')
  base=$(basename "$f")
  adb pull "$f" "${HOST_OUT_DIR}/${base}" 2>/dev/null
  echo "  ${HOST_OUT_DIR}/${base}"
done

echo ""
echo "[dump.sh] Done. CL files saved to ${HOST_OUT_DIR}/"
echo ""
echo "Next: inspect the .cl files, then run benchmarks with:"
echo "  CONV_CL_DIR=${HOST_OUT_DIR} CONV_BENCH_SHAPES='${CONV_DUMP_SHAPES}' ./temp_litert_conv_cl_run.sh"
