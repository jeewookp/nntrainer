#!/usr/bin/env bash
# temp_litert_cl_intercept.sh
#
# Captures the ACTUAL CL kernel sources that the closed-source delegate
# (libLiteRtGpuAccelerator.so) generates at runtime.
#
# How it works:
#   1. Builds cl_intercept as a shared library
#   2. Pushes it to device as "libOpenCL.so" (in the device folder)
#   3. Runs matmul_micro_benchmark — the delegate dlopen's "libOpenCL.so"
#      and finds our interceptor first (via LD_LIBRARY_PATH=.)
#   4. Our interceptor forwards all CL calls to the real
#      /system/vendor/lib64/libOpenCL.so, but saves the CL source text
#      for every clCreateProgramWithSource call
#   5. Pulls the captured .cl files back to the host
#
# The captured files are the REAL production kernels — including
# "convolution(conv_wave_memory)" that achieves 5+ TFLOPS.
#
# Usage:
#   INTERCEPT_SHAPES="1024x6144x1536" ./temp_litert_cl_intercept.sh
#
# Output:
#   runtime/cl_bench/intercepted/program_000.cl
#   runtime/cl_bench/intercepted/program_001.cl
#   ...

export ANDROID_NDK_HOME="${ANDROID_NDK_HOME:-$HOME/neo/android-ndk-r28b}"
set -euo pipefail

DEVICE_FOLDER="${DEVICE_FOLDER:-/data/local/tmp/litert_lm}"
INTERCEPT_SHAPES="${INTERCEPT_SHAPES:-1024x6144x1536}"
INTERCEPT_DTYPE="${INTERCEPT_DTYPE:-int8_per_tensor}"
INTERCEPT_ITERS="${INTERCEPT_ITERS:-2}"
INTERCEPT_WARMUP="${INTERCEPT_WARMUP:-1}"
TASKSET_MASK="${TASKSET_MASK:-f0}"
HOST_OUT_DIR="runtime/cl_bench/intercepted"
DEVICE_CL_DIR="${DEVICE_FOLDER}/cl_intercept"
MATMUL_ROSTER_CSV_DEVICE="${DEVICE_FOLDER}/matmul_roster.csv"

NDK_FALLBACK_R28B="${HOME}/neo/android-ndk-r28b"
case "${ANDROID_NDK_HOME}" in
  *android-ndk-r2[0-7]*)
    if [ -f "${NDK_FALLBACK_R28B}/source.properties" ]; then
      echo "[intercept.sh] WARNING: auto-upgrading NDK to ${NDK_FALLBACK_R28B}"
      export ANDROID_NDK_HOME="${NDK_FALLBACK_R28B}"
    fi ;;
esac

BAZEL=$(command -v bazelisk || command -v bazel)
[ -n "${BAZEL}" ] || { echo "[intercept.sh] bazel not found"; exit 1; }

echo "[intercept.sh] === CL Kernel Source Interceptor ==="
echo "[intercept.sh] SHAPES=${INTERCEPT_SHAPES}"
echo "[intercept.sh] DTYPE=${INTERCEPT_DTYPE}"
echo "[intercept.sh] ITERS=${INTERCEPT_ITERS} (just enough to trigger compilation)"
echo ""

# Step 1: Build the interceptor .so and matmul_micro_benchmark.
echo "[intercept.sh] Building cl_intercept + matmul_micro_benchmark ..."
"${BAZEL}" build \
  --config=android_arm64 \
  --define=litert_link_capi_so=true \
  --define=resolve_symbols_in_exec=false \
  //runtime/cl_bench:cl_intercept \
  //runtime/engine:matmul_micro_benchmark

# Find the interceptor .so — linkshared produces libcl_intercept.so
INTERCEPT_SO=$(find -L bazel-bin/runtime/cl_bench -name "libcl_intercept.so" 2>/dev/null | head -1)
if [ -z "${INTERCEPT_SO}" ]; then
  echo "[intercept.sh] ERROR: libcl_intercept.so not found under bazel-bin/"
  echo "  Trying alternative names..."
  find -L bazel-bin/runtime/cl_bench -name "*.so" 2>/dev/null
  exit 1
fi
echo "[intercept.sh] Interceptor: ${INTERCEPT_SO}"

MICRO_BIN=bazel-bin/runtime/engine/matmul_micro_benchmark
[ -f "${MICRO_BIN}" ] || { echo "[intercept.sh] matmul_micro_benchmark not found"; exit 1; }

LIBLITERT_SO=$(find -L bazel-bin -maxdepth 8 -type f -name "libLiteRt.so" \
                  ! -path "*solib*" 2>/dev/null | head -1)
if [ -z "${LIBLITERT_SO}" ]; then
  LIBLITERT_SO=$(find -L bazel-bin -type f -name "libLiteRt.so" 2>/dev/null | head -1)
fi
[ -n "${LIBLITERT_SO}" ] || { echo "[intercept.sh] libLiteRt.so not found"; exit 1; }

# Step 2: Backup real libOpenCL.so reference, push interceptor as libOpenCL.so.
echo ""
echo "[intercept.sh] Pushing interceptor + benchmark to device ..."
adb shell "mkdir -p ${DEVICE_FOLDER} ${DEVICE_CL_DIR}"
adb shell "rm -f ${DEVICE_CL_DIR}/*.cl"

# Save the original libOpenCL.so if it exists in device folder
adb shell "if [ -f ${DEVICE_FOLDER}/libOpenCL.so ]; then mv ${DEVICE_FOLDER}/libOpenCL.so ${DEVICE_FOLDER}/libOpenCL.so.bak; fi"

# Push interceptor AS libOpenCL.so — this is the key trick
adb push "${INTERCEPT_SO}" "${DEVICE_FOLDER}/libOpenCL.so" >/dev/null
adb push "${MICRO_BIN}" "${DEVICE_FOLDER}/matmul_micro_benchmark" >/dev/null
adb push "${LIBLITERT_SO}" "${DEVICE_FOLDER}/libLiteRt.so" >/dev/null
adb shell "chmod +x ${DEVICE_FOLDER}/matmul_micro_benchmark"

# Ensure GPU accelerator shlibs are present
PREBUILT_DIR=prebuilt/android_arm64
for f in libLiteRtGpuAccelerator.so libLiteRtOpenClAccelerator.so; do
  if ! adb shell "test -f ${DEVICE_FOLDER}/${f}"; then
    if [ -f "${PREBUILT_DIR}/${f}" ]; then
      echo "[intercept.sh] Pushing ${f} ..."
      adb push "${PREBUILT_DIR}/${f}" "${DEVICE_FOLDER}/${f}" >/dev/null
    fi
  fi
done

# Step 3: Run matmul_micro_benchmark with BOTH LD_PRELOAD and LD_LIBRARY_PATH.
# LD_PRELOAD ensures our clCreateProgramWithSource shadows the real one
# even if the delegate does dlopen with a full path to libOpenCL.so.
# We also keep libOpenCL.so as a renamed copy for dlopen("libOpenCL.so").
echo ""
echo "[intercept.sh] Running benchmark with interceptor (shapes: ${INTERCEPT_SHAPES}) ..."
echo "[intercept.sh] (Only ${INTERCEPT_ITERS} iters — just need CL compilation, not timing)"

# Also push as libcl_intercept.so for LD_PRELOAD (keep libOpenCL.so copy too)
adb push "${INTERCEPT_SO}" "${DEVICE_FOLDER}/libcl_intercept.so" >/dev/null

adb shell "cd ${DEVICE_FOLDER}; \
  LD_PRELOAD=./libcl_intercept.so \
  LD_LIBRARY_PATH=. \
  taskset ${TASKSET_MASK} ./matmul_micro_benchmark \
    --backend=gpu \
    --dtype=${INTERCEPT_DTYPE} \
    --shapes=${INTERCEPT_SHAPES} \
    --warmup=${INTERCEPT_WARMUP} \
    --iters=${INTERCEPT_ITERS} \
    --profile=false \
    --max_tensor_mb=512" 2>&1 | head -80

# Step 4: Restore original libOpenCL.so and pull captured files.
echo ""
echo "[intercept.sh] Restoring original libOpenCL.so ..."
adb shell "rm -f ${DEVICE_FOLDER}/libOpenCL.so"
adb shell "if [ -f ${DEVICE_FOLDER}/libOpenCL.so.bak ]; then mv ${DEVICE_FOLDER}/libOpenCL.so.bak ${DEVICE_FOLDER}/libOpenCL.so; fi"

echo "[intercept.sh] Pulling captured CL sources ..."
mkdir -p "${HOST_OUT_DIR}"
CL_COUNT=$(adb shell "ls ${DEVICE_CL_DIR}/*.cl 2>/dev/null | wc -l" | tr -d '\r ')
if [ "${CL_COUNT}" -gt 0 ]; then
  adb shell "ls ${DEVICE_CL_DIR}/*.cl" | while read -r f; do
    f=$(echo "$f" | tr -d '\r')
    base=$(basename "$f")
    adb pull "$f" "${HOST_OUT_DIR}/${base}" 2>/dev/null
    size=$(wc -c < "${HOST_OUT_DIR}/${base}" 2>/dev/null || echo "?")
    echo "  ${HOST_OUT_DIR}/${base} (${size} bytes)"
  done
  echo ""
  echo "=========================================="
  echo " Captured ${CL_COUNT} CL kernel source(s)"
  echo "=========================================="
  echo " Location: ${HOST_OUT_DIR}/"
  echo ""
  echo " These are the ACTUAL production kernels from"
  echo " libLiteRtGpuAccelerator.so — the same code that"
  echo " achieves 5+ TFLOPS on Adreno 830."
  echo ""
  echo " Look for 'convolution' or 'conv_wave' in the files."
else
  echo "[intercept.sh] WARNING: No .cl files captured."
  echo "  The delegate might have used clCreateProgramWithBinary"
  echo "  (cached compiled program) instead of source compilation."
  echo "  Try: adb shell 'rm -rf ${DEVICE_FOLDER}/cache/*' and re-run."
fi
echo ""
