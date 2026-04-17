#!/usr/bin/env bash
# Capture CL kernel sources via binary-patched delegate.
#
# The delegate loads "libOpenCL.so" by name. We patch the binary to load
# "libOpenCX.so" instead, then push our interceptor as "libOpenCX.so".
# Since the patched name is the same length, the ELF structure is preserved.
#
# Usage:
#   INTERCEPT_SHAPES="1024x6144x1536" ./temp_litert_cl_patch_intercept.sh

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
PREBUILT_DIR="prebuilt/android_arm64"

NDK_FALLBACK_R28B="${HOME}/neo/android-ndk-r28b"
case "${ANDROID_NDK_HOME}" in
  *android-ndk-r2[0-7]*)
    if [ -f "${NDK_FALLBACK_R28B}/source.properties" ]; then
      export ANDROID_NDK_HOME="${NDK_FALLBACK_R28B}"
    fi ;;
esac

BAZEL=$(command -v bazelisk || command -v bazel)
[ -n "${BAZEL}" ] || { echo "[patch.sh] bazel not found"; exit 1; }

echo "[patch.sh] === Binary-Patched CL Kernel Interceptor ==="
echo "[patch.sh] SHAPES=${INTERCEPT_SHAPES}"
echo "[patch.sh] DTYPE=${INTERCEPT_DTYPE}"
echo ""

# Step 1: Build interceptor + benchmark.
echo "[patch.sh] Building ..."
"${BAZEL}" build \
  --config=android_arm64 \
  --define=litert_link_capi_so=true \
  --define=resolve_symbols_in_exec=false \
  //runtime/cl_bench:cl_intercept \
  //runtime/engine:matmul_micro_benchmark

INTERCEPT_SO=$(find -L bazel-bin/runtime/cl_bench -name "libcl_intercept.so" 2>/dev/null | head -1)
[ -n "${INTERCEPT_SO}" ] || { echo "[patch.sh] libcl_intercept.so not found"; exit 1; }
MICRO_BIN=bazel-bin/runtime/engine/matmul_micro_benchmark
[ -f "${MICRO_BIN}" ] || { echo "[patch.sh] matmul_micro_benchmark not found"; exit 1; }
LIBLITERT_SO=$(find -L bazel-bin -maxdepth 8 -type f -name "libLiteRt.so" ! -path "*solib*" 2>/dev/null | head -1)
[ -n "${LIBLITERT_SO}" ] || { echo "[patch.sh] libLiteRt.so not found"; exit 1; }

# Step 2: Patch the delegate binaries.
# Replace "libOpenCL.so" → "libOpenCX.so" (same length, ELF-safe).
PATCH_DIR=$(mktemp -d)
echo "[patch.sh] Patching delegate .so files in ${PATCH_DIR} ..."

for so_name in libLiteRtOpenClAccelerator.so libLiteRtGpuAccelerator.so; do
  src="${PREBUILT_DIR}/${so_name}"
  if [ -f "${src}" ]; then
    cp "${src}" "${PATCH_DIR}/${so_name}"
    # Binary-safe replacement: same length string
    perl -pi -e 's/libOpenCL\.so/libOpenCX.so/g' "${PATCH_DIR}/${so_name}"
    # Verify the patch
    count_old=$(strings "${PATCH_DIR}/${so_name}" | grep -c "libOpenCL\.so" || true)
    count_new=$(strings "${PATCH_DIR}/${so_name}" | grep -c "libOpenCX\.so" || true)
    echo "  ${so_name}: libOpenCL.so→${count_old} remaining, libOpenCX.so→${count_new} patched"
  fi
done

# Also patch the interceptor to load the real lib (it already uses full path
# /system/vendor/lib64/libOpenCL.so, so no change needed there).

# Step 3: Push everything to device.
echo ""
echo "[patch.sh] Pushing to device ..."
adb shell "mkdir -p ${DEVICE_FOLDER} ${DEVICE_CL_DIR}"
adb shell "rm -f ${DEVICE_CL_DIR}/*.cl ${DEVICE_CL_DIR}/*.spv 2>/dev/null" || true

# Clear CL caches
adb shell "rm -rf ${DEVICE_FOLDER}/cache/* 2>/dev/null" || true

adb push "${MICRO_BIN}" "${DEVICE_FOLDER}/matmul_micro_benchmark" >/dev/null
adb push "${LIBLITERT_SO}" "${DEVICE_FOLDER}/libLiteRt.so" >/dev/null
adb shell "chmod +x ${DEVICE_FOLDER}/matmul_micro_benchmark"

# Push patched delegate .so files (overwrite originals temporarily)
for so_name in libLiteRtOpenClAccelerator.so libLiteRtGpuAccelerator.so; do
  if [ -f "${PATCH_DIR}/${so_name}" ]; then
    # Backup original on device
    adb shell "if [ -f ${DEVICE_FOLDER}/${so_name} ]; then cp ${DEVICE_FOLDER}/${so_name} ${DEVICE_FOLDER}/${so_name}.orig; fi"
    adb push "${PATCH_DIR}/${so_name}" "${DEVICE_FOLDER}/${so_name}" >/dev/null
    echo "  Pushed patched ${so_name}"
  fi
done

# Push interceptor as "libOpenCX.so" — the patched delegate will load THIS
adb push "${INTERCEPT_SO}" "${DEVICE_FOLDER}/libOpenCX.so" >/dev/null
echo "  Pushed interceptor as libOpenCX.so"

# Also ensure other prebuilts are present
for f in libLiteRtTopKOpenClSampler.so libLiteRtTopKWebGpuSampler.so libLiteRtWebGpuAccelerator.so libGemmaModelConstraintProvider.so; do
  if [ -f "${PREBUILT_DIR}/${f}" ] && ! adb shell "test -f ${DEVICE_FOLDER}/${f}"; then
    adb push "${PREBUILT_DIR}/${f}" "${DEVICE_FOLDER}/${f}" >/dev/null
  fi
done

# Step 4: Run.
echo ""
echo "[patch.sh] Running with patched delegate ..."
adb shell "cd ${DEVICE_FOLDER}; \
  LD_LIBRARY_PATH=. \
  taskset ${TASKSET_MASK} ./matmul_micro_benchmark \
    --backend=gpu \
    --dtype=${INTERCEPT_DTYPE} \
    --shapes=${INTERCEPT_SHAPES} \
    --warmup=${INTERCEPT_WARMUP} \
    --iters=${INTERCEPT_ITERS} \
    --profile=false \
    --max_tensor_mb=512" 2>&1 | tee temp_litert_cl_patch.log | head -200

# Step 5: Restore original delegate .so files.
echo ""
echo "[patch.sh] Restoring original delegate .so files ..."
for so_name in libLiteRtOpenClAccelerator.so libLiteRtGpuAccelerator.so; do
  adb shell "if [ -f ${DEVICE_FOLDER}/${so_name}.orig ]; then mv ${DEVICE_FOLDER}/${so_name}.orig ${DEVICE_FOLDER}/${so_name}; fi"
done
adb shell "rm -f ${DEVICE_FOLDER}/libOpenCX.so"

# Step 6: Pull captured files.
echo "[patch.sh] Pulling captured CL sources ..."
mkdir -p "${HOST_OUT_DIR}"
CL_COUNT=$(adb shell "ls ${DEVICE_CL_DIR}/*.cl ${DEVICE_CL_DIR}/*.spv 2>/dev/null | wc -l" | tr -d '\r ')
if [ "${CL_COUNT}" -gt 0 ]; then
  adb shell "ls ${DEVICE_CL_DIR}/*.cl ${DEVICE_CL_DIR}/*.spv 2>/dev/null" | while read -r f; do
    f=$(echo "$f" | tr -d '\r')
    base=$(basename "$f")
    adb pull "$f" "${HOST_OUT_DIR}/${base}" 2>/dev/null
    size=$(wc -c < "${HOST_OUT_DIR}/${base}" 2>/dev/null || echo "?")
    echo "  ${HOST_OUT_DIR}/${base} (${size} bytes)"
  done
  echo ""
  echo "=========================================="
  echo " Captured ${CL_COUNT} CL kernel source(s)!"
  echo "=========================================="
  echo " Location: ${HOST_OUT_DIR}/"
  echo ""
  echo " Look for 'wave_memory' or 'convolution' in the files:"
  echo "   grep -l 'wave_memory\|convolution' ${HOST_OUT_DIR}/*.cl"
else
  echo ""
  echo "[patch.sh] No .cl files captured."
  echo "  Check temp_litert_cl_patch.log for [cl_intercept] messages."
  grep "cl_intercept" temp_litert_cl_patch.log 2>/dev/null || echo "  (no intercept messages found)"
fi

# Cleanup temp dir
rm -rf "${PATCH_DIR}"
echo ""
