#!/usr/bin/env bash
# temp_litert_decode_kernel_capture.sh
#
# Captures the OpenCL kernel sources LiteRT's GPU delegate compiles
# when dispatching DECODE-style M=1 GEMV.  This is the kernel family
# we need to match in nntrainer's int4 GEMV path on Adreno 830 to
# close the ~5x decode TPS gap vs LiteRT-LM (~26 TPS Qwen3-4B
# normalized vs nntrainer 4.8 TPS).
#
# Strategy.  litert_lm_main routes OpenCL through the closed-source
# libLiteRtGpuAccelerator.so which bypasses LD_PRELOAD (likely via
# absolute-path dlopen or a private linker namespace), so a direct
# litert_lm_main intercept yields zero captured kernels (verified:
# decode at 49 TPS but [cl_intercept] never logged a single call).
#
# Instead we reuse matmul_micro_benchmark which IS interceptable
# (proven by temp_litert_cl_intercept.sh capturing 5 .cl files for
# M=1024xKxN), but feed it M=1 shapes so the LiteRT GPU delegate
# picks its decode-time GEMV variant.  The delegate's kernel choice
# is keyed on (dtype, shape) -- it'll lower 1xKxN through whatever
# gemv kernel it would have picked for an actual decode dispatch.
#
# Output: runtime/cl_bench/intercepted_decode/program_*.cl
# (kept separate from the existing matmul_micro_benchmark M=1024
# capture in runtime/cl_bench/intercepted/).
#
# Re-exec under bash if invoked via sh/dash.
if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi

# bazel's --config=android_arm64 (TF/LiteRT) refuses to register the
# Android cc_toolchain unless ANDROID_NDK_HOME is exported BEFORE the
# build invocation.  Mirror temp_litert_cl_intercept.sh.
export ANDROID_NDK_HOME="${ANDROID_NDK_HOME:-$HOME/neo/android-ndk-r28b}"
NDK_FALLBACK_R28B="${HOME}/neo/android-ndk-r28b"
case "${ANDROID_NDK_HOME}" in
  *android-ndk-r2[0-7]*)
    if [ -f "${NDK_FALLBACK_R28B}/source.properties" ]; then
      echo "[decode_intercept.sh] WARNING: auto-upgrading NDK to ${NDK_FALLBACK_R28B}"
      export ANDROID_NDK_HOME="${NDK_FALLBACK_R28B}"
    fi ;;
esac
if [ ! -f "${ANDROID_NDK_HOME}/source.properties" ]; then
  echo "[decode_intercept.sh] ERROR: ANDROID_NDK_HOME=${ANDROID_NDK_HOME} not a valid NDK"
  echo "[decode_intercept.sh] Set ANDROID_NDK_HOME to your r28b+ install before running."
  exit 1
fi

set -euo pipefail

DEVICE_FOLDER="${DEVICE_FOLDER:-/data/local/tmp/litert_lm}"
DEVICE_CL_DIR="${DEVICE_CL_DIR:-${DEVICE_FOLDER}/cl_intercept}"
HOST_OUT_DIR="${HOST_OUT_DIR:-runtime/cl_bench/intercepted_decode}"
TASKSET_MASK="${TASKSET_MASK:-f0}"
# Default to fp16: int8_per_tensor's M=1 codegen is buggy in this
# LiteRT build (BC-src "use of undeclared identifier 'q0'"), and
# matmul_micro_benchmark.cc's own comment explicitly notes the
# wrapped int8 path "tried and failed at for unrelated reasons --
# the delegate collapses the wrap back to fp16."  fp16 compiles
# fine at M=1 even though the .cc warns its CPU reference disagrees
# -- we only need the dispatch pattern, not numeric correctness.
INTERCEPT_DTYPE="${INTERCEPT_DTYPE:-fp16}"
INTERCEPT_ITERS="${INTERCEPT_ITERS:-2}"
INTERCEPT_WARMUP="${INTERCEPT_WARMUP:-1}"

# Qwen3-4B M=1 decode FC shapes (M x K x N):
#   q_proj    1 x 2560 x 4096
#   k/v_proj  1 x 2560 x 1024
#   o_proj    1 x 4096 x 2560
#   gate/up   1 x 2560 x 9728
#   down_proj 1 x 9728 x 2560
INTERCEPT_SHAPES="${INTERCEPT_SHAPES:-1x2560x4096,1x2560x1024,1x4096x2560,1x2560x9728,1x9728x2560}"

BAZEL=$(command -v bazelisk || command -v bazel || true)
[ -n "${BAZEL}" ] || { echo "[decode_intercept.sh] bazel not found"; exit 1; }

# -----------------------------------------------------------------------------
# Step 1: Build cl_intercept + matmul_micro_benchmark.
# -----------------------------------------------------------------------------
echo "[decode_intercept.sh] Building cl_intercept + matmul_micro_benchmark with ${BAZEL} ..."
${BAZEL} build \
  --config=android_arm64 \
  --copt=-O3 \
  //runtime/cl_bench:cl_intercept \
  //runtime/engine:matmul_micro_benchmark \
  2>&1 | tail -20

INTERCEPT_SO=$(find -L bazel-bin/runtime/cl_bench -name "libcl_intercept.so" 2>/dev/null | head -1)
MATMUL_BIN=bazel-bin/runtime/engine/matmul_micro_benchmark
LIBLITERT_SO=$(find -L bazel-bin -maxdepth 8 -type f -name "libLiteRt.so" \
                 ! -path "*runfiles*" 2>/dev/null | head -1)
if [ -z "${LIBLITERT_SO}" ]; then
  LIBLITERT_SO=$(find -L bazel-bin -type f -name "libLiteRt.so" 2>/dev/null | head -1)
fi
[ -n "${INTERCEPT_SO}" ] && [ -f "${INTERCEPT_SO}" ] || \
  { echo "[decode_intercept.sh] ERROR: libcl_intercept.so not found"; exit 1; }
[ -f "${MATMUL_BIN}" ] || \
  { echo "[decode_intercept.sh] ERROR: matmul_micro_benchmark not found"; exit 1; }
[ -n "${LIBLITERT_SO}" ] && [ -f "${LIBLITERT_SO}" ] || \
  { echo "[decode_intercept.sh] ERROR: libLiteRt.so not found"; exit 1; }
echo "[decode_intercept.sh]   INTERCEPT_SO=${INTERCEPT_SO}"
echo "[decode_intercept.sh]   MATMUL_BIN=${MATMUL_BIN}"
echo "[decode_intercept.sh]   LIBLITERT_SO=${LIBLITERT_SO}"

# -----------------------------------------------------------------------------
# Step 2: Push interceptor + binary to device.
# -----------------------------------------------------------------------------
echo "[decode_intercept.sh] Pushing to device ..."
adb shell "mkdir -p ${DEVICE_FOLDER} ${DEVICE_CL_DIR}"
adb shell "rm -f ${DEVICE_CL_DIR}/*.cl ${DEVICE_CL_DIR}/*.spv 2>/dev/null" || true
# Save vendor libOpenCL.so if present
adb shell "if [ -f ${DEVICE_FOLDER}/libOpenCL.so ]; then mv ${DEVICE_FOLDER}/libOpenCL.so ${DEVICE_FOLDER}/libOpenCL.so.bak; fi"
# Push interceptor masquerading as libOpenCL.so AND under its real name
adb push "${INTERCEPT_SO}" "${DEVICE_FOLDER}/libOpenCL.so" >/dev/null
adb push "${INTERCEPT_SO}" "${DEVICE_FOLDER}/libcl_intercept.so" >/dev/null
adb push "${MATMUL_BIN}"   "${DEVICE_FOLDER}/matmul_micro_benchmark" >/dev/null
adb push "${LIBLITERT_SO}" "${DEVICE_FOLDER}/libLiteRt.so" >/dev/null
adb shell "chmod +x ${DEVICE_FOLDER}/matmul_micro_benchmark"
# GPU accelerator shlibs
PREBUILT_DIR=prebuilt/android_arm64
for f in libLiteRtGpuAccelerator.so libLiteRtOpenClAccelerator.so; do
  if ! adb shell "test -f ${DEVICE_FOLDER}/${f}"; then
    if [ -f "${PREBUILT_DIR}/${f}" ]; then
      adb push "${PREBUILT_DIR}/${f}" "${DEVICE_FOLDER}/${f}" >/dev/null
    fi
  fi
done

# -----------------------------------------------------------------------------
# Step 3: Clear CL caches so delegate is forced to compile from source.
# -----------------------------------------------------------------------------
echo "[decode_intercept.sh] Clearing CL caches ..."
adb shell "rm -rf ${DEVICE_FOLDER}/cache/* 2>/dev/null; \
           rm -rf /data/local/tmp/cl_cache/* 2>/dev/null; \
           rm -rf /data/data/*/cache/cl_cache/* 2>/dev/null" 2>/dev/null || true

# -----------------------------------------------------------------------------
# Step 4: Run matmul_micro_benchmark with M=1 shapes + LD_PRELOAD.
# -----------------------------------------------------------------------------
echo ""
echo "[decode_intercept.sh] Running matmul_micro_benchmark with shapes: ${INTERCEPT_SHAPES}"
echo "[decode_intercept.sh] Captured kernels go to ${DEVICE_CL_DIR}/"
echo ""

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
    --max_tensor_mb=512" 2>&1 | tail -40

# -----------------------------------------------------------------------------
# Step 5: Restore vendor libOpenCL.so + pull captured kernels.
# -----------------------------------------------------------------------------
echo ""
echo "[decode_intercept.sh] Restoring vendor libOpenCL.so ..."
adb shell "rm -f ${DEVICE_FOLDER}/libOpenCL.so"
adb shell "if [ -f ${DEVICE_FOLDER}/libOpenCL.so.bak ]; then mv ${DEVICE_FOLDER}/libOpenCL.so.bak ${DEVICE_FOLDER}/libOpenCL.so; fi"

echo "[decode_intercept.sh] Pulling captured CL sources ..."
mkdir -p "${HOST_OUT_DIR}"
rm -f "${HOST_OUT_DIR}"/*.cl "${HOST_OUT_DIR}"/*.spv 2>/dev/null || true
CL_COUNT=$(adb shell "ls ${DEVICE_CL_DIR}/*.cl ${DEVICE_CL_DIR}/*.spv 2>/dev/null | wc -l" | tr -d '\r ')
if [ "${CL_COUNT}" -gt 0 ]; then
  adb shell "ls ${DEVICE_CL_DIR}/*.cl ${DEVICE_CL_DIR}/*.spv 2>/dev/null" | while read -r f; do
    f=$(echo "$f" | tr -d '\r')
    base=$(basename "$f")
    adb pull "$f" "${HOST_OUT_DIR}/${base}" 2>/dev/null
    size=$(wc -c < "${HOST_OUT_DIR}/${base}" 2>/dev/null || echo "?")
    echo "  ${HOST_OUT_DIR}/${base} (${size} bytes)"
  done
fi

echo ""
echo "===================================================================="
echo " Captured ${CL_COUNT} CL kernel source(s) for M=1 decode dispatches"
echo "===================================================================="
echo " Host:   ${HOST_OUT_DIR}/"
echo ""
echo " Compare against runtime/cl_bench/intercepted/ (the M=1024 prefill"
echo " capture).  Programs that appear ONLY here are decode-specialised"
echo " (M=1 GEMV variants); programs in both directories are shared."
echo ""
ls -la "${HOST_OUT_DIR}"/*.cl "${HOST_OUT_DIR}"/*.spv 2>/dev/null | head
