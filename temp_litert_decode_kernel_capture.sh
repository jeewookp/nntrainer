#!/usr/bin/env bash
# temp_litert_decode_kernel_capture.sh
#
# Capture the OpenCL kernel sources LiteRT-LM compiles when running
# DECODE-heavy (M=1) workloads.  Existing temp_litert_cl_intercept.sh
# captures kernels from a synthetic matmul_micro_benchmark; the
# kernels we care about for the nntrainer Stage 1 catchup
# (gemm_delegate_fp16_cl-equivalent for M=1) only show up under the
# real LiteRT engine running an actual model.
#
# This script combines the two:
#   - cl_intercept LD_PRELOAD setup from temp_litert_cl_intercept.sh
#   - litert_lm_main run from temp_litert.sh with prefill_tokens=1
#     (effectively skip prefill) and decode_tokens=256 (force lots
#     of M=1 dispatches to populate every decode kernel variant)
#
# Required env (matches the parent scripts):
#   ANDROID_NDK_HOME   path to NDK r28b+
#   adb                in PATH
#   model              ~/.cache/litert_lm_models/gemma-4-E2B-it.litertlm
#                      (set MODEL_PATH to override)
#
# Output:
#   runtime/cl_bench/intercepted_decode/program_*.cl
#     -- the actual kernel sources that ran during decode.
#   runtime/cl_bench/intercepted_decode/dispatch.log
#     -- timing per kernel name (if cl_intercept logs dispatches).
#
# Re-exec under bash if invoked via sh/dash.
if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi
set -e
set -o pipefail

DEVICE_FOLDER="${DEVICE_FOLDER:-/data/local/tmp/litert_lm}"
DEVICE_CL_DIR="${DEVICE_CL_DIR:-${DEVICE_FOLDER}/intercepted_cl}"
HOST_OUT_DIR="${HOST_OUT_DIR:-runtime/cl_bench/intercepted_decode}"
MODEL_PATH="${MODEL_PATH:-$HOME/.cache/litert_lm_models/gemma-4-E2B-it.litertlm}"
TASKSET_MASK="${TASKSET_MASK:-f0}"
PREFILL_TOKENS="${PREFILL_TOKENS:-1}"     # effectively skip prefill
DECODE_TOKENS="${DECODE_TOKENS:-256}"     # plenty of M=1 dispatches
ASYNC="${ASYNC:-false}"

# -----------------------------------------------------------------------------
# Step 1: Build cl_intercept + litert_lm_main.
# -----------------------------------------------------------------------------
# Prefer bazelisk (auto-pins bazel version via .bazelversion); fall back
# to plain bazel if bazelisk isn't installed.
if command -v bazelisk >/dev/null 2>&1; then
  BAZEL=bazelisk
elif command -v bazel >/dev/null 2>&1; then
  BAZEL=bazel
else
  echo "[decode_intercept.sh] ERROR: neither bazelisk nor bazel found in PATH"
  exit 1
fi
echo "[decode_intercept.sh] Building cl_intercept + litert_lm_main with ${BAZEL} ..."
${BAZEL} build \
  --config=android_arm64 \
  --copt=-O3 \
  //runtime/cl_bench:cl_intercept \
  //runtime/engine:litert_lm_main \
  2>&1 | tail -20

INTERCEPT_SO=bazel-bin/runtime/cl_bench/libcl_intercept.so
LITERT_BIN=bazel-bin/runtime/engine/litert_lm_main
LIBLITERT_SO=bazel-bin/runtime/engine/libLiteRt.so
[ -f "${INTERCEPT_SO}" ] || { echo "[decode_intercept.sh] cl_intercept .so not found"; exit 1; }
[ -f "${LITERT_BIN}" ]   || { echo "[decode_intercept.sh] litert_lm_main not found"; exit 1; }

# -----------------------------------------------------------------------------
# Step 2: Push interceptor + binary + model to device.
# -----------------------------------------------------------------------------
MODEL_BASENAME=$(basename "${MODEL_PATH}")
echo "[decode_intercept.sh] Pushing to device ..."
adb shell "mkdir -p ${DEVICE_FOLDER} ${DEVICE_CL_DIR}"
adb shell "rm -f ${DEVICE_CL_DIR}/*.cl"
# Save vendor libOpenCL.so if present
adb shell "if [ -f ${DEVICE_FOLDER}/libOpenCL.so ]; then mv ${DEVICE_FOLDER}/libOpenCL.so ${DEVICE_FOLDER}/libOpenCL.so.bak; fi"
# Push interceptor masquerading as libOpenCL.so AND under its real name
# for LD_PRELOAD.  Both names have to resolve so that whether the
# delegate dlopen()s libOpenCL.so or relies on LD_PRELOAD, our
# clCreateProgramWithSource shim runs.
adb push "${INTERCEPT_SO}" "${DEVICE_FOLDER}/libOpenCL.so" >/dev/null
adb push "${INTERCEPT_SO}" "${DEVICE_FOLDER}/libcl_intercept.so" >/dev/null
adb push "${LITERT_BIN}"   "${DEVICE_FOLDER}/litert_lm_main" >/dev/null
adb push "${LIBLITERT_SO}" "${DEVICE_FOLDER}/libLiteRt.so" >/dev/null
adb shell "chmod +x ${DEVICE_FOLDER}/litert_lm_main"
# GPU accelerator shlibs
PREBUILT_DIR=prebuilt/android_arm64
for f in libLiteRtGpuAccelerator.so libLiteRtOpenClAccelerator.so; do
  if ! adb shell "test -f ${DEVICE_FOLDER}/${f}"; then
    if [ -f "${PREBUILT_DIR}/${f}" ]; then
      adb push "${PREBUILT_DIR}/${f}" "${DEVICE_FOLDER}/${f}" >/dev/null
    fi
  fi
done
# Push model if not already there
if ! adb shell "test -f ${DEVICE_FOLDER}/${MODEL_BASENAME}"; then
  echo "[decode_intercept.sh] Pushing model ${MODEL_BASENAME} (~couple GB, slow) ..."
  adb push "${MODEL_PATH}" "${DEVICE_FOLDER}/${MODEL_BASENAME}" >/dev/null
fi

# -----------------------------------------------------------------------------
# Step 3: Clear CL caches so delegate is forced to compile from source.
# -----------------------------------------------------------------------------
echo "[decode_intercept.sh] Clearing CL caches ..."
adb shell "rm -rf ${DEVICE_FOLDER}/cache/* 2>/dev/null; \
           rm -rf /data/local/tmp/cl_cache/* 2>/dev/null; \
           rm -rf /data/data/*/cache/cl_cache/* 2>/dev/null" 2>/dev/null || true

# -----------------------------------------------------------------------------
# Step 4: Run litert_lm_main with decode-heavy params + LD_PRELOAD interceptor.
# -----------------------------------------------------------------------------
echo ""
echo "[decode_intercept.sh] Running litert_lm_main: prefill=${PREFILL_TOKENS} decode=${DECODE_TOKENS}"
echo "[decode_intercept.sh] Captured kernels go to ${DEVICE_CL_DIR}/"
echo ""

adb shell "cd ${DEVICE_FOLDER}; \
  LD_PRELOAD=./libcl_intercept.so \
  LD_LIBRARY_PATH=. \
  taskset ${TASKSET_MASK} ./litert_lm_main \
    --backend=gpu \
    --model_path=${DEVICE_FOLDER}/${MODEL_BASENAME} \
    --benchmark=true \
    --benchmark_prefill_tokens=${PREFILL_TOKENS} \
    --benchmark_decode_tokens=${DECODE_TOKENS} \
    --async=${ASYNC} \
    --enable_op_profiling=false" 2>&1 | tail -40

# -----------------------------------------------------------------------------
# Step 5: Restore vendor libOpenCL.so + pull captured kernels.
# -----------------------------------------------------------------------------
echo ""
echo "[decode_intercept.sh] Restoring vendor libOpenCL.so ..."
adb shell "rm -f ${DEVICE_FOLDER}/libOpenCL.so"
adb shell "if [ -f ${DEVICE_FOLDER}/libOpenCL.so.bak ]; then mv ${DEVICE_FOLDER}/libOpenCL.so.bak ${DEVICE_FOLDER}/libOpenCL.so; fi"

echo "[decode_intercept.sh] Pulling captured CL sources ..."
mkdir -p "${HOST_OUT_DIR}"
rm -f "${HOST_OUT_DIR}"/*.cl 2>/dev/null || true
adb pull "${DEVICE_CL_DIR}/" "${HOST_OUT_DIR}/" 2>&1 | tail -5 || true
# adb pull may put files in HOST_OUT_DIR/intercepted_cl -- flatten if so.
if [ -d "${HOST_OUT_DIR}/intercepted_cl" ]; then
  mv "${HOST_OUT_DIR}/intercepted_cl"/*.cl "${HOST_OUT_DIR}/" 2>/dev/null || true
  rmdir "${HOST_OUT_DIR}/intercepted_cl" 2>/dev/null || true
fi

CL_COUNT=$(ls "${HOST_OUT_DIR}"/*.cl 2>/dev/null | wc -l)
echo ""
echo "===================================================================="
echo " Captured ${CL_COUNT} CL kernel source(s) during decode-heavy run"
echo "===================================================================="
echo " Host:   ${HOST_OUT_DIR}/"
echo ""
echo " Compare against runtime/cl_bench/intercepted/ (prefill+decode mix"
echo " from the matmul_micro_benchmark intercept).  Programs that appear"
echo " ONLY here are decode-specialised.  Programs in both places are"
echo " shared between prefill and decode (likely 1x1 convs at different"
echo " global dispatch sizes)."
echo ""
ls -la "${HOST_OUT_DIR}"/*.cl 2>/dev/null | head
