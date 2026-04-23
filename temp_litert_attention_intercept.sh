#!/usr/bin/env bash
# temp_litert_attention_intercept.sh
#
# One-shot build + push + run that captures every CL kernel source
# LiteRT's GPU accelerator compiles during a real LLM prefill/decode —
# including (hopefully) the attention kernel libLiteRtGpuAccelerator.so
# uses for scaled-dot-product.
#
# Shares the model/prebuilt/push scaffolding of temp_litert.sh so we
# execute the same forward pass the reviewer has already validated.
# The only functional differences from temp_litert.sh are:
#   1. We ALSO build //runtime/cl_bench:cl_intercept and push the
#      resulting libcl_intercept.so, then LD_PRELOAD it before
#      litert_lm_main. Every clCreateProgramWithSource() call in
#      the closed-source delegate lands as ${DEVICE_CL_DIR}/program_NNN.cl.
#   2. We wipe on-device CL program caches before the run so the
#      delegate is forced to compile from source (otherwise it may
#      replay clCreateProgramWithBinary from a cached blob, leaving
#      us with zero .cl captures).
#   3. We disable --enable_op_profiling (profiling rebuilds the
#      delegate graph with instrumentation; we only need one clean
#      compile of every kernel path).
#   4. After the run we pull ${DEVICE_CL_DIR} back to the host as
#      runtime/cl_bench/intercepted_attn/ and scan each .cl for
#      attention-ish tokens (softmax / exp / fmax / work_group_reduce).
#
# Prereqs (same as temp_litert.sh):
#   - Bazelisk / Bazel 7.6.1+
#   - Android NDK r28b+ at $ANDROID_NDK_HOME
#   - adb connected, prebuilt/android_arm64/*.so fetched
#     (run ./fetch_android_prebuilts.sh once if missing)
#   - Model on host at $MODEL_PATH (download_gemma.sh)
#
# Usage:
#   ./temp_litert_attention_intercept.sh
#   PREFILL_TOKENS=437 ./temp_litert_attention_intercept.sh

# Re-exec under bash when invoked via `sh` (dash): we rely on
# `set -o pipefail`, `${var,,}`, arrays, [[ ]] … which dash does not
# support. Without this guard `sh temp_litert_attention_intercept.sh`
# dies on line 1 with "Illegal option -o pipefail".
if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi

export ANDROID_NDK_HOME="${ANDROID_NDK_HOME:-$HOME/neo/android-ndk-r28b}"
set -euo pipefail

# ----------------------------------------------------------------------------
# 1. Configure (defaults mirror temp_litert.sh).
# ----------------------------------------------------------------------------
DEVICE_FOLDER="${DEVICE_FOLDER:-/data/local/tmp/litert_lm}"
DEVICE_CL_DIR="${DEVICE_FOLDER}/cl_intercept"
HOST_OUT_DIR="${HOST_OUT_DIR:-runtime/cl_bench/intercepted_attn}"
MODEL_PATH_HOST="${MODEL_PATH:-$HOME/.cache/litert_lm_models/gemma-4-E2B-it.litertlm}"
PREFILL_TOKENS="${PREFILL_TOKENS:-1024}"
DECODE_TOKENS="${DECODE_TOKENS:-32}"
ASYNC="${ASYNC:-false}"
TASKSET_MASK="${TASKSET_MASK:-f0}"

if [ ! -f "${MODEL_PATH_HOST}" ]; then
  echo "[attn-intercept.sh] Model not found: ${MODEL_PATH_HOST}"
  echo "[attn-intercept.sh] Run ./download_gemma.sh first."
  exit 1
fi
if [ -z "${ANDROID_NDK_HOME:-}" ]; then
  echo "[attn-intercept.sh] ANDROID_NDK_HOME is not set."
  exit 1
fi

# Auto-upgrade from a stale NDK < r28 if r28b is already installed.
NDK_FALLBACK_R28B="${HOME}/neo/android-ndk-r28b"
case "${ANDROID_NDK_HOME}" in
  *android-ndk-r2[0-7]*)
    if [ -f "${NDK_FALLBACK_R28B}/source.properties" ]; then
      echo "[attn-intercept.sh] WARN: auto-upgrading NDK -> ${NDK_FALLBACK_R28B}"
      export ANDROID_NDK_HOME="${NDK_FALLBACK_R28B}"
    else
      echo "[attn-intercept.sh] NDK too old: ${ANDROID_NDK_HOME} (need r28b+)"
      exit 1
    fi ;;
esac
if [ ! -d "${ANDROID_NDK_HOME}" ]; then
  echo "[attn-intercept.sh] ANDROID_NDK_HOME does not exist: ${ANDROID_NDK_HOME}"
  exit 1
fi

BAZEL=$(command -v bazelisk || command -v bazel)
[ -n "${BAZEL}" ] || { echo "[attn-intercept.sh] bazel not found"; exit 1; }

echo "[attn-intercept.sh] BAZEL=${BAZEL}"
echo "[attn-intercept.sh] ANDROID_NDK_HOME=${ANDROID_NDK_HOME}"
echo "[attn-intercept.sh] MODEL (host)=${MODEL_PATH_HOST}"
echo "[attn-intercept.sh] DEVICE_FOLDER=${DEVICE_FOLDER}"
echo "[attn-intercept.sh] PREFILL_TOKENS=${PREFILL_TOKENS}"
echo "[attn-intercept.sh] DECODE_TOKENS=${DECODE_TOKENS}"
echo ""

# ----------------------------------------------------------------------------
# 2. Build litert_lm_main + cl_intercept.
# ----------------------------------------------------------------------------
echo "[attn-intercept.sh] Building litert_lm_main + cl_intercept ..."
"${BAZEL}" build \
  --config=android_arm64 \
  --define=litert_link_capi_so=true \
  --define=resolve_symbols_in_exec=false \
  //runtime/engine:litert_lm_main \
  //runtime/cl_bench:cl_intercept

BIN=bazel-bin/runtime/engine/litert_lm_main
[ -f "${BIN}" ] || { echo "[attn-intercept.sh] ${BIN} not built"; exit 1; }

INTERCEPT_SO=$(find -L bazel-bin/runtime/cl_bench -name "libcl_intercept.so" 2>/dev/null | head -1)
[ -n "${INTERCEPT_SO}" ] \
  || { echo "[attn-intercept.sh] libcl_intercept.so not found"; exit 1; }

LIBLITERT_SO=$(find -L bazel-bin -maxdepth 8 -type f -name "libLiteRt.so" \
                  ! -path "*solib*" 2>/dev/null | head -1)
if [ -z "${LIBLITERT_SO}" ]; then
  LIBLITERT_SO=$(find -L bazel-bin -type f -name "libLiteRt.so" 2>/dev/null | head -1)
fi
[ -n "${LIBLITERT_SO}" ] \
  || { echo "[attn-intercept.sh] libLiteRt.so not found"; exit 1; }

echo "[attn-intercept.sh] libLiteRt.so      : ${LIBLITERT_SO}"
echo "[attn-intercept.sh] libcl_intercept.so: ${INTERCEPT_SO}"

# Inventory of GPU accelerator shlibs (same as temp_litert.sh).
REQUIRED_PREBUILTS=(
  libLiteRtGpuAccelerator.so
  libLiteRtOpenClAccelerator.so
  libLiteRtTopKOpenClSampler.so
  libLiteRtTopKWebGpuSampler.so
  libLiteRtWebGpuAccelerator.so
  libGemmaModelConstraintProvider.so
)
PREBUILT_DIR=prebuilt/android_arm64
MISSING_PREBUILTS=()
for f in "${REQUIRED_PREBUILTS[@]}"; do
  [ -f "${PREBUILT_DIR}/${f}" ] || MISSING_PREBUILTS+=("${f}")
done
if [ "${#MISSING_PREBUILTS[@]}" -gt 0 ]; then
  echo "[attn-intercept.sh] Missing GPU accelerator shlibs in ${PREBUILT_DIR}:"
  for f in "${MISSING_PREBUILTS[@]}"; do echo "    - ${f}"; done
  echo "[attn-intercept.sh] Run ./fetch_android_prebuilts.sh"
  exit 1
fi

# ----------------------------------------------------------------------------
# 3. Push binary + prebuilts + interceptor + model.
# ----------------------------------------------------------------------------
echo ""
echo "[attn-intercept.sh] adb push ..."
adb shell "mkdir -p ${DEVICE_FOLDER} ${DEVICE_CL_DIR}"
adb shell "rm -f ${DEVICE_CL_DIR}/*.cl"

adb push "${BIN}"           "${DEVICE_FOLDER}/litert_lm_main"    >/dev/null
adb push "${LIBLITERT_SO}"  "${DEVICE_FOLDER}/libLiteRt.so"      >/dev/null
adb push "${INTERCEPT_SO}"  "${DEVICE_FOLDER}/libcl_intercept.so" >/dev/null

# Also push interceptor AS libOpenCL.so — belt-and-suspenders in case
# libLiteRtGpuAccelerator.so dlopen("libOpenCL.so")s by name (bypasses
# LD_PRELOAD since the real resolver caches the vendor path). With
# LD_LIBRARY_PATH=. the in-folder libOpenCL.so wins over
# /system/vendor/lib64/libOpenCL.so. Back up any existing on-device
# copy first so repeated runs don't clobber real state, and restore it
# at end-of-script (trap) on any exit path.
if adb shell "test -f ${DEVICE_FOLDER}/libOpenCL.so"; then
  # Only back up if the existing file is NOT our interceptor (avoid
  # backing up a stale interceptor as if it were vendor libOpenCL.so).
  if ! adb shell "cmp -s ${DEVICE_FOLDER}/libOpenCL.so ${DEVICE_FOLDER}/libcl_intercept.so"; then
    echo "[attn-intercept.sh] Backing up pre-existing ${DEVICE_FOLDER}/libOpenCL.so"
    adb shell "mv ${DEVICE_FOLDER}/libOpenCL.so ${DEVICE_FOLDER}/libOpenCL.so.bak"
  fi
fi
adb push "${INTERCEPT_SO}"  "${DEVICE_FOLDER}/libOpenCL.so"       >/dev/null
restore_libopencl() {
  if adb shell "test -f ${DEVICE_FOLDER}/libOpenCL.so.bak" 2>/dev/null; then
    adb shell "rm -f ${DEVICE_FOLDER}/libOpenCL.so; \
               mv ${DEVICE_FOLDER}/libOpenCL.so.bak ${DEVICE_FOLDER}/libOpenCL.so" \
      2>/dev/null || true
  else
    # No vendor copy was there -- just remove our shim so subsequent
    # non-intercepted runs pick up the real /system/vendor/.../libOpenCL.so.
    adb shell "rm -f ${DEVICE_FOLDER}/libOpenCL.so" 2>/dev/null || true
  fi
}
trap restore_libopencl EXIT

for f in "${REQUIRED_PREBUILTS[@]}"; do
  adb push "${PREBUILT_DIR}/${f}" "${DEVICE_FOLDER}/${f}" >/dev/null
done
adb shell "chmod +x ${DEVICE_FOLDER}/litert_lm_main"

# Model push is expensive; only push if size differs or missing
# (same logic as temp_litert.sh).
MODEL_BASENAME=$(basename "${MODEL_PATH_HOST}")
HOST_SIZE=$(stat -c '%s' "${MODEL_PATH_HOST}" 2>/dev/null || stat -f '%z' "${MODEL_PATH_HOST}")
DEV_SIZE=$(adb shell "stat -c '%s' ${DEVICE_FOLDER}/${MODEL_BASENAME} 2>/dev/null || echo 0" | tr -d '\r')
if [ "${DEV_SIZE}" = "${HOST_SIZE}" ]; then
  echo "[attn-intercept.sh] Model already on device, size matches. Skipping push."
else
  echo "[attn-intercept.sh] Pushing model (${HOST_SIZE} bytes) ..."
  adb push "${MODEL_PATH_HOST}" "${DEVICE_FOLDER}/${MODEL_BASENAME}"
fi

# Clear CL program caches so the delegate is forced to compile from
# source (otherwise clCreateProgramWithBinary replays a cached blob
# and no .cl files are ever created). Hit every location Adreno /
# LiteRT are known to use, then sweep everything under $DEVICE_FOLDER
# that looks cache-ish.
echo "[attn-intercept.sh] Clearing on-device CL caches ..."
adb shell "rm -rf ${DEVICE_FOLDER}/cache/*                2>/dev/null; \
           rm -rf ${DEVICE_FOLDER}/*.cache                2>/dev/null; \
           rm -rf ${DEVICE_FOLDER}/compilation_cache_*    2>/dev/null; \
           rm -rf ${DEVICE_FOLDER}/kernel_cache*          2>/dev/null; \
           rm -rf ${DEVICE_FOLDER}/.adrenocompiledkernel  2>/dev/null; \
           rm -rf /data/local/tmp/cl_cache/*              2>/dev/null; \
           rm -rf /data/local/tmp/kernel_cache/*          2>/dev/null; \
           rm -rf /data/local/tmp/.adrenocompiledkernel   2>/dev/null; \
           rm -rf /sdcard/.adrenocompiledkernel           2>/dev/null; \
           rm -rf /data/data/*/cache/cl_cache/*           2>/dev/null; \
           rm -rf /data/data/*/code_cache/.adrenocompiledkernel 2>/dev/null; \
           rm -rf /data/user_de/0/*/code_cache/.adrenocompiledkernel 2>/dev/null; \
           true" 2>/dev/null || true
# Nuke anything cache-ish that survived inside the device folder.
adb shell "find ${DEVICE_FOLDER} -maxdepth 3 \\
  \\( -iname '*cache*' -o -iname '*adreno*kernel*' -o -iname '*.bin.cached' \\) \\
  -print -exec rm -rf {} + 2>/dev/null; true" 2>/dev/null || true

# ----------------------------------------------------------------------------
# 4. Run litert_lm_main under LD_PRELOAD intercept.
#    --enable_op_profiling=false on purpose: profiling rebuilds the
#    delegate graph with instrumentation and we only need ONE clean
#    compile of every kernel. benchmark mode still exercises the full
#    prefill + decode graphs.
# ----------------------------------------------------------------------------
RUN_LOG=temp_litert_attn_intercept_run.log
echo ""
echo "[attn-intercept.sh] Running (taskset ${TASKSET_MASK}) ..."
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
    --report_peak_memory_footprint=true \
    --enable_op_profiling=false" \
  2>&1 | tee "${RUN_LOG}"

# Diagnose interceptor status from the run log. Three states:
#   (a) zero [cl_intercept] lines      -> interceptor did NOT load
#       (LD_PRELOAD ignored, or .so path wrong)
#   (b) "Real OpenCL loaded" but no "#N:" lines
#                                      -> interceptor loaded BUT delegate
#       used clCreateProgramWithBinary (cached blob; we didn't wipe all
#       caches)
#   (c) one or more "#N:" lines        -> interceptor worked; we have
#                                         captures
echo ""
echo "[attn-intercept.sh] Interceptor diagnostic:"
LOAD_HITS=$(grep -c '\[cl_intercept\] Real OpenCL loaded' "${RUN_LOG}" || true)
DUMP_HITS=$(grep -c '\[cl_intercept\] #' "${RUN_LOG}" || true)
echo "  'Real OpenCL loaded' markers: ${LOAD_HITS}"
echo "  per-program dump markers    : ${DUMP_HITS}"
if [ "${LOAD_HITS}" = "0" ]; then
  echo "  -> FATAL: interceptor never loaded. Check whether the"
  echo "     delegate resolves libOpenCL.so via dlopen with an absolute"
  echo "     vendor path; if so, consider the binary-patch approach in"
  echo "     temp_litert_cl_patch_intercept.sh."
elif [ "${DUMP_HITS}" = "0" ]; then
  echo "  -> Interceptor loaded but zero sources intercepted. The"
  echo "     delegate is using a precompiled blob via"
  echo "     clCreateProgramWithBinary. Hunt for leftover cache files:"
  echo "     adb shell 'find / -iname \"*.adrenocompiledkernel*\" 2>/dev/null'"
fi

# ----------------------------------------------------------------------------
# 5. Pull captures + scan for attention kernels.
# ----------------------------------------------------------------------------
echo ""
echo "[attn-intercept.sh] Pulling captured CL sources ..."
mkdir -p "${HOST_OUT_DIR}"
rm -f "${HOST_OUT_DIR}"/*.cl 2>/dev/null || true
CL_COUNT=$(adb shell "ls ${DEVICE_CL_DIR}/*.cl 2>/dev/null | wc -l" | tr -d '\r ')

if [ "${CL_COUNT}" -gt 0 ]; then
  adb shell "ls ${DEVICE_CL_DIR}/*.cl 2>/dev/null" | tr -d '\r' | while read -r f; do
    [ -z "$f" ] && continue
    base=$(basename "$f")
    adb pull "$f" "${HOST_OUT_DIR}/${base}" >/dev/null 2>&1
  done
  echo "=========================================="
  echo " Captured ${CL_COUNT} CL program(s) -> ${HOST_OUT_DIR}/"
  echo "=========================================="
  echo ""
  echo "-- Attention-ish matches (softmax / exp( / fmax / work_group_reduce / running_max / flash): --"
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
    echo "  (none — LiteRT likely runs softmax on CPU, splits attention"
    echo "   into matmul+softmax+matmul primitives, or ships a pre-compiled"
    echo "   binary blob. Inspect ${HOST_OUT_DIR}/ manually to confirm.)"
  fi
else
  echo "[attn-intercept.sh] WARN: no .cl captured — delegate likely used"
  echo "  clCreateProgramWithBinary from a cached blob. Check remaining"
  echo "  cache locations on device and re-run."
fi

echo ""
echo "Full run log: ${RUN_LOG}"
