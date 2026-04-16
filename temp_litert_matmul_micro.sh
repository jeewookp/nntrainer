#!/usr/bin/env bash
# temp_litert_matmul_micro.sh
#
# Standalone matmul micro benchmark runner. Splits out from temp_litert.sh
# so the per-shape latency loop can be re-run quickly without paying the
# cost of another 1024-token Gemma4 prefill on the device.
#
# Workflow (typical):
#
#   1. ./temp_litert.sh                    # builds litert_lm_main, runs
#                                          # prefill on device, pulls
#                                          # ./matmul_roster.csv to host
#   2. ./temp_litert_matmul_micro.sh       # builds matmul_micro_benchmark,
#                                          # pushes it to device, runs it
#                                          # against the roster, pulls
#                                          # ./matmul_micro.csv to host
#
# Step 2 can be re-run independently as long as ./matmul_roster.csv exists
# on the host (or on the device at ${MATMUL_ROSTER_CSV_DEVICE}). It does
# NOT re-build litert_lm_main, push the model, or run prefill again --
# only matmul_micro_benchmark itself plus its libLiteRt.so dependency.
#
# What you get:
#   - temp_litert_matmul_micro.log : full stdout/stderr of the run
#   - ./matmul_micro.csv           : CSV with one row per (M,N,K), columns
#                                    m,n,k,count,min_us,avg_us,p50_us,
#                                    p95_us,max_us,gflops,tflops_min,
#                                    tflops_avg
#
# Why it exists separately:
#   - The micro benchmark builds + compiles ONE single-FC tflite model
#     per shape via litert::CompiledModel::Create, which is fast on its
#     own but pays a large OpenCL kernel-compile / weight-upload cost
#     for every fresh CompiledModel. Iterating on the micro benchmark
#     during analysis (changing dtype, iters, shapes) is much faster
#     when prefill isn't re-run every time.
#   - Lets the user feed in custom shapes via --shapes= without going
#     through a full prefill cycle to seed the roster.
#
# Env var overrides:
#   DEVICE_FOLDER           where to push on device (default /data/local/tmp/litert_lm)
#   HOST_MATMUL_ROSTER_CSV  shape source on host (default ./matmul_roster.csv)
#   MATMUL_MICRO_DTYPE      int8_per_tensor (default; single FC op,
#                           int8 per-tensor weights. Fires
#                           `fully_connected_int8` for small M but
#                           falls back to fp16 convolution1x1 for
#                           large M.),
#                           int8_conv2d_per_tensor (single CONV_2D 1x1
#                           op, int8 per-tensor weights. Native CONV_2D
#                           trigger hits `convolution_int8(conv_wave_memory)`
#                           matching prefill's int8 matmul row; use
#                           this mode to get per-shape numbers that
#                           are directly comparable to prefill.),
#                           int8 (wrapped FC int8; delegate compiles
#                           as fp16 conv1x1),
#                           int8_chain (2-FC int8 chain; delegate
#                           compiles as fp16 conv1x1),
#                           int8_hybrid (int8 weights + fp32 acts;
#                           delegate accepts but does NOT trigger the
#                           int8 kernel),
#                           fp32 (baseline; matches the ~20% fp share
#                           via convolution(conv_wave_memory)),
#                           fp32_conv2d (single CONV_2D 1x1 fp32),
#                           or fp16 (broken)
#   MATMUL_MICRO_WARMUP     warmup iters per shape (default 5)
#   MATMUL_MICRO_ITERS      timed iters per shape (default 50)
#   MATMUL_MICRO_MAX_SHAPES if > 0, only benchmark first N unique shapes
#   MATMUL_MICRO_SHAPES     inline shapes list "1024x6144x1536,..." -- if
#                           set, used INSTEAD of the roster CSV
#   TASKSET_MASK            taskset mask for big-core pinning (default f0)
#   MATMUL_MICRO_CACHE_DIR  device path for the LiteRT OpenCL program
#                           cache (default ${DEVICE_FOLDER}/cache). On the
#                           first run each shape still pays full JIT +
#                           weight upload and writes the compiled program
#                           under this dir; subsequent runs with the same
#                           shape + dtype skip the JIT and load from
#                           cache, cutting the per-shape
#                           CompiledModel::Create wall-clock by ~10-100x
#                           on Adreno. Set MATMUL_MICRO_CACHE_DIR="" to
#                           disable (cold-compile baseline).
#   ANDROID_NDK_HOME        same as temp_litert.sh

export ANDROID_NDK_HOME="${ANDROID_NDK_HOME:-$HOME/neo/android-ndk-r28b}"

set -euo pipefail

# ----------------------------------------------------------------------------
# 1. Configure
# ----------------------------------------------------------------------------
DEVICE_FOLDER="${DEVICE_FOLDER:-/data/local/tmp/litert_lm}"
HOST_MATMUL_ROSTER_CSV="${HOST_MATMUL_ROSTER_CSV:-./matmul_roster.csv}"
# int8_per_tensor (default) = single FULLY_CONNECTED op with INT8
# weights using PER-TENSOR symmetric quant (single scale + single
# zero_point), FLOAT32 bias, FLOAT32 signature I/O.
#
# Why per-tensor and not per-channel: from origin/litert's
# `tflite/delegates/gpu/common/lstm_parser.cc:62`, the open source
# TFLite GPU CL delegate only constructs a `FullyConnectedInt8Attributes`
# (which is what triggers the int8 FC kernel selection in
# `operation_selector.cc::FULLY_CONNECTED_INT8`) when:
#
#     weights_tensor->type == kTfLiteInt8 &&
#     quant_params->scale->size == 1
#
# The closed-source ml_drift CL accelerator (libLiteRtGpuAccelerator.so)
# is a fork of this code and almost certainly uses the same trigger.
# All earlier int8 attempts in this builder used per-CHANNEL quant
# (n scales) which the trigger explicitly rejects -- that's why the
# kernel name stayed `convolution1x1(conv_wave_memory)` (fp16 path)
# instead of `convolution_int8(conv_wave_memory)` (the prefill match).
#
# Other valid values:
#   int8_chain  : 2-FC int8 chain (failed experiment).
#   fp32_conv2d : single CONV_2D 1x1 op, fp32 (failed -- same kernel
#                 entry point as FC).
#   int8        : 3-op QUANTIZE -> FC int8 -> DEQUANTIZE wrapper
#                 with per-channel quant (failed).
#   int8_hybrid : single FC op, per-channel int8 weights + fp32 acts
#                 + asymmetric_quantize_inputs=true (failed).
#   fp32        : single FC op, FLOAT32 throughout. Baseline.
#   fp16        : broken (CPU FC reference kernel asserts fp32).
# Default is int8_conv2d_per_tensor because that is the schema
# expected to trigger `convolution_int8(conv_wave_memory)` on the CL
# delegate, matching prefill's int8 matmul kernel. The older
# int8_per_tensor (FULLY_CONNECTED) path falls back to
# `convolution1x1(conv_wave_memory)` in fp16 for large M and is kept
# only for comparison. Override to int8_per_tensor etc. when you
# specifically want the FC rewrite path.
MATMUL_MICRO_DTYPE="${MATMUL_MICRO_DTYPE:-int8_conv2d_per_tensor}"
MATMUL_MICRO_WARMUP="${MATMUL_MICRO_WARMUP:-5}"
MATMUL_MICRO_ITERS="${MATMUL_MICRO_ITERS:-50}"
MATMUL_MICRO_MAX_SHAPES="${MATMUL_MICRO_MAX_SHAPES:-0}"
MATMUL_MICRO_SHAPES="${MATMUL_MICRO_SHAPES:-}"
# Op profiling: when true, enables LiteRT op profiling on each shape and
# dumps the Delegate Statistics summary to stderr so we can see exactly
# which CL kernel the GPU delegate picks (e.g.
# `convolution_int8(conv_wave_memory)` vs the fp32 fallback). Default
# is true while we're still chasing the kernel-selection mismatch
# between the synthetic micro and Gemma4 prefill -- the verbose output
# is the canonical signal for "is the int8 path being hit?". Turn off
# (`MATMUL_MICRO_PROFILE=false`) once the timings match prefill and
# you just want clean steady-state numbers.
MATMUL_MICRO_PROFILE="${MATMUL_MICRO_PROFILE:-true}"
TASKSET_MASK="${TASKSET_MASK:-f0}"

# OpenCL program cache. When set, matmul_micro_benchmark passes
# --cache_dir=<path> through to litert::GpuOptions::SetSerializationDir
# + SetSerializeExternalTensors(true) + a per-(M,N,K,dtype) model cache
# key. First run populates the cache (wall-clock unchanged). Subsequent
# runs with the same shape roster skip OpenCL JIT on CompiledModel::Create
# -- this is the main knob for closing the "per-shape JIT overhead" gap
# between the micro benchmark and the one-shot prefill CompiledModel.
#
# Default: on, pointing at a persistent on-device path. Set
# MATMUL_MICRO_CACHE_DIR="" to disable and re-measure cold-compile
# wall-clock. The cache dir is created lazily on the device before the
# run; it survives adb wipes of the binary because it sits under
# DEVICE_FOLDER alongside the prebuilts.
MATMUL_MICRO_CACHE_DIR="${MATMUL_MICRO_CACHE_DIR:-${DEVICE_FOLDER}/cache}"

MATMUL_ROSTER_CSV_DEVICE=${DEVICE_FOLDER}/matmul_roster.csv
MATMUL_MICRO_CSV_DEVICE=${DEVICE_FOLDER}/matmul_micro.csv
MATMUL_MICRO_LOG=temp_litert_matmul_micro.log
HOST_MATMUL_MICRO_CSV=./matmul_micro.csv

# Same NDK auto-upgrade dance as temp_litert.sh: r26d in the user's
# ~/.bashrc would silently override the default we set above, but the
# rust crate index won't compile against r26d's libc++. Auto-bump to
# r28b if it's installed at the canonical location.
NDK_FALLBACK_R28B="${HOME}/neo/android-ndk-r28b"
case "${ANDROID_NDK_HOME}" in
  *android-ndk-r2[0-7]*)
    if [ -f "${NDK_FALLBACK_R28B}/source.properties" ]; then
      echo "[micro.sh] WARNING: caller exported ANDROID_NDK_HOME=${ANDROID_NDK_HOME}"
      echo "[micro.sh]          auto-upgrading to ${NDK_FALLBACK_R28B}"
      export ANDROID_NDK_HOME="${NDK_FALLBACK_R28B}"
    else
      echo "[micro.sh] ANDROID_NDK_HOME points at NDK < r28: ${ANDROID_NDK_HOME}"
      echo "[micro.sh] Install r28b first (./install_android_ndk.sh)."
      exit 1
    fi
    ;;
esac

if [ ! -d "${ANDROID_NDK_HOME}" ]; then
  echo "[micro.sh] ANDROID_NDK_HOME does not exist: ${ANDROID_NDK_HOME}"
  exit 1
fi

BAZEL=$(command -v bazelisk || command -v bazel)
if [ -z "${BAZEL}" ]; then
  echo "[micro.sh] Neither bazelisk nor bazel found in PATH."
  exit 1
fi

echo "[micro.sh] BAZEL=${BAZEL}"
echo "[micro.sh] ANDROID_NDK_HOME=${ANDROID_NDK_HOME}"
echo "[micro.sh] DEVICE_FOLDER=${DEVICE_FOLDER}"
echo "[micro.sh] DTYPE=${MATMUL_MICRO_DTYPE}"
echo "[micro.sh] WARMUP=${MATMUL_MICRO_WARMUP}"
echo "[micro.sh] ITERS=${MATMUL_MICRO_ITERS}"
echo "[micro.sh] PROFILE=${MATMUL_MICRO_PROFILE}"
echo "[micro.sh] CACHE_DIR=${MATMUL_MICRO_CACHE_DIR:-(disabled)}"
if [ -n "${MATMUL_MICRO_SHAPES}" ]; then
  echo "[micro.sh] SHAPES (inline)=${MATMUL_MICRO_SHAPES}"
else
  echo "[micro.sh] HOST_MATMUL_ROSTER_CSV=${HOST_MATMUL_ROSTER_CSV}"
fi
if [ "${MATMUL_MICRO_MAX_SHAPES}" -gt 0 ]; then
  echo "[micro.sh] MAX_SHAPES=${MATMUL_MICRO_MAX_SHAPES}"
fi
echo ""

# ----------------------------------------------------------------------------
# 2. Build matmul_micro_benchmark for android_arm64 (mirrors temp_litert.sh
#    flags: separate libLiteRt.so via --define=litert_link_capi_so=true).
# ----------------------------------------------------------------------------
echo "[micro.sh] Building //runtime/engine:matmul_micro_benchmark ..."
"${BAZEL}" build \
  --config=android_arm64 \
  --define=litert_link_capi_so=true \
  --define=resolve_symbols_in_exec=false \
  //runtime/engine:matmul_micro_benchmark

MATMUL_MICRO_BIN=bazel-bin/runtime/engine/matmul_micro_benchmark
if [ ! -f "${MATMUL_MICRO_BIN}" ]; then
  echo "[micro.sh] Build succeeded but binary not at ${MATMUL_MICRO_BIN}"
  exit 1
fi

# Locate libLiteRt.so the same way temp_litert.sh does. The micro binary
# dlopens it, so it has to live next to the binary on the device. We
# prefer the non-solib copy under bazel-bin/external/litert/ but fall
# back to anything bazel produced.
LIBLITERT_SO=$(find -L bazel-bin -maxdepth 8 -type f -name "libLiteRt.so" \
                  ! -path "*solib*" 2>/dev/null | head -1)
if [ -z "${LIBLITERT_SO}" ]; then
  LIBLITERT_SO=$(find -L bazel-bin -type f -name "libLiteRt.so" 2>/dev/null | head -1)
fi
if [ -z "${LIBLITERT_SO}" ]; then
  echo "[micro.sh] libLiteRt.so not found under bazel-bin/."
  echo "    Run ./temp_litert.sh first, or rebuild with"
  echo "    --define=litert_link_capi_so=true."
  exit 1
fi
echo "[micro.sh] libLiteRt.so : ${LIBLITERT_SO}"

# GPU accelerator shlibs. Same inventory as temp_litert.sh -- they're
# LFS-tracked prebuilts and we assume temp_litert.sh has already pushed
# them. We re-push libLiteRt.so + the binary every run because they
# change with each rebuild, but the prebuilts only need to be present
# on device (we check, but don't re-push if the size matches).
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
  if [ ! -f "${PREBUILT_DIR}/${f}" ]; then
    MISSING_PREBUILTS+=("${f}")
  fi
done
if [ "${#MISSING_PREBUILTS[@]}" -gt 0 ]; then
  echo "[micro.sh] Missing GPU accelerator shlibs in ${PREBUILT_DIR}:"
  for f in "${MISSING_PREBUILTS[@]}"; do
    echo "    - ${f}"
  done
  echo "[micro.sh] Fetch with:  ./fetch_android_prebuilts.sh"
  exit 1
fi

# ----------------------------------------------------------------------------
# 3. Push binary + libLiteRt.so + (if missing on device) GPU accelerator
#    shlibs. Skip the model push entirely -- the micro benchmark builds
#    its own single-op tflite flatbuffers in process memory and never
#    touches the Gemma4 weights.
# ----------------------------------------------------------------------------
echo ""
echo "[micro.sh] adb push binary + libLiteRt.so ..."
adb shell "mkdir -p ${DEVICE_FOLDER}"
adb push "${MATMUL_MICRO_BIN}" "${DEVICE_FOLDER}/matmul_micro_benchmark" >/dev/null
adb push "${LIBLITERT_SO}" "${DEVICE_FOLDER}/libLiteRt.so" >/dev/null
adb shell "chmod +x ${DEVICE_FOLDER}/matmul_micro_benchmark"

for f in "${REQUIRED_PREBUILTS[@]}"; do
  if ! adb shell "test -f ${DEVICE_FOLDER}/${f}"; then
    echo "[micro.sh] device missing ${f}, pushing ..."
    adb push "${PREBUILT_DIR}/${f}" "${DEVICE_FOLDER}/${f}" >/dev/null
  fi
done

# ----------------------------------------------------------------------------
# 4. Resolve the shape source. Two modes:
#    a) inline: MATMUL_MICRO_SHAPES="MxNxK,..." -- used as-is, roster CSV
#       is ignored. Useful for ad-hoc shape sweeps.
#    b) roster: HOST_MATMUL_ROSTER_CSV (default ./matmul_roster.csv) is
#       pushed to the device and matmul_micro_benchmark consumes it
#       via --shapes_csv. If the host file is missing we fall back to
#       the on-device copy ${MATMUL_ROSTER_CSV_DEVICE} from a previous
#       temp_litert.sh run.
# ----------------------------------------------------------------------------
SHAPES_FLAG=""
if [ -n "${MATMUL_MICRO_SHAPES}" ]; then
  SHAPES_FLAG="--shapes=${MATMUL_MICRO_SHAPES}"
  echo ""
  echo "[micro.sh] using inline shapes: ${MATMUL_MICRO_SHAPES}"
else
  if [ -f "${HOST_MATMUL_ROSTER_CSV}" ]; then
    echo ""
    echo "[micro.sh] Pushing ${HOST_MATMUL_ROSTER_CSV} -> ${MATMUL_ROSTER_CSV_DEVICE} ..."
    adb push "${HOST_MATMUL_ROSTER_CSV}" "${MATMUL_ROSTER_CSV_DEVICE}" >/dev/null
  elif adb shell "test -f ${MATMUL_ROSTER_CSV_DEVICE}"; then
    echo ""
    echo "[micro.sh] host ${HOST_MATMUL_ROSTER_CSV} not found; using device cached"
    echo "[micro.sh]   ${MATMUL_ROSTER_CSV_DEVICE}"
  else
    echo ""
    echo "[micro.sh] ERROR: no matmul roster CSV found."
    echo "    Looked in:"
    echo "        host:   ${HOST_MATMUL_ROSTER_CSV}"
    echo "        device: ${MATMUL_ROSTER_CSV_DEVICE}"
    echo "    Run ./temp_litert.sh first to produce one, or pass shapes inline:"
    echo "        MATMUL_MICRO_SHAPES='1024x6144x1536,1024x1536x6144' \\"
    echo "          ./temp_litert_matmul_micro.sh"
    exit 1
  fi
  SHAPES_FLAG="--shapes_csv=${MATMUL_ROSTER_CSV_DEVICE}"
fi

MAX_SHAPES_FLAG=""
if [ "${MATMUL_MICRO_MAX_SHAPES}" -gt 0 ]; then
  MAX_SHAPES_FLAG="--max_shapes=${MATMUL_MICRO_MAX_SHAPES}"
fi

PROFILE_FLAG=""
case "${MATMUL_MICRO_PROFILE,,}" in
  1|true|yes|on)  PROFILE_FLAG="--profile=true"  ;;
  *)              PROFILE_FLAG="--profile=false" ;;
esac

# Cache dir: when set, pre-create on device and pass --cache_dir to the
# binary so litert::GpuOptions::SetSerializationDir has a writable
# target. Leave empty to disable the OpenCL program cache entirely.
CACHE_DIR_FLAG=""
if [ -n "${MATMUL_MICRO_CACHE_DIR}" ]; then
  adb shell "mkdir -p ${MATMUL_MICRO_CACHE_DIR}"
  CACHE_DIR_FLAG="--cache_dir=${MATMUL_MICRO_CACHE_DIR}"
fi

# ----------------------------------------------------------------------------
# 5. Run on device.
# ----------------------------------------------------------------------------
echo ""
echo "[micro.sh] Running matmul_micro_benchmark on device (taskset ${TASKSET_MASK}) ..."
adb shell "cd ${DEVICE_FOLDER}; \
  LD_LIBRARY_PATH=. \
  taskset ${TASKSET_MASK} ./matmul_micro_benchmark \
    --backend=gpu \
    --dtype=${MATMUL_MICRO_DTYPE} \
    ${SHAPES_FLAG} \
    --warmup=${MATMUL_MICRO_WARMUP} \
    --iters=${MATMUL_MICRO_ITERS} \
    ${MAX_SHAPES_FLAG} \
    ${PROFILE_FLAG} \
    ${CACHE_DIR_FLAG} \
    --csv_out=${MATMUL_MICRO_CSV_DEVICE}" \
  2>&1 | tee "${MATMUL_MICRO_LOG}"

# ----------------------------------------------------------------------------
# 6. Pull the result CSV back and print a quick preview.
# ----------------------------------------------------------------------------
if adb shell "test -f ${MATMUL_MICRO_CSV_DEVICE}"; then
  adb pull "${MATMUL_MICRO_CSV_DEVICE}" "${HOST_MATMUL_MICRO_CSV}" >/dev/null 2>&1
  echo ""
  echo "=========================================="
  echo " Matmul micro CSV (host: ${HOST_MATMUL_MICRO_CSV})"
  echo "=========================================="
  echo "    The CSV has one row per unique (M,N,K). Compare each row"
  echo "    against the prefill Delegate Statistics rows in"
  echo "    temp_litert_run.log to attribute aggregated"
  echo "    convolution(conv_wave_memory) / convolution_int8(...) cost."
  echo ""
  head -n 30 "${HOST_MATMUL_MICRO_CSV}" || true
  echo ""
  echo "Full log:    ${MATMUL_MICRO_LOG}"
  echo "Full CSV:    ${HOST_MATMUL_MICRO_CSV}"
else
  echo ""
  echo "[micro.sh] WARNING: ${MATMUL_MICRO_CSV_DEVICE} not produced. Check ${MATMUL_MICRO_LOG}."
  exit 1
fi
echo ""
