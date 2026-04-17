#!/usr/bin/env bash
# temp_litert_conv_bench.sh
#
# Matmul benchmark using TFLite GPU delegate's ConvGeneric kernel directly.
# This links the open-source TFLite GPU CL runtime (NOT the closed-source
# libLiteRtGpuAccelerator.so), so we get the same optimized
# "convolution(conv_wave_memory)" kernel that achieves 5+ TFLOPS on Adreno.
#
# Usage:
#   ./temp_litert_conv_bench.sh
#   CL_BENCH_SHAPES="1024x6144x1536" ./temp_litert_conv_bench.sh
#   CL_BENCH_PRECISION=fp32 ./temp_litert_conv_bench.sh
#
# Env vars:
#   DEVICE_FOLDER           (default /data/local/tmp/litert_lm)
#   HOST_MATMUL_ROSTER_CSV  (default ./matmul_roster.csv)
#   CL_BENCH_PRECISION      fp16 (default), fp32, or fp16acc32
#   CL_BENCH_WARMUP         (default 5)
#   CL_BENCH_ITERS          (default 50)
#   CL_BENCH_SHAPES         inline "MxNxK,..."
#   CL_BENCH_MAX_TENSOR_MB  (default 512)
#   TASKSET_MASK            (default f0)

export ANDROID_NDK_HOME="${ANDROID_NDK_HOME:-$HOME/neo/android-ndk-r28b}"

set -euo pipefail

DEVICE_FOLDER="${DEVICE_FOLDER:-/data/local/tmp/litert_lm}"
HOST_MATMUL_ROSTER_CSV="${HOST_MATMUL_ROSTER_CSV:-./matmul_roster.csv}"
CL_BENCH_PRECISION="${CL_BENCH_PRECISION:-fp16}"
CL_BENCH_WARMUP="${CL_BENCH_WARMUP:-5}"
CL_BENCH_ITERS="${CL_BENCH_ITERS:-50}"
CL_BENCH_SHAPES="${CL_BENCH_SHAPES:-}"
CL_BENCH_MAX_TENSOR_MB="${CL_BENCH_MAX_TENSOR_MB:-512}"
TASKSET_MASK="${TASKSET_MASK:-f0}"

CONV_BENCH_CSV_DEVICE="${DEVICE_FOLDER}/conv_bench.csv"
CONV_BENCH_LOG="temp_litert_conv_bench.log"
HOST_CONV_BENCH_CSV="./conv_bench.csv"
MATMUL_ROSTER_CSV_DEVICE="${DEVICE_FOLDER}/matmul_roster.csv"

NDK_FALLBACK_R28B="${HOME}/neo/android-ndk-r28b"
case "${ANDROID_NDK_HOME}" in
  *android-ndk-r2[0-7]*)
    if [ -f "${NDK_FALLBACK_R28B}/source.properties" ]; then
      echo "[conv_bench.sh] WARNING: auto-upgrading NDK to ${NDK_FALLBACK_R28B}"
      export ANDROID_NDK_HOME="${NDK_FALLBACK_R28B}"
    else
      echo "[conv_bench.sh] NDK < r28 not supported"; exit 1
    fi ;;
esac
[ -d "${ANDROID_NDK_HOME}" ] || { echo "[conv_bench.sh] NDK not found: ${ANDROID_NDK_HOME}"; exit 1; }

BAZEL=$(command -v bazelisk || command -v bazel)
[ -n "${BAZEL}" ] || { echo "[conv_bench.sh] bazel not found"; exit 1; }

echo "[conv_bench.sh] === ConvGeneric Matmul Benchmark ==="
echo "[conv_bench.sh] PRECISION=${CL_BENCH_PRECISION} WARMUP=${CL_BENCH_WARMUP} ITERS=${CL_BENCH_ITERS}"
if [ -n "${CL_BENCH_SHAPES}" ]; then
  echo "[conv_bench.sh] SHAPES=${CL_BENCH_SHAPES}"
else
  echo "[conv_bench.sh] ROSTER=${HOST_MATMUL_ROSTER_CSV}"
fi
echo ""

# Build.
echo "[conv_bench.sh] Building //runtime/cl_bench:conv_generic_bench ..."
"${BAZEL}" build \
  --config=android_arm64 \
  //runtime/cl_bench:conv_generic_bench

CONV_BENCH_BIN=bazel-bin/runtime/cl_bench/conv_generic_bench
[ -f "${CONV_BENCH_BIN}" ] || { echo "[conv_bench.sh] Binary not found"; exit 1; }

# Push.
echo ""
echo "[conv_bench.sh] adb push ..."
adb shell "mkdir -p ${DEVICE_FOLDER}"
adb push "${CONV_BENCH_BIN}" "${DEVICE_FOLDER}/conv_generic_bench" >/dev/null
adb shell "chmod +x ${DEVICE_FOLDER}/conv_generic_bench"

# Shapes.
SHAPES_FLAG=""
if [ -n "${CL_BENCH_SHAPES}" ]; then
  SHAPES_FLAG="--shapes=${CL_BENCH_SHAPES}"
else
  if [ -f "${HOST_MATMUL_ROSTER_CSV}" ]; then
    adb push "${HOST_MATMUL_ROSTER_CSV}" "${MATMUL_ROSTER_CSV_DEVICE}" >/dev/null
  elif ! adb shell "test -f ${MATMUL_ROSTER_CSV_DEVICE}"; then
    echo "[conv_bench.sh] ERROR: no roster CSV. Run ./temp_litert.sh first or set CL_BENCH_SHAPES."
    exit 1
  fi
  SHAPES_FLAG="--shapes_csv=${MATMUL_ROSTER_CSV_DEVICE}"
fi

# Run.
echo ""
echo "[conv_bench.sh] Running on device (taskset ${TASKSET_MASK}) ..."
adb shell "cd ${DEVICE_FOLDER}; \
  LD_LIBRARY_PATH=/system/vendor/lib64:. \
  taskset ${TASKSET_MASK} ./conv_generic_bench \
    --precision=${CL_BENCH_PRECISION} \
    ${SHAPES_FLAG} \
    --warmup=${CL_BENCH_WARMUP} \
    --iters=${CL_BENCH_ITERS} \
    --max_tensor_mb=${CL_BENCH_MAX_TENSOR_MB} \
    --csv_out=${CONV_BENCH_CSV_DEVICE}" \
  2>&1 | tee "${CONV_BENCH_LOG}"

# Pull results.
if adb shell "test -f ${CONV_BENCH_CSV_DEVICE}"; then
  adb pull "${CONV_BENCH_CSV_DEVICE}" "${HOST_CONV_BENCH_CSV}" >/dev/null 2>&1
  echo ""
  echo "=========================================="
  echo " ConvGeneric Matmul Bench CSV"
  echo "=========================================="
  echo "    Uses the same ConvGeneric kernel as production prefill."
  echo "    tflops_wall should be close to prefill's 5+ TFLOPS."
  echo ""
  head -n 30 "${HOST_CONV_BENCH_CSV}" || true
  echo ""
  echo "Full log: ${CONV_BENCH_LOG}"
  echo "Full CSV: ${HOST_CONV_BENCH_CSV}"
else
  echo "[conv_bench.sh] WARNING: CSV not produced. Check ${CONV_BENCH_LOG}."
  exit 1
fi
echo ""
