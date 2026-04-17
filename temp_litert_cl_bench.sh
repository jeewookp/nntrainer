#!/usr/bin/env bash
# temp_litert_cl_bench.sh
#
# Direct OpenCL matmul benchmark runner. Bypasses LiteRT entirely --
# dlopen's libOpenCL.so on device and dispatches a tiled GEMM kernel
# directly. This gives a clean wall-clock TFLOPS number without LiteRT's
# per-Run() overhead (~8.5ms framework + 2.1ms upload + 0.4ms
# weights_convert).
#
# Usage:
#   ./temp_litert_cl_bench.sh                                  # default shapes from matmul_roster.csv
#   CL_BENCH_SHAPES="1024x6144x1536,1024x1536x6144" ./temp_litert_cl_bench.sh  # inline shapes
#   CL_BENCH_DTYPE=fp32 ./temp_litert_cl_bench.sh              # fp32 mode
#
# Env var overrides:
#   DEVICE_FOLDER           where to push on device (default /data/local/tmp/litert_lm)
#   HOST_MATMUL_ROSTER_CSV  shape source on host (default ./matmul_roster.csv)
#   CL_BENCH_DTYPE          fp16 (default) or fp32
#   CL_BENCH_WARMUP         warmup iters per shape (default 5)
#   CL_BENCH_ITERS          timed iters per shape (default 50)
#   CL_BENCH_SHAPES         inline shapes "MxNxK,..." -- if set, used INSTEAD of roster CSV
#   CL_BENCH_MAX_SHAPES     if > 0, only benchmark first N unique shapes
#   CL_BENCH_MAX_TENSOR_MB  max total tensor memory per shape in MB (default 512)
#   TASKSET_MASK            taskset mask for big-core pinning (default f0)
#   ANDROID_NDK_HOME        same as temp_litert.sh

export ANDROID_NDK_HOME="${ANDROID_NDK_HOME:-$HOME/neo/android-ndk-r28b}"

set -euo pipefail

# ----------------------------------------------------------------------------
# 1. Configure
# ----------------------------------------------------------------------------
DEVICE_FOLDER="${DEVICE_FOLDER:-/data/local/tmp/litert_lm}"
HOST_MATMUL_ROSTER_CSV="${HOST_MATMUL_ROSTER_CSV:-./matmul_roster.csv}"
CL_BENCH_DTYPE="${CL_BENCH_DTYPE:-fp16}"
CL_BENCH_WARMUP="${CL_BENCH_WARMUP:-5}"
CL_BENCH_ITERS="${CL_BENCH_ITERS:-50}"
CL_BENCH_SHAPES="${CL_BENCH_SHAPES:-}"
CL_BENCH_MAX_SHAPES="${CL_BENCH_MAX_SHAPES:-0}"
CL_BENCH_MAX_TENSOR_MB="${CL_BENCH_MAX_TENSOR_MB:-512}"
TASKSET_MASK="${TASKSET_MASK:-f0}"

CL_BENCH_CSV_DEVICE="${DEVICE_FOLDER}/matmul_cl_bench.csv"
CL_BENCH_LOG="temp_litert_cl_bench.log"
HOST_CL_BENCH_CSV="./matmul_cl_bench.csv"
MATMUL_ROSTER_CSV_DEVICE="${DEVICE_FOLDER}/matmul_roster.csv"

# NDK auto-upgrade dance (same as temp_litert_matmul_micro.sh).
NDK_FALLBACK_R28B="${HOME}/neo/android-ndk-r28b"
case "${ANDROID_NDK_HOME}" in
  *android-ndk-r2[0-7]*)
    if [ -f "${NDK_FALLBACK_R28B}/source.properties" ]; then
      echo "[cl_bench.sh] WARNING: caller exported ANDROID_NDK_HOME=${ANDROID_NDK_HOME}"
      echo "[cl_bench.sh]          auto-upgrading to ${NDK_FALLBACK_R28B}"
      export ANDROID_NDK_HOME="${NDK_FALLBACK_R28B}"
    else
      echo "[cl_bench.sh] ANDROID_NDK_HOME points at NDK < r28: ${ANDROID_NDK_HOME}"
      echo "[cl_bench.sh] Install r28b first (./install_android_ndk.sh)."
      exit 1
    fi
    ;;
esac

if [ ! -d "${ANDROID_NDK_HOME}" ]; then
  echo "[cl_bench.sh] ANDROID_NDK_HOME does not exist: ${ANDROID_NDK_HOME}"
  exit 1
fi

BAZEL=$(command -v bazelisk || command -v bazel)
if [ -z "${BAZEL}" ]; then
  echo "[cl_bench.sh] Neither bazelisk nor bazel found in PATH."
  exit 1
fi

echo "[cl_bench.sh] === Direct OpenCL Matmul Benchmark ==="
echo "[cl_bench.sh] BAZEL=${BAZEL}"
echo "[cl_bench.sh] ANDROID_NDK_HOME=${ANDROID_NDK_HOME}"
echo "[cl_bench.sh] DEVICE_FOLDER=${DEVICE_FOLDER}"
echo "[cl_bench.sh] DTYPE=${CL_BENCH_DTYPE}"
echo "[cl_bench.sh] WARMUP=${CL_BENCH_WARMUP}"
echo "[cl_bench.sh] ITERS=${CL_BENCH_ITERS}"
echo "[cl_bench.sh] MAX_TENSOR_MB=${CL_BENCH_MAX_TENSOR_MB}"
if [ -n "${CL_BENCH_SHAPES}" ]; then
  echo "[cl_bench.sh] SHAPES (inline)=${CL_BENCH_SHAPES}"
else
  echo "[cl_bench.sh] HOST_MATMUL_ROSTER_CSV=${HOST_MATMUL_ROSTER_CSV}"
fi
if [ "${CL_BENCH_MAX_SHAPES}" -gt 0 ] 2>/dev/null; then
  echo "[cl_bench.sh] MAX_SHAPES=${CL_BENCH_MAX_SHAPES}"
fi
echo ""

# ----------------------------------------------------------------------------
# 2. Build matmul_cl_bench for android_arm64.
#    No LiteRT deps needed -- just a plain cc_binary with -ldl.
# ----------------------------------------------------------------------------
echo "[cl_bench.sh] Building //runtime/cl_bench:matmul_cl_bench ..."
"${BAZEL}" build \
  --config=android_arm64 \
  //runtime/cl_bench:matmul_cl_bench

CL_BENCH_BIN=bazel-bin/runtime/cl_bench/matmul_cl_bench
if [ ! -f "${CL_BENCH_BIN}" ]; then
  echo "[cl_bench.sh] Build succeeded but binary not at ${CL_BENCH_BIN}"
  exit 1
fi

# ----------------------------------------------------------------------------
# 3. Push binary to device.
#    No libLiteRt.so or accelerator shlibs needed -- this binary is
#    self-contained (only needs libOpenCL.so from the device vendor).
# ----------------------------------------------------------------------------
echo ""
echo "[cl_bench.sh] adb push binary ..."
adb shell "mkdir -p ${DEVICE_FOLDER}"
adb push "${CL_BENCH_BIN}" "${DEVICE_FOLDER}/matmul_cl_bench" >/dev/null
adb shell "chmod +x ${DEVICE_FOLDER}/matmul_cl_bench"

# ----------------------------------------------------------------------------
# 4. Resolve shape source (same logic as temp_litert_matmul_micro.sh).
# ----------------------------------------------------------------------------
SHAPES_FLAG=""
if [ -n "${CL_BENCH_SHAPES}" ]; then
  SHAPES_FLAG="--shapes=${CL_BENCH_SHAPES}"
  echo ""
  echo "[cl_bench.sh] using inline shapes: ${CL_BENCH_SHAPES}"
else
  if [ -f "${HOST_MATMUL_ROSTER_CSV}" ]; then
    echo ""
    echo "[cl_bench.sh] Pushing ${HOST_MATMUL_ROSTER_CSV} -> ${MATMUL_ROSTER_CSV_DEVICE} ..."
    adb push "${HOST_MATMUL_ROSTER_CSV}" "${MATMUL_ROSTER_CSV_DEVICE}" >/dev/null
  elif adb shell "test -f ${MATMUL_ROSTER_CSV_DEVICE}"; then
    echo ""
    echo "[cl_bench.sh] host ${HOST_MATMUL_ROSTER_CSV} not found; using device cached"
    echo "[cl_bench.sh]   ${MATMUL_ROSTER_CSV_DEVICE}"
  else
    echo ""
    echo "[cl_bench.sh] ERROR: no matmul roster CSV found."
    echo "    Looked in:"
    echo "        host:   ${HOST_MATMUL_ROSTER_CSV}"
    echo "        device: ${MATMUL_ROSTER_CSV_DEVICE}"
    echo "    Run ./temp_litert.sh first to produce one, or pass shapes inline:"
    echo "        CL_BENCH_SHAPES='1024x6144x1536,1024x1536x6144' \\"
    echo "          ./temp_litert_cl_bench.sh"
    exit 1
  fi
  SHAPES_FLAG="--shapes_csv=${MATMUL_ROSTER_CSV_DEVICE}"
fi

MAX_SHAPES_FLAG=""
if [ "${CL_BENCH_MAX_SHAPES}" -gt 0 ] 2>/dev/null; then
  MAX_SHAPES_FLAG="--max_shapes=${CL_BENCH_MAX_SHAPES}"
fi

# ----------------------------------------------------------------------------
# 5. Run on device.
# ----------------------------------------------------------------------------
echo ""
echo "[cl_bench.sh] Running matmul_cl_bench on device (taskset ${TASKSET_MASK}) ..."
adb shell "cd ${DEVICE_FOLDER}; \
  LD_LIBRARY_PATH=/system/vendor/lib64:. \
  taskset ${TASKSET_MASK} ./matmul_cl_bench \
    --dtype=${CL_BENCH_DTYPE} \
    ${SHAPES_FLAG} \
    --warmup=${CL_BENCH_WARMUP} \
    --iters=${CL_BENCH_ITERS} \
    --max_tensor_mb=${CL_BENCH_MAX_TENSOR_MB} \
    ${MAX_SHAPES_FLAG} \
    --csv_out=${CL_BENCH_CSV_DEVICE}" \
  2>&1 | tee "${CL_BENCH_LOG}"

# ----------------------------------------------------------------------------
# 6. Pull results.
# ----------------------------------------------------------------------------
if adb shell "test -f ${CL_BENCH_CSV_DEVICE}"; then
  adb pull "${CL_BENCH_CSV_DEVICE}" "${HOST_CL_BENCH_CSV}" >/dev/null 2>&1
  echo ""
  echo "=========================================="
  echo " Direct OpenCL Matmul Bench CSV"
  echo "=========================================="
  echo "    Compare tflops_wall and tflops_gpu against prefill's"
  echo "    Delegate Statistics. tflops_gpu should be close to"
  echo "    prefill's kernel-only TFLOPS (>5 for large shapes)."
  echo "    tflops_wall shows what we get without LiteRT overhead."
  echo ""
  head -n 30 "${HOST_CL_BENCH_CSV}" || true
  echo ""
  echo "Full log:    ${CL_BENCH_LOG}"
  echo "Full CSV:    ${HOST_CL_BENCH_CSV}"
else
  echo ""
  echo "[cl_bench.sh] WARNING: ${CL_BENCH_CSV_DEVICE} not produced. Check ${CL_BENCH_LOG}."
  exit 1
fi
echo ""
