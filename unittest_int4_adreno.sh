#!/usr/bin/env bash
#
# unittest_int4_adreno.sh - Build and run the gemm_int4_adreno_cl unit
#                           test on a connected Android device.
#
# Layered on the existing unittest.sh / tools/android_test.sh workflow
# but scoped to a single binary:
#   test/unittest/unittest_opencl_kernels_int4_adreno.cpp
#     -> test binary `unittest_opencl_kernels_int4_adreno`
#
# Usage:
#   sh unittest_int4_adreno.sh                # build + push + run everything
#   sh unittest_int4_adreno.sh --build-only   # build + push, skip execution
#   sh unittest_int4_adreno.sh --run-only     # skip build, rerun on device
#   sh unittest_int4_adreno.sh --filter "*1024*"
#                                             # narrow to a gtest filter
#
# Env overrides:
#   ADB_IP      -- if set, uses `adb -H $ADB_IP` (network adb)
#   DEVICE_DIR  -- where to drop the binary + .so on device.
#                  default /data/local/tmp/nntr_android_test
#
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

FILTER=""
BUILD_ONLY=0
RUN_ONLY=0
DEVICE_DIR="${DEVICE_DIR:-/data/local/tmp/nntr_android_test}"
TEST_BIN_NAME="unittest_opencl_kernels_int4_adreno"
RUN_LOG="${SCRIPT_DIR}/unittest_int4_adreno_run.log"

# --- arg parsing -------------------------------------------------------------
while [ $# -gt 0 ]; do
  case "$1" in
    --build-only) BUILD_ONLY=1; shift ;;
    --run-only)   RUN_ONLY=1;   shift ;;
    --filter)     shift; FILTER="$1"; shift ;;
    --filter=*)   FILTER="${1#--filter=}"; shift ;;
    -h|--help)
      sed -n '2,24p' "$0"
      exit 0
      ;;
    *)
      echo "unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

if [ -n "$ADB_IP" ]; then
  ADB_CMD="adb -H ${ADB_IP}"
else
  ADB_CMD="adb"
fi

# =============================================================================
# Step 1: Build libnntrainer.so with OpenCL (via package_android.sh)
#
# Skipped in --run-only mode. The packager leaves libnntrainer.so /
# libccapi-nntrainer.so / libOpenCL.so under
# builddir/android_build_result/lib/arm64-v8a, which is where the test
# binary's ndk-build step expects to find them.
# =============================================================================
if [ $RUN_ONLY -eq 0 ]; then
  echo "=========================================="
  echo "[Step 1/4] Building nntrainer (OpenCL)"
  echo "=========================================="

  # Clean stale builddir (matches temp.sh convention). Without this,
  # a half-finished prior build leaves behind a builddir/ with missing
  # meson-private/build.dat and package_android.sh's reconfigure fails.
  rm -rf builddir

  ./tools/package_android.sh \
    -Denable-opencl=true \
    -Denable-fp16=true \
    -Domp-num-threads=1

  # ==========================================================================
  # Step 2: Build the int4_adreno test binary via ndk-build
  #
  # The Android.mk module name must match LOCAL_MODULE in test/jni/Android.mk.
  # `MESON_ENABLE_OPENCL=1` is forwarded into the makefile so it picks
  # up the opencl / clblast LOCAL_SHARED/STATIC_LIBRARIES additions.
  # ==========================================================================
  echo ""
  echo "=========================================="
  echo "[Step 2/4] Building ${TEST_BIN_NAME} (ndk-build)"
  echo "=========================================="

  pushd test/jni >/dev/null
  ndk-build -j$(nproc) MESON_ENABLE_OPENCL=1 ${TEST_BIN_NAME}
  if [ $? -ne 0 ]; then
    echo "[ERROR] ndk-build for ${TEST_BIN_NAME} failed"
    popd >/dev/null
    exit 1
  fi
  popd >/dev/null

  # ==========================================================================
  # Step 3: Push the binary + required shared libraries to the device.
  #
  # ndk-build drops the executable under test/libs/arm64-v8a/ (when it's
  # configured with the default APP_ABI=arm64-v8a) or test/obj/local/arm64-v8a/
  # on some toolchain setups. The shared libraries live next to the test
  # binary under the same path the main unittest.sh uses.
  # ==========================================================================
  echo ""
  echo "=========================================="
  echo "[Step 3/4] Pushing to device"
  echo "=========================================="

  $ADB_CMD shell mkdir -p ${DEVICE_DIR}

  if [ -f test/libs/arm64-v8a/${TEST_BIN_NAME} ]; then
    UNITTEST_BIN=test/libs/arm64-v8a/${TEST_BIN_NAME}
  elif [ -f test/obj/local/arm64-v8a/${TEST_BIN_NAME} ]; then
    UNITTEST_BIN=test/obj/local/arm64-v8a/${TEST_BIN_NAME}
  else
    echo "[ERROR] ${TEST_BIN_NAME} binary not found under test/libs/ or test/obj/"
    exit 1
  fi

  echo "Pushing ${UNITTEST_BIN} -> ${DEVICE_DIR}/"
  $ADB_CMD push "${UNITTEST_BIN}" "${DEVICE_DIR}/"

  # Push the matching nntrainer shared libs. Different build layouts put
  # them in slightly different folders, so we try both the jni/
  # ndk-build output tree and the meson android_build_result tree.
  for so in libnntrainer.so libccapi-nntrainer.so libOpenCL.so libc++_shared.so; do
    if [ -f builddir/jni/arm64-v8a/${so} ]; then
      $ADB_CMD push "builddir/jni/arm64-v8a/${so}" "${DEVICE_DIR}/"
    elif [ -f builddir/android_build_result/lib/arm64-v8a/${so} ]; then
      $ADB_CMD push "builddir/android_build_result/lib/arm64-v8a/${so}" "${DEVICE_DIR}/"
    fi
  done

  echo "[OK] Push complete"
fi

if [ $BUILD_ONLY -eq 1 ]; then
  echo ""
  echo "[Done] Build-only mode. Skipping execution."
  exit 0
fi

# =============================================================================
# Step 4: Run on device.
#
# The test binary prints one "int4_gemm_adreno M=... K=... N=... gpu=...
# ms MSE=... (tol=...)" line per shape before the gtest OK/FAIL summary.
# Both stdout and stderr are tee'd into unittest_int4_adreno_run.log so
# failing shapes can be grepped out after the fact.
# =============================================================================
echo ""
echo "=========================================="
echo "[Step 4/4] Running tests on device"
echo "=========================================="

$ADB_CMD shell chmod +x "${DEVICE_DIR}/${TEST_BIN_NAME}"

if [ -n "$FILTER" ]; then
  GTEST_FILTER_ARG="--gtest_filter=${FILTER}"
else
  # Default: run every adreno int4 / int8 / A/B suite. `_kernels_*` covers:
  #   nntrainer_opencl_adreno_kernels_int4          (fp16 activation path)
  #   nntrainer_opencl_adreno_kernels_int8_int4     (DP4A path)
  #   nntrainer_opencl_adreno_kernels_ab            (A/B benchmark)
  GTEST_FILTER_ARG="--gtest_filter=nntrainer_opencl_adreno_kernels_*.*"
fi

echo "Filter: ${GTEST_FILTER_ARG}"
echo ""

$ADB_CMD shell "cd ${DEVICE_DIR} && export LD_LIBRARY_PATH=. && ./${TEST_BIN_NAME} ${GTEST_FILTER_ARG}" \
  2>&1 | tee "${RUN_LOG}"

echo ""
echo "=========================================="
echo " Run summary"
echo "=========================================="
echo ""
echo "--- Per-shape latency / MSE ---"
grep -E "int4_gemm_adreno M=|int8_int4_gemm_adreno M=" "${RUN_LOG}" \
  || echo "(no per-shape lines found)"

echo ""
echo "--- A/B benchmark (fp16 vs DP4A) ---"
grep -E "A/B bench|fp16 path|dp4a path|speedup|dp4a MSE" "${RUN_LOG}" \
  || echo "(no A/B bench lines found)"

echo ""
echo "--- gtest pass/fail tally ---"
grep -E "\[  PASSED  \]|\[  FAILED  \]|\[ RUN      \]|\[       OK \]" "${RUN_LOG}" \
  | tail -20 || echo "(no gtest lines found)"

echo ""
echo "Full log: ${RUN_LOG}"
echo ""
