#!/usr/bin/env bash
# Re-exec under bash when invoked via `sh` (dash): this script uses
# pushd/popd, set -e, and other bash-isms that dash chokes on.
# Without the guard `sh run_gemv_compare.sh` dies on the first
# pushd with "pushd: not found".
if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi
# run_gemv_compare.sh
#
# Build, push, and run the GPU-vs-CPU M=1 gemv unittest on the
# attached Android device.  Mirrors the build flow temp.sh uses for
# the full causallm app, but compiles only the unit tests via
# ndk-build under test/jni.
#
# Output: per-shape lines like
#   [gemv_compare] K=2560 N=4096  GPU=0.61 ms  CPU=0.20 ms  ratio(GPU/CPU)=3.05x
# for each Qwen3-4B FC dim, isolating the per-call cost on each
# path so we can chase the ~5x decode CPU/GPU gap one shape at a
# time.
#
# Requires:
#   * Android NDK on PATH (or via ndk-build wrapper used by the
#     existing test build, same as run_test.sh)
#   * adb connected to the target device
#   * nntrainer already built once via `tools/package_android.sh`
#     (the unit test links libnntrainer.so / libccapi-nntrainer.so
#     produced there).

set -e

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
DEVICE_DIR=/data/local/tmp/nntrainer/test
TEST_BIN=unittest_opencl_kernels_gemv_compare

cd "$REPO_ROOT"

# 1. Always rebuild nntrainer shared libs.  meson configure_file bakes
#    each .cl source into a C++ string symbol inside libnntrainer.so;
#    ndk-build for the unittest does NOT rebuild libnntrainer.so, so
#    any kernel source / dispatch-param edit (e.g. switching WG=16
#    to WG=64) is silently ignored until we re-run package_android.sh.
echo "[gemv_compare.sh] (re)building libnntrainer.so via package_android.sh"
./tools/package_android.sh -Dmmap-read=false -Domp-num-threads=1 \
                           -Dthread-backend=omp -Denable-opencl=true

# 2. Build the unit test via the test/jni Android.mk.  ndk-build
#    drops the executable in test/libs/arm64-v8a/.
echo "[gemv_compare.sh] Building $TEST_BIN ..."
pushd test/jni > /dev/null
ndk-build NDK_PROJECT_PATH=. APP_BUILD_SCRIPT=Android.mk \
          APP_PLATFORM=android-29 APP_ABI=arm64-v8a \
          APP_STL=c++_shared MESON_ENABLE_OPENCL=1 \
          $TEST_BIN -j$(nproc)
popd > /dev/null

TEST_EXE=test/libs/arm64-v8a/$TEST_BIN
[ -x "$TEST_EXE" ] || [ -f "$TEST_EXE" ] || {
  echo "[gemv_compare.sh] ERROR: $TEST_EXE not produced by ndk-build"
  exit 1
}

# 3. Push the executable + the nntrainer / ccapi / c++ shared libs
#    that it depends on.  The existing causallm runs already drop
#    these in $DEVICE_DIR; push fresh ones here so a partial /
#    stale state on device doesn't shadow our build.
adb shell "mkdir -p $DEVICE_DIR"
adb push "$TEST_EXE" "$DEVICE_DIR/" >/dev/null

NNTRAINER_LIB_DIR=builddir/android_build_result/lib/arm64-v8a
for lib in libnntrainer.so libccapi-nntrainer.so libc++_shared.so; do
  if [ -f "$NNTRAINER_LIB_DIR/$lib" ]; then
    adb push "$NNTRAINER_LIB_DIR/$lib" "$DEVICE_DIR/" >/dev/null
  fi
done
adb shell "chmod +x $DEVICE_DIR/$TEST_BIN"

# 4. Run.  taskset f0 pins to the big cluster (Snapdragon 8 Elite
#    Cortex-X / X+ cores); LD_LIBRARY_PATH=. picks up the libs we
#    just pushed alongside the binary; the gtest filter narrows to
#    just the gemv_compare suite (other suites in this binary or
#    in the same test set don't apply here).
echo ""
echo "[gemv_compare.sh] Running on device ..."
adb shell "cd $DEVICE_DIR; \
  export LD_LIBRARY_PATH=.; \
  taskset f0 ./$TEST_BIN \
    --gtest_filter='nntrainer_gemv_compare.*' \
    --gtest_color=no" \
  2>&1 | tee gemv_compare_run.log

echo ""
echo "[gemv_compare.sh] Full log: $REPO_ROOT/gemv_compare_run.log"
echo "[gemv_compare.sh] Summary lines:"
grep "\[gemv_compare\]" gemv_compare_run.log || echo "  (no [gemv_compare] lines -- run failed?)"
