#!/usr/bin/env bash
# run_gemv_compare.sh
#
# Build, push, and run the GPU-vs-CPU M=1 gemv unittest on the
# attached Android device.  Mirrors the build flow temp.sh uses for
# the full causallm app, but compiles only the unit tests via
# ndk-build under test/jni.

# Re-exec under bash when invoked via `sh` (dash): this script uses
# pushd/popd, set -e, and other bash-isms that dash chokes on.
# Without the guard `sh run_gemv_compare.sh` dies on the first
# pushd with "pushd: not found".
if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi
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

# 1. Ensure nntrainer shared libs exist (rebuild if missing).
if [ ! -f builddir/android_build_result/lib/arm64-v8a/libnntrainer.so ]; then
  echo "[gemv_compare.sh] libnntrainer.so missing -- running package_android.sh first"
  ./tools/package_android.sh -Dmmap-read=false -Domp-num-threads=1 \
                             -Dthread-backend=omp -Denable-opencl=true
fi

# 2. Build the unit test via the test/jni Android.mk.  ndk-build
#    drops the executable in test/libs/arm64-v8a/.
echo "[gemv_compare.sh] Building $TEST_BIN ..."
pushd test/jni > /dev/null
ndk-build NDK_PROJECT_PATH=. APP_BUILD_SCRIPT=Android.mk \
          APP_PLATFORM=android-29 APP_ABI=arm64-v8a \
          APP_STL=c++_shared MESON_ENABLE_OPENCL=1 \
          $TEST_BIN -j$(nproc)
popd > /dev/null

# ndk-build with NDK_PROJECT_PATH=. from test/jni produces the
# executable under test/jni/libs/<abi>/.  The previous version
# of this script looked under test/libs/ (off by one) and
# spuriously bailed even after a successful build.  Search both
# locations to be tolerant of either layout.
TEST_EXE=""
for cand in \
    test/jni/obj/local/arm64-v8a/$TEST_BIN \
    test/jni/libs/arm64-v8a/$TEST_BIN \
    test/libs/arm64-v8a/$TEST_BIN; do
  if [ -f "$cand" ]; then
    TEST_EXE="$cand"
    break
  fi
done
if [ -z "$TEST_EXE" ]; then
  echo "[gemv_compare.sh] ERROR: $TEST_BIN not found under any candidate path"
  echo "  ndk-build output above said it was produced -- where did it land?"
  find test -name "$TEST_BIN" -type f 2>/dev/null | head
  exit 1
fi
echo "[gemv_compare.sh] Test exe: $TEST_EXE"

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
