# set -e
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/neo/android-ndk-r26d/
# export PATH=$PATH:~/neo/android-ndk-r26d/
# export ANDROID_NDK=~/neo/android-ndk-r26d/

rm -rf builddir
./tools/package_android.sh -Denable-opencl=true
# ./tools/package_android.sh

cp -r $ANDROID_NDK/sources/third_party/googletest test/jni

ninja install -C builddir
mkdir -p libs/arm64-v8a
cp builddir/android_build_result/lib/arm64-v8a/*.so libs/arm64-v8a

ndk-build -C test/jni -j$(nproc) MESON_ENABLE_OPENCL=1 unittest_kai_q40_comparison_test NDK_DEBUG=0
# ndk-build -C test/jni -j$(nproc) unittest_kai_q40_comparison_test NDK_DEBUG=1

adb shell "mkdir -p /data/local/tmp/nntrainer/test"
adb push test/obj/local/arm64-v8a/unittest_kai_q40_comparison_test /data/local/tmp/nntrainer/test
adb push libs/arm64-v8a/*.so /data/local/tmp/nntrainer/test
adb shell chmod +x /data/local/tmp/nntrainer/test/unittest_kai_q40_comparison_test
adb shell "cd /data/local/tmp/nntrainer/test; export LD_LIBRARY_PATH=.; ./unittest_kai_q40_comparison_test $@"
adb pull /data/local/tmp/nntrainer/test/logs/. ./logs/
adb shell "rm /data/local/tmp/nntrainer/test/logs/*"

