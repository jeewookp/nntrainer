# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/neo/android-ndk-r26d/
# export PATH=$PATH:~/neo/android-ndk-r26d/
# export ANDROID_NDK=~/neo/android-ndk-r26d/

# ./tools/package_android.sh -Denable-opencl=true

ninja install -C builddir
cp builddir/android_build_result/lib/arm64-v8a/*.so libs/arm64-v8a
ndk-build -C Applications/temp/jni -j$(nproc)
adb push Applications/temp/libs/arm64-v8a/nntrainer_logistic /data/local/tmp/nntrainer/test
adb shell "mkdir -p /data/local/tmp/nntrainer/test"
adb push libs/arm64-v8a/*.so /data/local/tmp/nntrainer/test
adb shell chmod +x /data/local/tmp/nntrainer/test/nntrainer_logistic
adb shell "cd /data/local/tmp/nntrainer/test; export LD_LIBRARY_PATH=.; ./nntrainer_logistic $@"
adb pull /data/local/tmp/nntrainer/test/logs/. ./logs/
adb shell "rm /data/local/tmp/nntrainer/test/logs/*"