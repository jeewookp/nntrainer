set -e

rm -rf builddir

# Step 1: Build libnntrainer.so WITHOUT OpenCL first (for clean headers)
./tools/package_android.sh -Dmmap-read=false -Domp-num-threads=4 -Dthread-backend=omp

# Step 2: Build CausalLM app (uses non-OpenCL headers)
cd Applications/CausalLM
sh build_android.sh
cd ../..

# Step 3: Rebuild libnntrainer.so WITH OpenCL (replace the .so only)
rm -rf builddir
./tools/package_android.sh -Dmmap-read=false -Domp-num-threads=4 -Dthread-backend=omp -Denable-opencl=true

# Step 4: Copy OpenCL-enabled .so to app libs
cp builddir/android_build_result/lib/arm64-v8a/libnntrainer.so Applications/CausalLM/jni/libs/arm64-v8a/

adb shell "mkdir -p /data/local/tmp/nntrainer/test"
adb shell "mkdir -p /data/local/tmp/nntrainer/causallm/models/qwen3-4b"
adb push Applications/CausalLM/jni/libs/arm64-v8a/* /data/local/tmp/nntrainer/test

# adb push /home/jwhero94/nntr_qwen3-4b-q6_K-qint4-idx3-fp32-arm/* /data/local/tmp/nntrainer/causallm/models/qwen3-4b

adb shell chmod +x /data/local/tmp/nntrainer/test/nntrainer_causallm
adb shell "cd /data/local/tmp/nntrainer/test; export LD_LIBRARY_PATH=.; ./nntrainer_causallm /data/local/tmp/nntrainer/causallm/models/qwen3-4b"
adb pull /data/local/tmp/nntrainer/test/logs/. ./logs/
adb shell "rm /data/local/tmp/nntrainer/test/logs/*"
