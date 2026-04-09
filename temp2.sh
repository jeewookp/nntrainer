set -e

rm -rf builddir

# Build libnntrainer.so WITHOUT OpenCL (CPU only)
./tools/package_android.sh -Dmmap-read=false -Domp-num-threads=4 -Dthread-backend=omp

# Build CausalLM app
cd Applications/CausalLM
sh build_android.sh

adb shell "mkdir -p /data/local/tmp/nntrainer/test"
adb shell "mkdir -p /data/local/tmp/nntrainer/causallm/models/qwen3-4b"
adb push jni/libs/arm64-v8a/* /data/local/tmp/nntrainer/test

# adb push /home/jwhero94/nntr_qwen3-4b-q6_K-qint4-idx3-fp32-arm/* /data/local/tmp/nntrainer/causallm/models/qwen3-4b

adb shell chmod +x /data/local/tmp/nntrainer/test/nntrainer_causallm
adb shell "cd /data/local/tmp/nntrainer/test; export LD_LIBRARY_PATH=.; ./nntrainer_causallm /data/local/tmp/nntrainer/causallm/models/qwen3-4b"
adb pull /data/local/tmp/nntrainer/test/logs/. ./logs/
adb shell "rm /data/local/tmp/nntrainer/test/logs/*"
