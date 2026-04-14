set -e

rm -rf builddir

# Build libnntrainer.so with OpenCL
# [Test 2] Single-thread to isolate possible parallel weight-load race condition
./tools/package_android.sh -Dmmap-read=false -Domp-num-threads=1 -Dthread-backend=omp -Denable-opencl=true

# Build CausalLM app
cd Applications/CausalLM
sh build_android.sh

adb shell "mkdir -p /data/local/tmp/nntrainer/test"
adb shell "mkdir -p /data/local/tmp/nntrainer/causallm/models/qwen3-4b"
adb push jni/libs/arm64-v8a/* /data/local/tmp/nntrainer/test

# Push freshly-built nntrainer shared libs that build_android.sh leaves in
# builddir/android_build_result/lib/arm64-v8a/ but does NOT copy into
# Applications/CausalLM/jni/libs/arm64-v8a/. Without these, libcausallm_core.so
# on device may try to resolve symbols against a stale libnntrainer.so.
NNTRAINER_LIB_DIR=../../builddir/android_build_result/lib/arm64-v8a
for lib in libnntrainer.so libccapi-nntrainer.so libOpenCL.so libc++_shared.so; do
  if [ -f "$NNTRAINER_LIB_DIR/$lib" ]; then
    adb push "$NNTRAINER_LIB_DIR/$lib" /data/local/tmp/nntrainer/test/
  fi
done

# adb push /home/jwhero94/nntr_qwen3-4b-q6_K-qint4-idx3-fp32-arm/* /data/local/tmp/nntrainer/causallm/models/qwen3-4b

adb shell chmod +x /data/local/tmp/nntrainer/test/nntrainer_causallm

# ----------------------------------------------------------------------------
# Patch on-device nntr_config.json so that:
#   - tensor_pool's KV-cache fits under the Adreno 830 1 GB SVM per-allocation
#     limit (init_seq_len <= 1024, max_seq_len <= 2048 for Qwen3-4B)
#   - generation finishes quickly during dev iteration (num_to_generate = 8)
#
# We always re-derive from .bak so re-runs are idempotent. Originally the
# device config has init_seq_len=10240, max_seq_len=20480, num_to_generate=128
# which is what the user wants in production.
# ----------------------------------------------------------------------------
CFG=/data/local/tmp/nntrainer/causallm/models/qwen3-4b/nntr_config.json
INIT_SEQ_LEN_NEW=1024
MAX_SEQ_LEN_NEW=2048
NUM_TO_GENERATE_NEW=8

adb shell "[ -f ${CFG}.bak ] || cp ${CFG} ${CFG}.bak"
adb shell "cp ${CFG}.bak ${CFG}"
adb shell "sed -i \
  -e 's/\"init_seq_len\"[[:space:]]*:[[:space:]]*[0-9]*/\"init_seq_len\": ${INIT_SEQ_LEN_NEW}/' \
  -e 's/\"max_seq_len\"[[:space:]]*:[[:space:]]*[0-9]*/\"max_seq_len\": ${MAX_SEQ_LEN_NEW}/' \
  -e 's/\"num_to_generate\"[[:space:]]*:[[:space:]]*[0-9]*/\"num_to_generate\": ${NUM_TO_GENERATE_NEW}/' \
  ${CFG}"
echo "[temp.sh] patched on-device nntr_config.json:"
adb shell "grep -E '\"init_seq_len\"|\"max_seq_len\"|\"num_to_generate\"' ${CFG}"

# Capture stdout AND stderr so the [DIAG ...] traces from our diagnostics land
# in error.txt next to the prefill / generation TPS.
adb shell "cd /data/local/tmp/nntrainer/test; export LD_LIBRARY_PATH=.; ./nntrainer_causallm /data/local/tmp/nntrainer/causallm/models/qwen3-4b" 2>&1 \
  | tee ../../error.txt

cd ../..

adb pull /data/local/tmp/nntrainer/test/logs/. ./logs/ || true
adb shell "rm /data/local/tmp/nntrainer/test/logs/* 2>/dev/null || true"

# ----------------------------------------------------------------------------
# Diagnostic summary (extracted from error.txt for quick scanning)
# ----------------------------------------------------------------------------
echo ""
echo "=========================================="
echo " Diagnostic summary"
echo "=========================================="

echo ""
echo "--- MemoryPool::allocate traces ---"
grep -E "DIAG MemoryPool::allocate" error.txt || echo "(no MemoryPool DIAG lines found)"

echo ""
echo "--- dotQInteger / dotBatched-QINT4 traces (first 8) ---"
grep -E "DIAG dotQInteger|DIAG dotBatched-QINT4" error.txt || echo "(no dot DIAG lines found)"

echo ""
echo "--- attach_kai_buffer count ---"
KAI_COUNT=$(grep -c "DIAG attach_kai_buffer" error.txt || true)
echo "attach_kai_buffer invocations: ${KAI_COUNT}"

echo ""
echo "--- Generation snippet (post-assistant tag) ---"
awk '/<\|im_start\|>assistant/ { found=1; next } found { print }' error.txt | head -c 600
echo ""

echo ""
echo "--- Perf summary ---"
grep -E "prefill:|generation:|total:|peak memory|e2e time" error.txt || echo "(no perf lines)"
echo ""
echo "Full log: error.txt"
echo ""
echo "To restore the original (long-context) config on device:"
echo "  adb shell 'mv ${CFG}.bak ${CFG}'"
