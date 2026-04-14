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
# Patch on-device nntr_config.json so that tensor_pool's KV-cache fits
# under the Adreno 830 1 GB SVM per-allocation limit (init_seq_len <=
# 1024, max_seq_len <= 2048 for Qwen3-4B). num_to_generate is left at
# the production value (128) so generation TPS is measured over a full
# decode run.
#
# We always re-derive from .bak so re-runs are idempotent. Originally the
# device config has init_seq_len=10240, max_seq_len=20480, num_to_generate=128
# which is what the user wants in production.
# ----------------------------------------------------------------------------
CFG=/data/local/tmp/nntrainer/causallm/models/qwen3-4b/nntr_config.json
INIT_SEQ_LEN_NEW=1024
MAX_SEQ_LEN_NEW=2048
NUM_TO_GENERATE_NEW=128

adb shell "[ -f ${CFG}.bak ] || cp ${CFG} ${CFG}.bak"
adb shell "cp ${CFG}.bak ${CFG}"
adb shell "sed -i \
  -e 's/\"init_seq_len\"[[:space:]]*:[[:space:]]*[0-9]*/\"init_seq_len\": ${INIT_SEQ_LEN_NEW}/' \
  -e 's/\"max_seq_len\"[[:space:]]*:[[:space:]]*[0-9]*/\"max_seq_len\": ${MAX_SEQ_LEN_NEW}/' \
  -e 's/\"num_to_generate\"[[:space:]]*:[[:space:]]*[0-9]*/\"num_to_generate\": ${NUM_TO_GENERATE_NEW}/' \
  ${CFG}"
echo "[temp.sh] patched on-device nntr_config.json:"
adb shell "grep -E '\"init_seq_len\"|\"max_seq_len\"|\"num_to_generate\"' ${CFG}"

# Capture stdout AND stderr so the [DIAG ...] traces from our diagnostics
# land alongside the prefill / generation TPS.
#
# IMPORTANT: write to temp_run.log (NOT error.txt). error.txt is owned by
# the user as the canonical "this is what the failing run looked like"
# file that gets pushed to the repo for me to inspect; this script must
# not clobber it.
RUN_LOG=../../temp_run.log
adb shell "cd /data/local/tmp/nntrainer/test; export LD_LIBRARY_PATH=.; ./nntrainer_causallm /data/local/tmp/nntrainer/causallm/models/qwen3-4b" 2>&1 \
  | tee ${RUN_LOG}

cd ../..

adb pull /data/local/tmp/nntrainer/test/logs/. ./logs/ || true
adb shell "rm /data/local/tmp/nntrainer/test/logs/* 2>/dev/null || true"

# ----------------------------------------------------------------------------
# Diagnostic summary (extracted from temp_run.log for quick scanning)
# ----------------------------------------------------------------------------
RUN_LOG=temp_run.log

echo ""
echo "=========================================="
echo " Run summary"
echo "=========================================="

echo ""
echo "--- Generation snippet (post-assistant tag) ---"
awk '/<\|im_start\|>assistant/ { found=1; next } found { print }' ${RUN_LOG} | head -c 600
echo ""

echo ""
echo "--- Perf summary ---"
grep -E "prefill:|generation:|total:|peak memory|e2e time" ${RUN_LOG} || echo "(no perf lines)"

echo ""
echo "--- dotQInteger profile breakdown ---"
grep -A 4 "PROFILE FloatTensor::dotQInteger" ${RUN_LOG} || echo "(no PROFILE lines found)"

echo ""
echo "Full log: ${RUN_LOG}  (error.txt is intentionally untouched)"
echo ""
echo "To restore the original (long-context) config on device:"
echo "  adb shell 'mv ${CFG}.bak ${CFG}'"
