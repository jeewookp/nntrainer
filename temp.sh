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
# Push only the nntrainer-built shared libs we actually need. Skip
# libOpenCL.so so the device's vendor /vendor/lib64/libOpenCL.so is
# loaded directly.
for lib in libnntrainer.so libccapi-nntrainer.so libc++_shared.so; do
  if [ -f "$NNTRAINER_LIB_DIR/$lib" ]; then
    adb push "$NNTRAINER_LIB_DIR/$lib" /data/local/tmp/nntrainer/test/
  fi
done
# Remove any previously-pushed bundled libOpenCL.so so `.` doesn't
# still shadow the vendor one via prior-run residue.
adb shell "rm -f /data/local/tmp/nntrainer/test/libOpenCL.so" || true

# adb push /home/jwhero94/nntr_qwen3-4b-q6_K-qint4-idx3-fp32-arm/* /data/local/tmp/nntrainer/causallm/models/qwen3-4b

adb shell chmod +x /data/local/tmp/nntrainer/test/nntrainer_causallm

# ----------------------------------------------------------------------------
# Patch on-device nntr_config.json so that:
#   - tensor_pool's KV-cache fits under the Adreno 830 1 GB SVM per-
#     allocation limit (init_seq_len <= 1024, max_seq_len <= 2048 for
#     Qwen3-4B)
#   - num_to_generate is small enough that a dev iteration finishes in
#     a few seconds
#   - activation dtype is FP16 (model_tensor_type QINT4-FP16) so the
#     custom layers (Phase 1 + Phase 2 work) skip the CPU fp32<->fp16
#     conversion loops that dominated the prior profile.
#
# We always re-derive from .bak so re-runs are idempotent. Originally the
# device config has init_seq_len=10240, max_seq_len=20480,
# num_to_generate=128, model_tensor_type="QINT4-FP32" which is the
# production-latency-unconstrained config.
# ----------------------------------------------------------------------------
CFG=/data/local/tmp/nntrainer/causallm/models/qwen3-4b/nntr_config.json
INIT_SEQ_LEN_NEW=1024
MAX_SEQ_LEN_NEW=2048
NUM_TO_GENERATE_NEW=32
# Back on QINT4-FP16 to exercise the HalfTensor dot path. Phase 4 DIAG
# assertions in half_tensor.cpp will throw if any layer's output tensor
# turns out to be FP32 instead of FP16 (silent reinterpret_cast corruption
# hypothesis).
MODEL_TENSOR_TYPE_NEW=QINT4-FP16

adb shell "[ -f ${CFG}.bak ] || cp ${CFG} ${CFG}.bak"
adb shell "cp ${CFG}.bak ${CFG}"
adb shell "sed -i \
  -e 's/\"init_seq_len\"[[:space:]]*:[[:space:]]*[0-9]*/\"init_seq_len\": ${INIT_SEQ_LEN_NEW}/' \
  -e 's/\"max_seq_len\"[[:space:]]*:[[:space:]]*[0-9]*/\"max_seq_len\": ${MAX_SEQ_LEN_NEW}/' \
  -e 's/\"num_to_generate\"[[:space:]]*:[[:space:]]*[0-9]*/\"num_to_generate\": ${NUM_TO_GENERATE_NEW}/' \
  -e 's/\"model_tensor_type\"[[:space:]]*:[[:space:]]*\"[A-Za-z0-9_-]*\"/\"model_tensor_type\": \"${MODEL_TENSOR_TYPE_NEW}\"/' \
  ${CFG}"
echo "[temp.sh] patched on-device nntr_config.json:"
adb shell "grep -E '\"init_seq_len\"|\"max_seq_len\"|\"num_to_generate\"|\"model_tensor_type\"' ${CFG}"

# Capture stdout AND stderr so the [DIAG ...] traces from our diagnostics
# land alongside the prefill / generation TPS.
#
# IMPORTANT: write to temp_run.log (NOT error.txt). error.txt is owned by
# the user as the canonical "this is what the failing run looked like"
# file that gets pushed to the repo for me to inspect; this script must
# not clobber it.
RUN_LOG=../../temp_run.log
# NNTR_DELEGATE_FP16=1 enables the delegate conv_wave_memory kernel path
# (dequant int4→fp16 + wave memory dispatch). Unset to use default int4 path.
DELEGATE_ENV="${NNTR_DELEGATE_FP16:+NNTR_DELEGATE_FP16=1}"
# NNTRAINER_PROFILE_LAYER_SYNC=1 clFinishes after every layer and makes
# delegate conv clWaitForEvents post-dispatch, so every per-layer
# profile is honest wall-clock instead of the GPU pipeline tail
# leaking into whichever layer next SVMMap-fences. Production runs
# should unset it — it doubles layer wall time for zero end-user
# benefit.
# NNTR_DELEGATE_CONV_VERIFY=X,Y,Z triggers a per-call comparison of the
# default (128,1,4) conv vs a candidate local of the given shape on
# the REAL production weights/input. Each gemm_delegate_fp16_cl call
# logs [VERIFY ...] with per-call mismatch counts, max abs diff, and
# relative L2. Unset in normal runs.
VERIFY_ENV="${NNTR_DELEGATE_CONV_VERIFY:+NNTR_DELEGATE_CONV_VERIFY=$NNTR_DELEGATE_CONV_VERIFY}"
# Single timed run.
adb shell "cd /data/local/tmp/nntrainer/test; \
  export LD_LIBRARY_PATH=.; \
  export ${DELEGATE_ENV}; \
  export ${VERIFY_ENV}; \
  export NNTRAINER_RMSNORM_GPU=1; \
  export NNTRAINER_ATTN_GPU=1; \
  export NNTRAINER_PROFILE_LAYER_SYNC=1; \
  taskset f0 ./nntrainer_causallm /data/local/tmp/nntrainer/causallm/models/qwen3-4b" \
  2>&1 | tee ${RUN_LOG}

# ----------------------------------------------------------------------------
# Delegate conv work-group-size sweep. litert_lm's delegate_kernel_bench
# auto-tunes local[3] across 13 candidates and picks the fastest per shape.
# We're hardcoded to (128,1,4); set NNTR_CONV_LOCAL_SWEEP=1 to rerun the
# model under each candidate and print the resulting prefill TPS and
# conv(gpu) ms, then manually pick the best and bake it in.
if [ -n "$NNTR_CONV_LOCAL_SWEEP" ]; then
  echo ""
  echo "=========================================="
  echo " [SWEEP] conv kernel local-size candidates"
  echo "=========================================="
  SWEEP_LOG=../../temp_sweep.log
  > $SWEEP_LOG
  for L in "128,1,4" "64,1,4" "32,1,4" "64,2,4" "32,2,4" "128,1,2" "256,1,1"; do
    echo ""
    echo "--- local=${L} ---" | tee -a $SWEEP_LOG
    adb shell "cd /data/local/tmp/nntrainer/test; \
      export LD_LIBRARY_PATH=.; \
      export ${DELEGATE_ENV}; \
      export NNTRAINER_PROFILE_LAYER_SYNC=1; \
      export NNTR_DELEGATE_CONV_LOCAL=${L}; \
      taskset f0 ./nntrainer_causallm /data/local/tmp/nntrainer/causallm/models/qwen3-4b" 2>&1 \
      | grep -E "prefill:|conv\(gpu\)" | tee -a $SWEEP_LOG
  done
  echo ""
  echo "Sweep log: $SWEEP_LOG"
fi

cd ../..

adb pull /data/local/tmp/nntrainer/test/logs/. ./logs/ || true
adb shell "rm /data/local/tmp/nntrainer/test/logs/* 2>/dev/null || true"


# ----------------------------------------------------------------------------
# Single-run summary. Pulls the headline numbers + per-layer breakdowns
# from temp_run.log so `sh temp.sh` is self-contained without scrolling
# through the full device output. After the single timed run above we
# read temp_run.log (relative to repo root after `cd ../..`).
# ----------------------------------------------------------------------------
SUMMARY_LOG=temp_run.log
echo ""
echo "=========================================="
echo " Run summary"
echo "=========================================="

echo ""
echo "-- Gate resolution --"
grep "\[rms_norm\] NNTRAINER_RMSNORM_GPU\|\[mha_core\] NNTRAINER_ROPE_GPU" ${SUMMARY_LOG} \
  | head -4 \
  || echo "(no gate line)"

echo ""
echo "-- Perf --"
grep -E "prefill:|generation:|total:|peak memory|e2e time" ${SUMMARY_LOG} | head -5 \
  || echo "(no perf lines)"

echo ""
echo "-- Layer profiles (RMSNorm / Reshaped / MHA + sub-stages) --"
grep -A 8 "PROFILE RMSNormLayer prefill\|PROFILE ReshapedRMSNormLayer prefill\|PROFILE MHACoreLayer prefill" ${SUMMARY_LOG} \
  || echo "(no profile lines)"

echo ""
echo "-- Decode-path profiles (M==1, fired per generation token) --"
grep -A 8 "PROFILE MHACoreLayer decode\|PROFILE HalfTensor::dotQInteger decode\|PROFILE TieWordEmbedding (lm_head) decode\|PROFILE TieWordEmbedding (embedding) decode" ${SUMMARY_LOG} \
  || echo "(no decode profile lines)"

echo ""
echo "-- Generation snippet (first 300 chars after <|im_start|>assistant) --"
awk '/<\|im_start\|>assistant/ { f=1; next } f' ${SUMMARY_LOG} | head -c 300
echo ""

echo ""
echo "Full log: ${SUMMARY_LOG}"
echo ""
echo "To restore the original (long-context) config on device:"
echo "  adb shell 'mv ${CFG}.bak ${CFG}'"
