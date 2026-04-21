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
# NOTE: libOpenCL.so deliberately dropped — the bundled nntrainer
# libOpenCL.so runs the delegate conv kernel at 1.65 TFLOPS vs the
# device vendor driver's 2.8 TFLOPS. Without pushing our copy, the
# loader falls back to /system/vendor/lib64/libOpenCL.so.
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
# To get the honest per-layer wall-clock breakdown (conv attributed to
# fc instead of being absorbed into the next CPU layer's SVMMap fence),
# export NNTRAINER_PROFILE_LAYER_SYNC=1 before running. Default here is
# the async production path so the prefill TPS number is realistic.
adb shell "cd /data/local/tmp/nntrainer/test; \
  export LD_LIBRARY_PATH=.; \
  export ${DELEGATE_ENV}; \
  export NNTRAINER_PROFILE_LAYER_SYNC=1; \
  export NNTRAINER_DELEGATE_REPEAT_BENCH=32; \
  taskset f0 ./nntrainer_causallm /data/local/tmp/nntrainer/causallm/models/qwen3-4b" \
  2>&1 | tee ${RUN_LOG}

cd ../..

adb pull /data/local/tmp/nntrainer/test/logs/. ./logs/ || true
adb shell "rm /data/local/tmp/nntrainer/test/logs/* 2>/dev/null || true"

# ----------------------------------------------------------------------------
# Diagnostic: print DT_NEEDED of every .so / executable actually pushed to
# the device. The in-process maps dump shows libGLESv1_CM/v2/v3 loaded at
# bench entry, but the unittest process doesn't load those. Dropping
# -landroid from the CausalLM Android.mk didn't change this, meaning
# something in libnntrainer.so / libccapi-nntrainer.so / libcausallm_core.so
# (or a dep they pull in) still has libandroid.so or libGLES*.so in its
# DT_NEEDED. Use the NDK's aarch64 objdump (required since host readelf is
# x86_64-only) to dump the DYNAMIC section for each binary.
echo ""
echo "=========================================="
echo " DT_NEEDED of pushed binaries"
echo "=========================================="
NDK_OBJDUMP=$(find "$ANDROID_NDK" -name llvm-objdump 2>/dev/null | head -n 1)
if [ -n "$NDK_OBJDUMP" ]; then
  for f in Applications/CausalLM/jni/libs/arm64-v8a/nntrainer_causallm \
           Applications/CausalLM/jni/libs/arm64-v8a/libcausallm_core.so \
           builddir/android_build_result/lib/arm64-v8a/libnntrainer.so \
           builddir/android_build_result/lib/arm64-v8a/libccapi-nntrainer.so; do
    if [ -f "$f" ]; then
      echo ""
      echo "--- $f ---"
      "$NDK_OBJDUMP" -p "$f" 2>/dev/null | grep -E "NEEDED|SONAME" || true
    fi
  done
else
  echo "llvm-objdump not found under \$ANDROID_NDK; skipping DT_NEEDED dump"
fi

# ----------------------------------------------------------------------------
# Ground-truth: run the real delegate-conv-wave unittest to measure what the
# Adreno 830 actually achieves on M=437 N=4096 K=2560 in a fresh process.
# All of our in-process benches (PROFILING+OOO / PLAIN / UNITTEST_PRIMED)
# converge on ~1.65 TFLOPS, but the "2.81 TFLOPS" comparison target was
# never re-measured in this session. This runs the real unittest so we get
# a direct apples-to-apples number from the same device + driver + kernel.
# ----------------------------------------------------------------------------
echo ""
echo "=========================================="
echo " [TRUTH] running real unittest_delegate_conv_wave ModelShapes"
echo "=========================================="
sh unittest_delegate_conv_wave.sh 2>&1 | tee unittest_delegate_conv_wave.log || true

echo ""
echo "--- Unittest ModelShapes lines (ground truth TFLOPS per shape) ---"
grep -E "M=437|Delegate fp16|conv only|TOTAL:" unittest_delegate_conv_wave.log \
  || echo "(no ModelShapes lines found)"

# ----------------------------------------------------------------------------
# Bisection bench: nntr_delegate_bench_stage{0,1,2}. Same kernel / shape /
# setup as the unittest, but linked against libnntrainer + libccapi-nntrainer
# + libcausallm_core. Stage 0 only links them (DT_NEEDED, static init runs),
# Stage 1 additionally calls nntrainer::Engine::Global(), Stage 2 additionally
# registers a CausalLM model in the factory. Whichever stage first reports
# ~1.61 TFLOPS (instead of unittest's 2.36) is the culprit.
# ----------------------------------------------------------------------------
echo ""
echo "=========================================="
echo " [BISECT] nntr_delegate_bench stages 0 / 1 / 2"
echo "=========================================="
adb push nntrainer/tensor/cl_operations/cl_kernels/delegate_conv_wave.cl \
  /data/local/tmp/nntrainer/test/ >/dev/null
for stage in 0 1 2 3 4; do
  LIBS_BIN=Applications/CausalLM/jni/libs/arm64-v8a/nntr_delegate_bench_stage${stage}
  OBJ_BIN=Applications/CausalLM/jni/obj/local/arm64-v8a/nntr_delegate_bench_stage${stage}
  if [ -f "$LIBS_BIN" ]; then
    BIN=$LIBS_BIN
  elif [ -f "$OBJ_BIN" ]; then
    BIN=$OBJ_BIN
  else
    echo "[BISECT] nntr_delegate_bench_stage${stage} not built (checked $LIBS_BIN and $OBJ_BIN)"
    continue
  fi
  adb push "$BIN" /data/local/tmp/nntrainer/test/ >/dev/null
  adb shell chmod +x /data/local/tmp/nntrainer/test/nntr_delegate_bench_stage${stage}
  echo ""
  echo "--- nntr_delegate_bench stage=${stage} ---"
  adb shell "cd /data/local/tmp/nntrainer/test; \
    export LD_LIBRARY_PATH=.; \
    taskset f0 ./nntr_delegate_bench_stage${stage}" 2>&1 \
    | grep -E "STAGE|M=437|maps|Engine::Global|registerModel|GPU:"
done

# Stage 3 drop sub-bisection. We keep nntr_delegate_bench_stage3 (which
# calls Engine::Global() -> ClContext::initialize()) but vary the
# NNTR_INIT_LEVEL env var that ClContext::initialize() now reads to
# short-circuit its post-clInit sub-steps:
#   NNTR_INIT_LEVEL=0 : clInit() only (cl_ctx + cl_queue + initBuffers)
#   NNTR_INIT_LEVEL=1 : + initBlasClKernels()
#   NNTR_INIT_LEVEL=2 : + initAttentionClKernels()
#   NNTR_INIT_LEVEL=3 : + add_default_object() + setMemAllocator() [default]
# Whichever level first reproduces the 1.60 TFLOPS drop is the culprit.
echo ""
echo "=========================================="
echo " [BISECT 3x] stage 3 with varying NNTR_INIT_LEVEL"
echo "=========================================="
STAGE3_LIBS=Applications/CausalLM/jni/libs/arm64-v8a/nntr_delegate_bench_stage3
STAGE3_OBJ=Applications/CausalLM/jni/obj/local/arm64-v8a/nntr_delegate_bench_stage3
if [ -f "$STAGE3_LIBS" ]; then STAGE3=$STAGE3_LIBS
elif [ -f "$STAGE3_OBJ" ]; then STAGE3=$STAGE3_OBJ
else STAGE3=
fi
if [ -n "$STAGE3" ]; then
  # Confirm sub-bisection: init_level=0 should stay fast (clInit only),
  # init_level=1+ drops to 1.60 TFLOPS because initBlasClKernels runs.
  for lvl in 0 1; do
    echo ""
    echo "--- stage=3 NNTR_INIT_LEVEL=${lvl} ---"
    adb shell "cd /data/local/tmp/nntrainer/test; \
      export LD_LIBRARY_PATH=.; \
      export NNTR_INIT_LEVEL=${lvl}; \
      taskset f0 ./nntr_delegate_bench_stage3" 2>&1 \
      | grep -E "STAGE|M=437|NNTR_INIT_LEVEL|Engine::Global|GPU:"
  done
  # Binary-search which kernel inside initBlasClKernels tips the driver.
  # NNTR_INIT_LEVEL=1 restricts to exactly initBlasClKernels;
  # NNTR_BLAS_KERNEL_COUNT caps how many REG() calls actually execute.
  echo ""
  echo "[BISECT blas-kernel-count]"
  for cnt in 0 1 2 5 10 15 20 27; do
    echo ""
    echo "--- stage=3 LEVEL=1 BLAS_KERNEL_COUNT=${cnt} ---"
    adb shell "cd /data/local/tmp/nntrainer/test; \
      export LD_LIBRARY_PATH=.; \
      export NNTR_INIT_LEVEL=1; \
      export NNTR_BLAS_KERNEL_COUNT=${cnt}; \
      taskset f0 ./nntr_delegate_bench_stage3" 2>&1 \
      | grep -E "STAGE|M=437|BLAS_KERNEL_COUNT|initBlasClKernels"
  done
fi

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
# Match both FloatTensor (FP32 activation path) and HalfTensor (FP16
# activation path) profilers.
grep -A 4 "PROFILE .*Tensor::dotQInteger" ${RUN_LOG} || echo "(no PROFILE lines found)"

echo ""
echo "--- Layer wall-clock profile (prefill M>1) ---"
grep -E "PROFILE (RMSNormLayer|ReshapedRMSNormLayer|MHACoreLayer|SwiGLULayer|EmbeddingLayer|TieWordEmbedding)" ${RUN_LOG} || echo "(no layer PROFILE lines found)"

echo ""
echo "Full log: ${RUN_LOG}  (error.txt is intentionally untouched)"
echo ""
echo "To restore the original (long-context) config on device:"
echo "  adb shell 'mv ${CFG}.bak ${CFG}'"
