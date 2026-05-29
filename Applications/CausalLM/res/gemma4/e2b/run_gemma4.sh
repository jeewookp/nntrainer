#!/bin/bash
# Build, quantize (INT4), push and run Gemma4 E2B on Android device
# Usage: bash run_gemma4.sh [model_dir] [prompt] [--skip-quantize] [--force-build]
set -e
set -o pipefail

# ── Colors ─────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BLUE='\033[0;34m'; NC='\033[0m'
log_info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $1"; }
log_step()    { echo -e "\n${CYAN}[Step $1/$TOTAL_STEPS]${NC} $2\n${CYAN}──────────────────────${NC}"; }

# ── Args ────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NNTRAINER_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
CAUSALLM_DIR="$NNTRAINER_ROOT/Applications/CausalLM"

MODEL_HOST_DIR="${HOME}/models/gemma4"
PROMPT="Explain artificial intelligence briefly"
SKIP_QUANTIZE=false
FORCE_BUILD=false

for arg in "$@"; do
    case "$arg" in
        --skip-quantize)  SKIP_QUANTIZE=true ;;
        --force-build)    FORCE_BUILD=true ;;
        --*)              log_warning "Unknown option: $arg" ;;
        */*)              MODEL_HOST_DIR="$arg" ;;
        *)                PROMPT="$arg" ;;
    esac
done

FP32_BIN="$MODEL_HOST_DIR/nntr_gemma4_e2b_fp32.bin"
Q4_BIN="$MODEL_HOST_DIR/nntr_gemma4_e2b_q40.bin"
TOKENIZER="$MODEL_HOST_DIR/tokenizer.json"
CONFIG_JSON="$MODEL_HOST_DIR/config.json"
GEN_CONFIG_JSON="$MODEL_HOST_DIR/generation_config.json"
NNTR_CONFIG="$MODEL_HOST_DIR/nntr_config.json"

DEVICE_INSTALL_DIR="/data/local/tmp/nntrainer/causallm"
DEVICE_MODEL_DIR="$DEVICE_INSTALL_DIR/models/gemma4"
LIBS_DIR="$CAUSALLM_DIR/jni/libs/arm64-v8a"
BUILD_DIR="$NNTRAINER_ROOT/builddir"

TOTAL_STEPS=5

echo -e "${CYAN}╔══════════════════════════════════════╗${NC}"
echo -e "${CYAN}║   Gemma4 E2B – INT4 Run Script       ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════╝${NC}"
log_info "Model dir  : $MODEL_HOST_DIR"
log_info "Prompt     : $PROMPT"
log_info "Quantize   : $( [ "$SKIP_QUANTIZE" = true ] && echo 'skip' || echo 'Q4_0')"

# ── Step 1: Prerequisites ──────────────────────────────────────────────────
log_step 1 "Check prerequisites"

[ -z "$ANDROID_NDK" ] && { log_error "ANDROID_NDK not set"; exit 1; }
command -v adb &>/dev/null || { log_error "adb not found"; exit 1; }
adb devices | grep -q "device$" || { log_error "No Android device connected"; exit 1; }
DEVICE=$(adb devices | grep "device$" | head -1 | cut -f1)
log_success "Device: $DEVICE"

[ -f "$FP32_BIN" ] || { log_error "FP32 weight not found: $FP32_BIN\nRun run_converter.sh first."; exit 1; }
[ -f "$TOKENIZER" ] || { log_error "tokenizer.json not found"; exit 1; }
[ -f "$CONFIG_JSON" ] || { log_error "config.json not found"; exit 1; }
log_success "Model files found"

# ── Step 2: Build (NDK + Linux) ────────────────────────────────────────────
log_step 2 "Build"

export NNTRAINER_ROOT
cd "$CAUSALLM_DIR/jni"

# Android NDK build
BUILT_LIB="$LIBS_DIR/libcausallm_core.so"
NEEDS_NDK_BUILD=false
if [ "$FORCE_BUILD" = true ]; then
    NEEDS_NDK_BUILD=true
elif [ ! -f "$LIBS_DIR/nntrainer_causallm" ] || [ ! -f "$BUILT_LIB" ]; then
    NEEDS_NDK_BUILD=true
elif find "$CAUSALLM_DIR" \( -name "*.cpp" -o -name "*.h" -o -name "Android.mk" \) \
        -newer "$BUILT_LIB" | grep -q .; then
    log_info "Source files changed, rebuilding Android binary..."
    NEEDS_NDK_BUILD=true
fi

if [ "$NEEDS_NDK_BUILD" = true ]; then
    log_info "Building Android binary..."
    # NDK r23+ places both BUILD_EXECUTABLE and BUILD_SHARED_LIBRARY outputs
    # only in obj/local/ABI/ — not in NDK_LIBS_OUT/ABI/.
    # Only PREBUILT_SHARED_LIBRARY still copies to NDK_LIBS_OUT/ABI/.
    # Define both output paths up front.
    EXE_OBJ="$CAUSALLM_DIR/jni/obj/local/arm64-v8a/nntrainer_causallm"
    CORE_OBJ="$CAUSALLM_DIR/jni/obj/local/arm64-v8a/libcausallm_core.so"

    # Wipe all stale outputs so NDK must produce a fresh binary.
    rm -f "$BUILT_LIB"                # libs/libcausallm_core.so (stripped, if present)
    rm -f "$CORE_OBJ"                 # obj/local/.../libcausallm_core.so (unstripped)
    rm -rf "$CAUSALLM_DIR/jni/obj/local/arm64-v8a/objs/nntrainer_causallm"
    rm -f "$EXE_OBJ"
    rm -f "$LIBS_DIR/nntrainer_causallm"

    # Do NOT pipe through tail: set -o pipefail means a pipe masks ndk-build
    # failures (tail always exits 0).
    ndk-build \
        NDK_PROJECT_PATH=. \
        NDK_LIBS_OUT=./libs \
        NDK_OUT=./obj \
        APP_BUILD_SCRIPT=./Android.mk \
        NDK_APPLICATION_MK=./Application.mk \
        nntrainer_causallm causallm_core \
        -j$(nproc)

    # Copy freshly-built binaries from obj/local/ to libs/ so the push
    # step always finds them at a uniform path.  NDK r23+ no longer does
    # this automatically for BUILD_* modules (only PREBUILT_SHARED_LIBRARY
    # still copies to NDK_LIBS_OUT).
    [ -f "$EXE_OBJ" ]  && cp -f "$EXE_OBJ"  "$LIBS_DIR/nntrainer_causallm"  \
                        && log_info "  Copied nntrainer_causallm: obj/local → libs/"
    [ -f "$CORE_OBJ" ] && cp -f "$CORE_OBJ" "$LIBS_DIR/libcausallm_core.so" \
                        && log_info "  Copied libcausallm_core.so: obj/local → libs/"

    log_success "Android build done"
else
    log_success "Android binary up to date"
fi

# ── Ensure libnntrainer.so is up-to-date for Android ──────────────────────────
# OpenCL kernel sources (.cl files) are embedded as C++ string literals at
# meson configure/build time.  ndk-build cannot update them.  We must rebuild
# libnntrainer.so via package_android.sh when:
#   (a) builddir is missing or misconfigured for x86 (not Android), OR
#   (b) any .cl file is newer than the installed libnntrainer.so
_PREBUILT_SO="$BUILD_DIR/android_build_result/lib/arm64-v8a/libnntrainer.so"
_CL_DIR="$NNTRAINER_ROOT/nntrainer/tensor/cl_operations/cl_kernels"

# Reads the meson 'platform' option from builddir; prints "android", "none", etc.
_meson_platform() {
    python3 -c "
import json,sys
try:
    opts=json.load(open('$BUILD_DIR/meson-info/intro-buildoptions.json'))
    print(next((str(o.get('value','')) for o in opts if o.get('name')=='platform'), 'unknown'))
except: print('unknown')
" 2>/dev/null || echo "unknown"
}

_platform=$(_meson_platform)
_cl_changed=false
if [ -f "$_PREBUILT_SO" ] && \
   find "$_CL_DIR" -name "*.cl" -newer "$_PREBUILT_SO" 2>/dev/null | grep -q .; then
    _cl_changed=true
fi

if [ "$_platform" = "android" ] && [ "$_cl_changed" = true ]; then
    # builddir is correctly configured for Android and only CL sources changed:
    # incremental rebuild is sufficient.
    log_info "OpenCL kernels changed — incremental rebuild of libnntrainer.so..."
    ninja -C "$BUILD_DIR" install
    log_success "libnntrainer.so rebuilt (incremental)"
elif [ "$_platform" != "android" ]; then
    # builddir is missing or misconfigured (e.g. reconfigured to x86).
    # Full reconfigure + build via package_android.sh.
    log_info "builddir platform=$_platform (not Android) — rebuilding libnntrainer.so..."
    log_info "This may take several minutes..."
    bash "$NNTRAINER_ROOT/tools/package_android.sh" "$NNTRAINER_ROOT" --arm-arch=armv8.2-a
    log_success "libnntrainer.so rebuilt (full)"
fi
unset _PREBUILT_SO _CL_DIR _platform _cl_changed

# Linux meson build for nntr_quantize using a dedicated x86 builddir
# (arm-arch=none → AVX path; avoids ARM NEON recompile of libnntrainer)
QUANTIZE_BUILDDIR="$NNTRAINER_ROOT/builddir_quantize"
QUANTIZE_BIN="$QUANTIZE_BUILDDIR/Applications/CausalLM/nntr_quantize"
if [ "$SKIP_QUANTIZE" = false ]; then
    log_info "Building nntr_quantize (Linux x86)..."
    cd "$NNTRAINER_ROOT"
    # Stamp file records the exact options; delete builddir if stale
    OPTS_STAMP="arm-arch=none;enable-fp16=false;enable-opencl=false;enable-transformer=true"
    OPTS_FILE="$QUANTIZE_BUILDDIR/.nntr_quantize_opts"
    if [ "$FORCE_BUILD" = true ] && [ -d "$QUANTIZE_BUILDDIR" ]; then
        rm -rf "$QUANTIZE_BUILDDIR"
    elif [ -d "$QUANTIZE_BUILDDIR" ]; then
        if [ ! -f "$OPTS_FILE" ] || [ "$(cat "$OPTS_FILE" 2>/dev/null)" != "$OPTS_STAMP" ]; then
            log_warning "builddir_quantize has stale settings, cleaning..."
            rm -rf "$QUANTIZE_BUILDDIR"
        fi
    fi
    if [ ! -f "$QUANTIZE_BUILDDIR/build.ninja" ]; then
        meson setup "$QUANTIZE_BUILDDIR" \
            -Dplatform=none \
            -Denable-app=true \
            -Denable-transformer=true \
            -Denable-fp16=false \
            -Denable-opencl=false \
            -Darm-arch=none
        echo "$OPTS_STAMP" > "$OPTS_FILE"
    fi
    if ! meson compile -C "$QUANTIZE_BUILDDIR" nntr_quantize; then
        log_error "nntr_quantize build failed"; exit 1
    fi
    log_success "nntr_quantize built"
fi

# ── Step 3: Quantize to INT4 ───────────────────────────────────────────────
log_step 3 "Quantize (FP32 → Q4_0)"

# Write FP32 nntr_config for quantizer input
cat > "$NNTR_CONFIG" <<EOF
{
    "model_tensor_type": "FP32-FP32",
    "model_file_name": "nntr_gemma4_e2b_fp32.bin",
    "fc_layer_dtype": "FP32",
    "embedding_dtype": "FP32",
    "lora_rank": 0, "lora_alpha": 0, "lora_target": [],
    "bad_word_ids": [], "fsu": false, "fsu_lookahead": 2,
    "num_to_generate": 128, "init_seq_len": 128, "max_seq_len": 256,
    "batch_size": 1,
    "tokenizer_file": "${DEVICE_MODEL_DIR}/tokenizer.json",
    "sample_input": "Explain artificial intelligence briefly"
}
EOF

if [ "$SKIP_QUANTIZE" = false ]; then
    # Stamp file invalidates the Q4 bin when quantize params change.
    # Q6_K is incompatible with Gemma4-E2B (hidden_size=1152 is not divisible
    # by 256) and also conflicts with lmhead_dtype=Q4_0 when tie_word_embeddings
    # is true, causing "tensor dimension mismatch" at runtime.  Use Q4_0 for
    # the embedding so both dtypes match and block-alignment holds (1152/32=36).
    QUANTIZE_PARAMS="fc=Q4_0,embd=Q4_0,lmhead=Q4_0"
    QUANTIZE_STAMP="$MODEL_HOST_DIR/.nntr_quantize_stamp"
    if [ "$(cat "$QUANTIZE_STAMP" 2>/dev/null)" != "$QUANTIZE_PARAMS" ]; then
        rm -f "$Q4_BIN"
        log_info "Quantize params changed — removing stale bin for re-quantization"
    fi

    if [ ! -f "$Q4_BIN" ] || [ "$FP32_BIN" -nt "$Q4_BIN" ]; then
        log_info "Quantizing to Q4_0 (this may take a few minutes)..."
        "$QUANTIZE_BIN" "$MODEL_HOST_DIR" \
            --fc_dtype Q4_0 \
            --embd_dtype Q4_0 \
            --lmhead_dtype Q4_0 \
            --output_bin nntr_gemma4_e2b_q40.bin \
            -o "$MODEL_HOST_DIR"
        echo "$QUANTIZE_PARAMS" > "$QUANTIZE_STAMP"
        log_success "Quantized: $Q4_BIN ($(du -h "$Q4_BIN" | cut -f1))"
    else
        log_success "Q4_0 bin up to date"
    fi

    # Write Q4_0 nntr_config (overwrite)
    cat > "$NNTR_CONFIG" <<EOF
{
    "model_tensor_type": "Q4_0-FP32",
    "model_file_name": "nntr_gemma4_e2b_q40.bin",
    "fc_layer_dtype": "Q4_0",
    "embedding_dtype": "Q4_0",
    "lmhead_dtype": "Q4_0",
    "lora_rank": 0, "lora_alpha": 0, "lora_target": [],
    "bad_word_ids": [], "fsu": false, "fsu_lookahead": 2,
    "num_to_generate": 128, "init_seq_len": 128, "max_seq_len": 256,
    "batch_size": 1,
    "tokenizer_file": "${DEVICE_MODEL_DIR}/tokenizer.json",
    "sample_input": "Explain artificial intelligence briefly"
}
EOF
    PUSH_BIN="$Q4_BIN"
    PUSH_BIN_NAME="nntr_gemma4_e2b_q40.bin"
else
    PUSH_BIN="$FP32_BIN"
    PUSH_BIN_NAME="nntr_gemma4_e2b_fp32.bin"
    log_success "Skipped (using FP32)"
fi

# ── Step 4: Push ───────────────────────────────────────────────────────────
log_step 4 "Push to device"

adb shell "mkdir -p $DEVICE_INSTALL_DIR $DEVICE_MODEL_DIR"

# NDK r23+ no longer auto-copies PREBUILT_SHARED_LIBRARY outputs to
# NDK_LIBS_OUT.  Copy them explicitly so the push loop below can push the
# exact libnntrainer.so that was used at link time.  This is critical: if
# libcausallm_core.so was linked against a libnntrainer.so that exports
# svm_alloc_ints, the device must have that same (or newer) version or the
# Android dynamic linker will report "cannot locate symbol".
PREBUILT_LIB_DIR="$BUILD_DIR/android_build_result/lib/arm64-v8a"
for _plib in libnntrainer.so libccapi-nntrainer.so; do
    if [ -f "$PREBUILT_LIB_DIR/$_plib" ]; then
        cp -f "$PREBUILT_LIB_DIR/$_plib" "$LIBS_DIR/"
        log_info "  Staged prebuilt: $_plib"
    fi
done

for lib in libcausallm_core.so libnntrainer.so libccapi-nntrainer.so libc++_shared.so libomp.so; do
    [ -f "$LIBS_DIR/$lib" ] && \
        adb push "$LIBS_DIR/$lib" "$DEVICE_INSTALL_DIR/" && \
        log_success "  $lib" || true
done

adb push "$LIBS_DIR/nntrainer_causallm" "$DEVICE_INSTALL_DIR/"
adb shell "chmod 755 $DEVICE_INSTALL_DIR/nntrainer_causallm"
log_success "  nntrainer_causallm"

# Push model files.
# Use md5sum for correctness: file-size comparison silently misses same-size
# edits (e.g. "Q6_K" -> "Q4_0" in nntr_config.json).
push_if_changed() {
    local src="$1" dst="$2"
    local h_md5 d_md5
    if [ ! -f "$src" ]; then
        log_error "  $(basename "$src") not found on host: $src"; return 1
    fi
    h_md5=$(md5sum "$src" | cut -d' ' -f1)
    # adb shell md5sum returns non-zero when the file is absent on the device;
    # suppress the error with || so set -e does not silently abort the script.
    d_md5=$(adb shell "md5sum '$dst' 2>/dev/null" 2>/dev/null | awk '{print $1}' | tr -d '[:space:]') || d_md5=""
    if [ "$h_md5" != "$d_md5" ]; then
        log_info "  Pushing $(basename "$src") ($(du -h "$src" | cut -f1))..."
        adb push "$src" "$dst" || { log_error "  adb push failed for $(basename "$src")"; return 1; }
    fi
    log_success "  $(basename "$src")"
}

push_if_changed "$PUSH_BIN"     "$DEVICE_MODEL_DIR/$PUSH_BIN_NAME"
push_if_changed "$TOKENIZER"    "$DEVICE_MODEL_DIR/tokenizer.json"
push_if_changed "$NNTR_CONFIG"  "$DEVICE_MODEL_DIR/nntr_config.json"
push_if_changed "$CONFIG_JSON"  "$DEVICE_MODEL_DIR/config.json"
[ -f "$GEN_CONFIG_JSON" ] && push_if_changed "$GEN_CONFIG_JSON" "$DEVICE_MODEL_DIR/generation_config.json"

# ── Step 5: Run ────────────────────────────────────────────────────────────
log_step 5 "Run on device"

echo -e "${YELLOW}── Output ──────────────────────────────────────${NC}"
set +e
adb shell "cd $DEVICE_INSTALL_DIR && \
    export LD_LIBRARY_PATH=$DEVICE_INSTALL_DIR:\$LD_LIBRARY_PATH && \
    export OMP_NUM_THREADS=4 && \
    export NNTRAINER_ASYNC_PIPELINE=1 && \
    ./nntrainer_causallm $DEVICE_MODEL_DIR '$PROMPT'; echo \"[EXIT] \$?\"" 2>&1
ADB_EXIT=$?
set -e
echo -e "${YELLOW}────────────────────────────────────────────────${NC}"
log_info "adb shell exit: $ADB_EXIT"

[ $ADB_EXIT -eq 0 ] && log_success "Done!" || { log_error "Failed (adb exit $ADB_EXIT)"; exit $ADB_EXIT; }
