#!/bin/bash
# Build, quantize (INT4), push and run Gemma4 E2B on Android device
# Usage: bash run_gemma4.sh [model_dir] [prompt] [--skip-quantize] [--force-build]
set -e

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
if [ "$FORCE_BUILD" = true ] || [ ! -f "$LIBS_DIR/nntrainer_causallm" ]; then
    log_info "Building Android binary..."
    ndk-build \
        NDK_PROJECT_PATH=. \
        NDK_LIBS_OUT=./libs \
        NDK_OUT=./obj \
        APP_BUILD_SCRIPT=./Android.mk \
        NDK_APPLICATION_MK=./Application.mk \
        nntrainer_causallm causallm_core \
        -j$(nproc) 2>&1 | tail -10
    log_success "Android build done"
else
    log_success "Android binary up to date"
fi

# Linux meson build for nntr_quantize
QUANTIZE_BIN="$BUILD_DIR/Applications/CausalLM/nntr_quantize"
if [ "$SKIP_QUANTIZE" = false ]; then
    if [ "$FORCE_BUILD" = true ] || [ ! -f "$QUANTIZE_BIN" ]; then
        log_info "Building nntr_quantize (Linux)..."
        cd "$NNTRAINER_ROOT"
        if [ -f "$BUILD_DIR/build.ninja" ]; then
            meson configure "$BUILD_DIR" -Dplatform=none -Denable-app=true -Denable-transformer=true
        else
            meson setup "$BUILD_DIR" -Dplatform=none -Denable-app=true -Denable-transformer=true
        fi
        if ! ninja -C "$BUILD_DIR" nntr_quantize -j$(nproc); then
            log_error "nntr_quantize build failed"
            log_info "Available quantize-related targets:"
            ninja -C "$BUILD_DIR" -t targets | grep -i quantize || true
            exit 1
        fi
        log_success "nntr_quantize built"
    else
        log_success "nntr_quantize up to date"
    fi
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
    if [ ! -f "$Q4_BIN" ] || [ "$FP32_BIN" -nt "$Q4_BIN" ]; then
        log_info "Quantizing to Q4_0 (this may take a few minutes)..."
        "$QUANTIZE_BIN" "$MODEL_HOST_DIR" \
            --fc_dtype Q4_0 \
            --embd_dtype Q6_K \
            --lmhead_dtype Q4_0 \
            --output_bin nntr_gemma4_e2b_q40.bin \
            -o "$MODEL_HOST_DIR"
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
    "embedding_dtype": "Q6_K",
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

for lib in libcausallm_core.so libnntrainer.so libccapi-nntrainer.so libc++_shared.so libomp.so; do
    [ -f "$LIBS_DIR/$lib" ] && \
        adb push "$LIBS_DIR/$lib" "$DEVICE_INSTALL_DIR/" && \
        log_success "  $lib" || true
done

adb push "$LIBS_DIR/nntrainer_causallm" "$DEVICE_INSTALL_DIR/"
adb shell "chmod 755 $DEVICE_INSTALL_DIR/nntrainer_causallm"
log_success "  nntrainer_causallm"

# Push model files (skip if unchanged)
push_if_changed() {
    local src="$1" dst="$2"
    local hs=$(wc -c < "$src")
    local ds=$(adb shell "wc -c < $dst 2>/dev/null || echo 0" | tr -d '[:space:]')
    if [ "$hs" != "$ds" ]; then
        log_info "  Pushing $(basename $src) ($(du -h "$src" | cut -f1))..."
        adb push "$src" "$dst"
    fi
    log_success "  $(basename $src)"
}

push_if_changed "$PUSH_BIN"     "$DEVICE_MODEL_DIR/$PUSH_BIN_NAME"
push_if_changed "$TOKENIZER"    "$DEVICE_MODEL_DIR/tokenizer.json"
push_if_changed "$NNTR_CONFIG"  "$DEVICE_MODEL_DIR/nntr_config.json"
push_if_changed "$CONFIG_JSON"  "$DEVICE_MODEL_DIR/config.json"
[ -f "$GEN_CONFIG_JSON" ] && push_if_changed "$GEN_CONFIG_JSON" "$DEVICE_MODEL_DIR/generation_config.json"

# ── Step 5: Run ────────────────────────────────────────────────────────────
log_step 5 "Run on device"

echo -e "${YELLOW}── Output ──────────────────────────────────────${NC}"
adb shell "cd $DEVICE_INSTALL_DIR && \
    export LD_LIBRARY_PATH=$DEVICE_INSTALL_DIR:\$LD_LIBRARY_PATH && \
    ./nntrainer_causallm $DEVICE_MODEL_DIR '$PROMPT'" 2>&1
EXIT=$?
echo -e "${YELLOW}────────────────────────────────────────────────${NC}"

[ $EXIT -eq 0 ] && log_success "Done!" || { log_error "Failed (exit $EXIT)"; exit $EXIT; }
