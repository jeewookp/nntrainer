#!/bin/bash
# Build, push and run Gemma4 E2B on Android device
set -e

# ── Colors ─────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BLUE='\033[0;34m'; NC='\033[0m'
log_info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $1"; }
log_step()    { echo -e "\n${CYAN}[Step $1/$2]${NC} $3\n${CYAN}──────────────────────────────────${NC}"; }

# ── Config ──────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NNTRAINER_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
CAUSALLM_DIR="$NNTRAINER_ROOT/Applications/CausalLM"

MODEL_HOST_DIR="${1:-$HOME/models/gemma4}"          # local model dir
PROMPT="${2:-Explain artificial intelligence briefly}"

DEVICE_INSTALL_DIR="/data/local/tmp/nntrainer/causallm"
DEVICE_MODEL_DIR="$DEVICE_INSTALL_DIR/models/gemma4"
DEVICE_BIN="$DEVICE_INSTALL_DIR/nntrainer_causallm"
LIBS_DIR="$CAUSALLM_DIR/jni/libs/arm64-v8a"

TOTAL_STEPS=5

echo -e "${CYAN}╔══════════════════════════════════════╗${NC}"
echo -e "${CYAN}║       Gemma4 E2B – Run Script        ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════╝${NC}"
log_info "Model dir (host) : $MODEL_HOST_DIR"
log_info "Prompt           : $PROMPT"

# ── Step 1: Prerequisites ──────────────────────────────────────────────────
log_step 1 $TOTAL_STEPS "Check prerequisites"

[ -z "$ANDROID_NDK" ] && { log_error "ANDROID_NDK not set"; exit 1; }
command -v adb &>/dev/null || { log_error "adb not found"; exit 1; }
adb devices | grep -q "device$" || { log_error "No Android device connected"; exit 1; }

DEVICE=$(adb devices | grep "device$" | head -1 | cut -f1)
log_success "Device: $DEVICE"

BIN_FILE="$HOME/models/gemma4/nntr_gemma4_e2b_fp32.bin"
[ -f "$BIN_FILE" ] || { log_error "Weight file not found: $BIN_FILE\nRun run_converter.sh first."; exit 1; }
log_success "Weight file found ($(du -h "$BIN_FILE" | cut -f1))"

TOKENIZER="$MODEL_HOST_DIR/tokenizer.json"
[ -f "$TOKENIZER" ] || { log_error "tokenizer.json not found: $TOKENIZER"; exit 1; }
log_success "Tokenizer found"

# ── Step 2: Build ──────────────────────────────────────────────────────────
log_step 2 $TOTAL_STEPS "Build with NDK"

export NNTRAINER_ROOT
cd "$CAUSALLM_DIR/jni"

SKIP_BUILD=false
if [ -f "$LIBS_DIR/nntrainer_causallm" ]; then
    # Rebuild if any source is newer than binary
    NEWEST_SRC=$(find "$CAUSALLM_DIR" -name "*.cpp" -o -name "*.h" | \
                 xargs ls -t 2>/dev/null | head -1)
    if [ -n "$NEWEST_SRC" ] && [ "$NEWEST_SRC" -nt "$LIBS_DIR/nntrainer_causallm" ]; then
        log_info "Sources changed – rebuilding"
    else
        log_info "Binary up to date – skipping build (use --force-build to override)"
        SKIP_BUILD=true
    fi
fi

if [ "$3" = "--force-build" ]; then SKIP_BUILD=false; fi

if [ "$SKIP_BUILD" = false ]; then
    ndk-build \
        NDK_PROJECT_PATH=. \
        NDK_LIBS_OUT=./libs \
        NDK_OUT=./obj \
        APP_BUILD_SCRIPT=./Android.mk \
        NDK_APPLICATION_MK=./Application.mk \
        nntrainer_causallm causallm_core \
        -j$(nproc) 2>&1 | tail -20
    log_success "Build complete"
else
    log_success "Skipped"
fi

# ── Step 3: Generate nntr_config.json ─────────────────────────────────────
log_step 3 $TOTAL_STEPS "Generate nntr_config.json"

CONFIG_FILE="$MODEL_HOST_DIR/nntr_config.json"
cat > "$CONFIG_FILE" <<EOF
{
    "model_tensor_type": "FP32-FP32",
    "model_file_name": "nntr_gemma4_e2b_fp32.bin",
    "fc_layer_dtype": "FP32",
    "embedding_dtype": "FP32",
    "lora_rank": 0,
    "lora_alpha": 0,
    "lora_target": [],
    "bad_word_ids": [],
    "fsu": false,
    "fsu_lookahead": 2,
    "num_to_generate": 128,
    "init_seq_len": 128,
    "max_seq_len": 256,
    "batch_size": 1,
    "tokenizer_file": "${DEVICE_MODEL_DIR}/tokenizer.json",
    "sample_input": "Explain artificial intelligence briefly"
}
EOF
log_success "Written: $CONFIG_FILE"

# ── Step 4: Push files ─────────────────────────────────────────────────────
log_step 4 $TOTAL_STEPS "Push files to device"

adb shell "mkdir -p $DEVICE_INSTALL_DIR $DEVICE_MODEL_DIR"

# Libraries
for lib in libcausallm_core.so libnntrainer.so libccapi-nntrainer.so libc++_shared.so libomp.so; do
    [ -f "$LIBS_DIR/$lib" ] && adb push "$LIBS_DIR/$lib" "$DEVICE_INSTALL_DIR/" && \
        log_success "  $lib" || log_warning "  $lib (not found, skipping)"
done

# Executable
adb push "$LIBS_DIR/nntrainer_causallm" "$DEVICE_INSTALL_DIR/"
adb shell "chmod 755 $DEVICE_BIN"
log_success "  nntrainer_causallm"

# Model files (skip if already on device and same size)
push_if_changed() {
    local src="$1" dst="$2"
    local host_size=$(wc -c < "$src")
    local dev_size=$(adb shell "wc -c < $dst 2>/dev/null || echo 0" | tr -d '[:space:]')
    if [ "$host_size" != "$dev_size" ]; then
        log_info "  Pushing $(basename $src) ($(du -h "$src" | cut -f1))..."
        adb push "$src" "$dst"
        log_success "  $(basename $src)"
    else
        log_success "  $(basename $src) (unchanged, skipped)"
    fi
}

push_if_changed "$BIN_FILE"   "$DEVICE_MODEL_DIR/nntr_gemma4_e2b_fp32.bin"
push_if_changed "$TOKENIZER"  "$DEVICE_MODEL_DIR/tokenizer.json"
push_if_changed "$CONFIG_FILE" "$DEVICE_MODEL_DIR/nntr_config.json"

# HuggingFace config.json and generation_config.json
for f in config.json generation_config.json; do
    [ -f "$MODEL_HOST_DIR/$f" ] && \
        adb push "$MODEL_HOST_DIR/$f" "$DEVICE_MODEL_DIR/" && \
        log_success "  $f" || log_warning "  $f (not found)"
done

# ── Step 5: Run ────────────────────────────────────────────────────────────
log_step 5 $TOTAL_STEPS "Run on device"

echo ""
echo -e "${YELLOW}Output:${NC}"
echo "──────────────────────────────────────────────"
adb shell "cd $DEVICE_INSTALL_DIR && \
    export LD_LIBRARY_PATH=$DEVICE_INSTALL_DIR:\$LD_LIBRARY_PATH && \
    $DEVICE_BIN $DEVICE_MODEL_DIR '$PROMPT'" 2>&1
EXIT_CODE=$?
echo "──────────────────────────────────────────────"
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    log_success "Done!"
else
    log_error "Execution failed (exit code $EXIT_CODE)"
    exit $EXIT_CODE
fi
