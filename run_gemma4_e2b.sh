#!/bin/bash
# Build, convert, quantize (INT4), and run Gemma4 E2B from the project root.
#
# Usage:
#   bash run_gemma4_e2b.sh [MODEL_DIR] [PROMPT] [OPTIONS]
#
# Options:
#   --convert-only   Convert HF weights to FP32 binary, then exit
#   --skip-convert   Skip weight conversion (FP32 binary already exists)
#   --skip-quantize  Skip INT4 quantization (run with FP32)
#   --force-build    Force rebuild of binaries
#   MODEL_DIR        Path containing HF model files (default: ~/models/gemma4)
#   PROMPT           Inference prompt (default: "Explain artificial intelligence briefly")

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GEMMA4_DIR="$SCRIPT_DIR/Applications/CausalLM/res/gemma4/e2b"

MODEL_DIR="${HOME}/models/gemma4"
CONVERT_ONLY=false
SKIP_CONVERT=false
RUN_ARGS=()

for arg in "$@"; do
    case "$arg" in
        --convert-only)  CONVERT_ONLY=true ;;
        --skip-convert)  SKIP_CONVERT=true ;;
        --skip-quantize) RUN_ARGS+=("--skip-quantize") ;;
        --force-build)   RUN_ARGS+=("--force-build") ;;
        */*)             MODEL_DIR="$arg" ;;
        *)               RUN_ARGS+=("$arg") ;;
    esac
done

FP32_BIN="$MODEL_DIR/nntr_gemma4_e2b_fp32.bin"

# Weight conversion
if [ "$SKIP_CONVERT" = false ]; then
    if [ ! -f "$FP32_BIN" ]; then
        echo "[run_gemma4_e2b] Converting HF weights → FP32 binary..."
        bash "$GEMMA4_DIR/run_converter.sh" "$MODEL_DIR"
    else
        echo "[run_gemma4_e2b] FP32 binary already exists, skipping conversion"
    fi
fi

[ "$CONVERT_ONLY" = true ] && { echo "[run_gemma4_e2b] Conversion done."; exit 0; }

# Build + quantize + push + run
bash "$GEMMA4_DIR/run_gemma4.sh" "$MODEL_DIR" "${RUN_ARGS[@]}"
