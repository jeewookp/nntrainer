#!/bin/bash
# weight conversion script for Gemma4 E2B

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PATH="${1:-$HOME/models/gemma4}"
OUTPUT="${2:-${MODEL_PATH}/nntr_gemma4_e2b_fp32.bin}"
DTYPE="${3:-float32}"

echo "Model path : $MODEL_PATH"
echo "Output     : $OUTPUT"
echo "Dtype      : $DTYPE"

python "$SCRIPT_DIR/weight_converter.py" \
    --model_path "$MODEL_PATH" \
    --output "$OUTPUT" \
    --dtype "$DTYPE"
