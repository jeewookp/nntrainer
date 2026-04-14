#!/bin/sh
#
# test_seqlen_svm.sh
#
# Test Option A: shrink init_seq_len / max_seq_len in the on-device
# nntr_config.json so that tensor_pool's SVM allocation fits under the
# Adreno 830 per-allocation limit (1 GB), then re-run the Qwen3-4B
# inference using the already-pushed binary (NO rebuild).
#
# What this script does:
#   1. Back up the current on-device nntr_config.json
#   2. Patch init_seq_len -> 1024, max_seq_len -> 2048 with sed
#   3. Verify the patch
#   4. Run nntrainer_causallm and capture stdout+stderr -> error.txt
#   5. Extract the key [DIAG ...] lines so you can see the result at a
#      glance
#
# Usage (from nntrainer repo root):
#   sh test_seqlen_svm.sh
#
# Restoring the original config afterwards:
#   adb shell 'mv /data/local/tmp/nntrainer/causallm/models/qwen3-4b/nntr_config.json.bak \
#                 /data/local/tmp/nntrainer/causallm/models/qwen3-4b/nntr_config.json'
#

set -e

MODEL_DIR=/data/local/tmp/nntrainer/causallm/models/qwen3-4b
CFG=${MODEL_DIR}/nntr_config.json
BIN_DIR=/data/local/tmp/nntrainer/test

INIT_SEQ_LEN_NEW=1024
MAX_SEQ_LEN_NEW=2048

echo "=========================================="
echo " [1/5] Backing up on-device nntr_config.json"
echo "=========================================="
adb shell "[ -f ${CFG}.bak ] || cp ${CFG} ${CFG}.bak"
adb shell "ls -l ${CFG} ${CFG}.bak"

echo ""
echo "=========================================="
echo " [2/5] Patching init_seq_len -> ${INIT_SEQ_LEN_NEW},"
echo "                 max_seq_len  -> ${MAX_SEQ_LEN_NEW}"
echo "=========================================="
# Always patch from the pristine backup so repeated runs are idempotent.
adb shell "cp ${CFG}.bak ${CFG}"
adb shell "sed -i \
  -e 's/\"init_seq_len\"[[:space:]]*:[[:space:]]*[0-9]*/\"init_seq_len\": ${INIT_SEQ_LEN_NEW}/' \
  -e 's/\"max_seq_len\"[[:space:]]*:[[:space:]]*[0-9]*/\"max_seq_len\": ${MAX_SEQ_LEN_NEW}/' \
  ${CFG}"

echo ""
echo "=========================================="
echo " [3/5] Verifying patched config"
echo "=========================================="
adb shell "grep -E '\"init_seq_len\"|\"max_seq_len\"' ${CFG}"

echo ""
echo "=========================================="
echo " [4/5] Running nntrainer_causallm on device"
echo "        (binary is the one already pushed by the last temp.sh)"
echo "=========================================="
# Clean the logs dir first so the pull at the end is fresh.
adb shell "rm -f ${BIN_DIR}/logs/* 2>/dev/null || true"

adb shell "cd ${BIN_DIR} && export LD_LIBRARY_PATH=. && ./nntrainer_causallm ${MODEL_DIR}" 2>&1 \
  | tee error.txt

echo ""
echo "=========================================="
echo " [5/5] Extracted diagnostic summary"
echo "=========================================="

echo ""
echo "--- MemoryPool::allocate traces ---"
grep -E "DIAG MemoryPool::allocate" error.txt || echo "(no MemoryPool DIAG lines found)"

echo ""
echo "--- dotQInteger / dotBatched-QINT4 traces ---"
grep -E "DIAG dotQInteger|DIAG dotBatched-QINT4" error.txt || echo "(no dot DIAG lines found)"

echo ""
echo "--- attach_kai_buffer count ---"
KAI_COUNT=$(grep -c "DIAG attach_kai_buffer" error.txt || true)
echo "attach_kai_buffer invocations: ${KAI_COUNT}"

echo ""
echo "--- Generation output snippet (first 200 chars after assistant tag) ---"
awk '/<\|im_start\|>assistant/ { found=1; next } found { print }' error.txt | head -c 400
echo ""

echo ""
echo "--- Perf summary ---"
grep -E "prefill:|generation:|total:|peak memory|e2e time" error.txt || echo "(no perf lines)"

echo ""
echo "=========================================="
echo " Done. Full log: error.txt"
echo "=========================================="
echo ""
echo "Expected outcomes (success path):"
echo "  - [DIAG MemoryPool::allocate] tensor_pool pool_size < 1 GB"
echo "    and createSVMRegion SUCCESS for tensor_pool"
echo "  - [DIAG dotQInteger] svm_in=1 svm_out=1 svm_w=1"
echo "  - Coherent English text in the generation snippet"
echo ""
echo "To restore the original long context config:"
echo "  adb shell 'mv ${CFG}.bak ${CFG}'"
