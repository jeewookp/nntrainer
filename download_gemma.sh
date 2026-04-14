#!/usr/bin/env bash
# download_gemma.sh
#
# One-shot downloader for the Gemma 4 E2B LiteRT-LM model that Google's
# public benchmark (ai.google.dev/edge/litert-lm/overview) uses to report the
# "3,808 tokens/sec prefill" number on Samsung S26 Ultra GPU. On Adreno 830
# the same binary+model will run, just at a lower prefill TPS since the GPU
# is from a previous generation.
#
# Usage:
#   ./download_gemma.sh               # default dir ~/.cache/litert_lm_models
#   ./download_gemma.sh /path/to/dir  # custom output dir
#
# The script tries huggingface-cli first (preferred; handles resume,
# auth, sharding), and falls back to curl for a direct .litertlm file
# pull if the HF CLI isn't available. Gemma 4 is a gated model -- you
# must have accepted the license on HuggingFace and set HF_TOKEN before
# running, OR already be logged in via `huggingface-cli login`.
#
# Env overrides:
#   HF_REPO    (default: litert-community/gemma-4-E2B-it-litert-lm)
#   HF_FILE    (default: gemma-4-E2B-it.litertlm)
#   HF_TOKEN   (optional; HF access token. If unset, assumes CLI login.)

set -euo pipefail

OUT_DIR="${1:-$HOME/.cache/litert_lm_models}"
HF_REPO="${HF_REPO:-litert-community/gemma-4-E2B-it-litert-lm}"
HF_FILE="${HF_FILE:-gemma-4-E2B-it.litertlm}"

mkdir -p "${OUT_DIR}"
OUT_PATH="${OUT_DIR}/${HF_FILE}"

if [ -f "${OUT_PATH}" ]; then
  SIZE=$(stat -c '%s' "${OUT_PATH}" 2>/dev/null || stat -f '%z' "${OUT_PATH}")
  # Expect at least ~1 GB for an E2B int4 checkpoint.
  if [ "${SIZE}" -gt 500000000 ]; then
    echo "[download_gemma.sh] Already present: ${OUT_PATH}"
    echo "[download_gemma.sh] Size: ${SIZE} bytes"
    echo "[download_gemma.sh] Delete the file if you want to re-download."
    exit 0
  fi
  echo "[download_gemma.sh] Existing file looks truncated (${SIZE} bytes)."
  echo "[download_gemma.sh] Re-downloading ${OUT_PATH}..."
  rm -f "${OUT_PATH}"
fi

echo "[download_gemma.sh] Target:"
echo "    repo : ${HF_REPO}"
echo "    file : ${HF_FILE}"
echo "    out  : ${OUT_PATH}"
echo ""

# Preferred: huggingface-cli (resume + auth + sharding support)
if command -v huggingface-cli >/dev/null 2>&1; then
  echo "[download_gemma.sh] Using huggingface-cli"
  if [ -n "${HF_TOKEN:-}" ]; then
    HF_ARGS="--token ${HF_TOKEN}"
  else
    HF_ARGS=""
  fi
  # shellcheck disable=SC2086
  huggingface-cli download "${HF_REPO}" "${HF_FILE}" \
    --local-dir "${OUT_DIR}" \
    --local-dir-use-symlinks False \
    ${HF_ARGS}
elif command -v curl >/dev/null 2>&1; then
  echo "[download_gemma.sh] huggingface-cli not found; falling back to curl"
  URL="https://huggingface.co/${HF_REPO}/resolve/main/${HF_FILE}"
  if [ -n "${HF_TOKEN:-}" ]; then
    curl -L -f -o "${OUT_PATH}" \
      -H "Authorization: Bearer ${HF_TOKEN}" \
      "${URL}"
  else
    curl -L -f -o "${OUT_PATH}" "${URL}"
  fi
else
  echo "[download_gemma.sh] Neither huggingface-cli nor curl found. Install one:"
  echo "    pip install -U huggingface_hub[cli]"
  echo "    # or: apt install curl"
  exit 1
fi

echo ""
echo "[download_gemma.sh] Done. Model at: ${OUT_PATH}"
SIZE=$(stat -c '%s' "${OUT_PATH}" 2>/dev/null || stat -f '%z' "${OUT_PATH}")
echo "[download_gemma.sh] Size: ${SIZE} bytes"
