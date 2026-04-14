#!/usr/bin/env bash
# download_gemma.sh
#
# Downloader for the Gemma 4 E2B LiteRT-LM model that Google's public
# benchmark (ai.google.dev/edge/litert-lm/overview) uses to report the
# "3,808 tokens/sec prefill" number on Samsung S26 Ultra GPU. On Adreno 830
# the same binary+model will run, just at a lower prefill TPS since the
# GPU is from a previous generation.
#
# IMPORTANT: Gemma 4 is a gated model on HuggingFace. Before running this
# script you must:
#
#   1. Visit https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm
#      (log in, click "Agree and access"). One-time per HF account.
#
#   2. Create a READ token at https://huggingface.co/settings/tokens
#      and export it as HF_TOKEN:
#
#          export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
#
#      OR log in once with `huggingface-cli login` after `pip install -U
#      "huggingface_hub[cli]"`. Either works.
#
# Usage:
#   ./download_gemma.sh               # default dir ~/.cache/litert_lm_models
#   ./download_gemma.sh /path/to/dir  # custom output dir
#
# Env overrides:
#   HF_REPO    (default: litert-community/gemma-4-E2B-it-litert-lm)
#   HF_FILE    (default: gemma-4-E2B-it.litertlm)
#   HF_TOKEN   (optional; HF access token. If unset, tries HF cached login.)
#
# Fallback chain (first that succeeds wins):
#   1. huggingface-cli download         (preferred: resume, auth, sharding)
#   2. python3 + huggingface_hub        (same library, programmatic)
#   3. python3 + urllib.request         (no HF library needed, uses HF_TOKEN)
#   4. wget                             (needs HF_TOKEN)
#   5. curl                             (needs HF_TOKEN; snap curl is BROKEN,
#                                        we detect and refuse)
#
# Run with `bash`, NOT `sh`. `sh` on Ubuntu is dash which does not support
# `set -o pipefail` or `[[ ... ]]`.

set -e
set -u
set -o pipefail

OUT_DIR="${1:-$HOME/.cache/litert_lm_models}"
HF_REPO="${HF_REPO:-litert-community/gemma-4-E2B-it-litert-lm}"
HF_FILE="${HF_FILE:-gemma-4-E2B-it.litertlm}"

mkdir -p "${OUT_DIR}"
OUT_PATH="${OUT_DIR}/${HF_FILE}"

# Reject dash: this script uses bash-only features.
if [ -z "${BASH_VERSION:-}" ]; then
  echo "[download_gemma.sh] This script requires bash, not sh/dash."
  echo "    Please run: ./download_gemma.sh   (uses the #!/usr/bin/env bash shebang)"
  echo "    NOT:        sh download_gemma.sh  (dash does not support set -o pipefail)"
  exit 2
fi

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

# --- Helpers --------------------------------------------------------------

is_snap_binary() {
  # If `which <cmd>` resolves to /snap/... it's a snap-confined binary and
  # its network access is broken for HF TLS in the observed case (see
  # user's error: "curl: (35) Recv failure: Connection reset by peer").
  local cmd_path
  cmd_path=$(command -v "$1" 2>/dev/null || true)
  [[ -n "${cmd_path}" ]] && [[ "${cmd_path}" == /snap/* ]]
}

have() {
  command -v "$1" >/dev/null 2>&1
}

check_token_if_unauth_error() {
  if [ -z "${HF_TOKEN:-}" ] && ! have huggingface-cli; then
    echo ""
    echo "[download_gemma.sh] NOTE: no HF_TOKEN in env and no huggingface-cli"
    echo "[download_gemma.sh] login. Gemma 4 is a GATED model. Set HF_TOKEN:"
    echo ""
    echo "    1. Accept license: https://huggingface.co/${HF_REPO}"
    echo "    2. Create token:   https://huggingface.co/settings/tokens"
    echo "    3. export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx"
    echo "    4. ./download_gemma.sh"
    echo ""
  fi
}

HF_URL="https://huggingface.co/${HF_REPO}/resolve/main/${HF_FILE}"

# --- Attempt 1: huggingface-cli -------------------------------------------
if have huggingface-cli && ! is_snap_binary huggingface-cli; then
  echo "[download_gemma.sh] Trying huggingface-cli..."
  HF_ARGS=()
  if [ -n "${HF_TOKEN:-}" ]; then
    HF_ARGS+=(--token "${HF_TOKEN}")
  fi
  if huggingface-cli download "${HF_REPO}" "${HF_FILE}" \
      --local-dir "${OUT_DIR}" \
      --local-dir-use-symlinks False \
      "${HF_ARGS[@]}"; then
    echo "[download_gemma.sh] huggingface-cli download OK"
  else
    echo "[download_gemma.sh] huggingface-cli failed, trying next fallback..."
    rm -f "${OUT_PATH}"
  fi
fi

# --- Attempt 2: python3 + huggingface_hub --------------------------------
if [ ! -f "${OUT_PATH}" ] && have python3; then
  if python3 -c "import huggingface_hub" >/dev/null 2>&1; then
    echo "[download_gemma.sh] Trying python3 huggingface_hub..."
    if python3 - "${HF_REPO}" "${HF_FILE}" "${OUT_DIR}" <<'PY'
import os, sys
from huggingface_hub import hf_hub_download
repo, filename, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
token = os.environ.get("HF_TOKEN") or None
try:
    p = hf_hub_download(repo_id=repo, filename=filename,
                        local_dir=out_dir, token=token)
    print(f"[hf_hub_download] OK: {p}")
except Exception as e:
    print(f"[hf_hub_download] FAILED: {e}", file=sys.stderr)
    sys.exit(1)
PY
    then
      echo "[download_gemma.sh] python huggingface_hub OK"
    else
      echo "[download_gemma.sh] python huggingface_hub failed, trying next fallback..."
      rm -f "${OUT_PATH}"
    fi
  fi
fi

# --- Attempt 3: python3 + urllib.request ----------------------------------
if [ ! -f "${OUT_PATH}" ] && have python3; then
  echo "[download_gemma.sh] Trying python3 urllib.request (streaming)..."
  if python3 - "${HF_URL}" "${OUT_PATH}" <<'PY'
import os, sys, urllib.request
url, out = sys.argv[1], sys.argv[2]
req = urllib.request.Request(url)
token = os.environ.get("HF_TOKEN")
if token:
    req.add_header("Authorization", f"Bearer {token}")
try:
    with urllib.request.urlopen(req) as r:
        total = int(r.headers.get("Content-Length", "0"))
        print(f"[urllib] starting, content-length={total}")
        n = 0
        CHUNK = 1 << 20  # 1 MB
        with open(out, "wb") as f:
            while True:
                b = r.read(CHUNK)
                if not b:
                    break
                f.write(b)
                n += len(b)
                if total:
                    pct = 100.0 * n / total
                    print(f"  {n} / {total} bytes ({pct:.1f}%)", end="\r")
        print("")
    print(f"[urllib] wrote {n} bytes to {out}")
except urllib.error.HTTPError as e:
    print(f"[urllib] HTTPError {e.code}: {e.reason}", file=sys.stderr)
    if e.code in (401, 403):
        print("    -> This model is gated. Set HF_TOKEN (see top of script).",
              file=sys.stderr)
    sys.exit(2)
except Exception as e:
    print(f"[urllib] FAILED: {e}", file=sys.stderr)
    sys.exit(3)
PY
  then
    echo "[download_gemma.sh] python urllib OK"
  else
    echo "[download_gemma.sh] python urllib failed, trying next fallback..."
    rm -f "${OUT_PATH}"
  fi
fi

# --- Attempt 4: wget ------------------------------------------------------
if [ ! -f "${OUT_PATH}" ] && have wget && ! is_snap_binary wget; then
  echo "[download_gemma.sh] Trying wget..."
  WGET_ARGS=(-O "${OUT_PATH}" --content-on-error)
  if [ -n "${HF_TOKEN:-}" ]; then
    WGET_ARGS+=(--header "Authorization: Bearer ${HF_TOKEN}")
  fi
  if wget "${WGET_ARGS[@]}" "${HF_URL}"; then
    echo "[download_gemma.sh] wget OK"
  else
    echo "[download_gemma.sh] wget failed, trying next fallback..."
    rm -f "${OUT_PATH}"
  fi
fi

# --- Attempt 5: curl (only if not snap) -----------------------------------
if [ ! -f "${OUT_PATH}" ] && have curl; then
  if is_snap_binary curl; then
    echo "[download_gemma.sh] curl is snap-confined. Skipping (observed to"
    echo "    fail with 'Recv failure: Connection reset by peer' on HF TLS)."
    echo "    Install native curl with:   sudo apt install curl   after"
    echo "    removing snap:              sudo snap remove curl"
  else
    echo "[download_gemma.sh] Trying curl..."
    CURL_ARGS=(-L -f -o "${OUT_PATH}")
    if [ -n "${HF_TOKEN:-}" ]; then
      CURL_ARGS+=(-H "Authorization: Bearer ${HF_TOKEN}")
    fi
    if curl "${CURL_ARGS[@]}" "${HF_URL}"; then
      echo "[download_gemma.sh] curl OK"
    else
      echo "[download_gemma.sh] curl failed."
      rm -f "${OUT_PATH}"
    fi
  fi
fi

# --- Verify ---------------------------------------------------------------
if [ ! -f "${OUT_PATH}" ]; then
  check_token_if_unauth_error
  echo ""
  echo "[download_gemma.sh] ALL fallbacks failed. Fastest path forward:"
  echo ""
  echo "    # Install HF CLI into user site-packages (no sudo needed):"
  echo "    pip install --user -U 'huggingface_hub[cli]'"
  echo "    export PATH=\"\$HOME/.local/bin:\$PATH\""
  echo ""
  echo "    # Accept license on the web:"
  echo "    # https://huggingface.co/${HF_REPO}"
  echo ""
  echo "    # Authenticate (paste your READ token when asked):"
  echo "    huggingface-cli login"
  echo ""
  echo "    # Re-run:"
  echo "    ./download_gemma.sh"
  exit 1
fi

SIZE=$(stat -c '%s' "${OUT_PATH}" 2>/dev/null || stat -f '%z' "${OUT_PATH}")
echo ""
echo "[download_gemma.sh] Done. Model at: ${OUT_PATH}"
echo "[download_gemma.sh] Size: ${SIZE} bytes"
if [ "${SIZE}" -lt 500000000 ]; then
  echo "[download_gemma.sh] WARNING: Size <500 MB looks too small. You may"
  echo "    have downloaded an HTML error page instead of the model. First"
  echo "    few bytes:"
  head -c 256 "${OUT_PATH}" || true
  echo ""
  exit 1
fi
