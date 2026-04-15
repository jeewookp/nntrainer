#!/usr/bin/env bash
# fetch_android_prebuilts.sh
#
# Pull the LFS-tracked Android arm64 prebuilt accelerator shlibs from
# upstream google-ai-edge/LiteRT-LM into prebuilt/android_arm64/. This
# branch was imported without the LFS objects, so temp_litert.sh fails
# at adb-push time with "missing GPU accelerator shlibs".
#
# Files fetched (mirror of upstream prebuilt/android_arm64/):
#   libLiteRtGpuAccelerator.so       -- generic GPU dispatch entry point
#   libLiteRtOpenClAccelerator.so    -- OpenCL backend impl (Adreno path)
#   libLiteRtTopKOpenClSampler.so    -- top-k sampler running on OpenCL
#   libLiteRtTopKWebGpuSampler.so    -- top-k sampler running on WebGPU
#   libLiteRtWebGpuAccelerator.so    -- WebGPU backend impl
#   libGemmaModelConstraintProvider.so -- Gemma-specific decoding constraints
#
# Usage:
#   ./fetch_android_prebuilts.sh          # default: from upstream main
#   UPSTREAM_REF=v0.8.0 ./fetch_android_prebuilts.sh
#
# Env overrides:
#   UPSTREAM_REPO   default: google-ai-edge/LiteRT-LM
#   UPSTREAM_REF    default: main
#   PREBUILT_DIR    default: prebuilt/android_arm64
#   FORCE           set to 1 to re-download even if files already exist
#
# Run with `bash`, NOT `sh`.

set -e
set -u
set -o pipefail

if [ -z "${BASH_VERSION:-}" ]; then
  echo "[fetch_android_prebuilts.sh] This script requires bash, not sh/dash."
  exit 2
fi

UPSTREAM_REPO="${UPSTREAM_REPO:-google-ai-edge/LiteRT-LM}"
UPSTREAM_REF="${UPSTREAM_REF:-main}"
PREBUILT_DIR="${PREBUILT_DIR:-prebuilt/android_arm64}"

FILES=(
  libLiteRtGpuAccelerator.so
  libLiteRtOpenClAccelerator.so
  libLiteRtTopKOpenClSampler.so
  libLiteRtTopKWebGpuSampler.so
  libLiteRtWebGpuAccelerator.so
  libGemmaModelConstraintProvider.so
)

mkdir -p "${PREBUILT_DIR}"

have() {
  command -v "$1" >/dev/null 2>&1
}

is_snap_binary() {
  local cmd_path
  cmd_path=$(command -v "$1" 2>/dev/null || true)
  [[ -n "${cmd_path}" ]] && [[ "${cmd_path}" == /snap/* ]]
}

# Returns 0 if the file at $1 looks like a real ELF .so (size > 100 KB and
# starts with the ELF magic 0x7f 'E' 'L' 'F'). Returns 1 otherwise.
looks_like_elf_so() {
  local f="$1"
  [ -f "${f}" ] || return 1
  local size
  size=$(stat -c '%s' "${f}" 2>/dev/null || stat -f '%z' "${f}")
  if [ "${size}" -lt 102400 ]; then
    return 1
  fi
  local magic
  magic=$(head -c 4 "${f}" 2>/dev/null | od -An -tx1 | tr -d ' \n')
  if [ "${magic}" != "7f454c46" ]; then
    return 1
  fi
  return 0
}

download_one() {
  local name="$1"
  local out="${PREBUILT_DIR}/${name}"
  # raw.githubusercontent.com 302-redirects LFS files to media.githubusercontent.com.
  # Both curl -L and wget follow redirects.
  local url="https://github.com/${UPSTREAM_REPO}/raw/${UPSTREAM_REF}/${PREBUILT_DIR}/${name}"

  if [ "${FORCE:-0}" != "1" ] && looks_like_elf_so "${out}"; then
    local size
    size=$(stat -c '%s' "${out}" 2>/dev/null || stat -f '%z' "${out}")
    echo "[fetch_android_prebuilts.sh] ${name}: already present (${size} bytes), skip."
    return 0
  fi

  echo "[fetch_android_prebuilts.sh] ${name}: fetching from ${url}"
  rm -f "${out}.part"

  local ok=0

  # --- Attempt 1: curl ---
  if [ "${ok}" = "0" ] && have curl && ! is_snap_binary curl; then
    if curl -L -f -o "${out}.part" "${url}"; then
      ok=1
    else
      echo "[fetch_android_prebuilts.sh] curl failed for ${name}, trying next fallback..."
      rm -f "${out}.part"
    fi
  fi

  # --- Attempt 2: wget ---
  if [ "${ok}" = "0" ] && have wget && ! is_snap_binary wget; then
    if wget -q -O "${out}.part" "${url}"; then
      ok=1
    else
      echo "[fetch_android_prebuilts.sh] wget failed for ${name}, trying next fallback..."
      rm -f "${out}.part"
    fi
  fi

  # --- Attempt 3: python3 urllib.request ---
  if [ "${ok}" = "0" ] && have python3; then
    if python3 - "${url}" "${out}.part" <<'PY'
import sys, urllib.request
url, out = sys.argv[1], sys.argv[2]
try:
    with urllib.request.urlopen(url) as r, open(out, "wb") as f:
        while True:
            b = r.read(1 << 20)
            if not b:
                break
            f.write(b)
except Exception as e:
    print(f"[urllib] FAILED: {e}", file=sys.stderr)
    sys.exit(2)
PY
    then
      ok=1
    else
      rm -f "${out}.part"
    fi
  fi

  if [ "${ok}" != "1" ]; then
    echo "[fetch_android_prebuilts.sh] All fetch fallbacks failed for ${name}."
    return 1
  fi

  if ! looks_like_elf_so "${out}.part"; then
    local size
    size=$(stat -c '%s' "${out}.part" 2>/dev/null || stat -f '%z' "${out}.part")
    echo "[fetch_android_prebuilts.sh] ${name}: downloaded ${size} bytes but does not"
    echo "    look like an ELF shared object. First 256 bytes:"
    head -c 256 "${out}.part" || true
    echo ""
    echo "[fetch_android_prebuilts.sh] Likely an LFS pointer or HTML error page."
    echo "    The upstream file may be missing on ref '${UPSTREAM_REF}', or LFS"
    echo "    bandwidth/quota is exhausted. Try a tagged release instead:"
    echo "        UPSTREAM_REF=v0.8.0 ./fetch_android_prebuilts.sh"
    rm -f "${out}.part"
    return 1
  fi

  mv "${out}.part" "${out}"
  local size
  size=$(stat -c '%s' "${out}" 2>/dev/null || stat -f '%z' "${out}")
  echo "[fetch_android_prebuilts.sh] ${name}: OK (${size} bytes)"
  return 0
}

failed=()
for f in "${FILES[@]}"; do
  if ! download_one "${f}"; then
    failed+=("${f}")
  fi
done

echo ""
if [ "${#failed[@]}" -gt 0 ]; then
  echo "[fetch_android_prebuilts.sh] FAILED files:"
  for f in "${failed[@]}"; do
    echo "    - ${f}"
  done
  echo ""
  echo "Manual fetch:"
  echo "    git clone https://github.com/${UPSTREAM_REPO} /tmp/litert_lm_upstream"
  echo "    cd /tmp/litert_lm_upstream && git lfs pull --include='${PREBUILT_DIR}/*'"
  echo "    cp ${PREBUILT_DIR}/*.so $(pwd)/${PREBUILT_DIR}/"
  exit 1
fi

echo "[fetch_android_prebuilts.sh] All ${#FILES[@]} prebuilts present in ${PREBUILT_DIR}"
ls -la "${PREBUILT_DIR}"/*.so
