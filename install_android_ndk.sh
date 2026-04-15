#!/usr/bin/env bash
# install_android_ndk.sh
#
# One-shot installer for the Android NDK that LiteRT-LM / temp_litert.sh
# requires (r28b or newer, see docs/getting-started/build-and-run.md and
# temp_litert.sh's ANDROID_NDK_HOME guard). The previous r26d fails to
# compile crate_index__cxx-1.0.149 because its libc++ does not satisfy
# std::ranges::contiguous_range for rust::Slice::iterator.
#
# Usage:
#   ./install_android_ndk.sh                # install r28b into ~/neo/android-ndk-r28b
#   ./install_android_ndk.sh /custom/root   # install into /custom/root/android-ndk-r28b
#
# Env overrides:
#   NDK_VERSION    default: r28b
#   NDK_ROOT       default: $HOME/neo          (parent directory)
#   NDK_ZIP_CACHE  default: $HOME/.cache/litert_lm_ndk
#   NDK_SHA256     optional SHA256 of the zip to verify
#   FORCE          set to 1 to reinstall even if the target directory exists
#
# Fallback download chain (first that succeeds wins):
#   1. python3 + urllib.request   (streaming, progress)
#   2. wget                       (native, not snap-confined)
#   3. curl                       (native, not snap-confined)
#
# Run with `bash`, NOT `sh`. This script uses bash-only features.

set -e
set -u
set -o pipefail

# Reject dash.
if [ -z "${BASH_VERSION:-}" ]; then
  echo "[install_android_ndk.sh] This script requires bash, not sh/dash."
  echo "    Please run: ./install_android_ndk.sh"
  exit 2
fi

NDK_VERSION="${NDK_VERSION:-r28b}"
NDK_ROOT="${NDK_ROOT:-${1:-$HOME/neo}}"
NDK_ZIP_CACHE="${NDK_ZIP_CACHE:-$HOME/.cache/litert_lm_ndk}"
NDK_DIR="${NDK_ROOT}/android-ndk-${NDK_VERSION}"
NDK_ZIP_NAME="android-ndk-${NDK_VERSION}-linux.zip"
NDK_ZIP_PATH="${NDK_ZIP_CACHE}/${NDK_ZIP_NAME}"
NDK_URL="https://dl.google.com/android/repository/${NDK_ZIP_NAME}"

# ---- Minimum version gate ------------------------------------------------
# r20..r27 do not compile crate_index__cxx-1.0.149 (see temp_litert.sh).
case "${NDK_VERSION}" in
  r2[0-7]*)
    echo "[install_android_ndk.sh] Refusing to install NDK ${NDK_VERSION}."
    echo "    crate_index__cxx-1.0.149 will not compile against this NDK's libc++."
    echo "    Use NDK_VERSION=r28b (default) or newer."
    exit 1
    ;;
esac

mkdir -p "${NDK_ROOT}" "${NDK_ZIP_CACHE}"

have() {
  command -v "$1" >/dev/null 2>&1
}

is_snap_binary() {
  local cmd_path
  cmd_path=$(command -v "$1" 2>/dev/null || true)
  [[ -n "${cmd_path}" ]] && [[ "${cmd_path}" == /snap/* ]]
}

# ---- Already installed? --------------------------------------------------
if [ -f "${NDK_DIR}/README.md" ] && [ -f "${NDK_DIR}/source.properties" ]; then
  if [ "${FORCE:-0}" != "1" ]; then
    REV=$(grep -E '^Pkg.Revision' "${NDK_DIR}/source.properties" 2>/dev/null \
          || echo "Pkg.Revision=?")
    echo "[install_android_ndk.sh] Already installed: ${NDK_DIR}"
    echo "[install_android_ndk.sh] ${REV}"
    echo "[install_android_ndk.sh] Set FORCE=1 to reinstall."
    echo ""
    echo "Next step:"
    echo "    export ANDROID_NDK_HOME=\"${NDK_DIR}\""
    echo "    ./temp_litert.sh"
    exit 0
  fi
  echo "[install_android_ndk.sh] FORCE=1, reinstalling over ${NDK_DIR} ..."
  rm -rf "${NDK_DIR}"
fi

echo "[install_android_ndk.sh] Target:"
echo "    version : ${NDK_VERSION}"
echo "    root    : ${NDK_ROOT}"
echo "    out dir : ${NDK_DIR}"
echo "    zip     : ${NDK_ZIP_PATH}"
echo "    url     : ${NDK_URL}"
echo ""

# ---- Use cached zip if present and size is sane --------------------------
need_download=1
if [ -f "${NDK_ZIP_PATH}" ]; then
  CACHED_SIZE=$(stat -c '%s' "${NDK_ZIP_PATH}" 2>/dev/null || stat -f '%z' "${NDK_ZIP_PATH}")
  # NDK r28b Linux zip is ~800 MB. Reject anything < 500 MB as truncated.
  if [ "${CACHED_SIZE}" -gt 500000000 ]; then
    echo "[install_android_ndk.sh] Using cached zip (${CACHED_SIZE} bytes)."
    need_download=0
  else
    echo "[install_android_ndk.sh] Cached zip looks truncated (${CACHED_SIZE} bytes). Re-downloading."
    rm -f "${NDK_ZIP_PATH}"
  fi
fi

# ---- Attempt 1: python3 + urllib.request ---------------------------------
if [ "${need_download}" = "1" ] && have python3; then
  echo "[install_android_ndk.sh] Trying python3 urllib.request ..."
  if python3 - "${NDK_URL}" "${NDK_ZIP_PATH}" <<'PY'
import sys, urllib.request
url, out = sys.argv[1], sys.argv[2]
req = urllib.request.Request(url)
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
    sys.exit(2)
except Exception as e:
    print(f"[urllib] FAILED: {e}", file=sys.stderr)
    sys.exit(3)
PY
  then
    need_download=0
  else
    echo "[install_android_ndk.sh] python urllib failed, trying next fallback..."
    rm -f "${NDK_ZIP_PATH}"
  fi
fi

# ---- Attempt 2: wget -----------------------------------------------------
if [ "${need_download}" = "1" ] && have wget && ! is_snap_binary wget; then
  echo "[install_android_ndk.sh] Trying wget ..."
  if wget -O "${NDK_ZIP_PATH}" --content-on-error "${NDK_URL}"; then
    need_download=0
  else
    echo "[install_android_ndk.sh] wget failed, trying next fallback..."
    rm -f "${NDK_ZIP_PATH}"
  fi
fi

# ---- Attempt 3: curl -----------------------------------------------------
if [ "${need_download}" = "1" ] && have curl && ! is_snap_binary curl; then
  echo "[install_android_ndk.sh] Trying curl ..."
  if curl -L -f -o "${NDK_ZIP_PATH}" "${NDK_URL}"; then
    need_download=0
  else
    echo "[install_android_ndk.sh] curl failed."
    rm -f "${NDK_ZIP_PATH}"
  fi
fi

if [ "${need_download}" = "1" ]; then
  echo ""
  echo "[install_android_ndk.sh] ALL download fallbacks failed."
  echo "    Download manually from"
  echo "      https://developer.android.com/ndk/downloads#stable-downloads"
  echo "    and save to ${NDK_ZIP_PATH}, then re-run this script."
  exit 1
fi

# ---- Optional SHA256 verification ----------------------------------------
if [ -n "${NDK_SHA256:-}" ]; then
  echo "[install_android_ndk.sh] Verifying SHA256 ..."
  if have sha256sum; then
    GOT=$(sha256sum "${NDK_ZIP_PATH}" | awk '{print $1}')
  elif have shasum; then
    GOT=$(shasum -a 256 "${NDK_ZIP_PATH}" | awk '{print $1}')
  else
    echo "[install_android_ndk.sh] neither sha256sum nor shasum is available; skipping hash check."
    GOT=""
  fi
  if [ -n "${GOT}" ]; then
    if [ "${GOT}" != "${NDK_SHA256}" ]; then
      echo "[install_android_ndk.sh] SHA256 mismatch!"
      echo "    expected: ${NDK_SHA256}"
      echo "    got     : ${GOT}"
      exit 1
    fi
    echo "[install_android_ndk.sh] SHA256 OK (${GOT})"
  fi
fi

# ---- Extract -------------------------------------------------------------
if ! have unzip; then
  echo "[install_android_ndk.sh] 'unzip' is not installed."
  echo "    sudo apt install -y unzip       # Debian/Ubuntu"
  echo "    sudo dnf install -y unzip       # Fedora/RHEL"
  exit 1
fi

echo "[install_android_ndk.sh] Extracting to ${NDK_ROOT} ..."
# The zip contains a top-level android-ndk-<version> directory.
unzip -q -o "${NDK_ZIP_PATH}" -d "${NDK_ROOT}"

# ---- Verify layout -------------------------------------------------------
if [ ! -f "${NDK_DIR}/README.md" ] || [ ! -f "${NDK_DIR}/source.properties" ]; then
  echo "[install_android_ndk.sh] Extraction finished but expected files are missing:"
  echo "    ${NDK_DIR}/README.md"
  echo "    ${NDK_DIR}/source.properties"
  echo "Contents of ${NDK_ROOT}:"
  ls -la "${NDK_ROOT}" || true
  exit 1
fi

REV=$(grep -E '^Pkg.Revision' "${NDK_DIR}/source.properties" || echo "Pkg.Revision=?")
echo ""
echo "[install_android_ndk.sh] Installed: ${NDK_DIR}"
echo "[install_android_ndk.sh] ${REV}"
echo ""
echo "Next step:"
echo "    export ANDROID_NDK_HOME=\"${NDK_DIR}\""
echo "    ./temp_litert.sh"
