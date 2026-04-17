#!/usr/bin/env bash
# Restore original delegate .so files on device and clean up intercept artifacts.
set -euo pipefail

DEVICE_FOLDER="${DEVICE_FOLDER:-/data/local/tmp/litert_lm}"
PREBUILT_DIR="prebuilt/android_arm64"

echo "[restore.sh] Restoring original delegate .so files ..."

# Restore from .orig backups if they exist
for so_name in libLiteRtOpenClAccelerator.so libLiteRtGpuAccelerator.so; do
  adb shell "if [ -f ${DEVICE_FOLDER}/${so_name}.orig ]; then \
    mv ${DEVICE_FOLDER}/${so_name}.orig ${DEVICE_FOLDER}/${so_name}; \
    echo '  Restored ${so_name} from .orig backup'; \
  fi"
done

# Always re-push from prebuilt to ensure originals
for so_name in libLiteRtOpenClAccelerator.so libLiteRtGpuAccelerator.so; do
  if [ -f "${PREBUILT_DIR}/${so_name}" ]; then
    echo "  Pushing original ${so_name} from prebuilt ..."
    adb push "${PREBUILT_DIR}/${so_name}" "${DEVICE_FOLDER}/${so_name}" >/dev/null
  fi
done

# Clean up intercept artifacts
adb shell "rm -f ${DEVICE_FOLDER}/libOpenCX.so ${DEVICE_FOLDER}/libOpenCL.so 2>/dev/null" || true

echo "[restore.sh] Done."
echo ""
