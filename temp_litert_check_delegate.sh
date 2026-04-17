#!/usr/bin/env bash
# Check how the prebuilt delegate loads OpenCL.
set -euo pipefail

DEVICE_FOLDER="${DEVICE_FOLDER:-/data/local/tmp/litert_lm}"

echo "=== Dynamic dependencies ==="
echo ""
echo "--- libLiteRtGpuAccelerator.so ---"
adb shell "readelf -d ${DEVICE_FOLDER}/libLiteRtGpuAccelerator.so 2>/dev/null | grep -iE 'NEEDED|RPATH|RUNPATH'" || echo "(readelf failed)"

echo ""
echo "--- libLiteRtOpenClAccelerator.so ---"
adb shell "readelf -d ${DEVICE_FOLDER}/libLiteRtOpenClAccelerator.so 2>/dev/null | grep -iE 'NEEDED|RPATH|RUNPATH'" || echo "(readelf failed)"

echo ""
echo "=== clCreate* symbols in OpenCL accelerator ==="
adb shell "readelf -s ${DEVICE_FOLDER}/libLiteRtOpenClAccelerator.so 2>/dev/null | grep -i clCreate" || \
adb shell "nm -D ${DEVICE_FOLDER}/libLiteRtOpenClAccelerator.so 2>/dev/null | grep -i clCreate" || \
echo "(no clCreate symbols found)"

echo ""
echo "=== dlopen/dlsym references ==="
adb shell "readelf -s ${DEVICE_FOLDER}/libLiteRtOpenClAccelerator.so 2>/dev/null | grep -iE 'dlopen|dlsym'" || echo "(none)"
adb shell "readelf -s ${DEVICE_FOLDER}/libLiteRtGpuAccelerator.so 2>/dev/null | grep -iE 'dlopen|dlsym'" || echo "(none)"

echo ""
echo "=== strings: OpenCL library paths ==="
adb shell "strings ${DEVICE_FOLDER}/libLiteRtOpenClAccelerator.so 2>/dev/null | grep -iE 'libOpenCL|vendor.*lib|opencl'" || echo "(none)"
adb shell "strings ${DEVICE_FOLDER}/libLiteRtGpuAccelerator.so 2>/dev/null | grep -iE 'libOpenCL|vendor.*lib|opencl'" || echo "(none)"

echo ""
echo "=== All NEEDED libraries ==="
echo "GpuAccelerator:"
adb shell "readelf -d ${DEVICE_FOLDER}/libLiteRtGpuAccelerator.so 2>/dev/null | grep NEEDED" || echo "(failed)"
echo "OpenClAccelerator:"
adb shell "readelf -d ${DEVICE_FOLDER}/libLiteRtOpenClAccelerator.so 2>/dev/null | grep NEEDED" || echo "(failed)"
echo ""
