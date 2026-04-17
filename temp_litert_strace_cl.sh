#!/usr/bin/env bash
# Trace which OpenCL library files the delegate actually opens.
set -euo pipefail

DEVICE_FOLDER="${DEVICE_FOLDER:-/data/local/tmp/litert_lm}"
STRACE_LOG="${DEVICE_FOLDER}/strace.log"

echo "[strace.sh] Running matmul_micro_benchmark under strace ..."
adb shell "cd ${DEVICE_FOLDER}; \
  LD_PRELOAD=./libcl_intercept.so \
  LD_LIBRARY_PATH=. \
  strace -f -e trace=openat \
    -o ${STRACE_LOG} \
  taskset f0 ./matmul_micro_benchmark \
    --backend=gpu --dtype=int8_per_tensor \
    --shapes=1024x6144x1536 --warmup=1 --iters=1 \
    --profile=false --max_tensor_mb=512" 2>&1 | tail -20

echo ""
echo "=== OpenCL / CL related file opens ==="
adb shell "grep -iE 'opencl|cl_intercept|libCL|vendor.*lib' ${STRACE_LOG}" 2>/dev/null || echo "(none found)"

echo ""
echo "=== All .so file opens ==="
adb shell "grep '\.so' ${STRACE_LOG} | grep -v ENOENT | head -40" 2>/dev/null || echo "(none)"

echo ""
echo "=== Failed .so lookups (ENOENT) with 'cl' in name ==="
adb shell "grep -iE 'cl.*\.so|\.so.*cl' ${STRACE_LOG} | grep ENOENT | head -20" 2>/dev/null || echo "(none)"

echo ""
echo "Full strace log on device: ${STRACE_LOG}"
echo ""
