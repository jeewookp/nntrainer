#!/usr/bin/env bash
# temp_litert_list_tasks.sh
#
# Enumerates the CL kernel "task" classes that ship with the open-source
# TFLite GPU delegate under @litert. conv_cl_dump.cc proves the recipe
# works for ConvGeneric: #include the task header, instantiate the
# class, call AssembleCode() to get the CL source string. The same
# trick should work for attention IF such a task class exists -- which
# is the open question this script answers.
#
# Output:
#   - absolute path of the @litert gpu/common/tasks directory
#   - every .h in that directory that looks attention-relevant
#     (softmax, batch matmul, fused/fully-connected, SDPA, flash ...)
#   - total count of .h files so we know nothing was missed
#
# Once we see which task headers exist, we write attention_cl_dump.cc
# that instantiates the relevant class(es) and dumps their CL source.
#
# Usage:
#   sh temp_litert_list_tasks.sh
#   (no args, no prereqs beyond having bazel externals fetched -- i.e.
#    any target under @litert having been built previously, which is
#    already true if conv_cl_dump built.)

# Re-exec under bash when invoked via sh/dash -- we use arrays, [[ ]],
# and process substitution.
if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi

set -euo pipefail

# -----------------------------------------------------------------------------
# 1. Locate the @litert tasks directory.
#
# Bazel places external repos under one of several locations depending on the
# version. We try the common spots in order.
# -----------------------------------------------------------------------------
CANDIDATE_ROOTS=(
  "bazel-litert_lm/external/litert/tflite/delegates/gpu/common/tasks"
  "bazel-nntrainer/external/litert/tflite/delegates/gpu/common/tasks"
  "bazel-out/../../external/litert/tflite/delegates/gpu/common/tasks"
  "$( { bazel info output_base 2>/dev/null || true; } )/external/litert/tflite/delegates/gpu/common/tasks"
)

TASKS_DIR=""
for d in "${CANDIDATE_ROOTS[@]}"; do
  # Resolve symlinks and check existence. Some of these paths are symlinks
  # into $HOME/.cache/bazel/_bazel_*/... so we need -L.
  if [ -n "$d" ] && [ -d "$d" ]; then
    TASKS_DIR="$(cd "$d" 2>/dev/null && pwd -P)"
    break
  fi
done

# Last-resort fallback: brute-force find under bazel-* dirs and $HOME/.cache.
if [ -z "$TASKS_DIR" ]; then
  echo "[list-tasks] candidates missed, falling back to find ..."
  TASKS_DIR="$(find -L bazel-* "${HOME}/.cache/bazel" -type d \
    -path '*litert/tflite/delegates/gpu/common/tasks' 2>/dev/null \
    | head -1 || true)"
fi

if [ -z "$TASKS_DIR" ] || [ ! -d "$TASKS_DIR" ]; then
  echo "[list-tasks] ERROR: @litert tasks dir not found." >&2
  echo "  Tried:"
  for d in "${CANDIDATE_ROOTS[@]}"; do echo "    $d"; done >&2
  echo "  Make sure you have built at least one target against @litert first," >&2
  echo "  e.g.:  bazelisk build --config=android_arm64 //runtime/cl_bench:conv_cl_dump" >&2
  exit 1
fi

echo "================================================================"
echo " @litert tasks dir:"
echo "   $TASKS_DIR"
echo "================================================================"

# -----------------------------------------------------------------------------
# 2. Count total headers and list attention-relevant ones.
# -----------------------------------------------------------------------------
TOTAL_H=$(find "$TASKS_DIR" -maxdepth 1 -type f -name '*.h' | wc -l | tr -d ' ')
echo ""
echo "-- Total .h files in tasks/ : ${TOTAL_H} --"
echo ""

echo "-- Attention-relevant headers (attn|attention|flash|sdpa|softmax|batch.*matmul|fused|fully_connected) --"
find "$TASKS_DIR" -maxdepth 1 -type f -name '*.h' \
  | awk -F/ '{print $NF}' \
  | grep -iE 'attn|attention|flash|sdpa|softmax|batch.*matmul|fused|fully_connected' \
  | sort -u \
  | sed 's/^/    /' \
  || echo "    (no match)"

echo ""
echo "-- All .h in tasks/ (for reference) --"
find "$TASKS_DIR" -maxdepth 1 -type f -name '*.h' \
  | awk -F/ '{print $NF}' \
  | sort \
  | sed 's/^/    /'

# -----------------------------------------------------------------------------
# 3. Also check the sibling 'selectors' dir, which holds the factories that
#    pick a task class for a given op (e.g. SelectFullyConnected). Factories
#    reveal which concrete task class implements softmax/matmul/attention.
# -----------------------------------------------------------------------------
SEL_DIR="$(dirname "$TASKS_DIR")/selectors"
if [ -d "$SEL_DIR" ]; then
  echo ""
  echo "-- selectors/ attention-relevant .h --"
  find "$SEL_DIR" -maxdepth 1 -type f -name '*.h' \
    | awk -F/ '{print $NF}' \
    | grep -iE 'attn|attention|flash|sdpa|softmax|batch.*matmul|fused' \
    | sort -u \
    | sed 's/^/    /' \
    || echo "    (no match)"
fi
