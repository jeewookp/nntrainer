#!/bin/bash
#
# execute.sh — Build & run the GPU-native Gemma2-2B forward on an Adreno device.
#
# This wraps the full pipeline for the `nntrainer_qwen3_gpu` binary
# (Applications/CausalLM/gpu_native) running the Gemma2-2B model
# (NNTR_MODEL_GEMMA2=1) on a Qualcomm Adreno GPU over adb:
#
#   1. Build libnntrainer.so / libccapi-nntrainer.so for Android (arm64-v8a)
#   2. ndk-build the standalone nntrainer_qwen3_gpu executable
#   3. adb push the binary + shared libs + Gemma2 weight to the device
#   4. Run it on-device with the Gemma2-2B config selected
#
# Usage:
#   sh execute.sh [/path/to/gemma2-2b-qint4.bin]
#
# The weight path may also be given via the GEMMA2_WEIGHT env var. If NO weight
# is given, the script auto-picks the first *.bin in ~/models/gemma2 (override
# with LOCAL_WEIGHT_DIR), and if none is found there, falls back to a *.bin
# already on the device — so a plain `sh execute.sh` works once you've dropped
# the weight in ~/models/gemma2. Set PULL_WEIGHT=1 to also adb-pull a local copy.
# A real Adreno device (with vendor OpenCL) must be connected via adb,
# and ANDROID_NDK must point at your Android NDK.
#
# Missing git submodules (iniparser, ruy, OpenBLAS, minja, ...) are auto-inited
# before the build, so a fresh clone works without `git submodule update` first.
#
# Common env overrides:
#   ANDROID_NDK        (required) path to the Android NDK
#   GEMMA2_WEIGHT      local path to the Gemma2-2B QINT4 .bin weight file
#   ADB               adb binary to use            (default: adb)
#   INSTALL_DIR       on-device install directory  (default: /data/local/tmp/nntrainer/causallm)
#   NNTR_NUM_THREADS  CPU helper threads on device (default: 4)
#   REUSE_NNTR=1      reuse an existing nntrainer android builddir (skip step 1)
#   SKIP_BUILD=1      skip all building, just push + run existing artifacts
#   PULL_WEIGHT=1     adb-pull the on-device weight into ./weights/ as well
#   NNTR_GPU_LMHEAD   GPU lm_head matvec+argmax: 1=on (default), 0=CPU ref
#   NNTR_STAGE_PROFILE per-stage prefill breakdown: 1=on (default), 0=off
#   NNTR_HOST_TIMING  host-bridge stall timing:     1=on (default), 0=off
#
# Invoked as `sh execute.sh ...`? Re-exec under bash for the color logging and
# BASH_SOURCE handling below (dash lacks both).
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi
set -e

# ---------------------------------------------------------------------------
# Pretty logging
# ---------------------------------------------------------------------------
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BLUE='\033[0;34m'; NC='\033[0m'
log_info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $1"; }
log_header()  { echo -e "\n${CYAN}========================================${NC}";
                echo -e "${CYAN} $1 ${NC}";
                echo -e "${CYAN}========================================${NC}"; }
log_step()    { echo -e "\n${YELLOW}[Step $1]${NC} $2"; }

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
NNTRAINER_ROOT="$SCRIPT_DIR"
export NNTRAINER_ROOT

CAUSALLM_DIR="$NNTRAINER_ROOT/Applications/CausalLM"
JNI_DIR="$CAUSALLM_DIR/jni"
ABI="arm64-v8a"
LIBS_DIR="$JNI_DIR/libs/$ABI"

ADB="${ADB:-adb}"
INSTALL_DIR="${INSTALL_DIR:-/data/local/tmp/nntrainer/causallm}"
MODEL_DIR="$INSTALL_DIR/models/gemma2-2b"
NNTR_NUM_THREADS="${NNTR_NUM_THREADS:-4}"
TARGET="nntrainer_qwen3_gpu"

# Weight resolution order:
#   1. local file given via $1 or $GEMMA2_WEIGHT          -> pushed to the device
#   2. otherwise: first *.bin in $LOCAL_WEIGHT_DIR (~/models/gemma2 by default)
#      -> pushed to the device
#   3. otherwise: a *.bin already ON the device under the model dir, run in place
#      (set PULL_WEIGHT=1 to also adb-pull a local copy)
LOCAL_WEIGHT_DIR="${LOCAL_WEIGHT_DIR:-$HOME/models/gemma2}"
GEMMA2_WEIGHT="${1:-${GEMMA2_WEIGHT:-}}"
# No explicit weight? Auto-pick the first *.bin in the default local dir.
if [ -z "$GEMMA2_WEIGHT" ] && [ -d "$LOCAL_WEIGHT_DIR" ]; then
  GEMMA2_WEIGHT="$(ls -1 "$LOCAL_WEIGHT_DIR"/*.bin 2>/dev/null | head -1)"
fi
USE_DEVICE_WEIGHT=0   # set to 1 when we reuse a weight already on the device

log_header "GPU-native Gemma2-2B on Adreno"
log_info "NNTRAINER_ROOT : $NNTRAINER_ROOT"
log_info "Target binary  : $TARGET (NNTR_MODEL_GEMMA2=1)"
log_info "Install dir    : $INSTALL_DIR"

# ---------------------------------------------------------------------------
# Prerequisite checks
# ---------------------------------------------------------------------------
log_step "0" "Check prerequisites"

if ! command -v "$ADB" >/dev/null 2>&1; then
  log_error "adb not found (set ADB=/path/to/adb or add it to PATH)."
  exit 1
fi

if ! "$ADB" devices | awk 'NR>1 && $2=="device"{found=1} END{exit !found}'; then
  log_error "No Adreno device in 'adb devices' state. Connect a device and enable USB debugging."
  "$ADB" devices || true
  exit 1
fi
DEVICE_ID=$("$ADB" devices | awk 'NR>1 && $2=="device"{print $1; exit}')
log_info "Device         : $DEVICE_ID"

if [ -n "$GEMMA2_WEIGHT" ]; then
  # A local weight was given — it will be pushed to the device.
  [ -f "$GEMMA2_WEIGHT" ] || { log_error "Weight file not found: $GEMMA2_WEIGHT"; exit 1; }
  log_info "Weight (local) : $GEMMA2_WEIGHT ($(du -h "$GEMMA2_WEIGHT" | cut -f1))"
else
  # No local weight: look for one already on the device and use it in place.
  log_info "No local weight given — searching the device..."
  DEV_WEIGHT=$("$ADB" shell "ls -1 $MODEL_DIR/*.bin 2>/dev/null | head -1" | tr -d '\r')
  [ -z "$DEV_WEIGHT" ] && DEV_WEIGHT=$("$ADB" shell "find $INSTALL_DIR/models -name '*.bin' 2>/dev/null | head -1" | tr -d '\r')
  if [ -z "$DEV_WEIGHT" ]; then
    log_error "No .bin in $LOCAL_WEIGHT_DIR, none on the device under $INSTALL_DIR/models."
    log_info  "Put a weight in $LOCAL_WEIGHT_DIR/  (or pass one: sh execute.sh /path/to.bin)"
    exit 1
  fi
  USE_DEVICE_WEIGHT=1
  DEV_SZ=$("$ADB" shell "wc -c < '$DEV_WEIGHT'" 2>/dev/null | tr -dc '0-9')
  log_info "Weight (device): $DEV_WEIGHT (${DEV_SZ:-?} bytes) — using in place"
  if [ "${PULL_WEIGHT:-0}" = "1" ]; then
    mkdir -p "$NNTRAINER_ROOT/weights"
    LOCAL_COPY="$NNTRAINER_ROOT/weights/$(basename "$DEV_WEIGHT")"
    log_info "Pulling to $LOCAL_COPY ..."
    "$ADB" pull "$DEV_WEIGHT" "$LOCAL_COPY"
    log_success "Pulled $(basename "$DEV_WEIGHT")"
  fi
fi

if [ "${SKIP_BUILD:-0}" != "1" ] && [ -z "$ANDROID_NDK" ]; then
  log_error "ANDROID_NDK is not set (needed to build). Example: export ANDROID_NDK=/opt/android-ndk-r26d"
  log_info  "If you already have built artifacts, re-run with SKIP_BUILD=1."
  exit 1
fi
log_success "Prerequisites OK"

# ---------------------------------------------------------------------------
# Step 1: Build nntrainer for Android (libnntrainer.so, libccapi-nntrainer.so,
#         and the OpenCL loader prebuilt the gpu binary links against).
# ---------------------------------------------------------------------------
NNTR_LIB="$NNTRAINER_ROOT/builddir/android_build_result/lib/$ABI/libnntrainer.so"
OPENCL_LIB="$NNTRAINER_ROOT/builddir/opencl/lib/$ABI/libOpenCL.so"

if [ "${SKIP_BUILD:-0}" = "1" ]; then
  log_step "1" "Skip nntrainer build (SKIP_BUILD=1)"
elif [ "${REUSE_NNTR:-0}" = "1" ] && [ -f "$NNTR_LIB" ]; then
  log_step "1" "Reuse existing nntrainer android build (REUSE_NNTR=1)"
else
  log_step "1" "Build nntrainer for Android (arm64-v8a, OpenCL enabled)"
  cd "$NNTRAINER_ROOT"
  # The meson subprojects (iniparser, ruy, OpenBLAS, ...) and minja are git
  # submodules; an un-inited clone leaves them empty and the ndk build fails
  # with "iniparser.h not found". Initialise any that are missing.
  if [ -d .git ] || git rev-parse --git-dir >/dev/null 2>&1; then
    if [ -z "$(ls -A subprojects/iniparser 2>/dev/null)" ] || \
       [ -z "$(ls -A Applications/CausalLM/third_party/minja 2>/dev/null)" ]; then
      log_info "Initialising git submodules (iniparser, ruy, OpenBLAS, minja, ...)"
      git submodule update --init --recursive
    fi
  fi
  [ -d builddir ] && { log_info "Removing existing builddir..."; rm -rf builddir; }
  # enable-opencl=true is required: it runs jni/prepare_opencl.sh (creates
  # builddir/opencl that the CausalLM Android.mk links against), adds the
  # cl_operations include path (blas_kernel_interface.h) and -DENABLE_OPENCL=1.
  # The gpu-native binary needs OpenCL anyway. Extra flags via MESON_ARGS.
  ./tools/package_android.sh -Denable-opencl=true ${MESON_ARGS:-}
fi

if [ "${SKIP_BUILD:-0}" != "1" ]; then
  [ -f "$NNTR_LIB" ]   || { log_error "Missing $NNTR_LIB — nntrainer build failed."; exit 1; }
  [ -f "$OPENCL_LIB" ] || { log_error "Missing OpenCL loader prebuilt: $OPENCL_LIB
  package_android.sh must be built with enable-opencl=true."; exit 1; }
  log_success "nntrainer libraries ready"
fi

# ---------------------------------------------------------------------------
# Step 2: Build the standalone nntrainer_qwen3_gpu executable.
# ---------------------------------------------------------------------------
if [ "${SKIP_BUILD:-0}" = "1" ]; then
  log_step "2" "Skip gpu binary build (SKIP_BUILD=1)"
else
  log_step "2" "Build $TARGET (ndk-build)"
  command -v ndk-build >/dev/null 2>&1 || export PATH="$ANDROID_NDK:$PATH"
  # The CausalLM Android.mk declares a `tokenizers_c` prebuilt static lib;
  # ndk-build checks the file exists while parsing, even though the gpu-native
  # binary doesn't link it (only main.cpp + qwen3_forward.cpp). Drop in an empty
  # placeholder archive so parsing passes. (Building nntr_causallm instead would
  # need the real lib from Applications/CausalLM/build_tokenizer_android.sh.)
  TOK_LIB="$CAUSALLM_DIR/lib/libtokenizers_android_c.a"
  if [ ! -f "$TOK_LIB" ]; then
    log_info "Creating placeholder $TOK_LIB (gpu binary doesn't link the tokenizer)"
    mkdir -p "$(dirname "$TOK_LIB")"
    AR="$(command -v llvm-ar || command -v ar)"
    EMPTY_OBJ="$(mktemp --suffix=.o)" ; : > "$EMPTY_OBJ"
    "$AR" rcs "$TOK_LIB" 2>/dev/null || "$AR" rc "$TOK_LIB" "$EMPTY_OBJ" 2>/dev/null || true
    rm -f "$EMPTY_OBJ"
  fi
  # The gpu binary directly includes OpenCL-internal headers that meson does
  # NOT install (attention_kernels.h, blas_kernels.h, cl_tensor_view.h from
  # tensor/cl_operations; opencl_buffer.h from opencl/). Add those source dirs
  # via APP_CFLAGS (empty in Application.mk, so APP_CPPFLAGS is untouched).
  GPU_INCLUDES="-I$NNTRAINER_ROOT/nntrainer/tensor/cl_operations -I$NNTRAINER_ROOT/nntrainer/opencl"
  cd "$JNI_DIR"
  rm -rf libs obj
  ndk-build \
    NDK_PROJECT_PATH=. NDK_LIBS_OUT=./libs NDK_OUT=./obj \
    APP_BUILD_SCRIPT=./Android.mk NDK_APPLICATION_MK=./Application.mk \
    APP_CFLAGS="$GPU_INCLUDES" \
    "$TARGET" -j "$(nproc 2>/dev/null || echo 4)"
  log_success "$TARGET built"
fi

# ndk-build copies shared libs into libs/<abi>/ but leaves BUILD_EXECUTABLE
# output in obj/local/<abi>/. Mirror build_android.sh: copy the binary into
# libs/ so the push step finds everything in one place.
OBJ_DIR="$JNI_DIR/obj/local/$ABI"
if [ "${SKIP_BUILD:-0}" != "1" ] && [ ! -f "$LIBS_DIR/$TARGET" ] && [ -f "$OBJ_DIR/$TARGET" ]; then
  mkdir -p "$LIBS_DIR"
  cp "$OBJ_DIR/$TARGET" "$LIBS_DIR/$TARGET"
  chmod +x "$LIBS_DIR/$TARGET"
fi

if [ ! -f "$LIBS_DIR/$TARGET" ]; then
  log_error "Built binary not found in $LIBS_DIR or $OBJ_DIR (run without SKIP_BUILD first)."
  exit 1
fi

# ---------------------------------------------------------------------------
# Step 3: Push binary, shared libs and weight to the device.
# ---------------------------------------------------------------------------
log_step "3" "Push to device"
"$ADB" shell "mkdir -p $INSTALL_DIR $MODEL_DIR"

# The gpu binary needs these alongside it (LD_LIBRARY_PATH=$INSTALL_DIR).
# ndk-build does NOT stage them in libs/ for an executable target, so push each
# from its real build location. libc++_shared.so comes from the NDK sysroot.
ANDROID_LIB_DIR="$NNTRAINER_ROOT/builddir/android_build_result/lib/$ABI"
OPENCL_LIB_DIR="$NNTRAINER_ROOT/builddir/opencl/lib/$ABI"
LIBCXX="$(find "${ANDROID_NDK:-/nonexistent}" -name libc++_shared.so 2>/dev/null | grep -i aarch64 | head -1)"

push_artifact() { # <name> <candidate-paths...>
  local name="$1"; shift
  local p
  for p in "$@"; do
    if [ -f "$p" ]; then
      "$ADB" push "$p" "$INSTALL_DIR/$name" >/dev/null && log_info "pushed $name"
      return 0
    fi
  done
  log_warning "$name not found in build outputs (device vendor lib may provide it)"
  return 1
}

push_artifact "$TARGET"            "$LIBS_DIR/$TARGET" "$OBJ_DIR/$TARGET"
push_artifact libnntrainer.so      "$ANDROID_LIB_DIR/libnntrainer.so" "$LIBS_DIR/libnntrainer.so"
push_artifact libccapi-nntrainer.so "$ANDROID_LIB_DIR/libccapi-nntrainer.so" "$LIBS_DIR/libccapi-nntrainer.so"
push_artifact libOpenCL.so         "$OPENCL_LIB_DIR/libOpenCL.so" "$LIBS_DIR/libOpenCL.so"
push_artifact libc++_shared.so     "$LIBS_DIR/libc++_shared.so" "$OBJ_DIR/libc++_shared.so" "$LIBCXX"
"$ADB" shell "chmod 755 $INSTALL_DIR/$TARGET"

# Weight: reuse the on-device one, or push the local one (skipping the push
# when an identical-size copy is already there, to avoid re-sending GBs).
if [ "$USE_DEVICE_WEIGHT" = "1" ]; then
  log_info "using on-device weight: $DEV_WEIGHT (no push)"
else
  WEIGHT_NAME="$(basename "$GEMMA2_WEIGHT")"
  DEV_WEIGHT="$MODEL_DIR/$WEIGHT_NAME"
  LOCAL_SZ=$(wc -c < "$GEMMA2_WEIGHT" | tr -d ' ')
  DEV_SZ=$("$ADB" shell "[ -f $DEV_WEIGHT ] && wc -c < $DEV_WEIGHT" 2>/dev/null | tr -dc '0-9')
  if [ "$LOCAL_SZ" = "$DEV_SZ" ]; then
    log_info "weight already on device ($WEIGHT_NAME, $LOCAL_SZ bytes) — skipping push"
  else
    log_info "pushing weight $WEIGHT_NAME ($(du -h "$GEMMA2_WEIGHT" | cut -f1))..."
    "$ADB" push "$GEMMA2_WEIGHT" "$DEV_WEIGHT" >/dev/null
  fi
fi
log_success "Device ready"

# ---------------------------------------------------------------------------
# Step 4: Run Gemma2-2B on the Adreno GPU.
# ---------------------------------------------------------------------------
log_header "Run Gemma2-2B (GPU-native, Adreno)"
# NNTR_GPU_LMHEAD routes the lm_head matvec+argmax onto the GPU. Default
# ON so a plain `sh execute.sh` exercises it; override with
# NNTR_GPU_LMHEAD=0 sh execute.sh to fall back to the CPU reference.
GPU_LMHEAD="${NNTR_GPU_LMHEAD:-1}"
# Prefill stage profiler (per-op breakdown at M=256 / M=1024). Default ON
# so a plain `sh execute.sh` always shows where prefill time goes. The
# per-stage clFinish adds overhead that slightly inflates the profiled
# M=256/M=1024 wall, but the (a)-(h) attribution is what we want.
# Override with NNTR_STAGE_PROFILE=0 sh execute.sh for clean wall timing.
STAGE_PROFILE="${NNTR_STAGE_PROFILE:-1}"
HOST_TIMING="${NNTR_HOST_TIMING:-1}"
"$ADB" shell "cat > $INSTALL_DIR/run_gemma2_gpu.sh" << EOF
#!/system/bin/sh
export LD_LIBRARY_PATH=$INSTALL_DIR:\$LD_LIBRARY_PATH
export NNTR_NUM_THREADS=$NNTR_NUM_THREADS
export NNTR_MODEL_GEMMA2=1
export NNTR_GPU_LMHEAD=$GPU_LMHEAD
export NNTR_STAGE_PROFILE=$STAGE_PROFILE
export NNTR_HOST_TIMING=$HOST_TIMING
cd $INSTALL_DIR
./$TARGET "$DEV_WEIGHT"
EOF
"$ADB" shell "chmod 755 $INSTALL_DIR/run_gemma2_gpu.sh"

log_info "Launching: NNTR_MODEL_GEMMA2=1 ./$TARGET $DEV_WEIGHT"
echo ""
"$ADB" shell "sh $INSTALL_DIR/run_gemma2_gpu.sh"
echo ""
log_success "Done. Re-run on device any time with:"
log_info  "  $ADB shell sh $INSTALL_DIR/run_gemma2_gpu.sh"
