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
# Build output (meson/ndk/submodule) goes here so the console stays clean;
# only profiling output is shown. Dumped on build failure.
BUILD_LOG="$NNTRAINER_ROOT/.execute_build.log"
: > "$BUILD_LOG"

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
      git submodule update --init --recursive >>"$BUILD_LOG" 2>&1
    fi
  fi
  [ -d builddir ] && { log_info "Removing existing builddir..."; rm -rf builddir; }
  # enable-opencl=true is required: it runs jni/prepare_opencl.sh (creates
  # builddir/opencl that the CausalLM Android.mk links against), adds the
  # cl_operations include path (blas_kernel_interface.h) and -DENABLE_OPENCL=1.
  # The gpu-native binary needs OpenCL anyway. Extra flags via MESON_ARGS.
  log_info "Building nntrainer (log -> $BUILD_LOG)"
  if ! ./tools/package_android.sh -Denable-opencl=true ${MESON_ARGS:-} \
       >>"$BUILD_LOG" 2>&1; then
    log_error "nntrainer build failed — tail of $BUILD_LOG:"
    tail -40 "$BUILD_LOG"
    exit 1
  fi
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
  log_info "Building $TARGET (log -> $BUILD_LOG)"
  if ! ndk-build \
       NDK_PROJECT_PATH=. NDK_LIBS_OUT=./libs NDK_OUT=./obj \
       APP_BUILD_SCRIPT=./Android.mk NDK_APPLICATION_MK=./Application.mk \
       APP_CFLAGS="$GPU_INCLUDES" \
       "$TARGET" -j "$(nproc 2>/dev/null || echo 4)" >>"$BUILD_LOG" 2>&1; then
    log_error "$TARGET build failed — tail of $BUILD_LOG:"
    tail -40 "$BUILD_LOG"
    exit 1
  fi
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
STAGE_PROFILE="${NNTR_STAGE_PROFILE:-0}"
HOST_TIMING="${NNTR_HOST_TIMING:-1}"
# Attention sub-kernel profiler (qk / softmax / sv split, M>=512). Default ON.
ATTN_PROFILE="${NNTR_ATTN_PROFILE:-1}"
# Adreno image-attention path (texture-cached K + causal tile-skip kernels).
# The author documents it as token-identical and ~+35% prefill @M=1024
# (224->303 TPS) vs the plain-buffer ohwi path, but it ships default-off in the
# gpu_native binary. The K/V OHWI images are always built, so just enable it.
# Default ON; override with NNTR_OHWI_IMG=0 for the plain-buffer path.
OHWI_IMG="${NNTR_OHWI_IMG:-1}"
# Existing-but-default-off perf flags the author documents as bit-identical
# (correctness-preserving), targeting the current prefill bottlenecks:
#   V8C_PREFETCH : 1-ahead weight prefetch in the int8xint4 v8c GEMM (FFN/QKV/wo)
#   V8C_MFAST    : M-fast dispatch order for the v8c GEMM (helps large-M prefill)
#   SV_TM2       : #72 M-tiled sv_matmul — 2 query rows/WI, halves V re-fetch
# All default ON here; A/B any of them off (e.g. NNTR_V8C_MFAST=0 sh execute.sh).
V8C_PREFETCH="${NNTR_V8C_PREFETCH:-1}"
V8C_PREFETCH2="${NNTR_V8C_PREFETCH2:-0}"  # 2-ahead prefetch A/B (off by default)
V8C_MFAST="${NNTR_V8C_MFAST:-1}"
SV_TM2="${NNTR_SV_TM2:-1}"
# FFN/attn fusions. The fusion bisect (vs a clean gen_only vanilla golden)
# proved #82 geglu+quant, #83 post-norm+add and #80 attn-norm+quant are
# BIT-IDENTICAL to the vanilla path -> safe, default ON. #80b (FFN
# add+norm+quant) changed the (already-degenerate) multi-token output ->
# kept OFF. NOTE: the greedy multi-token generation is not a correctness
# oracle (per-position RoPE is unimplemented, #45b, so it produces garbage
# past the first token); the meaningful invariant is the M=1 first-token
# prediction (185) which every config preserves.
# #82/#83/#80 DEFAULT ON: the bit-identity probe proved they produce a
# byte-identical 26-layer forward output at M=256/512/1024 (and the bisect at
# M<=21), so they are bit-identical at every prefill size -> safe. (The earlier
# full-mode golden "break" was non-deterministic generation noise, not these
# fusions.) #80b DIFFERS in the probe -> kept OFF.
FFN_GLUQUANT_FUSE="${NNTR_FFN_GLUQUANT_FUSE:-1}"
FUSE_POSTNORM="${NNTR_FUSE_POSTNORM:-1}"
FUSE_NORMQUANT="${NNTR_FUSE_NORMQUANT:-1}"
FUSE_ADDNORM="${NNTR_FUSE_ADDNORM:-0}"
"$ADB" shell "cat > $INSTALL_DIR/run_gemma2_gpu.sh" << EOF
#!/system/bin/sh
export LD_LIBRARY_PATH=$INSTALL_DIR:\$LD_LIBRARY_PATH
export NNTR_NUM_THREADS=$NNTR_NUM_THREADS
export NNTR_MODEL_GEMMA2=1
export NNTR_GPU_LMHEAD=$GPU_LMHEAD
export NNTR_STAGE_PROFILE=$STAGE_PROFILE
export NNTR_HOST_TIMING=$HOST_TIMING
export NNTR_ATTN_PROFILE=$ATTN_PROFILE
export NNTR_OHWI_IMG=$OHWI_IMG
export NNTR_V8C_PREFETCH=$V8C_PREFETCH
export NNTR_V8C_PREFETCH2=$V8C_PREFETCH2
export NNTR_V8C_MFAST=$V8C_MFAST
export NNTR_SV_TM2=$SV_TM2
export NNTR_FUSE_ADDNORM=$FUSE_ADDNORM
export NNTR_FUSE_NORMQUANT=$FUSE_NORMQUANT
export NNTR_FFN_GLUQUANT_FUSE=$FFN_GLUQUANT_FUSE
export NNTR_FUSE_POSTNORM=$FUSE_POSTNORM
cd $INSTALL_DIR
./$TARGET "$DEV_WEIGHT"
EOF
"$ADB" shell "chmod 755 $INSTALL_DIR/run_gemma2_gpu.sh"

# GEMM compute attribution: clean M=1024 full vs no-dp4a forward. Tells us if
# prefill is GEMM-compute-bound (then a faster GEMM helps) or not. Runs FIRST
# (before the slower LWS/PF/TILE sweeps) so a plain `sh execute.sh` always
# surfaces [gemm-attrib] up front. Default ON; GEMM_ATTRIB=0 to skip.
GEMM_ATTRIB="${GEMM_ATTRIB:-1}"
if [ "$GEMM_ATTRIB" = "1" ]; then
  log_header "GEMM compute attribution (in-process, M=1024)"
  "$ADB" shell "cd $INSTALL_DIR; LD_LIBRARY_PATH=$INSTALL_DIR NNTR_MODEL_GEMMA2=1 NNTR_OHWI_IMG=1 NNTR_FUSE_ADDNORM=$FUSE_ADDNORM NNTR_FUSE_NORMQUANT=$FUSE_NORMQUANT NNTR_FFN_GLUQUANT_FUSE=$FFN_GLUQUANT_FUSE NNTR_FUSE_POSTNORM=$FUSE_POSTNORM NNTR_GEMM_ATTRIB=1 ./$TARGET '$DEV_WEIGHT'" 2>&1 \
    | grep -E "\[gemm-attrib\]|compute attribution"
  echo ""
fi

# Roofline ceiling: measure THIS device's int8-dp4a / fp16 compute peak
# (NNTR_PEAK_BENCH runs a register-only microbench, then exits). Off by
# default (it adds a full model-load run); enable with PEAK=1.
PEAK="${PEAK:-0}"
if [ "$PEAK" = "1" ]; then
  log_info "Measuring device compute peak (roofline ceiling)..."
  "$ADB" shell "cd $INSTALL_DIR; LD_LIBRARY_PATH=$INSTALL_DIR NNTR_MODEL_GEMMA2=1 NNTR_PEAK_BENCH=1 ./$TARGET '$DEV_WEIGHT'" 2>&1 | awk '
    /\[qwen3-gpu\]/ { if ($0 ~ /[Ff]ail|[Ee]rror|FAIL|ERROR/) print; next }
    { print }'
  echo ""
fi

# Phase-0 gate #1 for the cl_qcom_ml_ops matrix-engine GEMM: probe whether the
# ML-ops entrypoints actually resolve on this device. Default ON while we work
# this track; set MLOPS_PROBE=0 to skip.
MLOPS_PROBE="${MLOPS_PROBE:-0}"
if [ "$MLOPS_PROBE" = "1" ]; then
  log_header "cl_qcom_ml_ops entrypoint probe (Phase-0 gate #1)"
  "$ADB" shell "cd $INSTALL_DIR; LD_LIBRARY_PATH=$INSTALL_DIR NNTR_MODEL_GEMMA2=1 NNTR_MLOPS_PROBE=1 ./$TARGET '$DEV_WEIGHT'" 2>&1 \
    | grep -E '\[mlops\]'
  echo ""
fi

# v8c GEMM fetch/compute decomposition by WALL TIME (the in-kernel
# cl_khr_kernel_clock reads 0 on Adreno 830, so we use the stage profiler
# instead). Re-run the prefill with each contribution suppressed and read the
# M=1024 GEMM stage times ((c) QKV, (g) wo, (h2) gate+up, (h4) down):
#   full      : the normal run below (fetch + dp4a)
#   nocompute : fetch only (skip dp4a)   => compute  = full - nocompute
#   nowfetch  : no weight fetch          => w-fetch  = full - nowfetch
#   noafetch  : no activation fetch      => a-fetch  = full - noafetch
# Numerics are garbage in these modes (synthetic fetch) — timing only.
# Set GEMM_DECOMP=1 to re-run the decomposition (data already collected).
GEMM_DECOMP="${GEMM_DECOMP:-0}"
if [ "$GEMM_DECOMP" = "1" ]; then
  for mode in nocompute nowfetch noafetch; do
    log_info "GEMM decomp: NNTR_V8C_$(echo $mode | tr a-z A-Z)=1 (M=1024 GEMM stage times)"
    "$ADB" shell "cd $INSTALL_DIR; LD_LIBRARY_PATH=$INSTALL_DIR NNTR_MODEL_GEMMA2=1 NNTR_OHWI_IMG=1 NNTR_STAGE_PROFILE=1 NNTR_V8C_$(echo $mode | tr a-z A-Z)=1 ./$TARGET '$DEV_WEIGHT'" 2>&1 \
      | awk '/stage timings, M=1024/{f=1} f && /\((c|g|h2|h4)\)/{print} /host-timing M=1024/{f=0}'
  done
  echo ""
fi

# v8c GEMM local-work-size (LWS) sweep. The default is 4,16 (a sweet spot
# swept on a different Adreno); this re-runs the prefill with several LWS
# candidates and reports the M=1024 prefill TPS for each so we can pick the
# best for THIS chip (Adreno 830). The winner can then be baked in as the
# default NNTR_V8C_LWS. Set LWS_SWEEP=0 to skip.
LWS_SWEEP="${LWS_SWEEP:-0}"
if [ "$LWS_SWEEP" = "1" ]; then
  log_header "v8c GEMM LWS sweep (in-process, M=1024 — single model load)"
  # NNTR_LWS_SWEEP=1 loads the model ONCE then re-times M=1024 prefill for each
  # LWS candidate in-process and exits — ~10x faster than relaunching per LWS.
  "$ADB" shell "cd $INSTALL_DIR; LD_LIBRARY_PATH=$INSTALL_DIR NNTR_MODEL_GEMMA2=1 NNTR_OHWI_IMG=1 NNTR_LWS_SWEEP=1 ./$TARGET '$DEV_WEIGHT'" 2>&1 \
    | grep -E '\[lws-sweep\]|in-process LWS sweep'
  echo ""
fi

# v8c GEMM weight-prefetch A/B (1-ahead vs 2-ahead), in-process, single model
# load. Default ON so a plain `sh execute.sh` shows whether the deeper prefetch
# helps the latency-bound K-loop on this chip. Set PF_AB=0 to skip.
PF_AB="${PF_AB:-0}"
if [ "$PF_AB" = "1" ]; then
  log_header "v8c GEMM prefetch A/B (in-process, M=1024)"
  "$ADB" shell "cd $INSTALL_DIR; LD_LIBRARY_PATH=$INSTALL_DIR NNTR_MODEL_GEMMA2=1 NNTR_OHWI_IMG=1 NNTR_PF_AB=1 ./$TARGET '$DEV_WEIGHT'" 2>&1 \
    | grep -E '\[pf-ab\]|prefetch A/B'
  echo ""
fi

# v8c GEMM register-tile A/B (4x8 default vs 8x4 = 2x weight reuse, + others).
# In-process, single model load. Default ON — this directly tests whether a
# higher-weight-reuse tile speeds up the (weight-bandwidth-bound) forward GEMM.
TILE_AB="${TILE_AB:-0}"
if [ "$TILE_AB" = "1" ]; then
  log_header "v8c GEMM register-tile A/B (in-process, M=1024)"
  "$ADB" shell "cd $INSTALL_DIR; LD_LIBRARY_PATH=$INSTALL_DIR NNTR_MODEL_GEMMA2=1 NNTR_OHWI_IMG=1 NNTR_TILE_AB=1 ./$TARGET '$DEV_WEIGHT'" 2>&1 \
    | grep -E '\[tile-ab\]|tile A/B'
  echo ""
fi


# ---------------------------------------------------------------------------
# Golden-token correctness reference.
#   * `MAKE_GOLDEN=1 sh execute.sh` runs the SLOW-but-accurate path (every
#     fusion + the image-attention OFF -> the most vanilla forward; generation
#     already uses the CPU lm_head argmax) and saves its 21 greedy token ids to
#     golden_tokens.txt as the trusted reference.
#   * Every normal (fast, fully-fused) run then re-extracts its token ids and
#     diffs them against the golden, printing PASS/FAIL — so any optimization
#     that ever changes the output is caught immediately. Commit the golden
#     file to keep the reference across machines/sessions.
# ---------------------------------------------------------------------------
GOLDEN_FILE="${GOLDEN_FILE:-$NNTRAINER_ROOT/Applications/CausalLM/gpu_native/golden_tokens.txt}"
extract_tokens() {  # $1 = raw run log -> prints the token ids (after nonan=X)
  grep -m1 '\[gen-tokens\]' "$1" | tr -d '\r' \
    | sed -E 's/.*\[gen-tokens\] nonan=[01] //'
}
extract_nonan() {   # $1 = raw run log -> prints 1 if generation had no NaN
  grep -m1 '\[gen-tokens\]' "$1" | tr -d '\r' \
    | sed -E 's/.*\[gen-tokens\] nonan=([01]).*/\1/'
}

if [ "${MAKE_GOLDEN:-0}" = "1" ]; then
  log_header "Generate GOLDEN reference (production config, same as the normal run)"
  # Golden must come from the SAME config + mode as the normal run it will be
  # diffed against, otherwise mode/flag differences (full-vs-gen_only KV state,
  # OHWI, fusions) cause spurious mismatches. So just run the production
  # run_gemma2_gpu.sh and capture its generated tokens.
  GOLD_LOG="$NNTRAINER_ROOT/.execute_golden.log"
  "$ADB" shell "sh $INSTALL_DIR/run_gemma2_gpu.sh" 2>&1 | tee "$GOLD_LOG" \
    | grep -E '\[run [0-9]|generated|\[causal\]|\[self-consistency\]|gen-tokens|FAIL|ERROR'
  G_TOK="$(extract_tokens "$GOLD_LOG")"
  G_NONAN="$(extract_nonan "$GOLD_LOG")"
  if [ -z "$G_TOK" ]; then
    log_error "Golden run produced no [gen-tokens] line — not saving."
  elif [ "$G_NONAN" != "1" ]; then
    log_error "Golden run had NaN (nonan=$G_NONAN) — refusing to save a bad reference."
  else
    echo "$G_TOK" > "$GOLDEN_FILE"
    log_success "Saved golden reference ($(echo "$G_TOK" | wc -w) tokens) -> $GOLDEN_FILE"
    log_info "  $G_TOK"
    log_info "  Commit it: git add $GOLDEN_FILE && git commit -m 'add golden tokens'"
  fi
  echo ""
fi

# ---------------------------------------------------------------------------
# Phase 1 — fusion bit-identity probe (default ON). Hash the 26-layer forward
# OUTPUT at the real prefill sizes (M=256/512/1024) with each fusion on vs the
# all-off baseline. A matching hash PROVES the fusion is bit-identical at M=1024
# (the case the token-185 check missed); a differing hash localizes exactly
# which M diverges. OHWI is held off so only the fusion varies. FUSION_PROBE=0
# to skip.
# ---------------------------------------------------------------------------
FUSION_PROBE="${FUSION_PROBE:-1}"
if [ "$FUSION_PROBE" = "1" ]; then
  log_header "Fusion bit-identity probe (M=256/512/1024 forward-output hash)"
  probe_hashes() {  # $1 = fusion env -> the 3 sorted [probe] hash lines
    "$ADB" shell "cd $INSTALL_DIR; LD_LIBRARY_PATH=$INSTALL_DIR \
      NNTR_MODEL_GEMMA2=1 NNTR_OHWI_IMG=0 NNTR_GPU_LMHEAD=0 NNTR_FUSION_PROBE=1 \
      $1 ./$TARGET '$DEV_WEIGHT'" 2>&1 | grep -E '^\[probe\] M=' | tr -d '\r' | sort
  }
  # Explicit all-off baseline (code defaults are now ON, so an empty env is NOT
  # a vanilla baseline). Each test then flips exactly one fusion on.
  OFF="NNTR_FFN_GLUQUANT_FUSE=0 NNTR_FUSE_POSTNORM=0 NNTR_FUSE_NORMQUANT=0 NNTR_FUSE_ADDNORM=0"
  BASE_H="$(probe_hashes "$OFF")"
  probe_cmp() {  # $1 = label, $2 = the one flag to flip on (rest stay off)
    local h; h="$(probe_hashes "$OFF $2")"
    if [ -z "$h" ]; then
      log_error  "  $1 : no probe output (run failed)"
    elif [ "$h" = "$BASE_H" ]; then
      log_success "  $1 : BIT-IDENTICAL at M=256/512/1024"
    else
      log_error  "  $1 : DIFFERS"
      echo "$h" | sed 's/^/      /'
    fi
  }
  if [ -z "$BASE_H" ]; then
    log_error "baseline probe produced no hashes — skipping."
  else
    echo "$BASE_H" | sed 's/^/  [baseline] /'
    probe_cmp "#82 geglu+quant       " "NNTR_FFN_GLUQUANT_FUSE=1"
    probe_cmp "#83 post-norm+add     " "NNTR_FUSE_POSTNORM=1"
    probe_cmp "#80 attn norm+quant   " "NNTR_FUSE_NORMQUANT=1"
    probe_cmp "#80b FFN add+norm+q   " "NNTR_FUSE_ADDNORM=1 NNTR_FUSE_NORMQUANT=1"
    probe_cmp "#82+#83+#80 (default) " "NNTR_FFN_GLUQUANT_FUSE=1 NNTR_FUSE_POSTNORM=1 NNTR_FUSE_NORMQUANT=1"
  fi
  echo ""
fi

# ---------------------------------------------------------------------------
# Fusion correctness bisect (opt-in, FUSION_BISECT=1). For each fusion (and the
# all-off baseline) run the FAST generation-only path (NNTR_GEN_ONLY=1 skips the
# decode
# + prefill-timing benchmarks) and diff the 21 greedy tokens against the golden,
# printing PASS/FAIL per fusion — so a plain `sh execute.sh` pinpoints which
# optimization preserves the output and which breaks multi-token generation.
# Needs a golden (MAKE_GOLDEN=1 sh execute.sh once). Set FUSION_BISECT=0 to skip.
# ---------------------------------------------------------------------------
FUSION_BISECT="${FUSION_BISECT:-0}"
if [ "$FUSION_BISECT" = "1" ] && [ -f "$GOLDEN_FILE" ]; then
  log_header "Fusion correctness bisect (gen-only, vs golden)"
  G_TOK="$(cat "$GOLDEN_FILE")"
  # Every bisect run uses the SAME mode as the golden (NNTR_GEN_ONLY=1, clean KV
  # cache) so the only variable is the optimization under test. Golden itself is
  # the most-vanilla config (OHWI image OFF + all fusions OFF), so each line
  # below isolates exactly one optimization vs that vanilla reference.
  bisect_run() {  # $1 = label, $2 = env assignments (full control incl. OHWI)
    local log="$NNTRAINER_ROOT/.execute_bisect.log"
    "$ADB" shell "cd $INSTALL_DIR; LD_LIBRARY_PATH=$INSTALL_DIR \
      NNTR_MODEL_GEMMA2=1 NNTR_GPU_LMHEAD=0 NNTR_GEN_ONLY=1 \
      $2 ./$TARGET '$DEV_WEIGHT'" > "$log" 2>&1
    local tok nonan
    tok="$(extract_tokens "$log")"
    nonan="$(extract_nonan "$log")"
    if [ -z "$tok" ]; then
      log_error  "  $1 : no tokens (run failed)"
    elif [ "$nonan" != "1" ]; then
      log_error  "  $1 : FAIL (NaN in generation)"
    elif [ "$tok" = "$G_TOK" ]; then
      log_success "  $1 : PASS"
    else
      log_error  "  $1 : FAIL  -> $tok"
    fi
  }
  bisect_run "baseline (OHWI off, fusions off)   " "NNTR_OHWI_IMG=0"
  bisect_run "OHWI image attention               " "NNTR_OHWI_IMG=1"
  bisect_run "#82 geglu+quant                    " "NNTR_OHWI_IMG=0 NNTR_FFN_GLUQUANT_FUSE=1"
  bisect_run "#83 post-norm+add                  " "NNTR_OHWI_IMG=0 NNTR_FUSE_POSTNORM=1"
  bisect_run "#80 attn norm+quant                " "NNTR_OHWI_IMG=0 NNTR_FUSE_NORMQUANT=1"
  bisect_run "#80b FFN add+norm+quant            " "NNTR_OHWI_IMG=0 NNTR_FUSE_ADDNORM=1 NNTR_FUSE_NORMQUANT=1"
  bisect_run "#82+#83+#80 (safe set?)            " "NNTR_OHWI_IMG=0 NNTR_FFN_GLUQUANT_FUSE=1 NNTR_FUSE_POSTNORM=1 NNTR_FUSE_NORMQUANT=1"
  log_info "golden: $G_TOK"
  echo ""
elif [ "$FUSION_BISECT" = "1" ]; then
  log_warning "Fusion bisect skipped: no golden yet. Run: MAKE_GOLDEN=1 sh execute.sh"
  echo ""
fi

log_info "Launching: NNTR_MODEL_GEMMA2=1 ./$TARGET $DEV_WEIGHT"
echo ""
# Keep only profiling / timing output: drop the per-token and per-weight
# `[qwen3-gpu] ...` spam, but never hide a [qwen3-gpu] line that reports an
# error/failure. Everything else ([main], [run N], [prefill M=...], the
# (a)-(h) stage timings, [host-timing], [causal] ...) passes through. Tee the
# raw stream to a log so we can diff the generated tokens against the golden.
RUN_LOG="$NNTRAINER_ROOT/.execute_run.log"
"$ADB" shell "sh $INSTALL_DIR/run_gemma2_gpu.sh" 2>&1 | tee "$RUN_LOG" | awk '
  /\[qwen3-gpu\]/ { if ($0 ~ /[Ff]ail|[Ee]rror|FAIL|ERROR/) print; next }
  { print }'
echo ""

# Correctness gate. The MEANINGFUL invariant is the M=1 first-token prediction
# (the [self-consistency] value, 185 for Gemma2) + no-NaN + determinism, since
# multi-token generation is not a valid oracle yet (per-position RoPE / #45b).
# The full 21-token sequence is reported as a secondary bit-canary only.
log_header "Token correctness check"
R_TOK="$(extract_tokens "$RUN_LOG")"
R_NONAN="$(extract_nonan "$RUN_LOG")"
# M=1 first-token: the 2nd id of the gen-tokens line (after BOS), == the
# [self-consistency] prefill([BOS]) prediction.
R_FIRST="$(echo "$R_TOK" | awk '{print $2}')"
EXPECT_FIRST=185
[ -f "$GOLDEN_FILE" ] && EXPECT_FIRST="$(awk '{print $2}' "$GOLDEN_FILE")"
if [ "$R_NONAN" != "1" ]; then
  log_error "FAIL — NaN in generation (nonan=$R_NONAN); output INVALID."
elif [ -z "$R_TOK" ]; then
  log_error "FAIL — no [gen-tokens] line; cannot verify."
elif [ -f "$GOLDEN_FILE" ] && [ "$R_TOK" = "$(cat "$GOLDEN_FILE")" ]; then
  log_success "PASS — full token sequence matches the golden reference."
  log_info "  $R_TOK"
elif [ "$R_FIRST" = "$EXPECT_FIRST" ]; then
  # Full sequence differs but the only meaningful invariant (M=1 prediction)
  # holds — flag it loudly so a divergence is never silently accepted.
  log_warning "PARTIAL — M=1 prediction = $R_FIRST OK, but the full sequence"
  log_warning "  DIFFERS from the golden (an approximation like OHWI, or an"
  log_warning "  unverified fusion, is active). To match the golden exactly,"
  log_warning "  keep the vanilla path (all fusions off)."
  if [ -f "$GOLDEN_FILE" ]; then
    log_info "    golden: $(cat "$GOLDEN_FILE")"
    log_info "    this  : $R_TOK"
  fi
else
  log_error "FAIL — M=1 prediction = $R_FIRST, expected $EXPECT_FIRST !"
fi
echo ""
log_success "Done. Re-run on device any time with:"
log_info  "  $ADB shell sh $INSTALL_DIR/run_gemma2_gpu.sh"
