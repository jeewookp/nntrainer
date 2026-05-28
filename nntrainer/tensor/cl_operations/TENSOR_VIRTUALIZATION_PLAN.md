# Generic Tensor Virtualization Plan

Paper reference: ML Drift (arXiv:2505.00232) §3.1, §3.2, §3.6, §3.7, §3.8.

> "Tensor virtualization decouples the logical representation of a tensor
> from its physical storage on the GPU, allowing tensors to be realized
> using various types and numbers of GPU memory objects (textures, buffers,
> 1D image buffers, 2D textures, 3D textures, texture arrays)."

## Goal

Close the measured 17× prefill / 14× decode gap on Qwen3-0.6B (Adreno 830)
against paper-class numbers (~4900 / ~140 TPS). Profiling shows the gap
is not kernel-level (v8c GEMM hits 87% HW peak in isolation) but
**data-flow between layers**: per-FC activation upload + quantize +
read-back + CPU RMSNorm/SwiGLU/residual round-trip. Paper §3.6 closes
this with fused-on-GPU operators; §3.8 reshapes KV cache so attention
becomes a convolution; §3.7 splits prefill/decode into different quant
kernels.

## Paper-vs-state gap (confirmed 2026-05-28)

| Paper | Claim | Our state | Gap |
|---|---|---|---|
| §3.1 | 4-element SIMD C₄ layout for activations (~20% claim) | image2d view only for v8c FC weights | PHWC4 activations not applied |
| §3.2 | Decoupled logical/physical, shaders bound at codegen | `tv::TensorBacking` exists but disconnected from `nntrainer::Tensor` | Bridge missing (Step 1e) |
| §3.6 #1 | Single kernel: Q+K+V proj + RoPE + layout `(B,1,S,hq·dh)→(B·hkv,S·hq/hkv,dh)` | 3 separate GPU FCs + RoPE on CPU (`mha_core.cpp:~2248`) | 100% missing (Step 2) |
| §3.6 #2 | Single kernel: RMSNorm + residual + elementwise | RMSNorm 100% CPU NEON, residual separate layer | 100% missing (Step 4) |
| §3.6 #3 | Auto element-wise fusion w/ FC | No fusion logic | Low priority |
| §3.7 | Prefill = dedicated quant kernel + int8 GEMM; Decode = quant **inside** op kernel | Single v8c kernel for both; M-binning only in profiling | Stage split missing (Step 5) |
| §3.8 | K: OHWI `[cache_size, dh]` (Kᵀ form); V: reversed `[dh, cache_size]` | Row-major `[B,1,S,kv_width]` (`kv_cache_manager.cpp:39–48`); helpers in tree but unused | Layout reorder missing (Step 3) |
| Residency | Implicit GPU-resident activations between fused ops | Every v8c FC output `clEnqueueReadBuffer`'d to host (`blas_kernel_interface.cpp:1033–1035`) | Enabled by Steps 1e+2+4 |

## Five-step paper-ordered plan (user-confirmed 2026-05-28)

The previous three-step formulation collapsed §3.7 (stage-aware quant)
and §3.8 (KV OHWI) into the fused-kernel steps. After direct code/paper
comparison we split them out explicitly because:

- §3.8 reorder defines the **output contract** of Step 2 (fused QKV+RoPE
  must produce data in OHWI form so attention can read it as convolution).
- §3.7 decode-quant is a separate kernel from prefill-quant per paper;
  collapsing it into Step 2 would only address prefill.

### Step 1 — Foundation (paper §3.2)

#### 1a–1d — DONE (commit `b3d395f8`)

- ViewKind enum (BUFFER / IMAGE_1D / IMAGE_2D / IMAGE_3D)
- ViewSpec with depth + slice_pitch_bytes
- Factory helpers: PHWC4 (fp16/int8), OHWI K-cache, OHWI_T V-cache
- `TensorBackingPool` singleton (name → backing)
- v8c FC weights migrated to ViewKind::IMAGE_2D

#### 1e — `nntrainer::Tensor` ↔ `TensorBacking` bridge (1 day)

- Forward-declare `nntrainer::tv::TensorBacking` in `tensor.h`.
- Add opt-in `tv::TensorBacking* backing_ = nullptr` member.
- Add `setBacking(tv::TensorBacking*)` / `getBacking()` accessors.
- CPU layers ignore (default null); GPU layers can set/read.
- **Validation:** route v8c FC's output tensor through setBacking →
  getBacking and assert pointer identity. Expect zero TPS change; the
  point is to prove the brigde compiles + survives the layer chain.

**Exit criterion:** prefill TPS within ±5% of pre-bridge baseline (282
TPS on Qwen3-0.6B / SD8 Elite).

### Step 2 — Fused RoPE + Q/K/V layout kernel (paper §3.6 #1) [2-3 weeks]

Single OpenCL kernel taking post-RMSNorm activation (PHWC4 image2d
view via `TensorBacking`) + Q/K/V weight backings, producing Q/K/V in
attention-input layout `[B·hkv, S·hq/hkv, dh]`.

**Output contract (consumed by Step 3 KV cache writer):**

- Q tensor: `[B·hkv, S·hq/hkv, dh]`, packed PHWC4-style if possible
- K tensor: `[B·hkv, S, dh]` — direct OHWI-write-friendly form
- V tensor: `[B·hkv, dh, S]` — direct OHWI_T-write-friendly form

Replaces 3 FC dispatches + separate CPU RoPE with 1 kernel.

**Exit criterion:** bit-equivalent (relL2 < 0.5%) vs current pipeline.

### Step 3 — KV cache OHWI / OHWI_T migration (paper §3.8) [1 week]

K cache stored as `[cache_size, dh]` per head (convolution-weight form);
V cache as `[dh, cache_size]` (reversed). Dynamic append (paper does not
specify) keeps a static slab + an append cursor; new tokens written at
the cursor, attention reads `[0..cursor]`.

**Replaces** `kv_cache_manager.cpp:39–48` row-major allocation. Writer
is the Step 2 kernel's K/V output. Reader is attention (still
two-1×1-conv shape, now consuming OHWI/OHWI_T directly via image2d).

**Exit criterion:** Qwen3-0.6B coherent output, attention TPS unchanged
or better vs current `NNTR_MHA_GPU=1` path (~80 prefill TPS).

### Step 4 — Fused RMSNorm + residual + elementwise (paper §3.6 #2) [2-3 weeks]

Single GPU kernel: input activation + residual stream + γ vector →
normalized PHWC4 image2d output. Output `TensorBacking` consumed by
Step 2's next-layer invocation. Eliminates current per-layer
write_act (~127 ms / prefill@282) + CPU RMSNorm (~13 ms).

**Output contract:** PHWC4 image2d activation backed by `TensorBacking`
in `TensorBackingPool`, registered under `layer_{i}_norm_out`.

**Exit criterion:** bit-equivalent vs CPU RMSNorm + residual.

### Step 5 — Stage-aware quantization (paper §3.7) [1-2 weeks]

Two distinct v8c FC code paths chosen by stage:

- **Prefill (M > 1):** keep current dedicated quant kernel → int8 GEMM
  with pre-quantized weights → dequant on output. Already what v8c
  does today; just confirm the path is well-isolated.
- **Decode (M = 1):** new kernel with activation quantization **fused
  into the FC kernel itself**. Eliminates the quant launch + scratch
  upload that today dominates decode at 10 TPS.

May also feed back to fixing the early-EOS bug in
`project_kv_int8_gpu_wip` since both touch the same decode quant path.

**Exit criterion:** Qwen3-0.6B decode TPS ≥ 30 (3× current 10 TPS).

## Cumulative TPS expectations (Qwen3-0.6B / SD8 Elite)

| After step | Prefill | Decode |
|---|---|---|
| Today (baseline) | 282 | 10 |
| Step 2 (fused QKV+RoPE) | ~700 | ~12 |
| Step 3 (OHWI KV) | ~900 | ~15 |
| Step 4 (fused RMSNorm+residual) | ~2500 | ~25 |
| Step 5 (stage-aware quant) | ~3000 | ~80 |
| Paper-scaled target | ~4900 | ~140 |

Estimates assume each step independently closes ~half of the remaining
gap in its dominant regime. We will recalibrate after each step.

## Risks (rolled up)

1. **`nntrainer::Tensor` bridge is invasive** — mitigated by opt-in
   default-null pointer; CPU layers untouched.
2. **TensorPool integration deferred** — Step 1 uses a parallel
   `TensorBackingPool` keyed by tensor name, independent of nntrainer's
   TensorPool. May need integration later when layer-graph router is
   touched.
3. **Step 2 kernel complexity** — single kernel doing 4 jobs (3 GEMMs +
   RoPE + layout). Mitigate by progressive validation: first verify
   3 GEMMs fused, then add RoPE, then add layout transform.
4. **Step 3 dynamic append semantics** — paper does not specify; we
   use static slab + cursor pattern. Watch for cache-line / image
   width-alignment hazards on Adreno.
5. **Step 5 decode kernel correctness** — fused quant inside FC is
   the same code shape that breaks today in `NNTR_KV_INT8_GPU` path.
   Land Steps 2–4 first; revisit with their TensorBacking machinery.

## Non-goals (deferred)

- Auto element-wise fusion (paper §3.6 #3) — low ROI vs Steps 2/4.
- Generic codegen / device specialization (paper §3.4).
- Texture arrays / 3D textures for KV (paper §3.2 enumeration).
- Migrating existing v8c FC weight path to PHWC4 — keep current
  IMAGE_2D + RGBA UINT32 packing; PHWC4 is for **activations**.

## Per-step verification gates

| Step | Gate |
|---|---|
| 1e | Bridge round-trip pointer-identity test passes; v8c FC e2e TPS ≥ 280 |
| 2 | Fused kernel output relL2 < 0.5% vs unfused pipeline; output backing in pool |
| 3 | Attention reads OHWI/OHWI_T KV via image2d; Qwen3-0.6B coherent |
| 4 | Fused RMSNorm+residual relL2 < 0.5%; backing registered in pool |
| 5 | Decode TPS ≥ 30; prefill regression < 5% |
| e2e | Prefill TPS ≥ 1.5× current after each of Steps 2, 4; decode ≥ 3× after Step 5 |
