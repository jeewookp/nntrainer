# Generic Tensor Virtualization Plan

Paper reference: ML Drift (arXiv:2505.00232) §3.2.

> "Tensor virtualization decouples the logical representation of a tensor
> from its physical storage on the GPU. Tensors can be realized using
> various types and numbers of GPU memory objects (textures, buffers,
> 1D image buffers, 2D textures, 3D textures, texture arrays)."

## Goal

Make every GPU-resident nntrainer tensor accessible through the same
abstraction so that subsequent fused-op kernels (paper §3.6: RoPE +
Q/K/V layout; RMSNorm + residual + elementwise) can hand activations
across layer boundaries without CPU round-trips.

## Current state (as of commit `5bd21b9c`)

| Aspect | Status |
|--------|--------|
| `tv::TensorBacking` (cl_mem + cached image2d_from_buffer view) | ✅ exists |
| Encoding enum: FP32, FP16, INT8, INT4_OFFSET, INT4_2COMP | ✅ exists |
| Layout enum: ROW_MAJOR, OSV32_ISV2; PHWC4 + OHWI declared but unused | 🟡 partial |
| `ViewSpec`: 2D image only (`as_image` bool) | 🟡 hardcoded 2D |
| Usage: v8c FC weights only (`make_v8c_weight_backing`) | 🟡 weights only |
| Activations / KV / scores all use ad-hoc cl_mem in static scratches | ❌ outside abstraction |
| `nntrainer::Tensor` ↔ `TensorBacking` bridge | ❌ none |

## Target state

| Aspect | Target |
|--------|--------|
| ViewKind: BUFFER + IMAGE_1D + IMAGE_2D + IMAGE_3D | paper §3.2 |
| Layouts: ROW_MAJOR + PHWC4 (activation) + OHWI (KV) + OSV32_ISV2 (weight) | paper §3.1 §3.8 |
| Activations go through `TensorBacking` (not raw scratch) | paper §3.6 implied |
| Inter-layer handoff via `TensorBackingPool` (tensor name → backing) | needed for fused ops |
| `nntrainer::Tensor` can wrap a `TensorBacking` (zero-copy access from layer code) | bridge |

## Three-step plan (paper-ordered)

### Step 1 (this session and next): Foundation

- 1a. **Extend `cl_tensor_view.h`**: replace `as_image` bool with `ViewKind`
  enum. Add IMAGE_1D + IMAGE_3D. Add depth field to ViewSpec.
- 1b. **PHWC4 layout helpers**: compute width/height/row_pitch for
  packing a `[B, S, C]` activation tensor into a 2D image (paper §3.1).
- 1c. **OHWI layout helpers**: same for `[cache_size, dh]` K-cache.
- 1d. **`TensorBackingPool`**: process-singleton string→Backing map.
  Layers register outputs by name, consumers fetch by name. Lifetime
  tied to the model node (cleared on model release).
- 1e. **`nntrainer::Tensor` GPU-backing hook**: add a `setBacking(Backing*)`
  / `getBacking()` pair so non-FC GPU layers can move data through the
  same Tensor objects nntrainer's NetworkGraph already routes.

### Step 2: Fused RoPE + Q/K/V layout kernel (paper §3.6)

- Single OpenCL kernel takes the post-RMSNorm activation (`TensorBacking`,
  PHWC4 image2d view) plus Q/K/V weight backings (existing v8c
  TensorBackings) and produces Q/K/V in attention-input layout
  `[B·hkv, S·hq/hkv, dh]` — paper's quoted transform.
- Embeds the int8 activation quantization inline (decode case) or calls
  the separate quant kernel first (prefill case) — paper §3.7 stage-aware.
- Replaces three FC dispatches + RoPE on CPU with one kernel.

### Step 3: Fused RMSNorm + residual + elementwise (paper §3.6 Figure 4)

- Single OpenCL kernel takes the previous layer's output + the residual
  stream + the gamma vector and writes the normalized result. Output is
  a `TensorBacking` that the next fused kernel consumes.
- Eliminates the per-layer write_act → CPU NEON RMSNorm → write_act
  cycle that currently costs ~127ms (write_act) + ~13ms (RMSNorm) per
  prefill.

## Risks

1. **`nntrainer::Tensor` bridge is invasive.** The Tensor class is
   foundational; adding a GPU-backing pointer touches every consumer.
   Mitigation: opt-in (default null), only GPU layers set/check it,
   CPU layers ignore it.
2. **TensorPool integration.** TensorPool allocates host-side storage
   per-tensor. We need a parallel allocation path or a hook to allocate
   a TensorBacking instead. Mitigation: start with a separate pool
   indexed by tensor name; don't touch TensorPool.
3. **PHWC4 vs row-major for activation.** Paper §3.1 uses PHWC4. Our
   existing v8c FC kernel reads row-major activations via image2d.
   Mitigation: keep the existing v8c path intact; new PHWC4 path is
   additive for the fused kernels.

## Non-goals (deferred)

- Generic codegen / device specialization (paper §3.4) — defer.
- Migrate existing v8c FC to use ActivationBacking — defer to after the
  fused kernels are validated.
- Texture arrays, 3D textures for KV cache reshape — defer; current K/V
  layouts are 2D-friendly.

## Verification gate (per step)

| Step | Gate |
|------|------|
| 1a-c | Header compiles; ViewSpec equality + cache still works for v8c FC |
| 1d   | TensorBackingPool round-trips a backing across layer boundaries |
| 1e   | One existing GPU layer (v8c FC) exercises `setBacking` round-trip |
| 2    | Fused RoPE+QKV kernel produces bit-equivalent output vs current pipeline (relL2 < 0.5%) |
| 3    | Fused RMSNorm+residual produces bit-equivalent output |
| 2+3 e2e | Qwen3-0.6B prefill TPS ≥ 1.5× current (295 → 440+) |
