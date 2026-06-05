# Prefill plan — catching the ML Drift paper (Gemma2-2B, Adreno 830)

Goal: lift M=1024 prefill from **~820 tok/s** (current, golden-matching vanilla
path) toward the paper's **1250 tok/s**, while every step stays **bit-correct
vs `golden_tokens.txt`**.

## Non-negotiable process: golden-gated development
Every optimization is env-gated and only promoted to default after a plain
`sh execute.sh` prints **`PASS — full token sequence matches the golden`**.
The token-185 (M=1) check alone is NOT enough — it missed that the fusions
changed the M=1024 path. The full-sequence golden + an explicit M=1024
bit-identity check (Phase 1) are the gates.

## Where the time goes (measured, M=1024)
| bucket | share | reducible? |
|---|---:|---|
| v8c **weight fetch** | ~34% | only via real data-reuse (Phase 3) |
| floor (attention 15% + FFN elementwise + dispatch) | ~43% | fusion (Phase 2) |
| v8c act fetch | ~14% | gate+up fuse, quant fuse (Phase 2) |
| dp4a compute | ~9% | no (already ~87% of dp4a peak isolated) |

Ruled out this session (measured, not guessed): faster dp4a (compute is 9%),
LWS reshape (M-heavy collapses), conv-matmul (slower than v8c), naive LDS
(can't cut cross-workgroup re-fetch at the current 16-row workgroup M-extent).

## Phase 1 — recover the fusion wins, *correctly* (≈ +9%, 820→~890)
The #82 geglu+quant, #83 post-norm+add, #80 attn-norm+quant fusions are
bit-identical for M≤21 but changed the full-mode (M=1024) output, so they were
reverted. Re-land them behind a stronger gate:
1. Add an in-process **M=1024 bit-identity probe**: run one M=1024 layer chain
   with the fusion off vs on, compare the last-row argmax / a checksum of the
   output. Must be identical.
2. Root-cause the M=1024 diff (suspect: M_pad-row vs M-row handling, or a
   stride in the fused kernels at large M). Fix so the probe AND the golden
   both pass.
3. Promote each fusion to default only after both gates pass.
Risk: low. Payoff: the already-measured +9% prefill / +33% decode.

## Phase 2 — shrink the floor (≈ +3–6%, ~890→~930), each golden-gated
- **gate+up → one concatenated-N GEMM** (read the activation once, one dispatch
  instead of two): cuts part of act-fetch + dispatch. Needs weight repack at
  load; kernel unchanged (just larger N).
- Fuse the remaining FFN elementwise / cut dispatch count.
- Attention (15%): A/B a single-pass flash-attention variant vs the current
  two-conv OHWI path; keep only if golden-PASS.

## Phase 3 — the weight-fetch wall — RESOLVED (negative): public-OpenCL ceiling
34% of prefill is re-reading int4 weights across M-tiles. Every lever was tried
and measured (all bit-identity verified via the forward-hash probe):
- **3a. weight buffer vs image** (`NNTR_V8C_BUF`): the buffer path is an
  Intel-NEO specialization and does not even run on the Adreno build. Dead.
- **3b. weight-stationary LDS GEMM** (`NNTR_V8C_LDS`, `v8c_gemm_int8_int4_lds`):
  implemented properly and **bit-identical**, swept across every block size
  (BM/BN 32–64, LWS 64–256) AND K-unroll (KU 1/4/8 to cut barriers). Result:
  **7–13× SLOWER** at every config (≈55–99 TPS vs 730). Cutting the barrier
  count (KU) did NOT help, so barriers were not the cost — the Adreno **texture
  L2 already caches the weight re-reads more cheaply than manual LDS staging**.
  LDS is a dead end here. Kept env-gated as evidence (default off).
- **3c. transposed (coalesced) weight layout** (`NNTR_V8C_WT`): the paper's
  stated "primary driver." Implemented (host-transpose [K/32 x N] -> [N x K/32],
  V8C_WCOORD kernel reads). Result: **~4% SLOWER** (706 vs 734 TPS) — the Adreno
  texture cache already handles the 2D access locality, so channel-contiguous
  weights do not help; the K-contiguous default is already better here. The
  paper's weight-layout lever does NOT transfer to our int4-dp4a texture GEMM.
- **3d. vendor path** (QNN / `cl_qcom_ml_ops`): the paper does NOT use QNN — it
  is an OpenCL/Metal/Vulkan GPU framework (verified in the PDF). Its 1250 comes
  from its full co-designed conv/texture framework + per-device offline tuning,
  NOT a single transferable trick.

**Conclusion (data-backed):** every individual GPU lever — LWS reshape, buffer
path, a full LDS-blocked GEMM (block + K-unroll sweep), conv-matmul (all_gpu),
and the paper's own weight-layout transpose — is negative on this Adreno. Our
register-tiled int4-dp4a texture GEMM is already at the practical OpenCL ceiling
(~730 clean / ~810 best real); the texture hardware's 2D-tiled caching already
does what these optimizations attempt. Matching the paper's 1250 requires
re-implementing its entire co-designed conv framework (different compute model +
per-device tuning), a multi-week dedicated effort whose building-block (weight
layout) already measured negative here — i.e. high effort, low transfer odds.

## Phase 4 — make prefill *semantically* correct (per-position RoPE, #45b)
ALREADY IMPLEMENTED: `rope_fp16_batched` applies per-position RoPE (start_pos+M)
and is wired into forward_one_layer_v2, so the vanilla generation IS the real
RoPE-correct output and `golden_tokens.txt` is a valid reference. (The fusions
change generation only via scratch-state coupling, not RoPE.)

## Realistic outlook (updated)
Public-OpenCL ceiling ≈ 730 (clean) / ~810 (best real) tok/s, reproduced and
golden-verified. The fusion wins (#82/#83/#80) are bit-identical in isolation
but couple to the generation via leftover scratch, so they are held off to keep
the golden match. **1250 is vendor-primitive territory (QNN), a separate
project** — not achievable with the current OpenCL stack, as now proven by the
LWS / buffer / conv / LDS negative results.
