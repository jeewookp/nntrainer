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

## Phase 3 — the weight-fetch wall (the real 1250 gap, ~930→1100+)
34% of prefill is re-reading int4 weights across M-tiles. Levers, cheapest first:
- **3a. weight buffer vs image A/B** for the big FFN GEMMs (`NNTR_V8C_BUF`):
  cheap test — wide vectorized buffer loads may beat `read_imageui` at large N.
- **3b. weight-stationary LDS GEMM** (the textbook fix, done right): a workgroup
  stages a weight K-block into local memory *once* (coalesced, N-consecutive to
  keep the winning access pattern) and streams a large M-block through it with
  double-buffering. Target: weight-fetch 468 ms → ~150 ms ⇒ ~1100–1200 tok/s.
  High effort, must be bit-identical (golden-gated). This is the make-or-break.
- **3c. vendor path** (QNN / `cl_qcom_ml_ops` matrix engine) — the paper's
  likely actual advantage; highest effort, separate toolkit. Only if 3b plateaus.

## Phase 4 — make prefill *semantically* correct (per-position RoPE, #45b)
Independent of speed: multi-token prefill currently lacks per-position RoPE, so
the generated sequence past token 1 is not real text (the golden is a stable
but not-yet-coherent reference). Implementing #45b turns the golden into a true
coherent-text oracle and makes the prefill usable end-to-end. Can be done any
time; pairs naturally with Phase 1's M=1024 probe.

## Realistic outlook
Phases 1–2 (~930 tok/s) are high-confidence and safe. **Phase 3b is the
make-or-break for 1250** on public OpenCL; if it plateaus, the remaining gap is
genuinely vendor-primitive territory (Phase 3c), which is what the paper most
likely used.
