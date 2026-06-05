# Device-attached plan: Gemma2-2B prefill ~730 → 1250 tok/s (Adreno 830)

Assumes a **device-attached, fast-iteration** setup: I can build, flash, run,
and **profile** on the Adreno 830 in a tight loop (minutes, not a human-relayed
round-trip). That single change is what makes the paper's approach tractable:
every negative result so far (LWS / buffer / LDS / transpose) came from *blind*
A/Bs where we could measure end-to-end TPS but never *why* a kernel was slow.
With a profiler in the loop we diagnose the real limiter and tune like the paper
does ("per-device offline tuning").

Target gap: 730 → 1250 = **+71%**. Budget below sums to it; the GEMM is
make-or-break.

---

## Phase 0 — Tooling & measurement (1–2 days). The actual unlock.
Without this, everything is guessing. Build it first.
- **Snapdragon Profiler / Adreno GPU metrics** wired to the run: per-kernel
  occupancy, ALU vs texture(L1/L2) vs LDS utilization, stall reasons, wavefronts
  in flight, L2 hit rate, DRAM bytes. This tells us if the v8c GEMM is *really*
  L2-weight-bandwidth-bound (our hypothesis) or texture-fetch-rate / occupancy
  bound — they need different fixes.
- **Per-GEMM micro-bench harness**: isolate each FC shape (qkv/wo/gate/up/down)
  and measure GOPS + effective GB/s, hot-cache vs cold-cache, so we get a
  **roofline per kernel** and know each one's ceiling.
- **In-kernel timestamps** already exist (V8C_KCLOCK); extend to dump per-stage
  cycle + stall counters.
- **Keep the bit-identity probe + golden** as the correctness gate on every step.
Exit: a dashboard that, for any kernel, says "you are X% of the ALU roofline,
Y% of the BW roofline, occupancy Z%, top stall = …".

## Phase 1 — Diagnose the v8c GEMM's true limiter (2–3 days)
Use Phase 0 on the M=1024 prefill GEMMs. Decision tree:
- **If L2/DRAM-bandwidth bound** (weight re-reads thrash L2): the lever is real
  data reuse (Phase 2). Re-run the LDS GEMM *with the profiler* — the 7–13× blind
  slowdown almost certainly has a single diagnosable cause (LDS bank conflicts,
  occupancy collapse from LDS size, or un-hidden load latency). Fix that specific
  thing; a correctly-tuned LDS/blocked GEMM is standard and *should* win once the
  profiler shows the stall.
- **If texture-fetch-rate bound**: the OHWI/`O4×I4` packing + `read_imageui`
  width matters; sweep texel packing with the profiler (this is where the paper's
  layout win likely lives — but only measurable, not guessable).
- **If occupancy/latency bound**: tune register tile, LWS, prefetch depth against
  real occupancy numbers (our blind LWS sweep had no occupancy feedback).
Exit: a written, measured root-cause for the 34% weight-fetch + 14% act-fetch.

## Phase 2 — Crack the GEMM (the make-or-break, ~+30–40%). 1–2 weeks.
Drive the chosen lever from Phase 1 with the profiler in the loop. Two tracks,
pick by data:
- **Track A — fix the blocked/LDS GEMM** (if BW-bound): proper double-buffered,
  bank-conflict-free LDS staging tuned to the measured occupancy sweet spot.
  Target: weight-fetch 468 ms → ~150 ms.
- **Track B — conv-matmul + OHWI weight layout** (the paper's core): implement
  FC-as-1×1-conv on the texture pipeline with the `(G,SO,O4,HWD,SI,I4)` weight
  layout. all_gpu's conv was slow blindly; with the profiler, tune the conv
  kernel's tiling/threadgroup to the Adreno texture units. The paper's q8 number
  (1220, nearly == its 8/4/4 1250) proves *their* GEMM is NOT weight-fetch-bound
  — i.e. the conv+layout reads weights efficiently. Replicate that property.
Decision gate: if neither beats v8c at the roofline after profiler-guided tuning,
the GEMM is genuinely at the Adreno ceiling — stop here and bank Phases 3–4.

## Phase 3 — Operator fusion / cut the floor (43% → less, ~+10%). 3–5 days.
The floor is FFN elementwise + attention + ~hundreds of dispatches. We already
have bit-identical fused kernels (#82 geglu+quant, #83 post-norm+add, #80
norm+quant); they were correct in isolation but coupled to the generation via
leftover scratch. Device-attached fixes that fast:
- Add an **M=1024 forward bit-identity gate in CI** (the probe), promote each
  fusion only when it passes *and* the full golden matches.
- Fix the scratch-coupling root cause (zero/clear padded scratch rows so the
  generation is state-independent), then land #82/#83/#80 + fuse more (gate+up
  into one dispatch, RMSNorm into the next GEMM's epilogue). Cuts dispatch count
  and M×I round-trips.

## Phase 4 — Attention (15%, ~+7%). 3–5 days.
Profile the two-conv OHWI attention. Options, measured:
- Flash-attention-style single pass (no materialized M×N_kv scores) to cut the
  softmax round-trip; or int8 QK/SV (needs care — non-bit-identical, gate on a
  tolerance-based golden, not exact).
- The paper's KV-cache OHWI layout (§3.8) — K as OHWI(O=cache,I=dh), V reversed —
  if not already matching.

## Phase 5 — Stage-aware + per-device tuning (continuous). 3–5 days.
The paper's "adaptive kernel selection" + offline tuning. With fast iteration:
- Auto-sweep tile/LWS/packing per GEMM shape on *this* Adreno, bake the winners
  (the paper does exactly this offline). Our single global LWS leaves per-shape
  wins on the table.
- prefill→conv kernels, decode→FC kernels split (paper §3.7).

## Phase 6 — Quantization (optional, small). 2–3 days.
Sub-channel / group quant tuning (paper mentions it); we're already 8/4/4.

---

## Budget to 1250 (illustrative, measured-gated)
| phase | lever | est. |
|---|---|---:|
| 2 | GEMM (conv/layout/LDS, profiler-tuned) | 730 → ~1000–1100 |
| 3 | operator fusion (floor) | → ~1100–1180 |
| 4 | attention | → ~1180–1250 |
| 5 | per-device tuning | → 1250+ |

## Why fast iteration changes the verdict
- The blind negatives (LDS 7–13×, transpose −4%) had **no diagnosis** — we never
  saw occupancy, bank conflicts, or stall reasons. The paper's whole method is
  *measure-then-specialize per device*. With the profiler we can do that; without
  it we were guessing and (correctly) failing.
- Highest risk remains Phase 2: if profiling shows the v8c GEMM is already at the
  texture/L2 roofline with no recoverable stalls, 1250 may be unreachable even
  on GPU and the honest ceiling stands. Phase 0–1 answer that in the first week,
  cheaply, before committing to the big build.

## First week, concretely
1. Wire Snapdragon Profiler + per-GEMM roofline harness (Phase 0).
2. Root-cause the M=1024 v8c GEMM limiter (Phase 1) — re-run LDS *with* the
   profiler to see the real stall.
3. Go/no-go on Track A vs B for Phase 2 based on that single measurement.
