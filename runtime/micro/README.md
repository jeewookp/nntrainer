# runtime/micro — matmul microbenchmark (from-scratch rewrite)

Standalone matmul-only microbenchmark for the LiteRT GPU (Adreno CL)
and CPU backends. This is the from-scratch successor to
`runtime/engine/matmul_micro_benchmark.cc`; the two binaries coexist
during the rewrite so each commit can be verified against the older
tool.

## Why a rewrite

1. **Observability.** The old binary prints per-shape latency but not
   which GPU kernel the delegate actually picked. When the CL delegate
   silently falls back (e.g. the build is missing
   `--export-dynamic-symbol=LiteRt*`, or the accelerator shlibs were
   not pushed to the device) the numbers look plausible but come from
   the CPU path. This one parses the per-op profiler summary and
   writes the convolution kernel name (e.g.
   `convolution1x1(conv_wave_memory)` or
   `convolution_int8(conv_wave_memory)`) as a CSV column so every
   row is self-verifying.

2. **Reproducibility across reruns.** Adds `--cache_dir` for the
   OpenCL program cache so the second invocation skips on-device JIT,
   and `--runtime_lib_dir` as an explicit flag instead of relying on
   `LD_LIBRARY_PATH` alone.

3. **Modularity.** The flatbuffer builder, the benchmark loop, and
   the shape roster extractor live side-by-side under `runtime/micro/`
   rather than being spread across `runtime/engine/` and
   `runtime/util/`.

## Status

| Phase | Scope | Status |
| --- | --- | --- |
| 1 | BUILD rule, flag surface, environment creation sanity check | **this commit** |
| 2 | `single_fc_builder` (.tflite flatbuffer for one FC op) | pending |
| 3 | End-to-end benchmark loop for `--dtype=fp32`, CSV output with kernel name column | pending |
| 4 | `int8_per_tensor`, `int8_chain`, `fp32_conv2d` dtypes | pending |
| 5 | `temp_matmul_bench.sh` adb runner, prebuilt shlib push | pending |
| 6 | `shape_roster` extractor (.tflite → CSV of matmul M/N/K) | pending |
| 7 | Correctness & determinism gating tests | pending |

Every phase is a self-contained commit; this README tracks which knobs
are already live vs. reserved for a later phase.

## Flags (stable across phases)

The flag surface is frozen in phase 1 so the runner script and any
downstream automation can lock in flag names today.

| Flag | Default | Consumed by |
| --- | --- | --- |
| `--backend` | `gpu` | P1 (echo), P3 (options) |
| `--dtype` | `int8_per_tensor` | P1 (echo), P2/P4 (builder) |
| `--shapes` | `""` | P3 |
| `--shapes_csv` | `""` | P3 |
| `--warmup` | `5` | P3 |
| `--iters` | `50` | P3 |
| `--profile` | `false` | P3 |
| `--csv_out` | `""` | P3 |
| `--runtime_lib_dir` | `.` | P1 (echo), P2 (env wire) |
| `--cache_dir` | `""` | P2 |
| `--seed` | `0` | P2 |

## Build (Android arm64)

```
bazel build --config=android_arm64 //runtime/micro:matmul_bench
```

## Run (Android, phase 1 smoke test)

Phase 1 only proves the binary links and creates a LiteRT environment.
No matmul is actually executed.

```
adb push bazel-bin/runtime/micro/matmul_bench /data/local/tmp/
adb shell "cd /data/local/tmp && ./matmul_bench --backend=gpu --iters=10"
```

Expected stderr:

```
[matmul_bench] phase=1-scaffolding backend=gpu dtype=int8_per_tensor ...
[matmul_bench] env ready. benchmark loop not implemented in phase 1 -- exiting 0.
```

If the env line is missing and you see a LiteRT error instead, the
most common cause is that the accelerator shlibs were not pushed to
the same directory as the binary -- same failure mode as the existing
`matmul_micro_benchmark`. The phase 5 runner script will automate that
push.
