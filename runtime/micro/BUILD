# Copyright 2026 The ODML Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# runtime/micro -- standalone matmul microbenchmark, successor to
# runtime/engine/matmul_micro_benchmark.cc. See matmul_bench.cc for the
# rationale. Phase 1 of the rewrite only ships the scaffolding binary
# and its linker plumbing; the flatbuffer builder, benchmark loop, and
# shape roster land in follow-up phases.

# [Google-internal load of `cc_binary`]
# [Google-internal load of `cc_library`]

package(
    default_hdrs_check = "strict",
    default_visibility = [
        "//visibility:public",
    ],
)

licenses(["notice"])

# Primary binary target. The dynamic-symbol export block is identical
# to runtime/engine:matmul_micro_benchmark's because LiteRT's
# accelerator shlibs dlopen() the main executable and look up LiteRt*
# symbols at load time -- without --export-dynamic-symbol=LiteRt* the
# GPU delegate registers but never dispatches, and every run silently
# falls back to the CPU path while still looking plausible in the
# latency numbers.
cc_binary(
    name = "matmul_bench",
    srcs = ["matmul_bench.cc"],
    additional_linker_inputs = select({
        "@platforms//os:windows": [
            "@litert//litert/c:windows_exported_symbols.def",
        ],
        "@platforms//os:linux": [":matmul_bench.exported_symbols"],
        "//conditions:default": [],
    }),
    linkopts = select({
        "@litert//litert:litert_link_capi_so": [],
        "@platforms//os:ios": ["-Wl,-exported_symbol,_LiteRt*"],
        "@platforms//os:macos": ["-Wl,-exported_symbol,_LiteRt*"],
        "@platforms//os:windows": [
            "/DEF:$(location @litert//litert/c:windows_exported_symbols.def)",
        ],
        "@platforms//os:linux": [
            "-Wl,--dynamic-list=$(location :matmul_bench.exported_symbols)",
        ],
        "//conditions:default": ["-Wl,--export-dynamic-symbol=LiteRt*"],
    }) + select({
        "@platforms//os:android": ["-lEGL", "-lGLESv3"],
        "//conditions:default": [],
    }),
    deps = [
        "@com_google_absl//absl/base:log_severity",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/log:globals",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "//runtime/util:single_op_tflite_builder",
    ] + select({
        "@litert//litert:litert_link_capi_so": [
            "@litert//litert/cc:litert_api_with_dynamic_runtime",
        ],
        "//conditions:default": [
            "@litert//litert/cc:litert_common",
            "@litert//litert/cc:litert_environment",
        ],
    }),
)
