# Copyright 2025 The ODML Authors.
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

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

# Direct OpenCL matmul benchmark. Bypasses LiteRT entirely -- dlopen's
# libOpenCL.so on device and dispatches a tiled GEMM kernel directly.
# No LiteRT deps, no accelerator shlibs, just raw GPU compute.
cc_binary(
    name = "matmul_cl_bench",
    srcs = ["matmul_cl_bench.cc"],
    linkopts = [
        "-ldl",
    ] + select({
        "@platforms//os:android": ["-llog"],
        "//conditions:default": [],
    }),
)

# Matmul benchmark using TFLite GPU delegate's ConvGeneric kernel.
# Links the open-source TFLite GPU CL runtime directly.
cc_binary(
    name = "conv_generic_bench",
    srcs = ["conv_generic_bench.cc"],
    linkopts = select({
        "@platforms//os:android": ["-lEGL", "-lGLESv3", "-llog"],
        "//conditions:default": [],
    }),
    deps = [
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/status",
        "@litert//tflite/delegates/gpu/cl:cl_command_queue",
        "@litert//tflite/delegates/gpu/cl:cl_event",
        "@litert//tflite/delegates/gpu/cl:cl_operation",
        "@litert//tflite/delegates/gpu/cl:environment",
        "@litert//tflite/delegates/gpu/cl:opencl_wrapper",
        "@litert//tflite/delegates/gpu/cl:tensor",
        "@litert//tflite/delegates/gpu/common:data_type",
        "@litert//tflite/delegates/gpu/common:operations",
        "@litert//tflite/delegates/gpu/common:shape",
        "@litert//tflite/delegates/gpu/common:status",
        "@litert//tflite/delegates/gpu/common:tensor",
        "@litert//tflite/delegates/gpu/common/task:gpu_operation",
        "@litert//tflite/delegates/gpu/common/task:tensor_desc",
        "@litert//tflite/delegates/gpu/common/tasks:conv_generic",
    ],
)

# Dump ConvGeneric CL kernel source to .cl files for inspection/reuse.
cc_binary(
    name = "conv_cl_dump",
    srcs = ["conv_cl_dump.cc"],
    linkopts = select({
        "@platforms//os:android": ["-lEGL", "-lGLESv3", "-llog"],
        "//conditions:default": [],
    }),
    deps = [
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/status",
        "@litert//tflite/delegates/gpu/cl:cl_operation",
        "@litert//tflite/delegates/gpu/cl:environment",
        "@litert//tflite/delegates/gpu/cl:opencl_wrapper",
        "@litert//tflite/delegates/gpu/cl:tensor",
        "@litert//tflite/delegates/gpu/common:data_type",
        "@litert//tflite/delegates/gpu/common:operations",
        "@litert//tflite/delegates/gpu/common:shape",
        "@litert//tflite/delegates/gpu/common:status",
        "@litert//tflite/delegates/gpu/common/task:gpu_operation",
        "@litert//tflite/delegates/gpu/common/task:tensor_desc",
        "@litert//tflite/delegates/gpu/common/tasks:conv_generic",
    ],
)

# OpenCL interceptor: wraps libOpenCL.so to capture CL kernel sources
# from the closed-source delegate. Built as shared library, pushed to
# device as "libOpenCL.so" so the delegate loads it first.
# Benchmark the captured delegate CL kernel directly.
cc_binary(
    name = "delegate_kernel_bench",
    srcs = ["delegate_kernel_bench.cc"],
    linkopts = ["-ldl"] + select({
        "@platforms//os:android": ["-llog"],
        "//conditions:default": [],
    }),
)

cc_binary(
    name = "cl_intercept",
    srcs = ["cl_intercept.cc"],
    linkopts = ["-ldl"],
    linkshared = True,
)
