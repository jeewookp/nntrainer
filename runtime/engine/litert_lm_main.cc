// Copyright 2025 The ODML Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// ODML pipeline to execute or benchmark LLM graph on device.
//
// The pipeline does the following
// 1) Read the corresponding parameters, weight and model file paths.
// 2) Construct a graph model with the setting.
// 3) Execute model inference and generate the output.

#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <variant>

#include "absl/base/log_severity.h"  // from @com_google_absl
#include "absl/flags/flag.h"  // from @com_google_absl
#include "absl/flags/parse.h"  // from @com_google_absl
#include "absl/functional/any_invocable.h"  // from @com_google_absl
#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/log/globals.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "litert/cc/internal/scoped_file.h"  // from @litert
#include "runtime/conversation/conversation.h"
#include "runtime/conversation/io_types.h"
#include "runtime/engine/engine.h"
#include "runtime/engine/engine_factory.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"
#include "runtime/engine/shared_flags.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/util/status_macros.h"

ABSL_FLAG(std::string, backend, "gpu",
          "Executor backend to use for LLM execution (cpu, gpu, etc.)");
ABSL_FLAG(std::string, model_path, "", "Model path to use for LLM execution.");
ABSL_FLAG(std::string, input_prompt, "",
          "Input prompt to use for testing LLM execution.");
ABSL_FLAG(std::string, input_prompt_file, "", "File path to the input prompt.");

namespace {

using ::litert::lm::Backend;
using ::litert::lm::Conversation;
using ::litert::lm::ConversationConfig;
using ::litert::lm::EngineSettings;
using ::litert::lm::InputData;
using ::litert::lm::Message;
using ::litert::lm::ModelAssets;
using ::nlohmann::json;

absl::AnyInvocable<void(absl::StatusOr<Message>)> CreateMessageCallback() {
  return [](absl::StatusOr<Message> message) {
    if (!message.ok()) {
      std::cout << "Error: " << message.status() << std::endl;
      return;
    }
    if (message->is_null()) {
      std::cout << std::endl << std::flush;
      return;
    }
    for (const auto& content : (*message)["content"]) {
      std::cout << content["text"].get<std::string>();
    }
    std::cout << std::flush;
  };
}

// Gets the input prompt from the command line flag or file.
std::string GetInputPrompt() {
  const std::string input_prompt = absl::GetFlag(FLAGS_input_prompt);
  const std::string input_prompt_file = absl::GetFlag(FLAGS_input_prompt_file);
  if (!input_prompt.empty() && !input_prompt_file.empty()) {
    ABSL_LOG(FATAL) << "Only one of --input_prompt and --input_prompt_file can "
                       "be specified.";
  }
  if (!input_prompt.empty()) {
    return input_prompt;
  }
  if (!input_prompt_file.empty()) {
    std::ifstream file(input_prompt_file);
    if (!file.is_open()) {
      std::cerr << "Error: Could not open file " << input_prompt_file
                << std::endl;
      return "";
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
  }
  // If no input prompt is provided, use the default prompt.
  return "What is the tallest building in the world?";
}

absl::Status MainHelper(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  // Overrides the default for FLAGS_minloglevel to error.
  absl::SetMinLogLevel(absl::LogSeverityAtLeast::kError);
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kFatal);

  const std::string model_path = absl::GetFlag(FLAGS_model_path);
  if (model_path.empty()) {
    return absl::InvalidArgumentError("Model path is empty.");
  }
  ASSIGN_OR_RETURN(ModelAssets model_assets,  // NOLINT
                   ModelAssets::Create(model_path));
  auto backend_str = absl::GetFlag(FLAGS_backend);
  ASSIGN_OR_RETURN(Backend backend,
                   litert::lm::GetBackendFromString(backend_str));
  ASSIGN_OR_RETURN(
      EngineSettings engine_settings,
      EngineSettings::CreateDefault(std::move(model_assets), backend));

  // Wire --benchmark_prefill_tokens / --benchmark_decode_tokens into the
  // engine's BenchmarkParams. The docs at docs/getting-started/build-and-run.md
  // advertise these flags on litert_lm_main:
  //
  //     ./litert_lm_main --benchmark \
  //         --benchmark_prefill_tokens=1024 \
  //         --benchmark_decode_tokens=256 \
  //         --async=false
  //
  // but the previous code path in this file was just:
  //
  //     engine_settings.GetMutableBenchmarkParams() =
  //         litert::lm::proto::BenchmarkParams();
  //
  // i.e. always reset to a default-constructed BenchmarkParams and never
  // looked at the flag values. That is why our Adreno 830 run reported
  // "Prefill Turn 1: Processed 18 tokens" (the length of the default
  // "What is the tallest building in the world?" prompt) instead of the
  // 1024-token synthetic prefill the flag is supposed to trigger -- and
  // is why our prefill TPS (280.92 t/s) comes out about 7% of Google's
  // 3808 t/s reference number: we were measuring launch overhead on an
  // 18-token prompt, not steady-state prefill throughput.
  //
  // The wiring below matches exactly what runtime/engine/litert_lm_lib.cc
  // does at lines 596-599 for the advanced main path:
  //
  //     litert::lm::proto::BenchmarkParams benchmark_params;
  //     benchmark_params.set_num_prefill_tokens(
  //         settings.benchmark_prefill_tokens);
  //     benchmark_params.set_num_decode_tokens(
  //         settings.benchmark_decode_tokens);
  //     engine_settings.GetMutableBenchmarkParams() = benchmark_params;
  //
  // When both flags are 0 (their defaults), this produces a BenchmarkParams
  // with num_prefill_tokens=0 and num_decode_tokens=0, which is byte-
  // equivalent to the old default-constructed BenchmarkParams -- so the
  // existing "just run the prompt and print BenchmarkInfo" behavior is
  // preserved.
  {
    litert::lm::proto::BenchmarkParams benchmark_params;
    benchmark_params.set_num_prefill_tokens(
        absl::GetFlag(FLAGS_benchmark_prefill_tokens));
    benchmark_params.set_num_decode_tokens(
        absl::GetFlag(FLAGS_benchmark_decode_tokens));
    engine_settings.GetMutableBenchmarkParams() = std::move(benchmark_params);
  }

  // When --benchmark_prefill_tokens > 0, the synthetic-prefill path in
  // session_utils.cc:60-72 resizes the input token vector to
  // benchmark_prefill_token_count, which means (a) max_num_tokens on the
  // main executor has to be large enough to hold prefill + decode, and
  // (b) prefill_batch_sizes has to contain benchmark_prefill_tokens so
  // the compiled model selects the matching prefill_<N> signature (our
  // Gemma 4 E2B model ships prefill_128 and prefill_1024 as visible in
  // the error.txt magic_number_utils log). Without these two adjustments
  // the engine either errors out with "max_num_tokens too small" or
  // silently falls back to prefill_128 and halves the reported TPS.
  //
  // This mirrors litert_lm_advanced_main.cc:259-267.
  const int benchmark_prefill_tokens =
      absl::GetFlag(FLAGS_benchmark_prefill_tokens);
  const int benchmark_decode_tokens =
      absl::GetFlag(FLAGS_benchmark_decode_tokens);
  if (absl::GetFlag(FLAGS_benchmark) && benchmark_prefill_tokens > 0) {
    auto& main_executor_settings =
        engine_settings.GetMutableMainExecutorSettings();
    if (main_executor_settings.GetMaxNumTokens() == 0 &&
        benchmark_decode_tokens > 0) {
      main_executor_settings.SetMaxNumTokens(benchmark_prefill_tokens +
                                             benchmark_decode_tokens);
    }
    litert::lm::AdvancedSettings advanced_settings;
    if (main_executor_settings.GetAdvancedSettings().has_value()) {
      advanced_settings = *main_executor_settings.GetAdvancedSettings();
    }
    if (advanced_settings.prefill_batch_sizes.empty()) {
      advanced_settings.prefill_batch_sizes.insert(benchmark_prefill_tokens);
    }
    main_executor_settings.SetAdvancedSettings(advanced_settings);
  }

  // Propagate --enable_op_profiling into the AdvancedSettings carried by the
  // main executor. The GPU backend path in llm_executor_settings_utils.cc
  // consults this to call runtime_options.SetEnableProfiling(true) at
  // CompiledModel::Create time, which wires in the tflite::Profiler
  // infrastructure inside the compiled model. The executor then retrieves
  // the profiler via LiteRtCompiledModelGetProfiler and dumps a per-op
  // summary at exit.
  //
  // WARNING: on Adreno 830 OpenCL, enabling this corrupts decode output
  // (see temp_litert.sh OP_PROFILING env var and error.txt). Until the
  // delegate bug is understood, keep this off for correctness runs and
  // only flip it on when you want the per-op profile table.
  if (absl::GetFlag(FLAGS_enable_op_profiling)) {
    auto& main_executor_settings =
        engine_settings.GetMutableMainExecutorSettings();
    litert::lm::AdvancedSettings advanced_settings;
    if (main_executor_settings.GetAdvancedSettings().has_value()) {
      advanced_settings = *main_executor_settings.GetAdvancedSettings();
    }
    advanced_settings.enable_op_profiling = true;
    main_executor_settings.SetAdvancedSettings(advanced_settings);
  }

  // Create the engine.
  ASSIGN_OR_RETURN(auto engine, litert::lm::EngineFactory::CreateAny(
                                    std::move(engine_settings)));

  // Create the conversation.
  std::unique_ptr<Conversation> conversation;
  auto session_config = litert::lm::SessionConfig::CreateDefault();
  ASSIGN_OR_RETURN(auto conversation_config,
                   ConversationConfig::Builder()
                       .SetSessionConfig(session_config)
                       .Build(*engine));
  ASSIGN_OR_RETURN(conversation,
                   Conversation::Create(*engine, conversation_config));

  // Prepare the message to send.
  json content_list = json::array();
  const std::string input_prompt = GetInputPrompt();
  std::cout << "input_prompt: " << input_prompt << std::endl;
  content_list.push_back({{"type", "text"}, {"text", input_prompt}});

  // Send the message and wait for the response, asynchronously log the
  // response.
  RETURN_IF_ERROR(conversation->SendMessageAsync(
      json::object({{"role", "user"}, {"content", content_list}}),
      CreateMessageCallback()));
  RETURN_IF_ERROR(engine->WaitUntilDone(absl::Minutes(10)));

  // Print the benchmark info.
  auto benchmark_info = conversation->GetBenchmarkInfo();
  std::cout << std::endl << *benchmark_info << std::endl;
  return absl::OkStatus();
}

}  // namespace

int main(int argc, char** argv) {
  ABSL_CHECK_OK(MainHelper(argc, argv));
  return 0;
}
