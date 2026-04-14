// Copyright 2026 The ODML Authors.
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

#include "runtime/components/constrained_decoding/llguidance_schema_utils.h"

#include <string>
#include <unordered_set>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json

namespace litert::lm {
namespace {

std::string GenerateValueRule(const nlohmann::ordered_json& prop_schema) {
  // If the property has an enum, use the enum values as the rule.
  if (prop_schema.contains("enum") && prop_schema["enum"].is_array()) {
    std::vector<std::string> enum_vals;
    for (const auto& val : prop_schema["enum"]) {
      if (val.is_string()) {
        enum_vals.push_back(absl::StrFormat(R"(fc_esc_open "%s" fc_esc_close)",
                                            val.get<std::string>()));
      } else if (val.is_number() || val.is_boolean()) {
        enum_vals.push_back(absl::StrFormat(R"("%s")", val.dump()));
      }
    }
    if (!enum_vals.empty()) {
      return absl::StrFormat("(%s)", absl::StrJoin(enum_vals, " | "));
    }
  }

  // Otherwise, use the type to determine the rule.
  if (prop_schema.contains("type") && prop_schema["type"].is_string()) {
    std::string type = prop_schema["type"].get<std::string>();
    if (type == "string") return "custom_string";
    if (type == "number" || type == "integer") return "NUMBER";
    if (type == "boolean") return "BOOLEAN";
    if (type == "array") return "array";
    if (type == "object") return "object";
    if (type == "null") return "NULL";
  }

  // Fallback to generic json_value.
  return "json_value";
}

void ExtractToolProperties(const nlohmann::ordered_json& tool,
                           const std::string& tool_name,
                           std::vector<std::string>& tool_blocks,
                           std::vector<std::string>& required_props,
                           std::vector<std::string>& optional_props) {
  if (!tool.contains("parameters") || !tool["parameters"].is_object()) {
    return;
  }

  const auto& params = tool["parameters"];
  std::unordered_set<std::string> required_set;
  if (params.contains("required") && params["required"].is_array()) {
    for (const auto& req : params["required"]) {
      if (req.is_string()) {
        std::string req_str = req.get<std::string>();
        required_props.push_back(req_str);
        required_set.insert(req_str);
      }
    }
  }
  if (params.contains("properties") && params["properties"].is_object()) {
    for (const auto& [prop_name, prop_schema] : params["properties"].items()) {
      std::string pair_rule = absl::StrFormat(R"("%s" ":" %s)", prop_name,
                                              GenerateValueRule(prop_schema));

      if (required_set.contains(prop_name)) {
        tool_blocks.push_back(absl::StrFormat(R"(%s_req_%s: %s)", tool_name,
                                              prop_name, pair_rule));
      } else {
        optional_props.push_back(prop_name);
        tool_blocks.push_back(absl::StrFormat(R"(%s_opt_%s: %s)", tool_name,
                                              prop_name, pair_rule));
      }
    }
  }
}

void AppendRequiredProperties(const std::vector<std::string>& required_props,
                              const std::string& tool_name,
                              std::vector<std::string>& object_sequence) {
  for (const std::string& req : required_props) {
    if (!object_sequence.empty()) {
      object_sequence.push_back(R"(",")");
    }
    object_sequence.push_back(absl::StrFormat("%s_req_%s", tool_name, req));
  }
}

void AppendOptionalProperties(const std::vector<std::string>& optional_props,
                              const std::string& tool_name,
                              std::vector<std::string>& tool_blocks,
                              std::vector<std::string>& object_sequence) {
  if (optional_props.empty()) {
    return;
  }

  std::string opt_rule_name = absl::StrCat(tool_name, "_optional");

  std::vector<std::string> opt_pairs;
  std::vector<std::string> opt_pairs_with_comma;
  for (const std::string& opt : optional_props) {
    opt_pairs.push_back(absl::StrFormat("%s_opt_%s", tool_name, opt));
    opt_pairs_with_comma.push_back(
        absl::StrFormat(R"("," %s_opt_%s %s)", tool_name, opt, opt_rule_name));
  }
  std::string all_opts = absl::StrJoin(opt_pairs, " | ");
  std::string all_opts_with_comma = absl::StrJoin(opt_pairs_with_comma, " | ");

  if (!object_sequence.empty()) {
    object_sequence.push_back(opt_rule_name);
    tool_blocks.push_back(
        absl::StrFormat("%s: %s | \"\"", opt_rule_name, all_opts_with_comma));
  } else {
    // If there are only optional properties, the whole block is optional
    object_sequence.push_back(opt_rule_name);

    // The rule itself is: empty OR (any opt) followed by ("," any opt)*
    tool_blocks.push_back(absl::StrFormat(R"(%s: "" | (%s) ("," (%s))*)",
                                          opt_rule_name, all_opts, all_opts));
  }
}

void AppendToolRules(const nlohmann::ordered_json& tool,
                     const std::string& tool_name,
                     std::vector<std::string>& tool_blocks) {
  std::string tool_obj_rule = absl::StrCat(tool_name, "_object");

  std::vector<std::string> required_props;
  std::vector<std::string> optional_props;

  ExtractToolProperties(tool, tool_name, tool_blocks, required_props,
                        optional_props);

  std::vector<std::string> object_sequence;

  // Add required properties in strict order
  AppendRequiredProperties(required_props, tool_name, object_sequence);

  // Add optional properties logic (flexible order, duplicates allowed)
  AppendOptionalProperties(optional_props, tool_name, tool_blocks,
                           object_sequence);

  if (object_sequence.empty()) {
    tool_blocks.push_back(absl::StrFormat(R"(%s: "{" "}")", tool_obj_rule));
  } else {
    tool_blocks.push_back(absl::StrFormat(R"(%s: "{" %s "}")", tool_obj_rule,
                                          absl::StrJoin(object_sequence, " ")));
  }
}

std::string GetJsonGrammar(const LlgConstraintsOptions& options) {
  return absl::StrFormat(R"(
fc_esc_open: %s
fc_esc_close: %s

json_value: custom_string | NUMBER | BOOLEAN | NULL | object | array

custom_string: fc_esc_open /(.|\n)*/ fc_esc_close
array: "[" [json_value ("," json_value)*] "]"
object: "{" [pair ("," pair)*] "}"
pair: IDENTIFIER ":" json_value
IDENTIFIER: /[a-zA-Z_][a-zA-Z0-9_]*/

// Primitives (Standard JSON)
NUMBER: /-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?/
BOOLEAN: "true" | "false"
NULL: "null"
%%ignore /[ \t\r\n]+/)",
                         options.fc_open_quote, options.fc_close_quote);
}

std::string GetFunctionBlock(const std::vector<std::string>& tool_call_cases,
                             const LlgConstraintsOptions& options) {
  return absl::StrFormat(
      R"((fc_start (%s) fc_end)+ fc_resp
fc_start: %s
fc_end: %s
fc_resp: %s
)",
      absl::StrJoin(tool_call_cases, " | "), options.fc_code_fence_start,
      options.fc_code_fence_end, options.fc_function_response_start);
}

std::string GetTextOnlyBlock(const LlgConstraintsOptions& options) {
  return absl::StrFormat(
      R"(
FORBIDDEN_CALL : /.*%s.*/
SAFE_TEXT : /(.|\n)*/ & ~FORBIDDEN_CALL
start : SAFE_TEXT
)",
      options.fc_code_fence_start);
}

}  // namespace

absl::StatusOr<std::string> CreateLarkGrammarForTools(
    const nlohmann::ordered_json& tools, const LlgConstraintsOptions& options) {
  std::vector<std::string> tool_names;
  std::vector<std::string> tool_blocks;

  for (const auto& tool : tools) {
    if (!tool.contains("name") || !tool["name"].is_string()) {
      continue;
    }
    std::string tool_name = tool["name"].get<std::string>();
    tool_names.push_back(tool_name);
    AppendToolRules(tool, tool_name, tool_blocks);
  }

  std::string tool_union =
      absl::StrFormat(R"(TOOL_UNION: /%s/)", absl::StrJoin(tool_names, "|"));

  std::vector<std::string> tool_call_cases;
  tool_call_cases.reserve(tool_names.size());
  for (const auto& tool_name : tool_names) {
    tool_call_cases.push_back(
        absl::StrFormat(R"("call:" "%s" %s_object)", tool_name, tool_name));
  }

  std::string json_grammar = GetJsonGrammar(options);
  std::string function_block = GetFunctionBlock(tool_call_cases, options);
  std::string text_only_block = GetTextOnlyBlock(options);

  std::string start_rule;
  switch (options.constraint_mode) {
    case LlgConstraintMode::kTextOnly: {
      return text_only_block;
    }
    case LlgConstraintMode::kFunctionCallsOnly: {
      if (tool_names.empty()) {
        return absl::InvalidArgumentError(
            "No tools provided for FunctionCallsOnly mode.");
      }
      start_rule = absl::StrCat("start: ", function_block, "\n");
      break;
    }
    case LlgConstraintMode::kTextAndOrFunctionCalls: {
      if (tool_names.empty()) {
        return text_only_block;
      }
      start_rule = absl::StrFormat(
          R"(
start: TEXT_CONTENT? function_block_opt
TEXT_CONTENT: /(.|\n)+/
function_block_opt: function_block |
function_block: %s
)",
          function_block);
      break;
    }
  }

  return absl::StrCat(tool_union, "\n", absl::StrJoin(tool_blocks, "\n"), "\n",
                      json_grammar, "\n", start_rule);
}

}  // namespace litert::lm
