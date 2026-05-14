// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Eunju Yang <ej.yang@samsung.com>
 *
 * @file   transformer.cpp
 * @date   10 July 2025
 * @see    https://github.com/nntrainer/nntrainer
 * @author Eunju Yang <ej.yang@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This file defines Transformer's basic actions
 */

#include <cstdio>
#include <cstring>
#include <fstream>
#include <unordered_set>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <app_context.h>
#include <engine.h>
#include <layer_context.h>
#include <model.h>
#include <tensor.h>

#include <llm_util.hpp>
#include <tokenizers_cpp.h>
#include <transformer.h>

#include <embedding_layer.h>
#include <gate_up_layer.h>
#include <mha_core.h>
#include <qkv_layer.h>
#include <qkv_norm_layer.h>
#include <rms_norm.h>
#include <swiglu.h>
#include <tie_word_embedding.h>

namespace causallm {

std::string LoadBytesFromFile(const std::string &path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open file: " + path);
  }
  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);

  std::string buffer(size, ' ');
  if (!file.read(&buffer[0], size)) {
    throw std::runtime_error("Failed to read file: " + path);
  }
  return buffer;
}

ModelType strToModelType(std::string model_type) {

  std::string model_type_lower = model_type;
  std::transform(model_type_lower.begin(), model_type_lower.end(),
                 model_type_lower.begin(),
                 [](unsigned char c) { return std::tolower(c); });

  static const std::unordered_map<std::string, ModelType> model_type_map = {
    {"model", ModelType::MODEL},
    {"causallm", ModelType::CAUSALLM},
    {"embedding", ModelType::EMBEDDING}};

  if (model_type_map.find(model_type_lower) == model_type_map.end()) {
    return ModelType::UNKNOWN;
  }

  return model_type_map.at(model_type_lower);
}

Transformer::Transformer(json &cfg, json &generation_cfg, json &nntr_cfg,
                         ModelType model_type) {

  std::string config_model_type_str = "Model";
  if (nntr_cfg.contains("model_type")) {
    config_model_type_str = nntr_cfg["model_type"].get<std::string>();
  }

  ModelType config_model_type = strToModelType(config_model_type_str);

  if (model_type != config_model_type) {
    throw std::runtime_error("model_type mismatch. Class Type: " +
                             std::to_string(static_cast<int>(model_type)) +
                             ", Config Type: " + config_model_type_str);
  }

  // Initialize the model with the provided configurations
  // This is where you would set up the model layers, parameters, etc.
  setupParameters(cfg, generation_cfg, nntr_cfg);

  // prep tokenizer (optional: may be absent when running on a different host
  // e.g. quantize tool running on Linux with an Android tokenizer_file path)
  try {
    tokenizer = tokenizers::Tokenizer::FromBlobJSON(
      LoadBytesFromFile(nntr_cfg["tokenizer_file"]));
  } catch (const std::exception &e) {
    std::cerr << "[Warning] Tokenizer not loaded: " << e.what() << "\n";
  }
};

void Transformer::setupParameters(json &cfg, json &generation_cfg,
                                  json &nntr_cfg) {

  /** Initialize nntr prameters */
  BATCH_SIZE = nntr_cfg["batch_size"].get<unsigned int>();
  MODEL_TENSOR_TYPE = nntr_cfg["model_tensor_type"].get<std::string>();
  INIT_SEQ_LEN = nntr_cfg["init_seq_len"];
  MAX_SEQ_LEN = nntr_cfg["max_seq_len"];
  NUM_TO_GENERATE = nntr_cfg["num_to_generate"];
  MODEL_TENSOR_TYPE = nntr_cfg["model_tensor_type"];
  MEMORY_SWAP = nntr_cfg.contains("fsu") ? nntr_cfg["fsu"].get<bool>() : false;
  FSU_LOOKAHEAD = nntr_cfg.contains("fsu_lookahead")
                    ? nntr_cfg["fsu_lookahead"].get<unsigned int>()
                    : 1;
  EMBEDDING_DTYPE = nntr_cfg["embedding_dtype"];
  FC_LAYER_DTYPE = nntr_cfg["fc_layer_dtype"];

  if (cfg.contains("is_causal") && !cfg["is_causal"].is_null()) {
    IS_CAUSAL = cfg["is_causal"].get<bool>();
  } else if (cfg.contains("use_bidirectional_attention") &&
             !cfg["use_bidirectional_attention"].is_null()) {
    IS_CAUSAL = !cfg["use_bidirectional_attention"].get<bool>();
  }

  auto cfg_uint = [&](const char *k, unsigned int def = 0) -> unsigned int {
    return (cfg.contains(k) && !cfg[k].is_null()) ? cfg[k].get<unsigned int>() : def;
  };
  auto cfg_int = [&](const char *k, int def = 0) -> int {
    return (cfg.contains(k) && !cfg[k].is_null()) ? cfg[k].get<int>() : def;
  };

  NUM_VOCAB = cfg_uint("vocab_size");
  DIM = cfg_int("hidden_size");
  INTERMEDIATE_SIZE = cfg_int("intermediate_size");
  NUM_LAYERS = cfg_int("num_hidden_layers");
  NUM_HEADS = cfg_int("num_attention_heads");
  HEAD_DIM = (cfg.contains("head_dim") && !cfg["head_dim"].is_null())
               ? cfg["head_dim"].get<int>()
               : DIM / NUM_HEADS;
  NUM_KEY_VALUE_HEADS = (cfg.contains("num_key_value_heads") && !cfg["num_key_value_heads"].is_null())
                          ? cfg["num_key_value_heads"].get<int>()
                          : NUM_HEADS;
  SLIDING_WINDOW =
    (cfg.contains("sliding_window") && !cfg["sliding_window"].is_null())
      ? cfg["sliding_window"].get<unsigned int>()
      : UINT_MAX;
  SLIDING_WINDOW_PATTERN = (cfg.contains("sliding_window_pattern") && !cfg["sliding_window_pattern"].is_null())
                             ? cfg["sliding_window_pattern"].get<unsigned int>()
                             : 1;
  MAX_POSITION_EMBEDDINGS =
    (cfg.contains("max_position_embeddings") &&
     !cfg["max_position_embeddings"].is_null())
      ? cfg["max_position_embeddings"].get<unsigned int>()
      : 4096;
  ROPE_THETA =
    (cfg.contains("rope_theta") && !cfg["rope_theta"].is_null())
      ? cfg["rope_theta"].get<unsigned int>()
      : 10000;
  TIE_WORD_EMBEDDINGS =
    (cfg.contains("tie_word_embeddings") && !cfg["tie_word_embeddings"].is_null())
      ? cfg["tie_word_embeddings"].get<bool>()
      : true;
  NORM_EPS = (cfg.contains("rms_norm_eps") && !cfg["rms_norm_eps"].is_null())
               ? cfg["rms_norm_eps"].get<float>()
               : 1e-6f;
  GQA_SIZE = NUM_HEADS / NUM_KEY_VALUE_HEADS;

  return;
};

void Transformer::initialize() {

  // RegisterCustomLayers
  registerCustomLayers();

  // construct causalLM model
  constructModel();

  // setup model property
  std::vector<std::string> model_props = {
    withKey("batch_size", BATCH_SIZE), withKey("epochs", "1"),
    withKey("model_tensor_type", MODEL_TENSOR_TYPE)};
  if (MEMORY_SWAP) {
    model_props.emplace_back(withKey("fsu", "true"));
    model_props.emplace_back(withKey("fsu_lookahead", FSU_LOOKAHEAD));
  }

  model->setProperty(model_props);

  if (model->compile(ml::train::ExecutionMode::INFERENCE)) {
    throw std::invalid_argument("Model compilation failed.");
  }

  if (model->initialize(ml::train::ExecutionMode::INFERENCE)) {
    throw std::invalid_argument("Model initialization failed.");
  }

  is_initialized = true;

#ifdef DEBUG
  model->summarize(std::cout, ML_TRAIN_SUMMARY_MODEL);
#endif
}

void Transformer::constructModel() {

  // layers used in the model
  std::vector<LayerHandle> layers;

  // create model
  model = ml::train::createModel(ml::train::ModelType::NEURAL_NET);

  // create input layer
  layers.push_back(createLayer(
    "input", {withKey("name", "input0"),
              withKey("input_shape", "1:1:" + std::to_string(INIT_SEQ_LEN))}));

  // create embedding layer
  const std::string embedding_type =
    TIE_WORD_EMBEDDINGS ? "tie_word_embeddings" : "embedding_layer";

  layers.push_back(createLayer(
    embedding_type,
    {"name=embedding0", "in_dim=" + std::to_string(NUM_VOCAB),
     "weight_dtype=" + EMBEDDING_DTYPE, "out_dim=" + std::to_string(DIM),
     "scale=" + std::to_string(EMBEDDING_SCALE)}));

  // create transformer layers
  for (int i = 0; i < NUM_LAYERS; ++i) {
    std::vector<LayerHandle> transformer;
    if (i == 0)
      transformer = createTransformerDecoderBlock(0, "embedding0");
    else
      transformer = createTransformerDecoderBlock(
        i, "layer" + std::to_string(i - 1) + "_decoder_output");
    layers.insert(layers.end(), transformer.begin(), transformer.end());
  }

  // create rms_norm
  layers.push_back(createLayer(
    "rms_norm",
    {withKey("name", "output_norm"),
     withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("input_layers",
             "layer" + std::to_string(NUM_LAYERS - 1) + "_decoder_output"),
     withKey("packed", "false")}));

  // add created layers into the model
  for (auto &layer : layers) {
    model->addLayer(layer);
  }
};

void Transformer::load_weight(const std::string &weight_path) {

  if (!is_initialized) {
    throw std::runtime_error(
      "Transformer model is not initialized. Please call "
      "initialize() before load_weight().");
  }

  // NeuralNetwork::load() in INFERENCE mode spawns one thread per layer
  // (~247 threads) each mmapping the full weight file concurrently.  On
  // Android this causes the LMK to SIGKILL the process before loading
  // completes.  Instead: mmap the file once (lazy, no physical pages faulted
  // in), then memcpy each weight sequentially and immediately call
  // POSIX_MADV_DONTNEED to release those file-cache pages.  At any moment
  // the resident file-cache footprint is at most one weight's size (~5 MB),
  // preventing the ~2x OOM spike that ifstream readahead caused.
  int fd = ::open(weight_path.c_str(), O_RDONLY);
  if (fd < 0)
    throw std::runtime_error("Cannot open weight file: " + weight_path);

  struct stat st{};
  if (::fstat(fd, &st) < 0) {
    ::close(fd);
    throw std::runtime_error("fstat failed for: " + weight_path);
  }
  size_t f_size = static_cast<size_t>(st.st_size);

  char *view = static_cast<char *>(
    ::mmap(nullptr, f_size, PROT_READ, MAP_PRIVATE, fd, 0));
  ::close(fd);
  if (view == MAP_FAILED)
    throw std::runtime_error("mmap failed for: " + weight_path);

  // Disable kernel read-ahead so pages are fetched on demand only.
  ::posix_madvise(view, f_size, POSIX_MADV_RANDOM);

  std::fprintf(stderr, "[DIAG] load_weight: file=%zu bytes\n", f_size);
  std::fflush(stderr);

  // Track data pointers already loaded to handle shared weights
  // (e.g. TieWordEmbedding where embedding0 and output_of_causallm
  // reference the exact same Tensor buffer).
  std::unordered_set<void *> visited;
  size_t file_pos = 0;
  size_t bytes_done = 0;

  std::exception_ptr ex;
  model->forEachLayer(
    [&](ml::train::Layer &l, nntrainer::RunLayerContext &ctx, void *) {
      if (ex) return;
      try {
        for (unsigned i = 0; i < ctx.getNumWeights(); ++i) {
          nntrainer::Tensor &w = ctx.getWeight(i);
          void *ptr = static_cast<void *>(w.getData<uint8_t>());
          if (!ptr) continue;
          if (!visited.insert(ptr).second) continue; // shared weight already read

          size_t n = w.getMemoryBytes();
          if (file_pos + n > f_size)
            throw std::runtime_error(
              "Weight '" + ctx.getWeightName(i) + "' exceeds file size");

          std::memcpy(ptr, view + file_pos, n);

          // Release file-cache pages for this weight immediately after copy.
          ::posix_madvise(view + file_pos, n, POSIX_MADV_DONTNEED);

          file_pos += n;
          bytes_done += n;
          if (bytes_done % (128UL * 1024 * 1024) < n) {
            // Read VmRSS from /proc/self/status to track physical page commits.
            size_t rss_kb = 0;
            {
              std::ifstream st("/proc/self/status");
              std::string ln;
              while (std::getline(st, ln)) {
                if (ln.find("VmRSS:") == 0) {
                  std::sscanf(ln.c_str(), "VmRSS: %zu", &rss_kb);
                  break;
                }
              }
            }
            std::fprintf(stderr,
                         "[DIAG] load_weight: %zu MB written, RSS=%zu MB\n",
                         bytes_done / 1024 / 1024, rss_kb / 1024);
            std::fflush(stderr);
          }
        }
      } catch (...) {
        ex = std::current_exception();
      }
    },
    nullptr);

  ::munmap(view, f_size);
  if (ex) std::rethrow_exception(ex);
};

void Transformer::save_weight(const std::string &weight_path) {

  if (!is_initialized) {
    throw std::runtime_error(
      "Transformer model is not initialized. Please call "
      "initialize() before save_weight().");
  }

  try {
    model->save(weight_path, ml::train::ModelFormat::MODEL_FORMAT_BIN);
  } catch (const std::exception &e) {
    throw std::runtime_error("Failed to save model weights: " +
                             std::string(e.what()));
  }
};

void Transformer::save_weight(
  const std::string &weight_path, ml::train::TensorDim::DataType dtype,
  const std::map<std::string, ml::train::TensorDim::DataType>
    &layer_dtype_map) {

  if (!is_initialized) {
    throw std::runtime_error(
      "Transformer model is not initialized. Please call "
      "initialize() before save_weight().");
  }

  try {
    model->save(weight_path, ml::train::ModelFormat::MODEL_FORMAT_BIN, dtype,
                layer_dtype_map);
  } catch (const std::exception &e) {
    throw std::runtime_error("Failed to save model weights with dtype: " +
                             std::string(e.what()));
  }
};

void Transformer::run(const WSTR prompt, bool do_sample,
                      const WSTR system_prompt, const WSTR tail_prompt,
                      bool log_output) {
  if (!is_initialized) {
    throw std::runtime_error(
      "Transformer model is not initialized. Please call "
      "initialize() before run().");
  }
  ///@note This part should be filled in.
  /// The run action can be defined by the precedent classes.
}

std::vector<LayerHandle>
Transformer::createTransformerDecoderBlock(const int layer_id,
                                           std::string input_name) {

  std::vector<LayerHandle> layers;

  layers.push_back(createLayer(
    "rms_norm",
    {withKey("name", "layer" + std::to_string(layer_id) + "_attention_norm"),
     withKey("input_layers", input_name),
     withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("packed", "false")}));

  auto att_layer =
    createAttention(layer_id, INIT_SEQ_LEN, NUM_HEADS, HEAD_DIM,
                    "layer" + std::to_string(layer_id) + "_attention_norm",
                    "layer" + std::to_string(layer_id) + "_attention_norm",
                    "layer" + std::to_string(layer_id) + "_attention_norm");

  layers.insert(layers.end(), att_layer.begin(), att_layer.end());

  layers.push_back(createLayer(
    "addition",
    {withKey("name", "layer" + std::to_string(layer_id) + "_decoder_add"),
     withKey("input_layers", input_name + ",layer" + std::to_string(layer_id) +
                               "_attention_out")}));

  // NNTRAINER_FUSED_RMSNORM_GATE_UP=1 absorbs ffn_norm into the
  // GateUpLayer dispatch (one kernel: cooperative rms_norm + Gate
  // proj + Up proj, single drain).  GateUpLayer registers gamma as
  // its FIRST weight to preserve the bundle byte order originally
  // produced by (rms_norm gamma + ffn_up + ffn_gate).  The standalone
  // rms_norm layer is omitted in this mode and gate_up_layer wires
  // its input directly to decoder_add.  Read once at static init so
  // every layer in the model agrees on the topology.
  static const bool s_fused_rmsnorm_gate_up =
    std::getenv("NNTRAINER_FUSED_RMSNORM_GATE_UP") != nullptr;
  if (!s_fused_rmsnorm_gate_up) {
    layers.push_back(createLayer(
      "rms_norm",
      {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_norm"),
       withKey("input_layers",
               "layer" + std::to_string(layer_id) + "_decoder_add"),
       withKey("epsilon", std::to_string(NORM_EPS)),
       withKey("packed", "false")}));
  }

  auto ffn_input_name =
    s_fused_rmsnorm_gate_up
      ? ("layer" + std::to_string(layer_id) + "_decoder_add")
      : ("layer" + std::to_string(layer_id) + "_ffn_norm");
  auto ffn_layer = createMlp(layer_id, DIM, INTERMEDIATE_SIZE, ffn_input_name);
  layers.insert(layers.end(), ffn_layer.begin(), ffn_layer.end());

  layers.push_back(createLayer(
    "addition",
    {withKey("name", "layer" + std::to_string(layer_id) + "_decoder_output"),
     withKey("input_layers", "layer" + std::to_string(layer_id) +
                               "_decoder_add,layer" + std::to_string(layer_id) +
                               "_ffn_down")}));

  return layers;
}

std::vector<LayerHandle>
Transformer::createAttention(const int layer_id, int seq_len, int n_heads,
                             int head_dim, std::string query_name,
                             std::string key_name, std::string value_name) {

  std::vector<LayerHandle> layers;

  auto Q = "layer" + std::to_string(layer_id) + "_wq";
  auto K = "layer" + std::to_string(layer_id) + "_wk";
  auto V = "layer" + std::to_string(layer_id) + "_wv";
  auto A = "layer" + std::to_string(layer_id) + "_attention";
  auto O = "layer" + std::to_string(layer_id) + "_attention_out";

  // Q layer
  std::vector<std::string> q_params = {
    withKey("name", Q), withKey("unit", head_dim * n_heads),
    withKey("disable_bias", "true"), withKey("input_layers", query_name),
    withKey("weight_initializer", "ones")};
  layers.push_back(createLayer("fully_connected", q_params));

  // K layer
  std::vector<std::string> k_params = {
    withKey("name", K), withKey("unit", head_dim * n_heads / GQA_SIZE),
    withKey("disable_bias", "true"), withKey("input_layers", key_name),
    withKey("weight_initializer", "ones")};
  layers.push_back(createLayer("fully_connected", k_params));

  // V layer
  std::vector<std::string> v_params = {
    withKey("name", V), withKey("unit", head_dim * n_heads / GQA_SIZE),
    withKey("disable_bias", "true"), withKey("input_layers", value_name),
    withKey("weight_initializer", "ones")};
  layers.push_back(createLayer("fully_connected", v_params));

  // Attention core layer
  std::vector<std::string> a_params = {
    withKey("name", A),
    withKey("num_heads", n_heads),
    withKey("num_heads_kv", n_heads / GQA_SIZE),
    withKey("max_timestep", std::to_string(INIT_SEQ_LEN + NUM_TO_GENERATE)),
    withKey("sliding_window", (layer_id + 1) % SLIDING_WINDOW_PATTERN
                                ? SLIDING_WINDOW
                                : UINT_MAX),
    withKey("rope_theta", ROPE_THETA),
    withKey("max_new_tokens", std::to_string(NUM_TO_GENERATE)),
    withKey("is_causal", IS_CAUSAL ? "true" : "false"),
    withKey("input_layers", {Q, K, V})};
  layers.push_back(createLayer("mha_core", a_params));

  // O layer
  std::vector<std::string> o_params = {
    withKey("name", O), withKey("unit", DIM), withKey("disable_bias", "true"),
    withKey("input_layers", A), withKey("weight_initializer", "ones")};
  layers.push_back(createLayer("fully_connected", o_params));

  return layers;
}

std::vector<LayerHandle> Transformer::createMlp(const int layer_id, int dim,
                                                int hidden_dim,
                                                std::string input_name) {

  std::vector<LayerHandle> layers;

  // Single GateUpLayer replaces ffn_up + ffn_gate FCs.  Weight order
  // (up first, then gate) matches the legacy registration order so the
  // bundle's byte layout is preserved -- no repackaging needed.
  // Output index convention from gate_up_layer.cpp:
  //   gate_up(0) = up projection
  //   gate_up(1) = gate projection
  // swiglu's first input is gate, second is up, hence the (1),(0) order.
  //
  // NNTRAINER_FUSED_RMSNORM_GATE_UP=1 (decided in
  // createTransformerDecoderBlock above) skips the standalone
  // ffn_norm rms_norm layer; in that mode GateUpLayer absorbs the
  // RMSNorm by registering gamma as its FIRST weight (matching the
  // bundle byte position of the removed rms_norm gamma) and
  // dispatching fused_rmsnorm_gate_up_cl. The fused_rmsnorm property
  // tells the layer which weight layout to use at finalize().
  static const bool s_fused_rmsnorm_gate_up_for_mlp =
    std::getenv("NNTRAINER_FUSED_RMSNORM_GATE_UP") != nullptr;
  auto gate_up_name = "layer" + std::to_string(layer_id) + "_ffn_gate_up";
  std::vector<std::string> gate_up_params{
    withKey("name", gate_up_name),
    withKey("up_unit", hidden_dim),
    withKey("gate_unit", hidden_dim),
    withKey("disable_bias", "true"),
    withKey("input_layers", input_name),
    withKey("weight_initializer", "ones")};
  if (s_fused_rmsnorm_gate_up_for_mlp) {
    gate_up_params.push_back(withKey("fused_rmsnorm", "true"));
    gate_up_params.push_back(
      withKey("fused_rmsnorm_epsilon", std::to_string(NORM_EPS)));
  }
  layers.push_back(createLayer("gate_up_layer", gate_up_params));

  // Phase M: SwiGLU layer stays in the graph for prefill correctness.
  // When env-gated and step_size==1, GateUpLayer's fused kernel writes
  // the final swiglu_out directly into Uhidden_step, and SwiGLULayer's
  // pass-through (also env-gated) copies that into its output. For
  // prefill (M>1) both layers fall back to the canonical path.
  layers.push_back(createLayer(
    "swiglu",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_swiglu"),
     withKey("input_layers", gate_up_name + "(1)," + gate_up_name + "(0)")}));

  layers.push_back(createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_down"),
     withKey("unit", dim), withKey("disable_bias", "true"),
     withKey("input_layers",
             "layer" + std::to_string(layer_id) + "_ffn_swiglu"),
     withKey("weight_initializer", "ones")}));

  return layers;
}

void Transformer::registerCustomLayers() {
  ///
  const auto &ct_engine = nntrainer::Engine::Global();
  const auto app_context =
    static_cast<nntrainer::AppContext *>(ct_engine.getRegisteredContext("cpu"));

  try {
    app_context->registerFactory(nntrainer::createLayer<causallm::SwiGLULayer>);
    app_context->registerFactory(
      nntrainer::createLayer<causallm::RMSNormLayer>);
    app_context->registerFactory(
      nntrainer::createLayer<causallm::MHACoreLayer>);
    app_context->registerFactory(
      nntrainer::createLayer<causallm::TieWordEmbedding>);
    app_context->registerFactory(
      nntrainer::createLayer<causallm::EmbeddingLayer>);
    app_context->registerFactory(nntrainer::createLayer<causallm::QKVLayer>);
    app_context->registerFactory(
      nntrainer::createLayer<causallm::QKVNormLayer>);
    app_context->registerFactory(
      nntrainer::createLayer<causallm::GateUpLayer>);

  } catch (std::invalid_argument &e) {
    std::cerr << "failed to register factory, reason: " << e.what()
              << std::endl;
  }
}

} // namespace causallm
