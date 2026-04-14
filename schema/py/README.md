# LiteRT-LM Builder

Python tools for building, inspecting, and unpacking LiteRT-LM files.

This directory contains the source code for the `litert-lm-builder` Python package.

## Project Structure

- `litertlm_builder.py`: Core logic for building LiteRT-LM files.
- `litertlm_builder_cli.py`: Command-line interface for the builder.
- `litertlm_peek.py`: Core logic for inspecting LiteRT-LM files.
- `litertlm_peek_main.py`: Command-line interface for the peek tool.
- `pyproject.toml`: PEP 517 configuration for the package.
- `bundle_pypi_package.sh`: Script to bundle the package into a PyPI-ready wheel.

## Building and Packaging

To build the package and create a wheel, run the bundle script:

```bash
./bundle_pypi_package.sh
```

This script will:
1. Stage the files in a temporary directory.
2. Build Protobuf and FlatBuffer bindings using Bazel.
3. Rewrite imports to match the package structure.
4. Build the wheel using `uv`.

## Usage

After installing the package, you can use the CLI tools:

### litertlm-builder

```bash
litertlm-builder [options]
```

The tool supports two options: Subcommand Chaining (passing options directly) and TOML Configuration.

#### Example: Subcommand Chaining (Direct Arguments)

```bash
litertlm-builder \
  system_metadata --str Authors "ODML team" --int version 1 \
  llm_metadata --path schema/testdata/llm_metadata.pb \
  sp_tokenizer --path runtime/components/testdata/sentencepiece.model \
  tflite_model --path runtime/components/testdata/dummy_embedding_cpu_model.tflite --model_type embedder \
  output --path real.litertlm
```

#### Example: Using a TOML Configuration File

```bash
litertlm-builder toml --path example.toml output --path real_via_toml.litertlm
```

### litertlm-peek

```bash
litertlm-peek [options]
```

#### Example Usage

To inspect a `.litertlm` file:

```bash
litertlm-peek --litertlm_file real.litertlm
```

To extract (dump) the contents of a `.litertlm` file:

```bash
litertlm-peek --litertlm_file real.litertlm --dump_files_dir ./extracted_files
```

