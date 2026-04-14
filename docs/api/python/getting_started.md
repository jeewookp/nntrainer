# LiteRT-LM Python API

The Python API of LiteRT-LM for **Linux and MacOS** (Windows support is upcoming).
Features like **multi-modality** and **tools use** are supported, while **GPU
acceleration** is upcoming.

## Introduction

Here is a sample terminal chat app built with the Python API:

```python
import litert_lm

litert_lm.set_min_log_severity(litert_lm.LogSeverity.ERROR) # Hide log for TUI app

with litert_lm.Engine("path/to/model.litertlm") as engine:
  with engine.create_conversation() as conversation:
    while True:
      user_input = input("\n>>> ")
      for chunk in conversation.send_message_async(user_input):
        print(chunk["content"][0]["text"], end="", flush=True)
```

![](../kotlin/demo.gif)

## Getting Started

LiteRT-LM is available as a Python library. You can install the nightly version from PyPI:

```bash
# Using pip
pip install litert-lm-nightly

# Using uv
uv pip install litert-lm-nightly
```

### 1. Initialize the Engine

The `Engine` is the entry point to the API. It handles model loading and resource management. Using it as a context manager (with the `with` statement) ensures that native resources are released promptly.

**Note:** Initializing the engine can take several seconds to load the model.

```python
import litert_lm

# Initialize with the model path and optionally specify the backend.
# backend can be Backend.CPU (default). GPU support is upcoming.
with litert_lm.Engine(
    "path/to/your/model.litertlm",
    backend=litert_lm.Backend.CPU,
    # Optional: Pick a writable dir for caching compiled artifacts.
    # cache_dir="/tmp/litert-lm-cache"
) as engine:
    # ... Use the engine to create a conversation ...
    pass
```

### 2. Create a Conversation

A `Conversation` manages the state and history of your interaction with the model.

```python
# Optional: Configure system instruction and initial messages
messages = [
    {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
]

# Create the conversation
with engine.create_conversation(messages=messages) as conversation:
    # ... Interact with the conversation ...
    pass
```

### 3. Sending Messages

You can send messages synchronously or asynchronously (streaming).

**Synchronous Example:**

```python
# Simple string input
response = conversation.send_message("What is the capital of France?")
print(response["content"][0]["text"])

# Or with full message structure
# response = conversation.send_message({"role": "user", "content": "..."})
```

**Asynchronous (Streaming) Example:**

```python
# sendMessageAsync returns an iterator of response chunks
stream = conversation.send_message_async("Tell me a long story.")
for chunk in stream:
    # Chunks are dictionaries containing pieces of the response
    for item in chunk.get("content", []):
      if item.get("type") == "text":
        print(item["text"], end="", flush=True)
print()
```

### 4. Multi-Modality

Note: This requires models with multi-modality support, such as [Gemma3n](https://huggingface.co/google/gemma-3n-E2B-it-litert-lm).

```python
# Initialize with vision and/or audio backends if needed
with litert_lm.Engine(
    "path/to/multimodal_model.litertlm",
    audio_backend=litert_lm.Backend.CPU,
    # vision_backend=litert_lm.Backend.CPU, (GPU support is upcoming)
) as engine:
    with engine.create_conversation() as conversation:
        user_message = {
            "role": "user",
            "content": [
                {"type": "audio", "path": "/path/to/audio.wav"},
                {"type": "text", "text": "Describe this audio."},
            ],
        }
        response = conversation.send_message(user_message)
        print(response["content"][0]["text"])
```

### 5. Defining and Using Tools

Note: This requires models with tool support, such as [FunctionGemma](https://huggingface.co/google/functiongemma-270m-it).

You can define Python functions as tools that the model can call automatically.

```python
def add_numbers(a: float, b: float) -> float:
    """Adds two numbers.

    Args:
        a: The first number.
        b: The second number.
    """
    return a + b

# Register the tool in the conversation
tools = [add_numbers]
with engine.create_conversation(tools=tools) as conversation:
    # The model will call add_numbers automatically if it needs to sum values
    response = conversation.send_message("What is 123 + 456?")
    print(response["content"][0]["text"])
```

LiteRT-LM uses the function's docstring and type hints to generate the tool schema for the model.
