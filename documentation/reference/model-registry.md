# Model Registry

**File**: `models/registry.toml`

The built-in registry contains 20+ models across multiple architecture families. Models are downloaded on-demand when running `bitty pull <name>`.

## BitNet Models

| Name | Parameters | Status |
|------|-----------|--------|
| `bitnet-b1.58` | 1.58-bit | stable (default) |

## Llama Family

| Name | Parameters | Status |
|------|-----------|--------|
| `tinyllama:1.1b` | 1.1B | stable |
| `smollm2:1.7b` | 1.7B | stable |
| `llama3.2:1b` | 1B | stable |
| `llama3.2:3b` | 3B | stable |
| `llama3:8b` | 8B | stable |

## Qwen Family

| Name | Parameters | Status |
|------|-----------|--------|
| `qwen2.5:0.5b` | 0.5B | stable |
| `deepseek-r1:1.5b` | 1.5B | stable |
| `qwen3.5:2b` | 2B | experimental |
| `qwen3:4b` | 4B | experimental |
| `qwen3:8b` | 8B | experimental |
| `qwen3:32b` | 32B | experimental |

## Gemma Family

| Name | Parameters | Status |
|------|-----------|--------|
| `gemma3:4b` | 4B | experimental |
| `gemma3:12b` | 12B | experimental |
| `gemma3:27b` | 27B | experimental |

## Mistral Family

| Name | Parameters | Status |
|------|-----------|--------|
| `mistral:7b` | 7B | stable |
| `mistral-nemo:12b` | 12B | experimental |

## Phi Family

| Name | Parameters | Status |
|------|-----------|--------|
| `phi3.5:3.8b` | 3.8B | stable |

## Registry Format

Each model entry follows this structure:

```toml
[[models]]
name = "model-name"
tag = "version-tag"
display_name = "Human Readable Name"
backend = "bitnet|candle"
quantization = "bit1|Q4_K_M|Q4_0|..."
filename = "model_file.gguf"
source = "https://huggingface.co/org/model-gguf"
url = "https://huggingface.co/org/model-gguf/resolve/main/model_file.gguf"
temperature = 0.7
num_predict = 128
num_ctx = 2048
status = "stable|experimental|deprecated"
```

## Adding a Model

To add a new model to the registry:

1. Find or create a GGUF-quantized version of the model
2. Add an entry to `models/registry.toml`
3. Test with `bitty pull <name>` and `bitty run <name>`
4. Submit a PR
