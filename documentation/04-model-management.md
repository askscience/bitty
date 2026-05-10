# Model Management

## Model Registry

Bitty ships with a built-in model registry at `models/registry.toml` containing 20+ models.

### Available Models

| Model | Parameters | Backend | Quantization |
|-------|-----------|---------|-------------|
| `bitnet-b1.58` | 1.58-bit | BitNet | Bit1 |
| `tinyllama:1.1b` | 1.1B | Candle | Q4_K_M |
| `llama3.2:1b` | 1B | Candle | Q4_K_M |
| `llama3.2:3b` | 3B | Candle | Q4_K_M |
| `llama3:8b` | 8B | Candle | Q4_K_M |
| `qwen2.5:0.5b` | 0.5B | Candle | Q4_K_M |
| `phi3.5:3.8b` | 3.8B | Candle | Q4_K_M |
| `gemma3:4b` | 4B | Candle | Q4_K_M |
| `deepseek-r1:1.5b` | 1.5B | Candle | Q4_K_M |
| `smollm2:1.7b` | 1.7B | Candle | Q4_K_M |
| `mistral:7b` | 7B | Candle | Q4_K_M |

### Registry Entry Format

```toml
[[models]]
name = "bitnet-b1.58"
tag = "stable"
display_name = "BitNet b1.58"
backend = "bitnet"
quantization = "bit1"
filename = "bitnet_b1_58-Q4_K_M.gguf"
source = "https://huggingface.co/askscience/bitnet-b1.58-gguf"
url = "https://huggingface.co/askscience/bitnet-b1.58-gguf/resolve/main/bitnet_b1_58-Q4_K_M.gguf"
temperature = 0.7
num_predict = 128
num_ctx = 2048
status = "stable"
```

## Pulling Models

```bash
# Pull the default model
bitty pull bitnet-b1.58

# Pull a specific model
bitty pull llama3.2:1b
```

Models are downloaded to `~/.bitty/models/<model-name>/model.gguf`.

## Creating Custom Profiles with Modelfile

Bitty supports Ollama-style Modelfiles:

```dockerfile
FROM bitnet-b1.58
PARAMETER temperature 0.8
PARAMETER num_ctx 4096
SYSTEM "You are a helpful assistant."
TEMPLATE "{{ .Prompt }}"
MESSAGE user "Hello"
MESSAGE assistant "Hi! How can I help you today?"
LICENSE MIT
```

Create and use:

```bash
bitty create -f Modelfile
bitty run my-custom-model "Tell me a joke"
```

## Model Storage

```
~/.bitty/models/
├── bitnet-b1.58/
│   ├── model.gguf           # Weight file
│   └── model.gguf.metadata  # Cached metadata
├── llama3.2:1b/
│   └── model.gguf
└── ...
```

## Managing Models

```bash
# List installed models
bitty ls

# Show model details
bitty show bitnet-b1.58

# Remove a model
bitty rm tinyllama:1.1b

# Copy/alias a model
bitty cp bitnet-b1.58 my-bitnet

# View loaded models
bitty ps
```
