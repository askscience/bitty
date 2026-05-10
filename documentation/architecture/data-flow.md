# Data Flow

## Request Lifecycle

### Distributed Mode

```
User CLI                  Coordinator               Worker Ring
    │                          │                        │
    │  1. GenerateRequest      │                        │
    ├─────────────────────────►│                        │
    │                          │  2. Halda.assign()     │
    │                          │     (compute topology) │
    │                          │                        │
    │                          │  3. ForwardActivation   │
    │                          │  (embedding + layers)  │
    │                          ├───────────────────────►│
    │                          │                        │
    │                          │  ┌─── 4. Execute layer │
    │                          │  │    range, forward   │
    │                          │  │    activation ──────►│
    │                          │  │         (ring hop)   │
    │                          │  │  ◄─── activation ───┤
    │                          │  │       (ring back)   │
    │                          │  └── repeat until done │
    │                          │                        │
    │                          │  5. FinalLogits         │
    │                          │◄───────────────────────│
    │                          │                        │
    │                          │  6. SampleToken         │
    │                          ├───────────────────────►│
    │                          │                        │
    │  ◄─── TokenOutput ───────┤                        │
    │  (streaming)             │                        │
    │                          │                        │
    └──────────────────────────┘                        ┘
```

### Detailed Steps

#### 1. Generate Request
- User sends prompt string + generation parameters (temperature, max tokens, context size)
- CLI tokenizes the prompt using the model's tokenizer (HuggingFace tokenizers)
- Sends tokenized prompt to the coordinator via gRPC or Iroh

#### 2. Scheduling (Halda)
- Coordinator maintains a `Registry` of connected workers with their `HardwareProfile`
- On each request (or periodically), the Halda scheduler computes:
  - Layer assignments: which worker gets which layer range
  - Quantization per worker: based on tier (S/A → Q4, B → Q3, C/D → Q2)
  - Ring order: workers sorted by compute score for optimal pipeline
  - Memory budgets: respects per-node VRAM/RAM limits

#### 3. Forward Activation
- The embedding layer runs on the first assigned worker (the one holding layer 0)
- Initial activation tensor is sent with shape `[batch, seq_len, hidden_size]`
- Activation can be compressed using the configured codec (FP8, TopK, Delta)

#### 4. Ring Execution
- Each worker:
  a. Receives activation from previous worker (or coordinator)
  b. Verifies CRC32 checksum
  c. Runs `LayerExecutor::execute_range()` on its assigned layers
  d. Compresses output activation
  e. Forwards to next worker in ring
- The ring is traversed once per token

#### 5. Final Logits
- The last worker (holding LM head) computes logits over the vocabulary
- Returns `BitNetLogits` (f32 vector) to the coordinator

#### 6. Sampling
- Coordinator or final worker samples the next token:
  - `argmax()` if temperature == 0 (greedy)
  - `sample_with_temperature()`: softmax + xorshift64 PRNG
- Token is streamed back to user
- Process repeats from step 4 until:
  - Max tokens reached
  - EOS token generated
  - User stops generation

### Local Mode

```
User CLI → BitNetRuntime / CpuModel
              │
     ┌────────┴────────┐
     │  Embed tokens    │
     │  Forward layers  │
     │  Sample          │
     │  Decode          │
     └─────────────────┘
              │
         Token stream
```

In local mode, the entire model runs on a single machine using either:
- **GPU backend** (Candle/wgpu): `BitNetRuntime` with `SplitBitNetModel`
- **CPU backend**: `CpuModel` with optimized quantized matmul kernels

## Activation Data Flow

```
Input prompt (tokens)
    │
    ▼
┌─────────────────────┐
│  Embedding Layer    │  token IDs → hidden states
└─────────┬───────────┘
          │ activation tensor
          ▼
┌─────────────────────┐
│  Worker (Layers     │  self-attention + FFN / MLP / SSM
│   i..j)             │
│  ┌───────────────┐  │
│  │ LayerExecutor  │  │
│  │ .execute_range()│ │
│  └───────────────┘  │
└─────────┬───────────┘
          │ activation tensor (compressed)
          ▼
    (ring hop via Iroh)
          │
          ▼
┌─────────────────────┐
│  ...more workers...  │  repeat for assigned ranges
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Final Worker       │  LM head → logits
│  (LM Head)          │
└─────────┬───────────┘
          │ logits f32[]
          ▼
┌─────────────────────┐
│  Sampler            │  argmax / softmax + temperature
└─────────┬───────────┘
          │ token_id
          ▼
┌─────────────────────┐
│  Token Decoder      │  token_id → text
└─────────┬───────────┘
          │ "hello"
          ▼
       output
```

## Compression Pipeline

```
Before ring send:
  f16 activation [N, H]
       │
       ▼
  ActivationCodec
  ┌─────────────────┐
  │ FP8 Linear:     │  (sample/256 + 128).clamp(0,255)
  │ Sparse TopK 30% │  keep top 30% by magnitude
  │ Delta (passthru)│  no compression
  └─────────────────┘
       │
       ▼
  packed bytes + CRC32 checksum

After ring receive:
  packed bytes + CRC32 checksum
       │
       ▼
  verify CRC32
  decompress → f16 activation [N, H]
```
