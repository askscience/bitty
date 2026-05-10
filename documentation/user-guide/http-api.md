# HTTP API

Bitty provides both Ollama-compatible and OpenAI-compatible HTTP API endpoints.

**Default address**: `http://127.0.0.1:11435`

## Ollama-Compatible Endpoints

### Generate

```
POST /api/generate
```

**Request**:
```json
{
  "model": "bitnet-b1.58",
  "prompt": "The meaning of life is",
  "stream": true,
  "options": {
    "temperature": 0.7,
    "num_predict": 128,
    "num_ctx": 2048
  }
}
```

**Response** (streamed):
```json
{"model":"bitnet-b1.58","response":" to ","done":false}
{"model":"bitnet-b1.58","response":"find ","done":false}
{"model":"bitnet-b1.58","response":"purpose.","done":true,"context":[1,2,3],"total_duration":123456789}
```

### Chat

```
POST /api/chat
```

**Request**:
```json
{
  "model": "bitnet-b1.58",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ],
  "stream": true,
  "options": {
    "temperature": 0.7
  }
}
```

**Response** (streamed):
```json
{"model":"bitnet-b1.58","message":{"role":"assistant","content":"Hi"},"done":false}
{"model":"bitnet-b1.58","message":{"role":"assistant","content":" there!"},"done":true}
```

### List Models

```
GET /api/tags
```

**Response**:
```json
{
  "models": [
    {
      "name": "bitnet-b1.58:latest",
      "modified_at": "2024-01-01T00:00:00Z",
      "size": 1234567890
    }
  ]
}
```

### Show Model

```
POST /api/show
```

**Request**:
```json
{
  "model": "bitnet-b1.58"
}
```

### Pull Model

```
POST /api/pull
```

**Request**:
```json
{
  "model": "bitnet-b1.58"
}
```

## OpenAI-Compatible Endpoints

### List Models

```
GET /v1/models
```

**Response**:
```json
{
  "object": "list",
  "data": [
    {
      "id": "bitnet-b1.58",
      "object": "model",
      "created": 1704067200,
      "owned_by": "bitty"
    }
  ]
}
```

### Chat Completions

```
POST /v1/chat/completions
```

**Request**:
```json
{
  "model": "bitnet-b1.58",
  "messages": [
    {"role": "user", "content": "Hello!"}
  ],
  "temperature": 0.7,
  "max_tokens": 128,
  "stream": true
}
```

**Response** (streamed):
```json
{"id":"chatcmpl-123","object":"chat.completion.chunk","choices":[{"delta":{"content":"Hello"},"index":0}]}
{"id":"chatcmpl-123","object":"chat.completion.chunk","choices":[{"delta":{"content":" world"},"index":0}]}
{"id":"chatcmpl-123","object":"chat.completion.chunk","choices":[{"delta":{},"finish_reason":"stop","index":0}]}
```

### Completions

```
POST /v1/completions
```

**Request**:
```json
{
  "model": "bitnet-b1.58",
  "prompt": "Once upon a",
  "max_tokens": 64,
  "temperature": 0.8,
  "stream": false
}
```

**Response**:
```json
{
  "id": "cmpl-123",
  "object": "text_completion",
  "choices": [
    {
      "text": " time in a land far away...",
      "index": 0,
      "finish_reason": "length"
    }
  ],
  "usage": {
    "prompt_tokens": 3,
    "completion_tokens": 64,
    "total_tokens": 67
  }
}
```

## Metrics Endpoint

```
GET /metrics
```

Returns Prometheus-formatted metrics.

## Health Check

```
GET /health
```

Returns `{"status":"ok"}` if the server is running.
