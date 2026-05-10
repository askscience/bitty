# Security Architecture

## Authentication Modes

Bitty supports two authentication modes for cluster communication:

### InsecureLocal

```
AuthMode::InsecureLocal
```

- No authentication required
- Intended for local development and testing
- All nodes must be on `127.0.0.1` or `localhost`
- Validated in `endpoint.rs` — rejects non-local addresses

### PreSharedToken

```
AuthMode::PreSharedToken
```

- All nodes share a pre-arranged token
- Token is passed via CLI argument (`--token`) or invite URL
- Comparison uses **constant-time** verification to prevent timing attacks
- Implemented via `security.rs`: `compare_constant_time(a, b)` using XOR-based comparison

## Iroh Transport Security

- All P2P communication uses **Iroh QUIC connections** with built-in encryption (TLS 1.3)
- Node identities are established via Ed25519 keypairs stored in `~/.bitty/iroh-secret.key`
- ALPN protocol negotiation: `bitty/scheduler/0` and `bitty/worker/0`
- NAT traversal via Iroh relay servers (configurable, defaults to public relays)

## Input Validation

The `validation.rs` module provides bounds checking for all wire inputs:

- **Activations**: maximum dimension size (configurable, default 65536)
- **Prompts**: maximum length (configurable, default 65536 tokens)
- **Logits**: maximum vocabulary size (configurable, default 262144)
- **Paths**: maximum length, no path traversal (no `..` segments)

## Rate Limiting

The coordinator uses the `governor` crate for rate limiting:
- **Default**: 100 requests per second per worker
- Configurable via `NetworkConfig`
- Enforced in `network.rs` on the gRPC handler

## Threat Model

| Threat | Mitigation |
|--------|-----------|
| Unauthorized node joining cluster | Pre-shared token with constant-time verification |
| Eavesdropping on P2P traffic | Iroh QUIC TLS 1.3 encryption |
| Malicious activation data | CRC32 checksum verification per tensor |
| Path traversal in model paths | Path validation (no `..`, length limits) |
| Denial of service (DDoS) | Governor rate limiting (100 req/s default) |
| Token theft from logs | Secret redaction in `secrets.rs` |
| Man-in-the-middle on join | Worker-initiated connection to coordinator |

## Secret Management

- Authentication tokens are **never logged** — the `secrets.rs` module redacts token-like strings from log output
- The Iroh secret key is stored at `~/.bitty/iroh-secret.key` with restricted permissions
- Cluster invite URLs are one-shot: they include the token and endpoint, but are intended for ephemeral use
