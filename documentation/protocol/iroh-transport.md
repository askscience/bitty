# Iroh Transport

Bitty uses [Iroh](https://iroh.computer) for encrypted peer-to-peer transport between cluster nodes. Iroh provides QUIC-based connections with built-in NAT traversal via relay servers.

## ALPN Protocol IDs

```
Coordinator: bitty/scheduler/0
Worker:      bitty/worker/0
```

## Binary Frame Protocol

Protobuf messages are wrapped in a custom binary frame format on top of Iroh bidirectional streams.

### Frame Format

```
┌──────────────────────────────────────────────────┐
│ 4 bytes: Payload Length (big-endian u32)         │
├──────────────────────────────────────────────────┤
│ 1 byte:  Opcode                                  │
├──────────────────────────────────────────────────┤
│ 2 bytes: Token Length (big-endian u16)           │
├──────────────────────────────────────────────────┤
│ Token Length bytes: Auth Token (UTF-8)          │
├──────────────────────────────────────────────────┤
│ Payload Length bytes: Protobuf Payload           │
└──────────────────────────────────────────────────┘
```

### Opcodes

| Opcode | RPC | Description |
|--------|-----|-------------|
| 1 | Register | Worker registers with coordinator |
| 2 | Heartbeat | Worker sends health metrics |
| 3 | Generate | Start text generation |
| 4 | ForwardActivation | Forward tensor to next worker |
| 5 | FinalLogits | Return logits to coordinator |
| 6 | SampleToken | Sample next token |
| 7 | ApplyTopology | Apply new layer assignments |
| 8 | LoadShard | Load weight shard |
| 9 | Cleanup | Free resources |

## URI Format

Cluster invite URIs use the following format:

```
iroh://<endpoint_id>?token=<auth_token>&relay=<relay_url>&addr=<socket_addr>
```

Example:
```
iroh://abc123def456?token=my-secret-token&relay=https://relay.iroh.network&addr=192.168.1.100:50051
```

## Connection Lifecycle

```
Client (worker)                    Server (coordinator)
     │                                    │
     │  1. Resolve relay URL              │
     │  2. Establish QUIC connection      │
     ├───────────────────────────────────►│
     │  3. ALPN negotiation               │
     │     (bitty/scheduler/0)            │
     │                                    │
     │  4. Open bi-directional stream     │
     ├───────────────────────────────────►│
     │  5. Send Register frame            │
     │     (opcode=1, token, protobuf)    │
     │                                    │
     │  6. Receive RegisterResponse       │
     │◄───────────────────────────────────┤
     │                                    │
     │  7. Periodic Heartbeat frames      │
     ├───────────────────────────────────►│
     │                                    │
     │  8. Receive ForwardActivation      │
     │◄───────────────────────────────────┤
     │  9. Execute, send response         │
     ├───────────────────────────────────►│
     │                                    │
```

## Relay Configuration

- **Default**: `public` — uses Iroh's public relay servers
- **Custom**: Set `iroh_relays` in config to a relay URL
- **LAN only**: Workers on the same local network connect directly via QUIC, bypassing relays

## NAT Traversal

1. Iroh uses STUN-like mechanisms for NAT type detection
2. Relays are used as fallback when direct connections fail
3. Connections are encrypted end-to-end regardless of relay usage

## Identity

- Each node generates an Ed25519 keypair stored at `~/.bitty/iroh-secret.key`
- The public key serves as the node's Iroh endpoint ID
- Keys are persisted across restarts for stable identity
