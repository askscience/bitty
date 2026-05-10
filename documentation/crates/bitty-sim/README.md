# bitty-sim

**Location**: `crates/bitty-sim/`

**Purpose**: Deterministic in-process cluster simulation for testing ring execution, topology changes, and failure modes without real networking.

## Key Types

### SimulatedCluster

```rust
pub struct SimulatedCluster {
    workers: Vec<RingWorker<FakeLayerExecutor>>,
    topology: TopologyUpdate,
    coordinator: NetworkCoordinator,
}
```

Builds a full cluster in-process:
1. Creates N fake workers with tier-based hardware profiles
2. Runs Halda to compute topology
3. Assigns `FakeLayerExecutor` to each worker
4. Sets up the ring

### ChaosProfile

```rust
pub struct ChaosProfile {
    pub drop_node: Option<NodeId>,      // Simulate node failure
    pub corrupt_node: Option<NodeId>,   // Corrupt activations
    pub latency_ms: Range<u64>,         // Random latency injection
}
```

### SimulationReport

Returns detailed per-token metrics:
- `final_activation`: Output tensor after ring traversal
- `hop_latencies`: Latency for each ring hop
- `checksum_ok`: Whether all checksums passed
- `total_tokens`: Tokens generated

### StreamedSimulation

Handles multi-token autoregressive simulation with per-token reports.

## Usage

### CLI
```
cargo run -p bitty-sim -- --nodes 4 --layers 32 --tokens 10
```

### Quick Start
```rust
use bitty_sim::{SimulatedCluster, demo_profiles, demo_layers};

let profiles = demo_profiles(4);         // 4 worker profiles
let layers = demo_layers(32);            // 32 model layers
let mut cluster = SimulatedCluster::new(&profiles, &layers);

let report = cluster.run_tokens(10);     // Generate 10 tokens
println!("{:?}", report);
```

### Chaos Testing
```rust
let chaos = ChaosProfile {
    drop_node: Some("node_2".into()),
    corrupt_node: None,
    latency_ms: 10..50,
};
let report = SimulatedCluster::with_chaos(&profiles, &layers, &chaos)
    .run_tokens(5);
```

## Test Fixtures

| Function | Description |
|----------|-------------|
| `demo_profiles(n)` | Creates `n` tiered `HardwareProfiles` (S, A, B, C tiers) |
| `demo_layers(n)` | Creates `n` `LayerMetadata` entries with realistic sizes |
| `tier_for_index(i)` | Maps index to tier: first node → S, last → D |

## Determinism

The simulation is fully deterministic for a given seed:
- Same node count, layer count, and token count produces identical results
- No real I/O, no network, no GPU
- Enables reproducible tests for the Halda scheduler and ring execution
