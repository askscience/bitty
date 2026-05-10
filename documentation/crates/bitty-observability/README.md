# bitty-observability

**Location**: `crates/bitty-observability/`

**Purpose**: Prometheus metrics recording and structured tracing via the `tracing` crate.

## Metrics

All metrics use the `metrics` crate with a Prometheus exporter.

### Available Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `dlm_layer_latency_us` | Histogram | `node_id`, `layer` | Per-layer execution time in microseconds |
| `dlm_activation_bytes_total` | Counter | `node_id` | Total activation bytes transferred |
| `dlm_checksum_failures_total` | Counter | `node_id` | CRC32 checksum mismatch count |
| `dlm_tokens_generated_total` | Counter | `node_id`, `model` | Total tokens generated |
| `dlm_halda_runs_total` | Counter | — | Number of Halda scheduler invocations |
| `dlm_halda_duration_ms` | Histogram | — | Halda scheduler execution time |

### Functions

```rust
pub fn install_prometheus_recorder() -> PrometheusHandle
```

Returns a `PrometheusHandle` that exposes metrics at the standard `/metrics` endpoint. Call once at startup.

```rust
pub fn record_halda_run(duration: Duration)
```

Records a Halda scheduler run in both the counter and histogram metrics.

## Tracing

Initializes structured logging via the `tracing` crate:

```rust
pub fn init_tracing()
```

- Uses `tracing-subscriber` with `EnvFilter`
- Default log level: `info`
- Controlled via `RUST_LOG` environment variable
- Outputs to stderr with structured fields
- Compatible with `opentelemetry` for distributed tracing

## Usage

```rust
use bitty_observability::{install_prometheus_recorder, record_halda_run, init_tracing};

// At startup
init_tracing();
let _handle = install_prometheus_recorder();

// During operation
let start = std::time::Instant::now();
// ... run Halda ...
record_halda_run(start.elapsed());
```

## Prometheus Scrape Endpoint

The metrics are exposed via the HTTP API server's `/metrics` endpoint (port 11435 by default), which can be scraped by Prometheus or any OpenMetrics-compatible collector.
