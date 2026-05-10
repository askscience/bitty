# CLI Reference

## Usage

```
bitty <command> [options] [arguments]
```

## Commands

### `bitty run <model> [prompt]`

Run inference on a model. If no prompt is given, enters interactive mode.

```
bitty run bitnet-b1.58 "The meaning of life is"
bitty run llama3.2:1b  # interactive mode
```

Options:
- `--temperature` / `-t`: Sampling temperature (default: 0.7)
- `--max-tokens` / `-n`: Maximum tokens to generate (default: 128)
- `--context` / `-c`: Context window size (default: 2048)
- `--cluster` / `-C`: Target cluster name

### `bitty pull <model>`

Download a model from the registry.

```
bitty pull bitnet-b1.58
bitty pull tinyllama:1.1b
```

### `bitty ls` / `bitty list`

List installed models with file sizes.

```
bitty ls
```

### `bitty show <model>`

Display detailed model information.

```
bitty show bitnet-b1.58
```

Output: architecture, quantization, file size, parameter count, context length, metadata.

### `bitty rm <model>`

Remove a downloaded model.

```
bitty rm bitnet-b1.58
```

### `bitty cp <source> <target>`

Copy or alias a model profile.

```
bitty cp bitnet-b1.58 my-custom-model
```

### `bitty create`

Create a model profile from a Modelfile.

```
bitty create -f Modelfile
```

### `bitty ps`

Show currently loaded/running models.

```
bitty ps
```

### `bitty stop <model>`

Stop a running model or the background runtime.

```
bitty stop llama3.2:1b
bitty stop          # stop all
```

### `bitty start`

Start a background runtime that loads a model into memory.

```
bitty start bitnet-b1.58
```

### `bitty serve`

Start the HTTP API server.

```
bitty serve
# Listening on http://127.0.0.1:11435
```

Options:
- `--host`: Listen address (default: `127.0.0.1:11435`)

### `bitty chat`

Start an interactive chat session.

```
bitty chat
```

Options:
- `--model` / `-m`: Model to use (default: from config)
- `--temperature` / `-t`: Temperature (default: 0.7)

### `bitty node`

Start a distributed node. Acts as a leader (coordinator + worker) or joins an existing cluster.

```
bitty node                    # Interactive node setup
bitty node --leader           # Start as cluster leader
bitty node --join <invite>    # Join existing cluster
```

### `bitty cluster`

Manage cluster operations.

```
bitty cluster status          # Show cluster health
bitty cluster nodes           # List registered nodes
bitty cluster check           # Test cluster connectivity
bitty cluster benchmark       # Run cluster benchmark
```

### `bitty invite`

Generate a cluster invite URL to share with other nodes.

```
bitty invite
# iroh://<endpoint_id>?token=<token>&relay=<relay_url>&addr=<socket_addr>
```

### `bitty join <invite-url>`

Join a cluster using an invite URL.

```
bitty join iroh://abcd1234?token=secret&relay=...
```

### `bitty use <cluster>`

Switch the active cluster target.

```
bitty use my-cluster
```

### `bitty clusters`

List saved cluster aliases.

```
bitty clusters
```

### `bitty generate <model> <prompt>`

Generate text via the cluster, returning the full response.

```
bitty generate bitnet-b1.58 "Once upon a time"
```

### `bitty status`

Show cluster health summary.

```
bitty status
```

### `bitty models`

Browse models available on the cluster.

```
bitty models
```

### `bitty settings`

Get or set configuration values.

```
bitty settings                       # List all settings
bitty settings get default_model     # Get a value
bitty settings set temperature 0.9   # Set a value
```

### `bitty logs`

View or clear logs.

```
bitty logs          # Tail last 50 lines
bitty logs --clear  # Clear log file
```

### `bitty clean`

Remove models and state but keep configuration.

```
bitty clean
```

### `bitty reset`

Remove everything and start fresh.

```
bitty reset
```

### `bitty setup`

Interactive first-time setup wizard.

```
bitty setup
```

### `bitty help`

Show help for any command.

```
bitty help run
bitty help
```

### `bitty version`

Show version information.

```
bitty version
```
