"""Times each stage of a model alone, on the CPU or the GPU.

Stages run one after the other, each fed the hidden states the previous one produced, so a
machine that cannot hold the whole model on its GPU can still measure every stage there.
The sum of the per-token times is what a chain of such peers would need without the network.
"""
import os, struct, subprocess, sys, time

if len(sys.argv) < 3:
    sys.exit("usage: BITTY_STAGE_BIN=path/to/bitty-stage stage_bench.py cpu|gpu model.gguf [layers per stage]")
BIN = os.environ.get("BITTY_STAGE_BIN", "bitty-stage")
mode, MODEL = sys.argv[1], sys.argv[2]
STEP = int(sys.argv[3]) if len(sys.argv) > 3 else 5
PROMPT, STEPS = 16, 60
TEXT = ("A steam engine turns heat into motion. Water is boiled in a closed vessel and the steam "
        "pushes a piston inside a cylinder. The piston moves a rod, the rod turns a wheel, and a "
        "valve lets the used steam out so the cycle can start again. Early engines were slow and "
        "wasted most of their fuel, but they changed mines, mills, ships and railways for good.") * 2

def read(f, n):
    b = b""
    while len(b) < n:
        c = f.read(n - len(b))
        if not c: raise RuntimeError("engine closed")
        b += c
    return b

class Stage:
    def __init__(self, args):
        self.p = subprocess.Popen([BIN, "-m", MODEL] + args, stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        assert read(self.p.stdout, 4) == b"BSTG"
        self.layers = struct.unpack("<6I", read(self.p.stdout, 24))[1]
    def call(self, op, pos=0, n=0, payload=b""):
        self.p.stdin.write(struct.pack("<IQIII", op, 1, pos, n, len(payload)) + payload)
        self.p.stdin.flush()
        status, kind, rows, ln = struct.unpack("<4I", read(self.p.stdout, 16))
        data = read(self.p.stdout, ln)
        if status != 0: raise RuntimeError(data.decode(errors="replace"))
        return data
    def close(self):
        self.p.stdin.close(); self.p.wait()

v = Stage(["--vocab-only"])
tokens = v.call(6, payload=TEXT.encode())
v.close()
probe = Stage(["--start", "0", "--end", "1"])  # the vocabulary process does not report the layer count
LAYERS = probe.layers
probe.close()
ids = [tokens[i:i+4] for i in range(0, len(tokens), 4)][:PROMPT + STEPS]
assert len(ids) == PROMPT + STEPS, len(ids)

# inputs[k] is what the stage receives at call k: the prompt batch, then one token per call
inputs = [b"".join(ids[:PROMPT])] + ids[PROMPT:]
results = []
for start in range(0, LAYERS, STEP):
    end = min(start + STEP, LAYERS)
    gpu = mode == "gpu"
    args = ["--start", str(start), "--end", str(end), "-t", "8"]
    if gpu: args += ["--gpu-layers", str(LAYERS - start + 1)]
    t0 = time.time(); st = Stage(args); load = time.time() - t0
    op = 1 if start == 0 else 2
    outputs, times, pos = [], [], 0
    for k, payload in enumerate(inputs):
        n = PROMPT if k == 0 else 1
        t = time.time(); out = st.call(op, pos, n, payload); times.append(time.time() - t)
        pos += n; outputs.append(out)
    st.close()
    steps = sorted(times[1:]); med = steps[len(steps)//2] * 1000
    results.append(dict(stage=f"{start}..{end}", load_s=round(load, 1), prompt_ms=round(times[0]*1000),
                        token_ms=round(med, 1), mean_ms=round(sum(steps)/len(steps)*1000, 1)))
    print(mode, results[-1], flush=True)
    inputs = outputs
total = sum(r["token_ms"] for r in results)
print(mode, "sum of stage medians per token: %.1f ms -> %.2f tokens/s" % (total, 1000/total))
