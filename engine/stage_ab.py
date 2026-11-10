"""A/B of engine settings on one stage: runs each variant in turn and prints median ms per token.

usage: BITTY_STAGE_BIN=.../bitty-stage stage_ab.py model start end rounds tokens -- \
           name:ENV=v,ENV=v[:threads[:prefix command]] ...
example: stage_ab.py model.gguf 10 15 2 60 -- "default:" "spin1000:GOMP_SPINCOUNT=1000" "t12::12"
AB_EXTRA adds engine arguments, AB_GAP_MS an idle pause between calls.
"""
import os, struct, subprocess, sys, time
BIN = os.environ["BITTY_STAGE_BIN"]
model, start, end, rounds, tokens = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
configs = sys.argv[7:]
def read(f, n):
    b = b""
    while len(b) < n:
        c = f.read(n - len(b))
        if not c: raise RuntimeError("engine closed")
        b += c
    return b
def throttle():
    return int(open("/sys/devices/system/cpu/cpu0/thermal_throttle/package_throttle_count").read())
def run(spec, extra):
    parts = spec.split(":")
    env = dict(os.environ); 
    for kv in filter(None, parts[1].split(",")):
        k, v = kv.split("="); env[k] = v
    threads = parts[2] if len(parts) > 2 and parts[2] else "8"
    prefix = parts[3].split() if len(parts) > 3 else []
    args = prefix + [BIN, "-m", model, "--start", str(start), "--end", str(end), "-t", threads] + extra
    p = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=env)
    assert read(p.stdout, 4) == b"BSTG"; hs = struct.unpack("<6I", read(p.stdout, 24)); width = hs[2]
    def call(op, pos, n, payload):
        p.stdin.write(struct.pack("<IQIII", op, 1, pos, n, len(payload)) + payload); p.stdin.flush()
        status, kind, rows, ln = struct.unpack("<4I", read(p.stdout, 16)); data = read(p.stdout, ln)
        if status: raise RuntimeError(data.decode(errors="replace"))
    def payload(n, k):
        if start == 0: return struct.pack("<%dI" % n, *[(1000 + 37 * (k + i)) % 200000 for i in range(n)])
        import random; r = random.Random(k)
        return struct.pack("<%df" % (n * width), *[r.uniform(-1, 1) for _ in range(n * width)])
    op = 1 if start == 0 else 2
    th = throttle(); t = time.time(); call(op, 0, 16, payload(16, 0)); prompt = time.time() - t
    times = []
    for k in range(tokens):
        pl = payload(1, k + 1); time.sleep(float(os.environ.get("AB_GAP_MS", "0")) / 1000); t = time.time(); call(op, 16 + k, 1, pl); times.append(time.time() - t)
    p.stdin.close(); p.wait()
    times.sort()
    return prompt * 1000, times[len(times)//2] * 1000, times[0] * 1000, throttle() - th
extra = os.environ.get("AB_EXTRA", "").split()
res = {c.split(":")[0]: [] for c in configs}
for r in range(rounds):
    for c in configs:
        pr, med, best, th = run(c, extra)
        res[c.split(":")[0]].append(med)
        print("round %d %-28s prompt %6.0f ms  token median %6.1f  best %6.1f  throttle +%d" % (r, c.split(":")[0], pr, med, best, th), flush=True)
print("--- medians per config")
for name, v in res.items():
    print("%-28s %s" % (name, "  ".join("%.1f" % x for x in v)))
