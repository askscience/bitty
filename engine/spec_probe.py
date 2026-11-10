"""Two measurements for speculative decoding with a small draft model.

  spec_probe.py batch model start end [engine args...]
      time of one decode call with 1, 2, 4 and 8 tokens on a stage (ms, median)
  spec_probe.py accept target.gguf layers draft.gguf layers [tokens]
      greedy text of the target, and how often the draft's best token is the same one

BITTY_STAGE_BIN is the engine. Both models run whole, on CPU, with 8 threads.
"""
import json, os, random, statistics, struct, subprocess, sys, time

BIN = os.environ["BITTY_STAGE_BIN"]
PROMPTS = [
    "Spiega in italiano, in modo semplice, come funziona un motore a vapore.",
    "What were the main causes of the fall of the Western Roman Empire?",
    "Write a Python function that returns the n-th Fibonacci number, with a short docstring.",
    "Scrivi l'inizio di un racconto breve su un faro abbandonato.",
    "List five practical tips for learning a new language, one line each.",
    "Explain the difference between TCP and UDP to a beginner.",
]


class Engine:
    def __init__(self, args):
        self.p = subprocess.Popen([BIN] + args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        assert self.read(4) == b"BSTG"
        _, self.layers, self.width, self.vocab, self.start, self.end = struct.unpack("<6I", self.read(24))

    def read(self, n):
        b = b""
        while len(b) < n:
            c = self.p.stdout.read(n - len(b))
            if not c:
                raise RuntimeError("engine closed")
            b += c
        return b

    def call(self, op, session, pos, n, payload=b""):
        self.p.stdin.write(struct.pack("<IQIII", op, session, pos, n, len(payload)) + payload)
        self.p.stdin.flush()
        status, kind, rows, ln = struct.unpack("<4I", self.read(16))
        data = self.read(ln)
        if status:
            raise RuntimeError(data.decode(errors="replace"))
        return data

    def tokens(self, data):
        return list(struct.unpack("<%dI" % (len(data) // 4), data))

    def prompt(self, text):
        payload = b"".join(struct.pack("<I", len(part)) + part for part in (b"user", text.encode()))
        chat = self.call(7, 0, 0, 1, payload)
        return self.tokens(self.call(6, 0, 0, 1, chat))

    def best(self, session, pos, ids):
        """Feeds token ids, returns the best next token (engine started with --top-logits 1)."""
        data = self.call(1, session, pos, len(ids), struct.pack("<%dI" % len(ids), *ids))
        return struct.unpack("<If", data[:8])[0]

    def text(self, ids):
        return self.call(5, 0, 0, 0, struct.pack("<%dI" % len(ids), *ids)).decode(errors="replace")

    def close(self):
        self.p.stdin.close()
        self.p.wait()


def batch(model, start, end, extra):
    engine = Engine(["-m", model, "--start", start, "--end", end, "-t", "8", "--top-logits", "1"] + extra)
    rnd = random.Random(1)

    def payload(n):
        if engine.start == 0:
            return struct.pack("<%dI" % n, *[rnd.randrange(1000, 200000) for _ in range(n)])
        return struct.pack("<%df" % (n * engine.width), *[rnd.uniform(-1, 1) for _ in range(n * engine.width)])

    op = 1 if engine.start == 0 else 2
    pos = 32
    engine.call(op, 1, 0, pos, payload(pos))
    times = {1: [], 2: [], 4: [], 8: []}
    for _ in range(10):
        for n in times:
            data = payload(n)
            started = time.time()
            engine.call(op, 1, pos, n, data)
            times[n].append((time.time() - started) * 1000)
            pos += n
    engine.close()
    one = statistics.median(times[1])
    for n, values in times.items():
        median = statistics.median(values)
        print("%d token(s): %7.1f ms  (%.2f times one token, %.1f ms each)" % (n, median, median / one, median / n))


def accept(target_model, target_layers, draft_model, draft_layers, count):
    target = Engine(["-m", target_model, "--start", "0", "--end", target_layers, "-t", "8", "--top-logits", "1"])
    stop = set(target.tokens(target.call(8, 0, 0, 0)))
    runs = []
    for index, prompt in enumerate(PROMPTS):
        ids = target.prompt(prompt)
        out = [target.best(index + 1, 0, ids)]
        while len(out) < count and out[-1] not in stop:
            out.append(target.best(index + 1, len(ids) + len(out) - 1, [out[-1]]))
        target.call(3, index + 1, 0, 0)
        runs.append((ids, out))
        print("target %d: %d tokens: %s" % (index, len(out), target.text(out)[:70].replace("\n", " ")), flush=True)
    vocab = target.vocab
    target.close()

    draft = Engine(["-m", draft_model, "--start", "0", "--end", draft_layers, "-t", "8", "--top-logits", "1"])
    assert draft.vocab == vocab, "different vocabularies"
    assert draft.prompt(PROMPTS[0]) == runs[0][0], "the two models build different prompts"
    every = []
    for index, (ids, out) in enumerate(runs):
        guess = draft.best(index + 1, 0, ids)
        match = [guess == out[0]]
        for k in range(1, len(out)):
            match.append(draft.best(index + 1, len(ids) + k - 1, [out[k - 1]]) == out[k])
        draft.call(3, index + 1, 0, 0)
        every.append(match)
        print("draft %d: %d of %d the same (%.0f%%)" % (index, sum(match), len(match), 100 * sum(match) / len(match)), flush=True)
    draft.close()
    total = sum(len(m) for m in every)
    print("all: %d of %d the same (%.0f%%)" % (sum(map(sum, every)), total, 100 * sum(map(sum, every)) / total))
    # A round: the draft proposes k tokens, the target checks them in one pass and keeps the
    # leading ones that match plus its own token at the first miss.
    for k in (2, 3, 4, 6, 8):
        rounds = 0
        for match in every:
            i = 0
            while i < len(match):
                kept = 0
                while kept < k and i + kept < len(match) and match[i + kept]:
                    kept += 1
                i += kept + 1 if kept < k else kept
                rounds += 1
        print("draft of %d: %.2f tokens per pass of the target" % (k, total / rounds))
    json.dump(every, open(os.environ.get("SPEC_OUT", "/tmp/spec_matches.json"), "w"))


if sys.argv[1] == "batch":
    batch(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5:])
else:
    accept(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], int(sys.argv[6]) if len(sys.argv) > 6 else 120)
