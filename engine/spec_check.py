"""Checks that speculative decoding over stages follows the target model.

  spec_check.py target.gguf layers split draft.gguf draft_layers [tokens] [draft_len]

The target runs as two stages cut at `split`, started with --rollback. The draft runs whole.
Every third draft token is replaced by a wrong one, so the stages have to take tokens back
on most passes (SPOIL=0 leaves the drafts alone). BITTY_STAGE_BIN is the engine.

A batch of tokens and the same tokens one at a time do not give the same numbers, only close
ones, so the text can part from plain greedy decoding where two tokens are nearly as likely.
The check is therefore: the tokens of the speculative text go through the target one at a
time, and at each place the token that was chosen must be the best one or within GAP (0.5)
of it. A state that was not taken back correctly gives gaps of several units.
"""
import os, random, struct, subprocess, sys, time

BIN = os.environ["BITTY_STAGE_BIN"]
ROWS = 0x100
PROMPT = "Explain the difference between TCP and UDP to a beginner."


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
        return kind, rows, data


def ids(data):
    return list(struct.unpack("<%dI" % (len(data) // 4), data))


def pairs_of_rows(kind, rows, data):
    """(token, logit) pairs, best first, of each row of a TOP_LOGITS or TOP_ROWS answer."""
    if kind == 5:
        return [[struct.unpack("<If", data[i:i + 8]) for i in range(0, len(data), 8)]]
    out, at = [], 0
    for _ in range(rows):
        count = struct.unpack("<I", data[at:at + 4])[0]
        at += 4
        out.append([struct.unpack("<If", data[at + i * 8:at + i * 8 + 8]) for i in range(count)])
        at += count * 8
    return out


class Target:
    """Two stages in a row."""

    def __init__(self, model, layers, split, rollback):
        common = ["-m", model, "-t", "8", "--top-logits", "64", "--rollback", str(rollback)]
        self.a = Engine(common + ["--start", "0", "--end", str(split)])
        self.b = Engine(common + ["--start", str(split), "--end", str(layers)])

    def pairs(self, session, pos, tokens, rows):
        flag = ROWS if rows else 0
        _, _, hidden = self.a.call(1 | flag, session, pos, len(tokens), struct.pack("<%dI" % len(tokens), *tokens))
        return pairs_of_rows(*self.b.call(2 | flag, session, pos, len(tokens), hidden))

    def run(self, session, pos, tokens, rows):
        return [row[0][0] for row in self.pairs(session, pos, tokens, rows)]


def main():
    model, layers, split, draft_model, draft_layers = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], int(sys.argv[5])
    want = int(sys.argv[6]) if len(sys.argv) > 6 else 60
    k = int(sys.argv[7]) if len(sys.argv) > 7 else 4
    target = Target(model, layers, split, k)
    draft = Engine(["-m", draft_model, "-t", "8", "--top-logits", "1", "--start", "0", "--end", str(draft_layers)])
    payload = b"".join(struct.pack("<I", len(part)) + part for part in (b"user", PROMPT.encode()))
    prompt = ids(target.a.call(6, 0, 0, 1, target.a.call(7, 0, 0, 1, payload)[2])[2])

    # plain greedy
    started = time.time()
    plain, token = [], target.run(1, 0, prompt, False)[0]
    while len(plain) < want:
        plain.append(token)
        token = target.run(1, len(prompt) + len(plain) - 1, [token], False)[0]
    plain_time = time.time() - started

    # speculative
    rng = random.Random(7)
    started = time.time()
    text = list(prompt)                       # every token that is certain
    text.append(target.run(2, 0, prompt, False)[0])
    fed = len(prompt)                         # tokens the target holds
    draft_fed, dirty = 0, []                  # tokens the draft holds, drafts it holds beyond
    passes = accepted_all = taken_back = 0
    while len(text) - len(prompt) < want:
        # draft: catch up with the text, mark, then go on alone
        _, _, data = draft.call(1, 1, draft_fed, len(text) - draft_fed, struct.pack("<%dI" % (len(text) - draft_fed), *text[draft_fed:]))
        draft_fed = len(text)
        draft.call(9, 1, 0, 0)
        drafts = [struct.unpack("<I", data[:4])[0]]
        while len(drafts) < k:
            _, _, data = draft.call(1, 1, draft_fed + len(drafts) - 1, 1, struct.pack("<I", drafts[-1]))
            drafts.append(struct.unpack("<I", data[:4])[0])
        dirty = len(drafts) - 1
        spoil = int(os.environ.get("SPOIL", "3"))
        sent = [t if not spoil or (i + passes) % spoil else rng.randrange(1000, 5000) for i, t in enumerate(drafts)]

        rows = target.run(2, fed, [text[-1]] + sent, True)
        passes += 1
        accepted = 0
        for i, token in enumerate(rows):
            text.append(token)
            if i < len(sent) and sent[i] == token:
                accepted += 1
            else:
                break
        fed += 1 + accepted                   # the next pass starts here: the stages go back
        accepted_all += accepted
        taken_back += len(sent) - accepted
        # the draft held the true drafts, not the spoiled ones
        true_accepted = 0
        for i in range(accepted):
            if drafts[i] != sent[i]:
                break
            true_accepted += 1
        if accepted >= dirty and true_accepted == accepted:
            draft_fed += dirty
    spec = text[len(prompt):len(prompt) + want]
    spec_time = time.time() - started

    # the speculative text, one token at a time through the target
    limit = float(os.environ.get("GAP", "0.5"))
    worst, missing = 0.0, 0
    row = target.pairs(3, 0, prompt, False)[0]
    for i, token in enumerate(spec):
        logit = dict(row).get(token)
        if logit is None:
            missing += 1
        else:
            worst = max(worst, row[0][1] - logit)
        row = target.pairs(3, len(prompt) + i, [token], False)[0]

    print("plain: %d tokens in %.1fs; speculative: %d passes, %d drafts accepted, %d taken back, %.1fs" % (
        want, plain_time, passes, accepted_all, taken_back, spec_time))
    same = sum(1 for a, b in zip(plain, spec) if a == b)
    print("%d of %d tokens are those of plain greedy decoding" % (same, want))
    print("largest gap to the best token: %.3f; tokens outside the best 64: %d" % (worst, missing))
    ok = missing == 0 and worst <= limit
    print("ok" if ok else "FAILED")
    sys.exit(0 if ok else 1)


main()
