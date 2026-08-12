#!/usr/bin/env python3
"""Parse a perplexity-example log into a machine-readable results JSON.

Input:  the stdout log of `mistralrs/examples/advanced/perplexity` (Chunk lines
        + "Final perplexity ..." line).
Output: JSON {rung, ppl_mean, ppl_std, chunks: [...], n_chunks} + summary line.

The per-chunk ppl STRINGS are preserved verbatim: two runs printing identical
strings implies identical f32 perplexity bits (Rust's shortest-roundtrip float
Display), which is how the fused-Sinkhorn bit-identity check works.
"""
import json
import re
import sys

CHUNK_RE = re.compile(r"^Chunk (\d+)/(\d+) \((\d+) tokens\): Perplexity for .*?(\S+)s: (\S+)\s*$")
FINAL_RE = re.compile(r"^Final perplexity for .*: (\S+)±(\S+) ppl\s*$")


def parse(path):
    chunks = []
    mean = std = None
    with open(path, errors="replace") as f:
        for line in f:
            m = CHUNK_RE.match(line.strip())
            if m:
                chunks.append({"i": int(m.group(1)), "tokens": int(m.group(3)),
                               "seconds": m.group(4), "ppl": m.group(5)})
                continue
            m = FINAL_RE.match(line.strip())
            if m:
                mean, std = m.group(1), m.group(2)
    return chunks, mean, std


def main():
    if len(sys.argv) != 4:
        print("usage: parse_ppl.py <log> <rung-name> <out.json>", file=sys.stderr)
        sys.exit(2)
    log, rung, out = sys.argv[1], sys.argv[2], sys.argv[3]
    chunks, mean, std = parse(log)
    if mean is None:
        print(f"PPL[{rung}]: FAILED — no 'Final perplexity' line in {log} "
              f"({len(chunks)} chunk(s) seen). Check the log for the error.",
              file=sys.stderr)
        sys.exit(1)
    obj = {"rung": rung, "ppl_mean": mean, "ppl_std": std,
           "n_chunks": len(chunks), "chunks": chunks, "log": log}
    with open(out, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"PPL[{rung}]: {mean}±{std} over {len(chunks)} chunks -> {out}")


if __name__ == "__main__":
    main()
