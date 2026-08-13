#!/usr/bin/env python3
"""Twin-seed ensemble perplexity — offline analysis (session-2 experiment).

Inputs: two NDJSON logprob dumps from the perplexity example's
`--dump-logprobs` flag (one line per chunk: {"chunk":i,"tokens":[...],
"logprobs":[...]}), produced by TWO bakes of the same weights that differ only
in the Hadamard rotation seed (bake A: default seed; bake B:
ARC_QTIP_ROTATION_SEED=<other>), each run over the SAME corpus with the same
chunking. Decode reads the STORED rotation signs from the UQFF, so each bake
is self-consistent; what differs is the quantization error pattern.

Mechanism under test: if the two bakes' errors are decorrelated, averaging
their predictive distributions cancels error variance:

    p_ens(tok) = (p_A(tok) + p_B(tok)) / 2
    logp_ens   = logaddexp(logp_A, logp_B) - ln 2

SUCCESS SIGNAL: ppl_ensemble meaningfully below min(ppl_A, ppl_B) — evidence
of decorrelated-error cancellation on real weights (the bias-vs-variance
decomposition: ensemble gain comes from the variance share of quant error;
the bias share is whatever remains above the unquantized model).

Note: averaging only the TARGET-token probabilities is exact for perplexity
(ppl needs only p(target)); no full-vocab logits required.

Stdlib only. Runs offline on the pulled results tarball — zero GPU cost.

Usage:
  python3 ensemble_ppl.py results/lp_bakeA.ndjson results/lp_bakeB.ndjson \
      [--out results/ensemble_ppl.json]
"""
import argparse
import json
import math
import os
import sys


def load_dump(path):
    chunks = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    if not chunks:
        print(f"FATAL: {path} contains no chunks", file=sys.stderr)
        sys.exit(2)
    return chunks


def ppl(logprobs):
    return math.exp(-sum(logprobs) / len(logprobs))


def logaddexp(a, b):
    m = max(a, b)
    return m + math.log(math.exp(a - m) + math.exp(b - m))


def pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    denom = math.sqrt(sxx * syy)
    return sxy / denom if denom > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump_a", help="NDJSON logprob dump, bake A (default seed)")
    ap.add_argument("dump_b", help="NDJSON logprob dump, bake B (twin seed)")
    ap.add_argument("--out", default=None,
                    help="verdict JSON (default: ensemble_ppl.json next to dump A)")
    args = ap.parse_args()

    a_chunks, b_chunks = load_dump(args.dump_a), load_dump(args.dump_b)
    if len(a_chunks) != len(b_chunks):
        print(f"FATAL: chunk count differs (A={len(a_chunks)} B={len(b_chunks)}) — "
              f"different corpus/chunking, ensemble undefined", file=sys.stderr)
        sys.exit(2)

    per_chunk = []
    all_a, all_b, all_ens = [], [], []
    for ca, cb in zip(a_chunks, b_chunks):
        if ca["tokens"] != cb["tokens"]:
            print(f"FATAL: token stream differs in chunk {ca['chunk']} — "
                  f"tokenization mismatch, ensemble undefined", file=sys.stderr)
            sys.exit(2)
        la, lb = ca["logprobs"], cb["logprobs"]
        lens = [logaddexp(x, y) - math.log(2.0) for x, y in zip(la, lb)]
        all_a += la
        all_b += lb
        all_ens += lens
        per_chunk.append({
            "chunk": ca["chunk"],
            "n_tokens": len(la),
            "ppl_a": round(ppl(la), 5),
            "ppl_b": round(ppl(lb), 5),
            "ppl_ensemble": round(ppl(lens), 5),
        })

    ppl_a, ppl_b, ppl_ens = ppl(all_a), ppl(all_b), ppl(all_ens)
    best_single = min(ppl_a, ppl_b)
    gain = best_single / ppl_ens if ppl_ens > 0 else float("nan")

    diffs = [abs(x - y) for x, y in zip(all_a, all_b)]
    mean_abs_diff = sum(diffs) / len(diffs)
    big_disagree = sum(1 for d in diffs if d > 1.0) / len(diffs)
    r = pearson(all_a, all_b)
    # Per-chunk A-vs-B spread: how much of the error moves with the seed
    # (variance share) vs stays put (bias share).
    chunk_rel_spread = [
        abs(c["ppl_a"] - c["ppl_b"]) / min(c["ppl_a"], c["ppl_b"]) for c in per_chunk
    ]

    # >3% ensemble improvement over the best single bake = real cancellation
    # (per-chunk ppl noise on the mini corpus is well below that).
    success = gain >= 1.03
    verdict = {
        "inputs": {"a": args.dump_a, "b": args.dump_b,
                   "n_chunks": len(per_chunk), "n_tokens": len(all_a)},
        "summary": {
            "ppl_a": round(ppl_a, 5),
            "ppl_b": round(ppl_b, 5),
            "ppl_ensemble": round(ppl_ens, 5),
            "best_single": round(best_single, 5),
            "ensemble_gain_x": round(gain, 4),
            "success_signal": success,
            "note": ("ENSEMBLE WIN: decorrelated-error cancellation observed "
                     "(ppl_ensemble meaningfully < min(ppl_A, ppl_B))" if success else
                     "no meaningful ensemble gain (errors correlated, or bias-dominated)"),
        },
        "agreement": {
            "pearson_r_token_logprobs": round(r, 5),
            "mean_abs_logprob_diff_nats": round(mean_abs_diff, 5),
            "frac_tokens_disagree_gt_1nat": round(big_disagree, 5),
            "per_chunk_ab_rel_spread": [round(x, 5) for x in chunk_rel_spread],
            "interpretation": ("high spread + ensemble gain => variance-dominated quant error "
                               "(more seeds would keep helping); low spread + no gain => "
                               "bias-dominated (seed averaging exhausted)"),
        },
        "per_chunk": per_chunk,
    }
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.dump_a)),
                                   "ensemble_ppl.json")
    tmp = out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(verdict, f, indent=2)
    os.replace(tmp, out)

    print(f"ENSEMBLE-PPL: A={ppl_a:.3f} B={ppl_b:.3f} ensemble={ppl_ens:.3f} "
          f"(gain x{gain:.3f} over best single) | r={r:.3f} "
          f"mean|dlp|={mean_abs_diff:.3f} nats")
    print(f"VERDICT: {'SUCCESS — decorrelated-error cancellation' if success else 'no ensemble win'} "
          f"-> {out}")
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
