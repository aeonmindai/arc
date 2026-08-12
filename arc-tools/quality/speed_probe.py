#!/usr/bin/env python3
"""Decode/prefill speed probe via streaming completions (RUN-161).

Measures, against the running server:
  - prefill tok/s  (prompt_tokens_estimate / time-to-first-token)
  - decode tok/s   (streamed tokens after the first / remaining wall time)

Run it right after the server comes up: its decode number CALIBRATES the rest
of the session (GSM8K budget etc. — see the runbook's abort table; June H200
anchor: b=1 ~640 tok/s). Also used for the fused-Sinkhorn tok/s A/B.
"""
import argparse
import os

import qlib

# ~460 words of neutral filler -> a prefill in the several-hundred-token range,
# comfortably past the 128-token sliding window so the compressed path is timed too.
PROMPT_BODY = (
    "The history of computing spans mechanical calculators, vacuum tubes, "
    "transistors, integrated circuits, and modern accelerators. " * 24
    + "\n\nSummarize the passage above in a few sentences."
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=256, help="decode length")
    ap.add_argument("--runs", type=int, default=2, help="timed runs (first run also warms up)")
    ap.add_argument("--label", default="baseline", help="tag for the results file (e.g. fused_sinkhorn)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    qlib.ensure_dirs()
    qlib.health_or_die()
    out_path = args.out or os.path.join(qlib.RESULTS_DIR, f"speed_{args.label}.json")

    prompt = qlib.encode_chat(PROMPT_BODY)
    runs = []
    for i in range(args.runs):
        r = qlib.complete_stream(prompt, max_tokens=args.tokens, temperature=0.0)
        # ask a non-stream request for exact prompt_tokens once
        if i == 0:
            probe = qlib.complete(prompt, max_tokens=1, temperature=0.0)
            prompt_tokens = probe["prompt_tokens"]
        r["prompt_tokens"] = prompt_tokens
        r["prefill_tok_s"] = (round(prompt_tokens / r["ttft_s"], 1)
                              if r["ttft_s"] is not None and r["ttft_s"] > 0
                              and prompt_tokens else None)
        r.pop("text", None)
        runs.append(r)
        print(f"  run {i + 1}: ttft={r['ttft_s']}s "
              f"prefill~{r['prefill_tok_s']} tok/s decode={r['decode_tok_s']} tok/s "
              f"({r['stream_chunks']} chunks in {r['seconds']}s)")

    best = max(runs, key=lambda r: r["decode_tok_s"] or 0)
    summary = {
        "meta": qlib.run_meta({"eval": "speed_probe", "label": args.label,
                               "decode_tokens": args.tokens}),
        "summary": {
            "decode_tok_s_best": best["decode_tok_s"],
            "prefill_tok_s_best": max((r["prefill_tok_s"] or 0) for r in runs),
            "june_anchor_decode_tok_s": 640,
        },
        "runs": runs,
    }
    qlib.write_json(out_path, summary)
    print(f"\nSPEED[{args.label}]: decode {best['decode_tok_s']} tok/s | "
          f"prefill ~{summary['summary']['prefill_tok_s_best']} tok/s | "
          f"June H200 anchor ~640 tok/s decode")


if __name__ == "__main__":
    main()
