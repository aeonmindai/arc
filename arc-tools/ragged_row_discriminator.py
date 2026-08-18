#!/usr/bin/env python3
"""Is a short/empty row in a ragged batch the MASKING, or the MODEL?

Parent system: ArcGate (correctness gates).

A row that returns zero tokens with no error, on a build that just changed
per-row attention masking, at the widest batch yet, is exactly where the
two-gate silent-corruption failure would surface: with masking mis-applied a
short row attends compressed blocks it has not produced, and a row attending
garbage emitting an immediate end-of-sequence looks, from outside, precisely
like a well-behaved stop. Wrong output, no error, plausible table.

THE DISCRIMINATOR
    Run the SAME prompt at B=1, where the batch is uniform by construction and
    per-row masking is dormant (`resolve_ragged_rows` returns `None` for a
    uniform batch, and the engagement beacon confirms it never fires).

        B=1 long, B=64 empty   -> it is the MASKING. Correctness bug. Stop.
        B=1 also empty          -> it is the MODEL. Natural stop at temperature
                                   1.0 on a raw completion with no chat
                                   template. The batch run is clean.

Repeated `--trials` times because temperature-1.0 sampling is stochastic: one
long B=1 sample does not rule out a model that stops early some fraction of the
time, and a single trial is how you turn noise into a false verdict. Reports the
distribution and every `finish_reason`, which is the field that separates "the
model chose to stop" from "generation was cut".
"""
import argparse
import json
import statistics
import sys
import urllib.request

FILLER = (
    "The history of computing spans mechanical calculators, vacuum tubes, "
    "transistors, integrated circuits, and modern accelerators, each of which "
    "changed what a machine could hold in memory at one time. "
)


def make_prompt(words: int) -> str:
    """Byte-identical to the sweep's prompt builder."""
    body, n = [], 0
    while n < words:
        body.append(FILLER)
        n += len(FILLER.split())
    return (
        "".join(body)
        + "\n\nContinue the passage above at length, in detail, without stopping."
    )


def one(port: int, prompt: str, max_tokens: int, timeout: int) -> dict:
    payload = {
        "model": "default",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 1.0,
        "top_p": 0.95,
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.load(resp)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}
    choice = body["choices"][0]
    usage = body.get("usage") or {}
    return {
        "completion_tokens": usage.get("completion_tokens"),
        "finish_reason": choice.get("finish_reason"),
        "text_head": (choice.get("text") or "")[:80].replace("\n", " "),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=18777)
    ap.add_argument("--words", type=int, required=True, help="the row's prompt length")
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--timeout", type=int, default=300)
    args = ap.parse_args()

    prompt = make_prompt(args.words)
    print(f"prompt: {args.words} words, {len(prompt)} chars, max_tokens={args.tokens}")
    print(f"running {args.trials} B=1 trials (uniform batch -> per-row masking dormant)\n")

    results = []
    for i in range(args.trials):
        r = one(args.port, prompt, args.tokens, args.timeout)
        results.append(r)
        print(f"  trial {i}: {json.dumps(r)}", flush=True)

    good = [r for r in results if "error" not in r]
    if not good:
        print("\nENVIRONMENT UNFIT: every trial raised", file=sys.stderr)
        sys.exit(2)

    counts = [r["completion_tokens"] or 0 for r in good]
    reasons = [r["finish_reason"] for r in good]
    empty = sum(1 for c in counts if c == 0)
    short = sum(1 for c in counts if 0 < c < args.tokens)

    print(f"\ntoken counts: {counts}")
    print(f"median: {statistics.median(counts)}  empty: {empty}/{len(good)}  short: {short}/{len(good)}")
    print(f"finish_reasons: {reasons}")

    print("\nVERDICT")
    if empty or short:
        print(
            f"  The SAME prompt stops early at B=1 ({empty} empty, {short} short of "
            f"{args.tokens}), where per-row masking is DORMANT. A short row in the "
            f"batched run is therefore the MODEL's sampling, not the mask. "
            f"The ragged batch run stands."
        )
    else:
        print(
            f"  Every B=1 trial ran to {args.tokens} tokens, but the batched run "
            f"produced an empty/short row for this same prompt. Per-row masking is "
            f"the only difference between those two conditions. Treat as a "
            f"CORRECTNESS BUG in the ragged mask and stop."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
