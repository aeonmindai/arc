#!/usr/bin/env python3
"""V4 ragged-decode batch sweep — client-observed serving capacity.

Parent system: ArcLab (benchmarks) — measures ArcInfer/ArcSched + ArcKV.

WHY THIS EXISTS
---------------
V4 has no PagedAttention, so it decodes out of a dense `NormalCache`, and a
dense forward requires every row's cache to be the same length. A pool of `k`
distinct prompt lengths therefore split into `k` buckets, and
`select_running_bucket`'s coalescence rule could not merge them
(`(total - n_min)*gap <= n_min*256` fails by ~4x), so **exactly one bucket
decoded per step**. Measured on an idle H200: B=1 -> 9.3 tok/s aggregate,
B=8 -> 0.1 tok/s, engine telemetry pinned at `1 running, 5 waiting`.

Ragged dense decode (front-alignment + `RAGGED_LEAD_PAD` + the pinned decode
bucket key) is supposed to remove that. This script measures whether it did.

WHAT IT REFUSES TO DO
---------------------
* It does NOT read the engine's `Throughput (T/s)` line as serving capacity.
  That counter includes **prefill** tokens: it has read 358-667 T/s while
  client-visible generation was 0.1 tok/s -- wrong by ~3 orders of magnitude.
  Every rate reported here is measured at the socket.
* It does NOT report a rate without first proving the run engaged. Engagement
  is asserted IN ADVANCE (see `--expect-running`), not inferred from "no
  errors". A run that produces numbers while `1 running, 5 waiting` is a
  FAILED run, not a slow one.
* It does not average away a short row. Per-row generated-token counts are
  reported, and a row that stopped early is called out rather than folded in.

EXIT CODES (house rule: environment failure exits 2, never 1)
    0  the sweep ran and every assertion held
    1  the sweep ran and an ASSERTION FAILED (a real result: engagement did not
       climb, ragged rows were refused, token counts disagreed)
    2  the environment was not fit to measure (server down, flags unset, wrong
       model, log unreadable) -- NOTHING was measured, do not quote this run
"""
import argparse
import json
import os
import re
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request

FILLER = (
    "The history of computing spans mechanical calculators, vacuum tubes, "
    "transistors, integrated circuits, and modern accelerators, each of which "
    "changed what a machine could hold in memory at one time. "
)

# The engine's own telemetry line (`engine/logger.rs:108`).
RUNNING_RE = re.compile(r"(\d+) running, (\d+) waiting")

# Printed once by `dsv4_attention` the first time it masks a ragged cohort per
# row. Its ABSENCE means per-row masking never ran: the rows were uniform, the
# gate was off, or the binary is stale. A throughput table without it describes
# something other than ragged decode, so the sweep refuses to pass without it.
BEACON = "ARC-RAGGED-MASK-ENGAGED"

# The release-mode postcondition that fires when a batch reached the dense
# forward without having been made uniform (`kv_cache/mod.rs`). If this appears,
# the ragged path is MALFORMED, not merely slow.
POISON = [
    "ensure_uniform_batch_cache_lens",
    "sequences disagree about their cache length",
    "cache length",
]


def make_prompt(words: int) -> str:
    body, n = [], 0
    while n < words:
        body.append(FILLER)
        n += len(FILLER.split())
    return (
        "".join(body)
        + "\n\nContinue the passage above at length, in detail, without stopping."
    )


def spread_lengths(batch: int, distinct: int, base: int, step: int) -> list:
    """`distinct` different prompt lengths, dealt round-robin across `batch`.

    Round-robin rather than blocked so that no B is accidentally uniform: at
    B=8 with distinct=9 this reproduces the 8-distinct-length pool that
    measured `1 running, 5 waiting`, which is the point of comparison.
    """
    return [base + step * (i % distinct) for i in range(batch)]


class Row:
    def __init__(self, idx, words):
        self.idx, self.words = idx, words
        self.t_start = self.t_first = self.t_end = None
        self.tokens = 0
        self.error = None


def stream_one(port: int, row: Row, max_tokens: int, timeout: int):
    # `/v1/completions`, NOT `/v1/chat/completions`. The V4 checkpoint ships no
    # chat template ("No chat template will be used. Only prompts will be
    # accepted, not messages" at load), so the chat endpoint would reject every
    # row and the sweep would measure nothing. The completion endpoint takes the
    # raw prompt and is what the serving path under test actually needs.
    payload = {
        "model": "default",
        "prompt": make_prompt(row.words),
        "max_tokens": max_tokens,
        # Temperature > 1e-7 AND top_p < 1.0 is what selects the real sampler
        # path rather than argmax/sample_fast (`sampler.rs`); measuring the
        # greedy path would prove nothing about serving.
        "temperature": 1.0,
        "top_p": 0.95,
        "stream": True,
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    row.t_start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                body = line[5:].strip()
                if body == "[DONE]":
                    break
                try:
                    chunk = json.loads(body)
                except json.JSONDecodeError:
                    continue
                for choice in chunk.get("choices", []):
                    piece = (choice.get("delta") or {}).get("content")
                    if piece:
                        if row.t_first is None:
                            row.t_first = time.time()
                        row.tokens += 1
    except Exception as exc:  # noqa: BLE001 - a dead row is itself a datum
        row.error = f"{type(exc).__name__}: {exc}"
    row.t_end = time.time()


def read_log_tail(path, from_byte):
    """Return (text, new_offset). Never raises -- a missing log is env failure,
    handled by the caller, not a crash mid-sweep."""
    try:
        with open(path, "rb") as fh:
            fh.seek(from_byte)
            data = fh.read()
            return data.decode("utf-8", "replace"), from_byte + len(data)
    except OSError:
        return "", from_byte


def sweep_one(args, batch: int, log_offset: int):
    words = spread_lengths(batch, args.distinct, args.base_words, args.step_words)
    rows = [Row(i, w) for i, w in enumerate(words)]
    threads = [
        threading.Thread(target=stream_one, args=(args.port, r, args.tokens, args.timeout))
        for r in rows
    ]

    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    t1 = time.time()

    log_text, log_offset = read_log_tail(args.log, log_offset)
    observed = [(int(a), int(b)) for a, b in RUNNING_RE.findall(log_text)]
    # Ignore the tail-off samples: as rows finish, `running` legitimately
    # decays to 0. Peak and the median of the top decile describe the
    # steady state that actually served the batch.
    runs = sorted((r for r, _ in observed), reverse=True)
    top = runs[: max(1, len(runs) // 10)] if runs else []

    ok = [r for r in rows if r.error is None and r.tokens > 0]
    total_tokens = sum(r.tokens for r in ok)
    wall = t1 - t0
    aggregate = total_tokens / wall if wall > 0 else 0.0
    # Per-user decode rate excludes that row's own prefill (first-token) wait,
    # so it is a decode rate and not a latency-diluted average.
    per_user = []
    for r in ok:
        if r.t_first is not None and r.tokens > 1 and r.t_end > r.t_first:
            per_user.append((r.tokens - 1) / (r.t_end - r.t_first))

    return {
        "batch": batch,
        "distinct_prompt_lengths": len(set(words)),
        "wall_s": round(wall, 2),
        "rows_ok": len(ok),
        # A row that raised is a failure -- and so is a row that returned
        # cleanly with NOTHING, which is how a protocol mismatch presents (the
        # first run of this sweep hit exactly that: 200 OK, immediate [DONE],
        # zero tokens, no exception). Counting only `r.error` let those rows
        # vanish from both lists, which is the silent-success shape the house
        # rules forbid. Both are named here.
        "rows_failed": [f"row{r.idx}: {r.error}" for r in rows if r.error]
        + [f"row{r.idx}: returned 0 tokens with no error" for r in rows if not r.error and r.tokens == 0],
        "tokens_per_row": [r.tokens for r in rows],
        "tokens_exact": sorted(set(r.tokens for r in ok)) == [args.tokens],
        "total_tokens": total_tokens,
        "aggregate_tok_s": round(aggregate, 2),
        "per_user_tok_s": round(statistics.median(per_user), 2) if per_user else 0.0,
        "achieved_running_peak": max(top) if top else 0,
        "achieved_running_p90": round(statistics.median(top), 1) if top else 0,
        "telemetry_samples": len(observed),
        "ragged_mask_engaged": BEACON in log_text,
        "poison": sorted({p for p in POISON if p in log_text}),
    }, log_offset


def preflight(args):
    """Refuse to measure an environment that cannot produce a valid number."""
    problems = []

    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{args.port}/v1/models", timeout=15
        ) as resp:
            json.load(resp)
    except Exception as exc:  # noqa: BLE001
        problems.append(f"server not answering on :{args.port} ({exc})")

    if not os.path.exists(args.log):
        problems.append(f"engine log {args.log} does not exist -- engagement is unprovable")

    # One real generation before committing the card to a full sweep. The first
    # run of this sweep spent a whole GPU session discovering at the END that
    # every row returned 200 OK and zero tokens, because V4 ships no chat
    # template and the request went to `/v1/chat/completions`. A five-token
    # probe would have caught it in one second, so it now runs first and the
    # sweep refuses to start without it.
    if not problems:
        probe = Row(-1, 12)
        stream_one(args.port, probe, 5, 60)
        if probe.error:
            problems.append(f"probe generation raised: {probe.error}")
        elif probe.tokens == 0:
            problems.append(
                "probe generation returned 200 OK with ZERO tokens -- the endpoint or payload "
                "shape is wrong for this model (V4 ships no chat template, so `messages` is "
                "rejected and only `prompt` works). Nothing would be measured."
            )
        else:
            print(f"probe OK: {probe.tokens} tokens in {probe.t_end - probe.t_start:.2f}s")

    if args.pid:
        try:
            with open(f"/proc/{args.pid}/environ", "rb") as fh:
                env = dict(
                    kv.split("=", 1)
                    for kv in fh.read().decode("utf-8", "replace").split("\0")
                    if "=" in kv
                )
        except OSError as exc:
            problems.append(f"cannot read /proc/{args.pid}/environ ({exc})")
            env = {}
        xs = env.get("ARC_V4_XS_PER_SEQ")
        mtp = env.get("ARC_MTP_PER_SEQ_KV")
        if xs != "1":
            problems.append(f"ARC_V4_XS_PER_SEQ={xs!r} on the SERVER process, expected '1'")
        # The silent-corruption trap: XS on with MTP off used to form ragged
        # batches with `row_q0: None`. After the re-gate the model derives it
        # from the cache layout, so this is a warning about intent, not a
        # correctness gate -- but a sweep meant to exercise both must set both.
        if mtp != "1":
            problems.append(
                f"ARC_MTP_PER_SEQ_KV={mtp!r} on the SERVER process, expected '1' "
                "(never run this sweep with only the first flag)"
            )
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=18234)
    ap.add_argument("--log", default="/tmp/v4_ragged_serve.log")
    ap.add_argument("--pid", type=int, default=0, help="server PID, for flag preflight")
    ap.add_argument("--batches", default="1,8,32,64,128")
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--distinct", type=int, default=9)
    ap.add_argument("--base-words", type=int, default=40)
    ap.add_argument("--step-words", type=int, default=40)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument(
        "--expect-running",
        default="",
        help="comma-aligned minimum achieved running width per batch; the run "
        "FAILS (exit 1) if any point misses. Declare this BEFORE the run.",
    )
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    problems = preflight(args)
    if problems:
        print("ENVIRONMENT UNFIT -- nothing measured:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        sys.exit(2)

    batches = [int(b) for b in args.batches.split(",") if b]
    expect = [int(x) for x in args.expect_running.split(",") if x]
    if expect and len(expect) != len(batches):
        print("--expect-running must have one entry per batch", file=sys.stderr)
        sys.exit(2)

    _, offset = read_log_tail(args.log, 0)  # start from the live tail
    results, failures = [], []

    for i, b in enumerate(batches):
        res, offset = sweep_one(args, b, offset)
        results.append(res)
        print(json.dumps(res), flush=True)

        if res["telemetry_samples"] == 0:
            failures.append(f"B={b}: NO telemetry samples -- engagement unproven")
        if res["poison"]:
            failures.append(f"B={b}: ragged postcondition fired {res['poison']}")
        if res["rows_failed"]:
            failures.append(f"B={b}: {len(res['rows_failed'])} rows errored")
        if not res["tokens_exact"]:
            failures.append(
                f"B={b}: per-row token counts not exact -> {sorted(set(res['tokens_per_row']))}"
            )
        if expect and res["achieved_running_peak"] < expect[i]:
            failures.append(
                f"B={b}: achieved running width {res['achieved_running_peak']} "
                f"< declared minimum {expect[i]}"
            )
        # The engine logs every 5s and only when the window saw tokens
        # (`engine/logger.rs`), and `num_running` is sampled instantaneously at
        # log time. Settle longer than one interval so the window covering this
        # batch is flushed before the next batch's lines mix into it -- and so a
        # short batch cannot end with zero telemetry, which the sweep would
        # (correctly) refuse to call a measurement.
        time.sleep(7)

    # The beacon fires ONCE per process, so it cannot be required per-B (B=1 is
    # uniform by construction and must NOT engage it). Require it across the
    # whole run, scanning the entire log rather than the sweep's tail: a warmup
    # request may have engaged it before the first window opened.
    whole_log, _ = read_log_tail(args.log, 0)
    beacon_seen = BEACON in whole_log
    widened = [r for r in results if r["batch"] > 1 and r["achieved_running_peak"] > 1]
    if widened and not beacon_seen:
        failures.append(
            "the engine ran multi-row decode batches but dsv4_attention NEVER masked a "
            "ragged cohort per row -- the binary is stale, the gate is off, or the pool "
            "was uniform. This run does not measure ragged decode."
        )
    print(f"\nragged mask engaged (beacon seen): {beacon_seen}")
    print("\nrequested_B  achieved_peak  achieved_p90  aggregate_tok/s  per_user_tok/s  exact")
    for r in results:
        print(
            f"{r['batch']:>10}  {r['achieved_running_peak']:>13}  "
            f"{r['achieved_running_p90']:>12}  {r['aggregate_tok_s']:>15}  "
            f"{r['per_user_tok_s']:>14}  {str(r['tokens_exact']):>5}"
        )

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(results, fh, indent=2)

    if failures:
        print("\nASSERTIONS FAILED:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        sys.exit(1)
    print("\nall engagement assertions held")


if __name__ == "__main__":
    main()
