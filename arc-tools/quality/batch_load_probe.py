#!/usr/bin/env python3
"""Batch-first serving load probe — the fleet-capacity measurement.

Production is always B=32/64/128 concurrent users; b=1 is a kernel-latency
diagnostic only (speed_probe.py keeps that job). This probe fires B concurrent
requests at the running server for each B in a sweep and reports, per B:

  - PREFILL aggregate tok/s          (compute-bound leg — grouped GEMM)
  - DECODE aggregate tok/s           (bandwidth-bound leg — the $/Mtok number)
  - DECODE per-request tok/s p50/p95 (what one user feels at that load)
  - TTFT p50/p95
  - effective concurrency            (how many sequences the server ACTUALLY
                                      ran together — see "the vacuous trap")
  - $/Mtok at --cost-per-hour
  - errors

plus a `--duration N` closed-loop sustained mode (keep B in-flight for N
seconds — closer to production than one-shot batches).

PREFILL AND DECODE ARE NEVER BLENDED. They have different bottlenecks
(prefill = compute/grouped-GEMM, decode = weight bandwidth), so a single
blended rate is meaningless for capacity planning. Three prefill numbers are
reported, each labelled with how it was obtained:

  prefill_agg_tok_s          DERIVED from TTFT + prompt length:
                             sum(prompt_tokens) / (last TTFT - batch release).
                             Pure client wall-clock; INCLUDES HTTP, queueing
                             and scheduler overhead, so it is a conservative
                             LOWER BOUND on engine prefill throughput. This is
                             the table's headline prefill number.
  prefill_agg_server_tok_s   REAL SERVER INSTRUMENTATION: sum(prompt_tokens) /
                             sum(distinct prefill-step durations), where each
                             step duration is the server's own
                             usage.total_prompt_time_sec. Compute-only, so it
                             is the UPPER reference. See _prefill_clusters()
                             for how steps are identified.
  prefill_per_req_tok_s_p50  server per-sequence rate. Note this UNDERSTATES:
                             engine/mod.rs stamps EVERY sequence in a prefill
                             batch with the whole step's wall time, so one
                             sequence's "rate" is its share of a shared step.

THE VACUOUS TRAP THIS PROBE IS BUILT TO AVOID: a probe that asks for B=64 but
whose requests actually execute one-at-a-time prints a plausible number that
is really 64 x b=1, and nothing in the output reveals it. mistral.rs makes
this easy to hit by accident — `--max-seqs` defaults to 32, so an unmodified
server silently caps a B=64 or B=128 sweep. Therefore every batch measures its
own concurrency from the overlap of per-request DECODE windows [first token,
last token] (serial execution leaves those windows disjoint => concurrency 1)
and:
  - HARD FAILS (`FAIL:` marker, exit 1) below --min-concurrency-frac x B,
  - WARNS and reports `effective_B` when the server ran fewer than asked.
The check is mutation-tested both offline (test_batch_load_probe.py serialises
the mock behind a lock and asserts the probe FAILS) and on GPU (serve with
`--max-seqs 1` and confirm `FAIL: CONCURRENCY` fires). A concurrency check
that cannot fail is worse than none.

Measurement mechanics (verified against mistralrs-core/src/pipeline/sampling.rs,
mistralrs-core/src/sequence.rs and response.rs on this branch):
  - Requests use SSE streaming *internally* because TTFT, the decode-only rate
    and the concurrency check are all unmeasurable without first-token timing;
    streamed text is discarded, never stored. `--no-stream` switches to plain
    requests (exact usage counts, but NO TTFT, NO prefill/decode split and NO
    concurrency check — the probe says so and warns).
  - /v1/chat/completions streams carry exact `usage` on the FINAL chunk
    (sampling.rs "Send usage on final chunk" -> sequence.rs get_usage()), which
    is a superset of OpenAI's: completion_tokens, prompt_tokens AND
    total_prompt_time_sec / total_completion_time_sec. /v1/completions streams
    have NO usage field (CompletionChunkResponse) — one chunk == one decoded
    token there (speed_probe precedent), so we count chunks and lose the
    server-side prefill split.
  - Distinct prompts (unique token-0 salt + rotating topics, ~64 tokens each)
    defeat prefix effects even without --prefix-cache-n 0.

Machine-greppable markers (orchestrators poll these, not the prose):
  BATCH[B=n] / PREFILL[B=n] / CONC[B=n] / BATCHSWEEP[label] / WARNING[KV]
  WARN[CONCURRENCY] / WARN[CLIENT] / FAIL:

V4 PREREQUISITE: concurrent sequences on DeepSeek-V4 require the xs_history
per-sequence fix (PR #21) in the served build — the old per-model buffer
crashes or silently corrupts with >1 sequence in flight.

The runbook's "ONE scored request at a time" rule is for SCORED evals; this
probe is the deliberate exception (unscored throughput measurement).

Stdlib only — no pip deps (same constraint as qlib).
"""
import argparse
import json
import math
import os
import threading
import time
import urllib.error
import urllib.request

import qlib

GPU_COST_HR_DEFAULT = 4.92  # Runcrate H200, sessions 1-4

# ---------------------------------------------------------------- prompt pool

_TOPICS = [
    "the tradeoffs between batch size and per-user latency in LLM serving",
    "how paged attention changes memory fragmentation on long contexts",
    "why speculative decoding acceptance rates fall on creative writing",
    "the history of the fast Walsh-Hadamard transform in signal processing",
    "how mixture-of-experts routing decides which experts see a token",
    "the difference between tensor and pipeline parallelism for inference",
    "why KV-cache quantization hurts retrieval-heavy tasks more than chat",
    "how continuous batching schedulers interleave prefill and decode",
    "the role of rotary position embeddings in length extrapolation",
    "why grouped-query attention reduces memory bandwidth at decode time",
    "how trellis-coded quantization differs from uniform scalar quantization",
    "the economics of GPU rental pricing for always-on inference fleets",
    "why perplexity is an incomplete proxy for downstream task quality",
    "how chunked prefill bounds time-to-first-token under heavy load",
    "the interaction between CUDA graphs and dynamic batch shapes",
    "why multi-token prediction heads need acceptance-rate telemetry",
]

_FILLER = (
    "Cover the main mechanism, one concrete numeric example, the most common "
    "misconception, and when the standard advice breaks down in practice. "
    "Keep the answer tight and well organized."
)


def make_prompt(i):
    """Distinct-from-token-0 prompt, ~64 tokens: unique salt first, rotating
    topic, shared filler tail."""
    return (
        f"Case {i:05d}, priority {i % 7}: explain {_TOPICS[i % len(_TOPICS)]}. "
        + _FILLER
    )


# ---------------------------------------------------------------- HTTP client


def _build_request(prompt_text, args):
    """Returns (url, body_bytes) for one request."""
    if args.raw:
        payload = {
            "model": "default",
            "prompt": qlib.encode_chat(prompt_text),
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "stop": [qlib.eos_token],
        }
        url = qlib.URL
    else:
        payload = {
            "model": "default",
            "messages": [
                {"role": "system", "content": qlib.SYSTEM_PROMPT},
                {"role": "user", "content": prompt_text},
            ],
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
        }
        url = qlib.CHAT_URL
    if not args.no_stream:
        payload["stream"] = True
    return url, json.dumps(payload).encode()


def _one_request_stream(prompt_text, args):
    """One SSE request. Returns a record dict (monotonic timestamps)."""
    url, body = _build_request(prompt_text, args)
    req = urllib.request.Request(url, data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    t_first = None
    n_chunks = 0
    usage = None
    finish = None
    try:
        with urllib.request.urlopen(req, timeout=args.timeout) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[len("data:"):].strip()
                if payload == "[DONE]":
                    break
                try:
                    obj = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                choices = obj.get("choices") or []
                if not choices:
                    continue
                if t_first is None:
                    t_first = time.monotonic()
                n_chunks += 1
                fr = choices[0].get("finish_reason")
                if fr:
                    finish = fr
                if obj.get("usage"):  # chat endpoint: exact usage, final chunk
                    usage = obj["usage"]
    except Exception as e:  # noqa: BLE001 — every failure is one error datum
        return _err_record(e, t0)
    t_end = time.monotonic()
    if usage:
        tokens = usage.get("completion_tokens")
        prompt_tokens = usage.get("prompt_tokens")
    else:  # raw /v1/completions stream: one chunk == one token
        tokens = n_chunks
        prompt_tokens = None
    decode_tok_s = None
    if t_first is not None and tokens and tokens > 1 and t_end > t_first:
        decode_tok_s = round((tokens - 1) / (t_end - t_first), 2)
    return {
        "ok": tokens is not None and tokens > 0,
        "error": None if tokens else "zero tokens streamed",
        "ttft_s": round(t_first - t0, 3) if t_first is not None else None,
        "seconds": round(t_end - t0, 3),
        "completion_tokens": tokens,
        "prompt_tokens": prompt_tokens,
        "decode_tok_s": decode_tok_s,
        # server's OWN prefill-step wall time for this sequence (mistral.rs
        # Usage superset). None on the raw endpoint, which sends no usage.
        "prompt_time_s": _usage_prompt_time(usage),
        "finish_reason": finish,
        "_t0": t0, "_t_first": t_first, "_t_end": t_end,
    }


def _usage_prompt_time(usage):
    """usage.total_prompt_time_sec, or None. Zero means the server did not
    record a prefill step (prefix-cache hit) — treat as absent, not as an
    infinitely fast prefill."""
    if not usage:
        return None
    v = usage.get("total_prompt_time_sec")
    if v is None:
        return None
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if v > 0 else None


def _one_request_nostream(prompt_text, args):
    """One plain (non-stream) request. Exact usage counts, no TTFT."""
    url, body = _build_request(prompt_text, args)
    req = urllib.request.Request(url, data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    try:
        resp = json.load(urllib.request.urlopen(req, timeout=args.timeout))
    except Exception as e:  # noqa: BLE001
        return _err_record(e, t0)
    t_end = time.monotonic()
    usage = resp.get("usage") or {}
    tokens = usage.get("completion_tokens")
    ch = (resp.get("choices") or [{}])[0]
    secs = t_end - t0
    return {
        "ok": bool(tokens),
        "error": None if tokens else "no usage in response",
        "ttft_s": None,
        "seconds": round(secs, 3),
        "completion_tokens": tokens,
        "prompt_tokens": usage.get("prompt_tokens"),
        # e2e rate (includes prefill) — the honest number without streaming
        "decode_tok_s": round(tokens / secs, 2) if tokens and secs > 0 else None,
        "prompt_time_s": _usage_prompt_time(usage),
        "finish_reason": ch.get("finish_reason"),
        "_t0": t0, "_t_first": None, "_t_end": t_end,
    }


def _err_record(e, t0):
    t_end = time.monotonic()
    status = getattr(e, "code", None)
    return {
        "ok": False,
        "error": f"{type(e).__name__}: {e}"[:200] + (f" (HTTP {status})" if status else ""),
        "ttft_s": None, "seconds": round(t_end - t0, 3),
        "completion_tokens": None, "prompt_tokens": None,
        "decode_tok_s": None, "prompt_time_s": None, "finish_reason": None,
        "_t0": t0, "_t_first": None, "_t_end": t_end,
    }


def _one_request(prompt_text, args):
    if args.no_stream:
        return _one_request_nostream(prompt_text, args)
    return _one_request_stream(prompt_text, args)


# ---------------------------------------------------------------- batch modes


def run_oneshot_batch(bsz, args, prompt_counter):
    """Fire bsz concurrent requests (barrier-synchronized). Returns
    (records, wall_s, t_batch0)."""
    records = [None] * bsz
    barrier = threading.Barrier(bsz + 1)

    def worker(slot, prompt_text):
        barrier.wait()
        records[slot] = _one_request(prompt_text, args)

    threads = []
    for slot in range(bsz):
        p = make_prompt(next(prompt_counter))
        t = threading.Thread(target=worker, args=(slot, p), daemon=True)
        t.start()
        threads.append(t)
    t_batch0 = time.monotonic()
    barrier.wait()  # release all workers together
    for t in threads:
        t.join()
    wall_s = max(r["_t_end"] for r in records) - t_batch0
    return records, wall_s, t_batch0


def run_sustained(bsz, args, prompt_counter):
    """Closed loop: keep bsz requests in flight for args.duration seconds.
    Workers stop LAUNCHING at the deadline; in-flight requests finish and
    count (wall extends to the last completion, so no inflation)."""
    records = []
    lock = threading.Lock()
    barrier = threading.Barrier(bsz + 1)
    stop_at = [0.0]  # filled after barrier release

    def worker():
        barrier.wait()
        while time.monotonic() < stop_at[0]:
            with lock:
                idx = next(prompt_counter)
            rec = _one_request(make_prompt(idx), args)
            with lock:
                records.append(rec)

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(bsz)]
    for t in threads:
        t.start()
    t_batch0 = time.monotonic()
    stop_at[0] = t_batch0 + args.duration
    barrier.wait()
    for t in threads:
        t.join()
    wall_s = (max(r["_t_end"] for r in records) - t_batch0) if records else 0.0
    return records, wall_s, t_batch0


# ---------------------------------------------------------------- math


def percentile(values, q):
    """Linear-interpolation percentile of a list (q in [0,100])."""
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    k = (len(vals) - 1) * q / 100.0
    lo, hi = int(k), min(int(k) + 1, len(vals) - 1)
    return round(vals[lo] + (vals[hi] - vals[lo]) * (k - lo), 3)


def concurrency(records):
    """Measure how many requests the SERVER actually ran together.

    THE ASSERTION THIS EXISTS FOR: firing B threads at a server proves nothing
    about whether the server batched them. Under true continuous batching every
    sequence decodes over the same wall-clock window, so the per-request DECODE
    windows [first token, last token] all overlap. Under serial execution
    request i does not emit its first token until request i-1 has finished, so
    those windows are DISJOINT and the overlap collapses to 1 — regardless of
    how the client launched them. That is the signal.

    Client submit windows [t0, t_end] are deliberately NOT used: the barrier
    makes them overlap even when the server serialises, so they would produce
    a check that can never fail.

    Returns:
      max      peak simultaneously-decoding requests (sweep line). Serial => 1.
      mean     time-weighted mean in-flight decodes = sum(window) / union span.
               Serial => ~1.0; perfect B-way batching => ~B.
      windows  how many records contributed a usable decode window.
    """
    spans = [(r["_t_first"], r["_t_end"]) for r in records
             if r["ok"] and r["_t_first"] is not None
             and r["_t_end"] > r["_t_first"]]
    if not spans:
        return {"max": None, "mean": None, "windows": 0,
                "note": "no first-token timestamps (--no-stream or all errors)"}
    # (t, delta) sorted naturally puts -1 (close) before +1 (open) at an equal
    # timestamp, so two merely-touching windows never count as overlapping.
    events = sorted([(a, 1) for a, _ in spans] + [(b, -1) for _, b in spans])
    cur = peak = 0
    for _, delta in events:
        cur += delta
        peak = max(peak, cur)
    union = max(b for _, b in spans) - min(a for a, _ in spans)
    busy = sum(b - a for a, b in spans)
    return {
        "max": peak,
        "mean": round(busy / union, 2) if union > 0 else None,
        "windows": len(spans),
    }


def _prefill_clusters(records):
    """Group requests that shared one batched prefill step, using the server's
    own timing.

    Ground truth (mistralrs-core/src/engine/mod.rs): after a prefill step the
    engine loops over the WHOLE scheduled prompt batch and stamps every
    sequence in it with the same `prompt_exec_time`, surfaced as
    usage.total_prompt_time_sec. So an identical stamped duration is evidence
    of a shared step — but ms-quantised durations can collide across two
    genuinely separate steps, so we additionally require the requests to have
    STARTED DECODING together (within one step duration + 50 ms slack). Two
    same-length prefills that ran at different times therefore stay separate.

    Returns a list of {step_s, n, prompt_tokens} in first-token order.
    """
    rows = sorted(
        ((r["prompt_time_s"], r["_t_first"], r["prompt_tokens"] or 0)
         for r in records
         if r["ok"] and r.get("prompt_time_s") and r["_t_first"] is not None),
        key=lambda row: row[1])
    clusters = []
    for step_s, t_first, ptoks in rows:
        cur = clusters[-1] if clusters else None
        if (cur is not None and abs(cur["step_s"] - step_s) < 1e-9
                and t_first - cur["_t_first"] <= step_s + 0.05):
            cur["n"] += 1
            cur["prompt_tokens"] += ptoks
        else:
            clusters.append({"step_s": step_s, "n": 1, "prompt_tokens": ptoks,
                             "_t_first": t_first})
    return clusters


def prefill_rep(records, t_batch0):
    """Prefill-only throughput, kept strictly separate from decode.

    Two independently-sourced numbers, because neither alone is honest:

      agg_tok_s        DERIVED FROM TTFT + PROMPT LENGTH. All B prompts are
                       fully processed by the time the LAST of them emits its
                       first token, so sum(prompt_tokens) / (last TTFT - batch
                       release) is a true aggregate rate — but it also carries
                       HTTP, queueing and scheduler time, making it a
                       conservative LOWER BOUND. Headline (never overstates).
      agg_server_tok_s REAL INSTRUMENTATION: sum(prompt_tokens) over clustered
                       prefill steps / sum of those steps' server-reported
                       durations. Compute-only UPPER reference.
    """
    ok = [r for r in records if r["ok"]]
    ptoks = sum(r["prompt_tokens"] or 0 for r in ok)
    firsts = [r["_t_first"] for r in ok if r["_t_first"] is not None]
    out = {
        "method": "derived_from_ttft_and_prompt_len",
        "prompt_tokens_total": ptoks,
        "agg_tok_s": None,
        "agg_server_tok_s": None,
        "per_req_tok_s_p50": None,
        "server_step_s_max": None,
        "server_prefill_batch_max": None,
        "steps": 0,
    }
    if not ptoks:
        out["method"] = "unavailable_no_prompt_token_counts"
        return out
    if firsts:
        window = max(firsts) - t_batch0
        if window > 0:
            out["agg_tok_s"] = round(ptoks / window, 2)
    else:
        out["method"] = "unavailable_no_first_token_timestamps"

    clusters = _prefill_clusters(records)
    if clusters:
        step_total = sum(c["step_s"] for c in clusters)
        clustered_toks = sum(c["prompt_tokens"] for c in clusters)
        if step_total > 0:
            out["agg_server_tok_s"] = round(clustered_toks / step_total, 2)
        out["server_step_s_max"] = round(max(c["step_s"] for c in clusters), 4)
        out["server_prefill_batch_max"] = max(c["n"] for c in clusters)
        out["steps"] = len(clusters)
        out["method"] = "derived_from_ttft (headline) + server_total_prompt_time_sec"
    per_req = [(r["prompt_tokens"] or 0) / r["prompt_time_s"]
               for r in ok if r.get("prompt_time_s") and r["prompt_tokens"]]
    out["per_req_tok_s_p50"] = percentile(per_req, 50)
    return out


def aggregate_rep(records, wall_s, t_batch0):
    """Per-rep aggregate metrics from raw records."""
    ok = [r for r in records if r["ok"]]
    total_tokens = sum(r["completion_tokens"] or 0 for r in ok)
    firsts = [r["_t_first"] for r in ok if r["_t_first"] is not None]
    t_end_max = max((r["_t_end"] for r in records), default=t_batch0)
    agg_decode = None
    if firsts and total_tokens:
        decode_wall = t_end_max - min(firsts)  # wall after batch's first token
        if decode_wall > 0:
            agg_decode = round(total_tokens / decode_wall, 2)
    agg_e2e = round(total_tokens / wall_s, 2) if wall_s > 0 and total_tokens else None
    return {
        "requests": len(records),
        "ok": len(ok),
        "errors": len(records) - len(ok),
        "total_completion_tokens": total_tokens,
        "wall_s": round(wall_s, 3),
        "agg_decode_tok_s": agg_decode,
        "agg_e2e_tok_s": agg_e2e,
        "req_per_s": round(len(ok) / wall_s, 3) if wall_s > 0 else None,
        "concurrency": concurrency(records),
        "prefill": prefill_rep(records, t_batch0),
    }


def cost_per_mtok(agg_tok_s, gpu_cost_hr):
    """$/Mtok = gpu_cost_hr * 1e6 / (agg_tok_s * 3600)."""
    if not agg_tok_s:
        return None
    return round(gpu_cost_hr * 1e6 / (agg_tok_s * 3600.0), 2)


def kv_guard(bsz, prompt_est, max_tokens, max_ctx):
    """Estimate KV token need for a batch; WARN (never block) if it exceeds
    --max-ctx. Returns the dict recorded in the JSON."""
    est = bsz * (prompt_est + max_tokens)
    guard = {"B": bsz, "est_kv_tokens": est, "max_ctx": max_ctx or None,
             "warned": False}
    if max_ctx and est > max_ctx:
        guard["warned"] = True
        print(f"  WARNING[KV] B={bsz}: est KV need ~{est} tokens "
              f"(B x (prompt~{prompt_est} + decode {max_tokens})) exceeds "
              f"--max-ctx {max_ctx}; expect eviction/queueing — measuring anyway.")
    elif bsz >= 64:
        print(f"  KV[B={bsz}]: est ~{est} tokens "
              f"(B x (prompt~{prompt_est} + decode {max_tokens}))"
              + ("" if max_ctx else " — pass --max-ctx to compare vs budget"))
    return guard


def concurrency_verdict(bsz, conc, min_frac):
    """Turn a measured concurrency into a PASS / WARN / FAIL verdict.

    b=1 is exempt (it is the serial diagnostic by definition). Otherwise:

      FAIL  effective < max(2, ceil(min_frac x B)). effective==1 means the
            server ran the batch STRICTLY SERIALLY and every throughput number
            for this B is B copies of the b=1 number — the result is vacuous
            and the probe exits non-zero.
      WARN  the server ran them together but fewer than asked (e.g. mistral.rs
            `--max-seqs 32` capping a B=64 sweep). The measurement is real, but
            it is a measurement of effective_B, not of B, and must be reported
            as such — never as a silently dropped row.
      PASS  effective >= 0.9 x B.
    """
    eff = conc.get("max")
    out = {"B": bsz, "effective_B": eff, "mean_inflight": conc.get("mean"),
           "min_required": None, "verdict": "pass", "reason": None}
    if bsz < 2:
        out["verdict"] = "exempt"
        out["reason"] = "B=1 is the serial diagnostic; concurrency not applicable"
        return out
    if eff is None:
        out["verdict"] = "unavailable"
        out["reason"] = conc.get("note") or "no decode windows to measure"
        return out
    floor = max(2, math.ceil(bsz * min_frac - 1e-9))
    out["min_required"] = floor
    if eff < floor:
        out["verdict"] = "fail"
        out["reason"] = (
            "requests executed serially — every number at this B is B x b=1"
            if eff <= 1 else
            f"server ran only {eff} of {bsz} together (floor {floor})")
    elif eff < 0.9 * bsz:
        out["verdict"] = "warn"
        out["reason"] = (f"server ran {eff} of {bsz} together — this row "
                         f"measures B={eff}, not B={bsz} (check --max-seqs "
                         f"and KV budget)")
    return out


def _strip_private(records):
    return [{k: v for k, v in r.items() if not k.startswith("_")} for r in records]


# ---------------------------------------------------------------- main


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--batches", default="1,8,16,32,64",
                    help="comma-separated batch sizes to sweep")
    ap.add_argument("--include-128", action="store_true",
                    help="append B=128 to the sweep")
    ap.add_argument("--reps", type=int, default=2,
                    help="measured repetitions per B (mean reported)")
    ap.add_argument("--max-tokens", type=int, default=128, help="decode length")
    ap.add_argument("--warmup-tokens", type=int, default=32,
                    help="decode length of the one warmup batch per B")
    ap.add_argument("--raw", action="store_true",
                    help="raw /v1/completions + encoding_dsv4 template "
                         "(default: /v1/chat/completions, server template)")
    ap.add_argument("--no-stream", action="store_true",
                    help="plain requests: exact usage token counts, but no "
                         "TTFT and e2e (not decode-only) rates")
    ap.add_argument("--duration", type=int, default=0,
                    help="sustained mode: keep B in-flight for N seconds "
                         "(closed loop) instead of one-shot reps")
    ap.add_argument("--max-ctx", type=int, default=0,
                    help="server KV/context budget in tokens for the guard "
                         "warning (0 = unknown)")
    ap.add_argument("--cost-per-hour", "--gpu-cost-hr", type=float,
                    dest="gpu_cost_hr", default=GPU_COST_HR_DEFAULT,
                    help="box $/hr used for every $/Mtok figure; the rate is "
                         "printed alongside the result (default: H200 "
                         "%(default)s)")
    ap.add_argument("--min-concurrency-frac", type=float, default=0.5,
                    help="hard-fail a batch whose measured concurrency is "
                         "below this fraction of B (floor 2). 1.0 = demand "
                         "full concurrency; 0 = report only, never fail "
                         "(default: %(default)s)")
    ap.add_argument("--temperature", type=float, default=0.0)
    # top_p is NOT cosmetic: it selects which sampler path the server takes.
    # mistralrs-core/src/sampler.rs `needs_sort = top_k > 0 || (0 < top_p < 1)`.
    # At the defaults here (temperature 0 => argmax, top_p 1.0 => no sort) the
    # big-vocab GPU radix top-k path is never entered, so a sweep run at the
    # defaults cannot observe any change to it. Use --temperature 1.0
    # --top-p 0.95 (the measured-good chat.py setting) to exercise the
    # production sampled path.
    ap.add_argument("--top-p", dest="top_p", type=float, default=1.0,
                    help="nucleus cutoff. <1.0 puts the server on the "
                         "top-p sampling path (default: %(default)s)")
    ap.add_argument("--timeout", type=int, default=900, help="per-request timeout (s)")
    ap.add_argument("--label", default="batch", help="tag for the results file")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    batches = [int(b) for b in args.batches.split(",") if b.strip()]
    if args.include_128 and 128 not in batches:
        batches.append(128)

    qlib.ensure_dirs()
    qlib.health_or_die()
    out_path = args.out or os.path.join(qlib.RESULTS_DIR, f"batch_load_{args.label}.json")

    prompt_counter = iter(range(10 ** 9))
    endpoint = "completions_raw" if args.raw else "chat_completions"
    mode = f"sustained_{args.duration}s" if args.duration else f"oneshot_x{args.reps}"
    print(f"batch_load_probe [{args.label}] endpoint={endpoint} "
          f"stream={not args.no_stream} mode={mode} "
          f"batches={batches} max_tokens={args.max_tokens}")

    if args.no_stream:
        print("WARN[CONCURRENCY] --no-stream: no first-token timestamps, so "
              "the prefill/decode split AND the concurrency assertion are "
              "BOTH unavailable. Rates below are end-to-end (prefill+decode "
              "blended) and nothing verifies the server batched anything. "
              "Sanity-check mode only — never quote these as fleet numbers.")

    prompt_est = 64  # refined from the first warmup's measured prompt_tokens
    per_b = []
    for bsz in batches:
        # -- warmup (also refines prompt_est for the KV guard)
        wargs = argparse.Namespace(**vars(args))
        wargs.max_tokens = min(args.warmup_tokens, args.max_tokens)
        wrecs, _, _ = run_oneshot_batch(bsz, wargs, prompt_counter)
        measured_pt = [r["prompt_tokens"] for r in wrecs if r.get("prompt_tokens")]
        if measured_pt:
            prompt_est = max(measured_pt)
        werrs = sum(1 for r in wrecs if not r["ok"])
        if werrs:
            print(f"  warmup B={bsz}: {werrs}/{bsz} errors "
                  f"(first: {next(r['error'] for r in wrecs if not r['ok'])})")

        guard = kv_guard(bsz, prompt_est, args.max_tokens, args.max_ctx)

        # -- measured reps (or one sustained window)
        reps_out = []
        all_records = []
        n_reps = 1 if args.duration else args.reps
        for rep in range(n_reps):
            cpu0, w0 = time.process_time(), time.monotonic()
            if args.duration:
                recs, wall, t0 = run_sustained(bsz, args, prompt_counter)
            else:
                recs, wall, t0 = run_oneshot_batch(bsz, args, prompt_counter)
            cpu_frac = ((time.process_time() - cpu0)
                        / max(time.monotonic() - w0, 1e-9))
            agg = aggregate_rep(recs, wall, t0)
            agg["client_cpu_frac"] = round(cpu_frac, 3)
            agg["records"] = _strip_private(recs)
            reps_out.append(agg)
            all_records.extend(recs)
            print(f"  B={bsz} rep{rep + 1}: agg {agg['agg_decode_tok_s']} tok/s "
                  f"(e2e {agg['agg_e2e_tok_s']}) | wall {agg['wall_s']}s | "
                  f"conc {agg['concurrency']['max']} | "
                  f"errs {agg['errors']}/{agg['requests']}")
            # A saturated single-threaded client UNDERSTATES the server. At
            # B=128 the probe itself can become the bottleneck, which would be
            # a measurement artefact reported as a server limit.
            if cpu_frac > 0.8:
                print(f"  WARN[CLIENT] B={bsz} rep{rep + 1}: probe used "
                      f"{cpu_frac:.0%} of a CPU core parsing SSE — the CLIENT "
                      f"may be the bottleneck and these tok/s an UNDERSTATEMENT. "
                      f"Re-run with fewer tokens or split across two probes.")

        # -- mean across reps + pooled distributions
        # reps_out bound as a default so the closures capture THIS B's reps,
        # not the loop variable (ruff B023).
        def _mean(key, reps_out=reps_out):
            vals = [r[key] for r in reps_out if r[key] is not None]
            return round(sum(vals) / len(vals), 2) if vals else None

        dist = {
            "per_req_decode_tok_s_p50": percentile(
                [r["decode_tok_s"] for r in all_records], 50),
            "per_req_decode_tok_s_p95": percentile(
                [r["decode_tok_s"] for r in all_records], 95),
            "ttft_s_p50": percentile([r["ttft_s"] for r in all_records], 50),
            "ttft_s_p95": percentile([r["ttft_s"] for r in all_records], 95),
        }
        def _mean_sub(section, key, reps_out=reps_out):
            vals = [r[section][key] for r in reps_out
                    if r[section].get(key) is not None]
            return round(sum(vals) / len(vals), 2) if vals else None

        errors = sum(r["errors"] for r in reps_out)
        n_requests = sum(r["requests"] for r in reps_out)
        # headline: decode-only aggregate; e2e fallback in --no-stream mode
        mean_agg = _mean("agg_decode_tok_s") or _mean("agg_e2e_tok_s")

        # -- concurrency: worst rep governs (one serialised rep taints the row)
        conc_max = [r["concurrency"]["max"] for r in reps_out
                    if r["concurrency"]["max"] is not None]
        conc = {
            "max": min(conc_max) if conc_max else None,
            "mean": _mean_sub("concurrency", "mean"),
            "per_rep_max": [r["concurrency"]["max"] for r in reps_out],
            "note": reps_out[0]["concurrency"].get("note"),
        }
        verdict = concurrency_verdict(bsz, conc, args.min_concurrency_frac)
        prefill = {
            "method": reps_out[0]["prefill"]["method"],
            "agg_tok_s": _mean_sub("prefill", "agg_tok_s"),
            "agg_server_tok_s": _mean_sub("prefill", "agg_server_tok_s"),
            "per_req_tok_s_p50": _mean_sub("prefill", "per_req_tok_s_p50"),
            "server_prefill_batch_max": max(
                (r["prefill"]["server_prefill_batch_max"] or 0
                 for r in reps_out), default=0) or None,
        }
        entry = {
            "B": bsz,
            "kv_guard": guard,
            "mean": {
                "agg_tok_s_headline": mean_agg,
                "agg_decode_tok_s": _mean("agg_decode_tok_s"),
                "agg_e2e_tok_s": _mean("agg_e2e_tok_s"),
                "wall_s": _mean("wall_s"),
                "req_per_s": _mean("req_per_s"),
                "client_cpu_frac": _mean("client_cpu_frac"),
            },
            "prefill": prefill,
            "concurrency": conc,
            "concurrency_verdict": verdict,
            "dist": dist,
            "errors": errors,
            "requests": n_requests,
            "cost_per_mtok_usd": cost_per_mtok(mean_agg, args.gpu_cost_hr),
            "gpu_cost_hr_usd": args.gpu_cost_hr,
            "reps": reps_out,
        }
        per_b.append(entry)
        ttft = dist["ttft_s_p50"]

        # -- concurrency verdict FIRST: it decides whether the row means
        #    anything at all, so it must not be buried under the numbers.
        if verdict["verdict"] == "fail":
            print(f"FAIL: CONCURRENCY[B={bsz}] effective_B="
                  f"{verdict['effective_B']} (mean in-flight "
                  f"{verdict['mean_inflight']}) < required "
                  f"{verdict['min_required']} — {verdict['reason']}. "
                  f"THIS ROW IS NOT A B={bsz} MEASUREMENT.")
        elif verdict["verdict"] == "warn":
            print(f"WARN[CONCURRENCY] B={bsz}: {verdict['reason']}")
        print(f"CONC[B={bsz}] effective_B={verdict['effective_B']} "
              f"mean_inflight={verdict['mean_inflight']} "
              f"server_prefill_batch={prefill['server_prefill_batch_max']} "
              f"verdict={verdict['verdict']}")
        print(f"PREFILL[B={bsz}] agg {prefill['agg_tok_s']} tok/s "
              f"(derived from TTFT + prompt len; includes queueing — a LOWER "
              f"bound) | server-instrumented agg "
              f"{prefill['agg_server_tok_s']} tok/s (compute only) | "
              f"per-req p50 {prefill['per_req_tok_s_p50']} tok/s")
        print(f"BATCH[B={bsz}] decode agg {mean_agg} tok/s | "
              f"decode per-user p50 {dist['per_req_decode_tok_s_p50']} tok/s "
              f"(p95 {dist['per_req_decode_tok_s_p95']}) | "
              f"TTFT p50 {ttft if ttft is not None else '-'}s "
              f"(p95 {dist['ttft_s_p95'] if dist['ttft_s_p95'] is not None else '-'}) | "
              f"$/Mtok {entry['cost_per_mtok_usd']} @ ${args.gpu_cost_hr}/hr | "
              f"errors {errors}/{n_requests}")

    # -- sweep summary
    #    Only concurrency-VALID rows can be the headline. A serialised row is
    #    excluded from `peak` on purpose: it is B copies of b=1 wearing a
    #    batch's name, and letting it win the max() is exactly the vacuous
    #    result this probe exists to prevent.
    failed_conc = [e for e in per_b if e["concurrency_verdict"]["verdict"] == "fail"]
    scored = [e for e in per_b
              if e["mean"]["agg_tok_s_headline"]
              and e["concurrency_verdict"]["verdict"] != "fail"]
    peak = max(scored, key=lambda e: e["mean"]["agg_tok_s_headline"]) if scored else None
    b1 = next((e for e in per_b if e["B"] == 1), None)
    summary = {
        "peak_agg_decode_tok_s": peak["mean"]["agg_tok_s_headline"] if peak else None,
        "peak_B": peak["B"] if peak else None,
        "peak_effective_B": peak["concurrency_verdict"]["effective_B"] if peak else None,
        "peak_prefill_agg_tok_s": peak["prefill"]["agg_tok_s"] if peak else None,
        "per_user_p50_at_peak": peak["dist"]["per_req_decode_tok_s_p50"] if peak else None,
        "ttft_p50_at_peak": peak["dist"]["ttft_s_p50"] if peak else None,
        "ttft_p95_at_peak": peak["dist"]["ttft_s_p95"] if peak else None,
        "cost_per_mtok_at_peak_usd": peak["cost_per_mtok_usd"] if peak else None,
        "b1_decode_tok_s": (b1["dist"]["per_req_decode_tok_s_p50"] if b1 else None),
        "gpu_cost_hr_usd": args.gpu_cost_hr,
        "total_errors": sum(e["errors"] for e in per_b),
        "concurrency_failures": [e["B"] for e in failed_conc],
        "min_concurrency_frac": args.min_concurrency_frac,
    }
    out = {
        "meta": qlib.run_meta({
            "eval": "batch_load_probe", "label": args.label,
            "endpoint": endpoint, "stream": not args.no_stream,
            "mode": mode, "batches": batches,
            "max_tokens": args.max_tokens, "warmup_tokens": args.warmup_tokens,
            "temperature": args.temperature, "top_p": args.top_p,
            "prompt_tokens_est": prompt_est,
        }),
        "summary": summary,
        "per_b": per_b,
    }
    qlib.write_json(out_path, out)
    if peak:
        print(f"\nBATCHSWEEP[{args.label}]: peak decode "
              f"{summary['peak_agg_decode_tok_s']} tok/s @B={summary['peak_B']} "
              f"(effective_B={summary['peak_effective_B']}) | prefill @peak "
              f"{summary['peak_prefill_agg_tok_s']} tok/s | per-user p50 @peak "
              f"{summary['per_user_p50_at_peak']} tok/s | TTFT p50/p95 @peak "
              f"{summary['ttft_p50_at_peak']}/{summary['ttft_p95_at_peak']}s | "
              f"$/Mtok @peak {summary['cost_per_mtok_at_peak_usd']} "
              f"(${args.gpu_cost_hr}/hr) | b=1 p50 {summary['b1_decode_tok_s']} tok/s | "
              f"errors {summary['total_errors']}")
    else:
        print(f"\nBATCHSWEEP[{args.label}]: NO USABLE BATCHES — see {out_path}")
    print(f"-> {out_path}")

    if failed_conc:
        bs = ", ".join(f"B={e['B']}(eff {e['concurrency_verdict']['effective_B']})"
                       for e in failed_conc)
        print(f"FAIL: batch_load_probe[{args.label}] — the server did not run "
              f"these batches concurrently: {bs}. Their throughput numbers are "
              f"NOT batched-serving numbers. Check the server's --max-seqs "
              f"(mistral.rs defaults to 32, which silently caps B=64/128) and "
              f"the KV budget, then re-run.")
        return 1
    if not peak:
        print(f"FAIL: batch_load_probe[{args.label}] — no batch produced a "
              f"usable aggregate rate; see {out_path}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
