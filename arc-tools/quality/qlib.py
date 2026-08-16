#!/usr/bin/env python3
"""Shared helpers for the RUN-161 quality harness (arc-tools/quality).

Every script in this directory talks to a locally-served DeepSeek-V4-Flash
(mistralrs serve ... --prefix-cache-n 0) through the RAW /v1/completions
endpoint, using the model's OFFICIAL encoder (repo-root encoding_dsv4.py) —
exactly the chat.py machinery, so the bytes the model sees are faithful.

Design constraints inherited from RUN-161:
  - ONE request at a time (global compressor xs_history is not seq-managed).
  - Server must run with --prefix-cache-n 0.
  - Greedy = temperature 0 (mistral.rs maps it to argmax; June eval.py used it).
  - Sampled = temperature 1.0 / top_p 0.95 (the measured-good chat.py setting).

Stdlib only — no pip deps.
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request

# repo root (= two levels up from arc-tools/quality) holds encoding_dsv4.py
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from encoding_dsv4 import encode_messages, eos_token  # noqa: E402

BASE = os.environ.get("BASE_URL", "http://localhost:1234")
URL = f"{BASE}/v1/completions"
CHAT_URL = f"{BASE}/v1/chat/completions"

SYSTEM_PROMPT = "You are a helpful assistant."  # REQUIRED: bare prompts are OOD for V4-Flash

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
RESULTS_DIR = os.path.join(HERE, "results")


def ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)


def health_or_die(timeout=10):
    try:
        urllib.request.urlopen(f"{BASE}/health", timeout=timeout)
    except Exception as e:
        print(f"FATAL: cannot reach server at {BASE} ({e})", file=sys.stderr)
        print("       start it first (see GPU_SESSION_RUNBOOK.md) or set BASE_URL.", file=sys.stderr)
        sys.exit(2)


def encode_chat(user_msg, system=SYSTEM_PROMPT, thinking_mode="chat"):
    """Official V4 encoding for a single-turn chat."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user_msg})
    return encode_messages(
        messages,
        thinking_mode=thinking_mode,
        drop_thinking=True,
        add_default_bos_token=True,
    )


def complete(prompt, max_tokens=60, temperature=0.0, top_p=1.0, timeout=1800):
    """One raw completion. Returns dict: text, finish_reason, prompt_tokens,
    completion_tokens, seconds."""
    body = json.dumps({
        "model": "default",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stop": [eos_token],
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    resp = json.load(urllib.request.urlopen(req, timeout=timeout))
    dt = time.time() - t0
    ch = resp["choices"][0]
    usage = resp.get("usage", {})
    return {
        "text": ch["text"].replace(eos_token, ""),
        "finish_reason": ch.get("finish_reason"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "seconds": round(dt, 3),
    }


def chat_complete(user_msg, system=SYSTEM_PROMPT, max_tokens=60, temperature=0.0,
                  top_p=1.0, n_votes=None, vote_mode=None, timeout=3600):
    """One /v1/chat/completions request (server-side template — NOT the
    encoding_dsv4 path; run the runbook's template-parity probe before trusting
    it for scored evals).

    Arc Boost voting (PR #13, non-streaming only): pass `n_votes >= 2` to fan
    the request out to k chains decoded in the same forward-pass batch;
    `choices[0]` is the server-voted winner and the response's top-level
    `vote` field carries the tally (`vote_mode`: "majority" or
    "confidence_weighted", server default confidence_weighted).

    Returns dict: text, finish_reason, prompt_tokens, completion_tokens,
    seconds, vote (None unless the server voted).
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user_msg})
    payload = {
        "model": "default",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    if n_votes is not None:
        payload["n_votes"] = n_votes            # verified field: openai.rs ChatCompletionRequest
        if vote_mode is not None:
            payload["vote_mode"] = vote_mode    # "majority" | "confidence_weighted"
    body = json.dumps(payload).encode()
    req = urllib.request.Request(CHAT_URL, data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    resp = json.load(urllib.request.urlopen(req, timeout=timeout))
    dt = time.time() - t0
    ch = resp["choices"][0]
    usage = resp.get("usage", {})
    return {
        "text": (ch.get("message") or {}).get("content") or "",
        "finish_reason": ch.get("finish_reason"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "seconds": round(dt, 3),
        "vote": resp.get("vote"),
    }


def complete_stream(prompt, max_tokens=256, temperature=0.0, top_p=1.0, timeout=1800):
    """Streaming completion. Returns dict with text, ttft (s), decode_tok_s,
    completion_tokens, total seconds. Used by the speed probe."""
    body = json.dumps({
        "model": "default",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stop": [eos_token],
        "stream": True,
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    ttft = None
    n_chunks = 0
    text = []
    with urllib.request.urlopen(req, timeout=timeout) as r:
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
            piece = choices[0].get("text", "") or choices[0].get("delta", {}).get("content", "")
            if ttft is None:
                ttft = time.time() - t0
            n_chunks += 1
            text.append(piece)
    total = time.time() - t0
    decode_time = max(total - (ttft or 0.0), 1e-6)
    # chunk count ~= token count for completions streaming (one token per chunk)
    decode_tok_s = (n_chunks - 1) / decode_time if n_chunks > 1 else 0.0
    return {
        "text": "".join(text).replace(eos_token, ""),
        "ttft_s": round(ttft, 3) if ttft is not None else None,
        "stream_chunks": n_chunks,
        "decode_tok_s": round(decode_tok_s, 2),
        "seconds": round(total, 3),
    }


# ------------------------------------------------------------- degeneracy
#
# WHY THIS WAS REWRITTEN (wave39-BQ). The original detector was
#
#     words = text.split()
#     for p in 1..=8:  run of words[i] == words[i-p];  return True if run+p >= 14
#
# and it was wrong in BOTH directions, verified against the four archived
# GSM8K runs (s1/s2/s3/s9 results JSON):
#
#  * FALSE POSITIVE. `run + p >= 14` means ONE repetition of a 7-word phrase
#    trips it. Step-by-step algebra does that naturally: wave34-BL's single
#    reported "degenerate" (gsm8k_s9.json idx=211) is a CORRECT, cleanly
#    stopped 185-token answer whose LaTeX derivation lines
#      \( 3r + 5r + 10 = 42 \)  /  \( 8r + 10 = 42 \)
#    repeat at period 7. The published "1 degenerate" was a harness artifact.
#  * FALSE NEGATIVE, on its own headline example. `text.split()` collapses an
#    unspaced token loop into ONE word, so looks_degenerate("User" * 200) is
#    False — the docstring's first case was undetectable.
#  * FALSE NEGATIVE, on real loops. s1 idx=465 ("Thus final answer: ####30."
#    x N, period 13 words) and s1 idx=553 (period 22 words) are unambiguous
#    loops that the old p<=8 window could not see; both were scored clean.
#
# The replacement is a union of three checks, each tuned against real output:
#   1. word cycle   - short cycles, but now needs >= 4 full repeats / 40 words
#   2. char cycle   - catches unspaced loops the word path is blind to
#   3. n-gram saturation - distinct 4-grams in the tail; catches long-period
#                          loops that no small-period scan can reach
#
# SEPARATION (measured on 350 archived completions, 300-char tails):
#   worst distinct-4gram ratio among CORRECT + finish_reason=="stop" .. 0.574
#   best  distinct-4gram ratio among the four known loops ............ 0.418
# The 0.45 threshold sits in that gap. test_degeneracy.py asserts the gap
# still exists, so a future retune cannot silently close it.

DEGEN_WORD_MAX_PERIOD = 10      # word-cycle: periods 1..N considered
DEGEN_WORD_MIN_REPEATS = 4      # word-cycle: full cycles required
DEGEN_WORD_MIN_SPAN = 40        # word-cycle: words the cycle must cover
DEGEN_CHAR_MAX_PERIOD = 60      # char-cycle: periods 1..N considered
DEGEN_CHAR_MIN_REPEATS = 6      # char-cycle: full cycles required
DEGEN_CHAR_MIN_SPAN = 150       # char-cycle: chars the cycle must cover
DEGEN_NGRAM_N = 4               # saturation: n-gram size
DEGEN_NGRAM_WINDOW = 120        # saturation: trailing words examined
DEGEN_NGRAM_MIN_GRAMS = 40      # saturation: too-short tails are not judged
DEGEN_NGRAM_MAX_RATIO = 0.45    # saturation: distinct/total below this = loop


def _longest_cycle(seq, max_period, min_repeats, min_span):
    """Longest repeated cycle in `seq` meeting both bars, else None.

    Returns (period, span, end_index). `span` counts the whole periodic
    region (the seed cycle plus every repeat of it), so `span // period` is
    the number of full repetitions.
    """
    n = len(seq)
    best = None
    for p in range(1, max_period + 1):
        if n < p * min_repeats:
            break
        run = 0
        for i in range(p, n):
            if seq[i] == seq[i - p]:
                run += 1
                span = run + p
                if (run // p + 1 >= min_repeats and span >= min_span
                        and (best is None or span > best[1])):
                    best = (p, span, i)
            else:
                run = 0
        if best is not None:
            return best
    return None


def _ngram_saturation(text, n=DEGEN_NGRAM_N, window=DEGEN_NGRAM_WINDOW):
    """(distinct_ratio, n_grams) over the trailing `window` words, or
    (None, 0) when the tail is too short to judge."""
    words = text.split()[-window:]
    if len(words) < n + 8:
        return None, 0
    grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    return len(set(grams)) / len(grams), len(grams)


def degeneracy_report(text):
    """None for a healthy completion; a dict describing the loop otherwise.

    The dict always carries `kind` ("word_cycle" | "char_cycle" |
    "ngram_saturation") and enough numbers to audit the call without rerunning
    the model — results JSON stores it verbatim, so a flagged run can be
    triaged offline.
    """
    if not text:
        return None

    cyc = _longest_cycle(text.split(), DEGEN_WORD_MAX_PERIOD,
                         DEGEN_WORD_MIN_REPEATS, DEGEN_WORD_MIN_SPAN)
    if cyc is not None:
        period, span, _ = cyc
        return {"kind": "word_cycle", "period_words": period,
                "span_words": span, "repeats": span // period}

    cyc = _longest_cycle(text, DEGEN_CHAR_MAX_PERIOD,
                         DEGEN_CHAR_MIN_REPEATS, DEGEN_CHAR_MIN_SPAN)
    if cyc is not None:
        period, span, end = cyc
        return {"kind": "char_cycle", "period_chars": period,
                "span_chars": span, "repeats": span // period,
                "unit": text[end - period + 1:end + 1]}

    ratio, n_grams = _ngram_saturation(text)
    if ratio is not None and n_grams >= DEGEN_NGRAM_MIN_GRAMS \
            and ratio < DEGEN_NGRAM_MAX_RATIO:
        return {"kind": "ngram_saturation", "n": DEGEN_NGRAM_N,
                "distinct_ratio": round(ratio, 4), "n_grams": n_grams,
                "window_words": DEGEN_NGRAM_WINDOW}

    return None


def looks_degenerate(text):
    """Bool form of degeneracy_report() — kept for run_coherence.py and
    run_longctx.py, which gate PASS/FAIL on it."""
    return degeneracy_report(text) is not None


def write_json(path, obj):
    """Atomic-ish JSON write (tmp + rename) so a crash never truncates results."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def run_meta(extra=None):
    meta = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "base_url": BASE,
        # NOTE: these are the CLIENT process's gates. The gates that matter are
        # on the SERVER process — record those via the runbook's serve log names.
        "client_env_gates": {
            k: os.environ.get(k)
            for k in (
                "ARC_FUSED_SINKHORN", "ARC_QUANT_ATTENTION", "ARC_V4_WINDOW_ONLY",
                "ARC_FORCE_NAIVE_SDPA", "ARC_TIME_DECODE", "ARC_V4_STANDARD_DENSE",
                "ARC_V4_NO_ABSORBED_DECODE", "ARC_ALLOW_UNSTAMPED_QTIP",
                "ARC_MTP_LOG_ACCEPTANCE",
            )
            if os.environ.get(k) is not None
        },
    }
    if extra:
        meta.update(extra)
    return meta
