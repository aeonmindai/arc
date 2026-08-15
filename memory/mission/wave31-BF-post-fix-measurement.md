# wave31-BF — post-fix measurement: **BLOCKED, NOTHING MEASURED**

**Date:** 2026-08-15 · **Base:** `master` @ `221497717`
**Spend: $0.00. No instance was ever created.**

---

## 🔴 The headline: this session measured nothing, and that is the finding

The session was to answer two questions Jish asked directly:

1. **Throughput** — what did PR #54 (scheduler coalescing) and PR #56 (fused MoE
   gather cap 8 → derived) actually buy, re-running the wave26-AX sweep so the
   rows compare line by line?
2. **Quality** — what is Arc's real GSM8K, given 87.0% was taken on decode math
   that PR #35 superseded?

**Neither was answered. The GPU provider session is expired and a box could not
be rented at all.**

```
$ runcrate ps
Error: session expired — run: runcrate login
  (token refresh failed (HTTP 400):
   {"code":400,"error_code":"refresh_token_already_used",
    "msg":"Invalid Refresh Token: Already Used"})
```

The Runcrate **MCP server** returns the same thing:
`MCP server "claude.ai Runcrate" requires re-authorization (token expired)`.

`runcrate login` signs in **via browser**. It is interactive and only Jish can
complete it. There is no non-interactive path, and inventing one is not
authorized.

**Per RULE ZERO, this is a STOP, not a puzzle to route around.** The previous
session's failure mode was improvising a fallback when the intended path did not
work — it downloaded the 149 GB source and ran a full in-memory quantization on
a $4.85/hr H200, burning ~$10 for nothing. The correct response to a blocked
path is to stop and report. **Cost of stopping cleanly here: $0.00.**

> ⚠️ **UNVERIFIED AND WORTH CHECKING FIRST THING:** because the provider session
> is dead, **I could not run `runcrate ps` to confirm no instance from a previous
> session is still running and billing.** DOCTRINE D10 applies across sessions.
> An orphan H200 bills $0.082/min ≈ **$118/day** unnoticed. **Check this
> immediately after re-authenticating.**

---

## What was verified for free, and what it changes

All of this is laptop-only work that needed no GPU. It is why the *next* attempt
should succeed on the first try rather than the third.

### 1. The artifact is present, complete, and reachable ✅

Read from the HF API with the token in `~/.config/arc/env`:

| check | result |
|---|---|
| Token identity | `heydryft`, orgs `['aeonmind']` — HTTP 200 |
| `aeonmind/DeepSeek-V4-Flash-UQFF-qtip2` | HTTP 200, **private: True**, 15 files, **74.19 GB** |
| All 9 weight files | present; **every byte count matches the model card exactly** |
| `deepseek-ai/DeepSeek-V4-Flash` (overlay base) | HTTP 200, **public, ungated**, 73 files, **159.63 GB** |

So the artifact was never the problem this time. **Nothing blocks the run except
the provider login.**

Note the memory entry saying the token "writes `heydryft/*` ONLY; aeonmindai org
doesn't exist on HF" is consistent but easy to misread: the **HF** org is
`aeonmind` (no `ai`), it does exist, and the token can read it.

### 2. The runbook would have failed on contact — six separate errors

`GPU_SESSION_RUNBOOK_8.md` was the session's stated authority, and **its two
central commands were both broken.** All six are now fixed (see PR).

| # | Error | Consequence |
|---|---|---|
| 1 | **"There is no bake and no 149 GB source download"** — false. The artifact is an *overlay*; Arc builds from the source checkpoint and overlays quantized layers. | The exact false belief that led the last session to improvise. |
| 2 | `serve --from-uqff /workspace/uqff` had **no `-m`, no `-a`, and a directory where a file belongs** — three independent errors in one line. | Each alone yields the misleading `DummyLayer not replaced at index 1` error. |
| 3 | `run_gsm8k.py --shots 0 … --label s8` — **neither flag exists** (`run_gsm8k.py:141-161`). | argparse rejects the command; step S5 dies instantly **on a paid box**. |
| 4 | `run_coherence.py --label s8` — **no `--label`** either. | Same. |
| 5 | **Missing `--chat-template chat_templates/deepseek_v4.json`**, which the measured-working invocation carried (wave26-AX §2). | The probe and every scored eval post to `/v1/chat/completions` and rely on the server-side template. |
| 6 | **`ABORT-IF load > 3 min (expected ~11 s)`** — the threshold sat *below* the only cold-cache measurement we have. | **Would have aborted a healthy load.** The card's 12.94 s was warm-cache on the box that had just baked it; cold is **3 m 10 s**. Now 8 min, with `Applying ISQ` as the real signal. |

The corrected serve invocation, matching the model card's hardware-verified run
(517 tensors, 12.94 s):

```bash
mistralrs serve -p 1234 \
  -m /workspace/src \
  -a deepseekv4 \
  --from-uqff /workspace/uqff/qtip2-0.uqff \
  --max-seqs 128 --prefix-cache-n 0 \
  --max-seq-len 4096 --max-batch-size 128
```

**Why `-m` must be the source, confirmed in code not memory:** with
`--from-uqff` set, `normal.rs:679-687` deserializes the ISQ layers and leaves
base expert weights as `DummyLayer` placeholders — but **non-ISQ weights
(embeddings, norms, lm_head, compressor) still resolve through the source
VarBuilder.** Hence both trees on disk: **74.19 + 159.63 = 233.8 GB.**
Downloading the source *as the overlay base* is correct; downloading it *to
re-quantize* is what RULE ZERO forbids.

**`--from-uqff` takes the first shard FILE, not a directory** —
`mistralrs-cli/src/args/model.rs:91-95`: *"specifying the first shard …
automatically finds …"*.

### 3. Both fixes under test are genuinely in the tree ✅

* **PR #54** — `default_scheduler.rs`, +317 lines, merged in `140ac04dc`.
* **PR #56** — the hard cap is gone, replaced by
  `gather_policy::lut_fused_gather_preferred()` evaluated per call.

One correction to the briefing's framing: the cap was **not** raised "8 → 511".
The boundary is *derived per routing shape* from a traffic model
(`pairs ≤ 16 · E·(1−(1−k/E)^n)`), giving **~512 tokens at top-8 of 256** and
**~683 at top-6**. **V4-Flash is top-6**, so the live boundary is ~683.

**This matters for interpreting the sweep:** every batch in the planned sweep
(B ≤ 128) now sits *far* below the boundary, so **the whole sweep should stay on
the fused path** — where before, B=32 and B=64 fell off it. That is precisely
the inversion wave26-AX measured (aggregate 10.31 tok/s at 8 tokens/step → 5.07
at 13).

**But do not expect the fix alone to deliver the fleet thesis.** The module's
own docs are explicit (`gather_policy.rs:105-112`): the fused path issues one
GEMV per (token, expert) pair **with no dedup**, so raising the cap converts a
fallback that *degrades* with batch into one that is merely **flat per token**.
It does not produce the `E(B)` amortization — that needs the grouped GEMM, which
only the bitshift rung has. **Predicted shape of the result: B=32/64 stop
collapsing; aggregate goes roughly flat rather than rising steeply.**

### 4. The exact sweep to re-run, recovered from wave26-AX ✅

The briefing pointed at `memory/mission/wave26-AX-h200-measurement.md`. **It is
in the repo** (not in `~/.claude/.../memory/`, where a first look failed to find
it). Its §1 protocol, so the next session does not re-derive it:

> One H200 @ **$4.85/hr**. Server `--max-seqs 128 --prefix-cache-n 0`, chat
> template supplied. Probe `batch_load_probe.py`, `/v1/chat/completions`
> streaming, distinct ~68-token prompts, **64 decode tokens, 1 rep**,
> `--max-ctx 118000`, **temperature 0**.

⇒ the line-comparable command is:

```bash
python3 batch_load_probe.py \
  --batches 1,8,16,32,64 --include-128 \
  --reps 1 --max-tokens 64 \
  --max-ctx 153000 \
  --cost-per-hour 4.85 \
  --label wave31_postfix
```

Two deliberate deltas from wave26-AX, both to be stated in the writeup:

* **`--include-128`** — B=128 was never run before. At the sweep's short context
  it fits (runbook §3); at 2048 ctx it does not, and the largest feasible row is
  ~B=68. **Report the arithmetic if a row is dropped; never drop it silently.**
* **`--max-ctx 153000`, not 118000** — the baseline's 118k was sized to the
  in-situ bake's **89,543 MiB** footprint. The published artifact loads at
  **75,859 MiB** (wave26-AX §2), i.e. **13.7 GB more free**, so the KV guard
  should be set to the real boundary rather than inherited.

**`temperature 0` also settles the PR #52 question cheaply.** The probe sends
temperature=0 → `sampler.rs:324` maps `<1e-7` to greedy, which returns *above*
the radix branch. The `GPU radix top-k … falling back to CPU` line was **never
on the measured path**, so its absence in the new log confirms nothing about
throughput. Confirm it is gone, but **do not attribute any speed change to it.**

### 5. The baseline is not the clean comparison the briefing assumes ⚠️

The briefing says to re-run so rows "compare line by line". They will not, quite.
FACTS records wave26-AX as measured on an **in-situ qtip2 W=32 bake — NOT the
published UQFF** (which is **W=256 beam, hadamard-128**). Same probe, same
protocol, **different weights.**

**State this whenever the tables are put side by side.** The comparison is still
worth making — it is the only prior sweep — but it is not a pure A/B of the two
fixes, and reporting it as one would repeat exactly the class of error this
repo has been correcting.

---

## What the next session must do

1. **`runcrate login`** (Jish, interactive) — then `runcrate ps` **to check for
   an orphaned billing instance before anything else.**
2. Run §10 preflight (now in the runbook) — all four checks, free, before renting.
3. Then the runbook as written. The artifact side is verified and ready.

**Nothing else about this session was blocked.** The only missing input is a
browser login.

---

## Honesty note (DOCTRINE D9)

**No throughput number, no GSM8K number, and no `$/Mtok` figure appears in this
document, because none was measured.** The two questions Jish asked remain open.
The provisional **87.0% GSM8K** stays provisional and still must not be treated
as a baseline to defend — and when it is finally re-measured, a result on either
side of 87.0 is **not** a regression, because it will be the first number taken
on decode math that matches the reference.
