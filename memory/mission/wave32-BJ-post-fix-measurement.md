# wave32-BJ — post-fix measurement: **BLOCKED AGAIN, NOTHING MEASURED**

**Date:** 2026-08-15 · **Base:** `master` @ `3460656d3`
**Spend: $0.00. No instance was ever created.**
**This is the SECOND consecutive session blocked by the same expired login**
(see `wave31-BF-post-fix-measurement.md`, base `221497717`).

---

## 🔴 Headline

The four throughput fixes that landed today — **#54, #56, #59, #46** — are
**still unmeasured end to end.** So is GSM8K on the published artifact. The
blocker is not technical and not in Arc: **the GPU provider session is expired
and a box cannot be rented at all.**

```
$ runcrate ps
Error: session expired — run: runcrate login
  (token refresh failed (HTTP 400):
   {"code":400,"error_code":"refresh_token_already_used",
    "msg":"Invalid Refresh Token: Already Used"})
```

The Runcrate **MCP server** is dead the same way — three calls, same result:

```
mcp__claude_ai_Runcrate__list_instances  -> MCP server "claude.ai Runcrate" is not connected
mcp__claude_ai_Runcrate__billing_balance -> MCP server "claude.ai Runcrate" is not connected
```

**Proof it is expiry and not a transient**, decoded from the stored JWT rather
than inferred from the CLI's error message:

| field | value |
|---|---|
| `access_token` `exp` | `1780539751` = **2026-06-04 02:22:31Z** |
| now | 2026-08-15 20:43:55Z |
| **expired by** | **72.8 days** |
| `refresh_token` | already consumed — `refresh_token_already_used` |
| `~/.runcrate/config.yaml` mtime | Jun 4 02:22 — untouched since the token died |

`runcrate login` **opens a browser** (`runcrate login --help`: *"Opens your
browser to log in to Runcrate and select a workspace"*). There is no API-key
flag, no token flag, and `runcrate config set` only sets the deployment URL.
**Re-auth is interactive and only Jish can complete it.**

**Per RULE ZERO this is a STOP, not a puzzle to route around.** Stopping cleanly
cost **$0.00**. The session that improvised a fallback burned ~$10 to produce
nothing.

### ⚠️ The brief's premise was wrong, and that matters

This wave was dispatched with: *"Runcrate MCP is authenticated; balance $84.93;
zero instances running (verified)."* **All three claims are unverifiable from
here and the first is demonstrably false.** Whatever "verified" referred to, it
was not a live call — the token has been dead since June 4.

> 🔴 **CANNOT CONFIRM NO ORPHAN BOX IS BILLING.** DOCTRINE D10 applies across
> sessions. With no working API call I cannot run the `list_instances` that D10
> requires. An orphan H200 bills $0.082/min ≈ **$118/day** silently. **First
> action after `runcrate login`: `runcrate ps`, before anything else.**

---

## What was done instead — the free preflight, completed in full

§10 of the runbook says every non-GPU check is free and must happen before
renting. All of it ran. **Three of four pass; the fourth is the blocker.**

| # | check | result |
|---|---|---|
| 1 | GPU provider session alive | 🔴 **FAIL — the blocker** |
| 2 | Nothing already running and billing | ⚠️ **UNVERIFIABLE** (needs #1) |
| 3 | HF token valid, sees the private org | ✅ `heydryft ['aeonmind']` |
| 4 | Both repos reachable, overlay complete | ✅ overlay **15 files 74.19 GB, private**; source **73 files 159.63 GB**, public |

So the artifact side is **green and re-verified today**: the moment a box exists,
S1 can start. The blocker is exclusively the rental.

### Beyond §10 — checks that would otherwise have been paid for

**The four fixes are genuinely in `master` @ `3460656d3`** (all MERGED
2026-08-15):

| PR | title | merged |
|---|---|---|
| #54 | scheduler: coalesce length buckets so the whole batch runs | 18:19:14Z |
| #56 | qtip: derive the fused MoE gather boundary instead of capping at 8 | 18:29:27Z |
| #59 | v4: roll the xs compressor state (B=68 → 266) | 20:13:30Z |
| #46 | qtip: computed codebook on the LUT rung | 20:02:57Z |

**The probe's concurrency assertion is functional** — run offline on CPU, exit 0.
This clears the S2(b) ABORT-IF (*"the concurrency assertion is non-functional and
every number this session produces is unfalsifiable"*) **for free**, before it
could cost paid box time:

```
$ python3 arc-tools/quality/test_batch_load_probe.py
PASS chat sweep: tokens exact, agg 43.92 -> 175.0 tok/s, TTFT p50 0.255s, $/Mtok 7.81
PASS split+concurrency: prefill agg 1105.79 tok/s (server 1136.0) vs decode agg 175.0 tok/s; effective_B 4/4
PASS raw mode: chunk-counted 24 tok/req, KV warning fired (never blocked)
PASS sustained: 6 completions in 2.41s, agg 66.86 tok/s
PASS mutation(serial): probe exited 1, effective_B 1/4, mean in-flight 0.74, headline suppressed — the assertion CAN fail
PASS mutation(cap 2 of 8): FAIL at default gate; WARN + effective_B=2 reported even when the gate is disabled
PASS --cost-per-hour: $/Mtok 0.62 @ $0.39/hr
ALL PASS
```

Note these are **mock-server numbers proving the harness's arithmetic — they are
not Arc measurements** and must never be quoted as such.

**Eval CLI surfaces verified by argparse**, so no paid step dies on a typo:
`run_gsm8k.py` takes `--n/--seed/--out/--max-tokens/--eight-shot` — **no
`--shots`, no `--label`**, and omitting `--eight-shot` *is* 0-shot; default `--n`
is **150**, so `--n 100` must be explicit. `run_coherence.py` takes only `--out`
and `--skip-facts`. `batch_load_probe.py --batches` accepts the full
comma-separated `1,8,16,32,64,128,256`.

---

## 🔴 The one real bug this session found: the runbook was stale by 3.91×

**`GPU_SESSION_RUNBOOK_8.md` §3 still described the pre-#59 memory model**, and
it would have destroyed the session it was written for.

| | stale (pre-#59) | correct (post-#59) |
|---|---|---|
| per-token footprint | 424,018 B | **108,288 B** |
| max B @ 2048 ctx | ~74 | **~266** |
| `--max-ctx` for S3 | `153000` | **`545000`** |
| stated sweep ceiling | *"largest sweep row is **B=64**"* | **B=256 fits, ~2.2 GB spare** |

The failure would have been **silent and self-justifying**: `--max-ctx 153000`
makes the probe fire `WARNING[KV]` on B=128 and B=256, and §S3's own ABORT-IF
then instructs the operator to *"drop to the largest B that does not warn"*. The
runbook would have told a correct box to throw away **exactly the two rows the
session exists to produce** — and the writeup would have recorded a memory cap
that no longer exists as though it were physics.

Arithmetic is PR #59's own (`memory/mission/wave30-BE-rolling-xs.md`), verified
against `mistralrs-core/src/kv_cache/xs_rolling.rs`:

```
KV        43 layers x 2(K,V) x 1 head x 512 x 2 B      =  88,064 B/token  (unchanged)
xs pre    41 layers x 4096 x 2 B                       = 335,872 B/token
xs post   41.4 MB/seq @2048 ctx / 2048                 =  20,224 B/token   16.6x
          -> 423,936 -> 108,288 B/token                                     3.91x
B @2048   59e9 / (108,288 x 2048) = 266   |  B=256 = 56.8 GB of ~59 usable
```

§3 is rewritten, and §1/§S2/§S3/§S4 are made consistent with a B=256 sweep
(`--max-seqs 256`, `--batches 1,8,16,32,64,128,256`, `--max-ctx 545000`).

**Also corrected:** §3 headroom said *"141 − 68 GB artifact ⇒ 65 GB usable"*. The
artifact is **74.19 GB** on the HF listing and **75.7 GB resident** per the model
card. The section now uses PR #59's **59 GB**.

> **Noticed:** with `xs` no longer dominant, **attention KV is now 81% of the
> per-token budget** and becomes the next binding constraint at long context.
> V4 is MQA, `head_dim=512`, 43 full-cache layers — Arc stores 1,024
> B/token/layer where the reference stores 584. FP8 KV would roughly double
> feasible B again. Worth a separate change?

---

## What is still unmeasured — state this plainly, do not estimate it

**Every number below is UNKNOWN. No figure in this document is an Arc
measurement.** Predictions are recorded only so the next session can be scored
against them, and are labelled `[predicted]`, never `[measured]`.

**PART A — throughput.** The pre-fix baseline stands unchallenged
(wave26-AX, in-situ **W=32** bake — *not* the published W=256 artifact, and any
comparison must say so):

| B | 1 | 8 | 16 | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|---|---|
| pre-fix aggregate tok/s | 15.35 | 14.83 | 10.31 | 5.07 | 8.14 | *unreachable* | *unreachable* |
| post-fix | **?** | **?** | **?** | **?** | **?** | **?** | **?** |

Aggregate **fell** with batch — that is the defect under repair.
`[predicted]` #54 stops the B=64 collapse (~2×, explicitly not scaling); #56
makes the MoE path **flat per token, not amortizing**, so the prediction is
*"stops collapsing"*, **not** *"starts scaling"*; #59 is what makes B=128/256
reachable at all; **#46 will not appear** — it needs a bake to take effect at
serve time and the published artifact predates it.

**PART B — quality.** **87.0% is VOID** — PR #35 changed decode math after it
(SwiGLU clamp missing on 4 of 5 expert paths including the shared expert, which
every token traverses in every layer; YaRN wrongly applied to ratio-0 layers,
correct set exactly {0,1,43}). The next number is the **first** taken on math
matching the reference: **either side of 87.0 is a valid result and LOWER IS NOT
A REGRESSION.** The reference's 90.8 is **8-shot** against our **0-shot** — state
the shot count on every comparison.

**PART C — MTP acceptance.** Not investigated; the box never existed. Free-only
if the instrumentation already emits.

---

## Next session: do these in this order

1. **Jish runs `runcrate login`** (browser, interactive — nobody else can).
2. **`runcrate ps` immediately** — D10, confirm no orphan box is billing. Then
   `runcrate billing balance` for the real number.
3. Re-run §10 checks 3–4 (cheap, catches a rotated HF token).
4. Then §S0 onward. **§3 is now correct — use `--max-ctx 545000` and
   `--max-seqs 256`.**

**Do not dispatch another measurement wave until step 1 is done.** Two
consecutive sessions have now been spent discovering the same dead login. It is
the only blocker, and no amount of agent time moves it.
