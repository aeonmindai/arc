# wave50-CA — the `qtip2b` bake succeeded again. **The session lost its runcrate auth 3 minutes after `BAKE_RC=0`, before the artifact was uploaded.**

**Branch** `perf/qtip2b-published` · **PR** #77 (draft, base `master`)
**Date** 2026-08-16 · **Box** `arc-s14-qtip2b`, 1×A100-80GB PCIe (sm_80),
Montreal, `inspiring-williamson-a100_80g-8bfc`, $1.49/hr
Created **16:32:28Z** · self-destruct **ARMED 16:37:37Z** (fires **22:37:37Z**)
· 🔴 **NOT DELETED — see §1, this needs Jish**

| stage | result |
|---|---|
| bootstrap (build + 149 GB source pull) | ✅ 22 m 34 s, arc at `46ea6948d` |
| `ARC_QTIP_BEAM=256` verified from `/proc/36155/environ` | ✅ |
| bake, `qtip2b` beam W=256, 43 layers | ✅ **224.9 s/layer, 2 h 52 m 16 s, `rc=0`** |
| corrected **qtip2** model card published | ✅ weights proven untouched |
| search provenance read back from the artifact | ❌ **never ran — locked out** |
| publish `qtip2b` | ❌ **never ran — locked out** |
| prove it serves | ❌ **never ran — locked out** |
| GSM8K n=50 | ❌ **never ran — locked out** |

---

## 1. 🔴 What Jish has to do, and the clock on it

`runcrate` returned, 3 minutes after the bake finished:

```
Error: session expired — run: runcrate login
  (token refresh failed (HTTP 400): {"code":400,"error_code":
   "refresh_token_already_used","msg":"Invalid Refresh Token: Already Used"})
```

It is **not transient** — it reproduced on a second, serial `runcrate ps`.
Re-auth is a browser flow and needs a TTY, so an agent cannot do it.

**The artifact is intact on the box and is not backed up anywhere.** It is at
`/ephemeral/models/DeepSeek-V4-Flash/uqff` (14 files), `BAKE_RC=0`. The box
halts itself at **22:37:37Z**. **If nobody re-auths before then, this is
wave48-BY repeated: ~$9 of A100 time and a second good artifact lost to an
ephemeral disk.**

### Everything needed to finish is already staged on the box

No file has to be copied over again; the whole remainder is five commands.

```bash
runcrate login                       # browser flow — only Jish can do this

# 1. Search provenance, read from the ARTIFACT (qtip2b emits no bake header)
runcrate ssh arc-s14-qtip2b -- 'bash /root/postbake.sh'
#    ABORT-IF this does not print STAMP_OK ... trellis/beam(W=256)/mse

# 2. Publish (swaps in the corrected card, uploads, then diffs hub vs disk)
runcrate ssh arc-s14-qtip2b -- 'bash /root/publish.sh'

# 3. Serve, detached
runcrate ssh arc-s14-qtip2b -- 'nohup bash /root/serve.sh > /root/serve.log 2>&1 </dev/null & disown; echo LAUNCHED'
#    poll /root/serve.log for the listen line, then:
runcrate ssh arc-s14-qtip2b -- 'bash /root/prompts.sh'

# 4. Short GSM8K
runcrate ssh arc-s14-qtip2b -- 'nohup bash /root/gsm8k.sh > /root/gsm8k.log 2>&1 </dev/null & disown; echo LAUNCHED'

# 5. Tear down
runcrate instances delete arc-s14-qtip2b && runcrate ps && runcrate billing balance
```

Staged on the box: `postbake.sh`, `publish.sh`, `serve.sh`, `prompts.sh`,
`gsm8k.sh`, `pace.py`, `read_qtip_stamp.py`, `hf_repo_diff.py`,
`card_qtip2b.md`, and `.hf_token` (0600).

### 🔑 The probable cause is mine, and it is a protocol bug worth keeping

The bake is 2 h 52 m of dead wall time. Background watchers armed to wait it out
were **repeatedly reaped by the harness** (four separate `run_in_background`
tasks were killed mid-wait), so the wait was carried instead by batches of ~18
rapid, back-to-back `runcrate ssh` calls.

`refresh_token_already_used` is the signature of **concurrent OAuth refresh**:
several CLI invocations noticing an expiring access token at once, each
POSTing the same single-use refresh token, and the losers invalidating the
session for everyone. A long poll loop made of many short CLI calls is exactly
the shape that triggers it.

⇒ **Poll a long job with ONE on-box process, not N control-plane calls.** The
right shape is a single detached remote watcher writing an epoch-stamped status
file, sampled rarely — not ~1,000 `runcrate ssh` invocations over three hours.
This cost the session its second artifact.

> **Noticed:** the harness reaping `run_in_background` tasks is what forced the
> polling pattern in the first place. A supported "wait hours for a remote job"
> primitive would remove the whole failure mode. Worth a separate change?

---

## 2. Bake pace — 43 differenced consecutive intervals

Marker: `Detected INT4-packed MoE expert weights`
(`mistralrs-quant/src/distributed/layers.rs:1267`), one per MoE layer, emitted
as the expert stack is dequantized — so a marker-to-marker interval is one
layer's whole quantize. **44 markers ⇒ 43 intervals**, matching wave48-BY.

**Never a running average.** `(t_n − t_0)/n` folds layer 0's one-time init into
every layer; that is the error that produced the bogus 135 s/layer figure.

| interval | s | | interval | s |
|---|---|---|---|---|
| 0→1 | **230.0** *(carries init)* | | 12→13 | 224.6 |
| 1→2 | 225.7 | | 15→16 | 234.1 *(max)* |
| 2→3 | 224.6 | | 17→18 | 223.4 |
| 4→5 | 223.2 | | 18→19 | 220.6 *(min)* |
| 6→7 | 225.2 | | 20→21 | 221.7 |
| 8→9 | 227.7 | | 21→22 | 228.9 |

**Measured steady-state median: 224.9 s/layer** (21 steady intervals at the
23-marker read; range **220.6 – 234.1 s**).
**Total bake: 2 h 52 m 16 s** (16:55:33Z → 19:47:49Z, `BAKE_RC=0`), 43 layers,
≈ **$4.28** of A100 time.

### The projection method held; the box did not match wave48-BY's box

| | wave48-BY (`arc-s13`) | **this session (`arc-s14`)** | Δ |
|---|---|---|---|
| steady median | 211.3 s/layer | **224.9 s/layer** | **+6.4%** |
| engine's own search time | 196.1–196.2 s | **205.3 s** (median, n=22) | +4.7% |
| ⇒ non-search per-layer overhead | ~15.2 s | ~19.6 s | +29% |
| total | 2 h 40 m 03 s | **2 h 52 m 16 s** | +7.6% |

Both boxes are the same advertised part on the same provider at the same price
(1×A100-80GB PCIe, Montreal, $1.49/hr), both ran at 100% util and full 1410 MHz
boost. **So 211.3 s/layer is not a constant for "an A100" — it is one box.**
A ~6% box-to-box spread on nominally identical silicon is the honest error bar
to carry on any future s/layer projection, and the cost projection should be
quoted as a range, not a point.

The projection itself was excellent: **2.68 h** from the first 3 steady
intervals, **2.69 h** at 22 intervals, **2.87 h** actual — the residual being
the ~10 min of buffered artifact write that follows the last layer and is not
in any marker interval.

### Box health — clean throughout

100% GPU util, **146–185 W** of 300 W (49–62%, far above the 28.6% starvation
floor), SM clock **1410 MHz = full A100 PCIe boost**, 52–58 °C, 15.7 GB VRAM
(identical to wave48-BY, and 64 GB of the 80 left idle by the
`ARC_QTIP_EXPERT_BATCH=8` guard). **Zero** `fallback` / `panicked` /
`CUDA error` / `out of memory` lines across the whole bake, checked at five
points. Layer 25 — where an H200 bake once died on fragmentation — cleared
cleanly. `Expert unpack pool: 28 thread(s)`, `Unpacked 256 INT4 experts in 5.9s`.

### `ARC_QTIP_BEAM` was verified from the running process

Not from the launch script — from `/proc/36155/environ`, exactly as wave48-BY
established:

```
ARC_QTIP_BEAM=256
ARC_QTIP_EXPERT_BATCH=8
MISTRALRS_ISQ_SINGLETHREAD=1
```

Unset, `TrellisSearch::from_env` returns `Exhaustive` (`viterbi.rs:150`) at
~8,257 s/layer ⇒ ~98 h ⇒ ~$147, and the 6 h fuse fires around layer 2. The var
is load-bearing on this rung from PR #74 onward.

Corroborating evidence that the right rung ran: all 22 sampled
`Quantized fused experts (…) in …s` lines report **`Qtip2b`**.

---

## 3. ⚠️ What is NOT known about this artifact

**No quality claim of any kind is made here, and none may be inferred.**

- **The search provenance was never read back.** wave48-BY's artifact scanned
  132/132 `trellis/beam(W=256)/mse`; **this artifact was never scanned.**
  `ARC_QTIP_BEAM=256` in the process environment and `Qtip2b` in the per-layer
  log lines are strong evidence, but they are the *inputs*. D4 is satisfied by
  reading the *artifact*, and that did not happen. `postbake.sh` is the
  15-second command that would settle it.
- **It has never been loaded, let alone generated a token.** Whether PR #76's
  KV preallocation fix actually clears the V4 serving failure on hardware is
  **still unmeasured** — it remains proven on CPU only, exactly as wave49-BZ §6
  said. This session did not change that.
- **No GSM8K.** Nothing is known about `qtip2b` output quality on V4.
- **One box, one run.** No repeat.

---

## 4. ✅ The `qtip2` model card was corrected and published

Independent of the bake, so it survived the lockout.

The published card documented an invocation that **does not work**:

```
mistralrs run -m aeonmind/DeepSeek-V4-Flash-UQFF-qtip2 --from-uqff qtip2-0.uqff
```

`-m` points at the overlay instead of the source checkpoint, so anyone finding
the repo got `DummyLayer not replaced at index 1`. It also still described the
repo as private and its quality as the retired provisional 87.0%.

Replaced with the corrected card: the two-flag overlay form, public, and the
measured **GSM8K 96.0%** (96/100, ±3.8 pp, 0 degenerate, 0 truncated; n=100,
0-shot chat, t=0, 2048-cap, seed 161). The 87.0% is recorded as **retired, not
beaten** — different bake, superseded decode math, not a comparable baseline.

**The good artifact was not touched.** `upload_card.py` writes exactly one path
and reads the repo inventory either side of the write:

```
CARD_BEFORE sha=108fbc2e3d… files=15 bytes=74,190,197,268
CARD_AFTER  sha=0b80b279b5… files=15 bytes=74,190,207,603
CARD_OK README.md replaced; all 14 other files byte-identical (untouched)
```

The 10,335-byte delta is the README itself (875 → 11,210 B). Verified live on
the Hub afterwards.

`aeonmind/DeepSeek-V4-Flash-UQFF-qtip2b` remains **public and empty**
(`.gitattributes` only, 1,519 B) — the storage quota that blocked wave48-BY is
gone, and it was never the thing that blocked this session.

---

## 5. Spend

| | |
|---|---|
| box | created 16:32:28Z, **still running at lockout (19:53Z)** |
| rate | $1.49/hr |
| at lockout (3.35 h) | **$4.99** |
| if it runs to the 22:37:37Z fuse (6.09 h) | **$9.07** |

Balance was $49.80 at dispatch. **`runcrate ps` could not be confirmed empty —
the box is up and the CLI is locked out.**

---

## 6. Files

- `arc-tools/quality/read_qtip_stamp.py` — artifact-side search verifier,
  brought onto master's line from wave48-BY's branch. Decodes the tail **from
  the flags byte**; taking "the last two bytes" is wrong for a beam bake.
- `arc-tools/quality/test_read_qtip_stamp.py` — nine-arm D12 test, **re-run and
  passing on this branch** (`ALL ARMS PASS`).
- `arc-tools/quality/hf_repo_diff.py` — independent hub-vs-disk byte diff.
- `docs/model-cards/deepseek-v4-flash-uqff-qtip2.md` — corrected, **published**.
- `docs/model-cards/deepseek-v4-flash-uqff-qtip2b.md` — new, staged on the box,
  **not published**.
- `memory/mission/wave50-CA-qtip2b-published.md` — this record.

No engine code changed.
