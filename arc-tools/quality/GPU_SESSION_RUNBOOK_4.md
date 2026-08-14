# GPU SESSION 4 RUNBOOK — the payoff session: 90+ voting shot, GEMV autotune, MTP acceptance, grouped curve, twin-seed

**Target:** one Runcrate H200 (141GB HBM, ≥24 cores, ≥720GB disk; sessions 1-3
were NY, $4.92/hr). **Budget: ~6-7 GPU-hours ≈ $30-35** full scope; trip-wires
below. Every step has command, expected wall time, expected output, and an
**ABORT IF** rule. A driver (human or agent) executes it mechanically top to
bottom.

**Branch model:** the box checks out **`session4-runbook`** (= master ≥ PR #17
+ this harness + the batch-curve bench + the session-2 Rust patches as files)
and applies the three `arc-tools/quality/patches/s2_*.patch` files before the
one build of the session. The patches are the PR #11 deltas — MTP acceptance
telemetry, `ARC_QTIP_ROTATION_SEED`, perplexity `--dump-logprobs` — which are
**NOT on master** (root cause of the session-3 MTP acceptance miss, see step
6). All three are pre-verified to `git apply` cleanly against master
`381063914` (post-#17).

**What this session produces** (in `arc-tools/quality/results/` on the box):
`speed_s4_baseline.json`, `speed_s4_tuned.json` (+`gemv_tune_winners` table +
`qtip_gemv_bw_tuned.txt`), `gsm8k_chat_votes5.json`,
`gsm8k_chat_8shot_votes5.json` (the 90+ attempts), `speed_mtp2.json` +
`mtp_acceptance.txt` (+depth-4 variants if ahead), `grouped_curve.txt`,
`coherence.json`, `vote_smoke.json`, `template_parity.txt`, twin-seed inputs
`lp_bakeA.ndjson` + `lp_bakeB.ndjson` + `ppl_qtip2_seedB_mini.log`
(ensemble_ppl.json computed OFFLINE post-teardown), plus `bake4.log` /
`bakeB.log` excerpts and `guard_tests.txt`.

**Anchors (session-3, `memory/mission/gpu-run3-results/`):** b=1 decode
**13.99 tok/s** (prefill ~57), GSM8K 0-shot greedy 2048-cap **87.0%** (n=100,
±6.6pp, 2 degenerate, 9 truncated), coherence 6/6, grouped-GEMM parity 5/5,
GEMV 153–192 GB/s (3-4% peak, ~13-19µs/call fixed overhead). Published base
anchor: 90.8 (8-shot EM — different protocol).

**Session-wide rules** (lessons from sessions 1-3 — all of them earned):
- **ssh_execute heredocs do NOT survive** — upload every script with
  `file_upload`, then `bash script.sh` (session-1 lesson).
- **PID-kills, never pattern-kills**: capture `$!` at every launch
  (`SERVE_PID=$!`), kill that exact PID. A `pkill -f` once killed the wrong
  generation of a chained run (session-2 lesson).
- **Stall sentinel + status server come up FIRST** (step 0c) — the session-2
  chain once sat 30 min behind a zombie wait-loop, and the Runcrate token TTL
  (~8h) can cut ssh_execute mid-session; `curl http://<IP>:8899/status.txt`
  keeps working regardless (session-3 lesson).
- ONE scored request at a time on the raw endpoint; server always
  `--prefix-cache-n 0`. (Voting is the deliberate exception: its k chains are
  ONE request, one sequence group — that is what the step-3 vote smoke gates.)
- Gates OFF all session unless a step says otherwise: `ARC_QUANT_ATTENTION`,
  `ARC_QTIP_REFINE_SCALES`, `ARC_V4_WINDOW_ONLY`, `ARC_V4_STANDARD_DENSE`,
  `ARC_V4_NO_ABSORBED_DECODE`, `ARC_FORCE_NAIVE_SDPA`, `ARC_TIME_DECODE`,
  `ARC_ALLOW_UNSTAMPED_QTIP` (would let a pre-0.3.0 greedy artifact serve!),
  `MISTRALRS_ISQ_SINGLETHREAD`. (`ARC_QTIP_EXPERT_GREEDY` no longer exists —
  greedy is banned in code, DOCTRINE D4; a bake handed it hard-errors.)
  `ARC_MTP_LOG_ACCEPTANCE` / `ARC_QTIP_TUNE_TABLE` / `ARC_QTIP_ROTATION_SEED`
  only where their steps say so.
- perplexity binary and server CANNOT run simultaneously (CUDA OOM).
- All scored evals run against the ONE session-4 bake A; bake B is for
  logprob dumps only.
- No debugging beyond ABORT-IF rules — findings are deliverables, fixes
  happen offline.

**Timeline at a glance** (cumulative, expected):

| # | Step | Wall | Cum. |
|---|---|---|---|
| 0 | Rent + driver/toolkit gate + status/sentinel up + **guard tests** | 20m | 0:20 |
| 1 | Bootstrap (build ∥ 149GB download), branch + patches, one build | 50-70m | 1:20 |
| 2 | **GPU Viterbi bake** + pace gate (fallback → s2-binary pivot) | 30m | 1:50 |
| 3 | Serve + speed + coherence + template-parity + **vote smoke** | 20m | 2:10 |
| 4 | **GEMV autotune sweep** (server DOWN) + tuned re-serve + before/after | 40m | 2:50 |
| 5 | **Voting GSM8K ×2** (votes5, votes5+8shot) — the 90+ attempts | 2:30-3:00 | 5:40 |
| 6 | **MTP acceptance** (self-verifying, depth 2) | 25m | 6:05 |
| 7 | **Grouped-GEMM batch curve** B=1..64 (server DOWN) | 15m | 6:20 |
| 8 | **Twin-seed**: bake B (seed 161) + logprob dumps A/B | 55m | 7:15 |
| 9 | Tar to /srv/arcstatus + **DELETE INSTANCE** | 15m | 7:30 |
| 9b | Ensemble analysis — OFFLINE, box already gone | $0 | — |

**Trip-wires** (cumulative from instance creation): at **5:00** and step 5b
(8-shot run) not yet at n=60 → let it stop at its current n (resume-safe
JSON, keep ≥60). At **6:00** → drop twin-seed dump B (8b's second half). At
**6:30** → drop grouped curve. At **7:15 = hard teardown** no matter what.
Cut order overall: 8-shot n → twin-seed 9b-half → grouped curve → MTP
depth-4 (already conditional) → 8-shot config entirely. Never cut step 9.

---

## Step 0 — Rent + hygiene gates (20m)

Runcrate: 1× H200, ≥720GB disk, CUDA image. Add SSH key, note IP.

**0a. Driver/toolkit gate — before anything builds** (session-1: 40 min lost;
session-3: root of the bake crawl):

```bash
nvidia-smi | head -4          # "CUDA Version: X.Y" = driver's MAX supported
nvcc --version | tail -2      # toolkit (or ls /usr/local/cuda-*)
```

- **ABORT-IF toolkit > driver CUDA version** (e.g. driver 13.0, toolkit 13.1):
  builds fine, dies at runtime with `CUDA_ERROR_UNSUPPORTED_PTX_VERSION` —
  or worse, works around it into a **silent CPU-crawl bake** (session 3).
  Fix BEFORE building: `apt-get install -y cuda-toolkit-13-0` (match the
  driver), then `export CUDA_HOME=/usr/local/cuda-13.0
  PATH=/usr/local/cuda-13.0/bin:$PATH` — and KEEP those exports in every
  build shell for the whole session (session-3's pinning).
- Also check: H200 141GB visible; ≥600GB disk free. HF_TOKEN not needed.

**0a-BIS. BOX HEALTH GATE — MANDATORY, before the download starts** (session
5: ~1.5h and ~$7 lost). A rental can be individually bad in a way no compile
check will show. Box **s5a** (NY H200, 192.241.248.189) ran the *session-3
binary* `cca7a9c2e` — proven at ~30 s/layer — at **~3 min/layer**. A bisect
proved the slowdown was ENVIRONMENTAL, not our kernels. The signature:

| metric | s5a | reading |
|---|---|---|
| GPU utilization | **99%** | looks perfectly healthy — this is the trap |
| `power.draw` | **~132 W of a 700 W limit** (19%) | the tell |
| `clocks.sm` | at max | not throttling |
| temperature | fine | not thermal |

Clocks maxed + power floored = the device is **starved waiting on host↔device
transfers**. QTIP Viterbi quantize streams every expert CPU→GPU→CPU, so the
bake is transfer-heavy; a slow host/PCIe path on one rental crawls it.

```bash
curl -fsSL https://raw.githubusercontent.com/aeonmindai/arc/session4-runbook/arc-tools/quality/box_health_gate.sh -o /root/box_health_gate.sh
bash /root/box_health_gate.sh --json /srv/arcstatus/box_health.json   # ~90s
```

- Checks: GPU name/driver, **sustained `power.draw` under a ~60 s synthetic
  CUDA load** (auto-picks a prebuilt `viterbi_bake_bench`, else a torch
  matmul+H2D loop, else an nvcc'd burn), `clocks.sm` vs `clocks.max.sm`,
  PCIe gen/width, `nproc`, disk free. Writes PASS/FAIL lines + JSON.
- **ABORT-IF exit code 1** (sustained power < 28.6% of the enforced limit =
  **<200 W on a 700 W H200**, or any other FAIL line): **delete the instance
  and re-rent a different box — preferably a different region.** Do NOT debug
  the rental, do NOT "see how it goes". Deleting costs $0.08; s5a cost $7.
- Thresholds are **heuristics from one datapoint (s5a)**, tuned to catch a 6×
  slow box, not to grade a 10% slow one. `--min-power <W>` overrides.
- After step 1's build, re-run with the ground-truth pace probe — this is the
  highest-signal check and supersedes the proxies above:

```bash
bash /root/box_health_gate.sh --arc $ARC --with-bake-probe "$V4_DIR" \
  --json /srv/arcstatus/box_health_bake.json      # 3 min, then it kills the bake
```

  It starts the real `quantize`, counts `Detected INT4` layer-progress lines
  for 3 minutes, PID-kills it, and **FAILs at <3 lines** (session-3 healthy ≈
  6 per 3 min at ~30 s/layer; s5a ≈ 1). It also flags a `QTIP GPU quantize
  fallback` line and warns if the binary asks for >2 ISQ threads.

**0b. Boot** (build ∥ download; boot script honors `ARC_BRANCH` and re-runs
the toolkit gate):

```bash
curl -fsSL https://raw.githubusercontent.com/aeonmindai/arc/session2-runbook/arc-tools/boot_run161_h200.sh -o /root/boot.sh
ARC_BRANCH=session4-runbook nohup bash /root/boot.sh > /root/boot.log 2>&1 & disown
```

(If the boot script's clone ignores `ARC_BRANCH` — it did once in session 1 —
fix after boot: `cd $ARC && git fetch origin session4-runbook && git checkout
session4-runbook`.)

**0c. Status server + first sentinel — NOW, not later** (upload both scripts
with `file_upload`; they also live in `arc-tools/quality/` on the branch):

```bash
mkdir -p /root/logs
nohup bash /root/status_server.sh > /root/logs/status_server.log 2>&1 &
nohup bash /root/stall_sentinel.sh /root/boot.log 900 > /root/logs/sentinel_boot.log 2>&1 &
# from the DRIVING machine — must work before proceeding:
curl -s http://<BOX_IP>:8899/status.txt | head -5
```

From here on, every long-running step gets its own sentinel with the step's
PID. The status page is the ssh-independent escape hatch (token TTL ~8h).

**0d. cfg(cuda) qtip guard tests — FIRST, before any bake** (PR #17,
wave6-Q). Run as soon as the build finishes (they compile fast against the
build cache):

```bash
cd $ARC
cargo test -p mistralrs-quant --release --features cuda \
  cuda_3d_expert_quantize_does_not_fall_back_to_cpu 2>&1 | tee /root/logs/guard_tests.log
cp /root/logs/guard_tests.log $Q/results/guard_tests.txt
```

- Expected: **2 passed** — `qtip::tests::cuda_3d_expert_quantize_does_not_fall_back_to_cpu`
  (LUT rung) + `qtip::bitshift::tests::cuda_3d_expert_quantize_does_not_fall_back_to_cpu`
  (bitshift rung). Passing = PR #17's fallback counter
  (`gpu_quantize_cpu_fallback_count()`) stayed flat across a bake-path
  quantize on this box = the GPU Viterbi will engage.
- **If either FAILS**: the assert prints the new fallback count and the warn
  log names the exact environmental condition — one of:
  - `CUDA build but the QTIP kernels were not compiled in ('has_qtip_kernels'
    cfg absent; build-time compute cap < 8.0?)` → rebuild with the pinned
    toolchain from 0a in the SAME shell (the cfg comes from build.rs probing).
  - `CUDA device 0 initialization failed: <err>` → the actual error is in the
    log; usually the driver/toolkit mismatch from 0a.
  - `an NVIDIA driver is present on this machine but mistralrs-quant was
    built WITHOUT the 'cuda' feature` → the build command lost its features;
    rebuild with `--features "cuda flash-attn"` (NO cudnn — see step 1).
  Capture the warn line VERBATIM (it is a session deliverable either way),
  fix the named condition, re-run the tests. **ABORT-IF** they still fail
  after one targeted fix → bake would crawl at ~11min/layer; pivot directly
  to the s2-binary plan in step 2's fallback and file the warn line.

## Step 1 — Bootstrap: branch + patches + ONE build (50-70m, mostly ∥ download)

After `BOOTSTRAP_COMPLETE` in /root/boot.log:

```bash
export ARC=/mnt/work/arc V4_DIR=/mnt/models/DeepSeek-V4-Flash
export Q=$ARC/arc-tools/quality
cd $ARC
git log --oneline -1                       # must be session4-runbook tip (>= 381063914 master)
git apply arc-tools/quality/patches/s2_mtp_acceptance_telemetry.patch
git apply arc-tools/quality/patches/s2_rotation_seed_override.patch
git apply arc-tools/quality/patches/s2_ppl_dump_logprobs.patch
git diff --stat | tail -4                  # 3 files changed — record in the log
# cudnn feature: −62% decode on V4 (5.45 vs 14.58 tok/s), see session-4 — NEVER add it
cargo build --release --features "cuda flash-attn" 2>&1 | tail -3
cargo build --release -p mistralrs --example perplexity --features "cuda flash-attn" 2>&1 | tail -3
bash $Q/fetch_data.sh                      # now ALSO fetches gsm8k_train.jsonl (8-shot pool)
```

- The patches are env-gated additions only; defaults are untouched, so every
  binary from this one build is valid for every step (serve, bake, tests,
  benches, perplexity).
- Expected: `DATA_READY`; `gsm8k_test.jsonl` 1319 lines; `gsm8k_train.jsonl`
  7473 lines; 4 wikitext files.
- **ABORT IF** a patch fails to apply (master moved past `381063914` in a way
  that touches those hunks — should not happen): skip the FAILED patch only,
  note which; MTP telemetry patch failing kills step 6, seed patch kills
  step 8, dump patch kills 8's dumps.
- **ABORT IF** build fails twice on the same compile error (not network):
  capture log, tear down, fix offline.

**Why the build line dropped `cudnn` (session-4 measured finding):** the
cudnn-feature build decoded V4 at **5.45 tok/s vs 14.58 without it (−62%)** —
same box, same bake, only the feature flag differed
(`results/speed_s4_tuned.json` vs `speed_nocudnn.json`, gpu-run4-results).
HYPOTHESIS ONLY (unverified, needs an offline profile session): candle's
`cudnn` feature swaps the SDPA/attention backend to the cuDNN path, which
bypasses our flash-attn integration and with it the V4 absorbed-decode fast
path — turning every decode step's attention into the slow generic route.
Deeper investigation queued offline: profile one decode step under both
builds (`ARC_TIME_DECODE`/nsys), confirm which kernel serves attention, and
check whether cudnn also disables the fused decode kernels. Until that lands:
**no session build line ever includes `cudnn`.**

## Step 2 — GPU Viterbi bake + pace gate (30m) — with the current master build

PR #17 is merged: if this bake falls back to CPU, the log now SAYS WHY.

```bash
cd $ARC
env | grep -E "ARC_QTIP|MISTRALRS_ISQ" && echo "^^ UNSET THESE FIRST" || echo CLEAN
nohup ./target/release/mistralrs quantize text -m "$V4_DIR" -a deepseekv4 \
  --isq qtip2 -o "$V4_DIR/uqff/" > /root/logs/bake4.log 2>&1 & BAKE_PID=$!
nohup bash /root/stall_sentinel.sh /root/logs/bake4.log 900 $BAKE_PID \
  > /root/logs/sentinel_bake.log 2>&1 &
```

**Pace gate — at minute 3, hard rule (do NOT wait to "see how it goes"):**

```bash
sleep 180
grep -c "QTIP GPU quantize fallback" /root/logs/bake4.log     # MUST be 0
grep -E "Applying (immediate )?ISQ .*threads" /root/logs/bake4.log  # MUST be 1 thread
grep -c "Detected INT4" /root/logs/bake4.log                  # MUST be >=3 by minute 3
wc -c /root/logs/bake4.log; sleep 60; wc -c /root/logs/bake4.log   # still growing
```

- GPU pace is **~30s/layer → 44 layers ≈ 24 min** (session-3 measured). The
  gate: by minute 3 the log must show **no fallback warn** and visible
  per-layer progress (≥1 layer/min equivalent — log actively growing past
  the first expert stacks). CPU-crawl pace is ~11 min/layer: a log that has
  the fallback warn, or that freezes after the first stack, is the crawl.
- **Thread count MUST read 1** (session-5 trap). PR #20's Viterbi-default
  flipped `get_max_isq_cpu_threads()` to `None` = all cores, so the bake
  logged `Applying immediate ISQ in parallel on 24 threads` — 24 host threads
  submitting Viterbi work to ONE device, ~4-9 min/layer with no fallback
  warning. Session-3's fast bake logged `1 threads`. Fixed in code (the QTIP
  rungs now cap at 1 whenever the quantize runs on GPU, and the log carries
  the rationale); `MISTRALRS_ISQ_SINGLETHREAD=1` remains the belt-and-braces
  override on any older binary — including the s2-binary pivot below.
- **Slow bake with 1 thread and no fallback warn ⇒ it is the BOX, not us.**
  Do not bisect our kernels (session 5 did, and the session-3 binary was just
  as slow). Run `box_health_gate.sh` from step 0a-BIS, and on FAIL delete +
  re-rent.
- **ABORT-AND-DIAGNOSE if the fallback warn appears**: capture the full warn
  line (it names the condition — that line is a primary session deliverable
  for wave6-Q). `kill $BAKE_PID`, then pivot to the **s2-binary bake trick**
  (session-3's proven workaround):

```bash
# s2-binary pivot: bake with the session-2-era master build (pre-#12..#15;
# artifact is bit-identical — GPU/CPU parity was tested in session 3, and the
# session-3 87.0%/13.99 numbers were produced exactly this way).
cd /mnt/work && git -C arc worktree add /mnt/work/arc-s2 cca7a9c2e
cd /mnt/work/arc-s2
# same pinned toolchain exports as step 0a, then (NO cudnn — −62% decode, step 1):
cargo build --release --features "cuda flash-attn" 2>&1 | tail -3
nohup ./target/release/mistralrs quantize text -m "$V4_DIR" -a deepseekv4 \
  --isq qtip2 -o "$V4_DIR/uqff/" > /root/logs/bake4_s2.log 2>&1 & BAKE_PID=$!
# (~25m build + 24m bake; serve/tests still use the MASTER build in $ARC.
#  For step 8's bake B, apply s2_rotation_seed_override.patch in THIS
#  worktree too — it was cut against master but the touched hunks predate
#  #12, so try git apply; if it fails here, twin-seed bakes both come from
#  whichever build did bake A.)
```

**Bake done:**

```bash
ls -lh "$V4_DIR"/uqff/
export UQFF0=$(ls "$V4_DIR"/uqff/qtip2*.uqff | sort | head -1)
grep -E "Applying ISQ on|QTIP GPU quantize fallback" /root/logs/bake4*.log > $Q/results/bake4_log_excerpt.txt
```

- Expected ~68GB across 7 shards. **ABORT IF** bake exceeds 75m total (incl.
  pivot) or output <40GB / >120GB.

## Step 3 — Serve + baseline + template-parity + vote smoke (20m) — FAIL-FAST

```bash
cd $ARC
# --chat-template is REQUIRED: without it /v1/chat/completions 422s (session-4)
nohup ./target/release/mistralrs serve -p 1234 -m "$V4_DIR" -a deepseekv4 \
  --from-uqff "$UQFF0" --prefix-cache-n 0 \
  --chat-template chat_templates/deepseek_v4.json > /root/logs/serve_s4.log 2>&1 & SERVE_PID=$!
until curl -s localhost:1234/health >/dev/null; do sleep 5; done && echo UP
python3 $Q/speed_probe.py --label s4_baseline     # anchor: 13.99 tok/s (session 3)
# SESSION 5+: b=1 is the kernel-latency DIAGNOSTIC only — the headline speed
# metric is the batch_load_probe sweep (see SESSION 5 DELTA section below).
python3 $Q/run_coherence.py --skip-facts          # 6/6 gate
```

- **ABORT IF** coherence ≤3/6 (bake/build broken → triage 30m max, then
  teardown). **Flag** (not abort) if decode <10 tok/s: a kernel-fix
  regressed vs session 3 — record, continue; the autotune A/B is still valid
  relative to this baseline.

**3b. Chat-template parity probe** — voting rides `/v1/chat/completions`
(server-side template), which NO previous scored eval has used (the harness
deliberately uses raw completions + `encoding_dsv4`). Gate it:

```bash
python3 - <<'EOF' | tee $Q/results/template_parity.txt
import sys, os
sys.path.insert(0, os.environ["Q"])
import qlib
q = "What is 17 + 25? Please reason step by step, and put your final numeric answer after '####'."
raw  = qlib.complete(qlib.encode_chat(q), max_tokens=128, temperature=0.0)
chat = qlib.chat_complete(q, max_tokens=128, temperature=0.0)
print("RAW :", raw["text"][-120:].replace("\n"," "))
print("CHAT:", chat["text"][-120:].replace("\n"," "))
ok = "42" in raw["text"] and "42" in chat["text"]
print("TEMPLATE_PARITY:", "PASS" if ok else "FAIL")
EOF
```

- Both paths must answer 42 fluently. CHAT garbled/degenerate while RAW is
  clean = the served chat template mis-encodes V4 → **server-side voting is
  off**; use `--client-votes` in step 5 (raw path) and record the finding.

**3c. Vote smoke** — first-ever multi-chain decode on V4: the compressor's
`xs_history` is a single per-model cache (deepseek4.rs:791), never exercised
with sibling chains in one batch. Two chains, trivial prompt:

```bash
python3 - <<'EOF' | tee $Q/results/vote_smoke.json
import sys, os, json
sys.path.insert(0, os.environ["Q"])
import qlib
r = qlib.chat_complete(
    "What is 6 times 7? Answer with the number after '####'.",
    max_tokens=96, temperature=1.0, top_p=0.95,
    n_votes=2, vote_mode="confidence_weighted")
print(json.dumps({"text_tail": r["text"][-150:], "finish": r["finish_reason"],
                  "vote": r["vote"]}, indent=1))
EOF
```

- PASS = fluent winner text + a populated `vote` field (`winner_index`,
  `candidates[2]` with confidences). Degenerate/looping text or a 4xx =
  multi-chain interference (or template failure from 3b) → **step 5 switches
  to `--client-votes`**, and the failure detail is a wave finding.

## Step 4 — GEMV autotune sweep + before/after (40m) — server DOWN for the sweep

The `perf/gemv-autotune` work (example `qtip_gemv_tune`, applied at serve
time via `ARC_QTIP_TUNE_TABLE`) is a COORDINATED parallel deliverable — it
may still be an unmerged PR at session time. Resolve in this order:

```bash
kill $SERVE_PID; sleep 10
cd $ARC
git fetch origin perf/gemv-autotune && echo BRANCH_EXISTS \
  || gh -R aeonmindai/arc pr list --search "gemv-autotune" --json number,headRefName,state
# if it exists and is NOT in this checkout's history:
git merge --no-edit origin/perf/gemv-autotune \
  || { git merge --abort; git checkout origin/perf/gemv-autotune -- mistralrs-quant/; }
# verify the contract before relying on it (names were coordinated, VERIFY anyway):
git grep -l "ARC_QTIP_TUNE_TABLE" -- mistralrs-quant/ || echo "CONTRACT MISSING - SKIP STEP"
ls mistralrs-quant/examples/qtip_gemv_tune.rs || echo "EXAMPLE MISSING - SKIP STEP"
# NO cudnn (−62% decode, step 1); kernels changed -> rebuild
cargo build --release --features "cuda flash-attn" 2>&1 | tail -2
```

- **SKIP the whole step** (note it, continue to step 5 with the baseline
  serve env) if the branch/PR doesn't exist or the contract greps fail —
  10 minutes max deciding.

**Sweep + apply + before/after:**

```bash
cargo run --release -p mistralrs-quant --example qtip_gemv_tune --features cuda \
  2>&1 | tee $Q/results/gemv_tune_sweep.txt
# Winner-table contract (session-4 fix — a glob missed the filename and the
# tuned serve ran WITHOUT the table): the sweep's LAST line is
#   WINNER_TABLE_WRITTEN: <absolute path>
# Parse THAT line, never glob:
TUNE_TABLE=$(grep '^WINNER_TABLE_WRITTEN: ' $Q/results/gemv_tune_sweep.txt | tail -1 | cut -d' ' -f2)
[ -s "$TUNE_TABLE" ] || echo "WINNER TABLE MISSING - serve will use baked defaults"
cp "$TUNE_TABLE" $Q/results/gemv_tune_winners.json
export ARC_QTIP_TUNE_TABLE=$Q/results/gemv_tune_winners.json
# kernel-level after (compare against session-3 anchors 153-192 GB/s):
cargo run --release -p mistralrs-quant --example qtip_gemv_bw --features cuda \
  2>&1 | tee $Q/results/qtip_gemv_bw_tuned.txt
# end-to-end after (env must reach the SERVE process; verify with
#   tr '\0' '\n' < /proc/$SERVE_PID/environ | grep ARC_QTIP_TUNE_TABLE):
nohup env ARC_QTIP_TUNE_TABLE=$ARC_QTIP_TUNE_TABLE ./target/release/mistralrs serve \
  -p 1234 -m "$V4_DIR" -a deepseekv4 --from-uqff "$UQFF0" --prefix-cache-n 0 \
  --chat-template chat_templates/deepseek_v4.json \
  > /root/logs/serve_s4_tuned.log 2>&1 & SERVE_PID=$!
until curl -s localhost:1234/health >/dev/null; do sleep 5; done && echo UP
python3 $Q/speed_probe.py --label s4_tuned      # THE before/after tok/s pair
```

- Deliverable: `speed_s4_baseline.json` vs `speed_s4_tuned.json` + the
  GB/s delta at the kernel level. Session-3 profile said ~18.7µs/call fixed
  overhead and 3-4% peak BW — the sweep's win should show up as both a
  GB/s and a tok/s move if launch parameters were the binding constraint.
- **If tok/s does NOT move but GB/s did**: fixed overhead (launch/dispatch)
  dominates at b=1 — that conclusion is itself the deliverable; record both
  numbers, keep `ARC_QTIP_TUNE_TABLE` set anyway (helps larger b).
- **ABORT-IF-cheap**: sweep crashes or tuned serve fails health → unset
  `ARC_QTIP_TUNE_TABLE`, restart baseline serve, skip to step 5.

## Step 5 — Voting GSM8K: the 90+ attempts (2:30-3:00) — OBJECTIVE 1

Server up (tuned env if step 4 won, else baseline — RECORD WHICH in the log).
Both runs: n=100, seed 161, 2048-cap, k=5 chains, confidence-weighted
(verified against merged #13: fields `n_votes` / `vote_mode`, values
`"confidence_weighted"` | `"majority"`; non-streaming; chains sampled
t=1.0/p=0.95 — greedy chains would be 5 identical copies).

**5a. Voting only** (isolates the voting delta over 87.0):

```bash
python3 $Q/run_gsm8k.py --n 100 --max-tokens 2048 --votes 5 \
  2>&1 | tee -a /root/logs/gsm8k_votes.log
# -> results/gsm8k_chat_votes5.json   (resume-safe per problem)
```

**5b. Voting + 8-shot** (adds the published protocol's prompt shape; the
answer to "is the remaining gap protocol or quantization?"):

```bash
python3 $Q/run_gsm8k.py --n 100 --max-tokens 2048 --votes 5 --eight-shot \
  2>&1 | tee -a /root/logs/gsm8k_votes8.log
# -> results/gsm8k_chat_8shot_votes5.json
```

- Wall budgeting: 5 batched chains ≈ 2-2.5× single-chain wall (MoE expert
  reads amortize across sibling chains). Session-3 single-chain n=100 took
  ~70m; expect **~1:15-1:45 per run** at ≥13 tok/s (less if step 4 moved
  decode). 8-shot adds ~1.3-1.6K prefill tokens ≈ +25-30s/problem at ~57
  tok/s prefill — 5b is the longer run.
- **Rate check after 10 problems** (each run): >2.5 min/problem → this run
  finishes around n=60-70 in budget; let it run and stop it at the
  trip-wire — the per-problem JSON keeps whatever n it reached (keep ≥60,
  CI widens to ~±12pp; say so in the results note).
- **If step 3's smoke forced client-side voting**: swap `--votes 5` →
  `--votes 5 --client-votes` and cut to `--n 60` up front (client chains are
  sequential ≈5× wall; majority tally, no confidence weighting — record the
  protocol difference).
- **ABORT IF** running accuracy <40% after 20 problems while `coherence.json`
  was 6/6 and text_tails look fluent → extractor/template bug, not the model:
  save 3 text_tails, kill the run, re-run 5a as `--client-votes` (raw
  encoding path). Voting below the 87.0 greedy anchor at n≥60 is NOT an
  abort — it is a real (negative) result; record it.
- Success bar: **≥90.8 on either run beats the published single-shot base
  protocol number**; anything ≥88 on 5a alone already isolates a positive
  voting delta at matched quantization.

## Step 6 — MTP acceptance, self-verifying (25m) — OBJECTIVE 2

Why session 3 produced nothing: master's `log_acceptance_rate()` has **zero
call sites** — the counters accumulate but no serve-path code ever logs
them, so `grep "MTP acceptance"` could never match, and the chain's probe
step silently produced no artifact. The periodic logger is PR #11's
env-gated patch (`s2_mtp_acceptance_telemetry.patch`) — applied in step 1,
so THIS build has it. Every sub-step below verifies before spending.

```bash
kill $SERVE_PID; sleep 10
cd $ARC
nohup env ARC_MTP_LOG_ACCEPTANCE=1 ${ARC_QTIP_TUNE_TABLE:+ARC_QTIP_TUNE_TABLE=$ARC_QTIP_TUNE_TABLE} \
  ./target/release/mistralrs serve -p 1234 -m "$V4_DIR" -a deepseekv4 \
  --from-uqff "$UQFF0" --prefix-cache-n 0 --mtp-depth 2 \
  --chat-template chat_templates/deepseek_v4.json \
  > /root/logs/serve_mtp2.log 2>&1 & SERVE_PID=$!
until curl -s localhost:1234/health >/dev/null; do sleep 5; done && echo UP

# VERIFY 1 — engagement (must print the engaged line):
grep -E "MTP (speculative decode engaged|requested)" /root/logs/serve_mtp2.log
# VERIFY 2 — telemetry actually fires (128 decoded tokens at depth 2
# = ~128 proposed >= 2 log periods of 64):
python3 -c "
import sys, os; sys.path.insert(0, os.environ['Q']); import qlib
qlib.complete(qlib.encode_chat('Count from 1 to 40, one number per line.'), max_tokens=128)"
N=$(grep -c "MTP acceptance" /root/logs/serve_mtp2.log)
echo "acceptance lines: $N"; [ "$N" -ge 1 ] || echo "TELEMETRY DEAD - see below"
```

- `MTP requested (depth=2) but the loaded model has no MTP head` → the
  from-uqff path isn't exposing the head (PR #15 gap variant): record
  VERBATIM, skip step (that line is the deliverable). NO debugging.
- `acceptance lines: 0` after VERIFY 2 → the telemetry patch isn't in this
  binary (step-1 patch skipped?) or env didn't reach the server process
  (`tr '\0' '\n' < /proc/$SERVE_PID/environ | grep MTP`). Fix the named
  cause once; still 0 → record + skip.

**Only after both VERIFYs pass**, spend the timed probe:

```bash
python3 $Q/speed_probe.py --label mtp2         # -> results/speed_mtp2.json
grep "MTP acceptance" /root/logs/serve_mtp2.log | tail -20 | tee $Q/results/mtp_acceptance.txt
[ -s $Q/results/speed_mtp2.json ] && [ -s $Q/results/mtp_acceptance.txt ] \
  && echo STEP6_ARTIFACTS_OK || echo STEP6_ARTIFACTS_MISSING
```

- Deliverables: converged acceptance % (late lines) + decode tok/s vs
  step-3/4 baselines (speculative win ≈ `1 + depth×acceptance` minus verify
  overhead). **Only if ≥30m ahead of trip-wires**: repeat with
  `--mtp-depth 4` (`speed_mtp4.json`, append acceptance lines).

## Step 7 — Grouped-GEMM batch curve B=1..64 (15m) — server DOWN

The bench is ON THIS BRANCH (`mistralrs-quant/examples/qtip_grouped_curve.rs`,
CPU-smoke verified at authoring). It drives the public `gather_forward`
dispatch: grouped GEMM forced for every B via
`ARC_QTIP_ONDEVICE_MOE_MAX_TOKENS=0` (set internally), plus the b≤8
gather-GEMV reference for the crossover.

```bash
kill $SERVE_PID; sleep 10
cd $ARC
cargo run --release -p mistralrs-quant --example qtip_grouped_curve --features cuda \
  2>&1 | tee $Q/results/grouped_curve.txt
```

- Expected: parity already proven (session-3 5/5) — this is the THROUGHPUT
  first-look. Watch: us/call flat-ish as B grows (good: expert reads
  amortize) vs linear (bad: no batching win); the crossover B where grouped
  beats gemv; the labeled extrapolation table (MoE-GEMM-only floor) as the
  first measured input to FLEET.md §2's saturated-batch math.
- **ABORT-IF-cheap**: crashes → capture stderr, skip. 15 minutes total, no
  debugging.

## Step 8 — Twin-seed ensemble: bake B + dumps (55m) — server DOWN

First data on decorrelated-quantization-error averaging (session-2 design,
never yet completed on hardware). Needs the seed + dump patches from step 1.

**8a. Bake B (seed 161):**

```bash
cd $ARC
df -h / | tail -1                       # need ~70GB free
nohup env ARC_QTIP_ROTATION_SEED=161 ./target/release/mistralrs quantize text \
  -m "$V4_DIR" -a deepseekv4 --isq qtip2 -o "$V4_DIR/uqff_b/" \
  > /root/logs/bakeB.log 2>&1 & BAKEB_PID=$!
nohup bash /root/stall_sentinel.sh /root/logs/bakeB.log 900 $BAKEB_PID \
  > /root/logs/sentinel_bakeB.log 2>&1 &
sleep 120
grep "rotation seed overridden" /root/logs/bakeB.log    # MUST show 0xa1
grep -c "QTIP GPU quantize fallback" /root/logs/bakeB.log   # 0, same pace gate as step 2
```

- No `rotation seed overridden` line = the seed patch isn't in this binary →
  kill, skip step (record). Same 3-minute pace gate as step 2.
- **ABORT-IF-cheap**: bake fails or exceeds 45m → kill, `rm -rf
  "$V4_DIR/uqff_b"`, skip to step 9.

**8b. Logprob dumps A + B** (same corpus, same chunking — the ensemble
script hard-fails on mismatch):

```bash
export UQFF_B=$(ls "$V4_DIR"/uqff_b/qtip2*.uqff | sort | head -1)
./target/release/examples/perplexity -m "$V4_DIR" -a deepseekv4 \
  -f $Q/data/wiki.test_mini.raw -u "$UQFF0" --chunk-size 1024 \
  --dump-logprobs $Q/results/lp_bakeA.ndjson 2>&1 | tail -3
./target/release/examples/perplexity -m "$V4_DIR" -a deepseekv4 \
  -f $Q/data/wiki.test_mini.raw -u "$UQFF_B" --chunk-size 1024 \
  --dump-logprobs $Q/results/lp_bakeB.ndjson 2>&1 | tee $Q/results/ppl_qtip2_seedB_mini.log | tail -3
```

- ~12m each. Free extra data point: bake-B mini-ppl vs bake-A = seed
  sensitivity of the Viterbi bake.
- Trip-wire rule: if only ONE dump fits the budget, do NEITHER (the
  ensemble needs both; a lone dump is dead weight) — that is the 6:00 cut.

## Step 9 — Results out + TEARDOWN (15m) — NEVER CUT

```bash
cd $Q && tar czf /root/s4_results.tgz results/ && cp /root/s4_results.tgz /srv/arcstatus/
ls -lh /srv/arcstatus/s4_results.tgz
# from the driving machine (works even if the ssh token died):
curl -s -o s4_results.tgz http://<BOX_IP>:8899/s4_results.tgz && tar tzf s4_results.tgz
```

Verify the tarball lists every expected artifact from "What this session
produces" BEFORE deleting. Then **DELETE the Runcrate instance** (DELETE,
not stop). Verify via list_instances that it is gone. On EVERY resume of the
driving session: list_instances; box exists + session done/stalled → delete
immediately.

## Step 9b — Twin-seed ensemble analysis (OFFLINE, $0)

```bash
python3 arc-tools/quality/ensemble_ppl.py results/lp_bakeA.ndjson results/lp_bakeB.ndjson
```

- SUCCESS SIGNAL: `ppl_ensemble` ≥3% below `min(ppl_A, ppl_B)` (decorrelated
  errors cancel). High A/B spread + gain = variance-dominated (more seeds
  keep helping); low spread + no gain = bias-dominated (seed averaging
  exhausted). Either verdict is publishable.

---

## SESSION 5 DELTA — speed measurement is batch-first (headline change)

Directive: production is always B=32/64/128 concurrent users; **b=1 is a
kernel-latency diagnostic only**. From session 5 on, the speed-measurement
step's headline is the `batch_load_probe.py` sweep; `speed_probe.py` keeps
the diagnostic line.

```bash
# HEADLINE — batch-first serving measurement (chat endpoint, server template):
python3 $Q/batch_load_probe.py --label s5_baseline --max-ctx <server KV budget, tokens>
#   -> results/batch_load_s5_baseline.json + one BATCH[B=..] line per B (1,8,16,32,64)
# b=1 kernel-latency DIAGNOSTIC (unchanged; session-3/4 anchor 13.99 tok/s):
python3 $Q/speed_probe.py --label s5_b1_diag
# if budget allows — B=128 and/or the production-shaped sustained window:
python3 $Q/batch_load_probe.py --label s5_128 --include-128 --max-ctx <budget>
python3 $Q/batch_load_probe.py --label s5_sustained --batches 64 --duration 120
```

- **Two headline numbers per config**: aggregate decode tok/s at the best B
  (fleet economics) and per-user p50 decode tok/s at that B (product feel);
  TTFT p50/p95 rides along. The `BATCHSWEEP[...]` line prints all of them.
- **$/Mtok**, inline in the probe output: `$/Mtok = 4.92e6 / (agg_tok_s × 3600)`
  at $4.92/hr H200 — b=1 13.99 tok/s ⇒ $97.68/Mtok; 300 tok/s aggregate ⇒
  $4.56/Mtok. That division is the whole fleet argument in one line.
- **PREREQ**: the served build must include the xs_history per-seq fix
  (PR #21) — concurrent V4 sequences crash or silently corrupt without it.
  The probe's own warmup batch is the de-facto gate; step 3c's vote smoke is
  a weaker cousin (sibling chains, one request).
- The session rule "ONE scored request at a time" still holds for SCORED
  evals; the batch probe is the sanctioned exception (unscored throughput).
- KV guard: the probe estimates `B × (prompt + decode)` tokens before each B
  and WARNS (never blocks) when it exceeds `--max-ctx`.
- Pre-session validation, $0, no GPU: `python3 $Q/test_batch_load_probe.py`
  (mock-server smoke asserting the probe's math — wave-1 dry-run precedent).

### SESSION 5 DELTA — two hardware lessons, now enforced in code

**1. The box-health gate is MANDATORY (step 0a-BIS).** Session 5 rented an
H200 (s5a, NY, 192.241.248.189) where the *session-3 binary* baked at ~3
min/layer instead of ~30 s. Environmental, not ours: 99% util, **132 W of a
700 W limit**, clocks maxed, temps fine ⇒ starved on host↔device transfers.
Cost ~1.5h before detection. From session 6 on:

```bash
bash /root/box_health_gate.sh --json /srv/arcstatus/box_health.json          # after boot, ~90s
bash /root/box_health_gate.sh --arc $ARC --with-bake-probe "$V4_DIR" \
     --json /srv/arcstatus/box_health_bake.json                              # after the build, 3 min
```

Exit 1 ⇒ **delete the instance and re-rent a different box/region**. Never
debug a rental. Thresholds (<200 W sustained on a 700 W board; <3 layers per
3 min) are one-datapoint heuristics, documented as such in the script.

**2. Bake threading is fixed in code — stop relying on the env var.** Session
5 also logged `Applying immediate ISQ in parallel on 24 threads` where
session 3 logged `1 threads`: PR #20's Viterbi-default made
`IsqType::QtipBitshift2 | Qtip2b => get_max_isq_cpu_threads()` return `None`
(= all cores), so N CPU threads each pushed GPU Viterbi work at ONE device.
The QTIP rungs now resolve their thread cap against where the quantize
actually runs (`IsqQuantizeBackend`): **1 thread on a GPU-backed bake**, all
cores only for a genuinely CPU-side Viterbi. Every ISQ log line now carries
the rationale, e.g.

```
ISQ thread policy: 1 thread(s) — QTIP quantize runs in GPU kernels on one
device; extra host threads only contend for it (session-5 bake trap).
```

`MISTRALRS_ISQ_SINGLETHREAD=1` still wins over everything and is still the
right thing to export for the s2-binary pivot (that binary predates the fix).
On the bad box single-threading did NOT fix the crawl — that was the
environment — so both lessons are needed, and gate 1 is the one that saves
the money.

## Appendix — verified names cheat sheet (verified against master 381063914, 2026-08-14)

| Thing | Value | Where verified |
|---|---|---|
| Voting request fields | `n_votes: usize` (≥2 activates; 1 rejected by harness; `Some(0)` server-errors), `vote_mode: "majority"\|"confidence_weighted"` (default confidence_weighted; `"confidence-weighted"` also parses), optional `answer_regex`, `early_stop_confidence: f32`, `reasoning_budget: usize`; **`/v1/chat/completions` ONLY** (raw completions has no voting); mutually exclusive with `stream: true`; `n` must equal `n_votes` if both set | `mistralrs-server-core/src/openai.rs:619-641`, `chat_completion.rs parse_vote_options/parse_request`, `mistralrs-core/src/arc_boost.rs FromStr` |
| Voting response | `choices[0]` = winner; top-level `vote: {mode, winner_index, winner_answer, candidates:[{index, answer, confidence, weight}]}` | `mistralrs-core/src/response.rs:162-165`, `arc_boost.rs VoteOutcome/VoteCandidate` |
| Guard tests (run FIRST) | `cargo test -p mistralrs-quant --release --features cuda cuda_3d_expert_quantize_does_not_fall_back_to_cpu` → 2 tests (LUT `qtip::tests`, bitshift `qtip::bitshift::tests`) | PR #17 diff, `qtip/mod.rs` + `qtip/bitshift.rs` |
| Fallback counter | `mistralrs_quant::gpu_quantize_cpu_fallback_count()` — flat across a GPU-path quantize | PR #17, `qtip/mod.rs` |
| Fallback warn line | `QTIP GPU quantize fallback: <context> is quantizing on the CPU — <reason>. Expect ~10-20x slower per-layer bake times…` — reasons quoted in step 0d | PR #17, `note_gpu_quant_cpu_fallback` |
| 2-D mis-map warn | `…quantizing a NxK weight on the CPU Viterbi/greedy pipeline…` (≥4M weights) | PR #17, `warn_big_cpu_2d_quantize` |
| s2-binary pivot commit | `cca7a9c2e` (session-2 master = merge PR #10; pre-#12..#15) | wave6-Q window reconstruction |
| MTP flag + lines | `--mtp-depth 0..=8`; engaged: `MTP speculative decode engaged (depth=N)`; missing head: `MTP requested (depth=N) but the loaded model has no MTP head` | `mistralrs-cli/src/args/mod.rs:437-440`, `pipeline/mtp_pipeline.rs try_wrap_pipeline_with_mtp` |
| MTP telemetry | `ARC_MTP_LOG_ACCEPTANCE=1` → `MTP acceptance rate: X% (a/p accepted)` per 64 proposed — **patch-only** (`s2_mtp_acceptance_telemetry.patch`); master's `log_acceptance_rate()` has NO call sites (session-3 root cause) | patch file (applies clean to `381063914`); grep of master |
| Rotation seed | `ARC_QTIP_ROTATION_SEED=<u64\|0xhex>` → `QTIP rotation seed overridden: …` — **patch-only** (`s2_rotation_seed_override.patch`); decode reads STORED signs, so bakes stay self-consistent | patch file (applies clean); `qtip/mod.rs rotation_seed()` |
| Logprob dump | `perplexity … --dump-logprobs <path>` — **patch-only** (`s2_ppl_dump_logprobs.patch`) | patch file (applies clean) |
| GEMV autotune | MERGED (PR #19) + session-4 winners BAKED as dispatch defaults (v21 `w4_r2_i1_v2` for gate/up, v6 `w8_r4_i1_v2` for down); example `qtip_gemv_tune` ends with `WINNER_TABLE_WRITTEN: <path>` (parse it, never glob); serve-env `ARC_QTIP_TUNE_TABLE=<path>` overrides the baked table, no recompile | `mistralrs-quant/src/qtip/tune.rs`, `examples/qtip_gemv_tune.rs` |
| Batch-curve bench | `cargo run --release -p mistralrs-quant --example qtip_grouped_curve --features cuda` — this branch; CPU smoke verified | `mistralrs-quant/examples/qtip_grouped_curve.rs` |
| GSM8K harness knobs | `--votes K --vote-mode … [--client-votes] [--eight-shot] [--fewshot-seed 8] --max-tokens 2048`; output name gains `_8shot`/`_votes5`/`_cvotes5` suffixes; train pool `data/gsm8k_train.jsonl` (fetch_data.sh) | this branch, `run_gsm8k.py`/`qlib.py`/`fetch_data.sh` |
| Vote-smoke risk | compressor `xs_history` is one per-model `Mutex<SingleCache>` — multi-chain batches untested on V4 before step 3c | `mistralrs-core/src/models/deepseek4.rs:791` |
| Ops scripts | `status_server.sh` (HTTP :8899, /srv/arcstatus, 30s snapshots), `stall_sentinel.sh <log> <secs> [pid]` (PID-kill escalation), `box_health_gate.sh [--with-bake-probe <dir>]` (step 0a-BIS; exit 1 = delete + re-rent) | this branch, `arc-tools/quality/` |
| ISQ thread policy | `mistralrs_quant::isq_thread_policy(ty, device)` → `(threads, rationale)`; QTIP rungs cap at **1** when `IsqQuantizeBackend::Gpu`. Log line: `ISQ thread policy: N thread(s) — <rationale>` | `mistralrs-quant/src/lib.rs`, session-5 delta |
| Serve / bake commands | runbook 2 base (`serve -p 1234 … --from-uqff … --prefix-cache-n 0`; `quantize text … --isq qtip2`) **+ every serve line adds `--chat-template chat_templates/deepseek_v4.json`** — without it `/v1/chat/completions` 422s (session-4 root cause #1 of the vote-API failures) | runbook-2 appendix; session-4 finding |
