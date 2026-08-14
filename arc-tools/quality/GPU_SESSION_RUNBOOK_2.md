# GPU SESSION 2 RUNBOOK — Viterbi re-bake, quality table v2, PR revalidations

**Target:** one Runcrate H200 (141GB HBM, ≥24 cores, ≥720GB disk; session-1 box
was NY, $4.92/hr, 720GB — enough: 149GB model + 2×~70GB UQFF + build).
**Budget: ~7 GPU-hours ≈ $35** full scope (~6h before the twin-seed ensemble
addition), hard trip-wires below. Every step has command, expected wall time,
expected output, and an **ABORT IF** rule. A driver (human or agent) executes
it mechanically top to bottom.

**Branch model:** the box runs **`master`** with the `session2-runbook` branch
merged on top (harness + MTP acceptance telemetry + fixed boot script). This
runbook is written assuming PRs **#6** (qtip2b bitshift), **#9** (Viterbi
default bake) and **#10** (absorbed-MLA decode + GPU top-k) are **MERGED to
master** by session time. **As of authoring (2026-08-13) all three were still
OPEN** — step 0 verifies and tells you what to skip if any isn't.

**What this session produces** (in `arc-tools/quality/results/` on the box):
`speed_absorbed.json`, `speed_no_absorbed.json`, `ab_verdict_absorbed_vs_no_absorbed.json`,
`coherence.json`, `gsm8k_chat.json` (n=100, 1024-cap),
`longctx_fixed.json`, `longctx_standard_dense.json`, `longctx_window_only.json`,
`ppl_qtip2_c1024.json`, `ppl_qtip2_c64.json` (rebaked weights),
`ppl_sink_verdict.json` + `sinkhorn_verdict.json` + `speed_fused_sinkhorn.json`,
`speed_mtp_depth2.json` (+`_depth4`), `mtp_acceptance.txt`,
`qtip2b_parity.txt`, `qtip_gemv_bw.txt`, twin-seed ensemble inputs
`lp_bakeA.ndjson` + `lp_bakeB.ndjson` + `ppl_qtip2_seedB_mini.log`
(ensemble_ppl.json is computed OFFLINE post-teardown), plus `bake2.log` and
`bakeB.log` head/tail.

**Session-1 baseline (compare column — memory/mission/gpu-run1-results/results/):**

| Metric | Session 1 (greedy bake, 640-cap) | Session-2 SUCCESS |
|---|---|---|
| PPL qtip2_c1024 | **58.85** | **≤ ~29** (within 1.3× of q2k) |
| PPL q2k_c1024 | 22.50 — **REUSED, not rerun** | (same number, same corpus) |
| PPL qtip2_c64 | 226.9 | large drop expected |
| GSM8K | 64.0% (n=50, 33/50 truncated@640) | **≥ 70%** (n=100, 1024-cap) |
| coherence6 / facts / math | 6/6 · 21/22 · 8/8 | no regression |
| longctx default (pre-#7-fix) | coh 2/5, needle 4/4 (console evidence) | coh 5/5 post-fix |
| longctx window-only | coh 1/5, needle 0/4 | (repro rung, similar) |
| decode / prefill | 5.24 / ~60 tok/s (92% GPU util) | absorbed decode (#10) faster |
| Sinkhorn fused | REJECTED (drift + 4/6 divergence) | PASS post-#8 fix |

**Session-wide rules** (carried from session 1)
- ONE request at a time; server always `--prefix-cache-n 0`.
- Gates OFF all session unless a step says otherwise: `ARC_QUANT_ATTENTION`,
  `ARC_QTIP_REFINE_SCALES` (known-buggy), `ARC_V4_WINDOW_ONLY`,
  `ARC_V4_STANDARD_DENSE`, `ARC_V4_NO_ABSORBED_DECODE`, `ARC_FORCE_NAIVE_SDPA`,
  `ARC_TIME_DECODE`, `ARC_QTIP_EXPERT_GREEDY` (would silently undo #9!),
  `MISTRALRS_ISQ_SINGLETHREAD` (would mask the Viterbi thread check).
- perplexity binary and server CANNOT run simultaneously (CUDA OOM).
- All quality steps run against the ONE session-2 UQFF bake; never re-bake
  mid-session.
- Weights re-download every session (volume reuse not assumed): 30-60m,
  budgeted inside step 1. No debugging beyond ABORT-IF rules — findings are
  deliverables, fixes happen offline.
- ssh_execute NOTE (session-1 lesson): heredocs do NOT survive; upload scripts
  with file_upload, then `bash script.sh`.

**Timeline at a glance** (cumulative, expected):

| # | Step | Wall | Cum. |
|---|---|---|---|
| 0 | Rent box + **driver/toolkit gate** + PR-state check | 10m | 0:10 |
| 1 | Bootstrap (build ∥ 149GB download) + merge session2-runbook + aux builds | 50m | 1:00 |
| 2 | Eval data fetch | 2m | 1:02 |
| 3 | **Re-bake qtip2 with Viterbi (#9)** + engagement check | 30-60m | 1:50 |
| 4 | Serve + speed + coherence6 + **absorbed-decode A/B (#10)** | 35m | 2:25 |
| 5 | Full coherence battery + sinkhorn baseline capture | 15m | 2:40 |
| 6 | **GSM8K n=100 @1024-cap** (∥ qtip2b parity tests, #6) | 90-105m | 4:20 |
| 7 | **Long-context 3-config matrix (#7 reval)** | 40m | 5:00 |
| 8 | Server DOWN → qtip gemv microbench LUT vs bitshift (#6) | 10m | 5:10 |
| 8b | **Twin-seed bake B** (`ARC_QTIP_ROTATION_SEED=161`) | 35m | 5:45 |
| 9 | PPL ladder, qtip2 rungs only (`SKIP_Q2K=1`) | 35m | 6:20 |
| 9b | **Twin-seed dump B** (mini corpus, bake-B UQFF) | 12m | 6:32 |
| 10 | **Sinkhorn A/B revalidation (#8)**, both halves (dump A rides free) | 35m | 7:07 |
| 11 | **MTP acceptance (#5)**, depth 2 (4 only if ahead) | 20m | 7:27 |
| 12 | Pull results + **DELETE INSTANCE** | 15m | 7:42 |
| 12b | Twin-seed ensemble analysis — OFFLINE, box already gone | $0 | — |

**Budget trip-wires** (cumulative from instance creation): at **6:00** before
step 9 finishes → set `SKIP_C64=1`; at **6:45** → drop MTP depth-4; at
**7:00** → drop MTP entirely; at **7:45 = hard teardown** no matter what is
mid-flight. A torn-down box with partial results beats a complete run that
overruns. Time cuts drop, in order: MTP depth-4 → qtip2_c64 rung → MTP
entirely → gemv microbench → twin-seed (9b first — bake B is sunk cost, but
without dump B the ensemble can't be computed, so cutting 9b cuts 8b's value
too). Never cut step 12.

---

## Step 0 — Rent + driver/toolkit gate + PR states (10m)

Runcrate: 1× H200, ≥720GB disk, CUDA image. Add SSH key, note IP,
`ssh root@<IP>` must work.

**0a. Immediately** (session-1 lesson — 40 min lost to this):

```bash
nvidia-smi | head -4          # "CUDA Version: X.Y" = driver's MAX supported
nvcc --version | tail -2      # toolkit version (or ls /usr/local/cuda-*)
```

- **ABORT-IF toolkit > driver CUDA version** (e.g. driver 13.0, toolkit 13.1):
  the build succeeds then dies at runtime with
  `CUDA_ERROR_UNSUPPORTED_PTX_VERSION`. Fix BEFORE building:
  `apt-get install -y cuda-toolkit-<drv-major>-<drv-minor>` then
  `export CUDA_HOME=/usr/local/cuda-<drv>` (~10m) — or delete the box and pick
  an image whose toolkit matches. The session-2 boot script also enforces this
  gate and fails fast with the same fix instructions.
- Also check: H200 141GB visible; disk ≥600GB free; no nvcc at all → different
  image. You've burned ~$1, not $40.
- HF_TOKEN: not needed (public repos).

**0b. PR-state check** (from your machine, not the box):

```bash
for p in 6 9 10; do gh -R aeonmindai/arc pr view $p --json number,state,mergedAt; done
```

| PR not merged | Consequence |
|---|---|
| **#9** (Viterbi bake) | **ABORT the session** — objective 1 is impossible; a re-bake without it just reproduces ppl 58.85. Do not rent. |
| **#10** (absorbed decode) | Skip the step-4 A/B; decode stays ~5.24 tok/s → use the slow-lane GSM8K recalibration in step 6. |
| **#6** (qtip2b) | Skip step 6's parity tests and step 8 entirely (patch won't compile). |
| MTP-telemetry draft PR (from session2-runbook) | Nothing — step 1 merges the branch onto master anyway. |

## Step 1 — Bootstrap: clone+build ∥ download, merge harness (40-70m)

```bash
ssh root@<IP>
curl -fsSL https://raw.githubusercontent.com/aeonmindai/arc/session2-runbook/arc-tools/boot_run161_h200.sh -o /root/boot.sh
ARC_BRANCH=master nohup bash /root/boot.sh > /root/boot.log 2>&1 & disown
tail -f /root/boot.log    # STEP/OK/FAIL markers; Ctrl-C detaches
```

(The session2-runbook copy of boot.sh honors `ARC_BRANCH` — session 1's did
not — and runs the driver/toolkit gate before building.)

**While the build runs**, in a second shell, bring in the harness + telemetry
and pre-build the extra binaries needed later:

```bash
cd /root/work/arc
git fetch origin session2-runbook
git merge --no-edit origin/session2-runbook   # harness + MTP telemetry + patches
# fallback if the merge conflicts (should not — files are additive):
#   git merge --abort && git checkout origin/session2-runbook -- arc-tools/
#   (and note: MTP acceptance telemetry then missing unless its PR is merged)
cargo build --release -p mistralrs --example perplexity --features "cuda flash-attn" 2>&1 | tail -3
```

Wait for `BOOTSTRAP_COMPLETE` in /root/boot.log, then set session env (paste
into every new shell or append to /root/.bashrc):

```bash
export ARC=/root/work/arc V4_DIR=/root/models/DeepSeek-V4-Flash
export Q=$ARC/arc-tools/quality
mkdir -p /root/logs
```

- Expected: `arc at <sha> master` (+ merge commit), binaries in
  `$ARC/target/release/`, `$V4_DIR/.download_done` exists. Session-1 measured
  25m total (build ∥ download); budget 40-70m.
- Recovery: download stalls → rerun boot.sh (hf resumes). Candle git-dep fetch
  hiccup → `cargo fetch` once more.
- **ABORT IF** build fails twice on the same compile error (not network):
  master+session2-runbook is broken for this CUDA — capture log, tear down,
  fix offline.

## Step 2 — Eval data (2m)

```bash
bash $Q/fetch_data.sh
```

Expected `DATA_READY`, gsm8k_test.jsonl 1319 lines, 4 wikitext files. No abort
condition. NOTE: session-1's q2k rung numbers are reused for the ladder — do
NOT rerun q2k (`SKIP_Q2K=1` is set in step 9).

## Step 3 — Re-bake qtip2 with the Viterbi fix (30-60m) — OBJECTIVE 1

Same qtip2 config as session 1; #9 makes Viterbi+rotation the 3-D expert
default. Confirm the poison gate is not set, then bake:

```bash
env | grep -E "ARC_QTIP|MISTRALRS_ISQ" && echo "^^ UNSET THESE FIRST" || echo CLEAN
cd $ARC
nohup ./target/release/mistralrs quantize text -m "$V4_DIR" -a deepseekv4 \
  --isq qtip2 -o "$V4_DIR/uqff/" > /root/logs/bake2.log 2>&1 & disown
tail -f /root/logs/bake2.log
```

**Viterbi-engagement check — run ~2m in, do NOT wait for the end:**

```bash
grep "Applying ISQ on" /root/logs/bake2.log
```

- `Applying ISQ on 1 threads.` = **GREEDY engaged (bake is wrong)** — with #9
  merged, `IsqType::QtipBitshift2.get_max_isq_cpu_threads()` returns 1 only in
  Greedy mode. Kill the bake, check `ARC_QTIP_EXPERT_GREEDY` isn't set and #9
  is actually in `git log`; one retry. Still 1 thread → **ABORT session**
  (capture log, teardown) — everything downstream would re-measure the bad bake.
- `Applying ISQ on <N>1 threads.` (N = core count, e.g. 24) = Viterbi engaged.
  Rotation rides on the same mode switch (Greedy is what disabled Hadamard
  rotation), so threads>1 ⇒ Viterbi+rotation.
- Second signal: wall time. Session-1 greedy bake ≈ 4.5m; Viterbi is the
  10x-heavier search (June anchor ~8s/layer) → expect **15-60m**. A bake that
  finishes in <6m did NOT run Viterbi — treat as the 1-thread case.

```bash
ls -lh "$V4_DIR"/uqff/
export UQFF0=$(ls "$V4_DIR"/uqff/qtip2*.uqff | sort | head -1)
```

- Expected: shards totalling ~65-80GB (experts 2-bit; attention native FP8 +
  lm_head unquantized by design).
- **ABORT IF** bake exceeds 75m or output <40GB / >120GB.

## Step 4 — Serve + smoke + absorbed-decode A/B (35m) — OBJECTIVE 4, FAIL-FAST

**4a. Default server** (absorbed decode ON — it's #10's default):

```bash
cd $ARC
nohup ./target/release/mistralrs serve -p 1234 -m "$V4_DIR" -a deepseekv4 \
  --from-uqff "$UQFF0" --prefix-cache-n 0 > /root/logs/serve_absorbed.log 2>&1 & disown
until curl -s localhost:1234/health >/dev/null; do sleep 5; done && echo UP
python3 $Q/speed_probe.py --label absorbed        # THE calibration number
python3 $Q/run_coherence.py --skip-facts          # 6/6 gate (sampled t=1.0/p=0.95)
python3 $Q/run_sinkhorn_ab.py capture --label absorbed   # greedy identity half A
grep -i "radix top-k" /root/logs/serve_absorbed.log      # CPU-fallback warn = finding
```

- coherence6 runs sampled → it exercises #10's GPU radix top-k/top-p sampler.
  If the log shows `GPU radix top-k sampling failed; falling back to CPU`,
  record it in the results notes (that IS a finding) and continue.
- **ABORT IF** coherence6 ≤ 3/6 or all-degenerate → bake or build broken.
  Triage 30m max: restart with `ARC_FORCE_NAIVE_SDPA=1`; still garbage →
  capture logs + teardown.

**4b. Kill-switch server** (pre-#10 decode path):

```bash
pkill -f "mistralrs serve"; sleep 10
nohup env ARC_V4_NO_ABSORBED_DECODE=1 ./target/release/mistralrs serve -p 1234 \
  -m "$V4_DIR" -a deepseekv4 --from-uqff "$UQFF0" --prefix-cache-n 0 \
  > /root/logs/serve_no_absorbed.log 2>&1 & disown
until curl -s localhost:1234/health >/dev/null; do sleep 5; done && echo UP
python3 $Q/speed_probe.py --label no_absorbed
python3 $Q/run_sinkhorn_ab.py capture --label no_absorbed
python3 $Q/run_sinkhorn_ab.py compare --a absorbed --b no_absorbed
# -> writes ab_verdict_absorbed_vs_no_absorbed.json (does NOT touch sinkhorn_verdict.json)
```

- Verdict: decode tok/s delta (absorbed vs no_absorbed) + greedy token
  identity. Identity FAIL = real finding (absorbed decode changes numerics) —
  record, keep the DEFAULT path for the rest of the session, don't debug.
- **4c. Restart the default server** (rest of session runs on it):

```bash
pkill -f "mistralrs serve"; sleep 10
nohup ./target/release/mistralrs serve -p 1234 -m "$V4_DIR" -a deepseekv4 \
  --from-uqff "$UQFF0" --prefix-cache-n 0 > /root/logs/serve_base.log 2>&1 & disown
until curl -s localhost:1234/health >/dev/null; do sleep 5; done && echo UP
```

## Step 5 — Full coherence + Sinkhorn baseline capture (15m)

```bash
python3 $Q/run_coherence.py                              # + facts/math
python3 $Q/run_sinkhorn_ab.py capture --label baseline   # step-10's reference half
```

Expected: ≥ session-1 (6/6, 21/22, 8/8) — this is the rebaked-weights
coherence row of quality table v2. Baseline capture MUST be on this same
server process.

## Step 6 — GSM8K n=100, 1024-cap (90-105m) — OBJECTIVE 2

Kills the session-1 truncation artifact (33/50 hit the 640 cap):

```bash
python3 $Q/run_gsm8k.py --n 100 --max-tokens 1024      # seed 161, greedy, resume-safe
```

Budget recalibration from step-4a decode tok/s:
- **≥10 tok/s** (absorbed-decode win): full n=100 ≈ 60-90m. Run to completion.
- **5-8 tok/s**: start it, and hard-stop (Ctrl-C) at **100m elapsed** — the
  JSON is rewritten per problem, whatever n it reached (keep ≥50) is the
  result.
- **<4 tok/s**: something regressed vs session 1 — record, run `--n 50`.

**Concurrently (CPU + idle GPU gaps), qtip2b parity tests (#6) — OBJECTIVE 7a.**
The 4 cfg(cuda) tests use tiny tensors (~59GB HBM is free next to the server);
compile runs on CPU while GSM8K decodes:

```bash
cd $ARC && nohup bash -c 'cargo test -p mistralrs-quant --release --features cuda \
  qtip::bitshift::tests::cuda_ 2>&1 | tee /root/logs/qtip2b_parity.log; \
  cp /root/logs/qtip2b_parity.log $ARC/arc-tools/quality/results/qtip2b_parity.txt' \
  > /dev/null 2>&1 & disown
```

Expected: `4 passed` — `cuda_quantize_2b_matches_cpu_dequantize_cos_sim`,
`cuda_dequantize_2b_matches_cpu`, `cuda_fused_gemv_2b_matches_dequant_matmul`,
`cuda_gather_gemv_2b_matches_cpu`. Any failure = record verdict, do NOT debug.

- GSM8K expected: **≥70%** (session-1 64% was truncation-understated;
  published Base 8-shot anchor 90.8 is a different protocol). `truncated`
  should collapse vs 33/50.
- **ABORT IF** running accuracy <30% after 50 problems AND outputs gibberish
  (cross-check coherence.json first — 6/6 coherence + ~0 GSM8K = extractor
  bug, save 3 text_tails and continue).

## Step 7 — Long-context 3-config matrix (40m) — OBJECTIVE 3

Revalidates #7 (banded-mask fix for Standard layers) on GPU. Three server
configs, three DISTINCT output files (the script now refuses to overwrite —
session-1's clobber can't recur). Config 1 reuses the already-running default
server:

```bash
python3 $Q/run_longctx.py --label fixed                  # 1) post-#7 default

pkill -f "mistralrs serve"; sleep 10
nohup env ARC_V4_STANDARD_DENSE=1 ./target/release/mistralrs serve -p 1234 \
  -m "$V4_DIR" -a deepseekv4 --from-uqff "$UQFF0" --prefix-cache-n 0 \
  > /root/logs/serve_standard_dense.log 2>&1 & disown
until curl -s localhost:1234/health >/dev/null; do sleep 5; done && echo UP
python3 $Q/run_longctx.py --label standard_dense         # 2) pre-fix repro

pkill -f "mistralrs serve"; sleep 10
nohup env ARC_V4_WINDOW_ONLY=1 ./target/release/mistralrs serve -p 1234 \
  -m "$V4_DIR" -a deepseekv4 --from-uqff "$UQFF0" --prefix-cache-n 0 \
  > /root/logs/serve_window_only.log 2>&1 & disown
until curl -s localhost:1234/health >/dev/null; do sleep 5; done && echo UP
python3 $Q/run_longctx.py --label window_only            # 3) ablation
pkill -f "mistralrs serve"; sleep 10                     # server DOWN for 8/9/10
```

- Expected: `fixed` coherence **5/5** (fix validated); `standard_dense`
  degraded like session-1's 2/5 (repro = the fix is what changed it);
  `window_only` ≈ 1/5 + needle 0/4 (unchanged ablation).
- `fixed` NOT clean → real finding; record, continue (don't debug on the meter).

## Step 8 — qtip gemv microbench: LUT vs bitshift @ L=16 (10m) — OBJECTIVE 7b

Server is DOWN (quiet GPU — timings need it). Requires #6 merged:

```bash
cd $ARC
git apply arc-tools/quality/patches/qtip_gemv_bw_bitshift.patch   # pre-verified vs master+#6
cargo run --release -p mistralrs-quant --example qtip_gemv_bw --features cuda \
  2>&1 | tee $Q/results/qtip_gemv_bw.txt
git checkout -- mistralrs-quant/examples/qtip_gemv_bw.rs          # leave tree clean
```

- Output: per-rung (`qtip2/LUT` vs `qtip2b/bitshift`) us/call, GB/s, **%HBM
  peak**, and marginal-BW fit at the real decode shapes — the number that
  decides Stage-4 GEMV work. June LUT anchor: ~388 GB/s ≈ 8% of HBM
  (instruction-bound). The bitshift question: does removing the LUT dependency
  raise %HBM?
- Patch fails to apply / build fails (e.g. #6 changed after authoring) →
  skip, note it, move on. 10 minutes max, no debugging.

## Step 8b — Twin-seed bake B (35m; server DOWN) — ENSEMBLE EXPERIMENT

First-ever data point on twin-seed ensembling: a SECOND qtip2 bake identical
to step 3 except for the Hadamard rotation seed
(`ARC_QTIP_ROTATION_SEED=161`, vs default `0xA3C1_7B0F_5F2E_1D4D`). Decode is
seed-independent — the sign vector is STORED in the UQFF and every
forward/gather path reads the stored signs — so bake B is self-consistent and
its quantization error pattern is decorrelated from bake A's.

```bash
cd $ARC
df -h / | tail -1        # need ~70GB free for the second UQFF
nohup env ARC_QTIP_ROTATION_SEED=161 ./target/release/mistralrs quantize text \
  -m "$V4_DIR" -a deepseekv4 --isq qtip2 -o "$V4_DIR/uqff_b/" \
  > /root/logs/bakeB.log 2>&1 & disown
tail -f /root/logs/bakeB.log
grep "rotation seed overridden" /root/logs/bakeB.log   # MUST appear: seed=0xa1
grep "Applying ISQ on" /root/logs/bakeB.log            # >1 threads, same as step 3
export UQFF_B=$(ls "$V4_DIR"/uqff_b/qtip2*.uqff | sort | head -1)
```

- Expected: same size/shape as bake A (~65-80GB), similar wall time to step 3.
  The `QTIP rotation seed overridden: ARC_QTIP_ROTATION_SEED=0xa1` info line
  proves the override engaged.
- **ABORT-IF-cheap:** no seed-override line, bake fails, or exceeds 45m →
  kill it, `rm -rf "$V4_DIR/uqff_b"`, skip 9b, continue with step 9. Max 10
  minutes of looking at logs — this is a bonus experiment, not the session.

## Step 9 — PPL ladder, rebaked qtip2 (35m; server DOWN) — OBJECTIVE 2

```bash
MODEL_DIR="$V4_DIR" UQFF0="$UQFF0" SKIP_Q2K=1 bash $Q/run_ppl.sh
# if the 5:30 trip-wire hit: prepend SKIP_C64=1 too
```

- Rungs: qtip2_c1024 (~15m) + qtip2_c64 (~15m). q2k is skipped by design —
  session-1's stored 22.50 on the SAME corpus is the comparison rung.
- **HEADLINE GATE: qtip2_c1024 ≤ ~29** (1.3 × 22.50) = the Viterbi fix closed
  the gap. 30-45 = partial win (record; Hessian-aware costs are the known
  next lever). Still ~55+ = fix didn't engage or didn't help — cross-check the
  step-3 thread count in bake2.log.
- **ABORT IF** qtip2_c1024 > 50 → skip c64 rung, go to step 9b (identity and
  ensemble runs don't need good absolute ppl).

## Step 9b — Twin-seed dump B (12m; server DOWN) — ENSEMBLE EXPERIMENT

Bake-B logprobs on the SAME mini corpus + chunking the step-10 gate-off run
uses (that run doubles as dump A):

```bash
cd $ARC
./target/release/examples/perplexity -m "$V4_DIR" -a deepseekv4 \
  -f $Q/data/wiki.test_mini.raw -u "$UQFF_B" --chunk-size 1024 \
  --dump-logprobs $Q/results/lp_bakeB.ndjson 2>&1 | tee $Q/results/ppl_qtip2_seedB_mini.log
```

- Expected: `Dumped per-token logprobs to ...` + a Final perplexity close to
  bake A's mini-corpus number (same config, different seed).
- Also a free data point: bake-B ppl vs bake-A ppl = seed sensitivity of the
  Viterbi bake itself.
- **ABORT-IF-cheap:** any failure → skip (ensemble needs both dumps), ≤10m.

## Step 10 — Sinkhorn A/B revalidation post-#8 (35m) — OBJECTIVE 5

Half 1 — bit-identity through the perplexity binary (server still down).
`SINK_DUMP_OFF` makes the gate-OFF run double as the ensemble's bake-A dump —
no extra ppl pass needed:

```bash
MODEL_DIR="$V4_DIR" UQFF0="$UQFF0" SINK_DUMP_OFF=$Q/results/lp_bakeA.ndjson \
  bash $Q/run_ppl.sh --sinkhorn-ab
# PASS = "SINKHORN-PPL-IDENTITY: PASS (bit-identical per-chunk ppl)"
```

Half 2 — greedy token identity + tok/s on the served path:

```bash
cd $ARC
nohup env ARC_FUSED_SINKHORN=1 ./target/release/mistralrs serve -p 1234 \
  -m "$V4_DIR" -a deepseekv4 --from-uqff "$UQFF0" --prefix-cache-n 0 \
  > /root/logs/serve_fused.log 2>&1 & disown
until curl -s localhost:1234/health >/dev/null; do sleep 5; done && echo UP
python3 $Q/run_sinkhorn_ab.py capture --label fused
python3 $Q/speed_probe.py --label fused_sinkhorn
python3 $Q/run_sinkhorn_ab.py compare          # baseline vs fused -> sinkhorn_verdict.json
pkill -f "mistralrs serve"; sleep 10
```

- Session-1 verdict was REJECTED (ppl drift + 4/6 divergence); #8 merged the
  identity fix. **Both halves PASS** → kernel validated; record tok/s delta vs
  `speed_absorbed.json`. **Verdict only** — flipping the default ON requires a
  code change; do NOT leave the gate on for other steps. Either half FAILS →
  gate stays off, the divergence detail in the verdict JSON is the deliverable.
- If tok/s is identical to baseline AND identity trivially passes, grep
  serve_fused.log to confirm the kernel engaged (fallback is silent).

## Step 11 — MTP acceptance, depth 2 (20m) — OBJECTIVE 6

Telemetry: `ARC_MTP_LOG_ACCEPTANCE=1` (session2-runbook commit) logs
`MTP acceptance rate: X% (a/p accepted)` every 64 proposed tokens.

```bash
cd $ARC
nohup env ARC_MTP_LOG_ACCEPTANCE=1 ./target/release/mistralrs serve -p 1234 \
  -m "$V4_DIR" -a deepseekv4 --from-uqff "$UQFF0" --prefix-cache-n 0 \
  --mtp-depth 2 > /root/logs/serve_mtp2.log 2>&1 & disown
until curl -s localhost:1234/health >/dev/null; do sleep 5; done && echo UP
grep -E "MTP (speculative decode engaged|requested)" /root/logs/serve_mtp2.log
```

- MUST show `MTP speculative decode engaged (depth=2)`. If instead
  `MTP requested (depth=2) but the loaded model has no MTP head` — the
  UQFF-loaded pipeline isn't exposing the MTP head: record as a finding
  (PR #5 gap on the from-uqff path), skip this step. NO debugging.

```bash
python3 $Q/speed_probe.py --label mtp_depth2         # tok/s vs speed_absorbed.json
python3 $Q/run_gsm8k.py --n 5 --seed 7 --max-tokens 512 --out $Q/results/gsm8k_mtp_smoke.json
grep "MTP acceptance" /root/logs/serve_mtp2.log | tail -20 | tee $Q/results/mtp_acceptance.txt
```

- Deliverables: acceptance % (late lines = converged rate) + decode tok/s
  delta. Speculative win ≈ `1 + depth × acceptance` minus verify overhead.
- **Only if ≥30m ahead of the trip-wires**: repeat with `--mtp-depth 4`
  (`serve_mtp4.log`, `--label mtp_depth4`, append acceptance lines to
  `mtp_acceptance.txt`).

```bash
pkill -f "mistralrs serve"
```

## Step 12 — Pull results + TEARDOWN (15m) — DO NOT SKIP

```bash
# on the box
grep "Applying ISQ on" /root/logs/bake2.log > $Q/results/bake2_threads.txt
(head -30 /root/logs/bake2.log; echo ...; tail -10 /root/logs/bake2.log) > $Q/results/bake2_log_excerpt.txt
[ -f /root/logs/bakeB.log ] && grep -E "Applying ISQ on|rotation seed overridden" \
  /root/logs/bakeB.log > $Q/results/bakeB_log_excerpt.txt
cd $Q && tar czf /root/quality2_results.tgz results/ && ls -lh /root/quality2_results.tgz
# from your machine
scp root@<IP>:/root/quality2_results.tgz .
scp root@<IP>:"/root/logs/serve_*.log" ./logs_s2/ || true
tar tzf quality2_results.tgz        # verify BEFORE deleting the box
```

Then **DELETE the Runcrate instance** (DELETE, not stop — stopped boxes bill
disk). Verify via list_instances that it is gone. On EVERY resume of the
driving session: run list_instances; if the box exists and the session is
done/stalled, delete immediately.

## Step 12b — Twin-seed ensemble analysis (OFFLINE, $0) — ENSEMBLE EXPERIMENT

On your machine, from the extracted tarball (box is already deleted):

```bash
python3 arc-tools/quality/ensemble_ppl.py results/lp_bakeA.ndjson results/lp_bakeB.ndjson
```

- Reports `ppl_A`, `ppl_B`, `ppl_ensemble` (probability-averaged:
  `logp_ens = logaddexp(logp_A, logp_B) − ln 2`), per-chunk table, and
  agreement stats (Pearson r of per-token logprobs, mean |Δlogp|, per-chunk
  A-vs-B spread = the bias-vs-variance decomposition signal).
- **SUCCESS SIGNAL: `ppl_ensemble` ≥3% below `min(ppl_A, ppl_B)`** — the
  first-ever evidence of decorrelated-error cancellation between twin-seed
  bakes on real weights (exit code 0 = success). High A/B spread + gain =
  variance-dominated quant error (more seeds keep helping); low spread + no
  gain = bias-dominated (seed averaging exhausted).
- The script hard-fails on token-stream mismatch between the dumps (different
  corpus/chunking) — if that fires, the two ppl runs weren't run as written.

---

## Appendix — verified flags/env/log-lines cheat sheet

| Thing | Value | Where verified (2026-08-13) |
|---|---|---|
| Serve | `mistralrs serve -p 1234 -m <dir> -a deepseekv4 --from-uqff <shard0> --prefix-cache-n 0` | `mistralrs-cli/src/args/{mod,model}.rs` (master) |
| Bake | `mistralrs quantize text -m <dir> -a deepseekv4 --isq qtip2 -o <dir>/uqff/` | `mistralrs-cli/src/args/quantize.rs`; `"qtip2"` in `pipeline/isq.rs:149` |
| Viterbi default + opt-out | #9: `QtipMode::default_expert_mode()`; `ARC_QTIP_EXPERT_GREEDY=1` reverts (`ARC_QTIP_EXPERT_VITERBI` now a no-op) | PR #9 diff, `mistralrs-quant/src/qtip/mod.rs`, `lib.rs` |
| Viterbi engagement signal | `Applying ISQ on N threads.` — 1=Greedy, >1=Viterbi | `pipeline/isq.rs:604` + #9's `get_max_isq_cpu_threads` |
| Longctx repro gates | `ARC_V4_STANDARD_DENSE=1` (any value); `ARC_V4_WINDOW_ONLY=1` (any value) | `models/dsv4_attention.rs:97`, `models/deepseek4.rs:1316` (master) |
| Absorbed-decode kill switch | `ARC_V4_NO_ABSORBED_DECODE=1` (or `true`) | PR #10 diff, `models/dsv4_attention.rs` |
| GPU top-k fallback warn | `GPU radix top-k sampling failed; falling back to CPU` | PR #10 diff, `mistralrs-core/src/sampler.rs` |
| Sinkhorn gate | `ARC_FUSED_SINKHORN=1`, CUDA+F32 only, silent fallback | `models/dsv4_mhc.rs` (master, incl. #8 fix) |
| MTP | `--mtp-depth 0..=8` (clap-validated); engage/fallback lines quoted in step 11 | `mistralrs-cli/src/args/mod.rs:437`, `pipeline/mtp_pipeline.rs` |
| MTP telemetry | `ARC_MTP_LOG_ACCEPTANCE=1` → `MTP acceptance rate: ...` per 64 proposed | session2-runbook commit, `mtp_pipeline.rs record_acceptance` |
| qtip2b parity tests | `cargo test -p mistralrs-quant --release --features cuda qtip::bitshift::tests::cuda_` (4 tests) | PR #6 diff, `qtip/bitshift.rs` `mod tests` |
| gemv microbench | patch `arc-tools/quality/patches/qtip_gemv_bw_bitshift.patch`, compile-verified against master+#6 | `mistralrs-quant/examples/qtip_gemv_bw.rs` |
| Harness new knobs | `run_longctx.py --label/--force` (anti-clobber), `run_sinkhorn_ab.py` free labels + derived verdict path, `run_ppl.sh SKIP_Q2K/SKIP_C64/SINK_DUMP_OFF` | this branch |
| Boot script | honors `ARC_BRANCH` (default master) + driver/toolkit gate | this branch, `arc-tools/boot_run161_h200.sh` |
| Rotation seed override | `ARC_QTIP_ROTATION_SEED=<u64|0xhex>` (quantize-time only; decode reads STORED signs — verified: all forward/gather paths use `self.rotation_signs`, UQFF serializes them); logs `QTIP rotation seed overridden` | this branch, `mistralrs-quant/src/qtip/mod.rs rotation_seed()` |
| Logprob dump | `perplexity ... --dump-logprobs <path>` → NDJSON per chunk (target-token logprobs) | this branch, `mistralrs/examples/advanced/perplexity/main.rs` |
| Ensemble analysis | `ensemble_ppl.py <A.ndjson> <B.ndjson>` — offline, stdlib-only; validated on synthetic decorrelated-noise fixtures (gain x1.13, mismatch guard fires) | this branch, `arc-tools/quality/ensemble_ppl.py` |
