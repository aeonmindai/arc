# GPU SESSION RUNBOOK — Arc first quality numbers (V4-Flash + QTIP-2bit)

**Target:** one Runcrate H200 (141GB HBM, 44 cores, 1500GB disk, Helsinki,
~$4.85/hr). **Budget: 8 GPU-hours** — the plan below sums to ~5h with margin.
Every step has: command, expected wall time, expected output, failure recovery,
and an **ABORT IF** rule. A driver (human or agent) can execute it mechanically
top to bottom. The branch to run is **`run161-quality-harness`** (= RUN-161 +
this harness; candle comes as a git dep since 57a4ff203 — no sibling clone
needed by cargo).

**What this session produces** (all in `arc-tools/quality/results/` on the box):
`speed_baseline.json`, `coherence.json`, `gsm8k_chat.json`, `longctx.json`,
`ppl_qtip2_c1024.json`, `ppl_qtip2_c64.json`, `ppl_q2k_c1024.json`
(+ optional `ppl_q3k_c1024.json`), `sinkhorn_verdict.json` +
`ppl_sink_verdict.json`, `speed_fused_sinkhorn.json`, `decode_profile.txt`.

**Session-wide rules**
- ONE request at a time; server always runs `--prefix-cache-n 0` (global
  compressor xs_history is not sequence-managed).
- Gates that stay **OFF** all session unless a step says otherwise:
  `ARC_QUANT_ATTENTION` (2-bit attn: measured no-gain + breaks coherence),
  `ARC_QTIP_REFINE_SCALES` (known-buggy), `ARC_V4_WINDOW_ONLY` (ablation only),
  `ARC_FORCE_NAIVE_SDPA` (bisection only), `ARC_TIME_DECODE` (its syncs skew
  every timing — step 9 only).
- Paged attention auto-disables for V4 (head_dim 512 unsupported) — no flag needed.
- The perplexity binary and the server CANNOT run simultaneously (two model
  instances → CUDA OOM). Steps are ordered so only one is ever up.
- All quality steps run against the **same one UQFF bake** — never re-bake
  mid-session or numbers stop being comparable.

**Timeline at a glance** (cumulative, expected):

| # | Step | Wall time | Cum. |
|---|---|---|---|
| 0 | Rent box + SSH | 10m | 0:10 |
| 1 | Bootstrap (build ∥ 148GB download) | 40-70m | 1:20 |
| 2 | Eval data fetch | 2m | 1:22 |
| 3 | UQFF bake (qtip2) | 7-20m | 1:42 |
| 4 | Serve + smoke gate (speed, coherence6) | 15m | 1:57 |
| 5 | Full coherence + sinkhorn baseline capture | 15m | 2:12 |
| 6 | GSM8K 150 greedy | 30-60m | 3:12 |
| 7 | Long-context ladder | 15m | 3:27 |
| 8 | PPL ladder (server DOWN) | 60-90m | 4:57 |
| 9 | Fused-Sinkhorn A/B (bit-identity + tok/s) | 30m | 5:27 |
| 10 | ARC_TIME_DECODE re-profile | 15m | 5:42 |
| 11 | Pull results + **DELETE INSTANCE** | 15m | 5:57 |

~6h worst-case ≈ **$29**. If cumulative time hits **6:30 before step 8
finishes**, skip the optional q3k rung and step 10; if it hits **7:00**, jump
straight to step 11 — a torn-down box with partial results beats a complete
run that overruns the budget.

---

## Step 0 — Rent the box (10m)

Runcrate: 1× H200, Helsinki, 1500GB disk, latest CUDA 12.x image. Add your SSH
key. Note the IP. `ssh root@<IP>` must work before proceeding.

- **Check immediately:** `nvidia-smi` shows H200 141GB; `df -h` shows ≥1.2TB
  free on some mount; `nvcc --version` exists (or /usr/local/cuda-*/bin/nvcc).
- **ABORT IF** no nvcc-compatible CUDA ≥12.0 or disk <400GB → delete instance,
  pick another template. You've burned ~$1, not $40.
- HF_TOKEN: not needed (model repo is public — June download ran tokenless;
  GSM8K/wikitext are public). If `hf download` 401s anyway:
  `export HF_TOKEN=<token>` and rerun.

## Step 1 — Bootstrap: clone + build ∥ model download (40-70m)

```bash
ssh root@<IP>
curl -fsSL https://raw.githubusercontent.com/aeonmindai/arc/run161-quality-harness/arc-tools/boot_run161_h200.sh -o /root/boot.sh
ARC_BRANCH=run161-quality-harness nohup bash /root/boot.sh > /root/boot.log 2>&1 & disown
tail -f /root/boot.log   # STEP/OK/FAIL markers; Ctrl-C detaches, script keeps going
```

The script is idempotent/resumable: build runs concurrently with the ~148GB
download (background, `$WORK/dl.log`, `.download_done` marker). While it runs,
build the perplexity example too (needed by step 8):

```bash
cd /root/work/arc && cargo build --release -p mistralrs --example perplexity --features "cuda flash-attn" 2>&1 | tail -3
```

Wait for `BOOTSTRAP_COMPLETE` in /root/boot.log. Then set the session env
(paste into every new shell, or add to /root/.bashrc):

```bash
export ARC=/root/work/arc V4_DIR=/root/models/DeepSeek-V4-Flash
export Q=$ARC/arc-tools/quality
mkdir -p /root/logs
```

- Expected: `arc at <sha> run161-quality-harness`, binaries in
  `$ARC/target/release/`, `$V4_DIR/.download_done` exists.
- Recovery: download stalls → rerun boot.sh (hf resumes). Build fails on
  candle git dep → `cargo fetch` once more (transient network); still failing →
  check the dep pins to `aeonmindai/candle` branch `run-161-cuda-graph-capture`.
- **ABORT IF** build fails twice on the same compile error (not network): the
  branch is broken for this CUDA version — capture log, tear down, fix offline.

## Step 2 — Eval data (2m)

```bash
bash $Q/fetch_data.sh
```

Expected: `DATA_READY`, gsm8k_test.jsonl 1319 lines, 4 wikitext files.
Recovery: GitHub raw blocked → script auto-falls-back to HF parquet.
No abort condition (this step is trivial and offline-fixable).

## Step 3 — UQFF bake, qtip2 (7-20m)

```bash
cd $ARC
nohup ./target/release/mistralrs quantize text -m "$V4_DIR" -a deepseekv4 \
  --isq qtip2 -o "$V4_DIR/uqff/" > /root/logs/bake.log 2>&1 & disown
tail -f /root/logs/bake.log
ls -lh "$V4_DIR"/uqff/          # note the first shard name:
export UQFF0=$(ls "$V4_DIR"/uqff/qtip2*.uqff | sort | head -1)
```

- Expected: shards `qtip2-*.uqff` (or single `qtip2.uqff`), total **~65-80GB**
  (experts 2-bit trellis; attention native FP8 + lm_head are NOT quantized —
  that's by design, the "FP8-attention baseline" IS this config).
- June anchor: ~7 min (batched Viterbi across 44 cores).
- Recovery: OOM during bake → rerun with `--cpu` added (slower but bounded).
- **ABORT IF** bake exceeds 60m or output totals <40GB / >120GB — the ISQ
  regex set isn't matching what it should; capture log, stop session.

## Step 4 — Serve + smoke gate (15m) — FAIL-FAST POINT

```bash
cd $ARC
nohup ./target/release/mistralrs serve -p 1234 -m "$V4_DIR" -a deepseekv4 \
  --from-uqff "$UQFF0" --prefix-cache-n 0 > /root/logs/serve_base.log 2>&1 & disown
until curl -s localhost:1234/health >/dev/null; do sleep 5; done && echo UP
python3 $Q/speed_probe.py                       # calibrates the whole session
python3 $Q/run_coherence.py --skip-facts        # the 6/6 gate
```

- Expected: UQFF load ~11s + weights mmap; `SPEED[baseline]: decode ~640 tok/s`
  (June H200 anchor); `coherence6 6/6` or 5/6.
- Recovery: from-uqff "file not found" → `cd "$V4_DIR" &&` serve with relative
  `--from-uqff uqff/qtip2-0.uqff`. Server crash on load → check
  /root/logs/serve_base.log tail; one retry.
- **Budget recalibration:** decode <300 tok/s → run GSM8K with `--n 100`;
  <100 tok/s → `--n 50` and skip step 7's `--deep`.
- **ABORT IF** coherence6 ≤ 3/6 or every output degenerate → the bake or build
  is broken; do NOT proceed to burn hours on GSM8K. Triage (30m max): re-run
  with `ARC_FORCE_NAIVE_SDPA=1` server restart (SDPA-path bisection); if still
  garbage, capture logs + teardown.

## Step 5 — Full coherence + Sinkhorn baseline capture (15m)

```bash
python3 $Q/run_coherence.py                                # facts+math too
python3 $Q/run_sinkhorn_ab.py capture --label baseline     # greedy 6x128tok
```

Expected: facts ≥18/22, math ≥6/8 (June eval.py was in this range).
The baseline capture MUST happen on this same server process (same weights,
gate off) — it's the reference half of step 9's identity check.

## Step 6 — GSM8K (30-60m at ≥300 tok/s)

```bash
python3 $Q/run_gsm8k.py          # 150 problems, seed 161, greedy, resume-safe
```

- Progress + running accuracy + ETA print per problem; results JSON rewritten
  every item — an SSH drop or crash loses nothing (`--resume` default).
- Expected: accuracy in the 70-90% band (anchor: Base 8-shot 90.8, EXPECTED.md);
  `degenerate: 0-3`; `truncated` small.
- Recovery: repeated `[ERROR ...]` items → server died; restart serve (step 4
  block), rerun (resumes). ETA blowing past 90m → Ctrl-C, rerun with `--n 100`
  (already-done items are kept).
- **ABORT IF** running accuracy <30% after 50 problems AND outputs are
  gibberish (cross-check coherence.json — if coherence was 6/6 but GSM8K is
  ~0, suspect the answer extractor, not the model; save 3 `text_tail`s and
  continue anyway).

## Step 7 — Long-context ladder (15m)

```bash
python3 $Q/run_longctx.py            # add --deep if ≥45m ahead of schedule
```

- Validates the June compressor fix (337fd139a) — coherence rungs at ctx>128
  are the HARD gate; needle rungs are soft.
- Expected: coherence 5/5 (6/6 with --deep); needle ≥~half.
- Recovery/triage: coherence failures → restart server with
  `ARC_V4_WINDOW_ONLY=1` (env on serve command), rerun; if coherence recovers,
  the compressed branch is the culprit — that's a REAL finding, record it in
  the results and continue the session (don't debug on the meter).

## Step 8 — PPL ladder (60-90m; server must be DOWN)

```bash
pkill -f "mistralrs serve"; sleep 10
MODEL_DIR="$V4_DIR" UQFF0="$UQFF0" bash $Q/run_ppl.sh          # 3 rungs
# only if ≥60m of budget headroom remains:
MODEL_DIR="$V4_DIR" UQFF0="$UQFF0" RUN_Q3K=1 bash $Q/run_ppl.sh
```

- Rungs: qtip2@1024 (~15m), qtip2@64 on the 100KB slice (~15m), q2k@1024
  (load+in-situ quant ~20-40m + run ~15m). q3k is opt-in: ~120GB weights on
  141GB — OOM is an acceptable outcome, the script WARNs and moves on.
- Expected ordering: q3k < q2k ≈/≥ qtip2 (see EXPECTED.md health heuristics);
  wikitext-2 ppl for a healthy 280B-class model lands in single digits.
- Recovery: a rung with `no 'Final perplexity'` → read its .log tail; OOM on
  q2k → rerun ladder with `PPL_FILE=$Q/data/wiki.test_mini.raw` for that rung
  only (noisier but nonzero signal).
- **ABORT IF** qtip2_c1024 ppl is >50 (catastrophic — quant pipeline broken)
  → skip q2k/q3k, go to step 9 (its bit-identity halves don't need good ppl).

## Step 9 — Fused-Sinkhorn A/B: bit-identity + tok/s (30m)

Half 1 — numerical identity through the perplexity binary (server still down):

```bash
MODEL_DIR="$V4_DIR" UQFF0="$UQFF0" bash $Q/run_ppl.sh --sinkhorn-ab
# PASS = "SINKHORN-PPL-IDENTITY: PASS (bit-identical per-chunk ppl)"
```

Half 2 — greedy token identity + throughput on the served path:

```bash
cd $ARC
nohup env ARC_FUSED_SINKHORN=1 ./target/release/mistralrs serve -p 1234 \
  -m "$V4_DIR" -a deepseekv4 --from-uqff "$UQFF0" --prefix-cache-n 0 \
  > /root/logs/serve_fused.log 2>&1 & disown
until curl -s localhost:1234/health >/dev/null; do sleep 5; done && echo UP
python3 $Q/run_sinkhorn_ab.py capture --label fused
python3 $Q/speed_probe.py --label fused_sinkhorn
python3 $Q/run_sinkhorn_ab.py compare        # vs step-5 baseline capture
pkill -f "mistralrs serve"; sleep 10
```

- Verdict matrix: **both halves PASS** → fused kernel is validated on H200;
  record tok/s delta vs `speed_baseline.json` (the ~13k-launch Sinkhorn swarm
  was ~18% of decode — expect a visible win). **Either half FAILS** → gate
  stays off; `sinkhorn_verdict.json` carries the first-divergence details —
  that IS the deliverable, don't debug on the meter.
- Note: kernel only fires on CUDA+F32 comb tensors; if tok/s is identical to
  baseline AND identity trivially passes, grep serve_fused.log to confirm the
  kernel actually engaged (fall-through on error is silent by design).

## Step 10 — ARC_TIME_DECODE re-profile (15m; skip if >6:30 cum.)

```bash
cd $ARC
nohup env ARC_TIME_DECODE=1 ./target/release/mistralrs serve -p 1234 \
  -m "$V4_DIR" -a deepseekv4 --from-uqff "$UQFF0" --prefix-cache-n 0 \
  > /root/logs/serve_timed.log 2>&1 & disown
until curl -s localhost:1234/health >/dev/null; do sleep 5; done && echo UP
python3 $Q/speed_probe.py --runs 1 --tokens 128 --label timedecode
grep "ARC_TIME_DECODE" /root/logs/serve_timed.log | tail -40 > $Q/results/decode_profile.txt
pkill -f "mistralrs serve"
```

- Compares against the June split (fp8_matmul 31.5% / qtip_dequant 26.5% /
  gather_gemv 11.8% / Sinkhorn ~18%) — with the compressor fix + (if PASSed)
  fused Sinkhorn, the Sinkhorn share should collapse.
- The tok/s printed here is NOT comparable to step 4 (per-component syncs) —
  only the percentage split matters.

## Step 11 — Pull results + TEARDOWN (15m) — DO NOT SKIP

```bash
# on the box
cd $Q && tar czf /root/quality_results.tgz results/ && ls -lh /root/quality_results.tgz
# from your machine
scp root@<IP>:/root/quality_results.tgz .
scp root@<IP>:/root/logs/serve_base.log ./logs_serve_base.log   # optional logs
tar tzf quality_results.tgz    # verify BEFORE deleting the box
```

Then **DELETE the Runcrate instance** (not stop — DELETE; stopped instances
can still bill for disk). Verify in the Runcrate console/API that the instance
list no longer shows it. The box is disposable by design: everything
non-reproducible is in the tarball, the branch, and this runbook.

---

## Appendix — known-good env/flags cheat sheet (verified against branch code)

| Thing | Value | Where verified |
|---|---|---|
| Serve | `mistralrs serve -p 1234 -m <dir> -a deepseekv4 --from-uqff <shard0> --prefix-cache-n 0` | `mistralrs-cli/src/args/{model,server,mod}.rs` |
| Bake | `mistralrs quantize text -m <dir> -a deepseekv4 --isq qtip2 -o <dir>/uqff/` | `mistralrs-cli/src/args/quantize.rs` (dir mode names `qtip2.uqff`) |
| PPL | `target/release/examples/perplexity -m <dir> -a deepseekv4 -f <txt> [-u <uqff> \| -i q2k] --chunk-size N` | `mistralrs/examples/advanced/perplexity/main.rs` |
| ISQ names | `qtip2` (only QTIP rung; **no qtip3/qtip4 exist**), `q2k`, `q3k` | `mistralrs-core/src/pipeline/isq.rs`, `IsqType` |
| qtip2 scope | routed experts only; attention stays native FP8; lm_head excluded | `normal_loaders.rs` `isq_layer_regexes` (RUN-161 comments) |
| Sinkhorn gate | `ARC_FUSED_SINKHORN=1`, CUDA+F32 only, silent fallback | `models/dsv4_mhc.rs` |
| Decode profiler | `ARC_TIME_DECODE=1`, logs per-forward component split | `models/deepseek4.rs` |
| Ablations | `ARC_V4_WINDOW_ONLY=1`, `ARC_FORCE_NAIVE_SDPA=1` | `deepseek4.rs`, `attention/mod.rs` |
