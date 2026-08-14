# GPU SESSION 6 RUNBOOK — publish the bake, then measure the fleet

**Target:** one Runcrate H200 (141 GB HBM, ≥24 cores, ≥720 GB disk; sessions
1–5 were NY, **$4.92/hr**). **Balance at staging: $48.54.** Planned spend
**≈ $35.7 over ≈ 7:15**, hard teardown at **7:45 ≈ $38.1**, leaving ~$10 for a
bad-box re-rent and overrun.

Everything below is executed by a **single prepared driver**,
`arc-tools/quality/s6_driver.sh`, uploaded with `file_upload` and run detached.
The human/agent driving the session runs six commands total (§1) and then
polls one URL. This document exists so that nothing has to be *decided* while
the meter runs.

---

## 0. Why this session is worth renting

Ten PRs merged since session 5 changed what a rental can buy:

| PR | What it unlocked |
|---|---|
| **#33** CUDA beam-search bake kernel | projected **31–62 min** bake vs a measured **~6.2 h** exhaustive bake. Default NOT flipped — **hardware parity must pass first**. |
| **#34** greedy banned structurally + search stamped into UQFF 0.3.0 | a bake can no longer lie about which search produced it |
| **#28** batch prefill causality | `b_sz>1` prefill had **no causal mask** — every batched quality number before this is void |
| **#30** MTP draft seeding + real draft KV | every prior MTP acceptance number is void |
| **#35** SwiGLU clamp + YaRN layers {0,1,43} | changed decode math on **every token's path** — 87.0 % GSM8K and 12.50 PPL are now **PROVISIONAL** |
| **#31** calibration collector + `mistralrs calibrate` | makes DOCTRINE **D11**'s open question measurable |
| **#29 #32 #26 #25** | CPU beam + sensitivity probe · real V4 config fixture · indexer scoring · box-health gate + ISQ thread policy |

Four things have **never been measured on hardware**: the full-serving batch
sweep (D2's headline), a valid MTP acceptance rate, the diag(H) channel spread
of a V4 expert layer, and the beam kernel itself.

And one thing has never been *done*: **publishing the UQFF.** FACTS.md is
explicit — *"FIX THE COST, NOT THE MYSTERY: bake ONCE and reuse the UQFF … 
instead of re-baking every session."* Five sessions have re-baked. Step S4
ends that, and it runs before any measurement so a session that dies at S7
still leaves the artifact.

---

## 1. The six commands you actually type

```bash
# 1. rent  (Runcrate MCP: list_gpu_types -> create_instance -> instance_status)
# 2. upload the prepared scripts (file_upload — NEVER a heredoc through ssh_execute)
#      /root/s6_driver.sh          arc-tools/quality/s6_driver.sh
#      /root/s6_upload_uqff.py     arc-tools/quality/s6_upload_uqff.py
#      /root/box_health_gate.sh    arc-tools/quality/box_health_gate.sh
#      /root/stall_sentinel.sh     arc-tools/quality/stall_sentinel.sh
#      /root/stall_sentinel2.sh    arc-tools/quality/stall_sentinel2.sh
#      /root/s6_status_digest.sh   arc-tools/quality/s6_status_digest.sh
#      /root/boot.sh               arc-tools/boot_run161_h200.sh
#      /root/.hf_token             <the token, one line, no newline drama>

# 3. one ssh_execute to lock the token down and bring up observability
chmod 600 /root/.hf_token && mkdir -p /root/logs && \
nohup bash /root/status_server.sh > /root/logs/status_server.log 2>&1 & \
S6_START_EPOCH=$(date +%s) nohup bash /root/s6_status_digest.sh > /root/logs/digest.log 2>&1 & \
nohup bash /root/stall_sentinel2.sh /root/logs S6_COMPLETE /root/logs/s6.log > /root/logs/sentinel2.log 2>&1 & \
sleep 3; curl -s localhost:8899/status.txt | head -3

# 4. one ssh_execute to launch the session
(nohup bash /root/s6_driver.sh > /root/logs/s6.log 2>&1 < /dev/null &) ; sleep 5; head -4 /root/logs/s6.log

# 5. arm the local watchdog (§7) and STOP TOUCHING THE BOX

# 6. at S6_COMPLETE (or a trip-wire): pull the tarball, DELETE the instance
curl -s -o s6_results.tgz http://<BOX_IP>:8899/s6_results.tgz && tar tzf s6_results.tgz
```

`status_server.sh` also needs uploading if it is not already on the branch
checkout — it is, at `arc-tools/quality/status_server.sh`, but the box has no
repo yet at step 3, so upload it too.

---

## 2. Timeline and cost model

$4.92/hr = **$0.082/min**. Cumulative from *instance creation*.

| # | Step | Wall | Cum | $ step | Cum $ |
|---|---|---|---|---|---|
| S0 | **Box health gate — BEFORE the 149 GB download** | 3m | 0:03 | 0.25 | **0.25** |
| S1 | Bootstrap: build ∥ download, MTP patch, eval corpora | 55m | 0:58 | 4.51 | 4.76 |
| S2 | **CUDA beam parity gate** | 15m | 1:13 | 1.23 | 5.99 |
| S3 | **Bake, beam W=256**, header + pace gates | 60m | 2:13 | 4.92 | 10.91 |
| S4 | **UPLOAD THE UQFF** (68 GB) | 25m | 2:38 | 2.05 | 12.96 |
| S5 | Serve + **batch sweep** + b=1 diag + coherence | 20m | 2:58 | 1.64 | 14.60 |
| S6 | GSM8K n=100 2048-cap (re-measure the provisional 87.0) | 75m | 4:13 | 6.15 | 20.75 |
| S7 | gen-2 GEMV sweep, 98 variants (server down) | 20m | 4:33 | 1.64 | 22.39 |
| S8 | Calibration probe (server down, 25 min hard box) | 25m | 4:58 | 2.05 | 24.44 |
| S9 | Tuned serve + batch re-sweep | 12m | 5:10 | 0.98 | 25.42 |
| S10 | MTP acceptance | 20m | 5:30 | 1.64 | 27.06 |
| S11 | **Voting GSM8K k=5** — the 90+ attempt | 90m | 7:00 | 7.38 | 34.44 |
| S12 | Tar + teardown (**NEVER CUT**) | 15m | 7:15 | 1.23 | **35.67** |

Leaves **$12.87**. A box that fails S0 costs **~$0.30** to abandon — that is
the entire point of gating before the download. A box that fails the S3 pace
gate costs ~$6, which the remainder absorbs once.

### Trip-wires (enforced by the driver, not by memory)

| Cum | Wire | Driver behaviour |
|---|---|---|
| 2:51 | `WIRE_BAKE_H=2.85` | bake still running ⇒ PID-kill, `ABORT_BAKE_TRIPWIRE`, teardown |
| 3:18 | `WIRE_UPLOAD_H=3.30` | past this at end of S4 ⇒ log `TRIPWIRE`, expect to lose S11 |
| 6:00 | `WIRE_LASTCHANCE_H=6.00` | batch sweep drops `--include-128`; voting starts at `--n 60` |
| 7:15 | `WIRE_STOP_H=7.25` | any running eval is PID-killed; JSON keeps its partial `n`; go to S12 |
| 7:45 | `WIRE_TEARDOWN_H=7.75` ($38.13) | **HARD teardown, no matter what** — the watchdog shouts `BUDGET_WIRE` |

### Cut order (from the bottom, with one stated exception)

1. `--include-128` on both batch sweeps *(automatic past 6:00)*
2. **S11 voting `n`**: 100 → 60 → whatever it reached *(automatic; resume-safe JSON, keep n ≥ 60, CI widens to ~±12 pp — say so in the results note)*
3. **S9** tuned batch re-sweep *(S5 already gives the headline on baked defaults)*
4. **S11** voting entirely
5. **S10** MTP acceptance
6. **S8** calibration
7. **S7** gemv sweep

**Never cut:** S0, S2, S3, **S4**, S5, S12.

**The stated exception.** Mission priority lists voting (8) above MTP (9) and
calibration (10). The driver runs MTP and calibration *first* anyway, because
voting costs ~90 min for one number and **degrades gracefully** (per-problem
JSON), while MTP and calibration cost 20 and 25 min for one number each and
degrade to **nothing** if cut. Running the cheap ones first and letting the
expensive one absorb the trip-wire buys three numbers instead of one. Voting
is therefore last and *truncatable*, not droppable.

S6 (plain GSM8K) is deliberately **not** on the cut list: re-measuring the
provisional 87.0 is a correctness obligation after #35 changed decode math on
the shared-expert path every token traverses. If forced, it truncates like
voting.

---

## 3. Step-by-step: what runs, and every ABORT-IF

Markers in `CAPITALS` are what the driver prints and what the watchdog greps.

### S0 — Box health gate (3 min) — **before the download**

```bash
nvidia-smi --query-gpu=compute_cap --format=csv,noheader     # must be >= 8.0
nvidia-smi | grep 'CUDA Version'   vs   nvcc --version       # toolkit <= driver max
bash /root/box_health_gate.sh --burn-secs 60 --json /srv/arcstatus/box_health.json
```

- **ABORT-IF `compute_cap < 8.0`** (`ABORT_BOX_COMPUTE_CAP`). This is not
  pedantry: `mistralrs-quant/build.rs:113` sets `has_qtip_kernels` **only** at
  compute cap ≥ 80, and every CUDA parity test in S2 does
  `if !ffi::HAVE_QTIP_KERNELS { return Ok(()) }` — a **silent vacuous pass**
  (`qtip/mod.rs:5915`, `qtip/bitshift.rs:3309`). On a sub-8.0 box the gate
  would go green with no kernel under test.
- **ABORT-IF toolkit newer than the driver's max CUDA** (`ABORT_BOX_TOOLKIT`):
  builds fine, dies at runtime with `CUDA_ERROR_UNSUPPORTED_PTX_VERSION`
  (driver 580.173 = CUDA 13.0 max; cost session 1 forty minutes).
- **ABORT-IF `box_health_gate.sh` exits 1** (`ABORT_BOX_HEALTH`): sustained
  power < 28.6 % of the enforced limit (< 200 W on a 700 W H200) or any other
  FAIL line. **DELETE the instance and re-rent a different box, preferably a
  different region. Do not debug the rental.** s5a showed **99 % util at 132 W
  of 700 W**, clocks maxed, temps fine — starved on host↔device transfers,
  ~3 min/layer with a binary proven at 30 s/layer, ~1.5 h and ~$7 burned.
  Sampling is done **in a loop under confirmed load**: a single early sample
  reads ~80 W on a healthy box mid-ramp and cries wolf (s5b false alarm).

### S1 — Bootstrap (55 min, build ∥ download)

`boot.sh` honours `ARC_BRANCH` (`arc-tools/boot_run161_h200.sh:31`), clones the
candle fork as a sibling, starts the 149 GB download in the background, builds
`arc-cli` + `mistralrs-cli` with `--features "cuda flash-attn"`, and prints
`BOOTSTRAP_COMPLETE`.

- **NEVER add `cudnn`**: −62 % decode on V4 (5.45 vs 14.58 tok/s, same box,
  same bake, only the feature flag differed — session 4).
- The driver then applies **`patches/s6_mtp_acceptance_telemetry.patch`** and
  rebuilds `mistralrs-cli`. **This is load-bearing:**
  `mtp_pipeline.rs:690 log_acceptance_rate()` has **zero call sites on master**
  (verified at `d6ceaf1ad`), which is exactly why sessions 3/4/5 measured no
  acceptance rate. The session-2-era patch **no longer applies** (PR #30
  rewrote the file); the s6 patch is regenerated against `d6ceaf1ad`, applies
  clean, and `cargo check -p mistralrs-core` is green with zero new warnings.
- **ABORT-IF** `boot.sh` prints `FAIL:` or never reaches `BOOTSTRAP_COMPLETE`
  (`ABORT_BUILD`), or the binary is missing after the patched rebuild.
- **SKIP (not abort)** if the patch fails to apply: `SKIP_MTP_PATCH`, and S10
  will produce nothing. Record it; do not debug.

### S2 — **CUDA beam parity gate** (15 min) — THE GATE

```bash
cargo test -p mistralrs-quant --release --features cuda cuda_ -- --nocapture
```

That filter runs all six relevant tests in one compile:

| test | file | what it pins |
|---|---|---|
| `cuda_beam_matches_cpu_beam_bit_for_bit` | `qtip/mod.rs:4381` | CUDA beam == CPU beam **byte-identically** at **W ∈ {64,128,256}**, with a non-vacuity assert that the beam differs from exhaustive at some width |
| `cuda_beam_unpruned_matches_cuda_exhaustive` | `qtip/mod.rs:4469` | an unpruned beam reproduces the exhaustive DP |
| `cuda_exhaustive_matches_cpu_exhaustive_bit_for_bit` | `qtip/mod.rs:4526` | PR #33's exact-arithmetic fix (`__fadd_rn`/`__fmul_rn`/`__fdiv_rn` vs `--use_fast_math`) |
| `cuda_search_plan_never_substitutes_a_width` | `qtip/mod.rs:4281` | the dispatch may *translate*, never *substitute* |
| `cuda_3d_expert_quantize_does_not_fall_back_to_cpu` ×2 | `qtip/mod.rs:5915`, `qtip/bitshift.rs:3309` | the GPU quantize path really engages |

- **ABORT-IF the log contains `CUDA not available; skipping`**
  (`ABORT_BEAM_VACUOUS`) — the passes mean nothing.
- **The width ladder.** The parity test iterates 64 → 128 → 256 and fails at
  the **first** bad width, naming it: `W={width}: CUDA beam differs from the
  CPU beam in N/M bytes`. The driver parses that and picks the largest width
  strictly below it:

| Outcome | Action |
|---|---|
| all three pass | bake at **W=256** (PR #29: quality-neutral, matmul cos 0.96680 vs exhaustive 0.96495) |
| W=256 fails, 128 passes | bake at **W=128**, `BEAM_GATE=W128 (DEGRADED)` — −0.004 cos; **flag it in the results and in FACTS.md** |
| only W=64 passes | **do not bake.** −0.014 cos (0.95054) is a real regression, and the width is **stamped into the artifact forever** (#33's UQFF flags byte). Tear down. |
| none pass | tear down |

- **ABORT-IF the largest passing width < 128** (`ABORT_BEAM_PARITY`).
  **The fallback is NOT an exhaustive bake.** A ~6.2 h exhaustive bake is ~$30
  of a $48.54 balance, buys the old generation at full price, and leaves
  nothing for measurement — session 5 was killed for exactly this
  (*"I'm not willing to wait 6 hours so you can kill the viterbi machine
  that's wasting money"*). Capture `beam_parity.log`, tar, **DELETE**, fix on
  CPU where the fixtures are free.

### S3 — Bake with the beam (projected 31–62 min at W=256)

```bash
ARC_QTIP_BEAM=<W> MISTRALRS_ISQ_SINGLETHREAD=1 \
  mistralrs quantize text -m "$V4_DIR" -a deepseekv4 --isq qtip2 -o "$V4_DIR/uqff/" \
  --uqff-base-model deepseek-ai/DeepSeek-V4-Flash \
  --uqff-repo-id aeonmind/DeepSeek-V4-Flash-UQFF-qtip2
```

`--uqff-base-model` / `--uqff-repo-id` are not decoration: directory-mode
output generates a README and **prompts interactively** without them
(`args/quantize.rs:148-165`). A detached `nohup` bake that blocks on stdin is
an invisible stall. They also give the published repo a model card for free.

Gates, in order of how much money each has already cost:

1. **`BAKE_HEADER` must print** — polled, not slept for. Format from
   `qtip/mod.rs:670 bake_header_line`:
   `QTIP bake [qtip2]: mode=viterbi search=viterbi-beam(W=256) objective=mse (unweighted) rotation=hadamard-128`.
   **ABORT-IF absent** (`ABORT_BAKE_HEADER`): before #33 the GPU fast path
   returned *before* `log_bake_header`, so **no GPU bake has ever emitted a
   header**. A binary that cannot say what search it ran must not produce a
   published artifact (D4).
2. **ABORT-IF `search=greedy`** (`ABORT_BAKE_GREEDY`) — D4, greedy is banned
   forever, structurally. That would be a code regression, not a session problem.
3. **ABORT-IF the search is anything other than the beam we asked for**
   (`ABORT_BAKE_SEARCH`) — `cuda_search_plan` (`qtip/mod.rs:714`) may translate
   (a beam ≥ 2^L runs the exhaustive kernel) but must never substitute.
4. **ABORT-IF `rotation=` is not `hadamard-N`** (`ABORT_BAKE_ROTATION`) — D11
   makes the rotation the permanent default and its signs are serialized into
   the UQFF and consumed by the forward path.
5. **ABORT-IF a `QTIP GPU quantize fallback` line appears**
   (`ABORT_BAKE_FALLBACK`) — that is the ~11 min/layer CPU crawl (~8 h). The
   warn line names the condition and **is** the deliverable.
6. **ISQ thread policy must read 1 thread.** Fixed in code by #25, and
   `MISTRALRS_ISQ_SINGLETHREAD=1` is belt-and-braces. 24 host threads pushing
   Viterbi work at one device gave 4–9 min/layer with **no warning**
   (session-5 trap); session 3's fast bake logged `1 threads`.
7. **Pace gate, calibrated for the beam, not for exhaustive**:
   **≥ 2 layers at t=3 min** and **≥ 6 layers at t=10 min** (projection is
   42–85 s/layer; the exhaustive kernel's measured 510 s/layer would fail
   both). **ABORT-IF short** (`ABORT_BAKE_PACE`) — re-run
   `box_health_gate.sh --with-bake-probe`; on FAIL delete and re-rent. Do not
   bisect our kernels: session 5 did, and the session-3 binary was just as slow.
8. **ABORT-IF** the output is outside the 40–120 GB band (expected ~68 GB
   across 7 shards) or has no `.uqff` (`ABORT_BAKE_SIZE`).

### S4 — **UPLOAD THE UQFF** (25 min) — the irreversible step

```bash
python3 /root/s6_upload_uqff.py --folder "$V4_DIR/uqff" \
  --repo-id aeonmind/DeepSeek-V4-Flash-UQFF-qtip2 \
  --token-file /root/.hf_token --private
```

The driver first writes `arc_bake_manifest.json` into the uqff dir (arc commit,
base model, beam width, the verbatim bake header, GPU name, UTC) so the
artifact carries its own provenance, then uploads folder + manifest.

**Verified against current huggingface_hub docs, not from memory:**

- `HfApi.upload_folder` is the current API; it commits in batches and
  **resumes** an interrupted upload on re-run (Xet streaming pipeline). The
  driver retries once on failure for exactly that reason.
- **`upload_large_folder` / `hf upload-large-folder` are DEPRECATED** and
  slated for removal. Do not reach for them.
- `HF_XET_HIGH_PERFORMANCE=1` enables hf_xet's high-performance mode;
  `HF_HUB_ENABLE_HF_TRANSFER` is deprecated and no longer used.
- `HF_TOKEN` is read from the environment (`constants.py`
  `_get_token_from_environment`); the script passes it explicitly as well so a
  stale cached token cannot win.

**The teeth:** after uploading, the script calls `list_repo_files` and diffs
against the local inventory. **An upload that silently drops shards is worse
than no upload**, because the next session would trust it. Missing files ⇒
`UPLOAD_FAIL`, non-zero exit.

**Token discipline** (session 5 leaked one in plaintext and it had to be
rotated): 0600 file, never an argv, never printed, scrubbed out of any
exception text, filtered out of the driver's tee, absent from the status
digest's marker allow-list, and `shred`ed at S12. The dry run asserts the token
string never appears in the session log.

- **SKIP (not abort) if the token file is missing** — but understand what you
  are skipping: the bake is only worth something if it is published. Re-run
  `S6_ONLY=S4 bash /root/s6_driver.sh` after uploading the token.

### S5 — Serve + **the batch sweep** (20 min) — THE HEADLINE

```bash
mistralrs serve -p 1234 -m "$V4_DIR" -a deepseekv4 --from-uqff <first shard> \
  --prefix-cache-n 0 --chat-template chat_templates/deepseek_v4.json
python3 run_coherence.py --skip-facts        # 6/6 gate
python3 speed_probe.py --label s6_b1_diag    # DIAGNOSTIC ONLY (D2)
python3 batch_load_probe.py --label s6_baseline --batches 1,8,16,32,64 --include-128
```

- `--chat-template` is **required**: without it `/v1/chat/completions` 422s
  (session-4 root cause of the vote-API failures).
- Four numbers per B: **aggregate decode tok/s**, **per-user p50 at that B**,
  **TTFT p50/p95**, **$/Mtok** (`$/Mtok = 4.92e6 / (agg × 3600)`). The
  `BATCHSWEEP[...]` line carries all of them. The kernel-level anchor to beat
  is s4's **1,006 aggregate tok/s ⇒ ~$1.36/Mtok**, which was a microbench, not
  the engine.
- b=1 is a **kernel-latency diagnostic, never a headline** (D2). Anchor: 14.58.
- The probe is the **sanctioned exception** to "one scored request at a time";
  scored evals (S6, S11) stay serial.
- Pre-session, $0: `python3 test_batch_load_probe.py` (mock server, asserts the
  probe's arithmetic).
- **ABORT-IF the server never becomes healthy** (`ABORT_SERVE`).
- **Flag, don't abort**, if coherence < 6/6 or decode < 10 tok/s — record and
  continue; the A/B against S9 is still valid.

### S6 — GSM8K n=100, 2048-cap, 0-shot greedy (75 min)

87.0 % is **PROVISIONAL**. #35 added the SwiGLU clamp that was missing on 4 of
5 expert paths — **including the shared expert every token traverses in every
layer** — and moved YaRN off the ratio-0 layers **{0, 1, 43}** (the audit's
"0, 1, 42" was wrong; layer 42 is ratio-4). A number on either side of 87.0
**is not a regression**: it is the first number measured on math that matches
the reference. Say that in the results note. Reference: published V4-Flash-Base
**90.8 with 8-shot** — a different, easier protocol; always state it.

- Truncates at `WIRE_STOP_H`; the per-problem JSON keeps whatever `n` it reached.

### S7 — gen-2 GEMV sweep, 98 variants (20 min, server down)

```bash
unset ARC_TUNE_VARIANTS      # <- see below
cargo run --release -p mistralrs-quant --example qtip_gemv_tune --features cuda
```

**`ARC_TUNE_VARIANTS=all` is not a thing.** The parser
(`examples/qtip_gemv_tune.rs:211-257`) accepts `gen1`, `gen2`, id lists, and
inclusive ranges. Anything else selects nothing and falls back to the full grid
*with a warning* — which works by accident and looks like a bug in the log.
**Unset the variable.** The grid is 98 = 44 gen-1 (`qtip_bitshift_tune.cu:336`,
`static_assert(Q2B_GEN2_ID_BASE == 44)`) + 54 gen-2
(`qtip_bitshift_tune2.cu`, cp.async depth 2/3, SMEM-fed ILP, split-K 1/2/4,
producer-warp option).

- **Parse `WINNER_TABLE_WRITTEN: <path>` (`qtip_gemv_tune.rs:469`); never
  glob.** A session-4 glob missed the filename and the "tuned" serve silently
  ran without the table.
- **SKIP-IF** the sweep crashes or writes no table — the baked defaults (v21
  `w4_r2_i1_v2` gate/up, v6 `w8_r4_i1_v2` down) are still a valid row.

### S8 — Calibration probe: the diag(H) channel spread (25 min hard box)

```bash
timeout 1500 mistralrs calibrate text -m "$V4_DIR" -a deepseekv4 \
  --samples 8 --gram none --out results/v4_expert_stats.arccalib
cargo run --release -p mistralrs-quant --example stats_info -- <artifact> --layers
cargo run --release -p mistralrs-quant --example stats_info -- <artifact> --layer <expert layer>
```

`stats_info` prints `diag/tokens min … mean … max … (dynamic range NNNNx)`
(`examples/stats_info.rs:63`). That dynamic range **is** the diag(H) channel
spread, and it closes DOCTRINE D11's open question:

| measured spread | verdict the driver prints |
|---|---|
| < 1e4 | **D11 stands** — `no-rot + Hessian` stays ruled out (at 1,181:1 rotation wins 0.957 vs 0.874) |
| 1e4 – 1e5 | inconclusive band — keep rotation, widen the fixture family |
| > 1e5 | **D11 reopens** — re-run `probe_rotation_vs_hessian_sensitivity` at this dispersion |

**Known risk, stated up front so nobody debugs it on the meter.**
`mistralrs calibrate` loads the model **unquantized**: `commands/calibrate.rs`
builds a `ModelSelected` with `from_uqff: None` and no ISQ, and
`normal.rs:644` routes a calibration run to the regular device with
`loading_isq = false` (and `use_immediate` is already false because
`calibration_file.is_some()`). The V4 checkpoint is ~149 GB on disk against
141 GB of HBM. **It may not fit.**

- **SKIP-IF it OOMs or times out** (`SKIP_CALIB`) — and record the OOM **as the
  finding**: *Arc cannot calibrate its own flagship on one box, because
  `calibrate` has no `--from-uqff`.* That is a product gap worth a PR
  (the sweep only observes layer **inputs**, so a UQFF-loaded model would
  answer the same question inside 68 GB), not a session failure. **Do not
  debug it here.**

### S9 — Tuned serve + batch re-sweep (12 min)

Re-serve with `ARC_QTIP_TUNE_TABLE=<winners>` and re-run the sweep. Kernel
step-time wins are **~B× leveraged** — this is what turns a microbench delta
into a fleet number.

- **Verify the env reached the SERVE process**, not just the shell:
  `tr '\0' '\n' < /proc/$SERVE_PID/environ | grep ARC_QTIP_TUNE_TABLE`.
  The driver prints `TUNE_TABLE_IN_SERVE_ENV=yes|NO`; on `NO`, label the
  numbers as untuned rather than reporting a phantom win (session-4 bug).

### S10 — MTP acceptance (20 min) — the first valid number ever

Serve with `--mtp-depth 2` and `ARC_MTP_LOG_ACCEPTANCE=1`, decode 256 tokens,
then read `MTP acceptance rate: X% (a/p accepted)` (one line per 64 proposed).

- `MTP requested (depth=N) but the loaded model has no MTP head` ⇒ record
  **verbatim** and skip. That line is the deliverable.
- 0 acceptance lines ⇒ the S1 patch is not in this binary or the env did not
  reach the serve process. The driver prints which. Record and move on.
- Speculative win ≈ `1 + depth × acceptance` minus verify overhead; compare
  decode tok/s against S5/S9.

### S11 — Voting GSM8K k=5, confidence-weighted (≤90 min) — the 90+ attempt

`--votes 5 --vote-mode confidence_weighted`, n=100 (or 60 past the 6:00 wire),
2048-cap, seed 161. Chains are sampled t=1.0/p=0.95 — greedy chains would be
five identical copies. Unblocked by **#28** (batch prefill had no causal mask,
so every earlier batched-quality number is void) and **#21** (per-seq
`xs_history`).

- Success bar: **≥ 90.8 beats the published base protocol number**; **≥ 88**
  already isolates a positive voting delta at matched quantization.
- Voting *below* 87.0 at n ≥ 60 is **not** an abort — it is a real negative
  result. Record it (D9).
- Truncates at `WIRE_STOP_H`; keep n ≥ 60 and widen the stated CI to ~±12 pp.

### S12 — Results out + teardown (15 min) — NEVER CUT

Tar `arc-tools/quality/results/` to `/srv/arcstatus/s6_results.tgz`, pull it
over plain HTTP (works even if the Runcrate token TTL killed `ssh_execute`),
verify the listing, `shred` the token, then **DELETE the instance** (delete,
not stop) and confirm with `list_instances` + `billing_balance` in the same
message.

---

## 4. Expected artifacts

`arc-tools/quality/results/` on the box, tarred at S12:

```
box_health.json              beam_parity.txt            beam_width_used.txt
bake_log_excerpt.txt         arc_bake_manifest.json
batch_load_s6_baseline.json  batch_load_s6_tuned.json
speed_s6_b1_diag.json        speed_s6_b1_tuned.json     speed_s6_mtp2.json
coherence.json               gsm8k_chat.json            gsm8k_chat_votes5.json
gemv_tune_sweep.txt          gemv_tune_winners.json
v4_expert_stats.arccalib     calib_layers.txt           calib_layer_<i>.txt
mtp_acceptance.txt           + every /root/logs/*.log
```

Plus, off-box and permanent: **`aeonmind/DeepSeek-V4-Flash-UQFF-qtip2`** with
7 shards, a model card, and `arc_bake_manifest.json`.

---

## 5. Session-wide rules (each one was paid for)

- **`ssh_execute` mangles heredocs and complex quoting.** Upload scripts with
  `file_upload`, then `bash script.sh`. Never build a multi-line script inline.
- **Kill by PID, never by pattern.** `pkill -f 'foo'` matches and murders your
  own SSH command. Every launch captures `$!`; `stall_sentinel2.sh` brackets
  the first character of its `pgrep` pattern (`[m]istralrs`) so it cannot match
  itself.
- **Build without `cudnn`** (−62 % decode).
- **Check `nvidia-smi` CUDA version against `nvcc --version` before building.**
- **Every serve line needs `--chat-template chat_templates/deepseek_v4.json`.**
- **Scored evals run one request at a time**; the batch probe is the sanctioned
  exception.
- **Sample power in a loop with util > 50 % confirmed** — a single early sample
  reads ~80 W on a healthy box mid-ramp (s5b false alarm).
- **Gates OFF all session** unless a step sets them: `ARC_QUANT_ATTENTION`,
  `ARC_QTIP_REFINE_SCALES`, `ARC_V4_WINDOW_ONLY`, `ARC_V4_STANDARD_DENSE`,
  `ARC_V4_NO_ABSORBED_DECODE`, `ARC_FORCE_NAIVE_SDPA`, `ARC_TIME_DECODE`,
  `ARC_QTIP_HESSIAN`, and especially **`ARC_ALLOW_UNSTAMPED_QTIP`** (it would
  let a pre-0.3.0 unstamped artifact serve). The driver aborts
  (`ABORT_ENV_DIRTY`) if any is exported at start.
- **No debugging beyond the ABORT-IF rules.** Findings are deliverables; fixes
  happen offline where compute is free.

---

## 6. Observability

Three processes, brought up **before** the driver:

| Process | What it gives you |
|---|---|
| `status_server.sh` | `http://<IP>:8899/status.txt` — verbose (tail -3 of every log), and serves `s6_results.tgz` at teardown. Survives an expired Runcrate token. |
| `s6_status_digest.sh` | `http://<IP>:8899/s6.txt` — **small**: elapsed, spend, budget wire, one GPU line, last 12 event markers, stall markers. This is what the 60 s watchdog polls. |
| `stall_sentinel2.sh` | three-signal stall detection into `/srv/arcstatus/stall.txt`, folded into the digest |

Plus a per-step `stall_sentinel.sh <log> <secs> <pid>` armed by the driver on
every long step (boot, bake, gsm8k, voting, each serve) — that one **kills the
exact PID** on a frozen log.

**Why v2 exists.** v1 watches one log and kills on no-growth; that misses a step
that is genuinely working but silent for 20 minutes (it would kill it), and a
chain that *died between steps* (it stands down silently). v2 requires three
signals to agree:

```
logs frozen  AND  GPU util < 5%  AND  work processes alive   => STALL_DETECTED
no work processes  AND  S6_COMPLETE absent                   => STALL_DEADCHAIN
```

v2 never kills. Killing stays v1's job, bound to an exact PID.

---

## 7. The local watchdog (60 s, prints only new events)

```bash
prev=""; misses=0
while true; do
  s=$(curl -s --max-time 10 http://<IP>:8899/s6.txt 2>/dev/null || true)
  if [ -z "$s" ]; then misses=$((misses+1)); [ $misses -ge 4 ] && { echo "STATUS_SILENT x$misses"; misses=0; sleep 240; }; sleep 60; continue; fi
  misses=0
  cur=$(echo "$s" | grep -E '^:{6}|^HEALTH_|^BUILD_|^BEAM_|^BAKE_|^PACE\[|^UPLOAD_|^BATCH|^SPEED\[|^GSM8K\[|^WINNERS:|^CALIB_|^MTP_|^RESULTS_TGZ|^S6_COMPLETE|^ABORT_|^SKIP_|^TRIPWIRE|STALL_|panicked|CUDA_ERROR' | sort -u)
  comm -13 <(echo "$prev") <(echo "$cur") 2>/dev/null | grep -v '^$' || true
  prev="$cur"
  h=$(echo "$s" | grep -oE 'cum_h=[0-9.]+' | cut -d= -f2 | head -1)
  if [ -n "$h" ] && awk -v x="$h" 'BEGIN{exit !(x>7.0)}'; then echo "BUDGET_WIRE cum=${h}h — begin teardown"; sleep 300; fi
  sleep 60
done
```

`Monitor` with `persistent: true`, `timeout_ms: 3600000`. **`TaskStop` it the
moment the box is deleted.** One watchdog per session; replace, never stack.

---

## 8. Pre-flight validation performed at staging (all $0, no GPU)

| Check | Result |
|---|---|
| `bash -n` on `s6_driver.sh`, `s6_status_digest.sh`, `stall_sentinel2.sh`, `test_s6_driver.sh` | clean |
| `python3 -m py_compile s6_upload_uqff.py` | clean |
| `git apply --check patches/s6_mtp_acceptance_telemetry.patch` against `d6ceaf1ad` | applies clean |
| `cargo check -p mistralrs-core --lib` with the patch applied | **green, zero new warnings** |
| `s6_upload_uqff.py` guard paths (missing folder / no shard / empty token / no lib) | all refuse with `UPLOAD_FAIL`, exit 1 |
| **`bash test_s6_driver.sh`** — 7 mocked scenarios, 41 assertions | **ALL PASS** |

The dry run found three real bugs before they could cost rental time:

1. `grep -c … \| grep -qx 0` under `set -o pipefail` reports the *first* grep's
   exit 1 on zero matches — the fallback check would have **aborted every
   healthy bake**.
2. The `dynamic range` extraction matched the `e` in `range` and returned two
   lines, producing a malformed `CALIB_VERDICT`.
3. A blind `sleep 180` before reading the bake header was replaced by a poll,
   which both removed a race and now aborts a header-less bake in ~10 s instead
   of 3 paid minutes.

---

## 9. Appendix — verified names and citations (master `d6ceaf1ad`, 2026-08-14)

| Thing | Value | Verified at |
|---|---|---|
| Bake header | `QTIP bake [{rung}]: mode={greedy\|viterbi} search={…} objective={…} rotation={hadamard-N\|off}` | `mistralrs-quant/src/qtip/mod.rs:670` `bake_header_line` |
| Search tag | `viterbi-exhaustive` \| `viterbi-beam(W={width})` | `qtip/viterbi.rs:136` `TrellisSearch::tag` |
| GPU bake now emits the header | two call sites | `qtip/mod.rs:1437`, `qtip/mod.rs:1653` |
| `ARC_QTIP_BEAM` | unset/empty/`0`/`off`/`exhaustive` ⇒ exhaustive; `W ≥ 2^L` ⇒ exhaustive; unparsable ⇒ exhaustive **+ warn** (a typo can never lower quality) | `qtip/viterbi.rs:147` `from_env` |
| CUDA search dispatch | translates, never substitutes; error names `widths 1..=max` | `qtip/mod.rs:714` `cuda_search_plan` |
| Beam kernel max width | 256 (one thread owns one beam slot and one prefix group) | wave13-AF, `qtip_beam.cu` |
| Parity tests | `cuda_beam_matches_cpu_beam_bit_for_bit` (W ∈ 64/128/256), `cuda_beam_unpruned_matches_cuda_exhaustive`, `cuda_exhaustive_matches_cpu_exhaustive_bit_for_bit`, `cuda_search_plan_never_substitutes_a_width` | `qtip/mod.rs:4381 / :4469 / :4526 / :4281` |
| Parity failure message | `W={width}: CUDA beam differs from the CPU beam in {n}/{m} bytes` | `qtip/mod.rs:4438` |
| Vacuous-pass risk | `if !ffi::HAVE_QTIP_KERNELS { return Ok(()) }` — **silent**; `HAVE_QTIP_KERNELS = cfg!(has_qtip_kernels)` set only at compute cap ≥ 80 | `qtip/ffi.rs:12`, `mistralrs-quant/build.rs:113` |
| GPU-fallback guards | `cuda_3d_expert_quantize_does_not_fall_back_to_cpu` (LUT + bitshift rungs) | `qtip/mod.rs:5915`, `qtip/bitshift.rs:3309` |
| Fallback warn line | `QTIP GPU quantize fallback: … is quantizing on the CPU — <reason>` | PR #17, `note_gpu_quant_cpu_fallback` |
| ISQ thread policy | `ISQ thread policy: N thread(s) — <rationale>`; QTIP rungs cap at **1** on a GPU-backed bake | `mistralrs-quant/src/lib.rs`, PR #25 |
| quantize CLI | `quantize text -m <dir> -a deepseekv4 --isq qtip2 -o <dir>/` ; `--uqff-base-model` / `--uqff-repo-id` skip the interactive README prompts | `mistralrs-cli/src/args/quantize.rs:148-165` |
| serve CLI | `--from-uqff` (shards auto-discovered from the first), `--prefix-cache-n 0`, `--chat-template`, `--mtp-depth 0..=8` | `args/model.rs:95`, `args/mod.rs:426/431/456` |
| GEMV sweep | 98 variants = 44 gen-1 + 54 gen-2; ends with `WINNER_TABLE_WRITTEN: <abs path>` (parse it, never glob); `ARC_QTIP_TUNE_TABLE=<path>` overrides the baked table with no recompile | `qtip_bitshift_tune.cu:388` static_assert, `examples/qtip_gemv_tune.rs:469` |
| `ARC_TUNE_VARIANTS` | `gen1` \| `gen2` \| id list \| ranges (`44-97`). **No `all`.** Empty selection ⇒ full grid + warning. **Unset it to sweep everything.** | `examples/qtip_gemv_tune.rs:211-257` |
| calibrate CLI | `calibrate text -m <dir> -a <arch> [--corpus P] [--samples N] [--gram none\|block\|full] [--per-expert] -o <out.arccalib>` (`-o` required) | `mistralrs-cli/src/args/calibrate.rs` |
| calibrate loads **unquantized** | `from_uqff: None`, no ISQ; `loading_isq=false` and the regular device when a calibration file is set | `commands/calibrate.rs:191`, `mistralrs-core/src/pipeline/normal.rs:644` |
| channel-spread readout | `diag/tokens min … mean … max … (dynamic range {:.1}x)` | `mistralrs-quant/examples/stats_info.rs:63` |
| MTP telemetry | `log_acceptance_rate()` has **ZERO call sites on master** ⇒ `patches/s6_mtp_acceptance_telemetry.patch` is mandatory; gated by `ARC_MTP_LOG_ACCEPTANCE=1`, one line per 64 proposed | `mistralrs-core/src/pipeline/mtp_pipeline.rs:690`, `:737` |
| Voting request | `n_votes: usize` (≥2 activates), `vote_mode: "majority"\|"confidence_weighted"`; **`/v1/chat/completions` only**; incompatible with `stream: true` | `mistralrs-server-core/src/openai.rs`, `mistralrs-core/src/arc_boost.rs` |
| Voting response | `choices[0]` = winner; top-level `vote:{mode,winner_index,winner_answer,candidates[]}` | `mistralrs-core/src/response.rs` |
| GSM8K harness | `--n --seed 161 --max-tokens 2048 --votes K --vote-mode … [--client-votes] [--eight-shot]`; output name gains `_8shot`/`_votes5`/`_cvotes5` | `arc-tools/quality/run_gsm8k.py` |
| Batch probe | `--batches 1,8,16,32,64 [--include-128] [--duration N] [--max-ctx T]`; prints `BATCH[B=..]` per B and one `BATCHSWEEP[...]`; `$/Mtok = 4.92e6/(agg×3600)` | `arc-tools/quality/batch_load_probe.py` |
| Boot script | honours `ARC_BRANCH` (default master), picks `WORK`/`MODELS` from the first existing of `/ephemeral /workspace /mnt /root`, backgrounds the 149 GB download, prints `BOOTSTRAP_COMPLETE` | `arc-tools/boot_run161_h200.sh:31` |
| Box health gate | `[--burn auto\|example\|torch\|nvcc\|cargo] [--burn-secs N] [--min-power W] [--with-bake-probe DIR]`; **exit 1 = delete + re-rent**; thresholds documented as one-datapoint heuristics from s5a | `arc-tools/quality/box_health_gate.sh` |
| HF upload | `HfApi.upload_folder` (resumable, multi-commit); `upload_large_folder` **deprecated**; `HF_XET_HIGH_PERFORMANCE=1`; `HF_TOKEN` read from env | huggingface_hub docs, fetched 2026-08-14 |
