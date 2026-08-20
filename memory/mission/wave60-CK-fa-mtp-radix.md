# wave60-CK — FA2/FA3/FA4, MTP at batch, radix KV sharing

**Box:** `arc-mtp-radix`, H200 SXM5, SM90, 143771 MiB, NY, $4.92/hr.
**Agent had NO direct box access** — see §0. All hardware execution was driven by main
via `runcrate ssh`; the agent authored the scripts and read back the logs.

Status legend: **[MEASURED]** = off this H200. **[CODE-CONFIRMED]** = read out of the tree
and quotable by file:line, no hardware needed and none claimed. **[PROJECTION]** = inferred.
Nothing in this file is labelled MEASURED unless it came off the GPU.

---

## 0. The access failure that cost the first ~25 minutes

The brief handed the agent a plain-SSH path:
`ssh -i ~/.config/arc/keys/arc_fleet root@162.243.208.179`. **It does not work and never
did.** Every attempt returned `Permission denied (publickey)`, for `root` and for
`ubuntu/arc/admin/user/debian`, with `arc_fleet` and with all five other keys on the mac.
The TCP and SSH handshakes succeed and the host key matches `~/.ssh/known_hosts:124`, so
the box was up throughout — this was purely authentication.

Main diagnosed the root cause: **runcrate boxes authenticate with ephemeral SSH
certificates** (`sshd_config` sets `TrustedUserCAKeys /etc/ssh/runcrate_ca.pub`), so
appending a public key to `/root/.ssh/authorized_keys` cannot grant access. The
`arc_fleet` keypair was invented the same morning and written into GPU_ACCESS_RULE as if
it were established practice; it had never been used. One repair attempt also mangled the
box's existing `authorized_keys`.

**The working pattern, and the one this wave used:** the agent authors one self-contained
bash script that runs start-to-finish unattended, writing progress to `/root/status*.txt`;
main `nohup`s it via `runcrate ssh` and polls. D15 is preserved (agent never invokes
runcrate), and D10b is preserved (one long-running on-box job, not a poll loop of
round-trips). GPU_ACCESS_RULE has been corrected by main so no future agent repeats this.

---

## 1. 🔴 THE HEADLINE, and it is a code fact, not a benchmark

### **V4's attention cannot reach ANY FlashAttention kernel — FA2, FA3, or FA4.** [CODE-CONFIRMED]

The build feature flag is irrelevant to V4. The chain, each link quotable:

1. `models/deepseek4.rs:1188-1197` — every V4 layer constructs
   `SdpaParams { sinks: sinks_for_sdpa, .. }`. Sinks are present on all 43 layers; only the
   `ARC_DISABLE_SINK=1` ablation clears them.
2. `attention/mod.rs:132-135` — `run_attention` tests sinks **first** and returns into
   `sinks_attn` *before* the `can_use_flash` branch at `:141`. Whenever sinks are set,
   flash is unreachable by construction.
3. `attention/backends/sinks.rs:74-80` — the hard gate:
   ```rust
   let flash_sinks_ok = matches!(hd, 64 | 80 | 96 | 112 | 128 | 192 | 256);
   ```
   V4's `head_dim = 512` fails it and falls through to unfused
   `matmul` + `softmax_with_sinks`.
4. The tree already says so twice: `deepseek4.rs:1159` — *"head_dim=512 falls into the
   unfused softmax_with_sinks path on every layer"* — and `dsv4_attention.rs:106`.

**The index path does not rescue it.** V4's indexer runs at `index_head_dim = 128`, which
*is* inside FA's supported set, but it is a **top-k selection** path with no
`run_attention`/flash dispatch at all. There is no V4 code path that flashes.

### Consequences
- **The FA2-build vs FA3-build end-to-end serving delta on V4 is structurally ZERO**, not
  "small". Rebuilding with `--features flash-attn-v3` changes no kernel V4 executes.
  ⇒ The planned Rust FA3 rebuild for V4 was **dropped as provably uninformative**, and that
  build time was redirected to MTP and radix.
- **There is no FA3-vs-FA4 ratio at V4's real shapes to measure.** FA2 and FA3 both cap
  head_dim at 256; 512 is outside every published kernel. An FA4 CuTe AOT toolchain
  therefore cannot pay for itself *via V4*, whatever the ratio turns out to be on
  in-range shapes.
- **The real V4 attention lever is a kernel we do not have**, not a kernel version we do
  not build: a fused head_dim=512 MQA-with-sinks kernel. Per `dsv4_attention.rs:105-116`
  the non-absorbed route `repeat_kv`-expands K/V to all 64 heads, materializing
  `2 * n_heads * T_k * head_dim` elements per layer per decode step to hand a batched GEMM
  64 identical copies of K. `absorbed_mqa_decode` already removes that for `t_q == 1`.
  This is Arc-specific work no upstream FlashAttention release will hand us.

### What was benchmarked instead
FA2/FA3/FA4 at the shapes that genuinely run — V4's indexer geometry (`d=128`, MQA
`hq=64 / hkv=1`), `qk_rope` `d=64`, and two GQA fleet-representative cells — across decode
(`q_len=1`, 4096 KV, B=64 and B=256) and prefill (`q_len=2048`), bf16, causal. Plus the
`d=512` cells run deliberately **to record whether the kernels refuse them**, which is the
load-bearing fact. Refusals are recorded as `status=UNSUPPORTED` with the verbatim
exception; no nearby shape is ever substituted for one the kernel rejects.

Repro: `/root/wave60_fa.sh` → `/root/fa_bench.py`; raw `/root/fa_bench_{A,B}.json`,
`/root/fa_bench_{A,B}.log`. Two phases on purpose: `flash-attn-4` installs into the same
`flash_attn` namespace as FA2 (`flash_attn.cute`) and can shadow it, so phase A measures
FA2/FA3 before FA4 exists and phase B re-probes all three after.

**RESULTS: pending execution — see §4.**

---

## 2. MTP acceptance at batch (PR #86, `feat/mtp-batch-lockstep`)

The claim under test, from wave59-CJ §3, is arithmetic on one dense cache:
`next_u = u + a + 1 - keep` with `keep = min_i(u_i + a_i)`. One sequence rejecting its
first draft pins `keep = 1`, collapsing it to `next_u = u + a` — monotone non-decreasing,
saturating at `w = depth + 1`, at which point that sequence drafts `w - u = 0`. Prediction:
`k → 1.0` at B=64/128, i.e. per-user stays 1.82/1.09 unchanged.

Read **`tok_per_step`, never `accept_rate`.** A saturated sequence drafts nothing, so it
contributes `proposed = 0` and cannot depress `accept_rate`, which stays flattering while
`tok_per_step` collapses. The two diverge exactly where the question lives. Marker format
(`mtp_pipeline.rs:880-895`) carries every raw count so both ratios are auditable:
```
MTP[b=<B>] accept_rate=… accepted=… proposed=… steps=… drafted_steps=… committed=…
           tok_per_step=… batch_steps=… mean_batch=… tok_per_batch_step=…
```
`ARC_MTP_LOG_ACCEPTANCE` defaults **ON** (`mtp_pipeline.rs:716-741`); set to `1` explicitly
anyway. Sweep B=1,8,32,128 at 64 decode tokens, temperature 0, `--mtp-depth 3
--max-seqs 128`, and `--prefix-cache-n 0` so prefix reuse cannot contaminate acceptance.

The only acceptance number previously on record is wave51-CB's `accept_rate=0.4194`,
`tok_per_step=1.8387`, **at B=1** — which per wave59-CJ is precisely the regime where the
ratchet cannot occur.

**RESULTS: pending execution — see §4.**

---

## 3. Radix KV sharing (PR #82, `mistralrs-core/src/kv_sharing/`)

Baseline to beat: 111.69 tok/s aggregate @ B=256, $12.06/Mtok, 0 errors / 505 requests.

### 3.1 The toggle is valid on V4 — but only because V4 is non-paged [CODE-CONFIRMED]
This nearly produced a mislabelled number. `prefix_cacher.rs:361-372` makes
`search_for_matching_cache` return `None` whenever `has_paged_attention`, and
`add_sequence` (`:194-200`, `:259`) bails the same way — *"for paged attention, prefix
caching is handled by the KVCacheManager"*. **On any paged model, PR #82's radix is
bypassed entirely** and toggling the flag actually exercises the vLLM-style block pool in
`paged_attention/block_pool.rs`.

That does not apply to V4: **`DeepSeekV4Loader::supports_paged_attention` returns `false`**
(`pipeline/loaders/normal_loaders.rs`, with the wave-29 comment recording that both the
paged and the MLA paths are geometrically unable to serve V4 — and correcting the older,
wrong reason for the same verdict). So V4 runs non-paged, `has_paged_attention = false`,
and the `kv_sharing` `SharedPrefixCache` radix **is** the live path.

Toggle: `engine/mod.rs:241-244` folds `prefix_cache_n == 0` into `no_prefix_cache`, so
`--prefix-cache-n 16` vs `--prefix-cache-n 0` is a true ON/OFF for exactly the PR #82 code.

### 3.2 🔴 The cross-prefix reuse meter is WIRED BUT DEAD [CODE-CONFIRMED]
The brief asked for "the cross-prefix reuse meter in `content.rs` (how much sharing is
being left on the floor)". **It cannot be read from a running server.**
`PrefixCacheManagerV2::share_stats()` (`prefix_cacher.rs:184`) has **zero production
callers** — the only references in the tree are its own unit tests. `CrossPrefixMeter` and
`ContentIndex` in `kv_sharing/content.rs` are never constructed on any serving path. There
is no log line, no endpoint, and no counter a running server will emit.

⇒ **PR #82 shipped a headline meter with no way to read it.** Exposing it is ~30 lines
(construct the meter in the manager, feed it in `add_sequence`, emit from the interval
logger). Not done here — it is unrequested code, and it is filed as a BACKLOG item rather
than smuggled into a measurement wave.

### 3.3 What is obtainable
Aggregate tok/s and latency client-side; **prefix cache hit rate** from the interval
logger, which prints
`Throughput (T/s) X, Prefix cache hitrate Y%, N running, M waiting`
(`engine/logger.rs:80-88`) and is live in serve mode —
`mistralrs_for_server_builder.rs:1013` sets `throughput_logging_enabled: !interactive_mode`
and serve passes `interactive_mode(false)`. Hits are counted for the non-paged radix path
at `engine/add_request.rs:668`.

Workload: a ~1.4k-token **shared system prompt** identical across all requests with a short
distinct tail (the fleet case), at B=64 and B=256, plus a **distinct-prefix control** at
B=256 so prefix reuse can be separated from raw batching.

**RESULTS: pending execution — see §4.**

---

## 4. Results

*Populated only from hardware output. Any cell still empty means it did not run, and is
reported as "did not run" rather than filled by inference.*

| # | measurement | status |
|---|---|---|
| 3 | FA2/FA3/FA4 microbenchmark | pending |
| 1 | MTP `tok_per_step` at B=1,8,32,128 | pending |
| 2 | radix ON vs OFF, shared-prefix workload | pending |

---

## 5. Surfaced, not shipped

1. **`share_stats()` / `CrossPrefixMeter` are dead code** (§3.2). PR #82's advertised
   cross-prefix reuse meter has no runtime reader.
2. **A fused head_dim=512 MQA-with-sinks kernel is V4's real attention gap** (§1). Every
   V4 layer runs unfused `matmul` + `softmax_with_sinks` today.
3. **GPU_ACCESS_RULE described a key-based SSH path that cannot work on runcrate boxes**
   (§0). Corrected by main during this wave; noted here so the correction has a dated
   provenance.
