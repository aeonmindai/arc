# Wave 53 — CD: V4 + PagedAttention, settled by experiment. The flag is real, the win is not.

**Scope.** Flip `DeepSeekV4Loader::supports_paged_attention` behind an env var,
trace what actually changes, and run it on hardware. Branch
`feat/v4-paged-attn-probe`, PR #79. Base: `origin/master` @ `d4cf5a2d7`.
Every claim is CONFIRMED (code read, `file:line`) or MEASURED (on the box).

---

## 0. The one-paragraph answer

**The arm is reachable, the two objections on record do not bind, and the KV
hazards do not bind either — but flipping the flag is a net LOSS, not a win.**
Enabling it turns off V4's only CUDA-graph decode arm (they are arms of the same
`match`), turns off MTP, adds a full O(context) gather per layer per decode step,
and still does not reach GPU-autonomous decode because
`AutonomousDecodeRunner::capture` has no caller anywhere in the workspace. The
mission's causal chain — *flag false ⇒ `cache_config` None ⇒ CUDA graphs never
run* — has the right first link and the wrong second one: `cache_config` gates
only the **tier-3 autonomous decode loop**, and that loop has a second, harder
gate behind it. The tier-1/2 graph capture at `normal.rs:1554` never depended on
`cache_config` at all.

---

## 1. Record correction — nobody ever flipped this flag

The brief's 🔑 finding was that `git log -S 'supports_paged_attention'` shows the
value set `false` in `ba026e9d1`, *"perf(v4): ARC_TIME_DECODE per-component
decode profiler"* — a drive-by change inside a profiler commit.

**That is not what that commit is.** `git rev-list --parents -n 1 ba026e9d1`
returns the hash alone: **it is a root commit, no parents.** `--stat` says
**1,430 files changed, 604,635 insertions(+)**, and `normal_loaders.rs` appears
in it as a **new file** (`new file mode 100644`, `@@ -0,0 +1,5826 @@`). It is
this fork's squashed import of upstream mistral.rs, carrying whatever commit
message was on the tip at import time. `git log -S` found it because that is
where the *string* first appears in this history.

⇒ **The flag has been `false` since the first commit in this repository. It was
never flipped by anyone.** The "cheapest large win available" framing rests on a
`git log -S` artifact.

---

## 2. What actually changes when the flag is `true`

`pipeline/normal.rs:345-347` is the single gate:
```rust
if !self.inner.supports_paged_attention(&config)? {
    paged_attn_config = None;
}
```

Downstream of it, in order:

| # | Site | Effect |
|---|---|---|
| 1 | `normal.rs:1155-1206` | `calculate_cache_config` + `CacheEngine::new` run ⇒ `cache_config = Some(..)`, `cache_engine = Some(..)` |
| 2 | `normal.rs:1318` | `metadata.cache_config` becomes `Some` |
| 3 | `deepseek4.rs:3617-3623` | `AttentionImplementation::PagedAttention` ⇒ every layer gets `paged_attn: Some(PagedAttention::new(cfg.head_dim=512, ..))` |
| 4 | `deepseek4.rs:1537-1571` | the `Some(paged_attn)` arm of `match &self.paged_attn` becomes live |
| 5 | engine | the scheduler hands `CacheBackendMetadata::PagedAttention` instead of `DefaultInstructions` |

### 2a. Cache geometry is legal — head_dim=512 never binds

`deepseek4.rs:3783-3791` reports `num_kv_heads: 1`, `k_head_dim: 512`,
`v_head_dim: 512`. `CacheEngine` therefore allocates
K `[num_blocks, 1, 512/x, block_size, x]` and V `[num_blocks, 1, 512, block_size]`.

Both kernels take `head_size` as a **runtime `int32_t`** and index generically —
there is no `switch (head_size)` on either:
* `reshape_and_cache_kernel.cu:40-41,53-69` — `for (i = threadIdx.x; i < num_heads*head_size; i += blockDim.x)`, `dim3 block(std::min(num_heads*head_size, 512))`.
* `gather_kv_cache_kernel.cu:53,79-94,151` — same shape.

At BF16, `x = 16/sizeof(scalar_t) = 8`, and `512/8 = 64` divides evenly.
`dim3 block(min(1*512, 512)) = 512` threads. **Legal.**

### 2b. `--pa-cache-type` auto-falls back, so TurboQuant does not silently engage

`paged_attention/mod.rs:306-308`: *"TurboQuant only supports standard-layout
models with head_dim == 128"*. V4's 512 falls back to `PagedCacheType::Auto`
(unquantised) when the type is left at its default, and hard-errors only if
`--pa-cache-type turboquant` is forced. So the default CLI does **not** quietly
3.5-bit the paged KV.

---

## 3. Does V4 keep its own attention math? YES — this is the decisive answer

`deepseek4.rs:1561-1570`: the paged arm calls
`PagedAttention::cache_write_and_gather` (which is only `reshape_and_cache` +
`gather_kv_cache`) and then hands the gathered K/V straight to
**`dsv4_attention`** — the same function the non-paged arm uses at `:1609` and
`:1660`. `PagedAttention::forward` is never called from `deepseek4.rs`; the sole
`paged_attn.` call site in the file is `:1553`.

⇒ **Objection (1) — "head_dim=512 exceeds the paged kernel's sizes
(`pagedattention.cuh:714`)" — does not bind.** That switch lives inside
`PagedAttention::forward`. Confirmed; the in-code comment already said so.

⇒ **Objection (2) — "the MLA paged path cannot serve V4" — is TRUE but
IRRELEVANT on V4's path.** `flashinfer_mla_decode.cu:12-13` is how V2/V3 get
paged attention. V4 does not route through `mla/forward.rs` at all. The
objection is about a function V4 never calls, exactly like objection (1).

**Sliding window, attention sinks, and CSA/HCA folding all survive**, because
they live in `dsv4_attention`, which the paged arm still calls. Nothing
re-routes V4's attention to a kernel that cannot express them.

---

## 4. The KV-layout hazards do not bind either

The brief flagged this as the live hazard, and it is the right question — but
the answer is no, for a reason the code states outright.

* **1-wide V marker (PR #63):** built by `v4_v_marker`, used only inside
  `append_kv_mqa` (`deepseek4.rs:2352-2356`) and `append_graph_kv_mqa`
  (`:2382`). Both are called **only from the `None` (non-paged) arms**
  (`:1623`, `:1599`).
* **U8 FP8 K codes (PR #72):** `append_kv_mqa(kv_cache, &k, k_packed.filter(|_| v4_fp8_kv_enabled()))`
  — same call site, same arm.
* **The paged arm builds a full dense V on purpose.** `deepseek4.rs:1552`:
  `let v = k.copy()?;` — with its own comment explaining why `copy()` not
  `clone()` (aliased storage ⇒ `RwLock` self-deadlock in `reshape_and_cache`).
  `:1482-1486` states the split explicitly: *"A materialised `v` survives only on
  the PagedAttention arm."*

⇒ `reshape_and_cache` sees a dense BF16 `[N, 1, 512]` K and an identical dense
BF16 V. **Neither PR #63 nor PR #72 reaches it.** (Stale-doc note: the comment
at `:1475` names `append_v_marker`; the function is actually `v4_v_marker`.)

### 4a. And the counterfactual — a 1-wide V would fail LOUDLY, not silently

Worth answering anyway, because "does `reshape_and_cache` cope with a 1-wide V?"
is the right question to have asked. It does not cope, and it says so.
`mistralrs-paged-attn/src/cuda/backend/paged_attention.rs:541-544`:

```rust
let (num_tokens, num_heads, head_size) = k_l.shape().dims3()?;
if (num_tokens, num_heads, head_size) != v_l.shape().dims3()? {
    candle::bail!("shape mismatch k {:?} and v {:?}", k_l.shape(), v_l.shape())
}
```
followed by two more shape gates against the cache tensors (`:546-562`). Passing
V4's `[N, 1, 1]` marker would abort with
`shape mismatch k [N, 1, 512] and v [N, 1, 1]` before a single byte was written.

⇒ The 1-wide-V hazard is **real in principle, unreachable in practice, and
fail-loud if it ever becomes reachable.** That is the good outcome: it is not a
silent-corruption class, so a future optimisation that tried to reuse the marker
on the paged arm to halve the KV footprint would be stopped at the first token
rather than degrade quality invisibly. Note the footprint cost is genuine —
`v = k.copy()` makes paged V4 store **2× the bytes per token** (2,048 B/token/
layer) that the non-paged arm stores.

---

## 5. What the flag COSTS — the part nobody had traced

### 5a. It disables V4's only CUDA-graph decode arm

`deepseek4.rs:1537-1620` is one `match &self.paged_attn`:
```rust
match &self.paged_attn {
    Some(paged_attn) => { ...cache_write_and_gather + dsv4_attention... }   // :1538
    None if crate::layers::has_graph_mode_positions() && seq_len == 1 => {  // :1594
        ...append_graph_kv_mqa: fixed-width, shape-constant, capturable...
    }
    None => { ...plain KvCache... }                                         // :1620
}
```
Once `paged_attn` is `Some`, the `Some(..)` arm **shadows** the graph arm. The
fixed-capacity shape-constant KV write that exists precisely so a graph can be
captured becomes unreachable.

⇒ **`ARC_V4_PAGED_ATTN=1` and `ARC_V4_CAPTURE_PROBE=1` are mutually exclusive by
construction.** The flag the brief expected to *unlock* CUDA graphs is the flag
that *forecloses* the only graph path V4 has.

### 5b. It disables MTP

`mtp_pipeline.rs:2160-2166`: `take_fast_path` requires
`matches!(backend_metadata, CacheBackendMetadata::DefaultInstructions { .. })`.
Under paging the engine sends `PagedAttention`, so V4's MTP fast path is off.

### 5c. It does not reach GPU-autonomous decode — there is a second gate

`normal.rs:1841-1845` is the `cache_config` bail the brief identified, and yes,
it stops firing. But `normal.rs:1955-1961` then **returns `Ok(None)`
unconditionally** on the allocating call, and every later call hits
`normal.rs:1970`:
```rust
if !runner.is_captured() { return Ok(None); }
```
`AutonomousDecodeRunner::capture` / `capture_via_decode_forward`
(`arc-cuda-graph/src/autonomous.rs:297,213`) have **no caller anywhere in the
workspace.** Grepping all eight crates (`mistralrs-core`, `arc-cuda-graph`,
`arc-engine`, `arc-cli`, `arc-bench`, `mistralrs`, `mistralrs-server-core`,
`mistralrs-cli`), excluding `target/`:

* `capture_via_decode_forward` → 3 hits, all in `autonomous.rs`, and the two
  besides the definition are its own doc comments. **Zero call sites.**
* `.capture(` → 2 hits: `autonomous.rs:290` (inside
  `capture_via_decode_forward`, i.e. the dead function calling itself onward)
  and `pipeline/mod.rs:609`, which is a **comment** saying a pipeline *"must
  call `runner.capture(&forward_fn)`"*. Nothing does.

The whole capture chain is unreachable: the only entry point into it is a
function nobody calls.

⇒ `is_captured()` is permanently false. **`cache_config` was never the binding
constraint.** wave29-BC §3 called this "necessary but not sufficient"; it is
still exactly right, and the missing piece is the capture call, not the
`PagedAttentionMeta` bridge (`prime_for_step` **is** wired now, at
`normal.rs:1918` and `:1989` — that half of wave29's note is out of date).

### 5d. It adds an O(context) gather per layer per decode step

`cache_write_and_gather` gathers `N_total = sum_i(seqlen_i)` rows through
`gather_kv_cache` and lifts them to `[1, H, N_total, D]`
(`paged_attention.rs:499-509`) so `dsv4_attention` can read a dense tensor. Under
paging V4 pays block allocation **and** a full context gather, 43 layers × every
decode step. The non-paged arm does not (`raw_keep_span` narrows to the window).

### 5e. It is b=1 only

`v4_paged_dispatch_precheck` (`deepseek4.rs:841-852`) refuses `bs > 1`, because
the gather returns a **varlen pack, not a batch**. And the `xs` compressor
history in the extra `NormalCache` slots is never cloned in/out under the
engine's PagedAttention arm (wave29-BC §4b). Both hold.

---

### 5c-bis. The "three separate bails" are one cause seen three times, plus one the flag cannot fix

The profiler reported bails at `normal.rs:1554`, `normal.rs:1844`, and
`pipeline/mod.rs:1088`. Reading them:

| site | what it is | does the flag stop it? |
|---|---|---|
| `pipeline/mod.rs:1088` | `CacheBackendMetadata::PagedAttention {..} =>` — the `step()` arm V4 never enters | **yes** |
| `normal.rs:1844` | `autonomous_decode`: `cache_config == None` | **yes** — and the next gate (`is_captured()`, §5c) fires instead |
| `normal.rs:1554` | `if probe && seq_len == 1 && self.cuda_graph_runner.is_some()` | **no** — `probe` is `ARC_V4_CAPTURE_PROBE` (`normal.rs:1528`), nothing to do with paging |

So two of the three are the same root cause (flag false) observed at two depths,
and the third is an unrelated, separately env-gated probe that the flag makes
*worse*: `ARC_V4_CAPTURE_PROBE` drives `has_graph_mode_positions()`, and §5a
shows the arm it feeds is shadowed once `paged_attn` is `Some`.

### 5f. 🔴 THE LANDMINE — the flag makes `try_dedicated_decode` *reachable*, and it computes the wrong model

This one cuts the other way from §5a–§5e and is the most important thing in this
document. `graph_wrapped_forward` (`pipeline/mod.rs:627-644`) tries the
**DedicatedDecodePath** — the per-token Candle bypass the rental script calls
*"the ACTIVE CUDA-graph decode path"* — before falling through to
`forward_inputs`. `try_dedicated_decode` (`:649-670`) needs **three** things,
and every one of them is a paged-attention artifact:

```rust
let paged_meta   = model_inputs.paged_attn_meta.as_ref()?;   // :659
let cache_engine = metadata.cache_engine.as_ref()?;          // :663
let cache_config = metadata.cache_config.as_ref()?;          // :664
```

⇒ **With the flag `false`, V4 bails at `:659` before the cache checks. The
DedicatedDecodePath has never run on V4 at all.** Turning the flag on is exactly
what would make it reachable — so the brief's instinct that this flag gates a
graph path is *right*, just about a different path than the one it named.

**And that is the danger, not the win.** `DecodeConfig`
(`arc-cuda-graph/src/weights.rs:111-124`) describes a **dense Llama-shaped**
model: `intermediate_size` (one MLP, no experts), a **fused QKV** buffer
(`decode_forward.rs`: `qkv: [batch, q_dim + k_dim + v_dim]`, with `q`/`k`/`v`
aliased into it), plain RoPE, no sliding window, no attention sinks, no second
key set. V4 is none of those: one fused `wkv` producing a single 512-wide MQA
head, grouped `wo_a`/`wo_b`, a 256-expert MoE, mHC 4-D residual, and
sliding-window + sink + CSA/HCA attention. The construction site
(`normal.rs:1250-1256`) even infers `intermediate_size` by assuming *"gate_proj
is at index 1 + 4 (5th projection in first layer: q,k,v,o,gate)"* — V4's first
layer is `wq_a, wq_b, wkv, wo_a, wo_b`, so index 5 is not a gate projection.

Two things keep it from firing today, and both are accidents:
1. `_decode_weights` extraction fails on V4 anyway — session 15 logged
   `Decode path extraction failed: tensor_device_ptr requires CUDA tensor`,
   leaving `dedicated_decode: None`.
2. `ARC_NO_DEDICATED_DECODE=1`, which an 80 GB card needs regardless
   (`normal.rs:1228-1233`).

**⇒ If anyone ever flips this flag on for real AND the extraction starts
succeeding, V4 decode silently routes into a dense-transformer kernel stack and
produces garbage with no error.** That is a far better reason for caution than
either objection on record. The A/B below sets `ARC_NO_DEDICATED_DECODE=1` in
**both** legs precisely so this cannot confound the paging measurement.

---

## 6. What shipped

`mistralrs-core/src/pipeline/loaders/normal_loaders.rs` only:
* `v4_paged_attn_optin_from(Option<&str>) -> bool` — pure, `var == Some("1")`.
* `v4_paged_attn_optin()` — `OnceLock` wrapper + `once_log_info` on enable.
* `supports_paged_attention` returns `Ok(true)` iff the opt-in is on.
* Two tests (§7). Rationale block extended with §5's cost list.

**Default unchanged.** Strict `"1"` is deliberate: `ARC_V4_FP8_KV` shipped as
`!(v == "0")`, so *unset* meant *on*, and every V4 forward died for a day
(wave49-BZ / PR #76). An experiment flag must not be able to turn itself on.

---

## 7. Tests, and the mutation that fails each (D12)

* `v4_paged_attn_optin_requires_exactly_one` — `Some("1")` opts in; `None`,
  `Some("")`, `Some("0")`, `Some("true")`, `Some("yes")`, `Some(" 1")` do not.
  **Mutation:** widen to `var.is_some_and(|v| v != "0")` ⇒ the `None`, `""`,
  `"true"`, `"yes"`, `" 1"` rows fail. This is the wave43-BU bug in test form.
* `v4_supports_paged_attention_defaults_to_false` — drives the real
  `DeepSeekV4Loader::supports_paged_attention` in a process where the variable
  is unset. **Mutation:** change the tail to `Ok(true)` ⇒ fails.

`cargo test -p mistralrs-core --lib normal_loaders`: 15 passed, 0 failed.

**Honest limit.** These test the gate, not the arm. The arm is CUDA-only
(`cache_write_and_gather` is `#[cfg(all(feature = "cuda", target_family = "unix"))]`),
so only §8 can speak to it. BACKLOG "wired but dead" entry nine is not cleared by
the tests; it is addressed by §8.

---

## 7b. PRE-REGISTERED PREDICTION — written before the box finished

Recorded here, timestamped by the commit, so §8 can confirm or refute it rather
than rationalise whatever came back.

**Prediction: prompt 0 matches; prompts 1 and 2 diverge in the ON run.**

Mechanism: V4 keeps the compressor-input `xs` history in extra `NormalCache`
slots at index `num_hidden_layers + j` (`deepseek4.rs:3401-3405`), and the doc
on that field states the dependency outright — *"The history rides the same
per-sequence clone_in/clone_out machinery as the KV entries."* Under
`SchedulerOutput::PagedAttention` (`engine/mod.rs:556+`) there is **no
`CacheInstruction::In`/`Out` and no NormalCache reset** — I read the whole arm.
The paged KV itself is safe (per-sequence block tables), but the `xs` slots are
one process-wide buffer that is never cleared between requests. Request *n* > 0
therefore starts with request *n−1*'s history still in the slot.

**What would refute it:** all three prompts identical in the ON run. That would
mean something else clears those slots, and §5e / wave29-BC §4b are both wrong
about the mechanism.

**What would confirm it:** prompt 0 identical, prompts 1–2 not. Note this is a
*cross-request* leak at b=1 — strictly weaker than, and independent of, the
`bs > 1` varlen-pack corruption `v4_paged_dispatch_precheck` already refuses.

---

## 8. HARDWARE

**Box:** Runcrate `arc-w53-paged`, **A100 80GB PCIe**, 28 cores, 700 GB
`/ephemeral`, **$1.49/hr**. Self-destruct armed (`sleep 10800; shutdown -h now`,
`ARMED` confirmed) before any build, per the standing rule.

**Method:** one unattended script (`w53_probe.sh`, in this directory), polled by
one on-box log read every 5 min. Both runs: `--max-seqs 1`, `--prefix-cache-n 0`,
`--max-seq-len 4096`, `ARC_NO_DEDICATED_DECODE=1` (the extraction OOMs at 80 GB
with a ~74 GB artifact — `normal.rs:1228-1233`), `temperature 0.0` (greedy /
argmax), same 3 prompts, 48 tokens each.
* **OFF** — `--paged-attn off`, `ARC_V4_PAGED_ATTN` unset.
* **ON**  — `--paged-attn on --pa-cache-type auto --pa-memory-mb 2048`,
  `ARC_V4_PAGED_ATTN=1`.

`--pa-cache-type auto` is pinned deliberately: the CLI default is TurboQuant
(K4/V3, 3.5-bit), which would quantise the paged KV and break token identity for
a reason that has nothing to do with paging. (V4's `head_dim=512` would in fact
auto-fall-back to `Auto` anyway — `paged_attention/mod.rs:306-308` — but pinning
it removes the question.)

**Known margin risk, stated before the result.** Session 15 measured qtip2b
**resident at 78,801 MiB** after load on a 141 GB H200. The A100 has 81,920 MiB
total, so the OFF run has ≈2.6 GB of headroom and the ON run spends up to
another 2 GB of that on paged KV blocks. If the ON run OOMs where the OFF run
did not, that is a **box-size** result, not a verdict on the arm, and must be
re-run at `--pa-memory-mb 512` before it means anything. (b=1 needs far less:
2·1·512·2 B · 43 layers = **88 KB/token**, so 4096 tokens of context is 360 MB.)

**Pre-existing instability to not mis-attribute.** Session 15 saw the *plain,
non-paged* qtip2b server panic twice mid-run with
`kv_cache/mod.rs:498:54: shape mismatch on dim 1, 576 <> 64`, rebooting the
engine and turning 34 GSM8K items into `finish_reason: "error"`. A crash of that
signature in **either** leg is the known bug, not this flag.

<!-- RESULT -->


---

## 9. Surfaced, not shipped

> **Noticed:** the missing `runner.capture(&forward_fn)` call is the *actual*
> single thing standing between the workspace and GPU-autonomous decode, on
> every model, not just V4 — `prime_for_step` is wired, the runner allocates,
> the closure exists, and nothing ever captures. It is a one-call-site question,
> not a paging question. Worth a separate change?

> **Noticed:** `normal.rs:1772-1779`'s doc comment still says the
> `PagedAttentionMeta` → `prime_for_step` bridge does not exist. It does
> (`:1918`, `:1989`). A stale comment on the exact function whose gating is
> under investigation cost this wave real time. Worth a separate change?

> **Noticed:** `deepseek4.rs:1475` refers to `append_v_marker`, which does not
> exist (the function is `v4_v_marker`). Worth a separate change?
