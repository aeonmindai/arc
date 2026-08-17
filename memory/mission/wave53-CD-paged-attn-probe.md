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
(`arc-cuda-graph/src/autonomous.rs:213,297`) have **no caller in the entire
workspace** — `grep -rn "capture_via_decode_forward\|\.capture("` over
`mistralrs-core/`, `arc-engine/`, `arc-cuda-graph/src/bin/` returns exactly one
hit, and it is the comment at `pipeline/mod.rs:609` saying a pipeline *"must
call `runner.capture(&forward_fn)`"*. Nothing does.

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

## 8. HARDWARE

<!-- filled in below -->

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
