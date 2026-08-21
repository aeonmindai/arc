# wave65-CR — LRU eviction cannot see which addresses a live graph baked

Parent system: ArcInfer / ArcGraph, in the forked candle CUDA caching allocator
(`aeonmindai/candle`, `candle-core/src/cuda_backend/device.rs`).

Status: **surfaced, not fixed.** Two hazards, both silent, both filed here
because they have no owner otherwise.

Found while merging the two diverged candle lines — master's `89ab14ef`
("eviction must not run inside the capture window") and the capture branch's
`9211966e` ("a captured device->host copy writes into a FREED host Vec") — into
`9586979d` on `aeonmindai/candle@arcgraph/bounded-alloc-plus-prewarm`.

**No hardware leg was taken. Nothing here is measured on a GPU.** Every claim
below is derived from source. arc line numbers are read at `bb30559`; candle
claims are read at `9586979d`. Where a number came from someone else's recorded
measurement it is attributed inline and marked as theirs.

> **candle is cited by symbol, never by line — deliberately.** candle is not
> vendored in this repo, so a `device.rs:N` citation either fails to resolve or,
> worse, resolves by component-suffix match to `arc-profiler/src/device.rs`
> (268 lines) and lands on unrelated code. Both are `SYMBOL-ROT` under
> `mistralrs-core/tests/doc_citations.rs`, which has no waiver path. Every
> candle reference below therefore names the file and the symbol and lets the
> reader search at the stated rev. **Do not "fix" these by adding line
> numbers.**

---

## Hazard 1 — eviction frees buffers an instantiated graph writes into

### The mechanism

`AllocCache` gained a capacity and LRU eviction in candle `b2a4dd80` (default
cap 1 GiB, `DEFAULT_CAPACITY_BYTES`). Eviction sorts `free` by
`SizeClass::tick` and hands the oldest buffers back to the driver
(`evict_if_over_capacity`).

The cache stores `(bytes, ptr)` and nothing else. In particular it does **not**
record that an address is baked into a `CUgraphExec`'s memory nodes. After a
successful capture, the buffers the captured forward used are dropped by the
eager tensors that held them and land straight back in `free`, indistinguishable
from any other cached buffer. Evicting one returns to the driver memory that the
next `cuGraphLaunch` will write into.

### Why this is the steady state, not a corner case

The tempting reading is "the graph's buffers are hot, so LRU protects them."
It does not, and the reason is structural:

* **Replay never touches the allocator.** A `cuGraphLaunch` executes baked
  addresses. It calls no `cache_take`, so it refreshes no `tick`. The graph's
  buffers age monotonically from the moment the capture ends.
* **Eager traffic continues.** `default_verify_replays()` is 3
  (`arc-cuda-graph/src/graph.rs:278`), so eager forwards run alongside replays
  and keep stamping fresh ticks on every *other* size.
* **Nothing drains in between.** `drain_alloc_cache_and_free` is called only on
  pool-destroying paths — graph teardown (`graph.rs:56`) and cancel
  (`graph.rs:214`). A successful instantiate leaves the graph's buffers resident
  in the cache for the life of the graph.

So the graph's buffers start as the newest entries and become the oldest. The
first eager put that crosses the cap evicts them first. Master's own test docs
record the cap being reached in practice on V4 — *"Measured on V4 at a 1024 MiB
cap, the peak was 1026.1 MiB"* (candle `859c49c8` commit message; **their
measurement, not ours**), so "the cap never bites" is not available as a defence.

### The pre-warm makes it likelier, and multi-`bs` makes it certain

`prewarm_alloc_cache` (candle `1fa534db`) deliberately installs a large,
long-lived set of buffers at exactly the sizes the capture is about to bake.
arc's caller comments size that pool at *"~315 MiB"* (held-out arc commit
`492b7baa7`; **their figure**). Against a 1 GiB cap that is a third of the
budget in one bite.

And captures are keyed by batch size — `end_capture_and_cache(bs, ...)`
(`graph.rs:517`). Pre-warming for `bs=2` while `graph(bs=1)` is instantiated puts
graph(bs=1)'s baked addresses in the eviction pool.

For that reason the merge makes the pre-warm **non-evicting** by construction
(`install_prewarmed`). That closes the pre-warm's own window. It does **not**
close the general one: ordinary decode puts still evict.

### Why the obvious mitigations are not the fix

* **Raise the cap.** Turns a correctness bug into a tuning parameter, and the
  cap exists because the unbounded cache leaked (candle `b2a4dd80`: *"6.04 MiB
  per decoded token with no plateau"* — **their measurement**). Trading one back
  for the other is not progress.
* **Drain after instantiate.** The drain doc is explicit that draining under a
  replayable graph *"trades one use-after-free for another"*
  (`drain_alloc_cache_and_free`).
* **Never evict while a graph is live.** Restores the leak for the entire life of
  the graph, which is the whole run.

### The proposed fix — a pin the allocator can express

LRU cannot represent "this address is baked", so give it a way to say so.

1. A `pinned: HashSet<usize>` (or a `SizeClass::pinned` flag) marking byte sizes
   the capture path has declared replay-critical — naturally, the keys of the
   demand profile at instantiate time.
2. `evict_if_over_capacity` skips pinned sizes entirely, exactly as it already
   skips `deferred`.
3. `CudaGraphRunner` pins on successful instantiate and unpins in the same place
   it already drains (`graph.rs:56`, `:214`) — after `cuGraphExecDestroy`, before
   `cuMemPoolDestroy`. The lifetime is already correct there; only the pin is
   missing.
4. Retention stays bounded because the pinned set is bounded by the demand
   profile, which is bounded by one forward's allocations. If the pinned set
   alone exceeds the cap, that is a **reportable** condition, not a silent one.

This is deliberately not shipped in `9586979d`: it is a new mechanism on both
sides of the candle boundary, and the merge's job was the union of two existing
lines, not a third.

### The probe that settles it

One counter already exists for this, added by the merge:
`AllocCacheStats::evicted_at_demand_size` — eviction victims whose byte size
is in the capture demand profile.

> With capture enabled and a graph instantiated, poll `alloc_cache_stats()` per
> decode step and assert **`evicted_at_demand_size == 0`** for the life of the
> graph.

Non-zero is the fingerprint, observed *before* the crash rather than inferred
from it. The existing per-step logger (`mistralrs-core/src/pipeline/normal.rs:299`)
is where it belongs; it reads `s.misses` / `s.frees()` / `s.hits` today and does
not print this field yet.

Secondary A/B, same run: `capture_miss_count()` after the capture forward with
pre-warm on vs off (expect `>0 → 0`), and `alloc_cache_stats().prewarmed`
against the demand-profile sum.

---

## Hazard 2 — `ARC_NO_DEFERRED_FREE` silently disables the capture instruments

`set_capture_mode` returns early when `ARC_NO_DEFERRED_FREE` is set. That
predates both merged candle lines and was only ever documented as affecting
deferred-free. It does more than that now:

`capturing` never becomes true, therefore

* no capture window ever opens ⇒ `demand` stays empty ⇒ `prewarm_alloc_cache`
  plans nothing and returns `(0, 0)`;
* no miss is ever filed in `capture_misses` ⇒ `capture_miss_count()`
  returns 0 ⇒ the caller's "refuse to instantiate on a miss"
  gate passes **unconditionally**.

**A clean capture and a silently disabled capture are indistinguishable from
their instruments.** A run with this variable set reports the same numbers as a
perfect one.

This is the same class as the instrument failures already logged this session,
and it belongs to neither candle line, so nobody owns it unless it is written
down. `9586979d` adds an in-code warning comment in `set_capture_mode`'s preamble and
changes no behaviour — a runtime print would alter what log scrapers see, which
is a separate call.

> **Probe:** assert `ARC_NO_DEFERRED_FREE` is unset before trusting a zero from
> `capture_miss_count()` or a `(0, 0)` from `prewarm_alloc_cache`.

---

## Interaction with PR #182 and the KV `cat` growth

Three things are in flight at once and it is worth stating exactly how they sit
together, because two of them look independent and are not.

### #182 is not a neighbour of hazard 1 — it is its precondition

**PR #182** ("Bound the caching allocator: +6.04 MiB/token leak -> flat, for
0.15 ms/token", branch `perf/alloc-cache-bounded`, touching `Cargo.toml` and
`mistralrs-core/src/pipeline/normal.rs`) is the **arc-side landing of the same
allocator line hazard 1 is about**. Its candle side is master's `b2a4dd80` →
`89ab14ef` — one of the two lines merged into `9586979d`.

That matters more than a cross-reference. **Before the bound there was no
eviction at all**: the only routes back to the driver were
`set_alloc_cache_enabled(false)` and `drain_alloc_cache_and_free()`, both called
only on pool-destroying paths. Hazard 1 had nothing to fire with. After #182 it
does.

So #182 converts a bounded memory leak into a correctness risk that is live
exactly when a graph is live. That is still the right trade — an unbounded leak
is fatal on a long generation — but it **is** a trade, and it should be on the
record as one rather than discovered later. The pin proposed above is what pays
for it.

### The KV `cat` growth sets hazard 1's *rate*

Tonight's hardware run — **not ours; we took no hardware leg** — reports the
eager path OOMing at 22–30 tokens with ~11 MiB/token of genuine growth, the card
full at 143,151 of 143,771 MiB, traced to `cat_contiguous` (a candle function,
`candle-core/src/tensor_cat.rs`) on an F8E4M3 KV buffer.

Whatever that turns out to be, it bears directly on hazard 1: **retention only
reaches the cap because something presents byte sizes nothing asks for again.** A
buffer whose width tracks KV length is exactly that shape — it is the shape
`b2a4dd80` named when it measured *"a family of ~132 buffers stepping by 8 KiB
per token"* (**their measurement**). More distinct-size traffic per token means
the cap is crossed sooner and the LRU sweep runs more often, and the sweep is
precisely what hazard 1 says can reach a graph's baked buffers.

### This does not settle whether #182 and the KV growth are one bug or two

Another agent owns that question and this document must not pre-empt it. It does
not have to: **hazard 1's mechanism is unchanged under either verdict.**

* **One bug** — the growth *was* the cache retaining the `cat` chain's dead
  sizes, and #182's bound fixes it. Then eviction now fires by design, at the
  rate #182 measured, and hazard 1 fires with it.
* **Two bugs** — the ~11 MiB/token is genuinely live tensors the cache never
  holds. Then the OOM is independent of the allocator, and hazard 1 still fires
  whenever ordinary retention crosses the cap.

Either way the fix for hazard 1 is the same pin, and neither verdict makes it
unnecessary.

### Consequence for sequencing: hazard 1 is currently *masked*

A process that dies at token 22–30 never keeps a `CUgraphExec` alive long enough
for its baked buffers to age to the front of the eviction queue. Replay does not
refresh a tick, but ageing still takes tokens, and there are not enough of them
before the OOM.

So the `evicted_at_demand_size` probe cannot return a meaningful **zero or**
non-zero until the OOM is fixed. **A green probe today would be a green light
from an instrument that never had the chance to go red** — the same failure
shape as hazard 2, arriving by a different road. Fix the OOM first, then run the
probe, and treat any reading taken before that as void.

---

## Cross-reference — two root causes, one observed crash

Do not read either of these as closing the other.

**This item** is a *device-side* use-after-free: the allocator frees device
memory whose address a graph node writes to on replay.

**The `compressed_row_positions` thrash** found independently by the capture-lane
rebase is a *host-side* dangling pointer. `compressed_row_positions`
(`deepseek4.rs:3013`) memoizes into a **single-slot** thread-local
`Option<(t_c, ratio, Tensor)>` (`deepseek4.rs:3009`) keyed on `(t_c, ratio,
device)`. V4's layers strictly alternate ratio — *"Layers 0/1 = standard, even
2..=42 = CSA(4), odd 3..=41 = HCA(128)"* (`deepseek4.rs:6031-6032`, asserted
in-tree at `deepseek4.rs:6033-6041`) — so **every** consecutive compressed-layer
pair evicts the other's entry and the cache misses on all 41 compressed layers
of every forward. Each miss runs the strided `arange` constructor (`deepseek4.rs:3033`) →
`storage_from_cpu_storage` → `clone_htod`,
which under capture records the *host* pointer of a `Vec` that dies with the
expression → SIGSEGV on the first `cuGraphLaunch`.

Two distinct mechanisms, the same observed symptom. Both must hold.

Worth noting for sequencing: `9586979d` carries `arc_capture_retain_host`
(candle `864cd047`), which retains the host source of **every** capture-time H2D
copy including the `clone_htod` path — so it mitigates the second mechanism's
*crash* generically at the candle level, without fixing the cache thrash that
causes 41 redundant strided-`arange` builds per forward. That thrash is a real cost item
in its own right and is adjacent to wave65-CQ's item 2, which proposes memoizing
the `cos`/`sin` gathers on the same `(t_c, ratio, device)` key — **and would
inherit the same single-slot alternation defect if it copies the existing
pattern verbatim.** Whoever lands CQ item 2 should key on the pair, or use a
small map, not one slot.

---

## Related

* **PR #205** — the capture lane. Five commits were held out of it because they
  call candle APIs that exist only on the branch line: `492b7baa7`, `1c371b2cc`,
  `f7c65cb72`, `8968c1fa3`, `4475f9e71`. Against `9586979d` the first, second and
  fourth resolve; **`f7c65cb72` and `4475f9e71` must never be cherry-picked** —
  they pin candle to `5bdddcf` / `9211966`, which would revert master's
  eviction-inside-capture fix (`89ab14ef`). They collapse into one rev bump plus
  a lock regen.
* **PR #207 / wave65-CQ** — the V4 compressed-KV re-RoPE recompute, on the same
  `compressed_row_positions` code path. See the caveat above before landing its
  item 2.
* **`CAPTURE_LANE`** — op count is retired as a lever. Neither hazard here is an
  op-count item; both are correctness.
* Filed here rather than as a GitHub issue because `aeonmindai/arc` has issues
  disabled; `memory/mission/` is the tracker this project uses.

---

## What would change this document

Either hazard becomes a **defect with a number** the moment someone runs the
probe. Until then both are derivations from source, and the honest status is
"reachable by construction, unobserved on hardware."
