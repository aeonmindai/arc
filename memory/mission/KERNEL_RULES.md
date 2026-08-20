---
name: kernel-rules
description: "D16 dual-arch (Hopper AND Blackwell) mandatory for every kernel; D17 the moat is our byte formats, so every kernel touching them must be ours"
metadata: 
  node_type: memory
  type: project
  originSessionId: 4d267202-c569-48de-99ab-c497142fac75
  modified: 2026-08-19T02:22:53.631Z
---

# 🔴 SESSION-8 CLOSE — SEVEN RULES, EACH PAID FOR (2026-08-19)

*Consolidated from the census. Companions on branch `docs/census-session8`:
`memory/mission/CENSUS_SESSION8.md`, `memory/mission/LADDER_POST_CENSUS.md`.*

## 1. A NUMBER YOU DID NOT COMPILE IS AN ESTIMATE

**Hand-counting C++ undercounts SASS by 2.05× on this kernel family.** A
subagent's inst/weight answer was **retracted for exactly this** — it read the
source, counted the operations it could see, and reported the result as a
measurement. nvcc's output is not a transcription of your source.

**The method that works, and it costs no GPU:** `nvcc -cubin`, then difference
**two unroll depths** so prologue and epilogue cancel and you are left with the
inner loop alone. Validated to **1.3% against ncu**. Three points, not two —
the two-point rows in the compiled ladder are the only ones whose linearity is
unverified, and they are flagged as such.

⇒ **Label every uncompiled count "estimate", in the same sentence as the value.**

## 2. SCORE inst/weight AGAINST ITS **OWN** bpw BUDGET

`bits/weight = K/V` (`mistralrs-quant/src/qtip/mod.rs:376-381`). **The budget
scales *with* bpw.** So a 1-bpw geometry has roughly half the budget of a 2-bpw
one, and "1.38 is at the 1.41 budget" was a **1-bpw kernel scored against a
2-bpw budget** — 2.05× over, not at. That single error produced a retracted
claim that the full ceiling was one re-bake away.

⇒ **Before comparing two geometries, print `K/V` for both.** If they differ, you
are not allowed a shared budget line. The compiled ladder is safe *because* all
three of its geometries are 2 bpw.

⚠️ **AND A LADDER THAT HOLDS bpw FIXED HAS NOT PICKED A WINNER — IT HAS ONLY
PRICED ONE AXIS.** Holding all three geometries at 2 bpw made the instruction
comparison valid and simultaneously hid the fact that **the cheapest of the three
is quality-closed at that bit rate** (K8/V4/L12: −0.00698 `w_cos` vs a ±0.0008
band). The winning rung turned out to be **off the ladder entirely**, at 2.25 bpw.
⇒ **Price quality on the same sweep, or state in the same sentence that you have
not.** `memory/mission/FRONTIER_BITS_FOR_DECODE.md`.

## 3. A DEPRECATED CLI THAT PRINTS HELP AND EXITS **0** IS SILENT SUCCESS IN A NEW COSTUME

`huggingface-cli` → `hf`. The old binary still exists, still prints, still exits
**zero** — so every wrapper that checked `$?` saw a pass and every log looked
normal. **This is D18 wearing vendor clothes**: the absence of the work was
indistinguishable from the work.

⇒ **Gate on an artifact, never on an exit code** — the file appeared, the byte
count changed, the counter incremented. And **pin tool names**: a rename is a
silent-success generator for as long as the deprecation window lasts.

## 4. "MERGE FAILED" ≠ "MERGE CONFLICTED"

A **missing remote-tracking ref on a shallow clone** fails with the same shape
as a genuine content conflict. Main mislabelled one as the other and **nearly
dropped a required correctness fix out of a measurement** — the branch was fine;
the clone could not see it.

⇒ **Read the actual error, not the exit status.** `git fetch --unshallow` (or
`--depth`) before concluding anything about a conflict, and state which of the
two you observed. *Related and recurring: unpushed ≠ unlanded — check
`git merge-base --is-ancestor`, not the worktree.*

## 5. A FLEET SSH CERT WITH A 6.5-MINUTE TTL CANNOT BE USED BY AN AGENT

The documented access method was **structurally unusable**: the cert expires
inside the time it takes an agent to do anything with it, so every attempt fails
in a way that reads as a permissions problem.

⇒ **Install a durable pubkey once per instance**, then use it. And more
generally: **when a documented procedure fails identically for every caller,
suspect the procedure's own time constants before suspecting the callers.**

## 6. CANDLE: A NON-CONTIGUOUS OPERAND DOES A **HOST→DEVICE UPLOAD PER OP**

`candle-core/src/cuda_backend/mod.rs:56-61`. Worse, **`Shape::is_contiguous`
skips size-1 dims** (`shape.rs:159`) — so at **b=1** a slice-based fold looks
free, and at **B>1** the same code adds an upload on every op.

⇒ **MEASURE FOLDS AT B>1 OR NOT AT ALL.** A b=1 A/B on a slice-based
optimisation cannot see its own cost. This is the decode-side twin of the
already-recorded *"removing the 43 per-layer syncs makes it SLOWER"* result:
**a change that looks free at width 1 is the single most reliable way to ship a
regression to the fleet.**

## 7. CLIPPY'S LITERAL SUGGESTION CAN INVERT NaN SEMANTICS

`!(x > k)` → `x <= k` is **not** a refactor. `NaN > k` is false, so `!(x > k)` is
**true** for NaN — the original **rejects**; the rewrite **accepts**.

⇒ **Spell the NaN arm out** rather than taking the lint. A lint that changes
which inputs pass a guard is a behaviour change with a green checkmark on it.

## 8. A GUARD MUST BE SHOWN TO **FAIL**

**Three guards this session passed on unfixed code — including one written *for*
this rule.** That is the whole of D33 restated by counterexample: the guard you
just wrote is the least-tested code you have, and writing it *about* mutation
testing does not exempt it from mutation testing.

⇒ **Feed it the failing case and confirm it says so, every time, no exceptions
for the guard that enforces this.**

---

# D16 — Every kernel targets Hopper AND Blackwell

Jish, 2026-08-17: *"when you're writing stuff make sure you're writing for both hopper and blackwell"*

**SM90 (H100/H200) and SM100/SM103 (B200/B300) are both first-class targets.** Not
"sm90 now, retarget later." A kernel that dies at the next hardware generation is a patch,
not infrastructure — this is the same reasoning that chose CuTeDSL over FlashMLA-C++
(hand-written CUDA is arch-locked; CuTeDSL retargets because tile shapes are parameters).

## The state this rule corrects (measured 2026-08-17)

`mistralrs-quant/build.rs:52,117,134` already emits **SM89 / SM90 / SM100**, so Blackwell is
in the build matrix. But the kernels themselves are written to an **Ampere baseline**:

```
qtip_grouped_gemm.cu:8        sm_80
qtip_grouped_gemm.cu:327,444  __CUDA_ARCH__ >= 800
qtip2b_beam.cu:39,94          sm_80
qtip_bitshift_tune2.cu        sm_80, __CUDA_ARCH__ >= 800
```

**We are arch-portable by being arch-naive.** `__CUDA_ARCH__ >= 800` means "Ampere or
newer", so the keystone grouped-GEMM *runs* on H200 and B200 while exploiting neither —
no wgmma on Hopper, no tcgen05/tmem on Blackwell. "Runs on Blackwell" is not "written for
Blackwell."

Exception that proves the rule: `arc-cuda-graph/src/cuda/sampling_kernel.cu:25` does list
`sm_89 / sm_90 / sm_100`.

## What compliance means

- Arch-specialised paths where the hardware differs (wgmma/TMA on SM90; tcgen05/TMEM and
  2-CTA MMA on SM100), with a correct fallback — not a single `>= 800` path everywhere.
- Both arches in the build matrix AND in the dispatch.
- State in the PR which arches were **measured** vs merely compiled. We rent SM90 today, so
  SM100 will often be compile-verified only — say so, never imply it was benchmarked.

# D17 — The moat is our byte formats; kernels that touch them must be ours

Jish, 2026-08-17: *"how we pack our weights is our moat and if flashinfer has to be added it
needs to support how we pack."*

The moat is not one kernel. It is **"we store things in formats nobody else reads"**, and it
surfaces in two places:

| We pack | Kernel that touches it | Adoptable from upstream? |
|---|---|---|
| **Weights** — trellis/QTIP | the **GEMM** | **NO.** FlashInfer's grouped GEMM is stock CUTLASS: closed dtype enum, no weight-decode hook (`gemm/group_gemm.cuh:63,67`), `ElementA = ElementB` (`group_gemm_sm90.cuh:65-66`). Trellis is a *state machine*, not a dtype — decoding symbol N requires walking 1..N-1. |
| **KV cache** — TurboQuant (WHT) | the **attention kernel** | **NO.** Accepted KV formats are a closed set; even plain FP8 KV is unsupported at our head_dim on SM90. |

**The rule:** any kernel that reads or writes a format we invented has to be ours. Adopting
an upstream attention kernel silently costs KV compression — either we stop compressing, or
we decompress before every call, which destroys the benefit. Fewer bytes per token is what
lets more sequences fit, and that is throughput.

This is fleet-wide, not V4-specific — we want compressed KV on **every** model.

**Consequence:** upstream kernels are only adoptable on paths touching neither packed weights
nor compressed KV. Once TurboQuant is default-on (Jish's instruction), that surface is close
to empty. Any FlashInfer evaluation must be scored on *"can it read our bytes"*, not
*"can it express our attention shape"*.

Related: [[gpu-access-rule]] (D14/D15), [[doctrine]] (D1–D13).

---

# D36 — A NAIVE A-B-A-B DELTA ON A DRIFTING BOX IS **BIASED**, NOT MERELY NOISY

**Worked out 2026-08-18 by the trellis chain, before its own arms landed.** This invalidates the
*magnitude* of every naive interleaved A/B taken on `arc-v4-stack`.

In A-B-A-B-A-B at equal spacing, A occupies slots 0,2,4 and B occupies 1,3,5. So
**`mean(B) − mean(A)` carries exactly one slot of drift as bias.** Least-squares on this box's own
four-arm history — **66.99, 67.71, 68.05, 70.30 ms/T, monotone** (a genuine time trend, not
scatter) — gives a slope of **+0.91 ms/T per slot.**

**Against a true effect of −1.26 ms/T, the naive difference reads −0.35** — about **a quarter of
the truth, and shaped like a weak win rather than a measurement artefact.** That is exactly the
form nobody questions.

### The estimator to use instead
Fit **`y = a + b·slot + c·arm`** by least squares over all points and **report `c`**, with the
fitted drift slope `b` stated separately and the residuals shown rather than linearity assumed.

### Compute power BEFORE the run, and hold the design
With 3 pairs: residual df = 3, `Var(c) = 0.729·s²`; at the ≈0.38 ms/T residual scatter this box
showed, **SE(c) ≈ 0.33 and the 95% interval is ≈ ±1.0 ms/T against a predicted −1.26.** Marginal by
construction. **Clean separation needs ~6–8 pairs — decided in advance or not at all.**
⇒ **If the interval spans zero, write "this test could not resolve an X% effect" and STOP.** No
extra arms, no longer runs, no re-runs until it separates. **Choosing the design after seeing which
way the numbers fell is how a real 14% kernel win becomes a fabricated 3% token win.**

### What this does and does not overturn
- **Counts are immune** — allocations/token, launches/token, host-calls/step, D2H/step. Unaffected.
- **Kernel-level timings with huge SNR are safe** — trellis µs/call: 14% effect vs 0.16% drift ≈ 87:1.
- **⚠️ The `GpuApprox` delta is RETRO-INVALIDATED — stop citing it as a magnitude.** Naive read
  +1.485 ms/T; drift-corrected **c = +0.572 ± 0.855 ms/T on ONE residual dof** ⇒ **not
  distinguishable from zero** at four arms. (Four arms fit three parameters, leaving 1 dof —
  **six arms, A-B-A-B-A-B, giving 3 dof, is the standard shape from now on.**)
  **The CONCLUSION survives untouched**: *"GpuApprox is not the fix"* rests on the independent
  engagement evidence — `kv_fp8_quant` **73.49 → 134.18 µs/call, +83%** — not on that delta.
  Quote the engagement number, never the delta.

### 🔴 THE NIGHT'S SHARPEST DEMONSTRATION — a 2.43× "win" that was a compile finishing
Within **one interleaved pair, two minutes apart**, the trellis e2e read:
`A1 (baseline) 169.49 ms/T` → `B1 (optimised) 69.78 ms/T`.
B1 was within 4% of the 66.99 reference — **the `cicc`/`ptxas` swarm drained between the two arms.**
**≈100 ms/T of pure environment against a predicted −1.26 ms/T effect: noise ~80× signal.**

**Scanned casually, A1→B1 reads as a 2.4× win for the kernel. It is a compile finishing.** It would
have been the single most quotable number of the night.

**And it killed the pre-registered estimator, which the chain reported rather than fitting anyway:**
`y = a + b·slot + c·arm` assumes a smooth monotone trend; **a 2.43× step discontinuity is a regime
change, not drift**, so the linear detrend is inapplicable and **no `c` was reported.**
⇒ **A detrending model is itself a claim that must survive the residuals.** Show them; don't assume
linearity held.

### 🔑 INSTRUMENT THE ENVIRONMENT, NOT JUST THE RESULT

The mirror trap arrived one arm later: `A2 = 68.58` against `B1 = 69.78` would nominally show the
**optimised arm 1.2 ms/T SLOWER** — the same recovering host, the same meaninglessness, opposite
sign. **A contaminated window manufactures wins and regressions with equal ease**, so "the number
came out in the direction I expected" is worth nothing.

**The harness defect that made it unfixable:** `measure2.sh` logged neighbour state and GPU util
per round (which is how an earlier outlier was caught); **`e2e.sh` logged nothing per arm.** So the
contamination was visible in the numbers but there was **no objective, pre-specified,
outcome-independent criterion to exclude A1.** Dropping it *because it looks wrong* is post-hoc
selection on the outcome — **worse than reporting ENVFAIL.**

⇒ **Every timed harness must record, per arm: builder/compiler process count, load average, GPU
utilisation and power, and lock state.** With those logged, exclusion becomes a **rule stated in
advance** ("discard any arm with builders > 0") instead of a judgement made after seeing the
result. Without them, the only honest move left is to discard the entire run.

**This is the same lesson as the lock, one level up:** the thing that corrupts a measurement is
usually outside the thing being measured.

**Corollary — pre-register.** State the predicted effect size, the drift, and the estimator *before*
the run. That is the difference between a measurement and a search.

---

# D35b — NEVER SHARE `CARGO_TARGET_DIR` BETWEEN CHAINS

`/root/arc-wt/target` (13 G) was written by **`build.sh`, `build2.sh`, `build3.sh`, `devrun.sh`,
`unit.sh`, `pg_env.sh`, `graphstream-build.sh` and `arena/build.sh`** — at least four chains.
At 10:16:49Z a neighbour's build overwrote the arena chain's binary and **one exclusive bench slot
measured a binary containing none of its code.**

**Caught only by that chain's own "0 fingerprints" assert.** Without it, a clean null result would
have been reported and believed.

⇒ **Private `CARGO_TARGET_DIR` per chain, a frozen copy of the measured binary with its md5, and a
symbol/fingerprint check gating every run.** *Verify the binary carries your own symbols before
trusting any number.* Worktree isolation of *source* is not isolation of *output*.

---

# D35c — ⚠️ D35 AS FIRST WRITTEN WAS INCOMPLETE AND CORRUPTED A MEASUREMENT

**Main pushed "lock covers GPU work only, never compilation" fleet-wide. It fixed lock throughput
— the card went 0% → 99% — and it converted a QUEUEING problem into a CONTAMINATION problem.**

Builds now ran *concurrently with* timed runs. Result, 2026-08-18:

> The trellis e2e's **A1 arm — the UNMODIFIED kernel — returned 169.49 ms/T where its
> byte-identical reference is 66.99. 2.53× off.** A clean, well-formed, completely wrong number
> with no warning attached anywhere.

**Cause: host CPU starvation, not GPU contention.** 12 concurrent `cicc`/`ptxas`/`cc1plus` at
87–105% CPU from another chain's build (18 processes), **load 21.87 on 24 cores**. The card at that
moment: **21% util, 128 W of 700, memory-util 0%.** The benchmark process was pinned at 169% CPU
wanting more host.

**Why it is catastrophic here specifically: V4 B=1 decode is 49% host-bound** (32.64 ms of every
66.68 ms step is the GPU waiting on the host). **A half-host-bound benchmark against a 12-way
compile is the worst possible pairing.** On this fleet, *exclusivity must cover HOST CPU, not just
GPU allocation* — otherwise every B=1 number is hostage to whoever is compiling.

> **"The useful thing now is a quiet host, not just a free GPU."**

## THE PROTOCOL — two locks, always this order, deadlock-free

```sh
# BUILD (cargo / nvcc / cicc / ptxas / cc1plus) — SHARED: builds still parallelise with each other
exec 9>/root/locks/host.lock;  flock -s -w 7200 9
… cargo build / nvcc …

# BENCHMARK — needs a quiet host AND an exclusive GPU
exec 8>/root/locks/bench.lock; flock -x -w 7200 8   # serialise vs other benchmarks
exec 9>/root/locks/host.lock;  flock -x -w 7200 9   # drain and exclude ALL builders
… model load, warm-up, timed run …
```
Builders take **one** lock in shared mode ⇒ never block each other, never deadlock. A benchmark's
exclusive request drains builds and blocks new ones for its duration.

**Handling: contaminated windows are ENVFAIL (exit 2), no arm difference reported. Do NOT re-run,
extend, or add arms to escape a contaminated window** — that is choosing the design after seeing
the numbers.

### 🔑 VERIFIED QUIET BEATS NOMINALLY LOCKED — the protocol is not the condition

**A lock only quiets the chains that have adopted it.** Measured immediately after the two-lock
rollout: a chain acquired `host.lock` **exclusively and uncontended — with 12 compilers still
running**, because nobody else took it yet. **Holding the lock proved nothing.**

This is the same fault the arena chain found on the GPU side: **a rogue neighbour held 76 GB
*without* the lock, so holding the lock is not proof the card is free.** Two independent
discoveries of one principle:

> **Assert the PHYSICAL condition, never the protocol.** Count `cicc|ptxas|cc1plus|cc1|nvcc|rustc`
> and read `loadavg` directly; count GPU compute-apps directly. **Wait for actual quiet** (up to a
> stated budget), and **exit 2 if it never comes — refuse to emit a number rather than publish a
> contaminated one.** Never retry until the host happens to be quiet: re-running until conditions
> flatter you is how a contaminated arm becomes a clean-looking number.

Correct memory-guard shape, also from that chain: on insufficient VRAM, **release BOTH locks, wait,
re-queue** — never die, never squat on the lock while waiting.

### ⚠️ `pgrep -c` PRINTS `0` **AND** EXITS 1
So the natural `$(pgrep -c X || echo 0)` yields **`"0 0"`** and silently breaks the arithmetic — the
builder counter would have miscounted without ever erroring. Caught only because the chain **fed
the guard its failing case first** (D33). Proved both directions afterwards: **0** for a name
nothing is running, **8** for a deliberately-started `sleep`, **12** for the real compiler set.

### ⚠️ DISPOSAL — a stale unguarded run may still land
An earlier, **unguarded** sampler cost-curve run remains queued and writes to **`curve.log`**. If it
lands it may print a number taken beside active builds. **`curve.log` must NOT be quoted. Only
`curve2.log` (guarded) counts.** When a harness is hardened mid-flight, the old run does not
disappear — **name the dead output file explicitly or someone will quote it.**

**What saved this:** the chain had **designated the kernel-level micro-A/B primary IN ADVANCE** — a
200-iteration `cudaEvent` loop over a 12.58 MB working set with no host involvement per iteration,
which host starvation structurally cannot touch. **77.31 → 66.19 µs/call, −14.4%, drift 0.16%,
±0.05 µs over four interleaved rounds.** Had the e2e been primary, tonight produces either a
fabricated win or a fabricated regression. **Pre-registration is what made a wrecked run merely
wasteful instead of misleading.**

---

# D35 — THE BENCH LOCK COVERS GPU WORK ONLY. NOTHING ELSE.

**Measured 2026-08-18, session 8, with seven chains on one H200.** The benchmark queue read
**13–17 deep** and looked like the throughput limit. It wasn't. The card at that moment:

> `0 %, 0 MiB, 78.28 W` — **completely idle, with the exclusive lock held and waiters queued.**

Chains were holding `/root/locks/bench.lock` across **CPU work**. After the rule went out the card
read `99 %, 529 MiB, 127.07 W`. **The bottleneck was lock scope, not GPU capacity — and I had
already offered Jish a second $4.92/hr box to "fix" it.** Checking the *card* instead of the
*queue depth* is what found it.

**The rule.** The lock covers **anything that allocates GPU memory or times anything** — model
load, warm-up, the timed run. It covers **nothing else**: not compilation, not patching, not
script setup, not log parsing, not nsys report generation. **Acquire immediately before, release
immediately after.**

Two real offenders, both caught:
| what | cost |
|---|---|
| `flock -w 3600 … -c nvidia-smi --query-compute-apps=…` | **14 min queued** for a 50 ms status read needing no lock |
| `cuobjdump -elf` ×2 inside the locked section, for an engagement proof | **11.15 s per 294 MB binary ≈ 22 s of exclusive lock** for a static ELF read that never touches the GPU |

The second was found by the chain **auditing its own scripts when asked**, and fixed by
precomputing the symbol sets outside the lock and reducing the locked check to a `cmp`. It then
**re-proved the engagement check still had teeth**: comparing the baseline symbol set against
*itself* → `SAME -> would abort as vacuous`; the real comparison → `DIFFER`
(`...Li8ELi2ELb0E` 5 template params vs `...Li8ELi2ELb0ELb0ELi4E` 6). **Moving a check must be
followed by re-proving the check can fail** — D33 applied to a refactor.

## 🔴 THE 50-MINUTE SELF-DEADLOCK — and the three things that all look identical from outside

`/root/arc-wt/nsys_ab.sh` (pid 164153) **held `bench.lock` on fd 9, then a second copy of itself
(214047) was launched, whose `flock -w 7200 9` (214048) waited for the lock its own parent held.**
Stuck **50 minutes**, **13 chains queued**, card at **0%, 0 MiB, 78.79 W**, load 0.80. It would have
held for the full 7200 s timeout.

**Fix applied: kill the deadlocked CHILD branch, not the parent** — least destructive; the parent
released in **~2 s** and the card went to 97% immediately.

**It is NOT a script bug** — audited afterwards, no script in the tree re-invokes itself under
`flock`. It was an **operational double-launch**: starting a second copy of a lock-taking script
while the first still holds it. ⇒ **Write a pidfile before acquiring and refuse to start if a live
instance exists.**

### The three failure modes that are indistinguishable from outside
A deep queue plus an idle card was caused, on the same night, by **three unrelated things**:
1. a lock **held across CPU work** (`nsys stats`, `cuobjdump`, `count.py`)
2. **orphaned waiters** — `flock` children survive a killed runner with `ppid=1` and keep queue
   position for a parent that no longer exists (one chain unknowingly held 3)
3. this **self-deadlock**

**"The queue is deep" is a symptom, never a diagnosis.** Read the card *and* the lock holder's
actual process tree before concluding anything — and note main's own miss here: an offer to spend
$25 on a second box was made while the existing card sat idle.

## ⚠️ `flock` IS NOT FIFO — AN EARLY WAITER CAN STARVE INDEFINITELY

Live queue on the same box: two runs at **36:34** and **36:04** waiting while jobs that arrived at
**25:23** and **22:35** ran. The kernel makes no fairness guarantee, so a steady arrival rate
starves the oldest waiter.

**Therefore:** set generous `-w` budgets (5400–7200 s), and **when overtaken, escalate to main for
a slot — never shorten the measurement to escape the queue, and never re-queue in a tighter loop**
(that worsens it for everyone, including yourself). Main arbitrates; taking a slot unilaterally
hides the problem. **The chain that *asked* to be arbitrated is the reason the idle card was found
at all.**

---

# 🔴 A GUARD THAT CAN BE STEPPED OVER IS NOT A GUARD — detection ≠ enforcement

**2026-08-18, found by the arena chain auditing its own harness.** Its Part 3 guard's **detection
was sound and both limbs had fired on real events**: the GPU limb flagged a live neighbour
(`[note] GPU neighbour: [219393 219840]`), the host limb caught an injected `ptxas`.

**The defect was downstream of detection.** `bench_leg` runs inside `( … )`, so its `envfail` ends
**only the subshell** — the caller then proceeded to `decode_toks` and **printed a `RESULT` for an
arm the guard had already condemned.**

> **A guard that detects contamination but does not stop the pipeline produces a clean-looking
> number with a warning buried above it.** Verify the guard's *verdict propagates*, not merely that
> it fires. Subshells, pipes, `&&` chains and trap-less scripts all swallow a non-zero exit.

Fix: check the flag files **in the caller**, where the leg deliberately leaves them.
**And it was staged as `all.sh.next`, not applied live** — the byte-offset splice rule already
propagating between chains.

**Consequence stated in advance, which is the right instinct:** any Part 3 tok/s from the in-flight
run is **not automatically trustworthy** and each arm will be audited against its per-leg
`[host] peak_load1=… peak_builders=…` line, then **discarded rather than re-run** if contaminated.
**Parts 1–2 are unaffected — allocations/token, H2D/token, arena footprint and the bit-identical
verdict are COUNTS, immune to a busy host and a busy card. That is why they run first.**

---

## 📐 MEASUREMENT DISCIPLINE — index of what session 8 learned (all below, in order)
1. **D33** — feed a check its failing case; a single number cannot audit itself
2. **Walk the quiet branch too** — a guard can break on the all-clear path (`pgrep -c` = `0` + exit 1)
3. **Assert every resource you certify** — quiet host ≠ idle GPU
4. **Detection ≠ enforcement** ← *above*
5. **Verify the test's premise numerically** — the dtype gets a vote (BF16 ULP)
6. **Measure the artifact you ship**, not its components — `hybrid(A,B)` ≠ A and B
7. **D36** — naive A-B-A-B is *biased*; fit `y = a + b·slot + c·arm`, six arms
8. **Instrument the environment per arm** — so exclusion is a rule, never post-hoc selection
9. **D35b** — never share `CARGO_TARGET_DIR`; gate every run on a symbol check
10. **D35c** — the bench lock must cover host CPU; **verified quiet beats nominally locked**
11. **Never edit a running script; never launch a second copy** — byte-offset splice, self-deadlock
12. **"Supported, corroborated, not certified"** is a real grade — don't round it either way
13. **Counts are immune** to drift, host starvation and neighbours. **Prefer them. Run them first.**

---

# 🔑 THE SESSION'S CORE FAULT, IN ITS SHARPEST FORM

> **An error path is not finished when the error is PRODUCED, only when it is LEGIBLE at the
> boundary that consumes it.**

**2026-08-18.** Four instances across three layers, and the common shape is **NOT** *"an error was
missed"* — **in every case the error was correctly detected, then discarded, summarised, or encoded
into something the consumer could not read:**

| layer | the error was… | what the consumer saw |
|---|---|---|
| KV cache | a real `DriverError` | a fixed string, cause discarded ⇒ **investigators sent to the wrong file** |
| bench harness | a failed request | a zeroed row ⇒ **a full table of zeros for a run where nothing worked** |
| **SSE streaming** | a correct `ValidationError` | **a bare string in `data:`** — every OpenAI client parses `data:` as JSON, **discards the frame**, and sees only `[DONE]` ⇒ **200 OK with no content** |
| CUDA build | a kernel never compiled | **`BUILD_RC=0`** ⇒ a binary shipped without the change |
| *(tooling, 5th)* | a zero-token row | in **neither** the ok nor the failed list ⇒ `rows_failed: []` beside `total_tokens: 0` |

**🔑 The chat-endpoint case is the purest: the engine was never silent — the serving layer made it
silent. `add_request` DID reject a template-less model. The message was always sent, in a shape
nothing could read.** And **non-streaming was always correct** (422 with a JSON body) — *"which is
exactly why this read as a model problem rather than a protocol one."*

⇒ **When you fix an error path, check what the CONSUMER receives, not what the producer emits.**
And **a five-token preflight probe would have caught it in a second instead of a session.**

## ⚖️ A CORRECT DISAGREEMENT WITH MAIN — recorded because the reasoning beat the instruction
Main asked for a missing chat template to be a **hard error at load**. The chain made it a **loud
`warn!` naming the model and every path searched — not a refusal** — because **a template-less
checkpoint is legitimately serveable via `/v1/completions` with a raw prompt**, and a startup refusal
would break that deployment. Per-request became a named **422** identifying the model, the paths
searched, and which endpoint to use instead.
**It flagged the disagreement rather than quietly shipping something narrower than asked.** ⇒
**Accepted. The instruction was wrong.** *Reason from the deployment, not from the tidiest rule.*

## ⚠️ `serde_json::Value` INDEXING RETURNS `Null` FOR A **MISSING** KEY
So an assertion claiming to check *"present and null, not absent"* **cannot detect absence** — and it
**passed its own mutation.** **Second time in one session that a guard passed a mutation on first
writing** (the first: a `contains()` satisfied by the checked function's own *definition* line).
**Both caught only because every check is mutation-tested.** ⇒ **Write the guard, then try to fool
it — the guard you just wrote is the least-tested code you have.**

# D33 — THE ONE-LINE FORM OF THIS ENTIRE FILE

> **Before trusting a check, feed it the failing case and confirm it says so.**

Recorded 2026-08-18 by ArcGate, after building **two instruments in one session that could not
return the negative answer** — while spending that same session writing exactly this criticism
on other people's PRs (*"the probe as authored could only ever answer no"*; *"a green here is
not evidence about the thing that would be wrong"*).

| # | The instrument | Why it could not fail |
|---|---|---|
| 1 | `cargo check … 2>&1 \| tail -25` | the shell reports **`tail`'s** exit status ⇒ **exit 0 over a failed build** with two `E0061`s |
| 2 | `select(.conclusion != null)` over a GitHub check rollup | GitHub returns `.conclusion` as **`""`** for pending, not `null` ⇒ **every pending check counted as concluded**, so `16/16` appeared the instant the checks were *created* and never moved |

**Neither was a wrong conclusion about the code. Both were an instrument incapable of the
negative answer.** That is D18's mechanical form and it generalises past tests to **any**
instrument — a query, a filter, a monitor, a status sweep, a grep.

## ⚠️ D33 REFINEMENT — A GUARD MUST ASSERT EVERY RESOURCE IT IMPLICITLY CERTIFIES

**2026-08-18. A "quiet host" guard certified a run DIRTIER than the unguarded one.** It asserted
zero compilers but **never asserted an idle GPU** — and `bench.lock` does not exclude chains that
never take it. The *guarded* run executed with `before: gpu=100 %, apps=[223062,]`, producing
**1465 µs at W=1 against the unguarded 134 µs — 11× high.**

> **A guard that checks one resource silently certifies all of them.** Name every resource the
> measurement depends on and assert each. Fixed here to require `compute-apps` **empty before AND
> after**, exit 2 otherwise.

**Caught only because the guard PRINTED the GPU state it failed to ASSERT on.** Second time in one
session that "log the quantity, not just the verdict" was the only thing standing between us and a
silently wrong number.

## 🔴 NEVER EDIT A RUNNING SCRIPT — bash reads by BYTE OFFSET while executing

**2026-08-18.** `easured:: command not found` was **not** a syntax error. **Bash reads a script by
byte offset as it executes it**, so patching `run_curve.sh` under a live instance **spliced the old
and new files together and resumed mid-comment.** The run produced a well-formed `ENVFAIL`.

**The chain discarded that ENVFAIL rather than count it as the guard working** — correct, and the
subtle part: **a guard "firing" for the wrong reason is not evidence the guard works.** A green from
a Frankenstein script is worthless, and so is its red.

⇒ **Launch every run from an immutable copy** (`run_curve_v2.sh`), and compose with the pidfile
rule: **never edit a running script, and never launch a second copy of one.** Together those cover
both self-inflicted stalls this session — the 50-minute self-deadlock and this splice.

## 🔑 MEASURE THE ARTIFACT YOU INTEND TO SHIP — not its components

**2026-08-18, caught by the chain against its own plan, one step before flipping a default.**

The cost curve times **legacy vs bisect**. **The thing proposed for shipping is the HYBRID.**
A hybrid is a third artifact with costs neither component has:

> its branch-B path pays for phases 1–3 **twice** — enumeration tombstones `probs[]`, so the
> fallback must **rebuild** it — and **that cost appears in neither column of the table.**

⇒ **Benchmarking A and B tells you nothing reliable about `hybrid(A, B)`.** The gate for flipping
the default is **a hybrid timing arm**, not the curve already in hand. *"Should track legacy below
the budget"* is precisely the word this session has spent its time retracting.

**This is the same shape as the vacuous-parity and false-premise failures: the test measured
something adjacent to the claim.**

### Evidence language that is worth copying — "supported, corroborated, NOT certified"
Three cost-curve runs exist: the original **unguarded**, `curve2` (guard held `bench.lock` **while a
neighbour ran the GPU at 100%**), and `curve3` (verified `apps=[]`, `compilers=0` at start).
**`curve3` and the unguarded run agree within 0.5% at every support width; `curve2` was 2–11× high.**
Two independent runs converging while the known-dirty one is **grossly separable** is strong
evidence — but `curve3`'s *after*-bracket fails, because a neighbour arrived during the correctness
phase that follows the timings.
⇒ **"Supported, corroborated, not certified"** is the honest grade, and a grade of its own is worth
having. Don't round it up to "measured" or down to "unknown".

```
support   legacy µs   bisect µs   speedup     (curve3, verified-clean at START)
    1        133.6      1996.8       0.1×
    8        504.5      1994.2       0.3×
   64       3124.9      1993.9       1.6×
  512      24322.7      1995.0      12.2×
 4096     206124.8      1993.5     103.4×
12928     664854.0      1996.3     333.1×
```

## ✅ DOCUMENT THE DIVERGENCE, DON'T HIDE IT — hybrid sampler, full premise + branch proof

```
T1 peaked W=8       distinct   8/8   | reach   6 vs   6 | TV=0.0000 | A=20000 B=0    AGREE
T2 scrambled W=16   distinct  16/16  | reach   8 vs   8 | TV=0.0000 | A=20000 B=0    AGREE
T3 diffuse distinct distinct 256/256 | reach  47 vs  47 | TV=0.0000 | A=0  B=3000    AGREE
T4 tie boundary     distinct   1/16  | reach  15 vs  15 | TV=0.0000 | A=20000 B=0    AGREE
T5 diffuse TIED     distinct  21/128 | reach 111 vs 116 | TV=0.1103 | A=0  B=3000    DISAGREE (expected)
C  narrowed top_k=2 distinct  16/16  | reach   8 vs   2 | TV=0.5027 | A=20000 B=0    DISAGREE (expected)
```
**Every row carries its verified premise (`distinct n/m`) and which branch ran (`A`/`B`).** T3 is
the load-bearing row: `A=0 B=3000` proves the fallback actually executed, and `TV=0.0000` on a
**verified** 256/256-distinct support means bisection reproduces enumeration exactly.

**Two emergent properties, neither designed, both flagged rather than buried:**
- **T4** — the hybrid matches legacy **15 vs 15** where *pure* bisect kept 16, because `kept=15 <
  budget` routes to enumeration. **The hybrid inherits legacy's tie semantics for small nuclei**, so
  bisect's documented tie divergence never arises there.
- **T5** — on a **tied** diffuse support the fallback keeps **116 vs 111, TV=0.1103**: erring
  **wider, never narrower**, and only where many tokens share one stored probability and the
  reference's own choice among ties is index-order arbitrary. **A real, quantified, documented
  divergence — not a hidden one.**

## 🔴 THE ENABLE-TRANSITION IS ITS OWN STATE, AND IT IS WHERE THE BUG LIVES

**2026-08-18, the arena's first hardware trial.** Bucketing was applied **symmetrically**:
`cache_put` filed buffers under `bucket(bytes)`, which rounds **UP**. Correct for a *request*;
**wrong for a put of a buffer the arena did not size.**

> **Every buffer allocated BEFORE the arena was switched on has its exact size. Freed afterwards,
> it enters the bucketed free list advertising capacity it does not have — and the next request in
> that bucket gets a SHORT buffer.**

Decode survived (its requests rarely collided); **prefill's larger, more varied shapes hit it on
the first allocation** and died with `CUDA_ERROR_INVALID_VALUE`.

**The chain's own words: *"I guarded the miss path and missed the enable-transition path."*** ⇒
**Objects created before a feature is enabled have different properties from those created after.
That boundary is a distinct state and needs its own invariant and its own test.**

**Fix:** `bucket_down(P) ≤ P` on puts. Invariant restored — request `R` takes key `bucket(R) ≥ R`;
a buffer of physical size `P` sits under `bucket_down(P)`; matching keys therefore imply
`R ≤ key ≤ P`. Arena-allocated buffers are already on a bucket boundary, so steady-state reuse is
untouched.

**Falsified, not asserted:** two new tests state the invariant directly. With the fix **7/7 green**;
with `bucket_down` reverted to round up, **exactly those two go red** and every pre-existing test
stays green. **The tests detect the specific defect, not a proxy for it.**

### ⚠️ AND THE FAILURE WAS LUCKY — DO NOT EXPECT THE NEXT ONE TO BE
**A short buffer is a wrong-numbers bug by nature.** It surfaced as a hard `CUDA_ERROR_INVALID_VALUE`
only because prefill's first oversized request happened to trip the driver **rather than quietly
corrupting a tensor.** The same defect one shape smaller produces a silently wrong model.

**Two asserts earned their place in one night, both of which would otherwise have been silent
greens:** the **symbol assert** caught a benchmark running a binary containing none of its chain's
code, and the **fingerprint-count assert** (2 instead of 52) caught this arena bug.

## ⚠️ VERIFY THE TEST'S PREMISE NUMERICALLY — a mislabelled premise reads as a defect

A "distinct probabilities" case was built with step **0.005**. **The BF16 ULP near 6.0 is
0.015625**, so that support held only **21 of 128 distinct stored values** — **a tie test wearing a
distinctness label.** Its `DISAGREE` looked like a sampler defect and was a test artifact.
**Second time a tie boundary masqueraded as a defect in one session.**

⇒ **Count and print the actual distinct *stored* values per case.** A premise assumed from the
construction is not a premise; the dtype gets a vote.

## ✅ BRANCH COUNTERS MAKE "WHICH PATH RAN" AN OBSERVATION, NOT AN INFERENCE

Hybrid sampler validation, with per-branch execution counts emitted beside every verdict:
```
T1 peaked W=8      reach   6 vs   6 | TV=0.0000 | branchA=20000 branchB=0    AGREE
T2 scrambled W=16  reach   8 vs   8 | TV=0.0000 | branchA=20000 branchB=0    AGREE
T3 false premise   reach 111 vs 116 | TV=0.1103 | branchA=0 branchB=3000     DISAGREE
```
**Without `branchA=0 branchB=3000`, a disagreement cannot be attributed to the chooser versus the
fallback.** A hybrid is a *third* code path; evidence for its two branches does not transfer to the
thing that selects between them.

## ✅ TWO RUNS CONTAMINATED AT DIFFERENT LEVELS, AGREEING ON SHAPE, IS REAL EVIDENCE

Both cost-curve runs were contaminated by different factors, yet **bisect flat / legacy linear /
crossover near 64 held in both.** A conclusion resting on *shape* survives any uniform contamination
factor even when every absolute is unquotable. **Say which half you are claiming.**

## ⚠️ D33 REFINEMENT — WALK THE QUIET BRANCH TOO. A guard can break on the ALL-CLEAR path.

**2026-08-18.** A host-contention guard was validated **on a busy box (17 compilers running)** and
passed. It was **broken exactly when the host was quiet**: `pgrep -c` prints `0` **and exits 1**
when nothing matches, so the natural `$(pgrep -c X || echo 0)` yields **`"0\n0"`** and every integer
test after it dies with *"integer expression expected"*.

> **The guard degraded precisely in the state it existed to certify.** Feeding it the *failing*
> case (D33) was not enough — the failing case was the only branch ever exercised.

⇒ **Walk EVERY branch, including "nothing is wrong".** Re-validated by traversing all three:
quiet → `0` cleanly; fake `ptxas` → `1`; compiler gone → back to `0`.
**A guard validated only on the busy path is half-validated.**

**How it was visible at all: the guard logs its VALUE, not just its verdict.** A verdict-only guard
would have failed silently forever. *(This same `pgrep -c` trap independently bit two separate
chains tonight — it is not exotic.)*

## ⚠️ `$$` INSIDE A BASH SUBSHELL IS THE **PARENT'S** PID — use `$BASHPID`

**2026-08-18.** A chain's self-test ran `assert_idle` in a subshell expected to `die2` → `cleanup`,
and `cleanup` `rm -f`'d **the live parent's pidfile**. Later it read the missing pidfile as *"the
sweep isn't running"* and **killed its own 107 GB server** — *"I acted on a false signal I had
created."*

**Its first fix — have `cleanup` only reap a pidfile containing its own `$$` — DID NOT WORK**,
because **`$$` in a subshell reports the parent's PID**, so the guard matched and deleted anyway.
**Caught only because it checked the pidfile immediately after launching instead of trusting it.**
`$BASHPID` is the correct value.

⇒ **A known-but-unfixed bug in your instrumentation will eventually be read as evidence about the
system.** That is the sharpest form of the session's recurring fault — **you become the source of
the signal you trust.** Fix instrument bugs *when you find them*, not after they mislead you.

## ⚠️ A **HALF-APPLIED FIX** IS ITS OWN FAILURE MODE — and it looks exactly like the original bug

**2026-08-18.** A chain switched its load generator from `/v1/chat/completions` to `/v1/completions`
— **URL and payload changed, but the parser still read `delta.content`, which that endpoint never
sends.** Result: **zero tokens, identical symptom to the original chat-template bug it was fixing.**
**Cost: one GPU session.**

⇒ **Changing a call means changing the request AND the response handling.** When a protocol changes,
the parser is part of the protocol. **The symptom of a half-applied fix is indistinguishable from
"the fix didn't work" — which sends you back to the wrong hypothesis.**

### 🔑 THE PREFLIGHT PROBE PAID FOR ITSELF ON THE CHAIN THAT BUILT IT
The same chain had just added a **five-token probe before the long run**. It caught the half-applied
fix **in one second instead of at the end of a 35-minute sweep** — *"the only reason it cost a second
and not a session."*
⇒ **Run a trivial end-to-end probe before committing an exclusive card.** Cheapest insurance we have;
adopted the same day it was invented and immediately justified.

**Three silent-success shapes that tool now refuses, having originally permitted all three:**
zero-token rows vanishing from **both** the ok and failed lists · a missing engagement beacon ·
**a protocol mismatch discovered only after the card was committed.**

## ⏱️ AN HTTP **READ** TIMEOUT RESETS ON EVERY BYTE
A server dribbling one token every few seconds **outlives any per-read deadline forever** — B=32
took **50 minutes instead of 125 s**. **Any load generator needs an ABSOLUTE per-request deadline.**
**And this failure mode only appears against a pathologically slow server — exactly the condition
being measured**, so it will not show up in rehearsal.

## ✅ CONTROL THE OBVIOUS CONFOUND — the uniform-vs-spread A/B
Uniform arm set to **1536 tokens against the spread pool's mean of 1493**, so per-request prefill
work is comparable and **the result cannot be attributed to "uniform prompts are just shorter."**
Same binary, same card, same server instance, same `max_tokens`; **only the prompt-length
distribution varies**, interleaved A-B-A-B so drift is *fitted*, not assumed away.
**Prediction written into the script before the run** so it cannot be retrofitted, and **the spread
arm doubles as a reproducibility check** on the first run (which measured width 4, 0.3 tok/s).

## ⚠️ "VERIFY BEFORE CROSSING A BOUNDARY" APPLIES TO **IN-REPO** BOUNDARIES TOO

**2026-08-18.** A chain wrote a test against **remembered** in-repo APIs and got three wrong in one
file — `MockVendor::new(1.0)` (really `MockVendorConfig`), `SloTier::default()` (really
`SloTier::from_id(1)`), and a missing `#[async_trait]`. Its own diagnosis: *"I skipped it because
the boundary was in-repo."*
⇒ **The global rule is not about third-party crates; it is about any API you did not just read.**
A type in your own workspace is as unremembered as one in `node_modules`.

## 🔴 TESTS THAT PASS *BECAUSE* THE BUG IS INVISIBLE — use Miri for the UB class

`mistralrs-quant/src/utils/uqff.rs:238` deallocated a `Vec<T>` through a `Vec<u8>` — **alignment UB
that corrupts the heap on any UQFF write.** Verified by **Miri**, not assertion:
`incorrect layout on deallocation: alloc303 has size 16 and alignment 4, but gave size 16 and
alignment 1`; the fix exits 0 and is **copy-free** (borrow via `as_byte_slice`, header factored out
so the `Vec<T>` stays alive).

> ⚠️ **The pre-existing qtip/codebook/bitshift round-trip tests DO NOT detect it. The bytes came out
> right — which is exactly why it survived.** A test that checks output cannot see undefined
> behaviour that happens to produce correct output *this time*.

⇒ **A Miri lane over `mistralrs-quant`'s pure-Rust tests would catch this class automatically.**
Recommended, not built. **This is the moat's own format writer.**

## ⚠️ A HARNESS THAT RECORDS ZEROS FOR A RUN WHERE NOTHING WORKED
`arc-cli/src/bench/scheduler.rs:296` recorded a failed request as a zeroed `RequestResult{ok:false}`
with the cause dropped — **producing a full table of zeros for a phase in which nothing succeeded.**
**Identical in shape to the 14-row table of `RAW`.** Fixed: cause captured with `{e:#}` so the whole
chain survives, and **a phase with zero successful requests is an ERROR, not a record.**
**Sharpest mutation: keeping only the innermost cause (`{e}` not `{e:#}`) goes RED** — so the test
guards the chain, not merely the presence of a message.

## 🔧 THE `.cuh` FIX — a directory watch was NOT enough
cudaforge's `CacheEntry` keys on the **`.cu`'s own content hash, GPU arch, and args hash**;
`collect_headers` is exported and **never called**. So Cargo re-ran and **cudaforge still skipped**.
Fix folds a hash of every `.cuh` into the nvcc args as `-DARC_KERNEL_HEADERS_HASH`, moving
`args_hash` — one of the three `needs_rebuild` inputs.
**Proven behaviourally on the real tree:** deterministic across runs · **unmoved by `touch`ing a
`.cu`** (no over-invalidation) · `0xa330c331…` → `0x655b2b4b…` on a one-byte `qtip_codebook.cuh`
edit · back to baseline on revert. **Guard mutation-tested both ways, and the build refuses on zero
headers so it cannot pass vacuously.**

## 🔑 WHEN THE BUG MANIFESTS AS A SPEEDUP, PRE-COMMIT A **CEILING**, NOT A FLOOR

**2026-08-18.** The KV-write bug made the graph arm read **85.7 T/s against 17.4 T/s eager** —
because the prompt step failed and decode ran from unwritten cache. **A normal pass condition
("graph ≥ eager") would have celebrated the bug.**

The fix's verification inverts it, **pre-committed in the script rather than left to a reading
afterwards**:
> leg 2 **fails** if the graph arm produces no decode row · if the `kv-write: src … incompatible`
> refusal reappears · **or if graph comes out more than 1.2× eager — the speedup itself is the
> failure signature.**
> leg 1 **fails** if the instrument emits zero lines · if any `action=REALLOCATE` remains · or if no
> V-half (`src=[1,1,N,1]`) `reuse` line appears.

⇒ **Ask what the bug *looks like* before choosing the pass condition.** A correct run here must be
**SLOWER** than the broken one; "faster" is the alarm. **And write the condition into the script
before the run**, because after the numbers exist it is nearly impossible to choose a bound without
being influenced by them.

**Paired discipline from the same chain:** the binary's md5 (`106adb5b…`) was checked to be
**provably different from the seeded target dir's stale one (`bdb02686…`)** — so the gate confirms
*something was actually rebuilt*, not merely that a binary exists. **A checksum that never changes
proves nothing; a checksum that changed proves a rebuild.**

## 🔑 RULE OUT BY FUNCTIONAL FORM, NOT BY CONSTANTS

**2026-08-18.** Main relayed a hypothesis: the V-half realloc discards **~22 MB per sequence**, and
the arena leaks **~24 MB per step** — suspiciously close, mechanism plausible.

**The chain killed it on shape, and said explicitly why that argument is the stronger one:**
> Their mechanism fires **once per sequence** (`REALLOCATE` on first decode, `reuse` thereafter —
> visible in their own trace). The leak grew **linearly**: 2,098 MB @ step 100 → 4,527 @ 200 →
> 6,878 @ 300, a straight line of slope ~24 MB/step.
> **A one-off cannot produce a linear slope, whatever the denominators.**
> *"The argument I'm relying on isn't 22 ≠ 24, it's one-off vs linear — a difference in functional
> form, which survives being wrong about the constants."*

⇒ **When two quantities look alike, compare their SHAPES (constant / linear / super-linear), not
their values.** A magnitude argument dies if either constant is off by 2×; a form argument doesn't.
**Numerical coincidence is the weakest evidence available and the most seductive** — 22 vs 24 was
close enough that main relayed it as a live lead.

*(It was still real dead weight — ~512 KB × 43 layers, put once and never taken — i.e. the textbook
case for least-recently-**taken** eviction, and a clean confirmation the fix is aimed correctly at
~0.2% of the total.)*

## ⚖️ THRESHOLDS COMPUTED, NOT CHOSEN — and the test caught the chosen ones
First bucket thresholds were **64 KB / 1 MB**, picked by eye. **The test rejected them: the arena's
mean buffer is ~32 KB, so a 64 KB cut never bites.** Corrected to **1/4-octave above 4 KB, 1/2 above
64 KB** — which takes a growing tensor from **45 distinct sizes to 13** over 3,000 steps.
**And the test bound became a computed worst case (13, asserted ≤ 16) rather than a number that
looked right.** ⇒ **A constant you cannot derive is a constant you cannot defend.**

## ⚖️ ASSERT WHAT THE CLAIM DEPENDS ON — the guard's scope follows the claim's type

**Refinement of "a guard must assert every resource it certifies", and its necessary counterpart.**

A correctness run printed `load=11.96` at start — residual from a build that had just finished
(load average lags ~1 min). The chain **did not reject it**, and was right:

> **That run makes no timing claim.** It exists only to emit `ARC_KVSHAPE` lines. What it depends on
> is **card exclusivity** — because V4 must fit in memory at all — not a quiet host.
> **Had it been a timing run, that number alone would have been grounds to reject**, and its
> `run_curve.sh` guard *would* have.

⇒ **Two guards, deliberately different, because two claim types:**
| claim | must assert |
|---|---|
| **correctness / counts** | card exclusivity (so the model loads); host noise is irrelevant |
| **timing** | card exclusivity **and** a quiet host **and** interval/detrended estimation |

**The failure mode this prevents is over-guarding as much as under-guarding:** a correctness
observation rejected for host noise wastes an exclusive window and teaches the chain to loosen
guards generally. **Name the claim first, then scope the guard to it — and say which you are making.**

**Also worth keeping: `76 GB resident at 198% CPU` was verified as model load rather than a stall by
reading process state, not by inferring from a flat VRAM number.** A plateau is not evidence of a
hang; the process table is.

## 🔴 CANCELLING CI **WRITES** STATE — it does not merely discard it

**2026-08-18.** 26 superseded CI runs were cancelled to free a starved queue. The runs were genuinely
redundant — **but `ci-complete` uses `if: always()` over `needs:` every lane, so a CANCELLED
dependency makes the aggregate `FAILURE`.** Result: **#94 = 15 cancelled + `CI complete: FAILURE`;
#110 = 11 cancelled + 2 real inherited failures + `FAILURE`.**

> **"Nothing informative was destroyed" was true about the *runs* and false about the *record*.**
> Cancellation wrote a **permanent false negative** onto both PRs.

**And the gate's own guard called it `ok`** — it counted CANCELLED as "concluded and not failed", so
a PR whose lanes *it had cancelled* was indistinguishable from one that genuinely failed. **Third
instrument in one session that could not return the right answer for a state its author created**
(after the piped exit code and the `!= null` filter).

⇒ **CANCELLED is its own verdict, never a synonym for pass or fail.** Rebuilt as `arc-pr-gate`:
**PASS / FAIL / INDETERMINATE / PENDING / CONTRADICTION**, self-tested on all six shapes including a
synthetic replay of #94's exact state. ⇒ **Before any bulk action on shared state, ask what it
*writes*, not only what it *removes*.**

## ⚖️ "A RECORD 20 RUNS DEEP IS NOT EVIDENCE, IT IS A PROMISE OF EVIDENCE"
The concurrency change (#132) reverses a **documented, deliberate** decision — the `run_id` fallback
existed so master pushes never cancel each other, preserving per-commit master results as CI-side
bisect evidence. **It would have destroyed the run on `8c8161286`, the very commit that broke
master.** The gate put that argument *in its own PR body* rather than burying it.
**Why the trade still holds:** those runs were **queued, never completed** — 20 deep, with the fix
for the break they were re-deriving stuck behind them — while a local bisect gave a **causal** answer
in ~10 minutes. **If durable per-commit master results are wanted, concurrency is the wrong
mechanism; a post-merge job produces them without competing for the PR queue's runners.**

## 🔴 "A GREEN STRUCTURALLY INCAPABLE OF CONTAINING THE THING BEING MEASURED" — five variants, one session

**⚠️ `nsys --duration` starts at PROCESS LAUNCH, not at first decode.** V4 takes **~65 s to load**,
so a 60 s window **closed before a single token was generated** — and **`nsys` still exited 0.**

**Caught by three instruments, none of them the return code:** report **size** (341 KB vs 591 MB for
a real trace) · `assert_engaged` (no decode ⇒ no counter lines) · `count.py` (**0 logits markers**;
the only D2H copies present were 536 MB/2 MB/512 KB **model-load** transfers).
**Fix: `nsys_leg` refuses any duration under 120 s at source; harness uses 150 s.**

### The five faces of the same fault
| # | Looked successful because… | The thing it actually measured |
|---|---|---|
| 1 | **exit code** came from a pipe (`… \| tail`) | `tail`'s success, not the build's |
| 2 | **distinct-size count** (6 `MISS during capture` lines) | 6 unique sizes, not 1,535 occurrences |
| 3 | **run average** over a warming cache | history + present, never the steady state |
| 4 | **stale file** never truncated between runs | the *previous* run's numbers |
| 5 | **profiler window** opened before the workload existed | 65 s of model loading, zero decode |
| 6 | **no positive check** — "don't gate on exit code" without "assert work happened" | a complete **14-row table of `RAW`**; the sweep declared DONE 3 s after the last slot started |
| 7 | **parser matched the wrong delimiter** — ASCII `\|` against comfy-table's `UTF8_FULL` `│`/`┆` | would have returned empty on **every** run and read as *"the step never completed"*. **Caught before any GPU time was spent**, by validating the parser against a real prior bench log (`prefill=31.1 decode=6.3`) |

> **Every one exited 0. Every one was a green that could not possibly have contained the answer.**
> ⇒ **Ask of any green: "could this artifact physically contain the thing I claim to have
> measured?"** Size, marker count, and engagement counters answer that; return codes never do.

## 🔴 ABSENT MUST NOT BE INDISTINGUISHABLE FROM STALE

**2026-08-18.** A waiter reported arena counters that looked like a clean result —
`driver_allocs 3812.8`, `accounting OK` — from a per-leg log timestamped **11:41**, belonging to
the **previous defective build**, **three minutes before the new run had even reached that leg.**
`all.sh` never cleared per-leg logs between runs, so **a missing result read as last time's result.**

> **Same family as the stale binary and the lost stdout: the harness could not distinguish
> "no answer yet" from "an answer".**

⇒ **Truncate every output artifact at run start**, and arm waiters against a **freshly-truncated**
file rather than per-leg files that persist across runs. **A result must carry proof it belongs to
*this* run** — a run id, a start timestamp, or a file that provably did not exist a moment ago.

**Three variants of this one fault in a single session:**
| # | "absent" was read as… | caught by |
|---|---|---|
| 1 | a **stale binary** — benchmarked code containing none of the chain's own changes | a symbol/fingerprint assert |
| 2 | **no execution** — a counter printing to swallowed stdout from a spawned thread | a file-based probe |
| 3 | **last run's numbers** — per-leg logs never cleared | a timestamp that predated the run |

## ⚠️ NEVER EDIT A RUNNING SCRIPT — violated by the chain that wrote the rule
The same chain that **staged** an earlier fix to `all.sh.next` *specifically because* bash reads
lazily by byte offset then **patched `all.sh` in place while it was executing** an hour later.
Inserting ~370 bytes near the top shifts every later offset. **The process stayed alive and looked
fine** — the corruption surfaces at the *next* command boundary read from disk, not at edit time.
It killed the run rather than let it emit something plausible. **Cost: one exclusive-card window.**
⇒ **Knowing a rule is not following it. Make it mechanical: run from an immutable copy, always.**

## ⚠️ A PREDICTION RIGHT IN SUBSTANCE AND WRONG IN DETAIL IS WHERE "PROBABLY" BECOMES A GUESS

**2026-08-18.** ArcGate predicted #110/#94 would fail their **`Test Suite`** lanes from an inherited
break. **#110 failed `cargo check (cuda, workspace)` instead** — right cause, wrong lane.

**That mismatch is the danger point.** "It's probably the inherited break" is exactly the reasoning
that ships a real regression as collateral. It ran the **causal** test instead, on #110's head:
```
cargo check -p mistralrs-core --tests                 -> exit 101   (E0308, tokio vs std Mutex)
… cherry-pick the one-word fix onto the SAME tree …
cargo check -p mistralrs-core --tests                 -> exit 0
```
**Changed the one thing, the failure disappeared, reverted the probe immediately, branch untouched.**
Causal, not correlational. ⇒ **When a prediction lands off-target, escalate to a controlled
intervention — do not downgrade to "close enough".**

## ⚠️ THE LANE NAME IS NOT THE DIAGNOSIS

The failure surfaced in the **CUDA** lane. The error is **plain Rust** in
`paged_attention/scheduler.rs` — not CUDA-gated at all. That lane's *"Type-check CUDA-gated tests
(no codegen, no run)"* step simply **reaches the `--tests` target first**, before `Test Suite`
reported. **Reading "cuda lane red" as "CUDA problem" sends the next person hunting in the wrong
crate.** Read the *error*, not the job title.

## 🔑 "THE DISAGREEMENT, NOT THE VALUE, CARRIES THE INFORMATION"

**2026-08-18, the sharpest formulation of the redundant-quantity rule, from the arena chain.**

Its arena **reported itself healthy while silently bypassing itself for every small buffer**:
`accounting OK` · **4,976 cache hits/step** · 955 MB high-water · **bit-identical output over 52
steps**. Every headline green, all of them true.

**The only tell was `driver_frees = 3,672.8/token` being non-zero at all** — a quantity that is
*supposed to be impossible* when the arena is on. (Cause: a `bucket_down` fix returned `0` below 128
bytes, so `cache_put` refused every sub-128-byte buffer and it allocated **and** freed through the
driver, every step.)

> **"I'd have caught nothing had I emitted only the headline rate."**

⇒ **Emit the quantity whose value you can predict, not just the one you want.** A guard's power is
in the *pair* that must agree, and a metric that should be structurally zero is the most
informative thing you can print. **Two defects in one subsystem tonight, both found by instruments,
neither by inspection.**

## How it was caught — the transferable half

Not by re-reading the filter. **By noticing two of its own outputs contradicted each other:**
`done=16/16` and `ci=` (no `CI complete`) cannot both be true, because 16 green lanes must fire
`ci-complete`. That prompted opening one raw rollup, which showed
`Check (ubuntu-latest, stable): ` with an **empty** conclusion.

⇒ **Emit a second, redundant quantity whose disagreement with the first is impossible if both
are right.** A single number cannot audit itself; two that must agree can.

**No merge was made on the bad reading** — every merge to that point was decided on a rollup
read field-by-field, so nothing needed reversing. Worth noting: the *slower, manual* method was
the trustworthy one, and the automation built to replace it was the thing that lied.

---

# D18 — SILENT SUCCESS IS THE HOUSE FAULT. Assume the pass is fake until a failure can prove itself.

Named 2026-08-17 after the same bug appeared **eight times in one session**, across
kernels, harnesses, tooling and CI. It is not a kernel bug or a test bug. It is a
*shape*: **a failure path that returns the success value.**

## The eight instances

| # | Site | What it did |
|---|---|---|
| 1 | `candle-flash-attn-v3` varlen (`lib.rs:591`) | clamps `window_size_right` before `is_causal` is derived ⇒ `causal=true` silently becomes **full attention**. Upstream: 9/9 returning the non-causal result. |
| 2 | PR #88 diverted shapes | out-of-envelope shapes inherited the `(1,1)` flash placeholder mask ⇒ would have run **silently non-causal**. Caught *while fixing #1*. |
| 3 | TurboQuant launchers (×6) | `if (hs != 128) return;` — a **no-op leaving the output buffer uninitialised**. The `static_assert` everyone blamed was dead code with no callers. |
| 4 | `tensor_device_ptr` (`weights.rs:138-175`) | no `DType::I32` arm ⇒ per-token **CPU fallback** at ~10/sec, visible only as a WARN nobody reads. |
| 5 | **Copy 7 of the same family** | the `F8E4M3` arm was still a literal `as_cuda_slice::<u8>()` **inside the consolidated helper** — the fix for copies 1–5 shipped with a copy of the bug inside it. |
| 6 | wave61 harness | printed `RESULT=OK` while **every batch row was `None`**. The coherence canary proved the *model* was fine; nothing asserted that *numbers were produced*. |
| 7 | PR-merger + monitor | a transient `gh` failure returned an empty string, read as "resolved" ⇒ **silent exit 0, nothing merged.** Both tools, same defaulting. |
| 8 | PTX probe (main's own) | `cudaDeviceSynchronize()` returns success for a kernel that **never launched**; the fault sits in `cudaGetLastError()`, which the first probe never read. |

Adjacent, same family: `share_stats`/`CrossPrefixMeter`, `TurboQuantSingleCache`,
`PostLoadHook` — **wired but dead**, count 13+ in [[backlog]]. And `qs` (query stride)
and `softcapping`: accepted, plumbed through the FFI, **silently ignored**.

## The mechanical form — the one sentence to check code against

> **The absence of a signal was read as a specific signal.**

Every instance above has that shape. FA3's clamp, the missing dtype arm, `RESULT=OK`
over `None` rows, an empty `gh` response compared against `!= "OPEN"`. So the rule is
narrow enough to be checkable:

> **A status check must distinguish "not observed" from "observed negative" — never
> collapse them into a default.**

The correct form is to `continue` on empty rather than conclude, and to emit a line
for **every** terminal outcome including timeout, so silence cannot mean success.

Note this is a **different gate** from the wired-but-dead class ([[backlog]],
`unreachable_pub` + `cargo public-api`). Dead code and defaulted-to-success are
distinct failures that merely share a symptom: *a path whose breakage never reports
itself.* Two gates, not one.

Provenance worth keeping: the agent who wrote instance #7 was **actively hunting this
pattern in other people's code when it shipped its own copy** — and copy 7 was a
literal `as_cuda_slice::<u8>()` *inside the consolidated helper that fixed copies 1–5*.
The consolidation was both the fix and the carrier. **Vigilance was being applied at
maximum in both cases and did not prevent either.** That is the argument for
mechanical gates.

## The rules that follow

1. **A green result must prove work happened**, not merely that nothing threw.
   Assert non-empty output, non-zero counts, an engagement log line present in the
   treatment arm *and absent in the control*.
2. **Distinguish "it failed" from "we could not test it."** Separate exit codes:
   `0` pass · `1` genuine failure — the only strategy signal · `2` environment could
   not answer. Conflating 2 with 1 nearly reversed the FA4 decision on a laptop
   missing `libcuda`.
3. **Identity between two wrong runs is not evidence.** Both arms of an A/B on a
   broken box produce identical garbage: identity passes, ratio is exactly 1.00,
   everything green. Gate on a **coherence canary** — questions a working model
   cannot miss — before any timing counts.
4. **A broken box reports a BETTER number, not a worse one.** Garbage decodes fast.
   The failure mode is the number a session under pressure is least likely to query.
5. **Prefer a mechanical gate to another fix.** The dtype family ended with
   `device_ptr_supports_dtype` + an exhaustive `match DType` tripwire, so a new
   candle variant **breaks compilation**. Eight instances is a missing gate, not
   eight mistakes.
6. **When you fix one, grep for the family.** #52 fixed three copies, #53 two more,
   and the consolidation contained another. The sibling audit that found copy 7
   swept **337 sites**, ranking 43 suspect.

---

# D19 — Shared `/tmp` is a cross-agent hazard. Namespace every scratch path.

**This already caused wrong content to be published.** 2026-08-17: one agent wrote
`/tmp/pr_body.md`; a second agent overwrote it; the first then **published the second's
text to PR #99** before catching it. Restored, but the PR briefly carried another
workstream's words.

With 6–8 agents concurrent, every unqualified `/tmp/<name>` is a collision waiting to
happen — and the failure is silent, because a file that exists and parses looks exactly
like your own.

**Rules:**
- Namespace all scratch paths: `/tmp/arc-<agent-or-branch>/…`, or use a private dir.
- Never write a shared, guessable filename (`pr_body.md`, `out.json`, `status.txt`).
- Same applies **on the box**: `/root/status.txt` and `/root/*.log` are shared across
  every gate run. Prefix by wave/gate.
- Read-back-verify anything you are about to publish externally (PR body, model card,
  release note). Cheap, and the only thing that caught this one.

Related: the shared git clone was switched under agents **twice** the same day, and one
agent's commit landed on another's branch. **Every agent works in its own worktree.**
The user noticed his own clone sitting on a stray branch.

---

# D18, instance 10 — a fix that silently did not apply

Recorded because it is the sharpest instance yet: an agent's scripted edit added a
`cudaGetLastError` status check meant for `grouped_dtype!`, but landed it in
`dequant_dtype!` ~800 lines earlier. Both macros end with the identical
`unsafe { $launch(...); } drop(out_guard); wrap_cuda_slice(...)`, and **the edit
asserted its anchor was PRESENT, not UNIQUE — so it took the first match.**

Consequences: 15 compile errors (E0308 ×3 from a `()`-returning launcher, E0425 ×12
from names that exist only in the grouped function), and — far worse — **the grouped
GEMM still had no status check at all.** A D18 fix was itself silently not applied:
*"the anchor matched" was read as "it matched where I meant."*

**Rule: an edit anchor must be asserted UNIQUE, not merely present.** Count matches;
fail on ≠1. This is the same mechanical shape as the class it was fixing, which is why
it belongs in the same doctrine.

Still outstanding from the same finding: `dequant_dtype!` carries the identical D18
exposure — void launchers, so a failed launch returns the `alloc_zeros` buffer as a
valid all-zero result. Fixing it changes `qtip_dequantize.cu`'s ABI; logged in [[backlog]].

---

# D18, instance 11 — the verification layer itself was absent, and absence read as green

**The worst instance recorded.** Every prior one was a single code path lying about
itself. This one is the *whole CI lane* missing for an entire class of PR, presenting
as a pass.

**Provenance — the literal trigger, `.github/workflows/ci.yml:7-9` before the fix:**

```yaml
  pull_request:
    branches:
      - master
```

`cuda_compile_check.yaml` carried the same `branches: [master]`. A PR whose **base is
another PR's branch** — a stack — matches neither, so **not one lane ran**: no compile,
no test, no rustfmt, no clippy, no MSRV, and no nvcc. The single workflow with no branch
filter is `analysis.yaml`, which triggers on `pull_request_target`, so those PRs showed
**exactly one check, `comment`, and rendered green.**

Six open PRs were in that state simultaneously — the stack #95 → #100 → #102 → #103 →
#104, plus #98. Confirmed by check-run census: 16 checks on master-based PRs, **1** on
every stacked one. It also retroactively weakened every "CI green" statement relayed
about that stack; the agents' own local runs remained real evidence, but CI had
corroborated nothing.

**Mechanical form, unchanged:** *the absence of a signal was read as a specific signal.*
Here, "no CI is configured for this base branch" rendered as "checks passed."

Note the compounding with the macOS gap: a stacked PR touching `#[cfg(feature = "cuda")]`
Rust got **neither** the nvcc lane **nor** a local type-check, since `cargo check` on
macOS structurally cannot see those arms. #99's missing `qtip_grouped_tile_m` re-export
is exactly that hole.

## The fix, and why the trigger fix alone is not enough

1. `pull_request.branches: ['**']` in `ci.yml` and `cuda_compile_check.yaml`. Stacked
   PRs are normal in this repo and must get the full lane.
2. A **`ci-complete` guard job** that `needs` every lane, runs `if: always()`, and fails
   unless each reports `success` — treating `skipped` and `cancelled` as failures — and
   additionally asserts the lane **count**. A branch filter is only one route to "the
   lanes did not run"; a skipped job, a renamed job, an empty matrix, or an
   over-narrow `paths:` filter all reproduce it. `ci-complete` is the only job whose
   non-success is unambiguous, because unlike the individual lanes **it cannot pass by
   not existing**. It is the correct required status check for branch protection.

**Rule: a green PR must be green because checks RAN and passed, never because none were
configured. Assert the expected job set is present; never infer a pass from an empty
check list.** Corollary for tooling and for agents: when reading CI state, count the
checks before trusting the conclusion — `0 failures` out of `1 check` is not a pass.

---

# D18, instance 13 — a MISATTRIBUTED signal, and the first one in a DEPENDENCY

Recorded 2026-08-17, from #108's arch-matrix CI job on its first real run.

Every earlier instance was *absence* read as a specific signal. This one is worse:
a **real failure attributed to the wrong component**. Absence at least looks like
nothing; a misattributed signal looks like information and points somewhere false.

**Provenance — what `cargo build` reported:**

```
CompilationFailed { path: "src/cuda/gemv_bf16.cu", message: "nvcc error:\n\n" }
```

**What the raw log said two lines earlier:**

```
Segmentation fault (core dumped)
```

**nvcc crashed, and `cudaforge` surfaced the crash as a compile error with an empty
message.** Read as written, it says "Arc's `gemv_bf16.cu` does not compile" — a source
bug, in our code, on a specific file. The truth is a toolchain crash that says nothing
about that file at all. The only reason it was not filed as an Arc kernel bug is that
someone pulled the raw job log instead of trusting the build's own summary.

**Second instance, same window, same shape — also a dependency, also silent:**
`nvcc -arch=sm_90a` was accepted **without any diagnostic** and then emitted a
`compute_90` intermediate, so ptxas rejected the wgmma as *"not supported on .target
'sm_90'"*. The tool accepted a request it did not honour. Related: `nvcc
--list-gpu-code` never prints the arch-specific (`a`) variants at all, so the
`grep -qx sm_90a` check shipped in the wave65 gate and the wave66 box runner would have
reported *"toolkit cannot target sm_90a"* on a toolkit that can — a false negative
inferred from a listing that was never going to contain the answer.

## The rule

**An error message from a dependency is a claim, not a finding.** Before acting on a
third-party tool's attribution — especially before filing it against our own code —
confirm it against that tool's raw output. Concretely:

* **An empty or contentless error message is itself the signal.** `message: ""`,
  `nvcc error:\n\n`, a non-zero exit with no stderr — treat these as *"the wrapper lost
  the real error"*, never as *"the error was about the thing the wrapper named"*.
* **A crash is not a compile error.** SIGSEGV/SIGKILL from a compiler says nothing about
  the source file it happened to be holding. Separate "the toolchain died" from "the
  code is wrong" before assigning an owner — the same bucket split D16's gate already
  makes between a toolkit limit and an Arc failure.
* **Ask a tool what it DID, not what it claims to support.** Read back the artefact —
  the emitted `.target` line, the `cuobjdump --list-elf` output — rather than trusting
  a capability listing or the silent absence of a complaint. A flag accepted without
  error has not thereby been honoured.

Corollary: our own D18 discipline stops at our repository boundary, and the tools we
build on do not share it. Every dependency that reports on our behalf — build wrappers,
compilers, CI actions — is an unaudited narrator.

---

# D18, sub-pattern — VERIFICATION CODE IS NOT EXEMPT, and is where this bug hides best

Named 2026-08-17 after **four instances in one working window**, every one of them
inside code written *specifically to catch silent success*. The doctrine had been
applied to kernels, launchers and harnesses; it had not been turned on the tools
doing the checking. That is precisely where it was hiding.

| # | Site | The silent success |
|---|---|---|
| 1 | #94 regression pin (`cache_engine.rs` tests) | The test skipped every width already in `TURBOQUANT_DEFAULT_HEAD_DIMS`. Widening that const — the exact regression under test — **emptied the loop**, so it passed asserting nothing. A test written to prevent silent success, failing silently. |
| 2 | wgmma descriptor probe | Indexed a host `constexpr` array with a runtime index *inside a kernel*. nvcc rejects it outright — meaning **the file had never been compiled by anyone**, while existing as evidence. |
| 3 | Arch witness | Launched on the default stream. Under CUDA graph capture a launch is **appended to the graph, not executed**, so the witness returned its `memset` zero — read as a confident "no SM90 device code present". |
| 4 | CI monitor | Reported "all checks terminal" because it counted pending checks and found none. **An empty check list also has zero pending.** Same defaulting as D18 #7. |

Three of the four were found by *running* the tool, not by reading it. Instance 1 was
found only by mutating the constant the test claimed to pin — the test looked correct
and was correct-looking for the wrong reason.

## The rule

**A test, probe, witness, gate or monitor must be shown to FAIL when the property it
asserts is false.** Never accept a green from verification code that has not been seen
red. Concretely:

* **Mutate the thing being pinned** and confirm the assertion fires. If widening a
  constant, deleting a lane or reverting a fix leaves the check green, the check is
  decoration.
* **Assert the sample is non-empty before iterating it.** `for x in filtered` where
  `filtered` can be empty is a vacuous pass; count first, fail on zero.
* **Distinguish "no failures" from "no results".** Zero pending out of zero checks, an
  empty `gh` response, an all-zero output buffer and an uncompiled file are all
  *absence*, never *pass* — the D18 mechanical form, applied to the checker.
* **A probe must be proven to have executed**, not merely to have returned. Graph
  capture, a skipped dispatch arm and a file that never compiled all "return" fine.

Corollary, stated because it keeps being the expensive half: **the more a piece of code
exists to enforce rigour, the less likely anyone is to test it.** Audit the auditors
first.

---

# D18's mirror image — SILENT ERASURE. A measured result relabelled "unmeasured".

Recorded 2026-08-17, after Jish: *"And turboquant kv was fucking measured too."*

D18 says a failure path must not return the success value. This is the same shape
pointed the other way, in the knowledge base rather than the code: **an absent record
returned as evidence of absence.**

**What happened.** `4eba13905` (2026-04-06) records *"55 tok/s with TurboQuant = 46%
over Candle baseline"* on a B200, with correct output, from a harness that is in the
tree (`deploy/modal_b200.py`, `gpu="B200"`, `MODEL="Qwen/Qwen3-32B"`). No row for it
was ever written into [[facts]]. An honesty sweep then searched FACTS.md, found
nothing, and wrote **"never measured"** into README, ARC_V2, FLEET, BENCHMARKS,
PEAK_INFERENCE, RELEASE_NOTES, TAXONOMY, CODE_INDEX and two crate `//!` docs. wave61-CL
**quoted the commit** and still concluded "never measured", calling the model
"unstated" when the harness names it one file away. Successive agents copied the phrase
forward; main relayed it to Jish uncaught. The owner had to correct it.

**Two mechanisms, both worth checking for:**

* **Absence read as disproof.** "Not in FACTS.md" means *nobody wrote the row*. It is
  the same defaulting as an empty `gh` response read as "resolved" (D18 #7) — absence
  is never a verdict.
* **A true qualifier hardening into a false one.** `docs/PEAK_INFERENCE.md` correctly
  said *"no B200 has ever been rented."* The run was on **Modal**, so it left no rental
  line. "Never rented" became "no B200 evidence exists" became "never measured on a
  GPU, on any model." Each step looked like a faithful restatement of the last.

**The rule.** *Calling a measured result unmeasured is the same failure as calling an
unmeasured one measured — both misstate the evidence, and honesty discipline (D9) cuts
in both directions.* Before writing "never measured" about any feature:

* `git log --all --grep=<feature>` and `--all -S<number>`, and read `deploy/` and
  `arc-tools/` for harnesses — not just the rental ledger and FACTS.md.
* **Retract the claim at the granularity it fails at.** "4.27× is arithmetic" was
  right; "therefore nothing was measured" was a different claim smuggled in beside it.
  A narrow measurement is still a measurement — state its scope, do not delete it.
* When a run *is* found with no FACTS row, **write the row** in the same change. The
  missing row is what made the erasure possible.

---

# D20 — Merging a stacked PR: never `--delete-branch` on a parent

Learned the expensive way, 2026-08-17. Merging **#92** with `--delete-branch` deleted
`feat/mtp-per-seq-kv`, which **auto-closed #95** — GitHub closes a PR whose base ref
disappears; it does **not** retarget. And **#95 could not be reopened**, because
reopening requires the base ref to exist. Recovery: recreate the ref at the old head,
reopen, retarget to master, delete the ref again.

**The order, for every stacked chain:**
1. merge the parent **without** `--delete-branch`
2. **retarget the child** to master
3. *then* delete the parent branch

## Related: CI does not run on stacked PRs at all

`ci.yml` and `cuda_compile_check.yaml` gate on `pull_request: branches: [master]`, so a
PR based on another PR's branch gets **zero** lanes. The only automatic trigger is
`pull_request_target` → `Analysis`. On **#109** the CUDA lanes ran solely by hand via
`workflow_dispatch`, and **`ci.yml` has never run there at all** — while the PR read as
green. Six stacked PRs were in that state before it was found.

Fixed in **#106**: `branches: ['**']` plus a `ci-complete` guard that cannot pass by not
existing, and a `concurrency` block on `ci.yml` (the `head_ref || run_id` idiom, so master
pushes and scheduled runs never cancel each other — only PR iterations do). Its absence
meant every push across ~15 open PRs left superseded runs queued; 13 had to be cancelled
by hand.

**Make `ci-complete` the required check** — it is the only job whose non-success is
unambiguous.

## Citation discipline

The TurboQuant provenance chain needed a correction found by reading the object store
rather than relaying: `deploy/modal_b200.py` names the model, but **was not in the tree at
`4eba13905`** — it landed the next day at `404ee1aad`. Ordinary sequence, and it does not
weaken the finding, but the citation must say so. **A correction to a false record must
itself be unchippable**, especially when the original error began with an audit that
quoted a commit, called the model "unstated", and concluded "never measured" anyway.

---

# D21 — A SCOPING RESULT IS NOT A VERDICT. Build it, fix it, don't rank it down.

Jish, 2026-08-17: *"a frontier / novel system doesn't work != turn it off, it = work on it
further and fix it"* · *"No matter what you're keeping cuda graphs on"* · *"no limiting
beliefs"*.

**Arc's premise is building things nobody else has.** So "it doesn't work yet", "it has a
lower ceiling than hoped", "it costs throughput in this regime" are **scoping facts** —
they say *where* a system helps and *how much*. They are never grounds for shelving it.

## Three times in one evening, all the same reflex

| Finding | What it actually was | How it got framed (wrong) |
|---|---|---|
| MTP yields ~1.0 tok/step at B=128 | our cohort min-rollback is the cause; production engines do per-sequence advance | "MTP **refuted** at batch" |
| Per-seq advance: tok/step **+9%/+13%**, aggregate **−5×/−20%** | mechanism works, per-step cost regressed — a bug with a location | "a **regression, not a win**" |
| CUDA graphs: ~88% ceiling at b=1, ~26% at B=256 | tells you *which regime* graphs serve | "**not the priority**" |

Each time the underlying analysis was correct and useful. Each time it was converted into
a judgement about whether the system deserved to exist. **The reflex is toward tidiness —
closing a question feels like progress. On this project it deletes the work that matters.**

## The rule

1. **Report scope, not sentence.** "Helps at b=1, not at batch, ceiling X%" — never "not
   worth doing".
2. **A non-functional subsystem is unfinished, not marginal.** CUDA graphs were behind
   three default-off gates *and* discarded replayed logits — no token had ever come from a
   replay. That is a half-built feature, and a ceiling estimate says nothing about whether
   to finish it.
3. **Negative numbers point at a location.** tok/step up while aggregate down means steps
   got slower or fewer — decompose `agg = tok_per_step × steps/sec` and name the cost in
   ms. Don't stop at "it regressed".
4. **Turn it on and fix it** is the default disposition for anything novel. Off is a
   temporary state with an owner and a next step, never a conclusion.
5. **Do not apologise repeatedly.** Fix the reflex, record it here, move on. Jish has
   corrected this three times and should not have to a fourth.

## Standing context that must survive compaction

**Why Arc exists:** Runcrate rents GPUs. The wedge is **datacenter capacity** — one node
serving 4–8× more multiplies a fleet without buying a card. **×4–8 is credible; ~×1 is
shipped.**

**The moat is the byte formats, not any one kernel** — trellis weights (QTIP) and
compressed KV (TurboQuant) are formats nobody else reads, which is why the GEMM and the
attention kernel must be ours (see D17). Anyone can adopt a better attention kernel;
nobody can adopt our weight format without writing the decoder.

**Anti-pessimism protocol** lives in [[CEILINGS.json]] with the recorded pessimism
failures. Physics bounds and implementation gaps are different things: b=1 ceiling
**1,413 tok/s**, aggregate **~16,600 tok/s at B=256 on one H200**. Every measured gap so
far has been code, not law. Read it before saying anything is "not achievable".

## D18 #14 — verification code is not exempt, and is where this hides best

One session, seven instances, **all inside tools written to catch exactly this**:

| # | The tool | How it lied |
|---|---|---|
| 1 | a regression pin | skipped default widths, so widening the const **emptied its loop** and it passed asserting nothing |
| 2 | a CI monitor | reported "all checks terminal" because an **empty** check list also has zero pending |
| 3 | a crash guard | `grep -c` prints `0` *and* exits 1, so `\|\| echo 0` yields `"0\n0"`; the integer test errored and the branch **never fired while looking installed** |
| 4 | a lock cleanup | a live lock deleted on an unread assumption — one `cat` would have refuted it; skipped **because it was classified as housekeeping** |
| 5 | a probe's **control** | `mma.sync…f32.bf16.bf16` needs four type qualifiers, not three. A malformed control fails **looking like a result about the thing under test** |
| 6 | a per-arch diagnostic | added *to make a false failure interpretable*, itself used `-arch=sm_90a` ⇒ printed `OK sm_90a` for files compiled as `sm_90` |
| 7 | the arch gate | built its `ARC_CUDA_ARCHS` request **from the oracle it was meant to check** (`nvcc --list-gpu-code`, which never prints `a` variants) ⇒ resolved to `80`, built single-arch, and **passed** — proving the arch matrix while testing one architecture, unable to fail |

**Corollary — every dependency that reports on our behalf is an unaudited narrator.**
`nvcc` accepts `-arch=sm_90a` with **no diagnostic** and emits a `compute_90` intermediate,
so ptxas's *"not supported on .target 'sm_90'"* reads as a hardware limit rather than a flag
silently not honoured. `cudaforge` surfaced an nvcc **`Segmentation fault`** as an empty
`CompilationFailed{ message: "nvcc error:\n\n" }` — a *misattributed* signal, worse than
none. **Always: `-gencode arch=compute_90a,code=sm_90a`, emit PTX, read the `.target` back.
Believe the artefact, never the advertisement.**

**The discipline fails where the stakes look lowest** — cleanup, housekeeping, "just a
diagnostic". Three of the seven were caught by *running* the tool rather than reading it.

---

# D32 — "NEUTRAL" IS THE SIGNATURE OF AN UNREACHED PATH. Prove execution before believing a null.

**Twice in one session, two different mechanisms, the same false conclusion.** Both would have
been reported upward as "our new kernel costs nothing."

| # | The null | Why it was fake |
|---|---|---|
| 1 | ArcFlash 512: **−0.3%**, "neutral" | `ARC_FLASH_512` is consumed *downstream of an early return*. `dsv4_attention.rs:682` routes every V4 decode step into `absorbed_mqa_decode` first. The fused kernel was entered **0 times in 320 decode steps × 43 layers**; positive control `ABSORBED-MQA-DECODE` fired **12,000+**. The arm assertion passed because `flash_512_enabled()` is evaluated during *prefill*, so the mode line printed truthfully about a path the measured span never touched. |
| 2 | Trellis GEMM A/B, ~0% expected | `gather_forward` (`bitshift.rs:1800`) takes the fused gather-GEMV whenever `n_tokens <= ondevice_max_tokens`, and only falls through to the grouped GEMM below it. At B=128 **every call went to the GEMV.** `Path::Grouped::select()` sets `ARC_NO_QTIP_ONDEVICE_MOE` for exactly this reason; the new `--ab` path never called it. |

⇒ **A change that touches a hot path and moves nothing is more likely UNREACHED than genuinely
free.** Treat a null as suspicious until execution is proven: a **launch counter**, an
engagement marker *inside the arm*, `reachable && calls > 0` on the span. Instance 2 was caught
only because the counter gated the timing.

## The sharpest corollary — a negative control does not prove which code ran

> **A negative control proves the comparison is sensitive to its INPUTS. It says nothing about
> which code path produced them.**

Instance 2's parity gate printed **`parity OK`** while comparing the fused GEMV **to itself** —
and its negative control (a different routing draw must differ) **passed**, because it is a
genuine control for inputs. **Byte-equality between two runs of the same wrong kernel is
perfect and meaningless.** Gate parity on the same execution counter as the timing.

## And: reaching the kernel under test is a STEP, not an assumption

In instance 2 the harness already knew — `Path::Grouped::select()` existed precisely to force
the dispatch — and the new code path simply didn't inherit that knowledge. **When adding a new
entry point to an existing harness, enumerate what the old one did to make the measurement
valid.**

## Meta-rule, from the same chain hitting `pkill -f` again after writing two paragraphs on it

> **Knowing about a footgun is not the same as having removed it — the fix is deleting the tool
> from the script, not remembering not to use it.**

Same argument as mechanical gates over vigilance ([[D18]]). **A rule you have to remember is a
rule that fails under time pressure.**

**TWO CHAINS REACHED THIS INDEPENDENTLY, IN DIFFERENT DOMAINS, IN ONE SESSION — treat it as
settled.** The keystone chain hit the self-matching `pkill -f` pattern **twice**, the second
time *after writing two paragraphs about it*. And the prefill chain wired a cursor into one of
**two** dispatch paths — the exact family `wave64-CP §3` had documented hours earlier — having
**read that note and quoted it in its own summary**:

> I read that note, quoted it in my own summary, and then made the identical mistake. **The
> lesson is evidently not learned by reading it.**

**A rule violated by someone who has just quoted it is the strongest available evidence against
relying on rules.** The correct response to a recurrence is never "document it harder":
- delete the tool from the script (`pkill -f` → recorded PID);
- collapse the entry points so there is **one** place to wire (`make_prompt_chunk` applies the
  cursor itself, rather than patching two callers);
- write the test that makes the class **detectable** — *"every prompt-input path honours a
  non-zero cursor"* would have failed with **no GPU at all**.

⇒ **When a documented failure recurs, the deliverable is a structural change or a test, never a
better note.**

---

# D31 — CONFIG-DEPENDENT DISPATCH: the suite cannot fail on a branch no fixture's CONFIG reaches

Recorded 2026-08-18. Twice in one session, and **the second instance was found on hardware
while the whole suite was green.**

`CausalMasker` has **two** entry points. A new ragged channel was wired into one. Qwen2
reaches the other — because `qwen2.rs:30-46` deserialises `sliding_window` and **never reads
`use_sliding_window`**, so it loads `Some(32768)` even when HF says the window is off, and
`make_sliding_window_causal_mask_matrix` returns `None` at `tgt_len == 1`. Every ragged decode
batch then attended its zero-filled dead prefix. Symptom from outside: **433/512 tokens
returned with ZERO reported errors** — short, early-terminating generations, i.e. what wrong
tokens look like from the client.

**The suite was green throughout and could not have caught it**, because no fixture model
carried that config. This is not D25 (coverage without discrimination) — the line never ran at
all. It is a **fixture-space** gap, not an assertion gap, and coverage tools report it as
uncovered code that nobody notices among the rest.

**The rules:**
- **When adding a channel or a mask, enumerate every entry point AND every early return**, and
  ensure a fixture whose *config* reaches each. "I wired the dispatch" is not "every dispatch
  reaches it."
- **Suspect any behaviour gated on a deserialised config field.** `qwen3.rs:33`,
  `qwen3_moe.rs:33` and `qwen3_embedding.rs:28` gate on `use_sliding_window` correctly —
  three correct implementations next to one that doesn't is the signature.
- **Assume this gap is still open** unless someone closed it for *your* branch specifically.
  The chain that hit it closed it for the ragged channel only, **not for SWA generally**, and
  said so.

## Correction to the record — the completeness guard was REACTIVE, not foresight

Main wrote it up as the most valuable thing built in the session, implying it was designed
ahead of the failure. Its author corrected that: it was built **after** an arm printed
`ARM baseline OK` over a **dead server with 72 errored requests and exit 0**, and only because
they went back and read the token counts instead of the exit code.

**The transferable part is not "build guards."** It is:
1. the guard must resolve **cannot-answer** as a third state ([[D24]]); and
2. **it was verified RED against the saved bad JSON before being trusted** — including the
   subtlest case, `B=8 uniform: 0 errored, 8/512 tokens`.

*"A guard I hadn't seen fail would have been decoration."* That guard later killed a fake
2.84× that had zero reported errors and would have reached the founder.

---

# D30 — A NULL CONTROL GIVES YOU THE FLOOR FOR *THAT RUN*, NOT THE FLOOR. Interleave.

Recorded 2026-08-18. Corrects main's own guidance, which was wrong.

Main predicted that a noisy end-to-end floor would **collapse** on device time, and that if it
didn't, *"that is a finding about the instrument."* Measured:

```
run 1, span wall (dark timer):  floor 0.67%    effect 3.2%  -> 4.8x margin
run 2, device time:             floor 2.45%    effect 3.5%  -> 1.4x margin
run 2, span wall (same run):    floor 2.54%
```

**Device and wall floors track each other within a run (2.45% vs 2.54%) while both far exceed
run 1's 0.67%.** So the variance is in the *system* — GPU clocks, power, placement — not the
instrument. Sharpening the instrument could not have fixed it, and the floor varied **3.7×
between runs**. Run 1's comfortable 4.8× margin was a lucky draw.

**The rule:** one null control is **one sample of a floor that itself varies**. Treating it as
*the* floor is the same error as every other single-observation-as-property fault in this file.

## The fix — interleave, don't repeat

Running `U U U U F F F F` puts all between-window drift into the effect. Instead alternate at
the smallest cleanly-restartable unit and compare **paired**:

```
U F U N U F U N        not        U U U U  F F F F
```

Each treatment measurement sits against its temporal neighbours, so a clock excursion or
placement change hits both arms nearly equally and cancels. **Report the mean of the paired
differences and the spread of those differences** — that spread is the correct floor for a
paired design and is far tighter than any standalone null. Keep a null inside the interleave
so the floor is still measured under the same conditions rather than inferred.

Cheaper approximation when a full restart per arm is expensive: run the sequence in both
orders (`U F` then `F U`) and average. Cancels linear drift — most of it — for 2× the runs
rather than 8×.

## Separate the two claims

*Direction* and *magnitude* are different results with different confidence. Two independent
instruments agreeing on sign and size (3.2% wall, 3.5% device) establishes the regression
**exists**; a 1.4× margin over a floor sampled twice says little about **how big** it is. State
both: *"confirmed direction, ~3–3.5%, with a floor sampled twice and seen to vary 3.7×."*

And set the bar as **clears the floor with margin**, never merely clears the floor.

---

# D29 — THE THING YOU READ IS NOT THE THING THAT IS CURRENT. Re-read the reference, always.

Recorded 2026-08-17 after **three instances in one session**, each one a fact that was true
when observed and false when acted on. In a session where PRs are landing continuously, every
reference goes stale while you work.

| # | The stale reference | What it cost / nearly cost |
|---|---|---|
| 1 | a `min=1 → min=48` result attributed to the chain in front of me | wrong owner in a PR body; the requirement was nearly handed to a chain that could not act on it |
| 2 | a blocker scoped against a branch, while the fix had **already merged to master** | a subsystem reported as structurally blocked when it was gated on a flag |
| 3 | **`origin/master` on the box was `c7edf4e5a` (PR #77)** — a `git fetch` that only ever fetched the agent's own branch | an A/B that would have **attributed every merge since #77 to its own change**, producing a large, confident, entirely fake multiple |

**Instance 3 is the sharpest: nothing was broken.** The fetch worked, the build worked, the
benchmark would have run and produced a clean-looking number. The reference point was simply
old, and nothing in the pipeline is responsible for noticing that.

## The rules

- **Pin the baseline as a LITERAL SHA, and prove the pair.** "A commit you can name" is not
  enough — **a relative ref is still a moving ref.** `FIX~3` was correct at three commits and
  silently wrong at four, resolving to the agent's *own first commit* and putting half the
  change under test inside the "baseline". The version that cannot rot:
  `BASE=<sha> FIX=<sha>`, verified with `git merge-base --is-ancestor`, printing **the commit
  count under test** so the arithmetic is checked rather than trusted.
- `origin/master` is whatever your last fetch left behind — and `git fetch <branch>` does not
  update it.
- **Verify it in the script, not in your head.** `git log` the resolved SHA into the run's own
  output so the artefact carries its own baseline provenance. The agent that caught this added
  exactly that.
- **`git fetch <branch>` does not update `origin/master`.** On a long-lived box this is the
  default state, not an edge case.
- **Re-read master before scoping anything.** Especially mid-session with merges in flight:
  a blocker you found an hour ago may have been fixed forty minutes ago.
- **Attribution is a reference too.** Check who authored the commits before crediting work —
  the chain in front of you is not thereby the chain that did it.

Related: D23 (*explains ≠ verified*) — this is the same substitution applied to **freshness**
rather than causality. A reference that was correct is not thereby a reference that is correct.

---

# D28 — CHECK THE WHOLE POPULATION BEFORE BELIEVING ONE READING. Plus two tool footguns.

Recorded 2026-08-17.

## The method that works: profile-wide zero vs per-node zero

An agent measuring one span found `device_ns == 0` and — instead of concluding the span was
fast, or that its change had no effect — **checked every node in the profile. All 150 were
zero.** That is diagnostic in a way a single reading can never be:

> A per-node zero cannot distinguish *"this is fast"* from *"the instrument is off."*
> A population-wide zero can only mean the latter.

**Generalises past profilers to any instrument reporting many measurements at once** —
counters, timers, coverage, telemetry. Before believing one number from a multi-reading
instrument, look at the distribution. An implausibly uniform population is the instrument
talking about itself, not about the system.

Root cause in this case was D18's dark device timer: `device::timer_for` rejected a **NULL
`CUstream`**, which is CUDA's *legacy default stream*, not "no stream". Fixed on
`perf/profile-ragged-padding` by **probing instead of guarding**, proven with
`selftest ratio 24.6–44.2× PASS` and `unresolved_device_spans: 0`. **The fix is branch-local
— a chain measuring on another branch inherits the dark timer** and must cherry-pick it.
*An instrument fix is not a property of the repo until it is merged.*

## Footgun 1 — selecting a span by name can select the wrong one

There are **two** `mla_attn` nodes: `step.prompt…` (`calls=0`) and `step.decode…`
(`calls=2752`). Matching on name takes the first, i.e. prefill. The guard voided on
`calls=0` — **right outcome, wrong diagnosis**: it reported "unreached node" when the truth
was "wrong node". Select on the full path and require `calls > 0`. The profiler's `Node`
already carries a `reachable` flag documented as *"a zero from an unreached node and a zero
from a fast node are different answers and must not look the same"* — the distinction was
already solved in the schema; two chains hand-rolled it before reading it.

## FIXING A CONFLATION: rename one of the two quantities, don't just correct the uses

2026-08-18. `DecodeBuffers::batch_size` meant *allocated capacity* in one place and *sequences
served this step* in another. Correcting the uses would have left the next reader free to
re-conflate them. Renaming it to **`capacity`** turned the semantic bug into a **compile error
at all 13 call sites**, forcing each to be reconsidered rather than silently keeping the old
meaning.

**A rename is enforced by the compiler; a comment, an assertion, or a careful fix is not.**
Reach for it first whenever two quantities have been mistaken for each other.

Related, from the same change — **encode the policy as arithmetic in the test**: assert the
over-read is exactly `4,254,208` bytes rather than "reads too much"; walk a ladder that *dips*
back down to prove no shrink-thrash; pin the reallocation **count** (4 across 9 steps) so
growth is per high-water-mark, not per step. Falsifiable by a number beats falsifiable by a
judgement.

And **when two fixes land together, a single revert cannot attribute** — pre-register one
mutation arm per fix (M1: revert only A, M2: revert only B) so the surviving failure names
which one was load-bearing.

## The INVERSE of silent success — SILENT PROGRESS

Same defect pointed the other way, and arguably worse because it burns wall-clock instead of
producing a wrong number. A watcher reported **"BUILD STILL RUNNING"** about a build that had
died four minutes earlier — because `pgrep -f "cargo build …"` matched its own ssh command
line, so the check **could never return false**. Its author's framing:

> my liveness check could not distinguish "running" from "dead", and it defaulted to the
> reassuring answer

**Why this footgun is nastier than it looks: it fails safe-LOOKING in BOTH directions.**
An unbounded `until ! pgrep …` loop never exits (reads as *"still building"*); a bare
`pgrep && echo RUNNING` always says RUNNING (reads as *"healthy"*). **Neither presents as an
error**, so no exit code, no stderr, nothing to grep for.

**Wait on the ARTEFACT, never on a process name.** Binary exists with an mtime, or the log
carries an error line — a state with no ambiguous mode. Same rule as *believe the artefact,
not the advertisement* (`nvcc` accepting a flag it does not honour): a name is a claim, an
artefact is evidence.

**Then check the artefact's AGE.** Exit 2 if the binary is older than the source it claims to
be built from. A stale binary produces a **clean, flat result indistinguishable from an honest
negative — and biased toward "no effect."** Staleness does not add noise; it argues for your
null hypothesis.

## `set -e` DOES NOT SURVIVE A PIPE — and the fix is not `pipefail`

A guard printed **"PAIR PROVED" with 0 commits under test**, because `git am … | tail -2`
returns **`tail`'s** status, so `set -e` saw success. **Every `cmd | tail` / `| head` in a
guard has this property** — and piping to `tail` to keep output short is exactly what a
careful author does.

`set -o pipefail` only helps if you also stop swallowing the code. **The fix that generalises
is to print the QUANTITY, not the verdict.** `PAIR PROVED` is unfalsifiable by inspection;
`commits under test = 0` is refutable by anyone skimming the log, without re-running anything.

> **A guard must print the number it is asserting on, in the same breath as the verdict.**

## Footgun 1b — `cargo fmt -p <crate> -- <files>` IGNORES THE FILE ARGS and formats the crate

**Recorded 2026-08-18. This corrects main's own instruction, which was insufficient.**
Main had been telling every chain *"`cargo fmt -p <crate>`, never `--all`"*. A chain ran
`cargo fmt -p mistralrs-core -- <2 files>` and it reformatted **87 files** — the exact
fork-policy violation the rule exists to prevent (`fab114fe3` had to revert one already).

**The safe form is `rustfmt <path/to/file.rs>` directly — no cargo.**

Verification technique worth copying: that chain proved it added **zero** formatting noise by
showing master's `qwen2.rs` is itself **not** rustfmt-clean (5 pre-existing diffs at
187/348/518/539/656) and that its version carried the same 5 and no new ones.

## Footgun 2 — `pkill -f <pattern>` matches its own command line — THREE chains, one session

**Bit two independent chains in one session.** `pkill -f arcflash_span_ab` killed the parent
shell running the cleanup, before the relaunch could execute; the ragged-window chain hit the
identical thing. `pkill -f` scans the command line of *every* process including the one doing
the killing.

**The fix is the recorded PID (`$!`), not a cleverer pattern.** Every "unique token" is one
refactor away from appearing in a wrapper's command line, an `ssh` invocation, or a log line.
Both chains that hit this already captured `$!` in their main scripts and were bitten in an
**ad-hoc cleanup** — which is exactly where the discipline lapses, and exactly where D24's
"the stakes look lowest" observation applies.

Related: [[D24]] (three states), D18's verification-code sub-pattern.

---

# D27 — A MARGINAL COST AND AN ABSOLUTE COST ARE NOT COMPARABLE. Check the SHAPE of a number.

Recorded 2026-08-17, caught by an agent *before* it published the error.

The profiler measured `mla_attn` at **11.0 ms of the 19.4 ms marginal cost per additional
sequence (57%)** — a **slope**, `d(step time)/dB`, obtained by differencing across batch
sizes. The ArcFlash chain was about to measure an **absolute** `mla_attn` time from a single
fused-512 run and report it against that figure. Its own words: that would be *"reporting an
absolute against a slope and calling it a comparison."*

Correct method: measure marginally too — same span, two batch sizes, take the difference.

**The rule: before comparing two numbers, confirm they are the same KIND of number.**
Recurring shapes that get mixed here:
- **slope vs level** — per-additional-sequence cost vs total cost at one B.
- **instantaneous vs aggregate** — `IntervalLogger` printed `tps=474.20` while the harness
  measured ~12 tok/s aggregate; the logger samples momentarily **and counts prompt tokens as
  throughput** (`engine/mod.rs:583`), so engine-reported throughput is not comparable to
  harness aggregate anywhere.
- **wall vs decode-only** — 3.10× wall and 3.32× decode-only for the same A/B.
- **device vs host vs sync** — only separable once the device timer works at all.

Corollary: **state the shape in the units.** "11.0 ms/sequence (marginal, from B=1→32)" is
unmixable; "11.0 ms attention" invites the error. Every number entering [[facts]] carries its
shape and the range it was derived over.

Same family as D23 (*explains ≠ verified*): a number that *looks* comparable is not thereby
comparable, and the check is cheap.

---

# D26 — A CAPABILITY ALLOW-LIST SHARED ACROSS BACKENDS IS A TRAP

Recorded 2026-08-17, from the ArcFlash head_dim-512 work. Caught before push, by the agent
that introduced it.

```rust
let flash_sinks_ok = matches!(hd, 64|80|96|112|128|192|256);
if q.device().is_cuda()  && flash_sinks_ok { ...CUDA kernel... }
if q.device().is_metal() && flash_sinks_ok { ...Metal kernel... }
```

**One flag, two backends.** Adding `512` for CUDA **silently widened Metal**, whose sinks
kernel stops at 256 (`metal_kernels/mod.rs:3025` errors on any other head_dim;
`sdpa_with_sinks.metal` instantiates only `{64,80,96,128,256}`). A 512 head on Metal would
have gone from a **working unfused fallback** to a hard `CompilationError` — a regression
introduced by a change that reads as entirely correct at the call site you're editing.

**The rule: each backend advertises its own envelope.** Never let one predicate gate
dispatch to independently-compiled kernels. The shared list looks like DRY; it is actually
an unstated claim that every backend supports the same set, which nothing enforces and which
silently becomes false the first time one of them gains a capability.

**How it was caught, and this is the transferable part:** the agent went to verify Metal's
limit *before writing a comment asserting it*. The check it nearly skipped is the one that
found the bug. Same shape as D23 — the cheap verification deferred past the expensive step —
but here the trigger was **writing down a claim**, which forced the check. Stating a fact
you have not verified is uncomfortable enough to be useful; treat that discomfort as a prompt.

Corollary for widening ANY allow-list: enumerate every consumer of the predicate before
editing it, and confirm each independently supports the new value. Count the call sites; the
ArcFlash change touched **six** gating sites (two CUDA `switch`es and their `default` strings,
two Rust bails, two `flash_sinks_ok` gates) — a "fix" that updates four of six reads as
landed while the path stays dark.

---

# D25 — COVERAGE IS NOT DISCRIMINATION. A test can execute fully and be unable to fail.

Recorded 2026-08-17, from #95's mutation review. Distinct from every D18 instance: there the
code didn't run, or ran and defaulted to success. **Here the code ran, the assertion
evaluated, and it still could not have failed — because the fixture's data made the passing
and failing states byte-identical.**

The `xs` fixture was **zero-filled**. With zeros, "read the right tokens" and "read the wrong
tokens" produce the same bytes. So a test with **full path coverage** was blind to the defect
it existed to catch. Coverage measured *execution*; nothing measured *discrimination*.

Found only by running **both** mutations:
- tensor-trim → caught by the width assertion.
- logical-base-only → produces **every shape exactly right**, and is caught **only** by the
  per-row read. Under the zero-filled fixture, not caught at all.

The surviving mutation's cause was then named precisely — *"the fixture's data made two
different states indistinguishable"*, **not** *"the fixture couldn't reach the condition"* —
and the doc's overclaim was corrected rather than left standing. Naming it at that
granularity is what makes it fixable: the remedy is a new fixture sub-shape, not more paths.

**The rule:** a test's fixture must make the failure it targets **observable**. Before
trusting a green:
- **Mutate the defect in and confirm red.** A test never seen red is decoration (D18
  sub-pattern) — and this is the case where reading the test carefully still won't tell you.
- **Ask what the fixture's values erase.** Zeros, all-equal rows, uniform lengths, identity
  matrices and powers of two all collapse distinctions. If a wrong answer and a right answer
  render identically under the fixture's data, the test cannot see the bug however many lines
  it executes.
- **Coverage tools cannot detect this.** They report the line ran. Only mutation can report
  the assertion mattered.

## A THIRD trap, distinct from both — A CORRECT TEST OF THE ADJACENT THING

2026-08-18. Not a vacuous test. Not a fixture that erases the property. **A genuine test, red
for a real reason, pinning a real property — aimed one level too low to catch the failure.**

`a_non_zero_cursor_feeds_only_its_chunk_on_every_path` went red on a laptop in 4 seconds with
a message naming the exact defect. It was written *before* the fix, watched fail, and it
passes now. It is a good test. **And the bug it was written to catch was still there** — three
GPU runs later the mechanism still never engaged.

**The give-away, stated by its author, is checkable by inspection:**

> it passes the cursor in by hand, which is precisely the half that already worked

⇒ **A test that SUPPLIES the input the broken path is supposed to PRODUCE cannot detect that
the path fails to produce it.** It verifies the consumer while the producer is the defect.

**The check:** for any test of a mechanism, ask *which end of the wiring does this construct by
hand?* That end is not under test. If the failure could live there, the test is aimed one level
too low — move the entry point up until the path derives the input itself.

Same session, same chain: it also **withdrew a root cause** it had inferred from a grep of call
sites without reading what the call site was (`make_prompt_chunk(0, …, &load_device, …)` with a
synthetic BOS chunk is a **load-time warmup path**, not the serving path). ⇒ **grep finds call
sites; only reading tells you what they are.**

## The Rust-specific instance — `let (tx, _rx) = channel();` closes the channel immediately

2026-08-18, and it had poisoned **an entire test suite**. `seq_of_len` wrote
`let (dummy_sender, _rx) = channel(1);`. **`_rx` is a real binding, not a discard** (only bare
`_` discards), so it drops when the helper returns and the sender is closed from that moment.
⇒ **Every sequence the scheduler suite ever built carried an already-closed responder.** The
suite modelled a disconnected client in **100%** of its fixtures and could not have
distinguished a live client from a dead one. A correct new reap therefore looked like it
*broke* a passing test when it was in fact revealing one.

**Grep the tree for `, _rx)` and `, _tx)`** — same shape, same silence. Fix with a
thread-local (`LIVE_CLIENTS`) that holds each receiver for the fixture's lifetime.

## And the observers' version — greping for a string that only exists on the error path

The same defect was "corroborated" by **ZERO disconnect warnings** in the logs. There was no
such line to find: the only "disconnected" warning (`utils/mod.rs:39`) is reached solely when
reporting an *error* to a departed client, and the normal path emits nothing. **Two chains
searched for something that could not have been there and read its absence as evidence.**

Compounding it: `<N> running, <M> waiting` (`engine/logger.rs:83`) is **suppressed unless
`total_new_seqs != 0 && tokens_processed != 0`** — so an idle server holding phantoms prints
**nothing at all**, and a chain watching during a lull concludes the engine is clean.

⇒ **Before treating a log's silence as a signal, confirm the line exists on the path you are
watching and is not conditionally suppressed.**

Related: [[D24]] (a guard has three states), D18's verification-code sub-pattern, and D23
(*explains ≠ verified*) — this is that pattern in test data rather than in reasoning.

---

# D24 — A GUARD HAS THREE STATES, NOT TWO. And it can fail in EITHER direction.

Recorded 2026-08-17 after **two guards in the same script failed in opposite directions in
one run** — the sharpest illustration yet of why D18's discipline has to point both ways.

| | The guard | How it failed |
|---|---|---|
| **False clean** | tunable-engagement check | grepped its own output for `abort`; its success sentinel is the literal string **`NO_ABORT`**, which contains `ABORT`. Reported `✅ tunables ENGAGE` when they were provably inert. Visible only because the two control lines were byte-identical. |
| **False fail** | leg-validity check | labelled the treatment leg `UNPROVEN (0 tokens)` — but the zero tokens **were the consequence of the corruption under test**. It read the symptom of a true positive as a missing measurement. |

**Both had the same cause: each encoded exactly ONE expected failure shape.** The first
assumed failure looks like the word "abort" appearing; the second assumed a real result
always produces tokens.

**The rule:** a guard must resolve **three** states and never collapse them —
1. **pass** — the property held, and the instrument was shown able to say otherwise;
2. **fail** — the property was observed violated;
3. **cannot answer** — the instrument was inert, the leg didn't run, the sample was empty.

And two corollaries earned the hard way:

* **Never match a sentinel by substring.** Anchor it (`^NO_ABORT$`). An all-clear message
  that contains the failure keyword is not a hypothetical — it shipped.
* **Do not treat a symptom of the failure as absence of evidence.** If the thing under test
  kills the run, "no output" is the *result*, not a missing one. Ask whether the null could
  have been *caused* by the defect before recording it as unprovable.

**Engagement must be a behavioural difference, not the presence of a word.** The corrected
control runs the same deliberate fault with and without the instrument and requires the two
runs to *differ*; if both abort or neither does, it reports **cannot answer**.

Related: D18 (silent success), and its verification-code sub-pattern — this is that pattern
with the second direction added.

---

# D23 — A DECOMPOSITION IS ONLY EVIDENCE IF IT PREDICTS A VALUE IT WASN'T FITTED TO

Recorded 2026-08-17. An agent explained three observed allocation sizes as
`512 × {120, 168, 256} × 2`, read 512 as head_dim, and concluded "those are KV cache
capacities". Main endorsed it as the strongest evidence in the investigation. A fix was
built on it, measured, and **changed nothing** — because the buffer was misidentified.

The correct reading, from the same numbers: `4096 × {18, 19, 20, 21}` — hidden_size width,
**consecutive integers, one per decode step** — and `21` equalled `prompt(10) + warmup(8) +
deferred(3)` exactly. It is the `xs` hidden-state history, not the KV cache.

**The mechanical trap:** `512 × 120` and `4096 × 15` are *the same number*. Any integer
factors many ways, so a factorisation that merely reproduces the observed values has zero
information content — it cannot fail. The first decomposition's `{120, 168, 256}` matched no
grow pattern; the second's consecutive integers are a mechanism.

**The rule:** a decomposition, factorisation or numeric "explanation" counts as evidence only
when it **predicts a value it was not fitted to** — the next allocation, a size under
different settings, a count derivable independently. Until then it is a restatement of the
data wearing a causal costume.

Corollaries:
- **State the prediction before looking.** "If this is the KV cache, changing max_seqs
  changes it; if it's the hidden history, changing prompt length changes it." One cheap run
  discriminates; both fits survive arithmetic alone.
- **Endorsing a fit propagates it.** Main called this the strongest evidence in the
  investigation, which is how it reached a build. **An endorsement is a claim of its own** —
  check the arithmetic before amplifying, because downstream agents treat main's
  endorsement as settled.
- **The arithmetic was checkable the whole time.** It was checked only after a fix had been
  built on it. Cheap checks deferred past the expensive step is the recurring shape.

**The positive form — what to DO, not just what to avoid.** The prohibition alone leaves the
next agent stuck, so state the method:

> **Fit the decomposition, then make it predict a value it was not fitted to — the next
> allocation, the size under a changed setting, a count derivable independently — and
> believe it only after it survives that.**

`4096 × {18,19,20,21}` became evidence precisely because it predicted the series **across
runs it wasn't fitted to**, and because `21` was independently derivable as
`prompt(10) + warmup(8) + deferred(3)`. Two independent confirmations, neither available to
the `512 × {120,168,256}` reading, which explained its three numbers and nothing else.

## The general form — "EXPLAINS" IS NOT "VERIFIED". Three instances in one session.

D23 started as a rule about decompositions. One session produced three instances of the
same substitution, in three different shapes, so state it generally:

| # | The claim | What fit | What was never checked |
|---|---|---|---|
| 1 | `512 × {120,168,256}` = KV cache capacities | three observed sizes | that any integer factors many ways; `4096 × {18..21}` fits the same data and is the truth |
| 2 | heap corruption = pool destroyed while cache holds its pointers | the *shape* of all evidence — capture-only, varying diagnostic, moving abort site | whether the trigger **fires on the failing path**. `cuMemPoolDestroy` count in the crashing log: **0**. The fix's six drains never ran; the A/B never tested it |
| 3 | the `min=1 → min=48` result belongs to the chain I'm talking to | the chain was in front of me and the detail fit | who actually authored the commits |

**The substitution:** a hypothesis that *accounts for* the observations is treated as
*established by* them. Every one of these had a cheap available check that was skipped —
factor it the other way; grep the log for the trigger; read the commit author.

**The rule:** before acting on an explanation, name the observation that would be
**different** if it were false, and go look. If no such observation exists, the explanation
is a restatement, not a finding.

**Main's version of this fault is more expensive than an agent's.** An endorsement converts
a chain's tentative claim into the project's position, and downstream chains then treat it
as settled. Instance 1 reached a build because main called it "the strongest evidence in the
investigation" without redoing the arithmetic. **Check before amplifying; state confidence
explicitly; "an agent reports X" is not "X".**

Corollary on retractions: **a retraction that overshoots is still a false record.** Instance
3 was corrected to "nobody owns this requirement" when an owner had in fact been assigned.
Correct to the truth, not past it.

Same family as D18 (absence read as signal) pointed at *inference* rather than *status*:
here a **non-discriminating observation was read as a discriminating one**.

---

# D22 — `/root/locks/gpu.lock` IS NOT AN OCCUPANCY SIGNAL. Gate on `nvidia-smi`.

Recorded 2026-08-17 after it produced a wrong call by main and a correct refusal by an agent
on the same night.

**The lock is opt-in, and almost no chain takes it.** Measured on both live boxes:

| Box | Resident process | Lock state |
|---|---|---|
| `arc-prefill-curve` | `/root/wt-profkill/…` port 1239, 77 GB | **FREE** |
| `arc-graph-probe` | `/root/arc-batchinv/…` port 1240, 81 GB | **absent** |

Two different chains, two different worktrees, **neither holding the lock while both held
80 GB of GPU.** So `gpu.lock` is worthless in *both* directions: "FREE" does not mean idle,
and an absent lock file does not mean idle either. It only ever means *"the one chain that
opts in is not currently running"*.

This is the D18 mechanical form again — **the absence of a signal read as a specific
signal** — but sitting in shared infrastructure rather than in one tool, so every chain that
trusts it inherits the bug.

**Consequences already paid:** main read the lock as FREE while a server was running and
nearly acted on it. Separately, a lock cleanup deleted a *live* lock on an unread assumption
(D18 #14 instance 4). And an ownerless empty `gpu.lock` was chased twice as a corruption
mystery — there was no mystery: that is simply what the file looks like when nobody opted in.

**The rule:** occupancy is `nvidia-smi --query-compute-apps=pid,used_memory --format=csv`.
Never the lock. A measurement gate must refuse on a non-empty process list and exit **2**
(environment could not answer), not 1 — timings taken beside an 80 GB neighbour are not
attributable, and they look completely normal.

Corollary: this makes exclusivity a *convention*, not a mechanism. Until every chain opts in,
**assume a box is shared** and gate on the hardware, not on cooperation.

---

# D18, instance 15 — `grep -c` over a log that does not exist

Same night, caught before relay. A probe queried `MISSES: 0` and was about to report it as
the mechanism passing. The grep ran over a **server log that was never created** — the leg
had not run. Zero matches in a nonexistent file is indistinguishable from zero misses in a
successful run, and it happened to be **exactly the value the chain wanted to see**.

Caught only because the chain summary was empty and the status line had not moved in ten
minutes — i.e. by two *independent* liveness signals, not by the check itself.

**Rule, narrow enough to apply:** before grepping for a count, assert the file **exists and
is non-empty** as a separate step; and prove the counter can print **non-zero** by mutating
the condition it counts. *A zero that has never been observed non-zero is decoration.* This
is the same shape as the truncated-forward retraction and belongs with the verification-code
sub-pattern above — it is instance 8 of *that* family, and the tool was again one written to
catch exactly this.

---

**Settled for free, do not re-derive on a rented box:** register-A `wgmma` takes **7
operands** — `d, a, b-desc, scale-d, imm-scale-a, imm-scale-b, imm-trans-b`, **no
`imm-trans-a`** (a register operand has no layout to transpose). Established by making
ptxas discriminate: 7 operands ⇒ *"not supported on .target 'sm_90'"* (arguments
**validated**, only the target objected); 8 operands ⇒ *"Arguments mismatch"* (arguments
**rejected**). This is what PTX ISA §9.7.16.5.2 would say if its body weren't truncated in
the published HTML.

---

# D18, corollary — THREE WAYS A CHANGE READS AS COMPILING WHILE BEING BROKEN

All three were hit in one session (2026-08-17). Each produces a confident green from a
command that genuinely succeeded — the command was simply narrower than the claim being
made from it. Check all three before saying "it compiles".

| # | The narrow command | What it cannot see | How it was caught |
|---|---|---|---|
| 1 | `cargo check` on **macOS** | Every `#[cfg(feature = "cuda")]` arm. Not "might miss" — the arms are not compiled at all. | #99 shipped a missing `qtip_grouped_tile_m` re-export; only the nvcc lane found it. |
| 2 | `cargo check -p <crate>` | **Tests, examples and benches.** `-p` builds the lib target only. | #100's `E0063` (`missing field row_q0`) passed `cargo check -p mistralrs-core` and failed the lib-**test** build in CI. Only `cargo check --workspace --tests` reproduced it. |
| 3 | *(no command at all)* | A stacked PR — base ≠ `master` — matched no workflow trigger, so **zero lanes ran** and one cosmetic check rendered green. | D18 #11; fixed by `branches: ['**']` + `ci-complete`. |

**The command to actually run before claiming a change builds:**

```
cargo check --workspace --tests          # catches (2)
```
…and for anything touching cuda-gated code, **a box compile or the nvcc CI lane** — there
is no local substitute on macOS for (1).

Note the project guidance ("always run `cargo check`/`cargo c` before returning") is
**known-insufficient** as written: it is necessary, not sufficient, and satisfying it
literally is how (1) and (2) both shipped.

## The generalisation

**A green is scoped to the command that produced it, never to the claim you want to make
from it.** Before reporting "it compiles / tests pass / CI is green", state which command
produced that green and ask what it structurally cannot reach. Every instance above is the
D18 mechanical form once more — *absence of a signal read as a specific signal* — with the
absence created by the narrowness of the check rather than by a defaulting branch.

**Worked example, recorded because it happened inside the monitoring built to prevent it:**
a poll loop watching #100 computed only the *pending* count, having dropped the *failure*
count, and printed `pending=0`. The run had a failed test lane and two cancelled sibling
matrix legs. `pending=0` was true, and read as "ready to merge". The required `ci-complete`
check caught it; the human watching the summary did not. **When monitoring a job, every
poll must compute a failure term, not only a completion term** — and this is why a required
status check outranks a human reading lanes: the human reads a summary, and the summary is
where the defaulting lives.
