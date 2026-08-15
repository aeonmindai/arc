# Arc v2.0 release checklist (DRAFT — nothing here is executed yet)

Exact steps to cut the release. **Every item under "Requires Jish's
approval" must be explicitly approved by Jish before anything becomes
public — no tag, no binary, no HF upload, no announcement happens without
that sign-off.** The steps are written so an agent can execute them
mechanically once approved.

## 0. Requires Jish's approval (the go/no-go list)

| # | Decision | Options / default |
|---|---|---|
| A1 | **Release timing**: ship v2.0 on today's numbers, or hold for session 5's end-to-end batched-serving number (the serving-level $/Mtok) | Notes are written to be honest either way; holding upgrades the fleet story from kernel-level to serving-level |
| A2 | **Release notes content**: [docs/RELEASE_NOTES_v2.0.md](../RELEASE_NOTES_v2.0.md) as drafted (incl. limitations wording) | Approve / edit |
| A3 | **Version string**: `v2.0.0` tag name | Or `v2.0.0-rc1` first |
| A4 | **HF org + repo name** for the pre-baked UQFF upload | e.g. `<org>/DeepSeek-V4-Flash-UQFF` — org must be Jish's call |
| A5 | **When the HF repo flips public** (upload is `--private` first — see §4) | At announcement time |
| A6 | **Binary matrix scope** (§3): which variants actually get built for v2.0 vs "source build" fallback | Minimum viable: `linux-x86_64-cuda-hopper` + `macos-arm64-metal`; install.sh falls back to source for the rest |
| A7 | **Announcement channels and copy** (§6) | Each channel + each text needs sign-off |
| A8 | **License/attribution check**: NOTICE, LICENSE-APACHE (arc-*), LICENSE-MIT (mistralrs-*, FlashMLASparse) ship as-is | Approve |

## 1. Preconditions (repo state)

- [ ] PR #20 (sweep winners as dispatch defaults, cudnn ban, boot fixes, MTP
      KV-desync fix) **merged** — the release is wrong without it: v2.0
      claims the tuned dispatch defaults.
- [ ] PR #21 (per-sequence `xs_history`) — merge if green; it is *not* a
      release blocker because v2.0 claims no voting numbers, but shipping
      the crash fix is strictly better. If unmerged, the release notes'
      "voting is off" wording already covers it.
- [ ] This docs PR (sessions 3+4 fold + release notes draft) merged.
- [ ] CI green on `master` at the release commit (all lanes, incl. the two
      nvcc no-GPU lanes sm_80/sm_90).
- [ ] `cargo test -p mistralrs-core -p mistralrs-quant` clean at the release
      commit.
- [ ] Version bump: confirm the crate/CLI version reported by `arc --version`
      matches the tag (update `Cargo.toml` workspace version if not).
- [x] Docs consistency sweep: README.md carried session-2-era numbers
      (84.0%, 5.4 tok/s, ≈$77); reconciled to the BENCHMARKS.md current state
      (87.0%, 14.58 tok/s, ≈$123) in `docs/correct-kernel-stack-bound`.
- [ ] **Provisional labels intact.** GSM8K 87.0% / perplexity 12.50 /
      long-context were measured on pre-PR-#35 decode math and are labeled
      **provisional** in README.md, BENCHMARKS.md, FLEET.md and
      RELEASE_NOTES_v2.0.md. Either re-measure on post-#35 math before tagging,
      or confirm every one of those labels survived the freeze. Do not tag with
      the labels stripped and no re-measure.

## 2. Tag (only after §0 sign-off)

```bash
git checkout master && git pull
git tag -a v2.0.0 -m "Arc v2.0.0"   # annotated tag on the approved commit
git push origin v2.0.0
gh release create v2.0.0 -R aeonmindai/arc \
  --title "Arc v2.0" \
  --notes-file docs/RELEASE_NOTES_v2.0.md \
  --draft        # stays a GitHub draft until Jish flips it
```

The GitHub release is created as a **draft**; publishing it is part of the
announcement step (§6), not this one.

## 3. Prebuilt binary matrix

`install.sh` resolves assets from the latest GitHub release by variant name
(`{os}-{arch}-{variant}.tar.gz` containing the `arc` binary); anything
missing falls back to a source build, so an incomplete matrix degrades
gracefully.

| Asset variant | Target | Build features | Notes |
|---|---|---|---|
| `linux-x86_64-cuda-hopper` | H100/H200 (sm_90) | `cuda flash-attn` | The one that matters for the fleet story |
| `linux-x86_64-cuda-blackwell` | B200 (sm_100) | `cuda flash-attn` | Only if a Blackwell build box is available |
| `linux-x86_64-cuda-ada` | RTX 40xx / L40S (sm_89) | `cuda flash-attn` | Optional for v2.0 (A6) |
| `linux-x86_64-cuda-ampere` | A100 / RTX 30xx (sm_80) | `cuda flash-attn` | Optional for v2.0 (A6) |
| `linux-x86_64-cpu` | any | (none) | Cheap to build; include |
| `macos-arm64-metal` | Apple Silicon | `metal` | Build on macOS arm64 |

**WARNING — no `cudnn` in any CUDA release binary.** The `cudnn` feature
costs −62% decode on V4 (5.45 vs 14.58 tok/s, measured session 4). Every
CUDA asset above is `--features "cuda flash-attn"` and nothing more. A
release binary accidentally built with `cudnn` would ship the regression to
every user; check the feature list in the build log before packaging each
asset.

Packaging per asset:

```bash
cargo build --release -p arc-cli --features "<features per table>"
tar -czf arc-<variant>.tar.gz -C target/release arc
gh release upload v2.0.0 arc-<variant>.tar.gz -R aeonmindai/arc
```

Smoke test each asset: `./arc --version` on a matching box, plus
`arc validate --target-hbm 141 --model deepseek-ai/DeepSeek-V4-Flash --mock`
on the CUDA ones.

## 4. HF pre-baked UQFF upload

The `arc quantize` command's directory mode already generates the model-card
README and prints the exact upload command (its own hint):

```
hf upload <org>/DeepSeek-V4-Flash-UQFF <output-dir> --repo-type model --private
```

Plan:

1. Source of the artifact: **reuse the session-4 bake** (68 GB, 7 shards,
   `--isq qtip2` 2-bit trellis experts + FP8 attention, baked with the
   Viterbi default) if the session tarball is intact; otherwise re-bake on a
   rented H200 (~25 min build + ~24 min bake at 30 s/layer, ~$5 of GPU time)
   using the runbook-4 flow **without cudnn**.
2. Run `arc quantize` directory mode (or `--base-model`/`--repo-id` flags to
   skip the prompts) so the generated README names the base model
   `deepseek-ai/DeepSeek-V4-Flash` and the target repo from A4.
3. Upload with the printed `hf upload ... --private` command. The repo
   stays **private** until A5 says otherwise.
4. Model-card additions before flipping public: the BENCHMARKS.md headline
   table, the UQFF-0.2.1 reader requirement ("requires Arc ≥ v2.0 — older
   readers mis-decode rank-3 payloads"), and the serve line including
   `--chat-template chat_templates/deepseek_v4.json`.
5. Verify from a clean box: download via `arc serve -m <org>/DeepSeek-V4-Flash-UQFF`
   (or the documented `--from-uqff` path) and run the coherence probe before
   announcing.

## 5. Docs freeze check (same commit as the tag)

- [ ] README.md numbers == BENCHMARKS.md numbers (87.0 / 14.58 / ≈$123 /
      grouped-GEMM curve labeled kernel-level).
- [ ] FLEET.md tags intact ([measured] / [measured-kernel] / [projected]).
- [ ] RELEASE_NOTES_v2.0.md drops its DRAFT banner (single-line edit) —
      only at tag time, per A2.
- [ ] `docs/UQFF.md` mentions 0.2.1 rank-3 payloads and the reader
      requirement.

## 6. Announcement (all copy needs A7 sign-off)

Channels, in proposed order:

1. **GitHub release** — flip the §2 draft to published.
2. **HF model repo public** (A5) — the model card is the de-facto landing
   page for practitioners.
3. **X/Twitter thread** — the one-GPU 284B story + the honest
   measured-vs-projected framing; numbers only from BENCHMARKS.md.
4. **Runcrate site/blog** (runcrate.ai/arc is already the README's website
   link) — longer-form post.
5. Optional: HN "Show HN", r/LocalLLaMA — only if Jish wants the attention
   this week.

Rule for all copy: every number must exist in BENCHMARKS.md with its
protocol, kernel-level figures are always labeled kernel-level, and nothing
quotes the projected columns as achievements.

## 7. Post-release

- [ ] Open the session-5 tracking issue: end-to-end batched serving tok/s +
      $/Mtok, tuned-dispatch validation, MTP acceptance, voting after PR #21
      hardware validation — these are v2.1's headline candidates.
- [ ] Watch the first 48h of issues for artifact-download and
      chat-template-missing reports (the two known footguns).
