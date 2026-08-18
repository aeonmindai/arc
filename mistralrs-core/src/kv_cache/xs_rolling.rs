//! Rolling compressed state for the DeepSeek V4 compressor input (`xs`).
//!
//! # Why this exists
//!
//! V4's CSA/HCA layers feed a *compressor* (`V4Compressor::forward_from_xs`)
//! the history of the attention-block input, and use its output as a second,
//! distant-context key set inside the same softmax. The first correct
//! implementation (R3, PR #21) kept that history verbatim: one
//! `[B, T, hidden]` buffer per compressed layer, re-running the whole
//! compressor over the whole history on every decode step.
//!
//! That buffer, not the KV cache, is what caps batch size. Per token per
//! layer it is `hidden * 2 = 8192 B` against the KV's
//! `head_dim * 2 * 2 = 2048 B` — 4.0x — and 41 of V4's 43 layers carry one.
//!
//! # What the compressor actually needs
//!
//! `forward_from_xs` is a *strided local reduction*: output row `j` is a
//! softmax-weighted pool over the `ratio` tokens of group `j` (plus, when
//! `ratio == 4`, the `ratio` tokens of group `j - 1`, via
//! `overlap_transform`), followed by an RMSNorm. Row `j` is therefore final
//! the moment token `(j + 1) * ratio - 1` arrives and never changes again,
//! and the *only* raw tokens a future row can need are the last
//! `span_groups * ratio` of them.
//!
//! So the raw history is not state — it is a recompute buffer. This cache
//! keeps the state instead:
//!
//! * `comp` — the completed compressed rows, `[B, G, head_dim]` with
//!   `G = tokens / ratio`. `head_dim / ratio` bytes per token per layer
//!   instead of `hidden`: `8 * ratio` times smaller (32x at `ratio = 4`,
//!   1024x at `ratio = 128` for V4's `hidden = 4096`, `head_dim = 512`).
//! * `tail` — the raw rows behind `tokens`, bounded by
//!   `span_groups * ratio + margin` and independent of context length.
//!
//! # The contract with the rest of the engine
//!
//! Every cache entry in a `NormalCache` is truncated in **token** units by
//! the prefix cacher (`prefix_cacher.rs:316-321`), the MTP verify rollback
//! (`pipeline/mtp_pipeline.rs:1115-1127`) and speculative rejection
//! (`pipeline/speculative.rs:810,892`). This type therefore reports
//! `current_seq_len()` in tokens and maps a token-length truncation onto its
//! own two time bases itself — `comp` to `len / ratio` rows, `tail` to the
//! rows behind `len` — and *refuses* (`try_set_len` → `Err`) any truncation
//! that would land before `base`, where the raw rows needed to resume are
//! genuinely gone. The prefix cacher already treats that refusal as "no
//! match, prefill from scratch" (it is how every rotating/sliding-window
//! cache behaves); `margin` is what keeps the short MTP/speculative
//! rollbacks — which cannot decline — inside the retained window.
//!
//! Silently accepting such a truncation would resume the compressor from a
//! gap, i.e. corrupt the distant-context branch with no error. It is the
//! failure mode this type is written to make impossible.
//!
//! # 🔑 Per-row state (wave63-CO)
//!
//! Until this change `tokens` and `base` were **one scalar for the whole
//! batch**, which is what made this type the keystone blocker named by three
//! separate workstreams: PR #90 §6 ("the compressor's `xs` history has no
//! block table"), PR #92 §5.1 ("`XsRolling` cannot carry its own length …
//! third wave to land on it"), and wave29-BC §4b. A batched cache whose rows
//! sit at different token counts could not be represented at all, so no
//! sequence could advance its compressed KV independently of the others.
//!
//! They are now `Vec<usize>`, one entry per batch row, and the two time bases
//! carry the raggedness in the two ways their own geometry allows:
//!
//! * **`comp` is start-anchored and stays that way.** Physical column `j` is
//!   absolute block `j` for *every* row, exactly as before. A row that has
//!   completed fewer blocks simply has stale columns past its own
//!   `tokens_i / ratio`, and those are already `-inf` under the compressed
//!   branch's own causality threshold (`b < floor((q_abs + 1) / ratio)`,
//!   `dsv4_attention.rs:419-434`) once that threshold is evaluated per row —
//!   so no *extra* mask term is needed, only a per-row query position. What a
//!   ragged batch does need is a per-row **append offset**, which is the
//!   `slot_mapping`-shaped indirection [`XsRollingCache::scatter_comp_rows`]
//!   provides.
//! * **`tail` is end-anchored**, i.e. left-aligned in PR #92's sense: row `i`'s
//!   physical column `c` holds absolute token `tokens_i - W + c`, so every
//!   row's live run *ends* at the same column and the one shared append offset
//!   `Tensor::cat(&[tail, xs_new], 1)` uses is simultaneously correct for all
//!   of them. `base_i` stays the *semantic* resume point and may sit above the
//!   physical start when another row in the batch retained more.
//!
//! Today's uniform layout is the special case `W == tokens - base`, under
//! which end-anchored and start-anchored coincide **exactly** — which is why
//! [`XsRollingCache::advance`] keeps a verbatim uniform fast path and every
//! B=1 and every uniform batch runs byte-identical code.

use std::sync::OnceLock;

use candle_core::{DType, Device, Result, Tensor};

use super::{NormalCache, SingleCache};

/// Extra raw tokens retained behind the compressor's own requirement, so a
/// short rollback stays recoverable. MTP/speculative rejection drops at most
/// the number of drafted positions; 16 covers any configured MTP depth
/// (`--mtp-depth`) with room to spare, at a cost of `16 * hidden * 2 B` per
/// compressed layer (128 KB/layer/seq at V4's `hidden = 4096`).
pub const XS_TAIL_MARGIN_TOKENS: usize = 16;

/// `ARC_V4_XS_PER_SEQ=1` — allow the engine to *build* a batched `xs` cache
/// whose rows sit at different token counts. **Default OFF.**
///
/// The mechanism itself ([`XsRollingCache::advance`]'s ragged path,
/// [`XsRollingCache::set_row_lens`]) is always compiled and always available;
/// this flag gates only the two places that would otherwise never produce a
/// ragged cache — [`super::KvCache::supports_per_sequence_len`] and the
/// batching in [`super::NormalCacheManager::clone_in_cache`]. With the flag
/// off nothing in the engine can create more than one row's worth of state, so
/// every path takes the uniform branch and is byte-identical to what shipped
/// before this change.
pub fn xs_per_sequence_enabled() -> bool {
    #[cfg(test)]
    match test_override::current() {
        Some(on) => return on,
        None => {}
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        matches!(
            std::env::var("ARC_V4_XS_PER_SEQ").as_deref(),
            Ok("1") | Ok("true") | Ok("TRUE")
        )
    })
}

/// Test-only override for [`xs_per_sequence_enabled`].
///
/// The production read is latched in a `OnceLock` on purpose — a mid-run change
/// would split one batch's behaviour between its slots — which leaves no way to
/// exercise both sides of the flag in one test binary. This gives one.
///
/// It is **thread-local**, not a global: `cargo test` runs each test on its own
/// thread, so a process-wide override would leak into whatever else happened to
/// be running and make the flag-off assertions flaky rather than meaningful.
#[cfg(test)]
pub(crate) mod test_override {
    use std::cell::Cell;

    thread_local! {
        static STATE: Cell<Option<bool>> = const { Cell::new(None) };
    }

    pub(crate) fn current() -> Option<bool> {
        STATE.with(|s| s.get())
    }

    /// Run `f` with `ARC_V4_XS_PER_SEQ` forced to `on`, on this thread only.
    pub(crate) fn with<R>(on: bool, f: impl FnOnce() -> R) -> R {
        let prev = STATE.with(|s| s.replace(Some(on)));
        let out = f();
        STATE.with(|s| s.set(prev));
        out
    }
}

/// One compressor call: the batch rows that happen to need the *same* slice of
/// the retained window, plus where each of their outputs lands in `comp`.
///
/// Rows are grouped by `(off, len)` rather than run one at a time because both
/// are functions of `tokens_i mod ratio` alone — two rows at the same residue
/// need byte-for-byte the same window geometry and differ only in the
/// destination block. A uniform batch is therefore always exactly one group,
/// and a ragged one is at most `min(B, ratio, spread + 1)` groups: under MTP
/// the spread is bounded by the verify window (`depth + 1`), so ≤ 4 in
/// practice.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct XsCompressGroup {
    /// Physical column of the retained window where this group's slice starts.
    pub off: usize,
    /// Width of the slice, in tokens. Always a multiple of `ratio`.
    pub len: usize,
    /// Compressor output rows to keep, counted from the END of its output —
    /// the leading rows were computed against `overlap_transform`'s pad rather
    /// than against real predecessor tokens.
    pub take: usize,
    /// Batch rows in this group.
    pub rows: Vec<usize>,
    /// `comp` column each row's first kept output block is written to.
    /// Parallel to `rows`.
    pub dests: Vec<usize>,
}

/// Everything one [`XsRollingCache::advance`] has to do, derived from token
/// counts alone. Pure arithmetic — no tensors — so the geometry can be pinned
/// by unit tests independently of the compressor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct XsAdvancePlan {
    pub groups: Vec<XsCompressGroup>,
    pub tokens_new: Vec<usize>,
    pub base_new: Vec<usize>,
    /// Physical width of the retained window after this step.
    pub tail_width: usize,
    /// `comp.current_seq_len` after this step — `max_i tokens_new_i / ratio`.
    pub comp_rows_new: usize,
    /// The `comp` column band every write of this step falls inside, as
    /// `(start, len)`. `len == 0` means no group completed a block. Bounding
    /// the scatter to this band is what keeps the ragged path `O(B * spread)`
    /// instead of `O(B * capacity)`.
    pub band: (usize, usize),
    /// Every row is at the same token count, so the whole step is expressible
    /// as the single narrow + single append the uniform path has always used.
    pub uniform: bool,
}

/// Plan one advance. See [`XsAdvancePlan`].
///
/// `w_phys` is the physical width of the retained raw window (0 before the
/// first token); row `i`'s physical column `c` holds absolute token
/// `tokens[i] - w_phys + c`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn plan_xs_advance(
    tokens: &[usize],
    base: &[usize],
    w_phys: usize,
    t_new: usize,
    ratio: usize,
    span_groups: usize,
    margin: usize,
) -> Result<XsAdvancePlan> {
    debug_assert_eq!(tokens.len(), base.len());
    let uniform = tokens.iter().all(|&t| t == tokens[0]);

    // (off, len) → index into `groups`. Linear scan: there are at most a
    // handful of distinct residues, and a map would cost more than it saves.
    let mut groups: Vec<XsCompressGroup> = Vec::new();
    let mut tokens_new = Vec::with_capacity(tokens.len());
    let mut base_new = Vec::with_capacity(tokens.len());
    let mut tail_width = 0usize;
    let mut comp_rows_new = 0usize;
    let mut band_lo = usize::MAX;
    let mut band_hi = 0usize;

    for (i, (&tok, &bas)) in tokens.iter().zip(base).enumerate() {
        // Physical start of row `i`'s retained window, as an absolute token.
        let phys_start = tok.checked_sub(w_phys).ok_or_else(|| {
            candle_core::Error::msg(format!(
                "xs rolling cache: row {i} holds {tok} tokens but the retained window is {w_phys} \
                 wide, so it would begin before token 0. The window is end-anchored — every row's \
                 live run finishes at the same column — so this means `tokens` and `tail` were \
                 set inconsistently."
            ))
        })?;
        let tok_new = tok + t_new;
        let g_done = tok / ratio;
        let g_target = tok_new / ratio;
        comp_rows_new = comp_rows_new.max(g_target);

        if g_target > g_done {
            // Blocks `g_done..g_target` need raw tokens from group
            // `g_done + 1 - span_groups` onward.
            let need_start = (g_done + 1).saturating_sub(span_groups) * ratio;
            if need_start < phys_start {
                candle_core::bail!(
                    "xs rolling cache: compressor history gap — row {i}'s block {g_done} needs \
                     tokens from {need_start} but its retained window starts at {phys_start}. \
                     This means a truncation was applied without going through `set_len`."
                );
            }
            let off = need_start - phys_start;
            let len = g_target * ratio - need_start;
            let take = g_target - g_done;
            band_lo = band_lo.min(g_done);
            band_hi = band_hi.max(g_target);
            match groups
                .iter_mut()
                .find(|g| g.off == off && g.len == len && g.take == take)
            {
                Some(g) => {
                    g.rows.push(i);
                    g.dests.push(g_done);
                }
                None => groups.push(XsCompressGroup {
                    off,
                    len,
                    take,
                    rows: vec![i],
                    dests: vec![g_done],
                }),
            }
        }

        // Retain exactly what a future block can consume — including after a
        // rollback of up to `margin` tokens.
        //
        // The subtlety: "keep the last `span_groups * ratio + margin` tokens"
        // is NOT the same rule. A rollback that crosses a group boundary
        // invalidates that group's compressed row, and rebuilding it needs the
        // group's raw tokens from its START, not `margin` of them. So the
        // retention point is the one the WORST rollback would need — i.e. the
        // one computed at `tokens - margin`, not at `tokens`. It costs
        // nothing: the tail is only extended backwards during the first
        // `margin` tokens of a group, when the current group is that much
        // shorter, so the bound stays `span_groups * ratio + margin`.
        //
        // The floor is `base`, not the physical start: `base` is what this row
        // has *promised* to be able to resume from, and clamping to it keeps a
        // batched row's retention identical to the same sequence run alone
        // even when a neighbour caused the physical window to be wider.
        let rollback_floor = tok_new.saturating_sub(margin);
        let keep_from = ((rollback_floor / ratio + 1).saturating_sub(span_groups) * ratio).max(bas);
        let nb = keep_from.min(tok_new);
        tail_width = tail_width.max(tok_new - nb);
        tokens_new.push(tok_new);
        base_new.push(nb);
    }

    let band = if band_hi > band_lo {
        (band_lo, band_hi - band_lo)
    } else {
        (0, 0)
    };
    Ok(XsAdvancePlan {
        groups,
        tokens_new,
        base_new,
        tail_width,
        comp_rows_new,
        band,
        uniform,
    })
}

/// Rolling compressed-history state for one V4 compressor layer. See the
/// module docs.
#[derive(Debug, Clone)]
pub struct XsRollingCache {
    /// Completed compressed rows `[B, G, head_dim]` on seq dim 1.
    /// `comp.current_seq_len()` counts ROWS (groups), not tokens, and column
    /// `j` is absolute block `j` for every batch row.
    pub comp: SingleCache,
    /// Retained raw compressor-input rows, `[B, W, hidden]` on seq dim 1,
    /// **end-anchored**: row `i`'s column `c` holds absolute token
    /// `tokens[i] - W + c`. `None` before the first token.
    pub tail: Option<Tensor>,
    /// Per batch row: the first token that row is willing to resume from.
    /// `base[i] >= tokens[i] - W`; the two are equal for every uniform batch,
    /// which is every batch the engine builds with `ARC_V4_XS_PER_SEQ` off.
    base: Vec<usize>,
    /// Per batch row: total tokens seen — the length every other part of the
    /// engine means. One entry until a ragged batch is built.
    tokens: Vec<usize>,
    /// Compression ratio of the owning layer (V4: 4 for CSA, 128 for HCA).
    pub ratio: usize,
    /// How many consecutive groups of raw tokens one compressed row consumes:
    /// 2 for the overlapping (`ratio == 4`) compressor, 1 otherwise.
    pub span_groups: usize,
    /// Extra retained tokens for rollback ([`XS_TAIL_MARGIN_TOKENS`]).
    pub margin: usize,
    /// Width of a compressed row. Known at load time, and needed to
    /// materialise `comp.all_data` before the first group completes so the
    /// cache managers always have a tensor to batch.
    pub head_dim: usize,
}

impl XsRollingCache {
    pub fn new(
        ratio: usize,
        span_groups: usize,
        head_dim: usize,
        max_position_embeddings: usize,
    ) -> Self {
        assert!(ratio > 0, "XsRollingCache: ratio must be > 0");
        assert!(
            span_groups >= 1,
            "XsRollingCache: span_groups must be >= 1 (one row always needs its own group)"
        );
        // Rows, not tokens. `SingleCache::append` grows in `CACHE_GROW_SIZE`
        // blocks and bails if that overshoots `max_seq_len`, so round the row
        // ceiling up to a whole number of blocks.
        let grow = NormalCache::CACHE_GROW_SIZE;
        let max_rows = max_position_embeddings.div_ceil(ratio).div_ceil(grow) * grow;
        // Small initial allocation: an HCA layer (ratio 128) needs 16 rows at
        // a 2048-token context, and paying a 512-row block for it would give
        // back a third of the win.
        let init_rows = 64.min(max_rows);
        Self {
            comp: SingleCache::new(1, max_rows, init_rows),
            tail: None,
            base: vec![0],
            tokens: vec![0],
            ratio,
            span_groups,
            margin: XS_TAIL_MARGIN_TOKENS,
            head_dim,
        }
    }

    /// How many batch rows this cache currently describes.
    pub fn rows(&self) -> usize {
        self.tokens.len()
    }

    /// Per-row token counts and resume points, in batch order.
    pub fn row_lens(&self) -> (&[usize], &[usize]) {
        (&self.tokens, &self.base)
    }

    /// Install per-row token counts and resume points. Used by
    /// `NormalCacheManager::clone_in_cache` to assemble a batched cache from
    /// per-sequence ones, and by `clone_out_cache` to split it back.
    ///
    /// This is the *only* writer of per-row state outside [`Self::advance`] /
    /// [`Self::set_len`], and it lives on the cache rather than beside it on
    /// purpose: a second map keyed by sequence id is the divergence wave59-CJ
    /// documents and PR #92 removed. The lengths have exactly one owner.
    pub fn set_row_lens(&mut self, tokens: Vec<usize>, base: Vec<usize>) -> Result<()> {
        if tokens.len() != base.len() || tokens.is_empty() {
            candle_core::bail!(
                "xs rolling cache: row lengths must be non-empty and paired, got {} tokens / {} \
                 base",
                tokens.len(),
                base.len()
            );
        }
        let w = self.tail_width()?;
        for (i, (&t, &b)) in tokens.iter().zip(&base).enumerate() {
            if b > t {
                candle_core::bail!(
                    "xs rolling cache: row {i} would resume from token {b} but has only seen {t}"
                );
            }
            if t < w {
                candle_core::bail!(
                    "xs rolling cache: row {i} holds {t} tokens, fewer than the {w}-wide retained \
                     window it would have to end at"
                );
            }
        }
        self.tokens = tokens;
        self.base = base;
        Ok(())
    }

    /// [`Self::set_row_lens`] without the cross-check, for the one caller that
    /// is splitting *this* cache's own rows back out (`clone_out_cache`) and
    /// therefore cannot produce values the cache did not already hold.
    pub(crate) fn assign_row_lens(&mut self, tokens: Vec<usize>, base: Vec<usize>) {
        debug_assert_eq!(tokens.len(), base.len());
        debug_assert!(!tokens.is_empty());
        self.tokens = tokens;
        self.base = base;
    }

    /// Split batch row `row` back out into a per-sequence cache, given that
    /// row's already-chunked buffers. The inverse of the assembly
    /// `clone_in_cache` does, and the reason it is here rather than inline
    /// there: it has to restore the *original* invariant `width == tokens -
    /// base`, which a batched cache deliberately relaxes.
    ///
    /// The batched window is as wide as the greediest row needed. Handing that
    /// width to a row that needed less would let its own `advance` believe it
    /// holds history it never promised — and the window is end-anchored, so
    /// this row's share is its LAST `tokens - base` columns, not its first.
    pub fn split_row(&self, row: usize, comp: Tensor, tail: Tensor) -> Result<Self> {
        let (tokens, base) = self.row_lens();
        let i = if tokens.len() == 1 { 0 } else { row };
        let tok = *tokens.get(i).ok_or_else(|| {
            candle_core::Error::msg(format!(
                "xs rolling cache: row {row} is past the {} this cache describes",
                tokens.len()
            ))
        })?;
        let have = tail.dim(1)?;
        // `want > have` cannot happen (`set_row_lens` refuses such a batch),
        // but if it ever did, keeping the whole window and RAISING `base`
        // degrades to "refuses a rollback it could have served" — never to
        // resuming the compressor from a gap.
        let want = (tok - base[i]).min(have);
        let mut out = self.clone();
        out.comp.all_data = Some(comp);
        out.comp.current_seq_len = tok / self.ratio;
        out.tail = Some(if want == have {
            tail
        } else {
            tail.narrow(1, have - want, want)?.contiguous()?
        });
        out.tokens = vec![tok];
        out.base = vec![tok - want];
        Ok(out)
    }

    /// Physical width of the retained raw window (0 when there is none).
    pub fn tail_width(&self) -> Result<usize> {
        match self.tail.as_ref() {
            Some(t) => t.dim(1),
            None => Ok(0),
        }
    }

    /// Tokens seen — what `set_len` / the schedulers / the prefix cacher mean
    /// by "length". The batch maximum when the rows disagree, which is the
    /// width every dense consumer has to allocate for.
    pub fn current_seq_len(&self) -> usize {
        self.tokens.iter().copied().max().unwrap_or(0)
    }

    /// Whether every row is at the same token count. False only under
    /// `ARC_V4_XS_PER_SEQ`.
    pub fn rows_uniform(&self) -> bool {
        self.tokens.iter().all(|&t| t == self.tokens[0])
    }

    pub fn reset(&mut self) {
        self.comp.reset();
        self.tail = None;
        self.base = vec![0];
        self.tokens = vec![0];
    }

    /// First token index whose raw row is still retained. A truncation to a
    /// length below this cannot be resumed. The batch maximum — i.e. the most
    /// restrictive row — because a uniform truncation has to satisfy all of
    /// them.
    pub fn resumable_from(&self) -> usize {
        self.base.iter().copied().max().unwrap_or(0)
    }

    pub fn try_set_len(&self, len: usize) -> Result<()> {
        if !self.rows_uniform() {
            candle_core::bail!(
                "xs rolling cache: this cache carries per-row token counts ({:?}) and a single \
                 `set_len` cannot express a rollback for all of them — the retained window is \
                 end-anchored, so each row would need its own narrow. Split the batch first \
                 (`clone_out_cache`) and truncate the per-sequence caches.",
                self.tokens
            );
        }
        let tokens = self.tokens[0];
        let base = self.base[0];
        if len > tokens {
            candle_core::bail!("xs rolling cache: cannot extend from {tokens} to {len} tokens");
        }
        // Two things have to survive the rollback: the raw rows the truncated
        // tail is cut from (`len >= base`), and the raw rows the NEXT
        // compressed row will be built from — group `len/ratio + 1 -
        // span_groups` onward, which for the overlapping compressor reaches a
        // whole group further back than `len` itself. Checking only the first
        // would let `set_len` succeed and leave the gap for `advance` to hit.
        let resume_from = (len / self.ratio + 1).saturating_sub(self.span_groups) * self.ratio;
        if len < base || resume_from < base {
            candle_core::bail!(
                "xs rolling cache: rollback to {len} tokens is behind the retained raw window \
                 (resuming needs the compressor input from token {resume_from}, retained from \
                 {base}, currently at {tokens} tokens). Those rows are gone, so resuming would \
                 silently skip history; recompute the prefix instead."
            );
        }
        Ok(())
    }

    pub fn set_len(&mut self, len: usize) -> Result<()> {
        self.try_set_len(len)?;
        let tokens = self.tokens[0];
        if len == tokens {
            return Ok(());
        }
        // Compressed rows are token-aligned: row `j` covers tokens
        // `[j*ratio, (j+1)*ratio)`, so exactly `len / ratio` of them survive.
        self.comp.set_len(len / self.ratio)?;
        // The window is END-anchored, so a rollback drops columns from the
        // right. Under the uniform layout (`W == tokens - base`) that is
        // `narrow(1, 0, len - base)` — bit-identical to what this always did.
        let drop = tokens - len;
        let w = self.tail_width()?;
        let keep = w.saturating_sub(drop);
        self.tail = match self.tail.take() {
            Some(t) if keep > 0 => Some(t.narrow(1, 0, keep)?.contiguous()?),
            _ => None,
        };
        for t in self.tokens.iter_mut() {
            *t = len;
        }
        Ok(())
    }

    /// The first raw token any *future* compressed row can still need.
    ///
    /// Row `g` (the next one to complete) is built from group
    /// `g + 1 - span_groups` onward, so everything below this index is dead
    /// weight the cache is free to drop. It is the same expression
    /// [`Self::advance`] uses for `need_start`, named once so the retention
    /// rule and the trim rule cannot drift apart.
    pub fn compressor_needs_from(&self) -> usize {
        (self.comp.current_seq_len() + 1).saturating_sub(self.span_groups) * self.ratio
    }

    /// Raise this cache's `base` to `new_base`, dropping the leading raw rows.
    ///
    /// # Why this exists
    ///
    /// For a **single-row** cache `tail` is `tokens - base` wide and `base` is
    /// **not** a function of `tokens`: it is `canonical(high-water tokens)`,
    /// because [`Self::set_len`] narrows the tail without moving `base` and
    /// [`Self::advance`] never lowers it. Two sequences at the identical token
    /// count can therefore hold different-width tails — one that rolled back
    /// across a group boundary, one that arrived directly, one restored from a
    /// prefix-cache entry stored at a greater length.
    /// `NormalCacheManager::clone_in_cache` batches `tail` with `slice_set`,
    /// which demands an exact dim match, so that difference is the
    /// `shape mismatch on dim 1, 18 <> 22` class of failure — on the **plain
    /// decode path**, with no speculation involved (PR #93).
    ///
    /// Trimming to the batch's **largest** `base` reconciles them, and it is
    /// lossless by construction: the member that already sits at `base_max`
    /// proves no future compressed row needs a token below it. The rows dropped
    /// here only ever shortened how far a *rollback* could reach.
    ///
    /// Refuses (rather than silently dropping history) if `new_base` would cut
    /// into rows the next compressed row is built from, which is the same
    /// condition [`Self::advance`] bails on as a "compressor history gap".
    ///
    /// # 🔴 Why this refuses a multi-row cache instead of reading `base[0]`
    ///
    /// `new_base` is ONE number, and `drop` / `keep` below are physical column
    /// offsets applied across the whole batch dim. That is only meaningful when
    /// the cache describes one row. Once `base` is per-row, "trim **the** raw
    /// window" is not a statement this cache can evaluate: rows sharing a
    /// physical width `W` sit at `base[i] >= tokens[i] - W`, so a single
    /// `narrow` would move a different logical amount for each of them.
    ///
    /// Reading `base[0]` and proceeding would produce a plausible-looking wrong
    /// answer — the tensor would come out the right *shape*, which is exactly
    /// what the caller checks — so it refuses by name instead (D18). Every
    /// production caller passes a per-sequence cache (`rows() == 1`):
    /// `reconcile_xs_bases` runs before the batch is gathered, never after.
    pub fn trim_tail_to(&mut self, new_base: usize) -> Result<()> {
        if self.rows() > 1 {
            candle_core::bail!(
                "xs rolling cache: `trim_tail_to` is defined for a single-row cache, but this \
                 one carries {} rows (tokens {:?}, base {:?}). A scalar `new_base` cannot \
                 express a trim for rows that sit at different resume points — the retained \
                 window is shared and end-anchored, so one `narrow` moves a different logical \
                 amount per row. Split the batch first (`clone_out_cache`) and trim the \
                 per-sequence caches.",
                self.rows(),
                self.tokens,
                self.base
            );
        }
        let tokens = self.tokens[0];
        let base = self.base[0];
        if new_base <= base {
            return Ok(());
        }
        if new_base > tokens {
            candle_core::bail!(
                "xs rolling cache: cannot trim the raw window to start at {new_base}; the cache \
                 only holds {tokens} tokens"
            );
        }
        let need_start = self.compressor_needs_from();
        if need_start < new_base {
            candle_core::bail!(
                "xs rolling cache: trimming the raw window to {new_base} would drop tokens the \
                 next compressed row is built from (it needs them from {need_start}). Resuming \
                 would silently skip history."
            );
        }
        let keep = tokens - new_base;
        let drop = new_base - base;
        self.tail = match self.tail.take() {
            Some(t) if keep > 0 => Some(t.narrow(1, drop, keep)?.contiguous()?),
            _ => None,
        };
        self.base[0] = new_base;
        Ok(())
    }

    /// The compressed rows `[B, G, head_dim]`, or `None` when no group has
    /// completed yet (the sliding-window branch alone covers the context).
    ///
    /// `G` is the batch maximum. A row that has completed fewer blocks carries
    /// stale columns past its own `tokens_i / ratio`; the compressed branch's
    /// causality threshold (`b < floor((q_abs + 1) / ratio)`) already excludes
    /// exactly those columns once it is evaluated at the row's own absolute
    /// query position, so they need no additional mask term — only a per-row
    /// position, which is what `MtpSpeculativePipeline::target_masks_ragged_batches`
    /// still gates on.
    pub fn compressed_rows(&self) -> Result<Option<Tensor>> {
        if self.comp.current_seq_len() == 0 {
            return Ok(None);
        }
        self.comp.current_data()
    }

    /// Make this cache describe `b` batch rows. Broadcasting a single row is
    /// how a freshly built or freshly reset cache picks up the batch it is
    /// about to be advanced with.
    fn ensure_rows(&mut self, b: usize) -> Result<()> {
        if self.tokens.len() == b {
            return Ok(());
        }
        if self.rows_uniform() {
            // Uniform state carries no per-row information, so re-shaping it to
            // `b` rows is exact in BOTH directions — including the shrink a
            // batch that has just lost a sequence hands the next forward, and
            // the broadcast a freshly built or freshly reset cache needs. This
            // is every cache in the tree with `ARC_V4_XS_PER_SEQ` off, which is
            // why nothing about the pre-existing batch-composition behaviour
            // changes.
            let (t, ba) = (self.tokens[0], self.base[0]);
            self.tokens = vec![t; b];
            self.base = vec![ba; b];
            return Ok(());
        }
        // A RAGGED cache is different: dropping to `b` rows means choosing
        // which sequences survived, and this type has no way to know. Guessing
        // would silently hand one sequence another's compressed history.
        candle_core::bail!(
            "xs rolling cache: this cache describes {} batch rows at different token counts \
             ({:?}) but was handed {b}. Which rows survive is the caller's knowledge — split the \
             batch with `split_row` (what `clone_out_cache` does) and re-assemble the survivors, \
             rather than reshaping it here.",
            self.tokens.len(),
            self.tokens
        )
    }

    /// Grow `comp.all_data` so `need` rows fit, mirroring `SingleCache::append`'s
    /// `CACHE_GROW_SIZE` blocking exactly (the ragged path writes with
    /// `slice_set` instead of `append`, so it has to do this itself).
    fn reserve_comp_rows(
        &mut self,
        need: usize,
        b: usize,
        dtype: DType,
        dev: &Device,
    ) -> Result<()> {
        if self.comp.dim != 1 {
            candle_core::bail!(
                "xs rolling cache: the compressed rows must sit on seq dim 1, found dim {}",
                self.comp.dim
            );
        }
        if need > self.comp.capacity_seq_len {
            let diff = need - self.comp.capacity_seq_len;
            let blocks = diff.div_ceil(NormalCache::CACHE_GROW_SIZE);
            self.comp.capacity_seq_len += blocks * NormalCache::CACHE_GROW_SIZE;
            if self.comp.capacity_seq_len > self.comp.max_seq_len {
                candle_core::bail!(
                    "xs rolling cache: requested compressed capacity ({}) above max ({})",
                    self.comp.capacity_seq_len,
                    self.comp.max_seq_len
                );
            }
            let grown = Tensor::zeros((b, self.comp.capacity_seq_len, self.head_dim), dtype, dev)?;
            if let Some(old) = self.comp.all_data.as_ref() {
                grown.slice_set(old, 1, 0)?;
            }
            self.comp.all_data = Some(grown);
        } else if self.comp.all_data.is_none() {
            self.comp.all_data = Some(Tensor::zeros(
                (b, self.comp.capacity_seq_len, self.head_dim),
                dtype,
                dev,
            )?);
        }
        Ok(())
    }

    /// Write each batch row's freshly compressed blocks at **its own** `comp`
    /// column — the `slot_mapping`-shaped indirection a ragged batch needs and
    /// the one primitive `SingleCache::append`, which writes every row at one
    /// shared offset, cannot express.
    ///
    /// `writes` is `(batch_row, first_comp_column, blocks)` and `src` holds the
    /// blocks in the same order, `[N, head_dim]`.
    ///
    /// The scatter is bounded to the column band the plan says every write
    /// falls inside (`XsAdvancePlan::band`), which is `max_i g_target -
    /// min_i g_done` wide — 1 or 2 columns under MTP, never the whole
    /// capacity. Without that bound this would copy `B * capacity * head_dim`
    /// elements per layer per step and the ragged path would cost more than
    /// the raggedness buys.
    fn scatter_comp_rows(
        &mut self,
        src: &Tensor,
        idx_in_band: &[u32],
        band_start: usize,
        band_len: usize,
    ) -> Result<()> {
        let all = self.comp.all_data.as_ref().ok_or_else(|| {
            candle_core::Error::msg("xs rolling cache: compressed buffer missing")
        })?;
        let b = all.dim(0)?;
        let d = self.head_dim;
        let band = all.narrow(1, band_start, band_len)?.contiguous()?;
        let flat = band.reshape((b * band_len, d))?;
        let idx = Tensor::from_slice(idx_in_band, (idx_in_band.len(),), all.device())?;
        // Unique destinations, so `index_add` onto a zero buffer IS a scatter.
        // Candle has no strided overwrite-scatter that works on both the CPU
        // and CUDA backends; add-onto-zero plus a written/not-written select
        // does, in three kernels and with no per-row loop.
        let zeros = Tensor::zeros((b * band_len, d), all.dtype(), all.device())?;
        let scattered = zeros.index_add(&idx, &src.contiguous()?, 0)?;
        let ones = Tensor::ones((idx_in_band.len(), 1), all.dtype(), all.device())?;
        let written = Tensor::zeros((b * band_len, 1), all.dtype(), all.device())?
            .index_add(&idx, &ones, 0)?;
        let hit = written
            .ge(0.5f64)?
            .broadcast_as((b * band_len, d))?
            .contiguous()?;
        let merged = hit.where_cond(&scattered, &flat)?;
        all.slice_set(&merged.reshape((b, band_len, d))?, 1, band_start)?;
        Ok(())
    }

    /// Append `xs_new` (`[B, T_new, hidden]`), compress every group that
    /// completes, drop the raw rows no future row can need, and return the
    /// full compressed history `[B, G, head_dim]` (`None` while `G == 0`).
    ///
    /// `compress` is the layer's `V4Compressor::forward_from_xs`: it maps
    /// `[B, k*ratio, hidden]` to `[B, k, head_dim]`, and its row `i` depends
    /// only on the window's groups `i` and (when `span_groups == 2`) `i - 1`
    /// — with group `-1` of the window filled by `overlap_transform`'s pad.
    /// This function therefore always hands it `span_groups - 1` groups of
    /// real context ahead of the rows it keeps, and discards the padded
    /// leading row, EXCEPT at the true start of the sequence where the pad is
    /// the correct input.
    ///
    /// `compress` is `Fn` rather than `FnOnce` because a ragged batch needs one
    /// call per distinct window geometry. It is called exactly once for every
    /// uniform batch, including every B=1 sequence.
    pub fn advance<F>(&mut self, xs_new: &Tensor, compress: F) -> Result<Option<Tensor>>
    where
        F: Fn(&Tensor) -> Result<Tensor>,
    {
        let (b, t_new, _hidden) = xs_new.dims3()?;
        if t_new == 0 {
            return self.compressed_rows();
        }
        self.ensure_rows(b)?;
        let xs_new = xs_new.contiguous()?;

        if self.rows_uniform() {
            return self.advance_uniform(b, t_new, &xs_new, compress);
        }
        self.advance_ragged(b, t_new, &xs_new, compress)
    }

    /// The path every B=1 sequence and every uniform batch takes — unchanged
    /// from before per-row state existed, deliberately: it is what makes
    /// "token-identical with the flag off" a property of the code rather than
    /// of a test.
    fn advance_uniform<F>(
        &mut self,
        b: usize,
        t_new: usize,
        xs_new: &Tensor,
        compress: F,
    ) -> Result<Option<Tensor>>
    where
        F: Fn(&Tensor) -> Result<Tensor>,
    {
        let tokens = self.tokens[0];
        let base = self.base[0];

        // `window` covers tokens [win_start, tokens + t_new).
        let (window, win_start) = match self.tail.as_ref() {
            Some(tail) if tokens > base => (Tensor::cat(&[tail, xs_new], 1)?, base),
            _ => (xs_new.clone(), tokens),
        };
        let tokens_new = tokens + t_new;

        let g_done = self.comp.current_seq_len();
        let g_target = tokens_new / self.ratio;
        if g_target > g_done {
            // Rows g_done..g_target need raw tokens from group
            // `g_done + 1 - span_groups` onward.
            let need_start = self.compressor_needs_from();
            debug_assert_eq!(
                need_start,
                (g_done + 1).saturating_sub(self.span_groups) * self.ratio
            );
            if need_start < win_start {
                candle_core::bail!(
                    "xs rolling cache: compressor history gap — row {g_done} needs tokens from \
                     {need_start} but the retained window starts at {win_start}. This means a \
                     truncation was applied without going through `set_len`."
                );
            }
            let off = need_start - win_start;
            let len = g_target * self.ratio - need_start;
            let slice = window.narrow(1, off, len)?.contiguous()?;
            let rows = compress(&slice)?;
            let n_rows = rows.dim(1)?;
            let take = g_target - g_done;
            if n_rows < take {
                candle_core::bail!(
                    "xs rolling cache: compressor returned {n_rows} rows for a {len}-token \
                     window (ratio {}), expected at least {take}",
                    self.ratio
                );
            }
            // Drop the leading row(s) computed against `overlap_transform`'s
            // pad rather than against real predecessor tokens.
            let rows = rows.narrow(1, n_rows - take, take)?.contiguous()?;
            self.comp.append(&rows)?;
        }
        // Materialise the row buffer even when no group has completed yet:
        // `NormalCacheManager` batches `all_data` directly and unwraps it, so
        // a `None` here would panic the first time a sub-`ratio` prompt is
        // cloned out. Same shape `SingleCache::append` would have allocated.
        if self.comp.all_data.is_none() {
            self.comp.all_data = Some(Tensor::zeros(
                (b, self.comp.capacity_seq_len, self.head_dim),
                window.dtype(),
                window.device(),
            )?);
        }

        // Retain exactly what a future row can consume — see `plan_xs_advance`
        // for why the retention point is computed at `tokens - margin`.
        let rollback_floor = tokens_new.saturating_sub(self.margin);
        let keep_from = ((rollback_floor / self.ratio + 1).saturating_sub(self.span_groups)
            * self.ratio)
            .max(win_start);
        let new_base = keep_from.min(tokens_new);
        self.tail = Some(
            window
                .narrow(1, new_base - win_start, tokens_new - new_base)?
                .contiguous()?,
        );
        for (t, ba) in self.tokens.iter_mut().zip(self.base.iter_mut()) {
            *t = tokens_new;
            *ba = new_base;
        }
        debug_assert_eq!(b, self.tail.as_ref().unwrap().dim(0)?);

        self.compressed_rows()
    }

    /// The per-row path: one compressor call per distinct window geometry, and
    /// a `slot_mapping` scatter so each row's blocks land at its own `comp`
    /// column.
    fn advance_ragged<F>(
        &mut self,
        b: usize,
        t_new: usize,
        xs_new: &Tensor,
        compress: F,
    ) -> Result<Option<Tensor>>
    where
        F: Fn(&Tensor) -> Result<Tensor>,
    {
        let w_phys = self.tail_width()?;
        let plan = plan_xs_advance(
            &self.tokens,
            &self.base,
            w_phys,
            t_new,
            self.ratio,
            self.span_groups,
            self.margin,
        )?;

        let window = match self.tail.as_ref() {
            Some(tail) if w_phys > 0 => Tensor::cat(&[tail, xs_new], 1)?,
            _ => xs_new.clone(),
        };

        let (band_start, band_len) = plan.band;
        if band_len > 0 {
            self.reserve_comp_rows(plan.comp_rows_new, b, window.dtype(), window.device())?;
            let mut src_parts: Vec<Tensor> = Vec::with_capacity(plan.groups.len());
            let mut idx: Vec<u32> = Vec::new();
            for group in &plan.groups {
                let rows_idx =
                    Tensor::from_iter(group.rows.iter().map(|&r| r as u32), window.device())?;
                let sub = window
                    .index_select(&rows_idx, 0)?
                    .narrow(1, group.off, group.len)?
                    .contiguous()?;
                let out = compress(&sub)?;
                let n_rows = out.dim(1)?;
                if n_rows < group.take {
                    candle_core::bail!(
                        "xs rolling cache: compressor returned {n_rows} rows for a {}-token \
                         window (ratio {}), expected at least {}",
                        group.len,
                        self.ratio,
                        group.take
                    );
                }
                let kept = out
                    .narrow(1, n_rows - group.take, group.take)?
                    .contiguous()?;
                src_parts.push(kept.reshape((group.rows.len() * group.take, self.head_dim))?);
                for (&row, &dest) in group.rows.iter().zip(&group.dests) {
                    for j in 0..group.take {
                        idx.push((row * band_len + (dest - band_start) + j) as u32);
                    }
                }
            }
            let src = if src_parts.len() == 1 {
                src_parts.pop().unwrap()
            } else {
                Tensor::cat(&src_parts, 0)?
            };
            self.scatter_comp_rows(&src, &idx, band_start, band_len)?;
        }
        self.comp.current_seq_len = plan.comp_rows_new;
        if self.comp.all_data.is_none() {
            self.comp.all_data = Some(Tensor::zeros(
                (b, self.comp.capacity_seq_len, self.head_dim),
                window.dtype(),
                window.device(),
            )?);
        }

        // End-anchored retention: keep the last `tail_width` columns, which is
        // the widest any row needs. A row that needs fewer simply carries a few
        // real older tokens ahead of its own `base` — never read, because every
        // offset is derived from that row's own `base`.
        let w_win = window.dim(1)?;
        self.tail = Some(
            window
                .narrow(1, w_win - plan.tail_width, plan.tail_width)?
                .contiguous()?,
        );
        self.tokens = plan.tokens_new;
        self.base = plan.base_new;
        self.compressed_rows()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A uniform batch is exactly ONE compressor call, at the same `(off, len)`
    /// the scalar implementation used. If this ever splits, the "flag off is
    /// byte-identical" claim is gone.
    #[test]
    fn a_uniform_batch_plans_one_group_at_the_scalar_offsets() {
        // ratio 4, span 2 (CSA): 8 tokens seen, tail holds [0, 8), +4 new.
        let plan = plan_xs_advance(&[8, 8, 8, 8], &[0, 0, 0, 0], 8, 4, 4, 2, 16).unwrap();
        assert!(plan.uniform);
        assert_eq!(plan.groups.len(), 1);
        let g = &plan.groups[0];
        assert_eq!(g.rows, vec![0, 1, 2, 3]);
        assert_eq!(g.dests, vec![2, 2, 2, 2]);
        assert_eq!(g.take, 1);
        // g_done = 2, need_start = (2 + 1 - 2) * 4 = 4; phys_start = 8 - 8 = 0.
        assert_eq!(g.off, 4);
        assert_eq!(g.len, 12 - 4);
        assert_eq!(plan.comp_rows_new, 3);
        assert_eq!(plan.band, (2, 1));
    }

    /// The load-bearing claim of the whole design: window geometry is a
    /// function of `tokens mod ratio` ALONE, so rows are groupable and a
    /// ragged batch costs `#distinct residues` compressor calls, not `B`.
    #[test]
    fn window_geometry_depends_only_on_the_token_residue() {
        // Eight rows spanning two residues (mod 4) at two different block
        // counts: 20/24 share residue 0, 21/25 share residue 1.
        let toks = vec![20, 24, 21, 25, 20, 24, 21, 25];
        let base = vec![12, 16, 12, 16, 12, 16, 12, 16];
        let plan = plan_xs_advance(&toks, &base, 8, 4, 4, 2, 16).unwrap();
        assert!(!plan.uniform);
        assert_eq!(
            plan.groups.len(),
            2,
            "eight rows at two residues must cost two compressor calls, not eight"
        );
        for g in &plan.groups {
            assert_eq!(g.rows.len(), 4);
            // Every row in a group gets the SAME slice…
            assert!(g.len % 4 == 0);
            // …and a DIFFERENT destination block, which is the whole point.
            let mut d = g.dests.clone();
            d.sort_unstable();
            d.dedup();
            assert!(
                d.len() > 1,
                "a group whose rows all land on one block is not ragged"
            );
        }
    }

    /// A row that has not completed a new block contributes no compressor work
    /// at all — the ragged path must not fabricate a block for it.
    #[test]
    fn a_row_that_completes_no_block_joins_no_group() {
        // ratio 128 (HCA), 4 tokens in: row 0 crosses the 1024 boundary, row 1
        // does not. Both retain from block 7 (`base = 896`), so the physical
        // window is 126 wide for row 0 and 122 for row 1.
        let plan = plan_xs_advance(&[1022, 1018], &[896, 896], 126, 4, 128, 1, 16).unwrap();
        assert_eq!(plan.groups.len(), 1);
        assert_eq!(plan.groups[0].rows, vec![0]);
        assert_eq!(plan.groups[0].dests, vec![7]);
        assert_eq!(plan.groups[0].take, 1);
        assert_eq!(plan.comp_rows_new, 8);
        assert_eq!(plan.tokens_new, vec![1026, 1022]);
        assert_eq!(plan.band, (7, 1));
    }

    /// The scatter band is what keeps the ragged path cheap. It must cover
    /// every write and nothing more.
    #[test]
    fn the_scatter_band_covers_exactly_the_columns_written() {
        let plan = plan_xs_advance(&[16, 20, 24], &[8, 12, 16], 8, 4, 4, 2, 16).unwrap();
        let (start, len) = plan.band;
        for g in &plan.groups {
            for &d in &g.dests {
                assert!(
                    d >= start && d + g.take <= start + len,
                    "write {d} outside band"
                );
            }
        }
        // g_done spans 4..6, g_target spans 5..7 → band [4, 7).
        assert_eq!(plan.band, (4, 3));
    }

    /// Per-row retention must equal what the same row would compute alone: a
    /// neighbour holding more history may widen the physical window, but it may
    /// not move this row's `base`, or a rollback this row should refuse would
    /// start succeeding.
    #[test]
    fn a_neighbour_does_not_move_this_rows_base() {
        let ragged = plan_xs_advance(&[64, 65], &[48, 49], 17, 1, 4, 2, 16).unwrap();
        let alone_0 = plan_xs_advance(&[64], &[48], 16, 1, 4, 2, 16).unwrap();
        let alone_1 = plan_xs_advance(&[65], &[49], 16, 1, 4, 2, 16).unwrap();
        assert_eq!(ragged.base_new[0], alone_0.base_new[0]);
        assert_eq!(ragged.base_new[1], alone_1.base_new[0]);
        // The physical window is the widest of the two…
        assert_eq!(
            ragged.tail_width,
            alone_0.tail_width.max(alone_1.tail_width)
        );
        // …and it is wide enough for both rows' promises.
        assert!(ragged.tail_width >= ragged.tokens_new[0] - ragged.base_new[0]);
        assert!(ragged.tail_width >= ragged.tokens_new[1] - ragged.base_new[1]);
    }

    /// A window that would have to start before the retained rows is a
    /// *silent* history gap if it is not caught here.
    #[test]
    fn a_history_gap_is_refused_and_names_the_row() {
        // A 5-wide window is enough for row 0 (residue 0: needs `r + (span-1) *
        // ratio = 4`) and one short for row 1 (residue 2: needs 6). Row 1's
        // block 4 would have to read from token 12; its window starts at 13.
        let err = plan_xs_advance(&[16, 18], &[12, 14], 5, 4, 4, 2, 16)
            .unwrap_err()
            .to_string();
        assert!(err.contains("compressor history gap"), "{err}");
        assert!(err.contains("row 1"), "{err}");
        // …and the same batch with a window wide enough for both is fine.
        plan_xs_advance(&[16, 18], &[12, 14], 6, 4, 4, 2, 16).unwrap();
    }

    /// `set_len` on a ragged cache cannot be expressed by one narrow of an
    /// end-anchored window; it must refuse rather than silently truncate the
    /// wrong tokens.
    #[test]
    fn a_ragged_cache_refuses_a_single_shared_rollback() {
        let mut xs = XsRollingCache::new(4, 2, 8, 512);
        xs.tokens = vec![16, 20];
        xs.base = vec![8, 12];
        let err = xs.try_set_len(12).unwrap_err().to_string();
        assert!(err.contains("per-row token counts"), "{err}");
        // …and a uniform one still works exactly as before.
        xs.tokens = vec![20, 20];
        xs.base = vec![12, 12];
        xs.try_set_len(16).unwrap();
    }

    /// A batch that loses a sequence hands the next forward fewer rows. While
    /// every row agrees that is exact — the per-row vectors carry no
    /// information a uniform batch does not already have — and it is what every
    /// pre-existing caller (batch shrink, batch grow, a freshly reset cache)
    /// relies on. Once the rows disagree it stops being exact: which rows
    /// survived is the caller's knowledge, and guessing hands one sequence
    /// another's compressed history.
    #[test]
    fn reshaping_is_exact_while_uniform_and_refused_once_ragged() {
        let mut xs = XsRollingCache::new(4, 2, 8, 512);
        xs.assign_row_lens(vec![24, 24], vec![16, 16]);
        xs.ensure_rows(1).unwrap();
        assert_eq!(xs.row_lens(), (&[24usize][..], &[16usize][..]));
        xs.ensure_rows(3).unwrap();
        assert_eq!(xs.row_lens().0, &[24, 24, 24]);

        xs.assign_row_lens(vec![24, 25], vec![16, 16]);
        let err = xs.ensure_rows(1).unwrap_err().to_string();
        assert!(err.contains("different token counts"), "{err}");
        assert!(
            err.contains("split_row"),
            "the refusal must name the way out; got {err}"
        );
    }

    /// The flag is OFF unless explicitly asked for. Read once, latched, so a
    /// mid-run env change cannot split the batch's behaviour.
    #[test]
    fn per_sequence_xs_is_off_unless_requested() {
        assert!(
            !xs_per_sequence_enabled(),
            "ARC_V4_XS_PER_SEQ must default to off; the test process must not set it"
        );
        // …and the slot declines to carry its own length while it is off, which
        // is what keeps `MtpSpeculativePipeline::kv_advance` on the cohort rule.
        let xs = super::super::KvCache::XsRolling(Box::new(XsRollingCache::new(4, 2, 8, 512)));
        assert!(!xs.supports_per_sequence_len());
        test_override::with(true, || {
            assert!(
                xs.supports_per_sequence_len(),
                "with the flag on the compressor slot must stop being the blocker"
            );
        });
        assert!(
            !xs.supports_per_sequence_len(),
            "the override must not leak"
        );
    }
}
