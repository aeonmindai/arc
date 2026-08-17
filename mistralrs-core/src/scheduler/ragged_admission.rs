//! Whether one forward pass may carry sequences that sit at **different** cache
//! lengths — and the one rule the three layers that must agree about it read.
//!
//! # The assumption this removes
//!
//! Every layer of the decode path used to encode *"every sequence in a batch
//! sits at the same position"* as a different batch-wide scalar. PRs #92, #95,
//! #100, #102 and #103 removed it from the KV advance, the `xs` compressor
//! state, the attention mask, the fused step's `cache_len` and the MTP draft
//! chain. This module removes it from the layer that **admits** a batch, which
//! is where it was doing the most damage — because it was doing that damage to
//! *every* run, not only to speculative ones.
//!
//! Two sites carried it:
//!
//! * `scheduler/default_scheduler.rs` partitions the running set by
//!   [`crate::sequence::Sequence::cache_bucket_len`] and runs **exactly one
//!   bucket per step**. Sequences whose cache lengths differ therefore cannot
//!   share a forward pass *at all* — they take turns.
//! * `engine/mod.rs` issues [`crate::pipeline::CacheInstruction::In`] only when
//!   the scheduled completion-id list *changed*, so a stable cohort takes
//!   `Nothing` on every steady-state step and the batched cache is never
//!   re-assembled — which is exactly when a ragged cohort's front-alignment
//!   would have to be recomputed.
//!
//! # Why this is a general throughput bug, not an MTP one
//!
//! In real serving, requests arrive at different times with different prompt
//! lengths, so cache lengths **always** diverge. Length-bucketed admission
//! shatters precisely the batches a fleet claim depends on. The scheduler
//! simulation `bucket_shattering_law` (CPU, `default_scheduler.rs`) measures the
//! rule: with `B` running sequences spread over `D` distinct cache lengths far
//! enough apart that coalescing is refused, the mean number of sequences that
//! actually run per step is `B / D`. At `B = 128` over 128 distinct lengths
//! 64 tokens apart it is **1.00** — 0.8% of the admitted batch, in steady state.
//!
//! # The gate
//!
//! Admission is granted only when the pipeline **declares** it can serve a
//! ragged cohort ([`crate::pipeline::CacheManagerMixin::ragged_batch_admission`]).
//! The default declaration is a refusal carrying a reason, so a pipeline that
//! has not been taught to mask a left-aligned batch keeps the historical rule
//! and the run is *told which layer refused* rather than quietly served the
//! ceiling. That is D18: the absence of a signal must not be readable as a
//! specific signal.

/// The admission layer's decision for one run.
///
/// Constructed once, in `Engine::new`, from the pipeline's declaration; read by
/// the scheduler (which bucket key to build), by the engine (when to issue
/// `CacheInstruction::In`) and by the pipeline itself (whether the cohort it is
/// handed may be ragged). Holding it in one type is what stops the three from
/// drifting apart — each of them reading the same value is checkable, three
/// independent booleans are not.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct RaggedAdmission {
    granted: bool,
    refusal: Option<String>,
}

impl RaggedAdmission {
    /// The historical rule: one batch, one cache length. No reason recorded —
    /// nothing asked for anything else.
    pub fn refused() -> Self {
        Self {
            granted: false,
            refusal: None,
        }
    }

    /// Turn a pipeline's declaration into the run's decision.
    ///
    /// `Ok(())` grants; `Err(why)` refuses **and keeps `why`**, so the engine
    /// can log which layer said no. A refusal that loses its reason is the
    /// silent-fallback shape D18 exists to forbid.
    pub fn decide(declared: Result<(), String>) -> Self {
        match declared {
            Ok(()) => Self {
                granted: true,
                refusal: None,
            },
            Err(why) => Self {
                granted: false,
                refusal: Some(why),
            },
        }
    }

    /// May a batch carry sequences at different cache lengths?
    pub const fn granted(&self) -> bool {
        self.granted
    }

    /// Why not, when the pipeline declared a reason.
    pub fn refusal(&self) -> Option<&str> {
        self.refusal.as_deref()
    }

    /// Does the scheduler still partition its running set by cache length?
    ///
    /// The inverse of [`Self::granted`], named at the site that reads it so a
    /// mutation that flips one consumer and not the other is visible.
    pub const fn buckets_by_cache_length(&self) -> bool {
        !self.granted
    }

    /// Must the engine rebuild the batched cache on **every** decode step?
    ///
    /// A ragged cohort's rows are made to share one append offset by
    /// left-aligning them ([`crate::kv_cache::front_align_batch`]), and the
    /// alignment is a function of the batch's current lengths. Rows that advance
    /// by different amounts change it every step, so the `In`/`Nothing`
    /// optimisation — which is keyed on the *identity* of the cohort, not on its
    /// lengths — cannot be kept. It is also what sets the pipeline's
    /// `assembled_here`, which is the only place a per-row dead prefix is
    /// reported at all.
    pub const fn assembles_every_step(&self) -> bool {
        self.granted
    }
}

/// The engine's `pre_op` rule, as one expression.
///
/// 🔑 Structural, not numeric: with `granted == false` this is
/// `!no_kv_cache && cohort_changed` — the pre-change condition, character for
/// character — so a run that does not grant ragged admission issues exactly the
/// cache instructions it always did. There is no batch for which the two
/// branches differ and no test could tell them apart, which is why the identity
/// is asserted on the *predicate* rather than inferred from a decode result.
pub const fn issues_cache_in(no_kv_cache: bool, cohort_changed: bool, granted: bool) -> bool {
    !no_kv_cache && (cohort_changed || granted)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_refusal_keeps_its_reason() {
        let a = RaggedAdmission::decide(Err("cache slot 3 is a `Rotating`".to_string()));
        assert!(!a.granted());
        assert_eq!(a.refusal(), Some("cache slot 3 is a `Rotating`"));
        assert!(a.buckets_by_cache_length());
        assert!(!a.assembles_every_step());
    }

    #[test]
    fn a_grant_reaches_both_consumers() {
        let a = RaggedAdmission::decide(Ok(()));
        assert!(a.granted());
        assert!(a.refusal().is_none());
        // Both sites must move together: a grant that reached the scheduler but
        // not the engine would form ragged cohorts and then never re-align
        // them, which `ensure_uniform_batch_cache_lens` turns into a failed
        // request on the first divergent step.
        assert!(!a.buckets_by_cache_length());
        assert!(a.assembles_every_step());
    }

    /// The default is the historical rule, with no reason — nothing asked.
    #[test]
    fn the_default_is_the_historical_rule() {
        assert_eq!(RaggedAdmission::default(), RaggedAdmission::refused());
        assert!(!RaggedAdmission::refused().granted());
        assert!(RaggedAdmission::refused().refusal().is_none());
    }

    /// 🔑 Flag-off byte-identity for the engine site, asserted structurally:
    /// over every input, `granted == false` reproduces `!no_kv_cache &&
    /// cohort_changed` exactly.
    #[test]
    fn a_refused_run_issues_exactly_the_pre_change_cache_instructions() {
        for &no_kv in &[false, true] {
            for &changed in &[false, true] {
                assert_eq!(
                    issues_cache_in(no_kv, changed, false),
                    !no_kv && changed,
                    "no_kv_cache={no_kv} cohort_changed={changed}"
                );
            }
        }
    }

    /// And a granted run assembles on every step that has a cache at all —
    /// including the steady-state steps whose cohort id list did not change,
    /// which is the whole point of site (1).
    #[test]
    fn a_granted_run_assembles_on_a_stable_cohort() {
        assert!(
            issues_cache_in(false, false, true),
            "a stable cohort must still re-assemble, or the front-alignment is the \
             previous step's"
        );
        assert!(
            !issues_cache_in(true, true, true),
            "no_kv_cache still wins: there is no cache to assemble"
        );
    }
}
