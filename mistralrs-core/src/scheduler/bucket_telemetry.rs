//! Parent system: ArcInfer / ArcSched
//!
//! Process-wide counters for the one thing that decides how wide a decode step
//! actually is: **bucketing**.
//!
//! Both schedulers partition the running set by an exact cache length and run
//! **one bucket per step**, preempting the rest
//! (`default_scheduler.rs:bucket_and_waitlist_seqs_waiting`,
//! `paged_attention/scheduler.rs`). So a throughput number measured at "B=128"
//! is not necessarily a number about 128-wide steps — if the batch shattered
//! into 40 buckets, the engine ran 3-wide steps and the label lies.
//!
//! That is not a hypothetical confound for the per-sequence-advance A/B: the
//! whole point of ragged admission ([`super::RaggedAdmission`]) is to collapse
//! those buckets into one, so an aggregate throughput delta between the ON and
//! OFF arms could come from the KV mechanism, from the bucketing, or from both.
//! Without these counters the two are indistinguishable, and "aggregate did not
//! move" would be unattributable.
//!
//! The marker is emitted on the same log fence as `MTP[agg]`, so a harness that
//! differences cumulative counters across a wall-clock boundary gets the
//! scheduler's numbers for exactly the window it got MTP's.

use std::sync::atomic::{AtomicUsize, Ordering};

/// Cumulative bucketing counters. Monotone; difference two snapshots to get a
/// window.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SchedBuckets {
    /// Bucketing calls with at least one running sequence.
    pub calls: usize,
    /// Sum of the number of buckets formed.
    pub buckets: usize,
    /// Sum of the running-set size *before* selection — the batch the engine
    /// was offered.
    pub offered: usize,
    /// Sum of the winning bucket's size — the batch the engine actually ran.
    pub chosen: usize,
    /// Calls that formed more than one bucket, i.e. that preempted somebody.
    pub shattered: usize,
}

impl SchedBuckets {
    /// `self - earlier`, for differencing across a fence. Saturating, so a
    /// counter reset between snapshots degrades to zero rather than to a
    /// gigantic wrapped number that would read as a spectacular result.
    #[must_use]
    pub fn since(&self, earlier: &Self) -> Self {
        Self {
            calls: self.calls.saturating_sub(earlier.calls),
            buckets: self.buckets.saturating_sub(earlier.buckets),
            offered: self.offered.saturating_sub(earlier.offered),
            chosen: self.chosen.saturating_sub(earlier.chosen),
            shattered: self.shattered.saturating_sub(earlier.shattered),
        }
    }

    #[allow(clippy::cast_precision_loss)]
    fn per_call(&self, total: usize) -> Option<f64> {
        (self.calls > 0).then(|| total as f64 / self.calls as f64)
    }

    /// Mean buckets formed per scheduling step. 1.0 means the batch was never
    /// split; 40.0 at a nominal B=128 means the engine ran ~3-wide steps.
    #[must_use]
    pub fn buckets_per_step(&self) -> Option<f64> {
        self.per_call(self.buckets)
    }

    /// Mean size of the bucket that actually ran — the **real** batch width,
    /// as opposed to the number of sequences in flight.
    #[must_use]
    pub fn running_bucket_size(&self) -> Option<f64> {
        self.per_call(self.chosen)
    }

    /// Mean number of running sequences offered to the scheduler per step.
    #[must_use]
    pub fn offered_per_step(&self) -> Option<f64> {
        self.per_call(self.offered)
    }

    /// Mean sequences preempted per step (`offered - chosen`).
    #[must_use]
    pub fn preempted_per_step(&self) -> Option<f64> {
        self.per_call(self.offered.saturating_sub(self.chosen))
    }

    /// Fraction of steps that split the batch at all.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn shattered_frac(&self) -> Option<f64> {
        (self.calls > 0).then(|| self.shattered as f64 / self.calls as f64)
    }

    /// The machine-greppable one-liner, in the project's marker convention
    /// (`MTP[...]`, `SPEED[...]`). Every raw count is on the line so the ratios
    /// are auditable without trusting the formatter.
    #[must_use]
    pub fn marker(&self, scope: &str) -> String {
        let fmt = |v: Option<f64>| v.map_or_else(|| "n/a".to_string(), |x| format!("{x:.4}"));
        format!(
            "SCHED[{scope}] calls={} buckets={} offered={} chosen={} shattered={} \
             buckets_per_step={} running_bucket_size={} offered_per_step={} \
             preempted_per_step={} shattered_frac={}",
            self.calls,
            self.buckets,
            self.offered,
            self.chosen,
            self.shattered,
            fmt(self.buckets_per_step()),
            fmt(self.running_bucket_size()),
            fmt(self.offered_per_step()),
            fmt(self.preempted_per_step()),
            fmt(self.shattered_frac()),
        )
    }
}

static CALLS: AtomicUsize = AtomicUsize::new(0);
static BUCKETS: AtomicUsize = AtomicUsize::new(0);
static OFFERED: AtomicUsize = AtomicUsize::new(0);
static CHOSEN: AtomicUsize = AtomicUsize::new(0);
static SHATTERED: AtomicUsize = AtomicUsize::new(0);

/// Record one bucketing decision. Called once per scheduling step that had
/// anything to schedule.
pub fn record_bucketing(buckets: usize, offered: usize, chosen: usize) {
    if offered == 0 {
        return;
    }
    CALLS.fetch_add(1, Ordering::Relaxed);
    BUCKETS.fetch_add(buckets, Ordering::Relaxed);
    OFFERED.fetch_add(offered, Ordering::Relaxed);
    CHOSEN.fetch_add(chosen, Ordering::Relaxed);
    if buckets > 1 {
        SHATTERED.fetch_add(1, Ordering::Relaxed);
    }
}

/// Snapshot of the process-wide counters.
#[must_use]
pub fn sched_buckets() -> SchedBuckets {
    SchedBuckets {
        calls: CALLS.load(Ordering::Relaxed),
        buckets: BUCKETS.load(Ordering::Relaxed),
        offered: OFFERED.load(Ordering::Relaxed),
        chosen: CHOSEN.load(Ordering::Relaxed),
        shattered: SHATTERED.load(Ordering::Relaxed),
    }
}

/// The aggregate `SCHED[agg] …` line, or `None` when nothing was ever
/// scheduled — the honest answer to "how wide were the steps" when none ran is
/// *nothing*, not `0`.
#[must_use]
pub fn sched_bucket_marker() -> Option<String> {
    let snap = sched_buckets();
    (snap.calls > 0).then(|| snap.marker("agg"))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The ratios are what get quoted, so they are what must be pinned. A batch
    /// that shatters is the case the counters exist to make visible.
    #[test]
    fn a_shattered_batch_reports_the_width_that_actually_ran() {
        // Three steps: 128 offered, split 40 ways, 3 of them ran.
        let mut snap = SchedBuckets::default();
        for _ in 0..3 {
            snap.calls += 1;
            snap.buckets += 40;
            snap.offered += 128;
            snap.chosen += 3;
            snap.shattered += 1;
        }
        assert_eq!(snap.buckets_per_step(), Some(40.0));
        assert_eq!(snap.offered_per_step(), Some(128.0));
        // 🔑 The number a throughput result has to be read against: the engine
        // ran 3-wide steps while the harness called it B=128.
        assert_eq!(snap.running_bucket_size(), Some(3.0));
        assert_eq!(snap.preempted_per_step(), Some(125.0));
        assert_eq!(snap.shattered_frac(), Some(1.0));
        let m = snap.marker("agg");
        assert!(m.contains("running_bucket_size=3.0000"), "{m}");
        assert!(m.contains("buckets_per_step=40.0000"), "{m}");
    }

    /// An unshattered batch must be visibly different, or the counter cannot
    /// distinguish "the scheduler serialised us" from "it did not".
    #[test]
    fn an_intact_batch_reports_one_bucket_and_no_preemption() {
        let snap = SchedBuckets {
            calls: 10,
            buckets: 10,
            offered: 1280,
            chosen: 1280,
            shattered: 0,
        };
        assert_eq!(snap.buckets_per_step(), Some(1.0));
        assert_eq!(snap.running_bucket_size(), Some(128.0));
        assert_eq!(snap.preempted_per_step(), Some(0.0));
        assert_eq!(snap.shattered_frac(), Some(0.0));
    }

    /// Differencing across a fence is how a cell's numbers are taken; a reset
    /// in between must not read as a huge result.
    #[test]
    fn differencing_saturates_rather_than_wrapping() {
        let early = SchedBuckets {
            calls: 100,
            buckets: 200,
            offered: 1000,
            chosen: 500,
            shattered: 50,
        };
        let late = SchedBuckets {
            calls: 150,
            buckets: 260,
            offered: 1600,
            chosen: 900,
            shattered: 60,
        };
        let d = late.since(&early);
        assert_eq!(d.calls, 50);
        assert_eq!(d.buckets, 60);
        assert_eq!(d.running_bucket_size(), Some(8.0));
        // A counter that went backwards (process restart) reads as zero work,
        // never as a wrapped `usize`.
        assert_eq!(early.since(&late), SchedBuckets::default());
    }

    /// Nothing scheduled means no marker, not a marker full of zeros.
    #[test]
    fn no_scheduling_reports_nothing_rather_than_zero() {
        assert!(SchedBuckets::default().buckets_per_step().is_none());
        assert!(SchedBuckets::default().running_bucket_size().is_none());
    }
}
