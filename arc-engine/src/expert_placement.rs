//! Offline expert → device placement for expert parallelism.
//!
//! # Why this can be computed with no GPU
//!
//! V4's first `num_hash_layers` layers (3, in V4 Flash) do **not** route by
//! score. They use TD-MoE hash routing: `gate.tid2eid` is a fixed
//! `[vocab_size, top_k]` I64 table and a token's experts are
//! `tid2eid[token_id]`, unconditionally
//! (`mistralrs-core/src/models/deepseek4.rs`, `MoeGate::tid2eid`; reference
//! `inference/model.py`, `Gate.hash = layer_id < n_hash_layers`).
//!
//! Their expert load is therefore a **deterministic function of the tokenizer
//! and the corpus**, exactly computable offline. wave44-BV §3.2 flagged this
//! as "free to check and nobody has". This module checks it, and turns the
//! answer into an expert→rank assignment.
//!
//! # What `ep_size` does
//!
//! [`DeepSeekV4Config::ep_size`](crate::deepseek_v4::DeepSeekV4Config::ep_size)
//! is deserialized straight from the HF `config.json`. Until wave60-CK nothing
//! in the workspace read it. [`plan_placement`] is the offline half of that
//! wiring: it turns `ep_size` plus a hash table into the placement the serving
//! path consumes. The runtime half lives in
//! `mistralrs-core/src/models/deepseek4.rs` (`build_expert_parallel_plan`).

use mistralrs_core::{balancedness, Balancedness, ExpertPlacement};

use crate::deepseek_v4::DeepSeekV4Config;

/// Per-expert load implied by a hash-routing table, assuming every token id is
/// equally likely.
///
/// `table` is the flattened row-major `[vocab_size, top_k]` `gate.tid2eid`.
/// The load of expert `e` is the number of table entries naming it — with a
/// uniform token distribution that is proportional to the tokens it receives.
///
/// Uniform token frequency is the *pessimistic* assumption for balance: real
/// corpora concentrate on a small part of the vocabulary, which can only make
/// the distribution more skewed, never less. Use [`expert_loads_weighted`]
/// when a token histogram is available.
pub fn expert_loads_from_tid2eid(
    table: &[i64],
    top_k: usize,
    num_experts: usize,
) -> Result<Vec<f64>, String> {
    if top_k == 0 {
        return Err("tid2eid: top_k must be >= 1".to_string());
    }
    if !table.len().is_multiple_of(top_k) {
        return Err(format!(
            "tid2eid: table length {} is not a multiple of top_k {top_k}",
            table.len()
        ));
    }
    let vocab = table.len() / top_k;
    expert_loads_weighted(table, top_k, num_experts, &vec![1.0; vocab])
}

/// Per-expert load implied by a hash-routing table under a measured token
/// frequency distribution. `token_freq[t]` is how often token id `t` occurs.
pub fn expert_loads_weighted(
    table: &[i64],
    top_k: usize,
    num_experts: usize,
    token_freq: &[f64],
) -> Result<Vec<f64>, String> {
    if top_k == 0 {
        return Err("tid2eid: top_k must be >= 1".to_string());
    }
    if table.len() != token_freq.len() * top_k {
        return Err(format!(
            "tid2eid: table holds {} entries but token_freq has {} tokens at top_k {top_k}",
            table.len(),
            token_freq.len()
        ));
    }
    let mut loads = vec![0.0f64; num_experts];
    for (token, freq) in token_freq.iter().enumerate() {
        for slot in 0..top_k {
            let e = table[token * top_k + slot];
            if e < 0 {
                return Err(format!("tid2eid: negative expert id {e} at token {token}"));
            }
            let e = e as usize;
            if e >= num_experts {
                return Err(format!(
                    "tid2eid: expert id {e} at token {token} exceeds num_experts {num_experts}"
                ));
            }
            loads[e] += freq;
        }
    }
    Ok(loads)
}

/// Build the expert → rank assignment this config asks for.
///
/// `ep_size == 1` yields the trivial single-rank placement. Otherwise the
/// experts are bin-packed by `loads` so the ranks carry equal work — the same
/// greedy shape DeepSeek's EPLB uses, minus the redundant-expert replication
/// stage 1 deliberately does not ship.
pub fn plan_placement(cfg: &DeepSeekV4Config, loads: &[f64]) -> Result<ExpertPlacement, String> {
    let ep_size = cfg.ep_size.max(1);
    if loads.len() != cfg.n_routed_experts {
        return Err(format!(
            "expert placement: {} loads for {} routed experts",
            loads.len(),
            cfg.n_routed_experts
        ));
    }
    if ep_size == 1 {
        return ExpertPlacement::contiguous(loads.len(), 1).map_err(|e| e.to_string());
    }
    ExpertPlacement::balanced(loads, ep_size).map_err(|e| e.to_string())
}

/// How much better (or worse) a placement is than plain contiguous blocks,
/// on a given per-expert load.
#[derive(Debug, Clone, PartialEq)]
pub struct PlacementComparison {
    pub contiguous: Balancedness,
    pub planned: Balancedness,
}

impl PlacementComparison {
    /// Imbalance removed, as a fraction of the contiguous imbalance. `0.0`
    /// means the plan is no better; `1.0` means it is perfectly balanced.
    pub fn improvement(&self) -> f64 {
        let excess = self.contiguous.ratio - 1.0;
        if excess <= 0.0 {
            return 0.0;
        }
        ((excess - (self.planned.ratio - 1.0)) / excess).clamp(0.0, 1.0)
    }
}

/// Compare the planned placement against contiguous blocks on the same loads.
pub fn compare_to_contiguous(
    planned: &ExpertPlacement,
    loads: &[f64],
) -> Result<PlacementComparison, String> {
    let contiguous =
        ExpertPlacement::contiguous(loads.len(), planned.ep_size()).map_err(|e| e.to_string())?;
    Ok(PlacementComparison {
        contiguous: balancedness(
            &contiguous
                .per_rank_totals(loads)
                .map_err(|e| e.to_string())?,
        ),
        planned: balancedness(&planned.per_rank_totals(loads).map_err(|e| e.to_string())?),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A config with every field at its serde default: the published V4
    /// `config.json` supplies all of them, and every field in the struct
    /// carries a `#[serde(default)]`, so an empty object is a valid config.
    fn default_cfg() -> DeepSeekV4Config {
        serde_json::from_str("{}").expect("every DeepSeekV4Config field has a serde default")
    }

    /// A hash table in which experts are used with wildly different
    /// frequencies — expert `e` appears `e + 1` times.
    fn skewed_table(num_experts: usize, top_k: usize) -> (Vec<i64>, usize) {
        let mut entries: Vec<i64> = Vec::new();
        for e in 0..num_experts {
            for _ in 0..(e + 1) {
                entries.push(e as i64);
            }
        }
        // Pad to a whole number of rows.
        while !entries.len().is_multiple_of(top_k) {
            entries.push(0);
        }
        let vocab = entries.len() / top_k;
        (entries, vocab)
    }

    #[test]
    fn loads_count_table_entries_per_expert() {
        // vocab 3, top_k 2: token 0 → {0,1}, token 1 → {1,2}, token 2 → {2,2}.
        let table = vec![0i64, 1, 1, 2, 2, 2];
        let loads = expert_loads_from_tid2eid(&table, 2, 3).unwrap();
        assert_eq!(loads, vec![1.0, 2.0, 3.0]);
    }

    /// Token frequency must actually change the answer, otherwise the weighted
    /// form is decorative.
    #[test]
    fn token_frequency_reweights_the_loads() {
        let table = vec![0i64, 1, 1, 2, 2, 2];
        let uniform = expert_loads_from_tid2eid(&table, 2, 3).unwrap();
        let weighted = expert_loads_weighted(&table, 2, 3, &[10.0, 1.0, 1.0]).unwrap();
        assert_eq!(weighted, vec![10.0, 11.0, 3.0]);
        assert_ne!(uniform, weighted);
    }

    #[test]
    fn a_table_naming_an_out_of_range_expert_is_rejected() {
        let table = vec![0i64, 7];
        let err = expert_loads_from_tid2eid(&table, 2, 4).unwrap_err();
        assert!(err.contains("exceeds num_experts"), "{err}");

        let negative = vec![0i64, -1];
        let err = expert_loads_from_tid2eid(&negative, 2, 4).unwrap_err();
        assert!(err.contains("negative expert id"), "{err}");
    }

    #[test]
    fn ep_size_one_gives_the_trivial_placement() {
        let mut cfg = default_cfg();
        cfg.n_routed_experts = 8;
        cfg.ep_size = 1;
        let placement = plan_placement(&cfg, &[1.0; 8]).unwrap();
        assert_eq!(placement.ep_size(), 1);
        assert_eq!(placement.num_local_experts(), 8);
    }

    /// `ep_size` from the config is what decides the split — this is the
    /// assertion that the field is no longer inert.
    #[test]
    fn ep_size_from_the_config_decides_the_split() {
        let mut cfg = default_cfg();
        cfg.n_routed_experts = 8;
        for (ep_size, per_rank) in [(2usize, 4usize), (4, 2), (8, 1)] {
            cfg.ep_size = ep_size;
            let placement = plan_placement(&cfg, &[1.0; 8]).unwrap();
            assert_eq!(placement.ep_size(), ep_size);
            assert_eq!(placement.num_local_experts(), per_rank);
        }
    }

    /// The end-to-end offline question: given a real-shaped skewed hash table,
    /// does the planned placement actually beat contiguous blocks?
    #[test]
    fn planning_from_a_skewed_hash_table_beats_contiguous_blocks() {
        let (table, vocab) = skewed_table(256, 6);
        assert_eq!(table.len(), vocab * 6);
        let loads = expert_loads_from_tid2eid(&table, 6, 256).unwrap();

        let mut cfg = default_cfg();
        cfg.n_routed_experts = 256;
        cfg.ep_size = 2;
        let planned = plan_placement(&cfg, &loads).unwrap();

        let cmp = compare_to_contiguous(&planned, &loads).unwrap();
        assert!(
            cmp.contiguous.ratio > 1.4,
            "fixture cannot discriminate: contiguous is already balanced ({:?})",
            cmp.contiguous
        );
        assert!(
            cmp.planned.ratio < 1.001,
            "planned placement is still skewed: {:?}",
            cmp.planned
        );
        assert!(
            cmp.improvement() > 0.99,
            "improvement {}",
            cmp.improvement()
        );
    }

    /// A **uniform** table must be reported as already balanced — otherwise
    /// the previous test would pass on any input and prove nothing about the
    /// planner. This is the D12 pair: it asserts the metric can say "no
    /// improvement available".
    #[test]
    fn a_uniform_hash_table_needs_no_rebalancing() {
        let mut table: Vec<i64> = Vec::new();
        for t in 0..256usize {
            for slot in 0..6usize {
                table.push(((t + slot) % 256) as i64);
            }
        }
        let loads = expert_loads_from_tid2eid(&table, 6, 256).unwrap();
        assert!(loads.iter().all(|&l| (l - 6.0).abs() < 1e-9));

        let mut cfg = default_cfg();
        cfg.n_routed_experts = 256;
        cfg.ep_size = 2;
        let planned = plan_placement(&cfg, &loads).unwrap();
        let cmp = compare_to_contiguous(&planned, &loads).unwrap();
        assert!((cmp.contiguous.ratio - 1.0).abs() < 1e-9);
        assert!((cmp.planned.ratio - 1.0).abs() < 1e-9);
        assert_eq!(cmp.improvement(), 0.0);
    }

    #[test]
    fn a_load_vector_of_the_wrong_length_is_rejected() {
        let mut cfg = default_cfg();
        cfg.n_routed_experts = 8;
        cfg.ep_size = 2;
        let err = plan_placement(&cfg, &[1.0; 7]).unwrap_err();
        assert!(err.contains("7 loads for 8 routed experts"), "{err}");
    }
}
