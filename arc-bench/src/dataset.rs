//! Dataset loader: walk a directory of trajectory JSON files and
//! present them as an iterable, deterministic collection.

use crate::trajectory::{ParseError, Trajectory};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

/// In-memory dataset.
#[derive(Debug, Clone)]
pub struct Dataset {
    pub name: String,
    pub root: PathBuf,
    pub trajectories: Vec<Trajectory>,
}

#[derive(Debug, thiserror::Error)]
pub enum DatasetError {
    #[error("io error scanning {path}: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("trajectory parse error: {0}")]
    Parse(#[from] ParseError),
}

impl Dataset {
    /// Load every `*.json` file under `<root>/trajectories/`.
    pub fn load(root: impl AsRef<Path>) -> Result<Self, DatasetError> {
        let root_path = root.as_ref().to_path_buf();
        let traj_dir = root_path.join("trajectories");
        let name = root_path
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "unnamed".to_string());

        let entries = fs::read_dir(&traj_dir).map_err(|e| DatasetError::Io {
            path: traj_dir.display().to_string(),
            source: e,
        })?;

        let mut trajectories = Vec::new();
        for entry in entries {
            let entry = entry.map_err(|e| DatasetError::Io {
                path: traj_dir.display().to_string(),
                source: e,
            })?;
            let path = entry.path();
            if path.extension().and_then(|x| x.to_str()) != Some("json") {
                continue;
            }
            let trajectory = Trajectory::from_path(&path)?;
            trajectories.push(trajectory);
        }

        trajectories.sort_by(|a, b| a.id.cmp(&b.id));

        Ok(Self {
            name,
            root: root_path,
            trajectories,
        })
    }

    pub fn len(&self) -> usize {
        self.trajectories.len()
    }

    pub fn is_empty(&self) -> bool {
        self.trajectories.is_empty()
    }

    pub fn per_language(&self) -> BTreeMap<String, usize> {
        let mut out = BTreeMap::new();
        for t in &self.trajectories {
            *out.entry(t.language.clone()).or_insert(0) += 1;
        }
        out
    }

    pub fn token_totals(&self) -> (u64, u64) {
        let mut input = 0u64;
        let mut output = 0u64;
        for t in &self.trajectories {
            input += t.total_input_tokens_est();
            output += t.total_output_tokens_est();
        }
        (input, output)
    }

    pub fn length_distribution(&self) -> LengthDistribution {
        let mut short = 0u32;
        let mut medium = 0u32;
        let mut long = 0u32;
        for t in &self.trajectories {
            match t.assistant_turn_count() {
                0..=2 => short += 1,
                3..=5 => medium += 1,
                _ => long += 1,
            }
        }
        LengthDistribution {
            short,
            medium,
            long,
        }
    }

    pub fn iter(&self) -> std::slice::Iter<'_, Trajectory> {
        self.trajectories.iter()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LengthDistribution {
    pub short: u32,
    pub medium: u32,
    pub long: u32,
}
