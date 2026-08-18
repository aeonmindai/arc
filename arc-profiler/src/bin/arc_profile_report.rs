//! Turn one or more profile JSONs into one interactive HTML page.
//!
//! ```text
//! arc-profile-report -o batch-sweep.html b1.json b64.json b256.json
//! ```
//!
//! With more than one input the page gains a comparison view keyed by node
//! path, which is how a B=1 / B=64 / B=256 sweep gets read side by side. Never
//! needs a GPU and never touches the network.

use std::path::PathBuf;
use std::process::ExitCode;

use arc_profiler::{html, Profile};

fn main() -> ExitCode {
    let mut out = PathBuf::from("arc-profile.html");
    let mut inputs: Vec<PathBuf> = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "-o" | "--out" => match args.next() {
                Some(p) => out = PathBuf::from(p),
                None => {
                    eprintln!("--out needs a path");
                    return ExitCode::FAILURE;
                }
            },
            "-h" | "--help" => {
                eprintln!(
                    "usage: arc-profile-report [-o OUT.html] RUN.json [RUN.json ...]\n\
                     \n\
                     Renders arc-profiler JSON into one self-contained interactive page.\n\
                     Pass several runs (e.g. B=1, B=64, B=256) to get the comparison view."
                );
                return ExitCode::SUCCESS;
            }
            _ => inputs.push(PathBuf::from(a)),
        }
    }

    if inputs.is_empty() {
        eprintln!("no input JSON given; try --help");
        return ExitCode::FAILURE;
    }

    let mut profiles: Vec<Profile> = Vec::with_capacity(inputs.len());
    for path in &inputs {
        let bytes = match std::fs::read(path) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("{}: {e}", path.display());
                return ExitCode::FAILURE;
            }
        };
        match serde_json::from_slice::<Profile>(&bytes) {
            Ok(p) => {
                if p.schema != arc_profiler::SCHEMA {
                    // Refuse rather than render a page whose columns silently
                    // mean something else.
                    eprintln!(
                        "{}: schema is `{}`, this binary reads `{}`",
                        path.display(),
                        p.schema,
                        arc_profiler::SCHEMA
                    );
                    return ExitCode::FAILURE;
                }
                profiles.push(p);
            }
            Err(e) => {
                eprintln!("{}: {e}", path.display());
                return ExitCode::FAILURE;
            }
        }
    }

    let page = html::render(&profiles);
    if let Err(e) = std::fs::write(&out, page.as_bytes()) {
        eprintln!("{}: {e}", out.display());
        return ExitCode::FAILURE;
    }
    eprintln!(
        "wrote {} ({} runs, {} KiB)",
        out.display(),
        profiles.len(),
        page.len() / 1024
    );
    ExitCode::SUCCESS
}
