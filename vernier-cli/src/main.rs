//! `vernier` — the CLI front-end and benchmark harness for vernier-rs.
//!
//! This binary is the one place that names concrete backends. It parses args,
//! resolves a [`BackendKind`](backend_select::BackendKind), and
//! [`dispatch`](backend_select::dispatch)es a backend-generic
//! [`BackendTask`](backend_select::BackendTask). Swapping CPU for GPU is the
//! `--backend` flag; nothing in the library changes.

mod annotate;
mod args;
mod backend_select;
mod commands;
mod imageio;
mod pgm;

use args::{Command, TopLevel};
use backend_select::{BackendKind, dispatch};
use commands::benchmark::Benchmark;
use commands::detect_megarena::DetectMegarena;

fn main() {
    let top: TopLevel = argh::from_env();

    match top.command {
        Command::Bench(a) => {
            let Some(kind) = BackendKind::parse(&a.backend) else {
                eprintln!("unknown backend '{}'. try: cpu, gpu", a.backend);
                std::process::exit(2);
            };
            let task = Benchmark {
                size: a.size,
                iterations: a.iters,
                sigma: a.sigma,
                min_frequency: a.min_frequency,
                max_frequency: a.max_frequency,
                smoothing_sigma: a.smoothing_sigma,
            };
            let r = dispatch(kind, &task);
            println!(
                "backend={} size={}x{} iters={} mean={:.2}ms best={:.2}ms",
                r.backend, r.size, r.size, r.iterations, r.mean_ms, r.best_ms
            );
        }
        Command::DetectMegarena(a) => {
            let Some(kind) = BackendKind::parse(&a.backend) else {
                eprintln!("unknown backend '{}'. try: cpu", a.backend);
                std::process::exit(2);
            };
            let task = DetectMegarena {
                image_path: std::path::PathBuf::from(&a.image),
                physical_period: a.period,
                code_size: a.code_size,
                sigma: a.sigma,
                min_frequency: a.min_frequency,
                max_frequency: a.max_frequency,
                smoothing_sigma: a.smoothing_sigma,
                debug_image: a.debug_image.as_ref().map(std::path::PathBuf::from),
                verbose: a.verbose,
            };
            let report = dispatch(kind, &task);
            println!(
                "Estimated pose: x={:.4} µm, y={:.4} µm, θ={:.6} rad (quadrant k3={})",
                report.x, report.y, report.theta, report.k3
            );
        }
    }
}
