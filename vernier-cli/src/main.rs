//! `vernier` — the CLI front-end and benchmark harness for vernier-rs.
//!
//! This binary is the one place that names concrete backends. It parses args,
//! resolves a [`BackendKind`](backend_select::BackendKind), and
//! [`dispatch`](backend_select::dispatch)es a backend-generic
//! [`BackendTask`](backend_select::BackendTask). Swapping CPU for GPU is the
//! `--backend` flag; nothing in the library changes.

mod args;
mod backend_select;
mod commands;
mod pgm;
mod imageio;
mod annotate;

use args::{Command, TopLevel};
use backend_select::{dispatch, BackendKind};
use commands::benchmark::Benchmark;
use commands::detect::Detect;
use commands::inspect::Inspect;
use commands::roundtrip::Roundtrip;
use commands::analyse::Analyse;
use commands::detect_megarena::DetectMegarena;

fn main() {
    let top: TopLevel = argh::from_env();

    match top.command {
        Command::Detect(a) => {
            let Some(kind) = BackendKind::parse(&a.backend) else {
                eprintln!("unknown backend '{}'. try: cpu", a.backend);
                std::process::exit(2);
            };
            let task = Detect {
                size: a.size,
                cycles: a.cycles,
                period: a.period,
                sigma: a.sigma,
            };
            let report = dispatch(kind, &task);
            println!(
                "[{}] peak (m={:.2}, n={:.2}) -> pose x={:.4} y={:.4} theta={:.4} rad",
                report.backend,
                report.peak.0,
                report.peak.1,
                report.pose.x,
                report.pose.y,
                report.pose.theta,
            );
        }
        Command::Bench(a) => {
            let Some(kind) = BackendKind::parse(&a.backend) else {
                eprintln!("unknown backend '{}'. try: cpu", a.backend);
                std::process::exit(2);
            };
            let task = Benchmark {
                size: a.size,
                iterations: a.iterations,
                sigma: a.sigma,
            };
            let r = dispatch(kind, &task);
            println!(
                "[{}] {}x{} pipeline over {} iters: mean {:.3} ms, best {:.3} ms",
                r.backend, r.size, r.size, r.iterations, r.mean_ms, r.best_ms,
            );
            println!(
                "  (full path: upload + fwd FFT + bandpass + inv FFT + phase; transfer included)"
            );
        }
        Command::Roundtrip(a) => {
            let Some(kind) = BackendKind::parse(&a.backend) else {
                eprintln!("unknown backend '{}'. try: cpu", a.backend);
                std::process::exit(2);
            };
            let task = Roundtrip {
                size: a.size,
                period_px: a.period,
                true_theta: a.theta,
                sigma: a.sigma,
            };
            let r = dispatch(kind, &task);
            println!(
                "[{}] roundtrip {}x{}: true θ={:.4} rad, recovered θ={:.4} rad, error={:.2e} rad",
                r.backend, a.size, a.size, r.true_theta, r.recovered_theta, r.theta_error,
            );
        }
        Command::Inspect(a) => {
            let task = Inspect {
                size: a.size,
                period_px: a.period,
                theta: a.theta,
                sigma: a.sigma,
                out_dir: std::path::PathBuf::from(&a.out),
            };
            match task.run() {
                Ok(theta) => {
                    println!(
                        "saved {}x{} pipeline stages to '{}/' (recovered θ={:.4} rad)",
                        a.size, a.size, a.out, theta
                    );
                    println!(
                        "  stages: 00_pattern, 01_fft_magnitude, 02_bandpass_mask, \
                         03_isolated_lobe, 04_phase_wrapped, 05_phase_unwrapped (.pgm)"
                    );
                    println!("  view directly, or: convert {}/01_fft_magnitude.pgm fft.png", a.out);
                }
                Err(e) => {
                    eprintln!("inspect failed: {e}");
                    std::process::exit(1);
                }
            }
        }
        Command::Analyse(a) => {
            let task = Analyse {
                image_path: std::path::PathBuf::from(&a.image),
                sigma: a.sigma,
                min_frequency: a.min_frequency,
                max_frequency: a.max_frequency,
                smoothing_sigma: a.smoothing_sigma,
                window: !a.no_window,
                stages_dir: a.stages.as_ref().map(std::path::PathBuf::from),
            };
            if let Err(e) = task.run() {
                eprintln!("analyse failed: {e}");
                std::process::exit(1);
            }
        }
        Command::DetectMegarena(a) => {
            let task = DetectMegarena {
                image_path: std::path::PathBuf::from(&a.image),
                physical_period: a.period,
                code_size: a.code_size,
                sigma: a.sigma,
                min_frequency: a.min_frequency,
                max_frequency: a.max_frequency,
                smoothing_sigma: a.smoothing_sigma,
                window: !a.no_window,
                debug_image: a.debug_image.as_ref().map(std::path::PathBuf::from),
            };
            if let Err(e) = task.run() {
                eprintln!("detect-megarena failed: {e}");
                std::process::exit(1);
            }
        }
    }
}
