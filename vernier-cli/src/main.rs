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
use commands::render_megarena;
use commands::roundtrip_megarena::RoundtripMegarena;

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
        Command::RenderMegarena(a) => {
            let args = render_megarena::RenderMegarenaArgs {
                width: a.width,
                height: a.height,
                x: a.x,
                y: a.y,
                theta: a.theta,
                period_px: a.period,
                code_size: a.code_size,
                output: std::path::PathBuf::from(&a.output),
            };
            if let Err(e) = render_megarena::run(&args) {
                eprintln!("error: {e}");
                std::process::exit(1);
            }
        }
        Command::RoundtripMegarena(a) => {
            let Some(kind) = BackendKind::parse(&a.backend) else {
                eprintln!("unknown backend '{}'. try: cpu, gpu", a.backend);
                std::process::exit(2);
            };
            let task = RoundtripMegarena {
                width: a.width,
                height: a.height,
                true_x: a.x as f32,
                true_y: a.y as f32,
                true_theta: a.theta as f32,
                period_px: a.period as f32,
                code_size: a.code_size,
                sigma: a.sigma as f32,
                min_frequency: a.min_frequency,
                max_frequency: a.max_frequency,
                smoothing_sigma: a.smoothing_sigma as f32,
            };
            let r = dispatch(kind, &task);
            let swap_label = if r.swapped { "yes" } else { "no" };
            println!(
                "backend={}  size={}x{}  period={:.1}px  code={}  swapped={}",
                r.backend, a.width, a.height, a.period, a.code_size, swap_label
            );
            println!(
                "true:      x={:.4}  y={:.4}  θ={:.6} rad",
                r.true_x, r.true_y, r.true_theta
            );
            println!(
                "           fine_x={:.4}  fine_y={:.4}",
                r.true_fine_x, r.true_fine_y
            );
            println!(
                "recovered: x={:.4}  y={:.4}  θ={:.6} rad",
                r.recovered_x, r.recovered_y, r.recovered_theta
            );
            println!(
                "           fine_x={:.4}  fine_y={:.4}  (x_ps={}  y_ps={}  k1={}  k2={}  k3={})",
                r.recovered_fine_x, r.recovered_fine_y,
                r.x_ps, r.y_ps, r.k1, r.k2, r.k3
            );
            println!(
                "error abs: Δx={:.4}px  Δy={:.4}px  Δθ={:.2e} rad",
                r.abs_error_x, r.abs_error_y, r.error_theta
            );
            println!(
                "error fine: Δx={:.4}px  Δy={:.4}px",
                r.fine_error_x, r.fine_error_y
            );
        }
    }
}
