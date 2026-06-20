//! The benchmark command: time the real detection chain on synthetic patterns.
//!
//! The harness that answers the project's central question — does the GPU win at
//! these image sizes once host<->device transfer is paid for? Written as a
//! [`BackendTask`], so identical timing code runs against CPU now and GPU later;
//! comparing them is `vernier bench --backend cpu` vs `--backend gpu`.
//!
//! It times the full on-device path (upload -> forward FFT -> filter -> inverse
//! FFT -> phase -> download), because the transfer cost is part of the honest
//! comparison. Timing only the FFT kernel would flatter the GPU by hiding the
//! upload it cannot avoid.

use std::time::Instant;

use vernier_core::buffer::BufferLayout;
use vernier_core::{Complex32, ComputeBackend};
use vernier_detection::spectrum::{analyze_direction, forward};

use crate::backend_select::BackendTask;

/// Parameters for a benchmark run.
pub struct Benchmark {
    /// Square image side length (power of two recommended).
    pub size: usize,
    /// Number of timed iterations.
    pub iterations: usize,
    /// Band-pass filter width in bins.
    pub sigma: f32,
}

/// What a benchmark run reports.
pub struct BenchReport {
    /// Backend name.
    pub backend: String,
    /// Image side length used.
    pub size: usize,
    /// Iterations timed.
    pub iterations: usize,
    /// Mean wall-clock time per full pipeline iteration, in milliseconds.
    pub mean_ms: f64,
    /// Fastest single iteration, in milliseconds.
    pub best_ms: f64,
}

impl Benchmark {
    fn synthetic(size: usize) -> (Vec<Complex32>, BufferLayout) {
        use std::f32::consts::TAU;
        let layout = BufferLayout::packed(size, size);
        let mut data = Vec::with_capacity(layout.len());
        for _r in 0..size {
            for c in 0..size {
                let v = (TAU * 5.0 * c as f32 / size as f32).cos();
                data.push(Complex32::new(v, 0.0));
            }
        }
        (data, layout)
    }
}

impl BackendTask for Benchmark {
    type Output = BenchReport;

    fn run<B: ComputeBackend>(&self, backend: &B) -> BenchReport {
        let (data, layout) = Self::synthetic(self.size);
        let sigma = self.sigma as f64 as vernier_core::Real;

        // Warm-up: build/cache FFT plans so the timed loop is steady-state.
        {
            let mut buf = backend.upload(&data, layout).unwrap();
            forward(backend, &mut buf).unwrap();
            let spectrum = buf.clone();
            let _ = analyze_direction(backend, spectrum, sigma);
        }

        let mut best = f64::INFINITY;
        let mut total = 0.0;
        for _ in 0..self.iterations {
            let start = Instant::now();
            let mut buf = backend.upload(&data, layout).unwrap();
            forward(backend, &mut buf).unwrap();
            let spectrum = buf.clone();
            let _ = analyze_direction(backend, spectrum, sigma).unwrap();
            let ms = start.elapsed().as_secs_f64() * 1e3;
            total += ms;
            best = best.min(ms);
        }

        BenchReport {
            backend: backend.name().to_string(),
            size: self.size,
            iterations: self.iterations,
            mean_ms: total / self.iterations as f64,
            best_ms: best,
        }
    }
}
