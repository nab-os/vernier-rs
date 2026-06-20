//! The roundtrip command: the whole pipeline validated against ground truth.
//!
//! This is the "does it actually work" command. It:
//! 1. Generates a periodic pattern at a *known* pose (via `vernier-patterns`).
//! 2. Runs the real detection chain (forward FFT → band-pass → inverse FFT →
//!    phase → plane fit).
//! 3. Estimates the pose from the fitted plane.
//! 4. Reports the recovered orientation against the true one.
//!
//! Because the generator is the deliberate inverse of the estimator, a correct
//! pipeline recovers the input pose to within numerical tolerance — on whichever
//! backend. Running this with `--backend cpu` today and `--backend gpu` later is
//! how you prove the GPU path matches the CPU oracle.

use vernier_core::buffer::BufferLayout;
use vernier_core::{Complex32, ComputeBackend};
use vernier_detection::spectrum::{analyze_direction, forward};
use vernier_patterns::PatternPose;
use vernier_patterns::periodic::Periodic;

use crate::backend_select::BackendTask;

/// Roundtrip parameters.
pub struct Roundtrip {
    /// Square image side length.
    pub size: usize,
    /// Pattern period in pixels.
    pub period_px: f32,
    /// True orientation to render at, in radians.
    pub true_theta: f32,
    /// Band-pass filter width in bins.
    pub sigma: f32,
}

/// What roundtrip reports.
pub struct RoundtripReport {
    /// Backend used.
    pub backend: String,
    /// The true orientation the pattern was rendered at.
    pub true_theta: f64,
    /// The orientation the pipeline recovered.
    pub recovered_theta: f64,
    /// Absolute orientation error in radians.
    pub theta_error: f64,
}

impl BackendTask for Roundtrip {
    type Output = RoundtripReport;

    fn run<B: ComputeBackend>(&self, backend: &B) -> RoundtripReport {
        // 1. Generate the ground-truth image.
        let pose = PatternPose::new(0.0, 0.0, self.true_theta as vernier_core::Real);
        let pattern = Periodic::new(self.period_px as vernier_core::Real);
        let image = pattern.render(self.size, self.size, &pose);

        // Promote the real image to complex and upload.
        let layout = BufferLayout::packed(self.size, self.size);
        let complex: Vec<Complex32> = image
            .as_slice()
            .iter()
            .map(|&v| Complex32::new(v, 0.0))
            .collect();
        let mut buf = backend.upload(&complex, layout).unwrap();

        // 2. Real detection chain.
        forward(backend, &mut buf).unwrap();
        let spectrum = buf.clone();
        let dir = analyze_direction(backend, spectrum, self.sigma as vernier_core::Real).unwrap();

        // 3. Recovered orientation from the fitted plane gradients.
        let recovered = dir.plane.orientation() as f64;
        let truth = self.true_theta as f64;

        // Angular error wrapped into [-π, π].
        let mut err = recovered - truth;
        let pi = std::f64::consts::PI;
        while err > pi {
            err -= 2.0 * pi;
        }
        while err <= -pi {
            err += 2.0 * pi;
        }

        RoundtripReport {
            backend: backend.name().to_string(),
            true_theta: truth,
            recovered_theta: recovered,
            theta_error: err.abs(),
        }
    }
}
