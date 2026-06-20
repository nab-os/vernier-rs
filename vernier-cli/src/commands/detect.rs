//! The detect command: run the real detection chain on a synthetic pattern and
//! print the recovered pose.
//!
//! A demonstrable end-to-end path (image -> phase plane -> pose) before real
//! image loading exists. Written as a [`BackendTask`] so it runs on any backend.

use vernier_core::buffer::BufferLayout;
use vernier_core::{Complex32, ComputeBackend, Pose};
use vernier_detection::spectrum::{analyze_direction, forward};
use vernier_pose::{Calibration, periodic};

use crate::backend_select::BackendTask;

/// Detect parameters.
pub struct Detect {
    /// Image side length.
    pub size: usize,
    /// Pattern frequency in cycles across the width.
    pub cycles: usize,
    /// Pattern period in physical units (for calibration).
    pub period: f32,
    /// Band-pass filter width in bins.
    pub sigma: f32,
}

/// What detect reports.
pub struct DetectReport {
    /// Backend used.
    pub backend: String,
    /// Recovered (single-direction) pose.
    pub pose: Pose,
    /// Plane-implied spectral peak location (m, n).
    pub peak: (f64, f64),
}

impl BackendTask for Detect {
    type Output = DetectReport;

    fn run<B: ComputeBackend>(&self, backend: &B) -> DetectReport {
        use std::f32::consts::TAU;
        let layout = BufferLayout::packed(self.size, self.size);
        let mut data = Vec::with_capacity(layout.len());
        for _r in 0..self.size {
            for c in 0..self.size {
                let v = (TAU * self.cycles as f32 * c as f32 / self.size as f32).cos();
                data.push(Complex32::new(v, 0.0));
            }
        }

        let mut buf = backend.upload(&data, layout).unwrap();
        forward(backend, &mut buf).unwrap();
        let spectrum = buf.clone();
        let dir = analyze_direction(backend, spectrum, self.sigma as vernier_core::Real).unwrap();

        let calib = Calibration::new(self.period, self.size, self.size);
        let pose = periodic::estimate_single(&dir.plane, &calib);

        DetectReport {
            backend: backend.name().to_string(),
            pose,
            peak: (dir.peak.0 as f64, dir.peak.1 as f64),
        }
    }
}
