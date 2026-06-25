//! The inspect command: run the roundtrip and save every pipeline stage as an
//! image, so you can *see* the method work — input, FFT, isolated lobe, phase.
//!
//! It re-runs the detection chain stage by stage rather than threading a debug
//! flag through the library, downloading and dumping each intermediate buffer.
//! That keeps all visualization in the CLI; the library stays clean.
//!
//! Stages saved (into the output directory, as binary PGM):
//! - `00_pattern.pgm`        the generated input image
//! - `01_fft_magnitude.pgm`  log-scaled, fftshifted spectrum (lobes visible)
//! - `02_bandpass_mask.pgm`  the Gaussian filter, fftshifted
//! - `03_isolated_lobe.pgm`  spectrum after band-pass, log-scaled fftshifted
//! - `04_phase_wrapped.pgm`  wrapped phase of the inverse-FFT field
//! - `05_phase_unwrapped.pgm` the fitted plane evaluated over the image
//!
//! Unlike the other commands this is not a `BackendTask` — it needs to interleave
//! downloads with compute, and it only ever runs on the CPU backend (it is a
//! debug tool), so it takes a concrete `CpuBackend` directly.

use std::path::{Path, PathBuf};

use vernier_core::buffer::BufferLayout;
use vernier_core::{Complex32, ComputeBackend, Real};
use vernier_cpu::CpuBackend;
use vernier_detection::planefit::fit_plane;
use vernier_patterns::PatternPose;
use vernier_patterns::periodic::Periodic;

use crate::pgm;

/// Inspect parameters.
pub struct Inspect {
    /// Square image side length.
    pub size: usize,
    /// Pattern period in pixels.
    pub period_px: f32,
    /// Orientation to render at, in radians.
    pub theta: f32,
    /// Band-pass filter width in bins.
    pub sigma: f32,
    /// Directory to write the stage images into.
    pub out_dir: PathBuf,
}

impl Inspect {
    /// Runs the chain and writes all stage images. Returns the recovered
    /// orientation so the caller can print it alongside the saved files.
    pub fn run(&self) -> std::io::Result<f64> {
        std::fs::create_dir_all(&self.out_dir)?;
        let backend = CpuBackend::new();
        let (w, h) = (self.size, self.size);
        let dir = self.out_dir.as_path();

        // --- Stage 0: generate the input pattern ---
        let pose = PatternPose::new(0.0, 0.0, self.theta as Real);
        let pattern = Periodic::new(self.period_px as Real);
        let image = pattern.render(w, h, &pose);
        let intensities: Vec<f64> = image.as_slice().iter().map(|&v| v as f64).collect();
        pgm::save_unit(&stage(dir, "00_pattern.pgm"), w, h, &intensities)?;

        // Upload as complex.
        let layout = BufferLayout::packed(w, h);
        let complex: Vec<Complex32> = image
            .as_slice()
            .iter()
            .map(|&v| Complex32::new(v, 0.0))
            .collect();
        let mut buf = backend.upload(&complex, layout).unwrap();

        // --- Stage 1: forward FFT, save log magnitude (fftshifted) ---
        backend.fft2d(&mut buf).unwrap();
        let spec = backend.download(&buf).unwrap();
        let mag: Vec<f64> = spec.iter().map(|c| (c.norm_sqr() as f64).sqrt()).collect();
        pgm::save_log_magnitude(
            &stage(dir, "01_fft_magnitude.pgm"),
            w,
            h,
            &pgm::fftshift(w, h, &mag),
        )?;

        // Locate the lobe (same half-plane rule as detection).
        let (idx, _m) = backend.argmax_magnitude_halfplane(&buf, 0).unwrap();
        let (cx, cy) = (idx % w, idx / w);

        // --- Stage 2: the band-pass mask itself ---
        // Reconstruct the Gaussian for visualization (same formula as the
        // backend's bandpass_filter, evaluated to a [0,1] image).
        let mask = gaussian_mask(w, h, cx, cy, self.sigma as Real);
        pgm::save_unit(
            &stage(dir, "02_bandpass_mask.pgm"),
            w,
            h,
            &pgm::fftshift(w, h, &mask),
        )?;

        // --- Stage 3: apply band-pass, save the isolated lobe ---
        backend
            .bandpass_filter(&mut buf, cx, cy, self.sigma as Real)
            .unwrap();
        let filtered = backend.download(&buf).unwrap();
        let fmag: Vec<f64> = filtered
            .iter()
            .map(|c| (c.norm_sqr() as f64).sqrt())
            .collect();
        pgm::save_log_magnitude(
            &stage(dir, "03_isolated_lobe.pgm"),
            w,
            h,
            &pgm::fftshift(w, h, &fmag),
        )?;

        // --- Stage 4: inverse FFT, wrapped phase ---
        backend.ifft2d(&mut buf).unwrap();
        let phase_field = backend.extract_phase(&buf).unwrap();
        let phase_data = backend.download(&phase_field).unwrap();
        let wrapped: Vec<f64> = phase_data.iter().map(|c| c.re as f64).collect();
        pgm::save_linear(&stage(dir, "04_phase_wrapped.pgm"), w, h, &wrapped)?;

        // --- Stage 5: the fitted plane evaluated over the image ---
        let wrapped_real: Vec<Real> = phase_data.iter().map(|c| c.re as Real).collect();
        let plane = fit_plane(&wrapped_real, w, h, 0.5);
        let (cxf, cyf) = (w as Real / 2.0, h as Real / 2.0);
        let mut fitted = vec![0.0f64; w * h];
        for r in 0..h {
            let j = r as Real - cyf;
            for col in 0..w {
                let i = col as Real - cxf;
                fitted[r * w + col] = (plane.a * i + plane.b * j + plane.c) as f64;
            }
        }
        pgm::save_linear(&stage(dir, "05_phase_unwrapped.pgm"), w, h, &fitted)?;

        Ok(plane.orientation() as f64)
    }
}

fn stage(dir: &Path, name: &str) -> PathBuf {
    dir.join(name)
}

/// Evaluates the band-pass Gaussian to a [0,1] image for visualization. Mirrors
/// the backend's `bandpass_filter` gain formula with circular bin distance.
fn gaussian_mask(w: usize, h: usize, cx: usize, cy: usize, sigma: Real) -> Vec<f64> {
    let two_sigma_sq = 2.0 * sigma * sigma;
    let circular = |a: usize, c: usize, n: usize| -> Real {
        let d = a as isize - c as isize;
        let n = n as isize;
        let d = ((d % n) + n) % n;
        let d = if d > n / 2 { d - n } else { d };
        d as Real
    };
    let mut out = vec![0.0; w * h];
    for fy in 0..h {
        let dy = circular(fy, cy, h);
        for fx in 0..w {
            let dx = circular(fx, cx, w);
            let gain = (-(dx * dx + dy * dy) / two_sigma_sq).exp();
            out[fy * w + fx] = gain as f64;
        }
    }
    out
}
