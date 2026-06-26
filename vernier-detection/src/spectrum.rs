//! The real spectral detection chain, two directions (André et al. 2020/2021).

use vernier_core::buffer::{Buffer2D, BufferLayout};
use vernier_core::{ComputeBackend, ComputeJob, Real, Result, VernierError, Complex32};

use crate::planefit::PhasePlane;

/// Result of analyzing one pattern direction.
#[derive(Clone, Copy, Debug)]
pub struct DirectionResult {
    pub plane: PhasePlane,
    pub peak: (Real, Real),
    pub peak_bin: (usize, usize),
}

/// Full two-direction detection result.
#[derive(Clone, Debug)]
pub struct Detection {
    pub dir1: DirectionResult,
    pub dir2: DirectionResult,
    pub phase1: Vec<Real>,
    pub phase2: Vec<Real>,
    pub width: usize,
    pub height: usize,
}

/// Runs the forward FFT in place, leaving `buffer` in the frequency domain.
pub fn forward<B: ComputeBackend>(backend: &B, buffer: &mut B::Buffer2D) -> Result<()> {
    let mut job = backend.begin()?;
    job.fft2d(buffer)?;
    job.submit()
}

/// Reconstructs a `DirectionResult` and per-pixel phase map from the 3-element
/// plane buffer `[Complex32(a,0), Complex32(b,0), Complex32(c,0)]` returned by
/// `plane_fit_from_ifft`.
fn finish_direction_from_plane(
    plane_data: Vec<Complex32>,
    w: usize,
    h: usize,
    cx: usize,
    cy: usize,
) -> Result<(DirectionResult, Vec<Real>)> {
    let a = plane_data[0].re as Real;
    let b = plane_data[1].re as Real;
    let c = plane_data[2].re as Real;
    let plane = PhasePlane { a, b, c };
    let peak = plane.peak_location(w, h);
    let hw = w as Real / 2.0;
    let hh = h as Real / 2.0;
    let phase: Vec<Real> = (0..h)
        .flat_map(|row| {
            (0..w).map(move |col| a * (col as Real - hw) + b * (row as Real - hh) + c)
        })
        .collect();
    Ok((DirectionResult { plane, peak, peak_bin: (cx, cy) }, phase))
}

/// Analyzes the dominant direction in a pre-computed frequency-domain spectrum.
///
/// Single job: peak_search + bandpass + IFFT + GPU plane fit (3 floats downloaded).
pub fn analyze_direction<B: ComputeBackend>(
    backend: &B,
    mut spectrum: B::Buffer2D,
    sigma: Real,
) -> Result<DirectionResult> {
    let layout = spectrum.layout();
    let (w, h) = (layout.width, layout.height);

    let (peaks_buf, plane_buf) = {
        let mut job = backend.begin()?;
        let peaks_buf = job
            .peak_search(&mut spectrum, 0, 0, 0.5, sigma)?
            .ok_or_else(|| VernierError::Backend("no carrier peaks found".into()))?;
        let mut spec = job.copy_buffer(&spectrum)?;
        job.bandpass_from_peaks(&mut spec, &peaks_buf, 0, sigma)?;
        job.ifft2d(&mut spec)?;
        let pf = job.plane_fit_from_ifft(&spec, 0.5)?;
        job.submit()?;
        (peaks_buf, pf)
    };

    let peaks_data = backend.download(&peaks_buf)?;
    let cx = peaks_data[0].re as usize;
    let cy = peaks_data[1].re as usize;

    let plane_data = backend.download(&plane_buf)?;
    let a = plane_data[0].re as Real;
    let b = plane_data[1].re as Real;
    let c = plane_data[2].re as Real;
    let plane = PhasePlane { a, b, c };
    let peak = plane.peak_location(w, h);
    Ok(DirectionResult { plane, peak, peak_bin: (cx, cy) })
}

/// Uploads `data`, forward-transforms it, and analyzes both grid directions.
///
/// Single job: upload + FFT + peak_search + two-direction bandpass/IFFT/GPU plane fit.
/// Only 10 floats are downloaded after submission (4 peak coords + 3+3 plane coeffs).
pub fn analyze_two<B: ComputeBackend>(
    backend: &B,
    data: &[Complex32],
    layout: BufferLayout,
    sigma: Real,
    min_frequency: usize,
    max_frequency: usize,
    smoothing_sigma: Real,
) -> Result<Detection> {
    let (w, h) = (layout.width, layout.height);

    let (peaks_buf, plane_buf1, plane_buf2) = {
        let mut job = backend.begin()?;
        let mut buffer = job.upload(data, layout)?;
        job.fft2d(&mut buffer)?;
        let peaks_buf = job
            .peak_search(&mut buffer, min_frequency, max_frequency, smoothing_sigma, sigma)?
            .ok_or_else(|| VernierError::Backend("no carrier peaks found in spectrum".into()))?;
        let mut spec1 = job.copy_buffer(&buffer)?;
        let mut spec2 = job.copy_buffer(&buffer)?;
        job.bandpass_from_peaks(&mut spec1, &peaks_buf, 0, sigma)?;
        job.bandpass_from_peaks(&mut spec2, &peaks_buf, 1, sigma)?;
        job.ifft2d(&mut spec1)?;
        job.ifft2d(&mut spec2)?;
        let pf1 = job.plane_fit_from_ifft(&spec1, 0.5)?;
        let pf2 = job.plane_fit_from_ifft(&spec2, 0.5)?;
        job.submit()?;
        (peaks_buf, pf1, pf2)
    };

    let peaks_data = backend.download(&peaks_buf)?;
    let (cx1, cy1) = (peaks_data[0].re as usize, peaks_data[1].re as usize);
    let (cx2, cy2) = (peaks_data[2].re as usize, peaks_data[3].re as usize);

    let (dir1, phase1) =
        finish_direction_from_plane(backend.download(&plane_buf1)?, w, h, cx1, cy1)?;
    let (dir2, phase2) =
        finish_direction_from_plane(backend.download(&plane_buf2)?, w, h, cx2, cy2)?;

    Ok(Detection { dir1, dir2, phase1, phase2, width: w, height: h })
}
