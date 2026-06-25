//! The real spectral detection chain, two directions (André et al. 2020/2021).

use vernier_core::buffer::Buffer2D;
use vernier_core::{ComputeBackend, ComputeJob, Real, Result, VernierError};

use crate::planefit::{PhasePlane, fit_plane_to_unwrapped};
use crate::unwrap::quarters_unwrap_phase;

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

/// Analyzes one direction: band-pass at `(cx, cy)`, IFFT, extract phase,
/// unwrap, fit plane. Returns the fitted result and the per-pixel unwrapped
/// phase map. `spectrum` is a reference; a deep copy is made inside the job
/// so the original spectrum is not consumed or modified.
fn analyze_at<B: ComputeBackend>(
    backend: &B,
    spectrum: &B::Buffer2D,
    cx: usize,
    cy: usize,
    sigma: Real,
) -> Result<(DirectionResult, Vec<Real>)> {
    let layout = spectrum.layout();
    let (w, h) = (layout.width, layout.height);

    let phase_field = {
        let mut job = backend.begin()?;
        let mut spec = job.copy_buffer(spectrum)?;
        job.bandpass_filter(&mut spec, cx, cy, sigma)?;
        job.ifft2d(&mut spec)?;
        let pf = job.extract_phase(&spec)?;
        job.submit()?;
        pf
    };

    let data = backend.download(&phase_field)?;
    let wrapped: Vec<Real> = data.iter().map(|c| c.re as Real).collect();

    let mut unwrapped = wrapped.clone();
    quarters_unwrap_phase(&mut unwrapped, w, h);
    let plane = fit_plane_to_unwrapped(&unwrapped, w, h, 0.5);
    let peak = plane.peak_location(w, h);

    Ok((DirectionResult { plane, peak, peak_bin: (cx, cy) }, unwrapped))
}

/// Analyzes the dominant direction in a pre-computed frequency-domain spectrum.
pub fn analyze_direction<B: ComputeBackend>(
    backend: &B,
    spectrum: B::Buffer2D,
    sigma: Real,
) -> Result<DirectionResult> {
    let peaks = {
        let mut job = backend.begin()?;
        let mut mag = job.copy_buffer(&spectrum)?;
        let p = job
            .peak_search(&mut mag, 0, 0, 0.5, sigma)?
            .ok_or_else(|| VernierError::Backend("no carrier peaks found".into()))?;
        job.submit()?;
        p
    };

    let raw = backend.download(&peaks)?;
    let cx = raw[0].re as usize;
    let cy = raw[1].re as usize;
    let (dir, _) = analyze_at(backend, &spectrum, cx, cy, sigma)?;
    Ok(dir)
}

/// Forward-transforms `buffer` and analyzes both grid directions.
pub fn analyze_two<B: ComputeBackend>(
    backend: &B,
    buffer: &mut B::Buffer2D,
    sigma: Real,
    min_frequency: usize,
    max_frequency: usize,
    smoothing_sigma: Real,
) -> Result<Detection> {
    // Phase 1: FFT + two-peak search (all in one job).
    let buffer_peaks = {
        let mut job = backend.begin()?;
        job.fft2d(buffer)?;
        let p = job
            .peak_search(buffer, min_frequency, max_frequency, smoothing_sigma, sigma)?
            .ok_or_else(|| VernierError::Backend("no carrier peaks found in spectrum".into()))?;
        job.submit()?;
        p
    };

    let layout = buffer.layout();
    let (w, h) = (layout.width, layout.height);

    let (cx1, cy1, cx2, cy2) = {
        let peaks = backend.download(&buffer_peaks)?;
        (
            peaks[0].re as usize,
            peaks[1].re as usize,
            peaks[2].re as usize,
            peaks[3].re as usize,
        )
    };

    // Phase 2: independent bandpass + IFFT for each direction.
    let (dir1, phase1) = analyze_at(backend, buffer, cx1, cy1, sigma)?;
    let (dir2, phase2) = analyze_at(backend, buffer, cx2, cy2, sigma)?;

    Ok(Detection { dir1, dir2, phase1, phase2, width: w, height: h })
}
