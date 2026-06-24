//! The real spectral detection chain, two directions (André et al. 2020/2021).
//!
//! A 2D periodic/megarena pattern shows up in the Fourier domain as two carrier
//! peaks — one per grid direction — each with a conjugate twin. For each
//! direction the chain is:
//!
//! 1. Forward 2D FFT (done once, shared by both directions).
//! 2. Locate the direction's carrier peak in the canonical half-plane.
//! 3. Gaussian band-pass on that lobe (conjugate excluded).
//! 4. Inverse FFT -> spatial complex field.
//! 5. `arg(.)` per pixel -> wrapped phase map.
//! 6. Unwrap + least-squares plane fit -> high-resolution phase and gradients.
//!
//! The second direction is found by an angular cone exclusion around the first
//! peak's direction before searching again (C++ `PatternPhase::applyAngularCut`),
//! so the search returns the perpendicular carrier rather than a sideband of the
//! first. The two peaks are ordered so the one with the larger signed column
//! frequency is direction 1 (C++ convention).
//!
//! [`analyze_two`] returns a [`Detection`] carrying both [`DirectionResult`]s
//! AND both per-pixel unwrapped phase maps — the latter are what the absolute
//! megarena decode needs (it localizes coding cells against the phase). The
//! phase maps are the one larger thing that comes back to the host; the fitted
//! planes are the small summary.

use vernier_core::buffer::Buffer2D;
use vernier_core::{ComputeBackend, Real, Result, VernierError};

use crate::planefit::{PhasePlane, fit_plane_to_unwrapped};
use crate::unwrap::quarters_unwrap_phase;

/// Result of analyzing one pattern direction.
#[derive(Clone, Copy, Debug)]
pub struct DirectionResult {
    /// The fitted phase plane for this direction.
    pub plane: PhasePlane,
    /// Spectral peak location `(m, n)` in bins implied by the plane gradients.
    pub peak: (Real, Real),
    /// The integer bin the carrier was found at, `(fx, fy)`.
    pub peak_bin: (usize, usize),
}

/// Full two-direction detection result.
#[derive(Clone, Debug)]
pub struct Detection {
    /// First direction (larger signed column frequency, matching C++ convention).
    pub dir1: DirectionResult,
    /// Second (perpendicular) direction.
    pub dir2: DirectionResult,
    /// Per-pixel UNWRAPPED phase map for direction 1 (row-major). Unwrapped so the
    /// megarena decode can derive cell indices via round(φ/2π).
    pub phase1: Vec<Real>,
    /// Per-pixel UNWRAPPED phase map for direction 2.
    pub phase2: Vec<Real>,
    /// Image width.
    pub width: usize,
    /// Image height.
    pub height: usize,
}

/// Runs the forward FFT in place, leaving `buffer` in the frequency domain.
pub fn forward<B: ComputeBackend>(backend: &B, buffer: &mut B::Buffer2D) -> Result<()> {
    backend.fft2d(buffer)
}

/// Analyzes one direction given a frequency-domain spectrum and the carrier bin
/// to isolate. Returns the fitted result and the per-pixel unwrapped phase map.
///
/// `spectrum` is consumed (filtered + inverse-transformed). `cx, cy` is the
/// carrier bin for this direction. Uses quarter-based phase unwrapping seeded
/// at the image center (C++ `Spatial::quartersUnwrapPhase`).
fn analyze_at<B: ComputeBackend>(
    backend: &B,
    mut spectrum: B::Buffer2D,
    cx: usize,
    cy: usize,
    sigma: Real,
) -> Result<(DirectionResult, Vec<Real>)> {
    let layout = spectrum.layout();
    let (w, h) = (layout.width, layout.height);

    backend.bandpass_filter(&mut spectrum, cx, cy, sigma)?;
    backend.ifft2d(&mut spectrum)?;
    let phase_field = backend.extract_phase(&spectrum)?;

    let data = backend.download(&phase_field)?;
    let wrapped: Vec<Real> = data.iter().map(|c| c.re as Real).collect();

    // Quarter-based unwrap seeded at image center (C++ Spatial::quartersUnwrapPhase).
    let mut unwrapped = wrapped.clone();
    quarters_unwrap_phase(&mut unwrapped, w, h);
    let plane = fit_plane_to_unwrapped(&unwrapped, w, h, 0.5);
    let peak = plane.peak_location(w, h);

    Ok((
        DirectionResult {
            plane,
            peak,
            peak_bin: (cx, cy),
        },
        unwrapped,
    ))
}

/// Forward-transforms `buffer` and analyzes both grid directions.
///
/// Peak search matches C++ `PatternPhase::peaksSearch`:
/// 1. Compute magnitude of the frequency-domain buffer.
/// 2. Zero an annulus outside [`min_frequency`, `max_frequency`] bins from DC.
/// 3. Apply a separable Gaussian blur of `smoothing_sigma` (C++ `cv::GaussianBlur`).
/// 4. Find the dominant carrier in the canonical half-plane.
/// 5. Exclude an angular cone around peak 1's direction (C++ `applyAngularCut`,
///    half-width = `atan2(3·sigma, distance)`) and find peak 2.
/// 6. Swap so the peak with the larger signed column frequency is direction 1.
///
/// `sigma` is the band-pass filter width AND determines the angular cone width.
/// `min_frequency = 0` / `max_frequency = 0` disable the respective annulus bound.
pub fn analyze_two<B: ComputeBackend>(
    backend: &B,
    buffer: &mut B::Buffer2D,
    sigma: Real,
    min_frequency: usize,
    max_frequency: usize,
    smoothing_sigma: Real,
) -> Result<Detection>
where
    B::Buffer2D: Clone,
{
    forward(backend, buffer)?;
    let layout = buffer.layout();
    let (w, h) = (layout.width, layout.height);

    let ((cx1, cy1), (cx2, cy2)) = backend
        .peak_search(&*buffer, min_frequency, max_frequency, smoothing_sigma, sigma)
        .ok_or_else(|| VernierError::Backend("no carrier peaks found in spectrum".into()))?;

    eprintln!("peak 1: {},{}", cx1, cy1);
    eprintln!("peak 2: {},{}", cx2, cy2);

    let (dir1, phase1) = analyze_at(backend, buffer.clone(), cx1, cy1, sigma)?;
    let (dir2, phase2) = analyze_at(backend, buffer.clone(), cx2, cy2, sigma)?;

    Ok(Detection {
        dir1,
        dir2,
        phase1,
        phase2,
        width: w,
        height: h,
    })
}

/// Single-direction analysis (kept for the 1D periodic case and simple tests).
///
/// Forward-transforms, finds the one dominant carrier, runs the chain.
pub fn analyze_direction<B: ComputeBackend>(
    backend: &B,
    mut spectrum: B::Buffer2D,
    sigma: Real,
) -> Result<DirectionResult> {
    let layout = spectrum.layout();
    let (w, h) = (layout.width, layout.height);
    let (idx, _mag) = backend.argmax_magnitude_halfplane(&spectrum, 0)?;
    let (cx, cy) = (idx % w, idx / w);
    backend.bandpass_filter(&mut spectrum, cx, cy, sigma)?;
    backend.ifft2d(&mut spectrum)?;
    let phase_field = backend.extract_phase(&spectrum)?;
    let data = backend.download(&phase_field)?;
    let wrapped: Vec<Real> = data.iter().map(|c| c.re as Real).collect();
    let mut unwrapped = wrapped.clone();
    quarters_unwrap_phase(&mut unwrapped, w, h);
    let plane = fit_plane_to_unwrapped(&unwrapped, w, h, 0.5);
    let peak = plane.peak_location(w, h);
    Ok(DirectionResult {
        plane,
        peak,
        peak_bin: (cx, cy),
    })
}

/// Convenience single-direction analyze that forward-transforms first.
pub fn analyze<B: ComputeBackend>(
    backend: &B,
    spectrum: &mut B::Buffer2D,
    sigma: Real,
) -> Result<DirectionResult>
where
    B::Buffer2D: Clone,
{
    forward(backend, spectrum)?;
    let spectrum = spectrum.clone();
    analyze_direction(backend, spectrum, sigma)
}
