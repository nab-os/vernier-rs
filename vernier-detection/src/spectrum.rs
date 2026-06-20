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
//! The second direction is found by excluding a neighborhood of the first peak
//! before searching again, so the search returns the perpendicular carrier
//! rather than a sideband of the first.
//!
//! [`analyze_two`] returns a [`Detection`] carrying both [`DirectionResult`]s
//! AND both per-pixel wrapped phase maps — the latter are what the absolute
//! megarena decode needs (it localizes coding cells against the phase). The
//! phase maps are the one larger thing that comes back to the host; the fitted
//! planes are the small summary.

use vernier_core::buffer::Buffer2D;
use vernier_core::{ComputeBackend, Real, Result};

use crate::planefit::{fit_plane_cropped, PhasePlane};

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
    /// First (strongest) direction.
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
/// to isolate. Returns the fitted result and the per-pixel wrapped phase map.
///
/// `spectrum` is consumed (filtered + inverse-transformed). `cx, cy` is the
/// carrier bin for this direction.
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

    // Unwrap outward from the center (C++ quartersUnwrapPhase), then fit a plane
    // to the central 50% of the image (C++ RegressionPlane cropFactor=0.5).
    let mut unwrapped = wrapped;
    crate::unwrap::quarters_unwrap_phase(&mut unwrapped, w, h);
    let plane = fit_plane_cropped(&unwrapped, w, h, 0.5);
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
/// Finds the strongest carrier peak, then the strongest peak outside a
/// `exclude_radius` neighborhood of it (the perpendicular carrier), and runs the
/// phase chain on each. Returns both planes and both phase maps.
pub fn analyze_two<B: ComputeBackend>(
    backend: &B,
    buffer: &mut B::Buffer2D,
    sigma: Real,
    exclude_radius: usize,
) -> Result<Detection>
where
    B::Buffer2D: Clone,
{
    forward(backend, buffer)?;
    let layout = buffer.layout();
    let (w, h) = (layout.width, layout.height);

    // Peak 1: strongest carrier in the half-plane.
    let (idx1, _) = backend.argmax_magnitude_halfplane(buffer)?;
    let (mut cx1, mut cy1) = (idx1 % w, idx1 / w);

    // Peak 2: strongest carrier away from peak 1's neighborhood.
    let (idx2, _) =
        backend.argmax_magnitude_halfplane_excluding(buffer, cx1, cy1, exclude_radius)?;
    let (mut cx2, mut cy2) = (idx2 % w, idx2 / w);

    // Peak swap: ensure peak 1 has the larger signed-x frequency, matching C++
    // which swaps so mainPeak1.x() >= mainPeak2.x() in shifted-spectrum columns.
    let signed_x = |fx: usize| -> isize {
        let fx = fx as isize;
        let w = w as isize;
        if fx > w / 2 { fx - w } else { fx }
    };
    if signed_x(cx1) < signed_x(cx2) {
        std::mem::swap(&mut cx1, &mut cx2);
        std::mem::swap(&mut cy1, &mut cy2);
    }

    // Analyze each on its own copy of the spectrum (band-pass is destructive).
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
    let (idx, _mag) = backend.argmax_magnitude_halfplane(&spectrum)?;
    let (cx, cy) = (idx % w, idx / w);
    backend.bandpass_filter(&mut spectrum, cx, cy, sigma)?;
    backend.ifft2d(&mut spectrum)?;
    let phase_field = backend.extract_phase(&spectrum)?;
    let data = backend.download(&phase_field)?;
    let wrapped: Vec<Real> = data.iter().map(|c| c.re as Real).collect();
    let mut unwrapped = wrapped;
    crate::unwrap::quarters_unwrap_phase(&mut unwrapped, w, h);
    let plane = fit_plane_cropped(&unwrapped, w, h, 0.5);
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
    buffer: &mut B::Buffer2D,
    sigma: Real,
) -> Result<DirectionResult>
where
    B::Buffer2D: Clone,
{
    forward(backend, buffer)?;
    let spectrum = buffer.clone();
    analyze_direction(backend, spectrum, sigma)
}
