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
use vernier_core::{Complex32, ComputeBackend, Real, Result, VernierError};
use vernier_core::scalar::consts::{PI, TAU};

use crate::planefit::{fit_plane_to_unwrapped, PhasePlane};
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
    window: bool,
) -> Result<Detection>
where
    B::Buffer2D: Clone,
{
    if window {
        backend.hann_window(buffer)?;
    }
    forward(backend, buffer)?;
    let layout = buffer.layout();
    let (w, h) = (layout.width, layout.height);

    // Download spectrum for host-side C++ peak search.
    let spec = backend.download(buffer)?;

    let ((cx1, cy1), (cx2, cy2)) = find_peaks_cpp_style(
        &spec,
        w,
        h,
        sigma,
        min_frequency,
        max_frequency,
        smoothing_sigma,
    )
    .ok_or_else(|| VernierError::Backend("no carrier peaks found in spectrum".into()))?;

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

// ---------------------------------------------------------------------------
// C++ PatternPhase::peaksSearch implementation
// ---------------------------------------------------------------------------

/// Ports C++ `PatternPhase::peaksSearch` to Rust. Computes magnitude, applies
/// annulus mask + Gaussian blur, finds two peaks via half-plane search and
/// angular cone isolation, then orders them by signed column frequency.
///
/// Returns `None` if fewer than two valid bins exist (should not happen on any
/// non-trivial image).
fn find_peaks_cpp_style(
    spectrum: &[Complex32],
    width: usize,
    height: usize,
    sigma: Real,
    min_frequency: usize,
    max_frequency: usize,
    smoothing_sigma: Real,
) -> Option<((usize, usize), (usize, usize))> {
    let signed = |f: usize, n: usize| -> isize {
        let f = f as isize;
        let n = n as isize;
        if f > n / 2 { f - n } else { f }
    };

    // Magnitude array.
    let mut mag: Vec<Real> = spectrum.iter().map(|c| c.norm_sqr().sqrt()).collect();

    // Annulus mask: zero bins outside [min_frequency, max_frequency] from DC.
    // max_frequency = 0 means no upper limit.
    let min_r2 = (min_frequency * min_frequency) as Real;
    let max_r2 = if max_frequency > 0 {
        (max_frequency * max_frequency) as Real
    } else {
        Real::INFINITY
    };
    for fy in 0..height {
        let sfy = signed(fy, height) as Real;
        for fx in 0..width {
            let sfx = signed(fx, width) as Real;
            let r2 = sfx * sfx + sfy * sfy;
            if r2 < min_r2 || r2 > max_r2 {
                mag[fy * width + fx] = 0.0;
            }
        }
    }

    // Gaussian blur on magnitude (C++ cv::GaussianBlur).
    if smoothing_sigma > 0.0 {
        gaussian_blur_2d(&mut mag, width, height, smoothing_sigma);
    }

    // Peak 1: largest magnitude in canonical half-plane.
    let (cx1, cy1) = halfplane_argmax(&mag, width, height)?;
    let sfx1 = signed(cx1, width) as Real;
    let sfy1 = signed(cy1, height) as Real;

    // Angular cone exclusion around peak 1's direction (C++ applyAngularCut).
    let distance = (sfx1 * sfx1 + sfy1 * sfy1).sqrt();
    let center_angle = sfy1.atan2(sfx1);
    // C++: widthAngle = 2 * atan2(3*sigma, distance); we use half that as the
    // exclusion threshold so a bin is excluded when |angle_diff| < half_width.
    let half_width = (3.0 * sigma).atan2(distance);

    // Peak 2: largest magnitude outside the angular cone.
    let (cx2, cy2) =
        halfplane_argmax_angular_excl(&mag, width, height, center_angle, half_width)?;

    // Order so the larger signed column frequency is direction 1 (C++ convention:
    // swap if mainPeak1.x < mainPeak2.x in the shifted spectrum, equiv. to
    // sfx1 < sfx2 in unshifted).
    let sfx2 = signed(cx2, width) as Real;
    if sfx1 >= sfx2 {
        Some(((cx1, cy1), (cx2, cy2)))
    } else {
        Some(((cx2, cy2), (cx1, cy1)))
    }
}

/// Separable 2D Gaussian blur on a real-valued array, in place.
fn gaussian_blur_2d(data: &mut [Real], width: usize, height: usize, sigma: Real) {
    let radius = (3.0 * sigma).ceil() as usize;
    let n = 2 * radius + 1;
    let kernel: Vec<Real> = (0..n)
        .map(|i| {
            let x = i as Real - radius as Real;
            (-x * x / (2.0 * sigma * sigma)).exp()
        })
        .collect();
    let ksum: Real = kernel.iter().sum();
    let kernel: Vec<Real> = kernel.iter().map(|&k| k / ksum).collect();

    let mut tmp = vec![0.0_f32; width * height];

    // Blur along rows.
    for r in 0..height {
        for c in 0..width {
            let (mut v, mut w) = (0.0_f32, 0.0_f32);
            for (ki, &kv) in kernel.iter().enumerate() {
                let sc = c as isize + ki as isize - radius as isize;
                if sc >= 0 && (sc as usize) < width {
                    v += data[r * width + sc as usize] * kv;
                    w += kv;
                }
            }
            tmp[r * width + c] = if w > 0.0 { v / w } else { 0.0 };
        }
    }

    // Blur along columns.
    for r in 0..height {
        for c in 0..width {
            let (mut v, mut w) = (0.0_f32, 0.0_f32);
            for (ki, &kv) in kernel.iter().enumerate() {
                let sr = r as isize + ki as isize - radius as isize;
                if sr >= 0 && (sr as usize) < height {
                    v += tmp[sr as usize * width + c] * kv;
                    w += kv;
                }
            }
            data[r * width + c] = if w > 0.0 { v / w } else { 0.0 };
        }
    }
}

/// Argmax in the canonical half-plane (`sfx > 0`, or `sfx == 0` and `sfy > 0`).
fn halfplane_argmax(mag: &[Real], width: usize, height: usize) -> Option<(usize, usize)> {
    let signed = |f: usize, n: usize| -> isize {
        let f = f as isize;
        let n = n as isize;
        if f > n / 2 { f - n } else { f }
    };
    let mut best = Real::NEG_INFINITY;
    let mut result = None;
    for fy in 0..height {
        let sfy = signed(fy, height);
        for fx in 0..width {
            let sfx = signed(fx, width);
            if !(sfx > 0 || (sfx == 0 && sfy > 0)) {
                continue;
            }
            let m = mag[fy * width + fx];
            if m > best {
                best = m;
                result = Some((fx, fy));
            }
        }
    }
    result
}

/// Argmax in the canonical half-plane, excluding bins whose direction from DC
/// is within `half_width` radians of `center_angle`.
fn halfplane_argmax_angular_excl(
    mag: &[Real],
    width: usize,
    height: usize,
    center_angle: Real,
    half_width: Real,
) -> Option<(usize, usize)> {
    let signed = |f: usize, n: usize| -> isize {
        let f = f as isize;
        let n = n as isize;
        if f > n / 2 { f - n } else { f }
    };
    let mut best = Real::NEG_INFINITY;
    let mut result = None;
    for fy in 0..height {
        let sfy_i = signed(fy, height);
        let sfy = sfy_i as Real;
        for fx in 0..width {
            let sfx_i = signed(fx, width);
            let sfx = sfx_i as Real;
            if !(sfx_i > 0 || (sfx_i == 0 && sfy_i > 0)) {
                continue;
            }
            // Angular difference from center (shortest arc, in (-π, π]).
            let angle = sfy.atan2(sfx);
            let diff = ((angle - center_angle + PI).rem_euclid(TAU)) - PI;
            if diff.abs() < half_width {
                continue;
            }
            let m = mag[fy * width + fx];
            if m > best {
                best = m;
                result = Some((fx, fy));
            }
        }
    }
    result
}
