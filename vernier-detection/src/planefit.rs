//! Least-squares phase-plane fitting — where the resolution actually comes from.
//!
//! After a single spectral lobe is isolated and inverse-transformed, the
//! argument of the resulting complex field is the wrapped phase of one pattern
//! direction. The Vernier method does not read a single value from this: it fits
//! a *plane* `φ(i, j) = a·i + b·j + c` across the whole (unwrapped) phase map by
//! least squares (André et al. 2021, §III-A). Three payoffs fall out of the fit:
//!
//! - `c` is the high-resolution sub-period phase at the image center (the `i, j`
//!   here are counted from the center), i.e. `φ_x` / `φ_y` modulo 2π.
//! - `(a, b)` are the per-pixel phase gradients, which give the spectral peak
//!   location `(m, n) = (w·a/2π, h·b/2π)` and hence the orientation
//!   `θ = atan2(b, a)` (+ quadrant).
//!
//! Fitting across every pixel is the entire reason the method reaches
//! 1/1000-pixel resolution: it averages the redundant phase information spread
//! over the whole image rather than trusting one noisy bin.
//!
//! The fit must run on **unwrapped** phase — a plane fit over wrapped values is
//! corrupted by the 2π discontinuities. This module unwraps row-by-row and
//! column-wise before fitting; see [`fit_plane`].

use vernier_core::Real;

use crate::unwrap::unwrap_1d;

/// Coefficients of a fitted phase plane `φ(i, j) = a·i + b·j + c`, with `(i, j)`
/// measured from the image center.
#[derive(Clone, Copy, Debug)]
pub struct PhasePlane {
    /// Phase gradient along the column (x) axis, radians per pixel.
    pub a: Real,
    /// Phase gradient along the row (y) axis, radians per pixel.
    pub b: Real,
    /// Phase at the image center, radians. This is the high-resolution phase
    /// `φ` used for sub-period position (modulo 2π).
    pub c: Real,
}

impl PhasePlane {
    /// Orientation implied by the plane gradients, `atan2(b, a)` in `(-π, π]`.
    ///
    /// This is the in-image angle of the pattern direction before quadrant
    /// disambiguation (the `+ q·π/2` term, resolved later from the missing
    /// corner). Mirrors André et al. 2021, Eq. 3.
    pub fn orientation(&self) -> Real {
        self.b.atan2(self.a)
    }

    /// Spectral peak location `(m, n)` implied by the gradients, given image
    /// dimensions. `m = w·a/2π`, `n = h·b/2π` (André et al. 2022, Eq. 7).
    pub fn peak_location(&self, width: usize, height: usize) -> (Real, Real) {
        use vernier_core::scalar::consts::TAU;
        (width as Real * self.a / TAU, height as Real * self.b / TAU)
    }
}

/// Fits `φ(i, j) = a·i + b·j + c` to a wrapped phase map by least squares.
///
/// Steps:
/// 1. Unwrap the wrapped phases into a continuous surface (row unwrap to fix
///    horizontal jumps, then column unwrap of the first column to fix vertical
///    offset between rows — a simple separable 2D unwrap sufficient for the
///    near-planar phase of a periodic pattern).
/// 2. Solve the normal equations for the plane in coordinates centered on the
///    image, so `c` is the center phase directly.
///
/// `wrapped` is row-major, length `width * height`. `crop_factor ∈ [0, 1)`
/// mirrors the C++ `RegressionPlane::cropFactor` (default 0.5): only the center
/// `(1 − crop_factor)` fraction of each axis is used in the fit, which avoids
/// edge artefacts from the bandpass IFFT on real images. Pass `0.0` for no crop
/// (legacy behaviour, full image).
pub fn fit_plane(wrapped: &[Real], width: usize, height: usize, crop_factor: Real) -> PhasePlane {
    let phase = unwrap_2d(wrapped, width, height);
    fit_plane_to_unwrapped(&phase, width, height, crop_factor)
}

/// Separable 2D phase unwrap: unwrap each row, then reconcile rows via the first
/// column. Returns the unwrapped phase (a continuous surface), which is what the
/// plane fit needs — and what the megarena decode needs to compute cell indices
/// `round(φ/2π)` (the wrapped phase, confined to (-π, π], always rounds to 0).
///
/// Sufficient for the near-planar phase of a periodic pattern; steep or noisy
/// phase would want a quality-guided unwrap.
///
/// ## C++ parity note
///
/// The C++ library (`Spatial::quartersUnwrapPhase`) propagates outward from the
/// image center through four quadrants instead of unwrapping row-by-row. On
/// well-conditioned phase maps the two are equivalent: validated on a real
/// 856×856 megarena photo, both produce plane gradients identical to ~1e-15
/// (machine epsilon) with the same fit residual; they differ only in the
/// absolute offset `c`, which reflects the unwrap origin (a convention, not
/// accuracy). Quarter-propagation is more robust only when the seed row/column
/// (row 0, column 0 here) fall on a noisy or occluded region — a degraded-image
/// case. Kept separable for simplicity; quarter-propagation is a documented
/// upgrade if heavily degraded inputs become a target.
pub fn unwrap_2d(wrapped: &[Real], width: usize, height: usize) -> Vec<Real> {
    let mut phase = wrapped.to_vec();

    // Unwrap each row in place.
    for r in 0..height {
        let start = r * width;
        unwrap_1d(&mut phase[start..start + width]);
    }
    // Unwrap down the first column, then propagate each row's offset so rows are
    // mutually consistent.
    let mut first_col: Vec<Real> = (0..height).map(|r| phase[r * width]).collect();
    let before: Vec<Real> = first_col.clone();
    unwrap_1d(&mut first_col);
    for r in 0..height {
        let row_shift = first_col[r] - before[r];
        if row_shift != 0.0 {
            let start = r * width;
            for v in &mut phase[start..start + width] {
                *v += row_shift;
            }
        }
    }
    phase
}

/// Fits the plane to an already-unwrapped phase surface.
///
/// `crop_factor` trims a border of `(crop_factor/2) * dimension` pixels on each
/// side before fitting; coordinates remain centered on the FULL image so `c` is
/// still the phase at the full-image center. Mirrors C++ `RegressionPlane`.
pub(crate) fn fit_plane_to_unwrapped(
    phase: &[Real],
    width: usize,
    height: usize,
    crop_factor: Real,
) -> PhasePlane {
    // --- Least-squares plane fit, centered coordinates ---
    let col_off = ((width as Real * crop_factor) / 2.0) as usize;
    let row_off = ((height as Real * crop_factor) / 2.0) as usize;

    let cropped_w = width - 2 * col_off;
    let cropped_h = height - 2 * row_off;

    // C++ integer division
    let center_x = (cropped_w / 2) as Real;
    let center_y = (cropped_h / 2) as Real;

    // Accumulate normal-equation sums for [a, b, c].
    let (mut sii, mut sjj, mut sij) = (0.0, 0.0, 0.0);
    let (mut si, mut sj, mut sn) = (0.0, 0.0, 0.0);
    let (mut spi, mut spj, mut sp) = (0.0, 0.0, 0.0);

    for r in row_off..(height - row_off) {
        let j = (r - row_off) as Real - center_y;
        for col in col_off..(width - col_off) {
            let i = (col - col_off) as Real - center_x;
            let p = phase[r * width + col];
            sii += i * i;
            sjj += j * j;
            sij += i * j;
            si += i;
            sj += j;
            sn += 1.0;
            spi += p * i;
            spj += p * j;
            sp += p;
        }
    }

    // Solve the 3x3 symmetric system:
    // [sii sij si][a]   [spi]
    // [sij sjj sj][b] = [spj]
    // [si  sj  sn][c]   [sp ]
    let (a, b, c) = solve_3x3(
        [[sii, sij, si], [sij, sjj, sj], [si, sj, sn]],
        [spi, spj, sp],
    );

    PhasePlane { a, b, c }
}

/// Solves a 3x3 linear system by Cramer's rule. Adequate and clear for a
/// reference path; the matrix is tiny and well-conditioned for centered image
/// coordinates.
fn solve_3x3(m: [[Real; 3]; 3], v: [Real; 3]) -> (Real, Real, Real) {
    let det = det3(m);
    let mx = det3([
        [v[0], m[0][1], m[0][2]],
        [v[1], m[1][1], m[1][2]],
        [v[2], m[2][1], m[2][2]],
    ]);
    let my = det3([
        [m[0][0], v[0], m[0][2]],
        [m[1][0], v[1], m[1][2]],
        [m[2][0], v[2], m[2][2]],
    ]);
    let mz = det3([
        [m[0][0], m[0][1], v[0]],
        [m[1][0], m[1][1], v[1]],
        [m[2][0], m[2][1], v[2]],
    ]);
    (mx / det, my / det, mz / det)
}

fn det3(m: [[Real; 3]; 3]) -> Real {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

#[cfg(test)]
mod tests {
    use super::*;
    use vernier_core::scalar::consts::{PI, TAU};

    /// Build a wrapped phase map for a known plane and check we recover it.
    #[test]
    fn recovers_a_known_plane() {
        let (width, height) = (32, 32);
        let (true_a, true_b, true_c) = (0.30, -0.15, 0.4);
        let center_x = width as Real / 2.0;
        let center_y = height as Real / 2.0;

        let mut wrapped = vec![0.0; width * height];
        for r in 0..height {
            for col in 0..width {
                let i = col as Real - center_x;
                let j = r as Real - center_y;
                let mut p = true_a * i + true_b * j + true_c;
                // Wrap into (-π, π].
                p = ((p + PI).rem_euclid(TAU)) - PI;
                wrapped[r * width + col] = p;
            }
        }

        let plane = fit_plane(&wrapped, width, height, 0.0);
        assert!((plane.a - true_a).abs() < 1e-3, "a={}", plane.a);
        assert!((plane.b - true_b).abs() < 1e-3, "b={}", plane.b);
        // c recovered modulo 2π.
        let dc = ((plane.c - true_c + PI).rem_euclid(TAU)) - PI;
        assert!(dc.abs() < 1e-3, "c off by {dc}");
    }
}
