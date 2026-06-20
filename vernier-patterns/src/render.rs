//! Shared rasterization helpers for pattern generators.
//!
//! Small, generator-agnostic utilities. The interesting design note for the
//! project: rasterizing a pattern is *itself* an embarrassingly parallel,
//! per-pixel operation — exactly the kind of thing that becomes a trivial GPU
//! kernel later (generating synthetic validation frames on-device, with no
//! host round-trip). For now these are straightforward CPU loops; they are
//! deliberately written as per-pixel closures so the parallel structure is
//! already visible.

use vernier_core::{GrayImage, Real};

/// Fills a new [`GrayImage`] by evaluating `f(x, y)` at each pixel.
///
/// `f` receives pixel coordinates as `Real` and returns an intensity. The
/// per-pixel-closure shape mirrors a compute kernel: no inter-pixel dependence,
/// so this maps directly to a GPU dispatch when wanted.
pub fn render_with<F>(width: usize, height: usize, f: F) -> GrayImage
where
    F: Fn(Real, Real) -> Real,
{
    let mut img = GrayImage::zeros(width, height);
    let buf = img.as_mut_slice();
    for row in 0..height {
        for col in 0..width {
            buf[row * width + col] = f(col as Real, row as Real) as f32;
        }
    }
    img
}

/// Rotates pixel coordinates `(x, y)` by `-theta` about the image center,
/// mapping image space into the pattern's own (axis-aligned) frame.
///
/// Generators evaluate their pattern in this rotated frame, which is how a
/// single 1D periodic definition produces an arbitrarily-oriented pattern.
#[inline]
pub fn into_pattern_frame(x: Real, y: Real, cx: Real, cy: Real, theta: Real) -> (Real, Real) {
    let (dx, dy) = (x - cx, y - cy);
    let (s, c) = (-theta).sin_cos();
    (c * dx - s * dy, s * dx + c * dy)
}
