//! Shared rasterization helpers for pattern generators. The per-pixel closures
//! are independent, so they'd map cleanly onto a GPU kernel later; for now
//! they're plain CPU loops.

use vernier_core::{GrayImage, Real};

/// Fills a new [`GrayImage`] by evaluating `f(x, y)` at each pixel, where `f`
/// takes pixel coordinates and returns an intensity.
pub fn render_with<F>(width: usize, height: usize, f: F) -> GrayImage
where
    F: Fn(Real, Real) -> Real,
{
    let mut image = GrayImage::zeros(width, height);
    let pixels = image.as_mut_slice();
    for row in 0..height {
        for col in 0..width {
            pixels[row * width + col] = f(col as Real, row as Real) as f32;
        }
    }
    image
}

/// Rotates pixel coordinates `(x, y)` by `-theta` about the image center, into
/// the pattern's own axis-aligned frame. Evaluating a pattern in this frame is
/// how a single 1D definition produces an arbitrarily-oriented pattern.
#[inline]
pub fn into_pattern_frame(x: Real, y: Real, center_x: Real, center_y: Real, theta: Real) -> (Real, Real) {
    let (delta_x, delta_y) = (x - center_x, y - center_y);
    let (sin_theta, cos_theta) = (-theta).sin_cos();
    (cos_theta * delta_x - sin_theta * delta_y, sin_theta * delta_x + cos_theta * delta_y)
}
