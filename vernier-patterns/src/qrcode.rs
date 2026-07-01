//! QR-code-like pattern generation. Interface only for now — the cell encoding
//! is left unimplemented rather than guessed at.

use vernier_core::GrayImage;

use crate::PatternPose;

/// Parameters of a QR-like pattern.
#[derive(Clone, Copy, Debug)]
pub struct QrLike {
    /// Number of modules (cells) along each axis.
    pub modules: usize,
    /// Pixel size of one module.
    pub module_px: usize,
}

impl QrLike {
    /// Creates a QR-like pattern descriptor.
    pub fn new(modules: usize, module_px: usize) -> Self {
        Self { modules, module_px }
    }

    /// Renders the pattern at `pose`. Stub: returns a blank field of the right
    /// size until the module layout is implemented.
    pub fn render(&self, width: usize, height: usize, _pose: &PatternPose) -> GrayImage {
        // TODO(qr-encoding): rasterize the module grid (finder patterns + data)
        // rotated/translated by `pose`.
        GrayImage::zeros(width, height)
    }
}
