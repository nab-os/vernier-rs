//! QR-code-like pattern generation.
//!
//! One of the calibrated pattern families the library detects. Like megarena,
//! the data-carrying layout is pattern-specific; this module fixes the interface
//! and leaves the cell-encoding as a boundary rather than inventing one.

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

    /// Renders the pattern at `pose`.
    ///
    /// Boundary stub: returns a blank field at the correct dimensions. The
    /// module layout/encoding is the part to implement against the specific
    /// QR-like construction used by the Vernier patterns.
    pub fn render(&self, width: usize, height: usize, _pose: &PatternPose) -> GrayImage {
        // TODO(qr-encoding): rasterize the module grid (finder patterns + data)
        // rotated/translated by `pose`.
        GrayImage::zeros(width, height)
    }
}
