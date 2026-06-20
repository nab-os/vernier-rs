//! Stamp pattern generation.
//!
//! The last of the four calibrated pattern families. Interface fixed here; the
//! specific stamp layout is the boundary to implement against the pattern spec.

use vernier_core::GrayImage;

use crate::PatternPose;

/// Parameters of a stamp pattern.
#[derive(Clone, Copy, Debug)]
pub struct Stamp {
    /// Pixel size of the stamp tile.
    pub tile_px: usize,
}

impl Stamp {
    /// Creates a stamp pattern descriptor.
    pub fn new(tile_px: usize) -> Self {
        Self { tile_px }
    }

    /// Renders the stamp pattern at `pose`.
    ///
    /// Boundary stub: blank field at the right dimensions until the stamp tile
    /// layout is implemented.
    pub fn render(&self, width: usize, height: usize, _pose: &PatternPose) -> GrayImage {
        // TODO(stamp-layout): rasterize the stamp tile, posed by `pose`.
        GrayImage::zeros(width, height)
    }
}
