//! Stamp pattern generation. Interface only for now — the stamp tile layout is
//! left unimplemented.

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

    /// Renders the stamp pattern at `pose`. Stub: returns a blank field of the
    /// right size until the tile layout is implemented.
    pub fn render(&self, width: usize, height: usize, _pose: &PatternPose) -> GrayImage {
        // TODO(stamp-layout): rasterize the stamp tile, posed by `pose`.
        GrayImage::zeros(width, height)
    }
}
