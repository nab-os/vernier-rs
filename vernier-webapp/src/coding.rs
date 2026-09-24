//! The square-level end of the chain: the dewarped thumbnail the decoder builds
//! out of the phase maps, and the coding sites it reads off it.
//!
//! Everything here comes from `vernier_pose::checkerboard::read_squares` — the
//! decoder's own sampling, binarization and parity marking, kept instead of
//! discarded. This module only arranges the squares it hands back into a
//! canvas-shaped grid, one pixel per square, so the panel can draw them.

use vernier_core::Real;
use vernier_patterns::checkerboard::CodeLayout;
use vernier_pose::checkerboard::{CheckerboardCode, CheckerboardError, read_squares};
use vernier_spectral::spectrum::Detection;

/// A square the code inverted, in thumbnail cell coordinates.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Site {
    pub col: usize,
    pub row: usize,
}

/// The extracted thumbnail, square and padded so it can be blitted straight
/// onto a stage canvas and overlaid in the same coordinate space.
#[derive(Clone, PartialEq, Debug)]
pub struct Thumbnail {
    /// Side of the canvas, in cells. One cell is one pattern square.
    pub side: usize,
    /// Grey level per cell, row-major `side × side`, stretched over the squares
    /// actually sampled.
    pub levels: Vec<u8>,
    /// Which cells hold a square at all. The lattice is rarely square and never
    /// fills the frame, so the rest is padding, not black squares.
    pub present: Vec<bool>,
    /// The squares that break the checkerboard parity: the code's own marks.
    pub sites: Vec<Site>,
    /// Where the centre of the camera image lands, in cell coordinates.
    pub centre: (f64, f64),
    /// Squares sampled, and how many of those survived binarization.
    pub sampled: usize,
    pub binarized: usize,
    /// What the full decode made of the same squares.
    pub code: Result<CheckerboardCode, CheckerboardError>,
}

impl Thumbnail {
    /// Runs the square-level decode over one detection.
    ///
    /// `None` when the phases put no square inside the frame, which is what a
    /// pattern zoomed past the sampling radius looks like.
    pub fn extract(
        detection: &Detection,
        intensity: &[f32],
        order: u32,
        layout: CodeLayout,
    ) -> Option<Self> {
        let readout = read_squares(detection, intensity, order, layout)?;

        let (i_min, i_max) = readout.i_range;
        let (j_min, j_max) = readout.j_range;
        let side = (i_max - i_min + 1).max(j_max - j_min + 1) as usize;
        // Centre the lattice in the square canvas rather than cornering it.
        let pad_col = (side as i64 - (i_max - i_min + 1)) / 2;
        let pad_row = (side as i64 - (j_max - j_min + 1)) / 2;
        let cell_of = |i: i64, j: i64| {
            (
                (i - i_min + pad_col) as usize,
                (j - j_min + pad_row) as usize,
            )
        };

        // One stretch over the whole thumbnail, so a coding site reads as a
        // square of the wrong colour rather than as a locally odd grey.
        let (mut low, mut high) = (Real::INFINITY, Real::NEG_INFINITY);
        for square in &readout.squares {
            low = low.min(square.mean);
            high = high.max(square.mean);
        }
        let span = if (high - low).abs() < 1e-12 { 1.0 } else { high - low };

        let mut levels = vec![0u8; side * side];
        let mut present = vec![false; side * side];
        let mut sites = Vec::new();
        let mut binarized = 0usize;

        for square in &readout.squares {
            let (col, row) = cell_of(square.i, square.j);
            let cell = row * side + col;
            levels[cell] = (((square.mean - low) / span) * 255.0).clamp(0.0, 255.0) as u8;
            present[cell] = true;
            if square.is_white.is_some() {
                binarized += 1;
            }
            if square.is_coding_site == Some(true) {
                sites.push(Site { col, row });
            }
        }

        Some(Self {
            side,
            levels,
            present,
            sites,
            centre: (
                (readout.centre.0 - i_min as Real + pad_col as Real) as f64 + 0.5,
                (readout.centre.1 - j_min as Real + pad_row as Real) as f64 + 0.5,
            ),
            sampled: readout.squares.len(),
            binarized,
            code: readout.code,
        })
    }

    /// The decode's verdict, as one line for the readouts.
    pub fn verdict(&self) -> String {
        match &self.code {
            Ok(code) => format!(
                "square ({}, {}) · 1 in {:.0}",
                code.centre_square.0,
                code.centre_square.1,
                1.0 / code.false_accept.max(Real::MIN_POSITIVE),
            ),
            Err(error) => format!("{error}"),
        }
    }

    /// The two code windows as bit strings, when one was read.
    pub fn windows(&self) -> Option<(String, String)> {
        let code = self.code.as_ref().ok()?;
        let render = |bits: &[u8]| bits.iter().map(|&b| char::from(b'0' + b)).collect();
        Some((render(&code.x_window), render(&code.y_window)))
    }
}
