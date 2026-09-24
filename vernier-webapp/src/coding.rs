//! The lattice end of the chain: the dewarped thumbnail each decoder builds out
//! of the phase maps, and the coding sites it reads off it.
//!
//! Both coded patterns get here, by different routes. The checkerboard writes
//! its bits by inverting squares, so
//! `vernier_pose::checkerboard::read_squares_with_packing` hands back squares
//! and which of them break the parity. The megarena writes
//! its bits by removing dots, so `vernier_pose::absolute::read_cells` hands back
//! carrier cells and which of them carry a bit. Neither is recomputed here:
//! this module only arranges what they return into a canvas-shaped grid, one
//! pixel per lattice node, so the panel can draw it.

use vernier_core::Real;
use vernier_patterns::checkerboard::{CodeLayout, CodePacking};
use vernier_pose::absolute::{CellRole, read_cells};
use vernier_pose::checkerboard::read_squares_with_packing;
use vernier_spectral::spectrum::Detection;

/// Past this many markers the overlay stops being a picture and starts being a
/// cost. The lattice is visible without it at that density.
const MAX_SITES: usize = 3000;

/// Which decoder to run, and what it needs.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Source {
    Checkerboard { order: u32, layout: CodeLayout, packing: CodePacking },
    Megarena { order: u32 },
}

/// What a marked node turned out to be.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Mark {
    /// The code wrote a 0 here, and the node shows it: a square inverted, or a
    /// dot removed.
    Zero,
    /// A coding node the code left alone, which reads as a 1.
    One,
    /// A node both axes gate, so neither can be read from it.
    Ambiguous,
    /// The corner dropped from every megarena cell to fix the orientation. Not
    /// code, but part of how the code is found.
    Corner,
}

impl Mark {
    /// Fill colour, shared by the overlay and its legend.
    pub fn colour(self) -> &'static str {
        match self {
            Mark::Zero => "#ff8c42",
            Mark::One => "#4db5ff",
            Mark::Ambiguous => "#9aa0a6",
            Mark::Corner => "#b57bff",
        }
    }
}

/// Colour of the crosshair on the node the image centre falls in.
pub const CENTRE_COLOUR: &str = "#39ff88";

/// A lattice node worth marking, in thumbnail cell coordinates.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Site {
    pub col: usize,
    pub row: usize,
    pub mark: Mark,
}

/// The extracted thumbnail, square and padded so it can be blitted straight
/// onto a stage canvas and overlaid in the same coordinate space.
#[derive(Clone, PartialEq, Debug)]
pub struct Thumbnail {
    /// Side of the canvas, in cells. One cell is one lattice node.
    pub side: usize,
    /// Grey level per cell, row-major `side × side`, stretched over the nodes
    /// actually sampled.
    pub levels: Vec<u8>,
    /// Which cells hold a node at all. The lattice is rarely square and never
    /// fills the frame, so the rest is padding, not measurement.
    pub present: Vec<bool>,
    /// The nodes the code wrote to.
    pub sites: Vec<Site>,
    /// Where the centre of the camera image lands, in cell coordinates.
    pub centre: (f64, f64),
    /// What one node is called, for the readouts.
    pub unit: &'static str,
    /// Nodes sampled, and how many of those the decoder could judge.
    pub sampled: usize,
    pub judged: usize,
    /// The decode's verdict, ready to show.
    pub verdict: String,
    /// The two code windows as bit strings, when one was read.
    pub windows: Option<(String, String)>,
    /// What each colour on this panel means, in the order to show them. Only
    /// the marks this pattern can actually produce.
    pub legend: Vec<(&'static str, &'static str)>,
}

/// Lays a lattice out in a square canvas, centred, one cell per node.
struct Grid {
    side: usize,
    i_min: i64,
    j_min: i64,
    pad_col: i64,
    pad_row: i64,
}

impl Grid {
    fn new(i_range: (i64, i64), j_range: (i64, i64)) -> Self {
        let width = i_range.1 - i_range.0 + 1;
        let height = j_range.1 - j_range.0 + 1;
        let side = width.max(height);
        Self {
            side: side as usize,
            i_min: i_range.0,
            j_min: j_range.0,
            // Centre the lattice in the square canvas rather than corner it.
            pad_col: (side - width) / 2,
            pad_row: (side - height) / 2,
        }
    }

    fn cell_of(&self, i: i64, j: i64) -> (usize, usize) {
        (
            (i - self.i_min + self.pad_col) as usize,
            (j - self.j_min + self.pad_row) as usize,
        )
    }

    /// Same mapping for a fractional position, landing on the cell's middle.
    fn point_of(&self, i: Real, j: Real) -> (f64, f64) {
        (
            i - self.i_min as Real + self.pad_col as Real + 0.5,
            j - self.j_min as Real + self.pad_row as Real + 0.5,
        )
    }
}

/// Stretches node values over the full grey range and blits them into the grid.
/// One stretch over the whole thumbnail, so a coding site reads as a node of
/// the wrong brightness rather than as a locally odd grey.
fn paint_nodes(grid: &Grid, nodes: &[(i64, i64, Real)]) -> (Vec<u8>, Vec<bool>) {
    let (mut low, mut high) = (Real::INFINITY, Real::NEG_INFINITY);
    for &(_, _, value) in nodes {
        low = low.min(value);
        high = high.max(value);
    }
    let span = if (high - low).abs() < 1e-12 { 1.0 } else { high - low };

    let mut levels = vec![0u8; grid.side * grid.side];
    let mut present = vec![false; grid.side * grid.side];
    for &(i, j, value) in nodes {
        let (col, row) = grid.cell_of(i, j);
        let cell = row * grid.side + col;
        levels[cell] = (((value - low) / span) * 255.0).clamp(0.0, 255.0) as u8;
        present[cell] = true;
    }
    (levels, present)
}

fn bits_to_string(bits: &[u8]) -> String {
    bits.iter().map(|&b| char::from(b'0' + b)).collect()
}

impl Thumbnail {
    /// Runs the lattice-level decode over one detection.
    ///
    /// `None` when the decoder could not get as far as a lattice — a pattern
    /// zoomed past its sampling radius, or a frame too small to show one
    /// complete megarena cell.
    pub fn extract(detection: &Detection, intensity: &[f32], source: Source) -> Option<Self> {
        match source {
            Source::Checkerboard { order, layout, packing } => {
                Self::from_squares(detection, intensity, order, layout, packing)
            }
            Source::Megarena { order } => Self::from_cells(detection, intensity, order),
        }
    }

    /// The checkerboard: nodes are squares, and a site is a square whose colour
    /// breaks the parity. The code writes a 0 by inverting a square, so every
    /// site the parity finds is a 0 — the 1s are the squares it left alone, and
    /// those are invisible until the supercell grid is known.
    fn from_squares(
        detection: &Detection,
        intensity: &[f32],
        order: u32,
        layout: CodeLayout,
        packing: CodePacking,
    ) -> Option<Self> {
        let readout = read_squares_with_packing(detection, intensity, order, layout, packing)?;
        let grid = Grid::new(readout.i_range, readout.j_range);

        let nodes: Vec<(i64, i64, Real)> = readout
            .squares
            .iter()
            .map(|square| (square.i, square.j, square.mean))
            .collect();
        let (levels, present) = paint_nodes(&grid, &nodes);

        let sites: Vec<Site> = readout
            .squares
            .iter()
            .filter(|square| square.is_coding_site == Some(true))
            .map(|square| {
                let (col, row) = grid.cell_of(square.i, square.j);
                Site { col, row, mark: Mark::Zero }
            })
            .collect();

        let judged = readout
            .squares
            .iter()
            .filter(|square| square.is_white.is_some())
            .count();

        let (verdict, windows) = match &readout.code {
            Ok(code) => (
                format!(
                    "square ({}, {}) · 1 in {:.0}",
                    code.centre_square.0,
                    code.centre_square.1,
                    1.0 / code.false_accept.max(Real::MIN_POSITIVE),
                ),
                Some((
                    bits_to_string(&code.x_window),
                    bits_to_string(&code.y_window),
                )),
            ),
            Err(error) => (format!("{error}"), None),
        };

        Some(Self {
            side: grid.side,
            levels,
            present,
            sites: cap(sites),
            centre: grid.point_of(readout.centre.0, readout.centre.1),
            unit: "squares",
            sampled: readout.squares.len(),
            judged,
            verdict,
            windows,
            legend: vec![
                (Mark::Zero.colour(), "inverted square · bit 0"),
                (CENTRE_COLOUR, "image centre"),
            ],
        })
    }

    /// The megarena: nodes are carrier cells, and a site is a cell the code
    /// gates. A whole row or column goes dark for a 0, so the sites come in
    /// stripes, each coloured by the bit its triple read.
    fn from_cells(detection: &Detection, intensity: &[f32], order: u32) -> Option<Self> {
        let readout = read_cells(detection, intensity, order)?;
        let grid = Grid::new(readout.x_range, readout.y_range);

        // The dot site is what the thumbnail shows: bright where the dot is
        // there, dark where the code took it away. Cells at the frame edge can
        // miss the dot window entirely and have only a surround to show.
        let nodes: Vec<(i64, i64, Real)> = readout
            .cells
            .iter()
            .filter_map(|cell| Some((cell.x, cell.y, cell.white.or(cell.background)?)))
            .collect();
        let (levels, present) = paint_nodes(&grid, &nodes);

        let sites: Vec<Site> = readout
            .cells
            .iter()
            .filter(|cell| cell.role != CellRole::Carrier)
            .map(|cell| {
                let (col, row) = grid.cell_of(cell.x, cell.y);
                let mark = match (cell.role, cell.bit) {
                    (CellRole::MissingCorner, _) => Mark::Corner,
                    (_, Some(0)) => Mark::Zero,
                    (_, Some(_)) => Mark::One,
                    (_, None) => Mark::Ambiguous,
                };
                Site { col, row, mark }
            })
            .collect();

        let judged = readout.cells.iter().filter(|cell| cell.bit.is_some()).count();

        let (verdict, windows) = match &readout.code {
            Some(code) => (
                format!(
                    "quadrant {} · centre bit {} · {}",
                    code.k3, code.x_k_center, code.y_k_center
                ),
                Some((
                    bits_to_string(&code.x_window),
                    bits_to_string(&code.y_window),
                )),
            ),
            None => (
                "no window of consecutive triples matched the LFSR".to_string(),
                None,
            ),
        };

        Some(Self {
            side: grid.side,
            levels,
            present,
            sites: cap(sites),
            centre: grid.point_of(readout.centre.0, readout.centre.1),
            unit: "cells",
            sampled: readout.cells.len(),
            judged,
            verdict,
            windows,
            legend: vec![
                (Mark::Zero.colour(), "dot removed · bit 0"),
                (Mark::One.colour(), "dot kept · bit 1"),
                (Mark::Ambiguous.colour(), "both axes · unread"),
                (Mark::Corner.colour(), "dropped corner"),
                (CENTRE_COLOUR, "image centre"),
            ],
        })
    }
}

/// Keeps the overlay from turning into thousands of nodes on a dense lattice.
fn cap(mut sites: Vec<Site>) -> Vec<Site> {
    sites.truncate(MAX_SITES);
    sites
}
