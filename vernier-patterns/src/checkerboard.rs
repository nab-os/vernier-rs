//! Coded checkerboard: a 50/50 carrier with the megarena's LFSR position code.
//!
//! The carriers sit at ±45° to the square edges, and at a square centre
//! `φ₁ = π(i+j)`, `φ₂ = π(i−j)`, so the phases index the squares.
//!
//! Squares are grouped in 3×3 supercells. Two per supercell are coding sites
//! (one per axis); a site is painted against its parity when its bit is 0.
//! Flipping a square centre only shrinks the carrier, never shifts its phase,
//! and the sites alternate colour so the fill stays 50/50.
//!
//! Two layouts: `Squares` (upright squares, code along their edges) and
//! `Diamonds` (squares turned 45°, code along their diagonals, so the code grid
//! stays upright).
//!
//! The code period `3·(2ⁿ − 1)` is odd, so the pattern really repeats after two
//! periods. We only claim one.

use vernier_core::scalar::consts::{PI, SQRT_2};
use vernier_core::{GrayImage, Real};

use crate::PatternPose;
use crate::lfsr::Lfsr;
use crate::render::{into_pattern_frame, render_with};

/// Squares per supercell edge.
pub const CELL: i64 = 3;

/// x-code site within a supercell, as `(i mod 3, j mod 3)`.
pub const X_SITE: (i64, i64) = (1, 0);

pub const Y_SITE: (i64, i64) = (0, 1);

/// Same sites for [`CodeLayout::Diamonds`], as `(u mod 3, v mod 3)`.
pub const U_SITE: (i64, i64) = (1, 0);

pub const V_SITE: (i64, i64) = (0, 1);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CodeAxis {
    X,
    Y,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum CodeLayout {
    /// Upright squares, code along their edges `(i, j)`.
    #[default]
    Squares,
    /// Squares turned 45°, code along their diagonals `(i+j, i−j)`, which run
    /// along the pattern axes. Costs √2 of range, and each code band is one colour.
    Diamonds,
}

impl CodeLayout {
    /// Angle of the square lattice in the pattern frame.
    pub fn lattice_angle(self) -> Real {
        match self {
            Self::Squares => 0.0,
            Self::Diamonds => PI / 4.0,
        }
    }

    /// Pattern frame to square-lattice frame.
    pub fn to_lattice(self, x: Real, y: Real) -> (Real, Real) {
        match self {
            Self::Squares => (x, y),
            Self::Diamonds => ((x + y) / SQRT_2, (y - x) / SQRT_2),
        }
    }

    /// Square-lattice frame to pattern frame.
    pub fn from_lattice(self, x: Real, y: Real) -> (Real, Real) {
        match self {
            Self::Squares => (x, y),
            Self::Diamonds => ((x - y) / SQRT_2, (x + y) / SQRT_2),
        }
    }
}

/// Parameters of a coded checkerboard pattern.
#[derive(Clone, Debug)]
pub struct Checkerboard {
    /// Square side, in pixels.
    pub square_px: Real,
    pub order: u32,
    /// Same sequence for both axes.
    code: Lfsr,
    /// LFSR index at supercell 0.
    lfsr_offset: i64,
    /// Sub-samples per pixel edge; hard edges alias without it.
    supersample: u32,
    layout: CodeLayout,
}

impl Checkerboard {
    /// `None` if `order` is unsupported.
    pub fn new(square_px: Real, order: u32) -> Option<Self> {
        let code = Lfsr::maximal(order)?;
        Some(Self {
            square_px,
            order,
            code,
            lfsr_offset: 0,
            supersample: 4,
            layout: CodeLayout::Squares,
        })
    }

    pub fn with_code_layout(mut self, layout: CodeLayout) -> Self {
        self.layout = layout;
        self
    }

    pub fn code_layout(&self) -> CodeLayout {
        self.layout
    }

    pub fn with_lfsr_offset(mut self, offset: i64) -> Self {
        self.lfsr_offset = offset;
        self
    }

    pub fn with_supersample(mut self, supersample: u32) -> Self {
        self.supersample = supersample.max(1);
        self
    }

    pub fn code(&self) -> &Lfsr {
        &self.code
    }

    /// `a·√2`. Give this to the detector as the period, not the square side.
    pub fn carrier_period_px(&self) -> Real {
        self.square_px * SQRT_2
    }

    /// Range in code steps, `3·(2ⁿ − 1)`. Compare layouts with `range_px`.
    pub fn range_squares(&self) -> i64 {
        CELL * self.code.len() as i64
    }

    /// Range in pixels. Diagonal steps are `a/√2`.
    pub fn range_px(&self) -> Real {
        let steps = self.range_squares() as Real;
        match self.layout {
            CodeLayout::Squares => steps * self.square_px,
            CodeLayout::Diamonds => steps * self.square_px / SQRT_2,
        }
    }

    /// A pattern-frame offset, reduced to the smallest equivalent one modulo the
    /// code period. The period lattice is turned with the squares.
    pub fn wrap_offset(&self, dx: Real, dy: Real) -> (Real, Real) {
        let period = self.range_squares() as Real * self.square_px;
        let wrap = |e: Real| e - period * (e / period).round();
        let (x, y) = self.layout.to_lattice(dx, dy);
        self.layout.from_lattice(wrap(x), wrap(y))
    }

    pub fn code_bit(&self, index: i64) -> u8 {
        let k = (index + self.lfsr_offset).rem_euclid(self.code.len() as i64) as usize;
        self.code.bit_at(k)
    }

    pub fn coding_axis(i: i64, j: i64) -> Option<CodeAxis> {
        let within = (i.rem_euclid(CELL), j.rem_euclid(CELL));
        if within == X_SITE {
            Some(CodeAxis::X)
        } else if within == Y_SITE {
            Some(CodeAxis::Y)
        } else {
            None
        }
    }

    /// `(i+j, i−j)`. Only pairs with `u ≡ v (mod 2)` are real squares.
    pub fn diagonal_coords(i: i64, j: i64) -> (i64, i64) {
        (i + j, i - j)
    }

    pub fn diagonal_coding_axis(u: i64, v: i64) -> Option<CodeAxis> {
        let within = (u.rem_euclid(CELL), v.rem_euclid(CELL));
        if within == U_SITE {
            Some(CodeAxis::X)
        } else if within == V_SITE {
            Some(CodeAxis::Y)
        } else {
            None
        }
    }

    /// True at a coding site whose bit is 0.
    pub fn square_inverted(&self, i: i64, j: i64) -> bool {
        match self.layout {
            CodeLayout::Squares => match Self::coding_axis(i, j) {
                Some(CodeAxis::X) => self.code_bit(i.div_euclid(CELL)) == 0,
                Some(CodeAxis::Y) => self.code_bit(j.div_euclid(CELL)) == 0,
                None => false,
            },
            CodeLayout::Diamonds => {
                let (u, v) = Self::diagonal_coords(i, j);
                match Self::diagonal_coding_axis(u, v) {
                    Some(CodeAxis::X) => self.code_bit(u.div_euclid(CELL)) == 0,
                    Some(CodeAxis::Y) => self.code_bit(v.div_euclid(CELL)) == 0,
                    None => false,
                }
            }
        }
    }

    /// Uncoded colour: white when `i + j` is even.
    pub fn parity_is_white(i: i64, j: i64) -> bool {
        (i + j).rem_euclid(2) == 0
    }

    pub fn square_is_white(&self, i: i64, j: i64) -> bool {
        Self::parity_is_white(i, j) != self.square_inverted(i, j)
    }

    pub fn square_at(&self, x: Real, y: Real) -> (i64, i64) {
        let (x, y) = self.layout.to_lattice(x, y);
        (
            (x / self.square_px).floor() as i64,
            (y / self.square_px).floor() as i64,
        )
    }

    pub fn intensity_at(&self, x: Real, y: Real) -> Real {
        let (i, j) = self.square_at(x, y);
        if self.square_is_white(i, j) { 1.0 } else { 0.0 }
    }

    pub fn plain_intensity_at(&self, x: Real, y: Real) -> Real {
        let (i, j) = self.square_at(x, y);
        if Self::parity_is_white(i, j) {
            1.0
        } else {
            0.0
        }
    }

    /// `π(i+j)` at a square centre.
    pub fn phase1_at(&self, x: Real, y: Real) -> Real {
        let (x, y) = self.layout.to_lattice(x, y);
        PI * ((x + y) / self.square_px - 1.0)
    }

    /// `π(i−j)` at a square centre.
    pub fn phase2_at(&self, x: Real, y: Real) -> Real {
        let (x, y) = self.layout.to_lattice(x, y);
        PI * (x - y) / self.square_px
    }

    pub fn square_from_phases(phase1: Real, phase2: Real) -> (i64, i64) {
        let sum = phase1 / PI;
        let difference = phase2 / PI;
        (
            ((sum + difference) * 0.5).round() as i64,
            ((sum - difference) * 0.5).round() as i64,
        )
    }

    pub fn render(&self, width: usize, height: usize, pose: &PatternPose) -> GrayImage {
        self.render_field(width, height, pose, false)
    }

    pub fn render_plain(&self, width: usize, height: usize, pose: &PatternPose) -> GrayImage {
        self.render_field(width, height, pose, true)
    }

    fn render_field(
        &self,
        width: usize,
        height: usize,
        pose: &PatternPose,
        plain: bool,
    ) -> GrayImage {
        let center_x = width as Real / 2.0;
        let center_y = height as Real / 2.0;
        let n = self.supersample as Real;
        let step = 1.0 / n;
        let first = 0.5 * step - 0.5;

        render_with(width, height, |px, py| {
            let mut sum = 0.0;
            for sub_y in 0..self.supersample {
                for sub_x in 0..self.supersample {
                    let sx = px + first + sub_x as Real * step;
                    let sy = py + first + sub_y as Real * step;
                    let (xp, yp) = into_pattern_frame(sx, sy, center_x, center_y, pose.theta);
                    let (x, y) = (xp - pose.x, yp - pose.y);
                    sum += if plain {
                        self.plain_intensity_at(x, y)
                    } else {
                        self.intensity_at(x, y)
                    };
                }
            }
            sum / (n * n)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phases_index_the_squares() {
        let c = Checkerboard::new(7.0, 6).unwrap();
        for i in -10..10i64 {
            for j in -10..10i64 {
                let (x, y) = ((i as Real + 0.5) * 7.0, (j as Real + 0.5) * 7.0);
                assert_eq!(Checkerboard::square_from_phases(c.phase1_at(x, y), c.phase2_at(x, y)), (i, j));
            }
        }
    }

    #[test]
    fn code_does_not_shift_the_carrier_phase() {
        for layout in [CodeLayout::Squares, CodeLayout::Diamonds] {
            let c = Checkerboard::new(8.0, 8).unwrap().with_code_layout(layout);
            let n = c.range_squares();
            let mut im = 0.0;
            for i in 0..n {
                for j in 0..n {
                    let sign = if c.square_is_white(i, j) { 1.0 } else { -1.0 };
                    im += sign * (PI * (i + j) as Real).sin();
                }
            }
            assert!(im.abs() < 1e-6, "{layout:?}: {im}");
        }
    }

    #[test]
    fn half_the_image_is_white() {
        for layout in [CodeLayout::Squares, CodeLayout::Diamonds] {
            let c = Checkerboard::new(9.0, 8).unwrap().with_code_layout(layout);
            let img = c.render(512, 512, &PatternPose::new(13.7, -4.1, 0.37));
            let mean = img.as_slice().iter().map(|&v| v as Real).sum::<Real>() / img.as_slice().len() as Real;
            assert!((mean - 0.5).abs() < 0.02, "{layout:?}: {mean}");
        }
    }

    #[test]
    fn diamonds_are_turned_squares() {
        let squares = Checkerboard::new(8.0, 8).unwrap();
        let diamonds = squares.clone().with_code_layout(CodeLayout::Diamonds);
        let (s, c) = (PI / 4.0).sin_cos();
        for (x, y) in [(3.0, 5.0), (-20.5, 7.25), (41.0, -33.0)] {
            assert_eq!(diamonds.square_at(x, y), squares.square_at(c * x + s * y, c * y - s * x));
        }
    }
}
