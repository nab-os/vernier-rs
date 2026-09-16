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

/// Same sites for [`CodeLayout::Diagonals`], as `(u mod 3, v mod 3)`.
pub const U_SITE: (i64, i64) = (1, 0);

pub const V_SITE: (i64, i64) = (0, 1);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CodeAxis {
    X,
    Y,
}

/// Which directions the code runs along.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum CodeLayout {
    /// Along the square edges, `(i, j)`.
    #[default]
    LatticeAxes,
    /// Along the diagonals, `(i+j, i−j)`. Rendered at 45° you get diamonds on an
    /// upright code grid. Costs √2 of range, and each code band is one colour.
    Diagonals,
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
            layout: CodeLayout::LatticeAxes,
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
            CodeLayout::LatticeAxes => steps * self.square_px,
            CodeLayout::Diagonals => steps * self.square_px / SQRT_2,
        }
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
            CodeLayout::LatticeAxes => match Self::coding_axis(i, j) {
                Some(CodeAxis::X) => self.code_bit(i.div_euclid(CELL)) == 0,
                Some(CodeAxis::Y) => self.code_bit(j.div_euclid(CELL)) == 0,
                None => false,
            },
            CodeLayout::Diagonals => {
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
        PI * ((x + y) / self.square_px - 1.0)
    }

    /// `π(i−j)` at a square centre.
    pub fn phase2_at(&self, x: Real, y: Real) -> Real {
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

    /// Carrier amplitude summed over square centres, as (re, im).
    fn lattice_carrier(pattern: &Checkerboard, n: i64, axis1: bool, coded: bool) -> (Real, Real) {
        let (mut re, mut im) = (0.0, 0.0);
        for i in 0..n {
            for j in 0..n {
                let white = if coded {
                    pattern.square_is_white(i, j)
                } else {
                    Checkerboard::parity_is_white(i, j)
                };
                let c = if white { 1.0 } else { -1.0 };
                let phase = if axis1 {
                    PI * (i + j) as Real
                } else {
                    PI * (i - j) as Real
                };
                re += c * phase.cos();
                im += -c * phase.sin();
            }
        }
        let count = (n * n) as Real;
        (re / count, im / count)
    }

    #[test]
    fn builds_for_valid_order() {
        let c = Checkerboard::new(8.0, 8).unwrap();
        assert_eq!(c.order, 8);
        assert_eq!(c.code().len(), 255);
        assert_eq!(c.range_squares(), 765);
    }

    #[test]
    fn rejects_unsupported_order() {
        assert!(Checkerboard::new(8.0, 3).is_none());
    }

    #[test]
    fn only_coding_sites_are_ever_inverted() {
        let c = Checkerboard::new(8.0, 6).unwrap();
        for i in -30..30i64 {
            for j in -30..30i64 {
                if Checkerboard::coding_axis(i, j).is_none() {
                    assert!(!c.square_inverted(i, j), "({i},{j}) inverted off-site");
                }
            }
        }
    }

    #[test]
    fn coding_sites_follow_their_axis_bit() {
        let c = Checkerboard::new(8.0, 6).unwrap();
        for k in 0..20i64 {
            for l in 0..20i64 {
                let (xi, xj) = (CELL * k + X_SITE.0, CELL * l + X_SITE.1);
                assert_eq!(
                    c.square_inverted(xi, xj),
                    c.code_bit(k) == 0,
                    "x site of cell ({k},{l})"
                );
                let (yi, yj) = (CELL * k + Y_SITE.0, CELL * l + Y_SITE.1);
                assert_eq!(
                    c.square_inverted(yi, yj),
                    c.code_bit(l) == 0,
                    "y site of cell ({k},{l})"
                );
            }
        }
    }

    #[test]
    fn coding_sites_alternate_colour_so_fill_stays_balanced() {
        let c = Checkerboard::new(8.0, 8).unwrap();
        let n = 3 * 255;
        let mut white = 0i64;
        for i in 0..n {
            for j in 0..n {
                if c.square_is_white(i, j) {
                    white += 1;
                }
            }
        }
        let fill = white as Real / (n * n) as Real;
        assert!((fill - 0.5).abs() < 1e-3, "fill {fill} drifted from 50/50");
    }

    fn assert_design_properties(c: &Checkerboard, label: &str) {
        let n = 3 * 255;

        // 2 of 9 squares are sites.
        let mut sites = 0i64;
        for i in 0..n {
            for j in 0..n {
                let coding = match c.code_layout() {
                    CodeLayout::LatticeAxes => Checkerboard::coding_axis(i, j).is_some(),
                    CodeLayout::Diagonals => {
                        let (u, v) = Checkerboard::diagonal_coords(i, j);
                        Checkerboard::diagonal_coding_axis(u, v).is_some()
                    }
                };
                if coding {
                    sites += 1;
                }
            }
        }
        let density = sites as Real / (n * n) as Real;
        assert!(
            (density - 2.0 / 9.0).abs() < 1e-3,
            "{label}: coding-site density {density} should be 2/9"
        );

        let mut white = 0i64;
        for i in 0..n {
            for j in 0..n {
                if c.square_is_white(i, j) {
                    white += 1;
                }
            }
        }
        // Global only; diagonal bands cancel each other out.
        let fill = white as Real / (n * n) as Real;
        assert!((fill - 0.5).abs() < 2e-3, "{label}: fill {fill} drifted from 50/50");

        // No phase shift.
        for axis1 in [true, false] {
            let (_, coded_im) = lattice_carrier(c, n, axis1, true);
            assert!(
                coded_im.abs() < 1e-9,
                "{label}: code rotated carrier {}, imaginary part {coded_im}",
                if axis1 { 1 } else { 2 }
            );
        }
    }

    #[test]
    fn diagonal_layout_keeps_every_design_property() {
        let diagonal = Checkerboard::new(8.0, 8)
            .unwrap()
            .with_code_layout(CodeLayout::Diagonals);
        assert_design_properties(&diagonal, "diagonals");

        let axes = Checkerboard::new(8.0, 8).unwrap();
        assert_eq!(axes.code_layout(), CodeLayout::LatticeAxes);
        assert_design_properties(&axes, "lattice axes");
    }

    #[test]
    fn diagonal_layout_actually_moves_the_code() {
        let axes = Checkerboard::new(8.0, 8).unwrap();
        let diagonal = axes.clone().with_code_layout(CodeLayout::Diagonals);

        let mut differing = 0;
        for i in 0..90 {
            for j in 0..90 {
                if axes.square_is_white(i, j) != diagonal.square_is_white(i, j) {
                    differing += 1;
                }
            }
        }
        assert!(differing > 0, "the two layouts paint identical squares");
    }

    /// Worst colour bias among the sites sharing one code bit.
    fn worst_band_imbalance(c: &Checkerboard) -> Real {
        let n = 240;
        let mut bands: std::collections::HashMap<(bool, i64), (i64, i64)> =
            std::collections::HashMap::new();
        for i in -n..n {
            for j in -n..n {
                let (u, v) = Checkerboard::diagonal_coords(i, j);
                let (axis, band) = match c.code_layout() {
                    CodeLayout::LatticeAxes => match Checkerboard::coding_axis(i, j) {
                        Some(CodeAxis::X) => (true, i.div_euclid(CELL)),
                        Some(CodeAxis::Y) => (false, j.div_euclid(CELL)),
                        None => continue,
                    },
                    CodeLayout::Diagonals => match Checkerboard::diagonal_coding_axis(u, v) {
                        Some(CodeAxis::X) => (true, u.div_euclid(CELL)),
                        Some(CodeAxis::Y) => (false, v.div_euclid(CELL)),
                        None => continue,
                    },
                };
                let entry = bands.entry((axis, band)).or_insert((0, 0));
                if Checkerboard::parity_is_white(i, j) {
                    entry.0 += 1;
                } else {
                    entry.1 += 1;
                }
            }
        }
        // Skip bands clipped by the window.
        bands
            .values()
            .filter(|(w, b)| w + b >= 40)
            .map(|&(w, b)| ((w - b).abs() as Real) / ((w + b) as Real))
            .fold(0.0, Real::max)
    }

    #[test]
    fn lattice_axis_code_bands_are_colour_balanced() {
        let c = Checkerboard::new(8.0, 8).unwrap();
        let worst = worst_band_imbalance(&c);
        assert!(
            worst < 0.2,
            "a code band is {:.0}% colour-biased, so flipping it shifts the local \
             brightness",
            worst * 100.0
        );
    }

    #[test]
    fn diagonal_code_bands_are_monochrome_by_construction() {
        // Colour is u mod 2, so a line of constant u is one colour. Moving the
        // sites can't fix it.
        let c = Checkerboard::new(8.0, 8)
            .unwrap()
            .with_code_layout(CodeLayout::Diagonals);
        let worst = worst_band_imbalance(&c);
        assert!(
            worst > 0.99,
            "expected fully monochrome code bands, got {:.0}% bias",
            worst * 100.0
        );
    }

    #[test]
    fn diagonal_layout_costs_sqrt2_of_range() {
        let axes = Checkerboard::new(8.0, 8).unwrap();
        let diagonal = axes.clone().with_code_layout(CodeLayout::Diagonals);

        assert_eq!(axes.range_squares(), diagonal.range_squares());
        let ratio = axes.range_px() / diagonal.range_px();
        assert!((ratio - SQRT_2).abs() < 1e-9, "range ratio {ratio} should be sqrt(2)");
    }

    #[test]
    fn code_is_phase_neutral_on_the_lattice() {
        // An imaginary part would bias the fine pose.
        let c = Checkerboard::new(8.0, 8).unwrap();
        let n = 3 * 255;
        for axis1 in [true, false] {
            let (plain_re, plain_im) = lattice_carrier(&c, n, axis1, false);
            let (coded_re, coded_im) = lattice_carrier(&c, n, axis1, true);
            assert!((plain_re - 1.0).abs() < 1e-9, "plain carrier {plain_re}");
            assert!(plain_im.abs() < 1e-9, "plain carrier not real");
            assert!(
                coded_im.abs() < 1e-9,
                "coded carrier has imaginary part {coded_im} — phase bias"
            );
            // ~1/9 of squares flipped.
            assert!(
                (coded_re - 0.78).abs() < 0.02,
                "coded carrier amplitude {coded_re}, expected ≈0.78"
            );
        }
    }

    #[test]
    fn phases_index_the_squares() {
        let c = Checkerboard::new(7.0, 6).unwrap();
        for i in -10..10i64 {
            for j in -10..10i64 {
                let x = (i as Real + 0.5) * c.square_px;
                let y = (j as Real + 0.5) * c.square_px;
                let recovered =
                    Checkerboard::square_from_phases(c.phase1_at(x, y), c.phase2_at(x, y));
                assert_eq!(recovered, (i, j), "square ({i},{j})");
            }
        }
    }

    #[test]
    fn carriers_peak_on_white_squares() {
        let c = Checkerboard::new(7.0, 6).unwrap();
        for i in -6..6i64 {
            for j in -6..6i64 {
                let x = (i as Real + 0.5) * c.square_px;
                let y = (j as Real + 0.5) * c.square_px;
                let expected = if Checkerboard::parity_is_white(i, j) {
                    1.0
                } else {
                    -1.0
                };
                assert!((c.phase1_at(x, y).cos() - expected).abs() < 1e-9);
                assert!((c.phase2_at(x, y).cos() - expected).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn carrier_period_is_the_square_diagonal() {
        let c = Checkerboard::new(10.0, 6).unwrap();
        assert!((c.carrier_period_px() - 10.0 * SQRT_2).abs() < 1e-12);
        let step = c.carrier_period_px() / SQRT_2;
        let before = c.phase1_at(3.0, 4.0);
        let after = c.phase1_at(3.0 + step, 4.0 + step);
        assert!((after - before - 2.0 * PI).abs() < 1e-12);
    }

    #[test]
    fn renders_expected_dimensions_and_range() {
        let c = Checkerboard::new(6.0, 8).unwrap();
        let img = c.render(64, 48, &PatternPose::new(2.5, -1.25, 0.2));
        assert_eq!(img.width(), 64);
        assert_eq!(img.height(), 48);
        for &v in img.as_slice() {
            assert!((0.0..=1.0).contains(&v), "intensity {v} out of range");
        }
    }

    #[test]
    fn rendered_mean_sits_at_half_scale() {
        let c = Checkerboard::new(9.0, 8).unwrap();
        let img = c.render(512, 512, &PatternPose::new(13.7, -4.1, 0.37));
        let mean: Real =
            img.as_slice().iter().map(|&v| v as Real).sum::<Real>() / img.as_slice().len() as Real;
        assert!((mean - 0.5).abs() < 0.02, "mean intensity {mean}");
    }
}
