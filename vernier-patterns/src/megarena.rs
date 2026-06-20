//! Megarena absolute-pattern generation (André et al. 2020 §II, 2021 §II-C).
//!
//! A megarena pattern is a 2D grid of dots whose *presence* encodes an absolute
//! position. It carries two superimposed kinds of information:
//!
//! - **Fine**: the regular periodicity of the dots is a carrier whose phase
//!   gives sub-period position (the part `vernier-detection` measures).
//! - **Absolute**: specific dots are removed in a structured way so a coarse
//!   decode recovers *which* period you are in (the part `vernier-pose::absolute`
//!   recovers).
//!
//! ## The encoding, concretely
//!
//! Bits are encoded along each axis, **3 periods per bit**. Of those three
//! periods, the two outer ones are always present; the **central** period is
//! present for bit `1` and absent for bit `0` (2020 §II). Removing only the
//! central period keeps the carrier strong and — crucially — does not shift its
//! phase, so the absolute code does not corrupt the fine measurement.
//!
//! Bit values come from a maximal [`Lfsr`](crate::lfsr::Lfsr) so every window of
//! `order` bits is unique → absolute position. The all-ones word is implicitly
//! avoided by the LFSR's structure, guaranteeing a `0` (hence a missing period,
//! an embedded clock) in every window.
//!
//! The 2D pattern is the product of the x-code and the y-code: a dot at grid
//! node (col, row) is present iff *both* the column's period and the row's
//! period are present. Finally, one corner dot of the elementary cell is removed
//! to break the π/2 rotation ambiguity so orientation is absolute over 2π.

use vernier_core::scalar::consts::TAU;
use vernier_core::{GrayImage, Real};

use crate::PatternPose;
use crate::lfsr::Lfsr;
use crate::render::{into_pattern_frame, render_with};

/// Parameters of a megarena pattern.
#[derive(Clone, Debug)]
pub struct Megarena {
    /// Spatial period in pixels (one dot period).
    pub period_px: Real,
    /// LFSR order = bits per unique window. Determines absolute range.
    pub order: u32,
    /// The absolute code sequence (shared along both axes).
    code: Lfsr,
}

impl Megarena {
    /// Builds a megarena pattern with the given pixel period and LFSR order.
    ///
    /// Returns `None` if `order` is outside the supported LFSR range (3..=16).
    pub fn new(period_px: Real, order: u32) -> Option<Self> {
        let code = Lfsr::maximal(order)?;
        Some(Self {
            period_px,
            order,
            code,
        })
    }

    /// Decides whether the period at integer index `p` along an axis is present.
    ///
    /// Periods are grouped in threes (one bit each). Index within the triple:
    /// - 0, 2 (outer) → always present (carrier).
    /// - 1 (central)  → present iff the bit for this triple is `1`.
    ///
    /// The triple index `p / 3` selects the code bit via the LFSR.
    fn period_present(&self, p: i64) -> bool {
        // Map possibly-negative period index into a non-negative phase for
        // grouping and code lookup. The pattern is conceptually infinite; we
        // tile the finite LFSR sequence across it.
        let triple = p.div_euclid(3);
        let within = p.rem_euclid(3); // 0, 1, or 2
        if within != 1 {
            return true; // outer periods always present
        }
        // Central period: present iff this triple's code bit is 1.
        let k = triple.rem_euclid(self.code.len() as i64) as usize;
        self.code.bit_at(k) == 1
    }

    /// Renders the megarena pattern at `pose` into a `width × height` image.
    ///
    /// Intensity model: a separable product of two axis carriers, each gated by
    /// the present/absent encoding of its period. A pixel is bright where both
    /// the x carrier and the y carrier are bright *and* both their periods are
    /// present. The π/2-breaking corner removal is applied per elementary cell.
    pub fn render(&self, width: usize, height: usize, pose: &PatternPose) -> GrayImage {
        let cx = width as Real / 2.0;
        let cy = height as Real / 2.0;
        let period = self.period_px;

        render_with(width, height, |px, py| {
            // Into the pattern's own (axis-aligned) frame.
            let (xp, yp) = into_pattern_frame(px, py, cx, cy, pose.theta);

            // Continuous period coordinate along each axis (shifted by pose).
            let ux = (xp - pose.x) / period;
            let uy = (yp - pose.y) / period;

            // Which integer period this pixel falls in, along each axis.
            let pxi = ux.floor() as i64;
            let pyi = uy.floor() as i64;

            // Encoding gate: is the period present on each axis?
            let x_on = self.period_present(pxi);
            let y_on = self.period_present(pyi);

            // π/2 corner removal: in each 3×3 elementary cell, drop the dot at a
            // fixed corner (triple-local (0,0)) so orientation is unambiguous.
            let corner_removed = pxi.rem_euclid(3) == 0 && pyi.rem_euclid(3) == 0;

            if !x_on || !y_on || corner_removed {
                return 0.0; // dark: missing dot / removed corner
            }

            // Present dot: a bright spot whose intensity follows the carrier so
            // the spectral phase is well defined. Cosine carrier mapped to [0,1].
            let carrier_x = 0.5 + 0.5 * (TAU * ux).cos();
            let carrier_y = 0.5 + 0.5 * (TAU * uy).cos();
            carrier_x * carrier_y
        })
    }

    /// The absolute code sequence, for the decoder to match windows against.
    pub fn code(&self) -> &Lfsr {
        &self.code
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_for_valid_order() {
        let m = Megarena::new(20.0, 8).unwrap();
        assert_eq!(m.order, 8);
        assert_eq!(m.code().len(), 255);
    }

    #[test]
    fn rejects_unsupported_order() {
        assert!(Megarena::new(20.0, 2).is_none());
    }

    #[test]
    fn outer_periods_always_present() {
        let m = Megarena::new(20.0, 6).unwrap();
        // Within every triple, indices 0 and 2 are present regardless of code.
        for triple in 0..10 {
            assert!(m.period_present(triple * 3)); // within = 0
            assert!(m.period_present(triple * 3 + 2)); // within = 2
        }
    }

    #[test]
    fn central_period_follows_code() {
        let m = Megarena::new(20.0, 6).unwrap();
        for triple in 0..20i64 {
            let central = triple * 3 + 1;
            let expected = m
                .code()
                .bit_at(triple.rem_euclid(m.code().len() as i64) as usize)
                == 1;
            assert_eq!(m.period_present(central), expected, "triple {triple}");
        }
    }

    #[test]
    fn renders_expected_dimensions_and_range() {
        let m = Megarena::new(16.0, 8).unwrap();
        let img = m.render(64, 64, &PatternPose::IDENTITY);
        assert_eq!(img.width(), 64);
        assert_eq!(img.height(), 64);
        for &v in img.as_slice() {
            assert!((0.0..=1.0).contains(&v), "intensity {v} out of range");
        }
    }
}
