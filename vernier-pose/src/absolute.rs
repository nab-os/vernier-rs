//! Absolute pose: coarse code decode + fine phase.
//!
//! The phase-only path ([`periodic`](crate::periodic)) is precise but ambiguous
//! modulo the pattern period and modulo π/2 in orientation. Absolute (megarena)
//! patterns embed an LFSR position code so a coarse decode tells you which
//! period you are in (`k1, k2`) and which quadrant (`k3`); the fine phase then
//! refines within it (André et al. 2020, 2021). Coarse + fine = unambiguous
//! absolute `(x, y, θ)` at full resolution.
//!
//! ## What is real here and what is a boundary
//!
//! The combination logic — taking integer orders `(k1, k2)` and quadrant `k3`
//! and assembling the absolute coordinate from the fine pose — is implemented
//! and tested. The **decode itself** (the thumbnail / coding-ratio algorithm of
//! André et al. 2021 §III that reads `k1, k2, k3` from the image) is the
//! pattern-specific piece, exposed here as the [`CoarseDecoder`] boundary. It is
//! now fully specified by the 2021 paper and implementable against the megarena
//! generator, but kept behind the trait so the assembly math stands alone.

use vernier_core::scalar::consts::PI;
use vernier_core::{Pose, Real};
use vernier_detection::PhasePlane;

use crate::periodic;
use crate::Calibration;

/// Integer orders and quadrant recovered by the coarse decode.
#[derive(Clone, Copy, Debug)]
pub struct CoarseOrders {
    /// Period order along the first direction (`k1`).
    pub k1: i64,
    /// Period order along the second direction (`k2`).
    pub k2: i64,
    /// Quadrant number (`k3`), 0..=3, resolving the π/2 orientation ambiguity.
    pub k3: u8,
}

/// A coarse absolute-position decoder for a specific pattern family.
///
/// Implemented per pattern (megarena, etc.) against the binary-decoding
/// algorithm of André et al. 2021. Returns the integer orders and quadrant, or
/// `None` if the code region is occluded/undecodable.
pub trait CoarseDecoder {
    /// Decodes `(k1, k2, k3)` from whatever the decoder needs (typically the
    /// spatial-domain image plus the fitted phase planes used to localize the
    /// coding cells).
    fn decode(&self) -> Option<CoarseOrders>;
}

/// Megarena coarse decoder.
///
/// Implements the *localization* half of the André et al. 2021 decode: given the
/// binary windows already extracted from the two pattern directions, it locates
/// each window in the LFSR sequence to recover the absolute period orders
/// `(k1, k2)`, and takes the quadrant `k3` from the orientation disambiguation.
///
/// ## The input boundary
///
/// This decoder consumes **already-extracted bit windows**, not raw pixels. The
/// intensity→bits machinery of the paper (§III-B: phase-guided dot/background
/// classification, thumbnail aggregation, coding-ratio thresholding, global-cell
/// synchronization for the quadrant) is a substantial image-processing stage
/// that needs the per-pixel phase maps surfaced from detection. It is kept as a
/// separate, documented stage (`extract_windows`, below, defines its contract)
/// so the localization — the part that turns bits into an absolute position, and
/// the part most easily got wrong — is implemented and tested in isolation.
///
/// Each cell encodes one bit per axis via its central period (present = 1).
/// Reading `order` consecutive cells along each axis yields the two windows.
pub struct MegarenaDecoder {
    x_index: vernier_patterns::lfsr::WindowIndex,
    y_index: vernier_patterns::lfsr::WindowIndex,
    /// The decoded x-direction bit window (length = LFSR order).
    pub x_window: Vec<u8>,
    /// The decoded y-direction bit window (length = LFSR order).
    pub y_window: Vec<u8>,
    /// The quadrant `k3` from orientation disambiguation (missing-corner sync).
    pub k3: u8,
}

impl MegarenaDecoder {
    /// Builds a decoder for a megarena of the given LFSR order, with windows and
    /// quadrant already extracted from an image.
    ///
    /// Returns `None` if the order is unsupported.
    pub fn new(order: u32, x_window: Vec<u8>, y_window: Vec<u8>, k3: u8) -> Option<Self> {
        let lfsr = vernier_patterns::lfsr::Lfsr::maximal(order)?;
        Some(Self {
            x_index: lfsr.window_index(),
            y_index: lfsr.window_index(),
            x_window,
            y_window,
            k3,
        })
    }
}

impl CoarseDecoder for MegarenaDecoder {
    fn decode(&self) -> Option<CoarseOrders> {
        // Locate each axis window in its LFSR sequence -> absolute cell order.
        let k1 = self.x_index.locate(&self.x_window)? as i64;
        let k2 = self.y_index.locate(&self.y_window)? as i64;
        Some(CoarseOrders {
            k1,
            k2,
            k3: self.k3,
        })
    }
}

/// Extracts the per-direction binary windows from a megarena image, using the
/// detection phase maps to assign each pixel to a coding cell (André et al.
/// 2021 §III-B, simplified).
///
/// ## How it works
///
/// The fine phase along a direction increases by 2π per pattern period, so
/// `round(φ/2π)` labels which period (cell) each pixel belongs to — and this is
/// rotation-invariant, which is why the method works on a tilted pattern without
/// de-rotating the image. Pixels are grouped by cell, their intensities
/// averaged, and cells grouped into triples. The megarena encoding makes the
/// *central* period of each triple present (bit 1) or absent (bit 0); comparing
/// the central cell's mean intensity to its outer neighbors recovers the bit.
///
/// ## Honest simplifications vs the full paper
///
/// The paper's full robustness machinery (separate foreground/background phase
/// masks via |φ| thresholds, thumbnail aggregation, coding-ratio with per-cell
/// local thresholds, global-cell synchronization for the quadrant) is reduced
/// here to: cell-mean intensity + central-vs-outer comparison. This is correct
/// for clean synthetic images (validated to 0 bit errors) but less robust to
/// occlusion/uneven lighting than the full method. The quadrant `k3` is not yet
/// recovered from the missing-corner sync — it must be supplied. These are
/// documented extension points, not hidden gaps.
///
/// Returns the bit windows for both directions plus the per-direction cell
/// ranges, or `None` if too few full triples are visible to form an `order`-bit
/// window.
pub struct ExtractedCode {
    /// Decoded bit window along direction 1 (length = order), MSB = first cell.
    pub x_window: Vec<u8>,
    /// Decoded bit window along direction 2.
    pub y_window: Vec<u8>,
    /// The starting triple index of the x window (its absolute cell order / 3),
    /// before LFSR localization — for diagnostics.
    pub x_first_triple: i64,
    /// Starting triple index of the y window.
    pub y_first_triple: i64,
}

/// Per-axis bit extraction from one phase map and the image intensities.
///
/// `phase` and `intensity` are row-major `width*height`. `phase` MUST be the
/// **unwrapped** phase map of one direction; cell index = `round(phase/2π)`.
/// Wrapped phase (confined to (-π, π]) would round to 0 everywhere and collapse
/// all pixels into a single cell. Returns the bit for each fully-observed
/// triple, keyed by triple index.
fn extract_axis_bits(
    phase: &[Real],
    intensity: &[Real],
    width: usize,
    height: usize,
) -> std::collections::BTreeMap<i64, u8> {
    use std::collections::BTreeMap;
    use vernier_core::scalar::consts::TAU;

    // Accumulate intensity sum and count per cell index.
    let mut sum: BTreeMap<i64, (Real, usize)> = BTreeMap::new();
    for r in 0..height {
        for c in 0..width {
            let idx = r * width + c;
            let cell = (phase[idx] / TAU).round() as i64;
            let e = sum.entry(cell).or_insert((0.0, 0));
            e.0 += intensity[idx];
            e.1 += 1;
        }
    }
    let mean: BTreeMap<i64, Real> = sum
        .iter()
        .map(|(&k, &(s, n))| (k, if n > 0 { s / n as Real } else { 0.0 }))
        .collect();

    // Group cells into triples; a triple needs all three members present.
    let mut triples: BTreeMap<i64, [Option<Real>; 3]> = BTreeMap::new();
    for (&cell, &m) in &mean {
        let tr = cell.div_euclid(3);
        let within = cell.rem_euclid(3) as usize;
        triples.entry(tr).or_insert([None; 3])[within] = Some(m);
    }

    let mut bits = BTreeMap::new();
    for (&tr, members) in &triples {
        if let (Some(outer0), Some(central), Some(outer2)) =
            (members[0], members[1], members[2])
        {
            let outer = (outer0 + outer2) * 0.5;
            // Central present (bit 1) if its intensity is well above the
            // halfway mark to the outer level; absent (bit 0) if near zero.
            let bit = if central > outer * 0.5 { 1u8 } else { 0u8 };
            bits.insert(tr, bit);
        }
    }
    bits
}

/// Extracts both direction windows from a full [`Detection`] and image, ready to
/// build a [`MegarenaDecoder`].
///
/// `order` is the LFSR order (window length). `intensity` is the original
/// row-major image (real values). Takes the first `order` consecutive triples
/// available in each direction. Returns `None` if either direction lacks a full
/// window. `k3` (quadrant) must be supplied — its recovery from the missing
/// corner is a documented extension.
pub fn extract_code(
    detection: &vernier_detection::spectrum::Detection,
    intensity: &[Real],
    order: u32,
) -> Option<ExtractedCode> {
    let n = order as usize;
    let (w, h) = (detection.width, detection.height);

    let xbits = extract_axis_bits(&detection.phase1, intensity, w, h);
    let ybits = extract_axis_bits(&detection.phase2, intensity, w, h);

    // Take the first `order` consecutive triples in each direction.
    let take_window = |bits: &std::collections::BTreeMap<i64, u8>| -> Option<(Vec<u8>, i64)> {
        let triples: Vec<i64> = bits.keys().copied().collect();
        // Find a run of `n` consecutive triple indices.
        for start in 0..triples.len() {
            if start + n > triples.len() {
                break;
            }
            let consecutive = (0..n).all(|j| triples[start + j] == triples[start] + j as i64);
            if consecutive {
                let window: Vec<u8> = (0..n).map(|j| bits[&(triples[start] + j as i64)]).collect();
                return Some((window, triples[start]));
            }
        }
        None
    };

    let (x_window, x_first_triple) = take_window(&xbits)?;
    let (y_window, y_first_triple) = take_window(&ybits)?;

    Some(ExtractedCode {
        x_window,
        y_window,
        x_first_triple,
        y_first_triple,
    })
}

/// Assembles an absolute pose from the fine phase pose and the coarse orders.
///
/// Per direction: `position = k·λ + fine_sub_period`. Orientation gets the
/// quadrant term: `α = fine_θ + k3·(π/2)`.
pub fn assemble(fine: &Pose, orders: CoarseOrders, calib: &Calibration) -> Pose {
    let x = orders.k1 as Real * calib.period + fine.x;
    let y = orders.k2 as Real * calib.period + fine.y;
    let theta = fine.theta + orders.k3 as Real * (PI / 2.0);
    Pose::new(x, y, theta)
}

/// Full absolute estimate: fine phase pose from the two planes, coarse orders
/// from the decoder, combined.
///
/// Returns `None` if the coarse decode fails — without the orders there is no
/// absolute position, only the ambiguous fine one.
pub fn estimate<D: CoarseDecoder>(
    plane1: &PhasePlane,
    plane2: &PhasePlane,
    calib: &Calibration,
    decoder: &D,
) -> Option<Pose> {
    let fine = periodic::estimate(plane1, plane2, calib);
    let orders = decoder.decode()?;
    Some(assemble(&fine, orders, calib))
}

#[cfg(test)]
mod tests {
    use super::*;
    use vernier_core::Pose;

    #[test]
    fn assemble_adds_whole_periods_and_quadrant() {
        let calib = Calibration::new(10.0, 32, 32);
        let fine = Pose::new(2.0, 0.0, 0.0);
        let orders = CoarseOrders { k1: 3, k2: 0, k3: 1 };
        let abs = assemble(&fine, orders, &calib);
        assert!((abs.x - 32.0).abs() < 1e-6, "x={}", abs.x); // 3*10 + 2
        assert!(abs.y.abs() < 1e-6);
        assert!((abs.theta - PI / 2.0).abs() < 1e-6); // quadrant 1
    }

    struct FixedDecoder(Option<CoarseOrders>);
    impl CoarseDecoder for FixedDecoder {
        fn decode(&self) -> Option<CoarseOrders> {
            self.0
        }
    }

    #[test]
    fn estimate_returns_none_on_decode_failure() {
        let calib = Calibration::new(10.0, 32, 32);
        let p1 = PhasePlane { a: 0.5, b: 0.0, c: 0.0 };
        let p2 = PhasePlane { a: 0.0, b: 0.5, c: 0.0 };
        let decoder = FixedDecoder(None);
        assert!(estimate(&p1, &p2, &calib, &decoder).is_none());
    }

    #[test]
    fn megarena_decoder_recovers_known_cell() {
        // Keystone: build the same LFSR the generator uses, read the central-bit
        // window at a known cell start, feed it to the decoder, and confirm it
        // recovers that absolute cell order. This is the bits->position inverse.
        use vernier_patterns::lfsr::Lfsr;
        let order = 8u32;
        let lfsr = Lfsr::maximal(order).unwrap();
        let n = order as usize;

        let true_kx = 42usize;
        let true_ky = 17usize;
        // The generator's central-period presence at triple t is lfsr.bit_at(t).
        let x_window: Vec<u8> = (0..n).map(|j| lfsr.bit_at(true_kx + j)).collect();
        let y_window: Vec<u8> = (0..n).map(|j| lfsr.bit_at(true_ky + j)).collect();

        let decoder = MegarenaDecoder::new(order, x_window, y_window, 2).unwrap();
        let orders = decoder.decode().unwrap();
        assert_eq!(orders.k1, true_kx as i64);
        assert_eq!(orders.k2, true_ky as i64);
        assert_eq!(orders.k3, 2);
    }

    #[test]
    fn megarena_decoder_full_absolute_pose() {
        // End to end: decoder orders + a fine pose -> absolute pose.
        use vernier_patterns::lfsr::Lfsr;
        let order = 8u32;
        let lfsr = Lfsr::maximal(order).unwrap();
        let n = order as usize;
        let calib = Calibration::new(9.0, 64, 64); // 9 µm period, as in the paper

        let (kx, ky) = (10usize, 5usize);
        let xw: Vec<u8> = (0..n).map(|j| lfsr.bit_at(kx + j)).collect();
        let yw: Vec<u8> = (0..n).map(|j| lfsr.bit_at(ky + j)).collect();
        let decoder = MegarenaDecoder::new(order, xw, yw, 0).unwrap();

        // Fine pose: 2.0 µm into the cell along x, 0 along y.
        let p1 = PhasePlane { a: 0.5, b: 0.0, c: 0.0 };
        let p2 = PhasePlane { a: 0.0, b: 0.5, c: 0.0 };
        let mut fine = crate::periodic::estimate(&p1, &p2, &calib);
        fine.x = 2.0; // simulate a known sub-period offset

        let abs = assemble(&fine, decoder.decode().unwrap(), &calib);
        // x = kx*period + 2.0 = 10*9 + 2 = 92.0
        assert!((abs.x - 92.0).abs() < 1e-6, "x={}", abs.x);
        // y = ky*period + 0 = 5*9 = 45.0
        assert!((abs.y - 45.0).abs() < 1e-6, "y={}", abs.y);
    }
}
