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

use crate::Calibration;
use crate::periodic;

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
        let k1 = self.x_index.locate(&self.x_window)? as i64;
        let k2 = self.y_index.locate(&self.y_window)? as i64;
        Some(CoarseOrders {
            k1,
            k2,
            k3: self.k3,
        })
    }
}

/// The coding-cell orientation inferred from the thumbnail's 3×3 global cell.
///
/// Ports `MegarenaCell::getCodeOrientation` (C++): after folding all white-pool
/// cells into a 3×3 mean (`globalCell`), the orientation is found by matching
/// the best-fit template across 36 candidates (3 coding rows × 3 coding cols ×
/// 4 quadrant placements). The template assigns +1 to always-present cells,
/// −1 to the missing corner, and 0 to the coding row/column.
#[derive(Clone, Copy, Debug)]
pub struct CodingOrientation {
    /// mod-3 residue of x-direction coding cells (C++ `coding1`).
    pub coding1: i64,
    /// mod-3 residue of y-direction coding cells (C++ `coding2`).
    pub coding2: i64,
    /// mod-3 residue of the missing-corner x-class (C++ `missing1`).
    pub missing1: i64,
    /// mod-3 residue of the missing-corner y-class (C++ `missing2`).
    pub missing2: i64,
    /// Winning quadrant index 0..=3 (C++ `quadrant`).
    pub quadrant: u8,
}

/// Decoded bit windows plus the derived quadrant, ready to build a [`MegarenaDecoder`].
pub struct ExtractedCode {
    /// Decoded bit window along direction 1 (length = order), in LFSR-natural
    /// order (reversed relative to the image scan direction when msb1=false).
    pub x_window: Vec<u8>,
    /// Decoded bit window along direction 2 (LFSR-natural order).
    pub y_window: Vec<u8>,
    /// The starting triple index of the x window — for diagnostics.
    pub x_first_triple: i64,
    /// Starting triple index of the y window — for diagnostics.
    pub y_first_triple: i64,
    /// Quadrant derived from the missing-corner MSB rule, 0..=3.
    pub k3: u8,
    /// MSB flag for direction 1: true = forward LFSR (missing corner before coding row).
    pub msb1: bool,
    /// MSB flag for direction 2.
    pub msb2: bool,
    /// LFSR bit index of the image centre along direction 1 (C++ `K_center`).
    pub x_k_center: i64,
    /// LFSR bit index of the image centre along direction 2.
    pub y_k_center: i64,
    /// Correct bitSeq period-shift for direction 1 (what C++ `findCodePosition`
    /// returns for the MSB=1 / forward direction, corrected for coding residue).
    pub x_periodshift: i64,
    /// Correct bitSeq period-shift for direction 2.
    pub y_periodshift: i64,
}

// ─── Internal types ──────────────────────────────────────────────────────────

struct CellPools {
    white: std::collections::BTreeMap<(i64, i64), (Real, u64)>,
    background: std::collections::BTreeMap<(i64, i64), (Real, u64)>,
}

impl CellPools {
    fn white_mean(&self, cell: (i64, i64)) -> Option<Real> {
        self.white
            .get(&cell)
            .filter(|&&(_, count)| count > 0)
            .map(|&(sum, count)| sum / count as Real)
    }
    fn background_mean(&self, cell: (i64, i64)) -> Option<Real> {
        self.background
            .get(&cell)
            .filter(|&&(_, count)| count > 0)
            .map(|&(sum, count)| sum / count as Real)
    }
}

// ─── Pool accumulation ───────────────────────────────────────────────────────

/// Accumulates white-dot and background intensity pools per 2D cell.
///
/// Ports `MegarenaThumbnail::computeThumbnail` (C++).
///
/// Thresholds match C++: white within Chebyshev radius 0.125 of cell center
/// (C++ `|fmod(phase,2π)| ≤ π/4` AND both axes), background beyond radius
/// 0.375 (C++ `|fmod(phase,2π)| ≥ 3π/4` OR either axis).
fn accumulate_cell_pools(
    phase_x: &[Real],
    phase_y: &[Real],
    intensity: &[Real],
    width: usize,
    height: usize,
) -> CellPools {
    use std::collections::BTreeMap;
    use vernier_core::scalar::consts::TAU;

    let white_r: Real = 0.125; // C++: frac ≤ 1/8 of period, both axes
    let bg_r: Real = 0.375; // C++: frac ≥ 3/8 of period, either axis

    let mut white: BTreeMap<(i64, i64), (Real, u64)> = BTreeMap::new();
    let mut background: BTreeMap<(i64, i64), (Real, u64)> = BTreeMap::new();

    for row in 0..height {
        for col in 0..width {
            let flat_index = row * width + col;
            let fx = phase_x[flat_index] / TAU;
            let fy = phase_y[flat_index] / TAU;
            let cell_x = fx.round();
            let cell_y = fy.round();
            let rx = (fx - cell_x).abs();
            let ry = (fy - cell_y).abs();
            let cell = (cell_x as i64, cell_y as i64);
            let value = intensity[flat_index];
            if rx < white_r && ry < white_r {
                let entry = white.entry(cell).or_insert((0.0, 0));
                entry.0 += value;
                entry.1 += 1;
            } else if rx > bg_r || ry > bg_r {
                let entry = background.entry(cell).or_insert((0.0, 0));
                entry.0 += value;
                entry.1 += 1;
            }
        }
    }
    CellPools { white, background }
}

// ─── Orientation detection ───────────────────────────────────────────────────

/// Returns `(missing1, missing2)` from `(coding1, coding2, quadrant)`.
///
/// Ports the missing-corner derivation in `MegarenaCell::getCodeOrientation`.
fn missing_from_coding(coding1: i64, coding2: i64, quadrant: u8) -> (i64, i64) {
    let missing1 = match quadrant {
        0 | 1 => {
            if coding1 == 2 {
                1
            } else {
                2
            }
        }
        _ => {
            if coding1 == 0 {
                1
            } else {
                0
            }
        }
    };
    let missing2 = match quadrant {
        0 | 2 => {
            if coding2 == 2 {
                1
            } else {
                2
            }
        }
        _ => {
            if coding2 == 0 {
                1
            } else {
                0
            }
        }
    };
    (missing1, missing2)
}

/// Computes the C++ thumbnail frame offset for a given detection.
///
/// C++ indexes the 3×3 global cell as `(round(phase/(2π)) + length/2) % 3`
/// rather than `round(phase/(2π)) % 3`. This function returns `(length1/2,
/// length2/2)` so callers can apply the same shift and stay frame-aligned.
fn cpp_frame_offsets(detection: &vernier_detection::spectrum::Detection) -> (i64, i64) {
    use vernier_core::scalar::consts::TAU;
    let (width_f64, height_f64) = (detection.width as f64, detection.height as f64);
    let mag1 = (detection.dir1.plane.a.powi(2) + detection.dir1.plane.b.powi(2)).sqrt() as f64;
    let mag2 = (detection.dir2.plane.a.powi(2) + detection.dir2.plane.b.powi(2)).sqrt() as f64;
    let pixel_period = (TAU as f64 / mag1 + TAU as f64 / mag2) / 2.0;
    let make_odd_len = |dim: f64| -> i64 {
        let mut length = (dim / pixel_period) as i64 + 1;
        if length % 2 == 0 {
            length += 1;
        }
        length
    };
    let len1 = make_odd_len(height_f64);
    let len2 = make_odd_len(width_f64);
    (len1 / 2, len2 / 2)
}

/// Detects the coding-cell orientation via the 3×3 global-cell template match.
///
/// Ports `MegarenaCell::getGlobalCell` + `MegarenaCell::getCodeOrientation`.
///
/// `offset1` and `offset2` are the C++ frame offsets (`length1/2`, `length2/2`)
/// from `cpp_frame_offsets`. The global cell is built with the shifted index
/// `(cx + offset1) % 3` to match C++'s `phaseIteration % 3`. The returned
/// orientation is converted back to the physical frame (cx % 3) so that
/// `decode_axis_bits` can use it directly without further adjustment.
fn detect_coding_orientation(
    pools: &CellPools,
    offset1: i64,
    offset2: i64,
) -> Option<CodingOrientation> {
    let mut sum = [[0.0f64; 3]; 3];
    let mut cnt = [[0u64; 3]; 3];
    for (&(cell_x, cell_y), &(intensity_sum, pixel_count)) in &pools.white {
        if pixel_count > 0 {
            let i = (cell_x + offset1).rem_euclid(3) as usize;
            let j = (cell_y + offset2).rem_euclid(3) as usize;
            sum[i][j] += intensity_sum as f64;
            cnt[i][j] += pixel_count;
        }
    }
    if cnt.iter().flatten().any(|&count| count == 0) {
        return None;
    }
    let global: [[f64; 3]; 3] =
        std::array::from_fn(|i| std::array::from_fn(|j| sum[i][j] / cnt[i][j] as f64));

    let mut best_score = f64::NEG_INFINITY;
    let mut best_nc_sum = f64::NEG_INFINITY;
    let mut best: Option<CodingOrientation> = None;

    // Tolerance for "effectively equal" primary scores: f32-precision phase
    // noise can produce ties where C++ (f64) would not. The secondary key
    // (total non-coding sum) breaks ties correctly because the always-white
    // non-coding region has higher average intensity than regions that include
    // coding or coding-col cells.
    // 5e-4 is comfortably above the f32-noise-induced score error (~1e-5) for
    // near-tie cases, while remaining below any genuine orientation score gap
    // (empirically ≥ 0.001 for well-imaged patterns).
    let eps: f64 = 5e-4;

    for coding1 in 0i64..3 {
        for coding2 in 0i64..3 {
            for quadrant in 0u8..4 {
                let (missing1, missing2) = missing_from_coding(coding1, coding2, quadrant);
                let nc_iter = (0i64..3)
                    .flat_map(|i| (0i64..3).map(move |j| (i, j)))
                    .filter(move |&(i, j)| i != coding1 && j != coding2);
                let nc_sum: f64 = nc_iter
                    .clone()
                    .map(|(i, j)| global[i as usize][j as usize])
                    .sum();
                let score: f64 = nc_iter
                    .map(|(i, j)| {
                        let weight = if i == missing1 && j == missing2 {
                            -1.0
                        } else {
                            1.0
                        };
                        weight * global[i as usize][j as usize]
                    })
                    .sum();
                // Lexicographic (score, nc_sum): prefer strictly better score,
                // or effectively-equal score with higher non-coding total.
                let is_better =
                    score > best_score + eps || (score >= best_score - eps && nc_sum > best_nc_sum);
                if is_better {
                    best_score = score;
                    best_nc_sum = nc_sum;
                    // Convert from shifted C++ frame back to physical (cx%3) frame.
                    let phys = |v: i64, off: i64| (v - off).rem_euclid(3);
                    best = Some(CodingOrientation {
                        coding1: phys(coding1, offset1),
                        coding2: phys(coding2, offset2),
                        missing1: phys(missing1, offset1),
                        missing2: phys(missing2, offset2),
                        quadrant,
                    });
                }
            }
        }
    }
    best
}

// ─── Bit extraction ──────────────────────────────────────────────────────────

/// Decodes the per-triple bit window along one axis.
///
/// Ports `MegarenaAbsoluteDecoding::getCodeSequence` / `MegarenaThumbnail::getCodeSequence`.
///
/// Inner-loop rules (matching C++ exactly):
/// - **Background**: accumulated from ALL perpendicular cells.
/// - **Coding-white**: accumulated only from non-coding perpendicular cells
///   (`b % 3 != perp_coding_residue`) — excludes the perpendicular coding column
///   whose dots may be absent.
/// - **White reference** (adjacent ±1 along the coding axis): also restricted
///   to non-coding perpendicular cells; the missing-corner cell is excluded.
///
/// Bit decision (nearest-reference, threshold-free):
/// ```text
/// |coding − background| < |whiteRef − coding|  →  bit 0 (absent)
/// otherwise                                    →  bit 1 (present)
/// ```
fn decode_axis_bits(
    pools: &CellPools,
    axis_x: bool,
    coding_residue: i64,
    perp_coding_residue: i64,
    axis_missing: i64,
    perp_missing: i64,
) -> std::collections::BTreeMap<i64, u8> {
    use std::collections::{BTreeMap, BTreeSet};

    let mut x_positions: BTreeSet<i64> = BTreeSet::new();
    let mut y_positions: BTreeSet<i64> = BTreeSet::new();
    for &(cell_x, cell_y) in pools.white.keys().chain(pools.background.keys()) {
        x_positions.insert(cell_x);
        y_positions.insert(cell_y);
    }

    let (coding_axis, perp_axis): (&BTreeSet<i64>, &BTreeSet<i64>) =
        if axis_x { (&x_positions, &y_positions) } else { (&y_positions, &x_positions) };

    let cell_at = |axis_pos: i64, perp_pos: i64| -> (i64, i64) { if axis_x { (axis_pos, perp_pos) } else { (perp_pos, axis_pos) } };

    let mut bits = BTreeMap::new();
    for &axis_pos in coding_axis {
        if axis_pos.rem_euclid(3) != coding_residue {
            continue;
        }
        let mut coding_sum = 0.0;
        let mut coding_count = 0u64;
        let mut white_sum = 0.0;
        let mut white_count = 0u64;
        let mut background_sum = 0.0;
        let mut background_count = 0u64;

        for &perp_pos in perp_axis {
            // Background: all perpendicular positions.
            if let Some(mean) = pools.background_mean(cell_at(axis_pos, perp_pos)) {
                background_sum += mean;
                background_count += 1;
            }
            // Coding-white and white-reference: non-coding perp positions only.
            if perp_pos.rem_euclid(3) != perp_coding_residue {
                if let Some(mean) = pools.white_mean(cell_at(axis_pos, perp_pos)) {
                    coding_sum += mean;
                    coding_count += 1;
                }
                for neighbor in [axis_pos - 1, axis_pos + 1] {
                    // C++ MegarenaAbsoluteDecoding applies the missing-corner
                    // exclusion only for sequence 1 (axis_x=true). For sequence 2
                    // (axis_x=false) a C++ operator-precedence bug means
                    // `index2 ± 1 % 3` evaluates as `index2 ± 1` (not
                    // `(index2 ± 1) % 3`), so the missing corner is never
                    // excluded from the y-direction white reference.
                    let include = if axis_x {
                        neighbor.rem_euclid(3) != axis_missing || perp_pos.rem_euclid(3) != perp_missing
                    } else {
                        true
                    };
                    if include {
                        if let Some(mean) = pools.white_mean(cell_at(neighbor, perp_pos)) {
                            white_sum += mean;
                            white_count += 1;
                        }
                    }
                }
            }
        }

        if coding_count == 0 || white_count == 0 || background_count == 0 {
            continue;
        }
        let mean_coding = coding_sum / coding_count as Real;
        let mean_white = white_sum / white_count as Real;
        let mean_back = background_sum / background_count as Real;

        let bit = if (mean_coding - mean_back).abs() < (mean_white - mean_coding).abs() {
            0u8
        } else {
            1u8
        };
        bits.insert(axis_pos.div_euclid(3), bit);
    }
    bits
}

// ─── Public extraction API ───────────────────────────────────────────────────

/// Decodes the full per-triple bit maps for both directions (diagnostic API).
///
/// Returns `(x_bits, y_bits)`, each mapping triple index → bit, for every
/// fully-observed coding triple. Useful for debug visualization. Uses the
/// global-cell orientation detection to find the correct coding residues and
/// missing corner, matching C++ behavior.
pub fn decode_bit_maps(
    detection: &vernier_detection::spectrum::Detection,
    intensity: &[Real],
) -> (
    std::collections::BTreeMap<i64, u8>,
    std::collections::BTreeMap<i64, u8>,
) {
    let (width, height) = (detection.width, detection.height);
    let pools = accumulate_cell_pools(&detection.phase1, &detection.phase2, intensity, width, height);
    let (off1, off2) = cpp_frame_offsets(detection);
    let orient = detect_coding_orientation(&pools, off1, off2).unwrap_or(CodingOrientation {
        coding1: 1,
        coding2: 1,
        missing1: 2,
        missing2: 2,
        quadrant: 0,
    });
    (
        decode_axis_bits(
            &pools,
            true,
            orient.coding1,
            orient.coding2,
            orient.missing1,
            orient.missing2,
        ),
        decode_axis_bits(
            &pools,
            false,
            orient.coding2,
            orient.coding1,
            orient.missing2,
            orient.missing1,
        ),
    )
}

/// Returns the 3×3 global-cell mean intensities and the detected coding
/// orientation for the given image.
///
/// This is a diagnostic function: it runs `accumulate_cell_pools` and
/// `detect_coding_orientation` without proceeding to full code extraction.
/// Returns `None` if any of the 9 global-cell bins is empty.
pub fn detect_orientation(
    detection: &vernier_detection::spectrum::Detection,
    intensity: &[Real],
) -> Option<([[f64; 3]; 3], CodingOrientation)> {
    let (width, height) = (detection.width, detection.height);
    let pools = accumulate_cell_pools(&detection.phase1, &detection.phase2, intensity, width, height);
    let (off1, off2) = cpp_frame_offsets(detection);

    // Build the global cell in the physical (unshifted) frame for display.
    let mut sum = [[0.0f64; 3]; 3];
    let mut cnt = [[0u64; 3]; 3];
    for (&(cell_x, cell_y), &(intensity_sum, pixel_count)) in &pools.white {
        if pixel_count > 0 {
            let i = cell_x.rem_euclid(3) as usize;
            let j = cell_y.rem_euclid(3) as usize;
            sum[i][j] += intensity_sum as f64;
            cnt[i][j] += pixel_count;
        }
    }
    if cnt.iter().flatten().any(|&count| count == 0) {
        return None;
    }
    let global: [[f64; 3]; 3] =
        std::array::from_fn(|i| std::array::from_fn(|j| sum[i][j] / cnt[i][j] as f64));
    let orient = detect_coding_orientation(&pools, off1, off2)?;
    Some((global, orient))
}

pub fn extract_code(
    detection: &vernier_detection::spectrum::Detection,
    intensity: &[Real],
    order: u32,
) -> Option<ExtractedCode> {
    let window_size = order as usize;
    let (width, height) = (detection.width, detection.height);

    let pools = accumulate_cell_pools(&detection.phase1, &detection.phase2, intensity, width, height);
    let (off1, off2) = cpp_frame_offsets(detection);
    let orient = detect_coding_orientation(&pools, off1, off2)?;

    // ─────────────────────────────────────────────────────────────
    // AXIS CONTRACT (CRITICAL FIX)
    // enforce C++ meaning:
    // coding1 → x-axis stream
    // coding2 → y-axis stream
    // ─────────────────────────────────────────────────────────────

    let x_bits = decode_axis_bits(
        &pools,
        true,
        orient.coding1,
        orient.coding2,
        orient.missing1,
        orient.missing2,
    );

    let y_bits = decode_axis_bits(
        &pools,
        false,
        orient.coding2,
        orient.coding1,
        orient.missing2,
        orient.missing1,
    );

    let lfsr = vernier_patterns::lfsr::Lfsr::maximal(order)?;
    let widx = lfsr.window_index();

    let take_window = |bits: &std::collections::BTreeMap<i64, u8>| -> Option<(Vec<u8>, i64)> {
        let triples: Vec<i64> = bits.keys().copied().collect();

        // Collect all starts of consecutive runs of length window_size.
        let mut candidates: Vec<i64> = Vec::new();
        for start in 0..triples.len() {
            if start + window_size > triples.len() {
                break;
            }
            let consecutive = (0..window_size)
                .all(|j| triples[start + j] == triples[start] + j as i64);
            if consecutive {
                candidates.push(triples[start]);
            }
        }

        // Prefer the window whose center (first_triple + (n-1)/2) is closest to
        // triple 0 (the image centre). Image-edge cells are most likely to have
        // insufficient pixels and produce bit errors that create false LFSR matches.
        candidates.sort_by_key(|&t| {
            // Multiply by 2 to keep integer arithmetic with half-integer centres.
            (2 * t + window_size as i64 - 1).unsigned_abs()
        });

        for first_triple in candidates {
            let window: Vec<u8> =
                (0..window_size).map(|j| bits[&(first_triple + j as i64)]).collect();
            // all-ones is a valid LFSR state; widx.locate provides full validation.
            if widx.locate(&window).is_some() {
                return Some((window, first_triple));
            }
        }

        None
    };

    let (mut x_window, x_first_triple) = take_window(&x_bits)?;
    let (mut y_window, y_first_triple) = take_window(&y_bits)?;

    // ─────────────────────────────────────────────────────────────
    // MSB / quadrant (unchanged, matches C++)
    // ─────────────────────────────────────────────────────────────
    let msb1 = (orient.missing1 + 1).rem_euclid(3) == orient.coding1;
    let msb2 = (orient.missing2 + 1).rem_euclid(3) == orient.coding2;

    let k3 = match (msb1, msb2) {
        (true, true) => 0u8,
        (true, false) => 3u8,
        (false, false) => 2u8,
        (false, true) => 1u8,
    };

    // ─────────────────────────────────────────────────────────────
    // K_center and period-shift derivation:
    //
    // C++ bitSeq places LFSR bit k at position 3*(k+order-1)+1 = 3k+3*(n-1)+1.
    // The physical coding residue `orient.coding1` (0/1/2) shifts which bitSeq
    // column aligns with the image-centre triple (T=0), giving:
    //
    //   For msb=true  (forward LFSR):
    //     k_T0      = k1 - first_triple   (LFSR bit at physical triple T=0)
    //     periodshift = 3*k_T0 + 3*(n-1)+1 - coding1
    //
    //   For msb=false (reversed LFSR):
    //     k_T0      = k1 + first_triple + n - 1   (same formula, reversed window)
    //     periodshift = 3*(k_T0 + n)               (reversal sets eff. coding=1)
    //
    // K_center = k_T0 - 1 for msb=true (legacy field; periodshift is preferred).
    //
    // For msb=true (forward LFSR): ps = 3*k_t0 + 3*(n-1) + 1 - coding_phys
    // For msb=false (reversed LFSR): ps = 3*k_t0 + 3*(n-1) + 1 + coding_phys
    // The sign of coding_phys flips because reversing the LFSR reverses the
    // direction of the coding-residue offset within each 3-cell triple.
    let (x_k_center, x_periodshift) = if msb1 {
        let k1 = widx.locate(&x_window)? as i64;
        let k_t0 = k1 - x_first_triple;
        let ps = 3 * k_t0 + (3 * (window_size as i64 - 1) + 1) - orient.coding1;
        (k_t0 - 1, ps)
    } else {
        x_window.reverse();
        let k1 = widx.locate(&x_window)? as i64;
        let k_t0 = k1 + x_first_triple + window_size as i64 - 1;
        let ps = 3 * k_t0 + (3 * (window_size as i64 - 1) + 1) + orient.coding1;
        (k_t0 - 1, ps)
    };

    let (y_k_center, y_periodshift) = if msb2 {
        let k1 = widx.locate(&y_window)? as i64;
        let k_t0 = k1 - y_first_triple;
        let ps = 3 * k_t0 + (3 * (window_size as i64 - 1) + 1) - orient.coding2;
        (k_t0 - 1, ps)
    } else {
        y_window.reverse();
        let k1 = widx.locate(&y_window)? as i64;
        let k_t0 = k1 + y_first_triple + window_size as i64 - 1;
        let ps = 3 * k_t0 + (3 * (window_size as i64 - 1) + 1) + orient.coding2;
        (k_t0 - 1, ps)
    };

    Some(ExtractedCode {
        x_window,
        y_window,
        x_first_triple,
        y_first_triple,
        k3,
        msb1,
        msb2,
        x_k_center,
        y_k_center,
        x_periodshift,
        y_periodshift,
    })
}

// ─── Assembly ────────────────────────────────────────────────────────────────

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

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use vernier_core::Pose;

    #[test]
    fn assemble_adds_whole_periods_and_quadrant() {
        let calib = Calibration::new(10.0, 32, 32);
        let fine = Pose::new(2.0, 0.0, 0.0);
        let orders = CoarseOrders {
            k1: 3,
            k2: 0,
            k3: 1,
        };
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
        let p1 = PhasePlane {
            a: 0.5,
            b: 0.0,
            c: 0.0,
        };
        let p2 = PhasePlane {
            a: 0.0,
            b: 0.5,
            c: 0.0,
        };
        let decoder = FixedDecoder(None);
        assert!(estimate(&p1, &p2, &calib, &decoder).is_none());
    }

    #[test]
    fn megarena_decoder_recovers_known_cell() {
        use vernier_patterns::lfsr::Lfsr;
        let order = 8u32;
        let lfsr = Lfsr::maximal(order).unwrap();
        let window_size = order as usize;

        let true_kx = 42usize;
        let true_ky = 17usize;
        let x_window: Vec<u8> = (0..window_size).map(|j| lfsr.bit_at(true_kx + j)).collect();
        let y_window: Vec<u8> = (0..window_size).map(|j| lfsr.bit_at(true_ky + j)).collect();

        let decoder = MegarenaDecoder::new(order, x_window, y_window, 2).unwrap();
        let orders = decoder.decode().unwrap();
        assert_eq!(orders.k1, true_kx as i64);
        assert_eq!(orders.k2, true_ky as i64);
        assert_eq!(orders.k3, 2);
    }

    #[test]
    fn megarena_decoder_full_absolute_pose() {
        use vernier_patterns::lfsr::Lfsr;
        let order = 8u32;
        let lfsr = Lfsr::maximal(order).unwrap();
        let window_size = order as usize;
        let calib = Calibration::new(9.0, 64, 64);

        let (kx, ky) = (10usize, 5usize);
        let xw: Vec<u8> = (0..window_size).map(|j| lfsr.bit_at(kx + j)).collect();
        let yw: Vec<u8> = (0..window_size).map(|j| lfsr.bit_at(ky + j)).collect();
        let decoder = MegarenaDecoder::new(order, xw, yw, 0).unwrap();

        let p1 = PhasePlane {
            a: 0.5,
            b: 0.0,
            c: 0.0,
        };
        let p2 = PhasePlane {
            a: 0.0,
            b: 0.5,
            c: 0.0,
        };
        let mut fine = crate::periodic::estimate(&p1, &p2, &calib);
        fine.x = 2.0;

        let abs = assemble(&fine, decoder.decode().unwrap(), &calib);
        assert!((abs.x - 92.0).abs() < 1e-6, "x={}", abs.x); // 10*9 + 2
        assert!((abs.y - 45.0).abs() < 1e-6, "y={}", abs.y); // 5*9 + 0
    }

    #[test]
    fn missing_from_coding_matches_cpp_table() {
        assert_eq!(missing_from_coding(0, 0, 0), (2, 2));
        assert_eq!(missing_from_coding(0, 0, 1), (2, 1));
        assert_eq!(missing_from_coding(0, 0, 2), (1, 2));
        assert_eq!(missing_from_coding(0, 0, 3), (1, 1));
        assert_eq!(missing_from_coding(2, 1, 0), (1, 2));
    }
}
