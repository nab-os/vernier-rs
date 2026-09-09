//! Absolute decode for the coded checkerboard
//! ([`vernier_patterns::checkerboard`]).
//!
//! The pattern's fine pose comes from the two diagonal carriers like any
//! periodic pattern; this module recovers the missing integer part — *which*
//! square of a 12 285-square-wide board the image centre is looking at.
//!
//! # The chain
//!
//! 1. **Sample.** The fitted phase planes map every pixel to a point on the
//!    square lattice: with `s = φ₁/π` and `d = φ₂/π`, the continuous square
//!    coordinates are `î = (s+d)/2`, `ĵ = (s−d)/2` — and `x = a(î + ½)`,
//!    `y = a(ĵ + ½)`. Pixels near a lattice point (both residuals small) are
//!    pooled into that square's mean intensity.
//! 2. **Binarize.** The pattern is 50/50 white by construction, so the mean of
//!    all square samples is already a good threshold; one k-means refinement
//!    puts it midway between the two populations.
//! 3. **Parity-error map.** A square is a *defect* when its colour disagrees
//!    with `(i+j) mod 2`. Only coding sites are ever defects, so this map is
//!    ~1/9 dense. The opposite parity hypothesis gives an ~8/9-dense map, which
//!    is how the black/white ambiguity resolves itself — no search needed.
//! 4. **Locate the coding sites.** Averaging the defect map over `(i mod 3,
//!    j mod 3)` lights up exactly two of the nine positions at ~50%, the other
//!    seven at ~0. Whether a site carries the x or the y code is read off its
//!    structure: an x site's defects depend only on `i`, so its per-column means
//!    are 0 or 1 while its per-row means all sit at ~½.
//! 5. **Read and localize.** One bit per supercell per axis; `order` bits locate
//!    the window in the LFSR sequence.
//!
//! # Why the hypotheses must be over-determined
//!
//! A maximal LFSR of order `n` contains *every* non-zero `n`-bit word exactly
//! once. So a window always localizes — including a window read in the wrong
//! orientation, which localizes to the wrong place with full confidence. Single
//! windows can therefore never be used to choose between hypotheses. Two things
//! fix this here:
//!
//! - the relative offset of the two coding sites, `(0,1) − (1,0) ≡ (2,1) mod 3`,
//!   is not invariant under rotation, so exactly one of the four quarter-turns
//!   reproduces it;
//! - every bit beyond the first `order` is a check bit. With `r` spare bits a
//!   surviving wrong hypothesis is rejected with probability `1 − 2⁻ʳ`, and a
//!   typical field of view leaves `r` in the tens.
//!
//! # Handedness has to be fixed before that works
//!
//! Rotations are not the only way the frame can arrive wrong. Both axes carry
//! the *same* LFSR, which makes the pattern exactly invariant under transposing
//! `i` and `j` — swap the two coding sites and their two codes and you get the
//! pattern back. So a mirrored reading of the frame decodes just as
//! convincingly as the true one, into a position reflected about the diagonal,
//! and no amount of check bits will separate them.
//!
//! Nothing in the image is mirrored, though: the mirroring comes from the peak
//! search, which may hand back the two carrier directions in either order. That
//! is recoverable from the carriers themselves. The pattern's own gradients,
//! `∇φ₁ = (π/a)(1, 1)` and `∇φ₂ = (π/a)(1, −1)`, have a negative cross product,
//! and rotating the pattern does not change that sign. Measuring a positive one
//! therefore means the directions came back swapped, and negating `φ₂` restores
//! the pattern's own convention — after which only the four rotations remain.
//!
//! # Why parity is not used as a check
//!
//! It is tempting to demand that the frame offset preserve the checkerboard
//! parity — that `Δi + Δj` be even — since the defect map is only sparse when it
//! does. The check is invalid: the code period `3·(2ⁿ − 1)` is odd, so `Δi` is
//! only known modulo an odd number and its parity depends on which
//! representative the decode happened to return. Position is reported modulo one
//! code period per axis, and the check bits do the discriminating.

use std::collections::BTreeMap;

use vernier_core::scalar::consts::{PI, TAU};
use vernier_core::{Pose, Real};
use vernier_patterns::checkerboard::{
    CELL, Checkerboard, CodeLayout, U_SITE, V_SITE, X_SITE, Y_SITE,
};
use vernier_patterns::lfsr::{Lfsr, WindowIndex};
use vernier_spectral::spectrum::Detection;

use crate::Calibration;
use crate::absolute::{CoarseDecoder, CoarseOrders};

/// Half-width, in square units, of the window around a lattice point whose
/// pixels are pooled into that square's sample. Kept well inside the square
/// (whose cell in `(s, d)` is the diamond `|Δs| + |Δd| ≤ 1`) so edge pixels,
/// which carry the transition, never enter a sample.
const SAMPLE_RADIUS: Real = 0.35;

/// Minimum pixels pooled before a square's mean is trusted.
const MIN_SAMPLES: u64 = 4;

/// Spare bits required beyond `order` on each axis. Three gives a 1-in-8 chance
/// of a wrong hypothesis surviving *per candidate*, and there are only eight.
const MIN_SPARE_BITS: usize = 3;

/// Bits a winning hypothesis may still get wrong. Zero would be the strongest
/// possible evidence but would also let one misread square — a speck of dust on
/// one coding site — fail an otherwise unambiguous decode.
const MAX_BIT_ERRORS: usize = 1;

/// The four index transforms `(i, j) → (m₀i + m₁j, m₂i + m₃j)` left once the
/// carrier handedness is canonical: the quarter-turns. Their coding-site offsets
/// are `(2,1)`, `(1,1)`, `(1,2)`, `(2,2)` — all different, so the offset alone
/// names the quadrant. The mirrored four are excluded upstream by the `φ₂` sign
/// fix, not here; including them would reintroduce an ambiguity the code cannot
/// resolve.
const TRANSFORMS: [[i64; 4]; 4] = [
    [1, 0, 0, 1],   // identity
    [0, -1, 1, 0],  // +90°
    [-1, 0, 0, -1], // 180°
    [0, 1, -1, 0],  // −90°
];

/// What the decode recovered.
#[derive(Clone, Debug)]
pub struct CheckerboardCode {
    /// Index of the winning entry in [`TRANSFORMS`].
    pub transform: usize,
    /// Translation from the (transformed) measured frame to the pattern frame:
    /// `pattern = transform(measured) + delta`.
    pub delta: (i64, i64),
    /// Absolute pattern square index under the image centre.
    pub centre_square: (i64, i64),
    /// Continuous pattern square coordinates `(î, ĵ)` of the image centre —
    /// `centre_square` plus the sub-square part carried by the fine phase.
    pub centre: (Real, Real),
    /// LFSR position of the x-direction window.
    pub k_x: i64,
    /// LFSR position of the y-direction window.
    pub k_y: i64,
    /// The `order` bits read along each direction.
    pub x_window: Vec<u8>,
    /// The y-direction window.
    pub y_window: Vec<u8>,
    /// Spare bits that agreed with the located sequence — the evidence the
    /// hypothesis is the right one.
    pub check_bits: usize,
}

/// Coarse decoder over an already-extracted [`CheckerboardCode`], for the shared
/// [`CoarseDecoder`] plumbing.
pub struct CheckerboardDecoder {
    code: CheckerboardCode,
}

impl CheckerboardDecoder {
    /// Wraps an extracted code.
    pub fn new(code: CheckerboardCode) -> Self {
        Self { code }
    }
}

impl CoarseDecoder for CheckerboardDecoder {
    fn decode(&self) -> Option<CoarseOrders> {
        Some(CoarseOrders {
            k1: self.code.k_x,
            k2: self.code.k_y,
            k3: (self.code.transform % 4) as u8,
        })
    }
}

/// Why a decode could not be produced.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CheckerboardError {
    /// No maximal LFSR is known for the requested order.
    UnsupportedOrder(u32),
    /// Too few squares were sampled — occlusion, defocus, or a field of view
    /// smaller than `3·(order + 3)` squares across.
    NotEnoughSquares,
    /// No index transform reproduced the coding-site geometry, or none of the
    /// surviving candidates passed the spare-bit check.
    NoConsistentHypothesis,
}

impl std::fmt::Display for CheckerboardError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedOrder(n) => write!(f, "unsupported LFSR order {n}"),
            Self::NotEnoughSquares => {
                write!(
                    f,
                    "too few squares sampled to read {MIN_SPARE_BITS} spare bits"
                )
            }
            Self::NoConsistentHypothesis => {
                write!(
                    f,
                    "no orientation hypothesis reproduced the coding geometry"
                )
            }
        }
    }
}

impl std::error::Error for CheckerboardError {}

/// Reads the absolute code out of a completed [`Detection`] plus the spatial
/// intensity image.
pub fn extract_code(
    detection: &Detection,
    intensity: &[f32],
    order: u32,
) -> Result<CheckerboardCode, CheckerboardError> {
    extract_code_with_layout(detection, intensity, order, CodeLayout::LatticeAxes)
}

/// [`extract_code`] for a pattern whose code was written along `layout`.
///
/// The layout is not inferred. It could be — the two put their coding sites in
/// different places, so the wrong one scores badly — but that would spend eight
/// more hypotheses on a property the caller already knows, and a silent
/// misdetection here returns a confident wrong position rather than an error.
pub fn extract_code_with_layout(
    detection: &Detection,
    intensity: &[f32],
    order: u32,
    layout: CodeLayout,
) -> Result<CheckerboardCode, CheckerboardError> {
    let lfsr = Lfsr::maximal(order).ok_or(CheckerboardError::UnsupportedOrder(order))?;
    let index = lfsr.window_index();

    // Canonicalize the carrier handedness before anything reads the frame. The
    // pattern's own (∇φ₁, ∇φ₂) pair has a negative cross product; a positive one
    // means the peak search returned the directions swapped, and the frame is
    // mirrored. Negating φ₂ undoes that, leaving only a rotation to find.
    let plane1 = &detection.dir1.plane;
    let plane2 = &detection.dir2.plane;
    let sign2: Real = if plane1.a * plane2.b - plane1.b * plane2.a > 0.0 {
        -1.0
    } else {
        1.0
    };

    let samples = accumulate_squares(
        &detection.phase1,
        &detection.phase2,
        sign2,
        intensity,
        detection.width,
        detection.height,
    );
    if samples.len() < 9 * (order as usize + MIN_SPARE_BITS) {
        return Err(CheckerboardError::NotEnoughSquares);
    }
    let white = binarize(&samples);

    // The image centre in continuous square coordinates of the measured frame.
    //
    // The squares were keyed off the unwrapped phase *maps*, so the centre phase
    // has to come from the same maps or the two disagree by whole turns — and a
    // 2π error in direction 2 is a whole square of absolute position. The plane
    // fit gives the low-noise value; the map at the centre pixel (the unwrap
    // origin, hence the most trustworthy sample) gives the right multiple of 2π.
    let centre_pixel = (detection.height / 2) * detection.width + detection.width / 2;
    let snap = |fitted: Real, measured: Real| fitted + TAU * ((measured - fitted) / TAU).round();
    let phase1 = snap(plane1.c, detection.phase1[centre_pixel]);
    let phase2 = sign2 * snap(plane2.c, detection.phase2[centre_pixel]);
    let centre_measured = (
        (phase1 / PI + phase2 / PI) * 0.5,
        (phase1 / PI - phase2 / PI) * 0.5,
    );

    let mut best: Option<CheckerboardCode> = None;
    for (transform_index, matrix) in TRANSFORMS.iter().enumerate() {
        let Some(candidate) = try_transform(
            &white,
            centre_measured,
            transform_index,
            matrix,
            layout,
            &lfsr,
            &index,
            order,
        ) else {
            continue;
        };
        if best
            .as_ref()
            .is_none_or(|b| candidate.check_bits > b.check_bits)
        {
            best = Some(candidate);
        }
    }

    best.ok_or(CheckerboardError::NoConsistentHypothesis)
}

/// Pools pixel intensities into the square each one sits in, keyed by the square
/// indices the phase planes point at. Pixels near a square's edge are dropped.
fn accumulate_squares(
    phase1: &[Real],
    phase2: &[Real],
    sign2: Real,
    intensity: &[f32],
    width: usize,
    height: usize,
) -> BTreeMap<(i64, i64), (Real, u64)> {
    let mut pools: BTreeMap<(i64, i64), (Real, u64)> = BTreeMap::new();
    for pixel in 0..width * height {
        let s = phase1[pixel] / PI;
        let d = sign2 * phase2[pixel] / PI;
        let i = ((s + d) * 0.5).round();
        let j = ((s - d) * 0.5).round();
        if (s - (i + j)).abs() >= SAMPLE_RADIUS || (d - (i - j)).abs() >= SAMPLE_RADIUS {
            continue;
        }
        let entry = pools.entry((i as i64, j as i64)).or_insert((0.0, 0));
        entry.0 += intensity[pixel] as Real;
        entry.1 += 1;
    }
    pools
}

/// Thresholds the pooled square means into colours. Starts at the overall mean —
/// which the pattern's 50/50 balance puts between the two populations — then
/// takes one k-means step to centre it.
fn binarize(samples: &BTreeMap<(i64, i64), (Real, u64)>) -> BTreeMap<(i64, i64), bool> {
    let means: Vec<((i64, i64), Real)> = samples
        .iter()
        .filter(|&(_, &(_, count))| count >= MIN_SAMPLES)
        .map(|(&key, &(sum, count))| (key, sum / count as Real))
        .collect();

    let mut threshold = means.iter().map(|&(_, m)| m).sum::<Real>() / means.len().max(1) as Real;
    for _ in 0..4 {
        let (mut low, mut low_n, mut high, mut high_n) = (0.0, 0usize, 0.0, 0usize);
        for &(_, mean) in &means {
            if mean < threshold {
                low += mean;
                low_n += 1;
            } else {
                high += mean;
                high_n += 1;
            }
        }
        if low_n == 0 || high_n == 0 {
            break;
        }
        threshold = 0.5 * (low / low_n as Real + high / high_n as Real);
    }

    means
        .into_iter()
        .map(|(key, mean)| (key, mean >= threshold))
        .collect()
}

/// The coordinates a layout indexes its code in, for a square at `(i, j)`.
///
/// Everything upstream of this — sampling, binarizing, the parity hypothesis,
/// the quarter-turn search — is about the checkerboard itself and is identical
/// for both layouts. Only *which square carries which bit* differs, and that is
/// entirely a question of the frame the residues and supercells are counted in.
fn code_coords(layout: CodeLayout, i: i64, j: i64) -> (i64, i64) {
    match layout {
        CodeLayout::LatticeAxes => (i, j),
        CodeLayout::Diagonals => (i + j, i - j),
    }
}

/// The two coding-site residues a layout places within a supercell.
fn layout_sites(layout: CodeLayout) -> ((i64, i64), (i64, i64)) {
    match layout {
        CodeLayout::LatticeAxes => (X_SITE, Y_SITE),
        CodeLayout::Diagonals => (U_SITE, V_SITE),
    }
}

/// The defect map of one rotated frame: every sampled square, and whether its
/// colour disagrees with the checkerboard parity.
struct DefectMap {
    squares: Vec<((i64, i64), bool)>,
    /// Shift applied to `i` to make the parity hypothesis the sparse one.
    parity_shift: i64,
}

fn try_transform(
    white: &BTreeMap<(i64, i64), bool>,
    centre_measured: (Real, Real),
    transform_index: usize,
    matrix: &[i64; 4],
    layout: CodeLayout,
    lfsr: &Lfsr,
    index: &WindowIndex,
    order: u32,
) -> Option<CheckerboardCode> {
    let map = build_defect_map(white, matrix);

    // A coded checkerboard defects about 1 square in 9. Far off that and this is
    // not one — or the binarization failed — so don't spend 9 hypotheses on it.
    let rate = map.squares.iter().filter(|&&(_, defect)| defect).count() as Real
        / map.squares.len() as Real;
    if !(0.02..0.30).contains(&rate) {
        return None;
    }

    // The frame offset is only known modulo the supercell, so try all nine. Each
    // choice says exactly which squares carry the x code and which the y, and the
    // check bits say whether it was right. Reading the sites this way — rather
    // than looking for whichever positions defect most — survives the stretches
    // of the sequence that are almost all ones, where the coding sites are
    // indistinguishable from ordinary squares by defect rate alone.
    let mut best: Option<CheckerboardCode> = None;
    for delta_i_residue in 0..CELL {
        for delta_j_residue in 0..CELL {
            let candidate = try_origin(
                &map,
                centre_measured,
                transform_index,
                matrix,
                (delta_i_residue, delta_j_residue),
                layout,
                lfsr,
                index,
                order,
            );
            if let Some(candidate) = candidate
                && best
                    .as_ref()
                    .is_none_or(|b| candidate.check_bits > b.check_bits)
            {
                best = Some(candidate);
            }
        }
    }
    best
}

/// Rotates the sampled squares into one candidate frame and marks the squares
/// whose colour disagrees with the checkerboard parity.
///
/// The parity hypothesis is not searched: only coding sites are ever painted
/// against parity, so the right hypothesis defects about 1 square in 9 and the
/// wrong one about 8 in 9. Shifting `i` by one flips between them, and the shift
/// is folded into the frame offset later.
fn build_defect_map(white: &BTreeMap<(i64, i64), bool>, matrix: &[i64; 4]) -> DefectMap {
    let apply = |i: i64, j: i64| (matrix[0] * i + matrix[1] * j, matrix[2] * i + matrix[3] * j);
    let transformed: Vec<((i64, i64), bool)> = white
        .iter()
        .map(|(&(i, j), &is_white)| (apply(i, j), is_white))
        .collect();

    let disagreeing = transformed
        .iter()
        .filter(|&&((i, j), is_white)| is_white != Checkerboard::parity_is_white(i, j))
        .count();
    let parity_shift = i64::from(disagreeing * 2 > transformed.len());

    let squares = transformed
        .iter()
        .map(|&((i, j), is_white)| {
            let square = (i + parity_shift, j);
            (
                square,
                is_white != Checkerboard::parity_is_white(square.0, square.1),
            )
        })
        .collect();

    DefectMap {
        squares,
        parity_shift,
    }
}

/// Tests one frame-offset hypothesis: given `delta_residue = (Δi, Δj) mod 3`,
/// the coding sites sit at known residues, so their bits can simply be read and
/// checked against the sequence.
#[allow(clippy::too_many_arguments)]
fn try_origin(
    map: &DefectMap,
    centre_measured: (Real, Real),
    transform_index: usize,
    matrix: &[i64; 4],
    delta_residue: (i64, i64),
    layout: CodeLayout,
    lfsr: &Lfsr,
    index: &WindowIndex,
    order: u32,
) -> Option<CheckerboardCode> {
    // pattern = ours + delta, so a site at pattern residue r sits at ours r − Δ.
    let site_residue = |pattern: (i64, i64)| {
        (
            (pattern.0 - delta_residue.0).rem_euclid(CELL),
            (pattern.1 - delta_residue.1).rem_euclid(CELL),
        )
    };
    let (site_a, site_b) = layout_sites(layout);
    let a_residue = site_residue(site_a);
    let b_residue = site_residue(site_b);

    let (x_bits, x_first_cell) = read_site_bits(map, layout, a_residue, true)?;
    let (y_bits, y_first_cell) = read_site_bits(map, layout, b_residue, false)?;
    let (k_x, x_checks) = localize(&x_bits, lfsr, index, order)?;
    let (k_y, y_checks) = localize(&y_bits, lfsr, index, order)?;

    // The first read supercell of each axis is LFSR position k, so the pattern
    // index of its coding square is 3k + the site's offset in the supercell.
    // The site's residue is what put it in this hypothesis, so `delta` lands on
    // `delta_residue` mod 3 by construction — nothing to re-check there.
    //
    // These are offsets in the *code* frame, which is `(i, j)` for the lattice
    // layout and `(u, v) = (i+j, i−j)` for the diagonal one.
    let delta_a = (CELL * k_x + site_a.0) - (CELL * x_first_cell + a_residue.0);
    let delta_b = (CELL * k_y + site_b.1) - (CELL * y_first_cell + b_residue.1);

    // Back into `(i, j)`. A real frame offset is an integer translation of the
    // square lattice, so in `(u, v)` it always has `Δu + Δv = 2Δi` — an odd sum
    // names no lattice translation at all, and rejecting it prunes hypotheses
    // that could otherwise score check bits by coincidence.
    let (delta_i, delta_j) = match layout {
        CodeLayout::LatticeAxes => (delta_a, delta_b),
        CodeLayout::Diagonals => {
            if (delta_a + delta_b).rem_euclid(2) != 0 {
                return None;
            }
            ((delta_a + delta_b) / 2, (delta_a - delta_b) / 2)
        }
    };

    let (i, j) = centre_measured;
    let raw_i =
        matrix[0] as Real * i + matrix[1] as Real * j + (map.parity_shift + delta_i) as Real;
    let raw_j = matrix[2] as Real * i + matrix[3] as Real * j + delta_j as Real;

    // Position is known modulo one code period per code axis, and for both
    // layouts that comes to the same wrap in `(i, j)`.
    //
    // For the diagonal layout that is worth spelling out, because wrapping `u`
    // and `v` separately is wrong: `period = 3·(2ⁿ−1)` is odd, so a lone period
    // of `u` flips the `u ≡ v (mod 2)` parity and names no square. Only offsets
    // with `Δu ≡ Δv (mod 2)` are real, so the ambiguity lattice is generated by
    // `(P, P)` and `(P, −P)` — which in `(i, j)` is exactly `(P, 0)` and
    // `(0, P)`, the same lattice the axis layout wraps against.
    let period = (CELL * lfsr.len() as i64) as Real;
    let wrap = |value: Real| value - period * (value / period).floor();
    let (centre_i, centre_j) = (wrap(raw_i), wrap(raw_j));

    Some(CheckerboardCode {
        transform: transform_index,
        delta: (map.parity_shift + delta_i, delta_j),
        centre_square: (centre_i.round() as i64, centre_j.round() as i64),
        centre: (centre_i, centre_j),
        k_x,
        k_y,
        x_window: x_bits[..order as usize].to_vec(),
        y_window: y_bits[..order as usize].to_vec(),
        check_bits: x_checks + y_checks,
    })
}

/// Majority-votes one bit per supercell from the squares at `residue`, and
/// returns the longest run of consecutive supercells with the index of its first
/// cell. `by_i` selects which axis indexes the supercells: the x code runs along
/// `i`, the y code along `j`.
fn read_site_bits(
    map: &DefectMap,
    layout: CodeLayout,
    residue: (i64, i64),
    by_i: bool,
) -> Option<(Vec<u8>, i64)> {
    let mut votes: BTreeMap<i64, (usize, usize)> = BTreeMap::new();
    for &((i, j), defect) in &map.squares {
        let (first, second) = code_coords(layout, i, j);
        if (first.rem_euclid(CELL), second.rem_euclid(CELL)) != residue {
            continue;
        }
        let cell = if by_i { first } else { second }.div_euclid(CELL);
        let entry = votes.entry(cell).or_insert((0, 0));
        entry.0 += usize::from(defect);
        entry.1 += 1;
    }

    // A defect means the site was painted against parity, which is the `0` bit.
    let bits: Vec<(i64, u8)> = votes
        .into_iter()
        .map(|(cell, (defective, total))| (cell, u8::from(2 * defective <= total)))
        .collect();

    let mut best: (usize, usize) = (0, 0); // (start index, length)
    let mut run_start = 0usize;
    for position in 1..=bits.len() {
        let broken = position == bits.len() || bits[position].0 != bits[position - 1].0 + 1;
        if broken {
            if position - run_start > best.1 {
                best = (run_start, position - run_start);
            }
            run_start = position;
        }
    }
    if best.1 == 0 {
        return None;
    }
    Some((
        bits[best.0..best.0 + best.1]
            .iter()
            .map(|&(_, b)| b)
            .collect(),
        bits[best.0].0,
    ))
}

/// Locates the bit run in the sequence and scores it against every spare bit.
///
/// Every window of `order` bits localizes, so a single anchor cannot be trusted:
/// one misread bit inside it sends the whole run to the wrong place. Each
/// possible anchor is therefore tried and scored over the *whole* run, and the
/// best is kept only if it explains all but [`MAX_BIT_ERRORS`] of the bits.
/// Returns the LFSR position of the run's first bit and the spare-bit count that
/// backs it.
fn localize(bits: &[u8], lfsr: &Lfsr, index: &WindowIndex, order: u32) -> Option<(i64, usize)> {
    let order = order as usize;
    if bits.len() < order + MIN_SPARE_BITS {
        return None;
    }
    let length = lfsr.len();
    let mut best: Option<(i64, usize)> = None; // (position of bit 0, agreements)

    for anchor in 0..=bits.len() - order {
        let Some(found) = index.locate(&bits[anchor..anchor + order]) else {
            continue;
        };
        let first = (found + length - anchor % length) % length;
        let agreements = bits
            .iter()
            .enumerate()
            .filter(|&(offset, &bit)| lfsr.bit_at(first + offset) == bit)
            .count();
        if best.is_none_or(|(_, previous)| agreements > previous) {
            best = Some((first as i64, agreements));
        }
    }

    let (first, agreements) = best?;
    if bits.len() - agreements > MAX_BIT_ERRORS {
        return None;
    }
    Some((first, bits.len() - order))
}

/// Absolute pose from a completed detection: the code fixes which square the
/// image centre sits on, the fine phase places it within that square.
///
/// `square_size` is the physical side of one checkerboard square (the carrier
/// period a detector is calibrated with is `square_size·√2`). Positions come
/// back in the same unit.
pub fn solve_checkerboard(
    detection: &Detection,
    intensity: &[f32],
    square_size: Real,
    order: u32,
) -> Result<(Pose, CheckerboardCode), CheckerboardError> {
    solve_checkerboard_with_layout(
        detection,
        intensity,
        square_size,
        order,
        CodeLayout::LatticeAxes,
    )
}

/// [`solve_checkerboard`] for a pattern whose code was written along `layout`.
///
/// Only the coarse stage depends on the layout. The fine pose comes from the
/// carrier phases, and those are set by the square geometry, which both layouts
/// share — so everything below the `extract_code` call is common.
pub fn solve_checkerboard_with_layout(
    detection: &Detection,
    intensity: &[f32],
    square_size: Real,
    order: u32,
    layout: CodeLayout,
) -> Result<(Pose, CheckerboardCode), CheckerboardError> {
    let code = extract_code_with_layout(detection, intensity, order, layout)?;

    // x = a(î + ½), y = a(ĵ + ½) — the inverse of the pattern's phase definition.
    let x = square_size * (code.centre.0 + 0.5);
    let y = square_size * (code.centre.1 + 0.5);

    // Orientation: the fine plane angle, turned by the quarter-turn the winning
    // transform undid.
    let quadrant = (code.transform % 4) as Real;
    let theta = detection.dir1.plane.orientation() - quadrant * (PI / 2.0);

    let gradient = (detection.dir1.plane.a.powi(2) + detection.dir1.plane.b.powi(2)).sqrt();
    let pixel_size = square_size * core::f64::consts::SQRT_2 * gradient / TAU;

    Ok((Pose::new_2d(x, y, theta, pixel_size), code))
}

/// Convenience wrapper matching the megarena entry point's shape.
pub fn solve(
    detection: &Detection,
    intensity: &[f32],
    calib: &Calibration,
    order: u32,
) -> Result<Pose, CheckerboardError> {
    // `Calibration::period` is the carrier period; the square side is that over √2.
    let square_size = calib.period / core::f64::consts::SQRT_2;
    solve_checkerboard(detection, intensity, square_size, order).map(|(pose, _)| pose)
}
