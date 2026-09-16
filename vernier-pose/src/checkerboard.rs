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
//! 2. **Binarize.** Locally: each square is compared with the midpoint of the
//!    median intensities of the two parity classes in its 7×7 neighbourhood.
//!    The medians ignore the ~1/9 of squares the code inverts, and a local
//!    threshold follows uneven illumination, which a single global one cannot.
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
use vernier_core::buffer::BufferLayout;
use vernier_core::{Complex32, ComputeBackend, Pose, Real};
use vernier_patterns::checkerboard::{
    CELL, Checkerboard, CodeLayout, U_SITE, V_SITE, X_SITE, Y_SITE,
};
use vernier_patterns::lfsr::{Lfsr, WindowIndex};
use vernier_spectral::spectrum::{Detection, analyze_two};

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

/// Fewest bits a winning hypothesis may get wrong, whatever the run length. Zero
/// would be the strongest possible evidence but would also let one misread
/// square — a speck of dust on one coding site — fail an otherwise unambiguous
/// decode. Longer runs are allowed more; see [`allowed_bit_errors`].
const MAX_BIT_ERRORS: usize = 1;

/// Largest acceptable chance that a wrong placement of one code passes the
/// check bits by luck, counted over every anchor tried.
const FALSE_ACCEPT_LIMIT: Real = 1e-4;

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
    /// The detection locked onto the line the code puts at one third of a
    /// carrier, not the carrier itself. Decoding it would at best fail and at
    /// worst return a confident wrong position. [`detect_checkerboard`] retries
    /// such a detection automatically.
    SubharmonicLock,
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
            Self::SubharmonicLock => {
                write!(
                    f,
                    "carrier detection locked onto the code's one-third line; \
                     detect with detect_checkerboard"
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

    // A lock onto the code's one-third line decodes, when it decodes at all, to
    // a confidently wrong position; refuse it rather than report one.
    if subharmonic_ratios(detection, intensity)
        .iter()
        .any(|&r| r > SUBHARMONIC_LIMIT)
    {
        return Err(CheckerboardError::SubharmonicLock);
    }
    let index = lfsr.window_index();

    // Canonicalize the carrier handedness before anything reads the frame. The
    // pattern's own (∇φ₁, ∇φ₂) pair has a negative cross product; a positive one
    // means the peak search returned the directions swapped, and the frame is
    // mirrored. Negating φ₂ undoes that, leaving only a rotation to find.
    let plane1 = &detection.dir1.plane;
    let plane2 = &detection.dir2.plane;
    let sign2 = handedness(detection);

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

/// Radius, in squares, of the neighbourhood that sets a square's local
/// black/white threshold. Seven squares across follows an illumination ramp
/// closely while still holding about 24 squares of each parity.
const THRESHOLD_RADIUS: i64 = 3;

/// Squares of each parity a neighbourhood needs before its medians are trusted.
/// Near the edge of the view the neighbourhood is widened until it has them.
const MIN_PER_PARITY: usize = 6;

/// A square whose neighbourhood contrast is below this fraction of the typical
/// contrast is dropped: its colour is a coin toss (a vignetted corner, a patch
/// blurred flat), and a dropped vote is better than a random one.
const MIN_RELATIVE_CONTRAST: Real = 0.15;

fn median(values: &mut [Real]) -> Real {
    let mid = values.len() / 2;
    let (_, m, _) = values.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
    *m
}

/// Thresholds the pooled square means into colours, *locally*.
///
/// One global threshold fails under uneven illumination: in the dark part of
/// the frame white squares fall below it and read as black. Worse, the squares
/// that vote on one code bit are correlated in position, so the misreads arrive
/// together -- and for the diagonal layout, whose code bands are single-
/// coloured, they invert whole bits.
///
/// So each square is judged against its own neighbourhood, using the
/// checkerboard's structure. Squares of one parity share a colour and the other
/// parity the other colour, except for the coding sites the code inverts --
/// about one square in nine, a minority of each parity class. The *median* of
/// each parity class therefore ignores the code and estimates the local black
/// and white levels directly, and their midpoint is the threshold. It does not
/// need to know which parity is white, and it follows any lighting that varies
/// slowly over a few squares.
///
/// The minimum pool size adapts to the tile size. A fixed floor of
/// `MIN_SAMPLES` pixels discarded almost every square below ~3.5 px tiles,
/// where the sampling window holds only one or two pixel centres; half the
/// typical pool size keeps the floor at `MIN_SAMPLES` for normal tiles and lets
/// small ones through.
fn binarize(samples: &BTreeMap<(i64, i64), (Real, u64)>) -> BTreeMap<(i64, i64), bool> {
    let mut counts: Vec<Real> = samples.values().map(|&(_, n)| n as Real).collect();
    if counts.is_empty() {
        return BTreeMap::new();
    }
    let typical_count = median(&mut counts);
    let min_count = ((typical_count / 2.0).floor() as u64).clamp(1, MIN_SAMPLES);

    let means: BTreeMap<(i64, i64), Real> = samples
        .iter()
        .filter(|&(_, &(_, count))| count >= min_count)
        .map(|(&key, &(sum, count))| (key, sum / count as Real))
        .collect();
    if means.is_empty() {
        return BTreeMap::new();
    }

    // Dense grid over the sampled squares, for neighbourhood lookups.
    let (i_min, i_max) = means.keys().fold((i64::MAX, i64::MIN), |(lo, hi), &(i, _)| (lo.min(i), hi.max(i)));
    let (j_min, j_max) = means.keys().fold((i64::MAX, i64::MIN), |(lo, hi), &(_, j)| (lo.min(j), hi.max(j)));
    let width = (i_max - i_min + 1) as usize;
    let height = (j_max - j_min + 1) as usize;
    let mut grid = vec![Real::NAN; width * height];
    for (&(i, j), &m) in &means {
        grid[(j - j_min) as usize * width + (i - i_min) as usize] = m;
    }

    let mut judged: Vec<((i64, i64), Real, Real, Real)> = Vec::with_capacity(means.len());
    let (mut even, mut odd) = (Vec::new(), Vec::new());
    for (&(i, j), &m) in &means {
        let mut radius = THRESHOLD_RADIUS;
        loop {
            even.clear();
            odd.clear();
            for nj in (j - radius).max(j_min)..=(j + radius).min(j_max) {
                for ni in (i - radius).max(i_min)..=(i + radius).min(i_max) {
                    let v = grid[(nj - j_min) as usize * width + (ni - i_min) as usize];
                    if v.is_nan() {
                        continue;
                    }
                    if (ni + nj).rem_euclid(2) == 0 {
                        even.push(v);
                    } else {
                        odd.push(v);
                    }
                }
            }
            let enough = even.len() >= MIN_PER_PARITY && odd.len() >= MIN_PER_PARITY;
            if enough || radius >= 4 * THRESHOLD_RADIUS {
                break;
            }
            radius *= 2;
        }
        if even.is_empty() || odd.is_empty() {
            continue;
        }
        let (level_even, level_odd) = (median(&mut even), median(&mut odd));
        judged.push(((i, j), 0.5 * (level_even + level_odd), (level_even - level_odd).abs(), m));
    }
    if judged.is_empty() {
        return BTreeMap::new();
    }

    let mut contrasts: Vec<Real> = judged.iter().map(|&(_, _, c, _)| c).collect();
    let typical_contrast = median(&mut contrasts);
    judged
        .into_iter()
        .filter(|&(_, _, contrast, _)| contrast >= MIN_RELATIVE_CONTRAST * typical_contrast)
        .map(|(key, threshold, _, mean)| (key, mean >= threshold))
        .collect()
}

/// Sign applied to `φ₂` to make the carrier frame right-handed; see the module
/// docs on handedness.
fn handedness(detection: &Detection) -> Real {
    let (p1, p2) = (&detection.dir1.plane, &detection.dir2.plane);
    if p1.a * p2.b - p1.b * p2.a > 0.0 { -1.0 } else { 1.0 }
}

/// Largest ratio a true carrier lock reaches between the image's amplitude at
/// three times a carrier's frequency and at the carrier itself. On a true lock
/// that is the pattern's third harmonic, about 1/9 (measured 0.09-0.14 clean,
/// 0.20 under heavy noise); on a lock onto the code's one-third line it is the
/// true carrier, 1.3-3.1 measured even under blur, noise and illumination ramps.
const SUBHARMONIC_LIMIT: Real = 0.5;

fn signed_bin(bin: usize, n: usize) -> i64 {
    if bin > n / 2 { bin as i64 - n as i64 } else { bin as i64 }
}

/// Hann-windowed spectral amplitudes of one image at chosen FFT bins.
///
/// The windowed, mean-removed image is built once; each amplitude is then a
/// separable sum. Because `e^{-i(fx·x + fy·y)}` factors into a column term and
/// a row term, every row is first collapsed against the column phasors and
/// the row sums are combined with the row phasors — two passes of plain
/// multiply-adds per frequency, with no trigonometry per pixel.
struct Demodulator {
    width: usize,
    height: usize,
    xs: Vec<Real>,
    ys: Vec<Real>,
    /// Windowed, mean-removed samples on the subsampled grid, row-major.
    samples: Vec<Real>,
}

impl Demodulator {
    fn new(intensity: &[f32], width: usize, height: usize, step: usize) -> Self {
        let mean = intensity.iter().map(|&v| v as Real).sum::<Real>() / intensity.len() as Real;
        let xs: Vec<usize> = (0..width).step_by(step).collect();
        let ys: Vec<usize> = (0..height).step_by(step).collect();
        let hann = |i: usize, n: usize| 0.5 - 0.5 * (TAU * i as Real / (n - 1) as Real).cos();
        let wx: Vec<Real> = xs.iter().map(|&x| hann(x, width)).collect();
        let mut samples = Vec::with_capacity(xs.len() * ys.len());
        for &y in &ys {
            let wy = hann(y, height);
            for (k, &x) in xs.iter().enumerate() {
                samples.push((intensity[y * width + x] as Real - mean) * wx[k] * wy);
            }
        }
        Self {
            width,
            height,
            xs: xs.into_iter().map(|x| x as Real).collect(),
            ys: ys.into_iter().map(|y| y as Real).collect(),
            samples,
        }
    }

    fn amplitude(&self, bx: i64, by: i64) -> Real {
        let fx = TAU * bx as Real / self.width as Real;
        let fy = TAU * by as Real / self.height as Real;
        let (cx, sx): (Vec<Real>, Vec<Real>) = self.xs.iter().map(|&x| ((fx * x).cos(), (fx * x).sin())).unzip();
        let cols = self.xs.len();
        let (mut re, mut im) = (0.0, 0.0);
        for (r, &y) in self.ys.iter().enumerate() {
            let row = &self.samples[r * cols..(r + 1) * cols];
            let (mut rc, mut rs) = (0.0, 0.0);
            for k in 0..cols {
                rc += row[k] * cx[k];
                rs += row[k] * sx[k];
            }
            let (cy, sy) = ((fy * y).cos(), (fy * y).sin());
            // cos(a+b) = cos a cos b − sin a sin b;  sin(a+b) = sin a cos b + cos a sin b
            re += cy * rc - sy * rs;
            im -= cy * rs + sy * rc;
        }
        (re * re + im * im).sqrt()
    }
}

/// Per-carrier ratio of the image's amplitude at three times the detected peak
/// to its amplitude at the peak itself; see [`SUBHARMONIC_LIMIT`].
///
/// The code the checkerboard carries puts a spectral line at exactly one third
/// of each carrier, in the same direction: the coding sites repeat every three
/// squares, on top of the checkerboard's own alternation. When the bits in
/// view make that line stronger than the carrier's own peak -- or blur, which
/// attenuates the carrier more than the lower line, makes it so -- the peak
/// search locks onto it. Everything downstream then samples squares three times
/// too big and either fails or, worse, decodes to a wrong position with full
/// confidence. On a true lock, three times the frequency is the pattern's third
/// harmonic, about a ninth as strong; on a false one it is the true carrier.
///
/// Three times a peak beyond Nyquist cannot be a sub-harmonic lock (the true
/// carrier would have to be above Nyquist), so such a carrier scores zero.
pub fn subharmonic_ratios(detection: &Detection, intensity: &[f32]) -> [Real; 2] {
    let (w, h) = (detection.width, detection.height);
    [detection.dir1.peak_bin, detection.dir2.peak_bin].map(|(px, py)| {
        let (bx, by) = (signed_bin(px, w), signed_bin(py, h));
        let (tx, ty) = (3 * bx, 3 * by);
        if tx.abs() + 2 >= (w / 2) as i64 || ty.abs() + 2 >= (h / 2) as i64 {
            return 0.0;
        }
        // Subsampling by two is safe while everything stays under half Nyquist.
        let step = if tx.abs() + 2 < (w / 4) as i64 && ty.abs() + 2 < (h / 4) as i64 { 2 } else { 1 };
        let demod = Demodulator::new(intensity, w, h, step);
        let base = demod.amplitude(bx, by);
        let mut third: Real = 0.0;
        // The peak bin is exact to +-0.5 bin, so three times it to +-1.5.
        for dy in -2..=2 {
            for dx in -2..=2 {
                third = third.max(demod.amplitude(tx + dx, ty + dy));
            }
        }
        third / base.max(1e-12)
    })
}

/// Two-carrier detection for the coded checkerboard, guarded against locking
/// onto the code's one-third line.
///
/// Runs [`analyze_two`]; if a carrier looks like a sub-harmonic lock (see
/// [`subharmonic_ratios`]), runs it again with the minimum frequency raised
/// above that false peak -- the true carrier sits at three times its radius --
/// and keeps whichever detection is the better lock. Callers that detect with
/// `analyze_two` directly still get a
/// [`CheckerboardError::SubharmonicLock`] from the decoder rather than a wrong
/// position, but they miss the retry.
pub fn detect_checkerboard<B: ComputeBackend>(
    backend: &B,
    intensity: &[f32],
    layout: BufferLayout,
    sigma: Real,
    min_frequency: usize,
    max_frequency: usize,
    smoothing_sigma: Real,
) -> vernier_core::Result<Detection> {
    let complex: Vec<Complex32> = intensity.iter().map(|&v| Complex32::new(v, 0.0)).collect();
    let first = analyze_two(backend, &complex, layout, sigma, min_frequency, max_frequency, smoothing_sigma)?;
    let ratios = subharmonic_ratios(&first, intensity);
    if ratios.iter().all(|&r| r <= SUBHARMONIC_LIMIT) {
        return Ok(first);
    }

    let (w, h) = (first.width, first.height);
    let false_radius = [first.dir1.peak_bin, first.dir2.peak_bin]
        .iter()
        .zip(ratios)
        .filter(|&(_, r)| r > SUBHARMONIC_LIMIT)
        .map(|(&(px, py), _)| {
            let (bx, by) = (signed_bin(px, w), signed_bin(py, h));
            ((bx * bx + by * by) as Real).sqrt()
        })
        .fold(0.0, Real::max);
    // Between the false line (r) and the true carrier (3r).
    let raised = min_frequency.max((2.0 * false_radius).ceil() as usize);
    let Ok(second) = analyze_two(backend, &complex, layout, sigma, raised, max_frequency, smoothing_sigma)
    else {
        return Ok(first);
    };
    let worst = |d: &Detection| subharmonic_ratios(d, intensity).into_iter().fold(0.0, Real::max);
    if worst(&second) < worst(&first) { Ok(second) } else { Ok(first) }
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
    // square lattice, so in `(u, v)` it always has `Δu + Δv = 2Δi` — even.
    //
    // But `delta_a` and `delta_b` come from LFSR positions, so they are only
    // known modulo one code period `P = 3·(2ⁿ−1)`, and `P` is odd: the true
    // `Δu` may be `delta_a + P`, with the opposite parity. An odd sum therefore
    // does not mean a wrong hypothesis — rejecting it discarded the correct one
    // about half the time, which is what 100 random poses exposed and 8 fixed
    // ones did not. Lift instead: adding `P` to `delta_b` restores the parity,
    // and the two possible lifts differ by exactly `P` in `(i, j)`, which the
    // wrap below removes, so the answer modulo `P` is unique.
    let (delta_i, delta_j) = match layout {
        CodeLayout::LatticeAxes => (delta_a, delta_b),
        CodeLayout::Diagonals => {
            let period = CELL * lfsr.len() as i64;
            let delta_b = if (delta_a + delta_b).rem_euclid(2) != 0 {
                delta_b + period
            } else {
                delta_b
            };
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

/// How many bits of a run of `bits` a placement may get wrong and still pass.
///
/// A fixed allowance of one is right for a short run and far too strict for a
/// long one: a clean image read in the diagonal layout returns ~36 bits per
/// code, and two misread cells at the edge of the view were enough to reject
/// the true placement, while every wrong placement disagreed on 9–12 bits.
///
/// So the allowance is the most errors for which a *wrong* placement — whose
/// check bits agree with the sequence at random — would still pass with a
/// probability under [`FALSE_ACCEPT_LIMIT`], counting every anchor `localize`
/// tries. For 36 bits that is 2; for the ~21 bits of a typical lattice read it
/// stays at [`MAX_BIT_ERRORS`].
fn allowed_bit_errors(bits: usize, order: usize) -> usize {
    let spare = bits.saturating_sub(order);
    let anchors = (spare + 1) as Real;
    let chance = (0.5 as Real).powi(spare as i32);
    let (mut allowed, mut cumulative, mut choose) = (MAX_BIT_ERRORS, 0.0, 1.0);
    for errors in 0..=spare {
        if errors > 0 {
            choose *= (spare - errors + 1) as Real / errors as Real;
        }
        cumulative += choose;
        if cumulative * chance * anchors > FALSE_ACCEPT_LIMIT {
            break;
        }
        allowed = allowed.max(errors);
    }
    allowed
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
    if bits.len() - agreements > allowed_bit_errors(bits.len(), order) {
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

#[cfg(test)]
mod tests {
    use super::*;


    /// The per-pixel sum the separable demodulator replaces.
    fn naive_amplitude(intensity: &[f32], width: usize, height: usize, bx: i64, by: i64, step: usize) -> Real {
        let mean = intensity.iter().map(|&v| v as Real).sum::<Real>() / intensity.len() as Real;
        let (fx, fy) = (TAU * bx as Real / width as Real, TAU * by as Real / height as Real);
        let (mut re, mut im) = (0.0, 0.0);
        for y in (0..height).step_by(step) {
            let wy = 0.5 - 0.5 * (TAU * y as Real / (height - 1) as Real).cos();
            for x in (0..width).step_by(step) {
                let wx = 0.5 - 0.5 * (TAU * x as Real / (width - 1) as Real).cos();
                let v = (intensity[y * width + x] as Real - mean) * wx * wy;
                let phase = fx * x as Real + fy * y as Real;
                re += v * phase.cos();
                im -= v * phase.sin();
            }
        }
        (re * re + im * im).sqrt()
    }

    #[test]
    fn separable_demodulator_matches_the_per_pixel_sum() {
        let (w, h) = (96, 64);
        let image: Vec<f32> = (0..w * h)
            .map(|p| {
                let (x, y) = ((p % w) as f32, (p / w) as f32);
                0.5 + 0.3 * (0.7 * x + 0.2 * y).sin() + 0.1 * (0.05 * x * y).cos()
            })
            .collect();
        for step in [1, 2] {
            let demod = Demodulator::new(&image, w, h, step);
            for (bx, by) in [(0, 0), (5, -3), (-12, 7), (20, 11), (-3, -9)] {
                let fast = demod.amplitude(bx, by);
                let slow = naive_amplitude(&image, w, h, bx, by, step);
                assert!((fast - slow).abs() <= 1e-9 * slow.max(1.0), "bin ({bx},{by}) step {step}: {fast} vs {slow}");
            }
        }
    }

    #[test]
    fn bit_error_allowance_grows_with_run_length_but_stays_safe() {
        assert_eq!(allowed_bit_errors(8, 8), MAX_BIT_ERRORS);
        assert_eq!(allowed_bit_errors(21, 8), MAX_BIT_ERRORS);
        assert_eq!(allowed_bit_errors(36, 8), 2);
        // Never below the old floor, and monotone in the run length.
        let mut previous = 0;
        for bits in 8..80 {
            let allowed = allowed_bit_errors(bits, 8);
            assert!(allowed >= MAX_BIT_ERRORS);
            assert!(allowed >= previous, "allowance fell at {bits} bits");
            previous = allowed;
        }
    }
}
