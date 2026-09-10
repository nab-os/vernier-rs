//! Which code layout survives a worse image: lattice axes, or diagonals?
//!
//! Both layouts are given the *same* pose set and the *same* noise realisation
//! at every level — the degradation is seeded from the condition, not from the
//! layout — so the comparison is paired and a difference cannot be luck.
//!
//! Two numbers are reported per cell. The decode rate is what matters in the
//! end, but it is a cliff: it sits at 100% until it collapses. Check bits are
//! the spare bits a winning hypothesis got right beyond the ones used to
//! localize, so they measure how much margin is left *before* the cliff, and
//! they move first.
//!
//! Run with `cargo run --release --example layout_robustness -p vernier-pose`.

use vernier_core::Complex32;
use vernier_core::buffer::BufferLayout;
use vernier_cpu::CpuBackend;
use vernier_patterns::PatternPose;
use vernier_patterns::checkerboard::{Checkerboard, CodeLayout};
use vernier_pose::checkerboard::extract_code_with_layout;
use vernier_spectral::spectrum::analyze_two;

const SIZE: usize = 512;
const SQUARE: f64 = 8.0;
const ORDER: u32 = 8;
const POSES: usize = 100;

/// Deterministic noise, so a rerun reproduces the table exactly.
struct Lcg(u64);

impl Lcg {
    fn next_u32(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        // The top 32 bits. `>> 33` kept only 31, which pinned `unit()` to
        // [0, 0.5): every pose landed at negative x, y and theta, and the
        // Box-Muller noise was biased.
        (self.0 >> 32) as u32
    }

    fn unit(&mut self) -> f64 {
        (self.next_u32() as f64 + 0.5) / (u32::MAX as f64 + 1.0)
    }

    /// Box-Muller, one sample per call (the second is discarded; this is not
    /// hot enough to care).
    fn normal(&mut self) -> f64 {
        let (u1, u2) = (self.unit().max(1e-12), self.unit());
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Degradation {
    /// Additive sensor noise.
    Noise,
    /// Multiplicative illumination ramp across the diagonal — the case a global
    /// binarization threshold is least able to absorb.
    Gradient,
    /// Defocus.
    Blur,
}

impl Degradation {
    fn label(self) -> &'static str {
        match self {
            Degradation::Noise => "additive noise (sigma)",
            Degradation::Gradient => "illumination ramp (+/- amplitude)",
            Degradation::Blur => "defocus blur (sigma px)",
        }
    }

    fn levels(self) -> &'static [f64] {
        match self {
            Degradation::Noise => &[0.0, 0.15, 0.30, 0.45, 0.60, 0.80],
            Degradation::Gradient => &[0.0, 0.30, 0.60, 0.80, 1.00],
            Degradation::Blur => &[0.0, 1.0, 1.6, 2.0, 2.4],
        }
    }
}

fn blur(image: &mut [f32], sigma: f64) {
    if sigma <= 0.0 {
        return;
    }
    let radius = (3.0 * sigma).ceil() as isize;
    let kernel: Vec<f64> = (-radius..=radius)
        .map(|d| (-(d * d) as f64 / (2.0 * sigma * sigma)).exp())
        .collect();
    let norm: f64 = kernel.iter().sum();

    let mut pass = vec![0.0f32; image.len()];
    // Horizontal, then vertical; edges clamp.
    for row in 0..SIZE {
        for col in 0..SIZE {
            let mut acc = 0.0;
            for (k, weight) in kernel.iter().enumerate() {
                let x = (col as isize + k as isize - radius).clamp(0, SIZE as isize - 1) as usize;
                acc += weight * image[row * SIZE + x] as f64;
            }
            pass[row * SIZE + col] = (acc / norm) as f32;
        }
    }
    for col in 0..SIZE {
        for row in 0..SIZE {
            let mut acc = 0.0;
            for (k, weight) in kernel.iter().enumerate() {
                let y = (row as isize + k as isize - radius).clamp(0, SIZE as isize - 1) as usize;
                acc += weight * pass[y * SIZE + col] as f64;
            }
            image[row * SIZE + col] = (acc / norm) as f32;
        }
    }
}

fn degrade(image: &mut [f32], kind: Degradation, level: f64, seed: u64) {
    match kind {
        Degradation::Noise => {
            let mut rng = Lcg(seed);
            for pixel in image.iter_mut() {
                *pixel = (*pixel as f64 + level * rng.normal()).clamp(0.0, 1.0) as f32;
            }
        }
        Degradation::Gradient => {
            for row in 0..SIZE {
                for col in 0..SIZE {
                    let t = (row + col) as f64 / (2 * SIZE) as f64 - 0.5;
                    let gain = 1.0 + 2.0 * level * t;
                    let p = &mut image[row * SIZE + col];
                    // Clipping is part of the effect: a drifting local mean is
                    // what pushes a real sensor into saturation.
                    *p = ((*p as f64) * gain).clamp(0.0, 1.0) as f32;
                }
            }
        }
        Degradation::Blur => blur(image, level),
    }
}

/// Poses drawn from a fixed-seed generator: translations uniform over +/-2000 px
/// (hundreds of squares, never on a square boundary by construction) and
/// orientations uniform over +/-36 degrees, the same spread as the original
/// 8-pose set. Seeded, so both layouts -- and every rerun -- see the same set.
fn poses() -> Vec<PatternPose> {
    let mut rng = Lcg(0x9e37_79b9_7f4a_7c15);
    (0..POSES)
        .map(|_| {
            let x = (rng.unit() - 0.5) * 4000.0;
            let y = (rng.unit() - 0.5) * 4000.0;
            let theta = (rng.unit() - 0.5) * 2.0 * 0.63;
            PatternPose::new(x, y, theta)
        })
        .collect()
}

/// Worker threads: one per core. Each decode is independent, so the sweep scales
/// almost linearly -- 100 poses per cell is otherwise an hour of wall time.
fn threads() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4)
}


fn expected_square(pose: &PatternPose, square: f64) -> (i64, i64) {
    (
        (-pose.x / square - 0.5).round() as i64,
        (-pose.y / square - 0.5).round() as i64,
    )
}

/// Returns (correct decodes, total, mean check bits over the correct ones).
fn evaluate(
    layout: CodeLayout,
    square: f64,
    kind: Degradation,
    level: f64,
) -> (usize, usize, f64) {
    let pattern = Checkerboard::new(square, ORDER).unwrap().with_code_layout(layout);
    let period = 3 * pattern.code().len() as i64;
    let all = poses();
    let chunk = all.len().div_ceil(threads());

    let per_thread: Vec<(usize, f64)> = std::thread::scope(|scope| {
        let handles: Vec<_> = all
            .chunks(chunk)
            .enumerate()
            .map(|(c, slice)| {
                let pattern = &pattern;
                scope.spawn(move || {
                    let backend = CpuBackend::new();
                    let buffer = BufferLayout::packed(SIZE, SIZE);
                    let (mut correct, mut checks) = (0usize, 0f64);
                    for (offset, pose) in slice.iter().enumerate() {
                        let index = c * chunk + offset;
                        let mut image = pattern.render(SIZE, SIZE, pose).as_slice().to_vec();
                        // Seed from the condition only, never the layout: both
                        // layouts see the same noise.
                        degrade(
                            &mut image,
                            kind,
                            level,
                            0x5eed ^ ((index as u64) << 8) ^ (level.to_bits() >> 40),
                        );

                        let complex: Vec<Complex32> =
                            image.iter().map(|&v| Complex32::new(v, 0.0)).collect();
                        let Ok(detection) =
                            analyze_two(&backend, &complex, buffer, 4.0, 10, 0, 0.0)
                        else {
                            continue;
                        };
                        let Ok(code) = extract_code_with_layout(&detection, &image, ORDER, layout)
                        else {
                            continue;
                        };
                        let (want_i, want_j) = expected_square(pose, square);
                        if (code.centre_square.0 - want_i).rem_euclid(period) == 0
                            && (code.centre_square.1 - want_j).rem_euclid(period) == 0
                        {
                            correct += 1;
                            checks += code.check_bits as f64;
                        }
                    }
                    (correct, checks)
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().expect("worker panicked")).collect()
    });

    let correct: usize = per_thread.iter().map(|r| r.0).sum();
    let checks: f64 = per_thread.iter().map(|r| r.1).sum();
    let mean = if correct > 0 { checks / correct as f64 } else { 0.0 };
    (correct, all.len(), mean)
}

fn main() {
    // Two ways to make the comparison fair, because the layouts cannot be
    // matched on everything at once.
    //
    // At equal square size the diagonal layout's code steps are a/sqrt(2)
    // apart rather than a, so it packs ~1.4x more code bits into the same view.
    // That is extra redundancy -- and it is exactly the property that costs it
    // sqrt(2) of absolute range. Comparing there flatters it.
    //
    // Matching the range instead (diagonal square scaled by sqrt(2)) equalises
    // the code step, the bits in view and the absolute range, and makes the
    // diagonal layout pay its real price: a sqrt(2) coarser carrier.
    let configs: [(&str, f64, f64); 2] = [
        ("equal square size (diagonals gets ~1.4x more bits in view)", SQUARE, SQUARE),
        (
            "equal range and bits in view (diagonals pays a coarser carrier)",
            SQUARE,
            SQUARE * std::f64::consts::SQRT_2,
        ),
    ];

    println!(
        "Paired robustness comparison, {POSES} poses per cell, {SIZE}px, order {ORDER}."
    );
    println!("Each cell: correct decodes / total  (mean check bits on the correct ones)");

    for (title, square_axes, square_diag) in configs {
        println!("\n================================================================");
        println!("{title}");
        println!(
            "  lattice-axes square {square_axes:.2} px (code step {:.2}, carrier {:.2})",
            square_axes,
            square_axes * std::f64::consts::SQRT_2
        );
        println!(
            "  diagonals    square {square_diag:.2} px (code step {:.2}, carrier {:.2})",
            square_diag / std::f64::consts::SQRT_2,
            square_diag * std::f64::consts::SQRT_2
        );
        println!("================================================================");

        for kind in [Degradation::Noise, Degradation::Gradient, Degradation::Blur] {
            println!("\n{}", kind.label());
            println!("  {:>8}   {:>22}   {:>22}", "level", "lattice axes", "diagonals");
            for &level in kind.levels() {
                let (ca, na, ma) = evaluate(CodeLayout::LatticeAxes, square_axes, kind, level);
                let (cd, nd, md) = evaluate(CodeLayout::Diagonals, square_diag, kind, level);
                println!(
                    "  {level:>8.2}   {:>10} ({:>6.1} bits)   {:>10} ({:>6.1} bits)",
                    format!("{ca}/{na}"),
                    ma,
                    format!("{cd}/{nd}"),
                    md
                );
            }
        }
    }
}
