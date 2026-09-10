//! Is the illumination result a layout property, or an artefact of the ramp's
//! direction?
//!
//! `layout_robustness` ramps the illumination along `(row + col)`. At θ=0 that
//! is exactly the direction `u = i+j` the diagonal layout writes its code along,
//! so each of its code bands sits at one illumination level and — since every
//! site in a band shares a colour — the whole band misreads together. The
//! lattice layout's constant-`i` bands cut across that ramp, so each one samples
//! many levels and its majority vote averages over them.
//!
//! That would favour the lattice layout for a reason that has nothing to do with
//! the code geometry. So sweep the ramp direction: if the advantage survives
//! every direction it is real, and if it flips it was the ramp all along.
//!
//! Run with `cargo run --release --example gradient_direction -p vernier-pose`.

use vernier_core::Complex32;
use vernier_core::buffer::BufferLayout;
use vernier_cpu::CpuBackend;
use vernier_patterns::PatternPose;
use vernier_patterns::checkerboard::{Checkerboard, CodeLayout};
use vernier_pose::checkerboard::extract_code_with_layout;
use vernier_spectral::spectrum::analyze_two;

const SIZE: usize = 512;
const ORDER: u32 = 8;
const POSES: usize = 100;

/// Deterministic generator, so a rerun reproduces the table exactly.
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
}

/// Matched range and bits in view, as in the second `layout_robustness` config.
const SQUARE_AXES: f64 = 8.0;
const SQUARE_DIAG: f64 = 8.0 * std::f64::consts::SQRT_2;

/// Direction the illumination ramp runs, in degrees from the image x axis.
const DIRECTIONS: [f64; 4] = [0.0, 45.0, 90.0, 135.0];
const LEVELS: [f64; 4] = [0.0, 0.6, 0.8, 1.0];

fn ramp(image: &mut [f32], degrees: f64, level: f64) {
    if level <= 0.0 {
        return;
    }
    let (sin, cos) = degrees.to_radians().sin_cos();
    // Project onto the ramp direction, normalised so the span is [-0.5, 0.5]
    // whatever the angle — otherwise a diagonal ramp would be √2 stronger.
    let half = SIZE as f64 * 0.5;
    let extent = half * (cos.abs() + sin.abs());
    for row in 0..SIZE {
        for col in 0..SIZE {
            let (x, y) = (col as f64 - half, row as f64 - half);
            let t = (x * cos + y * sin) / (2.0 * extent);
            let gain = 1.0 + 2.0 * level * t;
            let p = &mut image[row * SIZE + col];
            *p = ((*p as f64) * gain).clamp(0.0, 1.0) as f32;
        }
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


fn evaluate(layout: CodeLayout, square: f64, degrees: f64, level: f64) -> (usize, usize) {
    let pattern = Checkerboard::new(square, ORDER).unwrap().with_code_layout(layout);
    let period = 3 * pattern.code().len() as i64;
    let all = poses();
    let chunk = all.len().div_ceil(threads());

    let correct: usize = std::thread::scope(|scope| {
        let handles: Vec<_> = all
            .chunks(chunk)
            .map(|slice| {
                let pattern = &pattern;
                scope.spawn(move || {
                    let backend = CpuBackend::new();
                    let buffer = BufferLayout::packed(SIZE, SIZE);
                    let mut correct = 0usize;
                    for pose in slice {
                        let mut image = pattern.render(SIZE, SIZE, pose).as_slice().to_vec();
                        ramp(&mut image, degrees, level);
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
                        let want = (
                            (-pose.x / square - 0.5).round() as i64,
                            (-pose.y / square - 0.5).round() as i64,
                        );
                        if (code.centre_square.0 - want.0).rem_euclid(period) == 0
                            && (code.centre_square.1 - want.1).rem_euclid(period) == 0
                        {
                            correct += 1;
                        }
                    }
                    correct
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().expect("worker panicked")).sum()
    });
    (correct, all.len())
}

fn main() {
    println!("Illumination ramp vs its direction. Matched range and bits in view:");
    println!("  lattice axes square {SQUARE_AXES:.2} px, diagonals square {SQUARE_DIAG:.2} px");
    println!("  {POSES} poses per cell (theta uniform over +/-36 deg), correct decodes / total\n");

    println!(
        "  {:>6}  {:>10}  {:>14}  {:>12}",
        "ramp", "level", "lattice axes", "diagonals"
    );
    for degrees in DIRECTIONS {
        for level in LEVELS {
            if level == 0.0 && degrees != DIRECTIONS[0] {
                continue; // the undegraded row is the same for every direction
            }
            let (ca, na) = evaluate(CodeLayout::LatticeAxes, SQUARE_AXES, degrees, level);
            let (cd, nd) = evaluate(CodeLayout::Diagonals, SQUARE_DIAG, degrees, level);
            println!(
                "  {:>5.0} deg  {:>10.2}  {:>14}  {:>12}",
                degrees,
                level,
                format!("{ca}/{na}"),
                format!("{cd}/{nd}")
            );
        }
        println!();
    }
}
