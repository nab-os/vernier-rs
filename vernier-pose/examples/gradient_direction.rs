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
const POSES: usize = 8;

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

fn poses() -> Vec<PatternPose> {
    (0..POSES)
        .map(|s| {
            let s = s as f64;
            PatternPose::new(s * 137.0 + 3.3, s * -83.0 - 5.7, (s - 3.5) * 0.18)
        })
        .collect()
}

fn evaluate(layout: CodeLayout, square: f64, degrees: f64, level: f64) -> (usize, usize) {
    let pattern = Checkerboard::new(square, ORDER).unwrap().with_code_layout(layout);
    let period = 3 * pattern.code().len() as i64;
    let backend = CpuBackend::new();
    let buffer = BufferLayout::packed(SIZE, SIZE);

    let mut correct = 0usize;
    let all = poses();
    for pose in &all {
        let mut image = pattern.render(SIZE, SIZE, pose).as_slice().to_vec();
        ramp(&mut image, degrees, level);

        let complex: Vec<Complex32> = image.iter().map(|&v| Complex32::new(v, 0.0)).collect();
        let Ok(detection) = analyze_two(&backend, &complex, buffer, 4.0, 10, 0, 0.0) else {
            continue;
        };
        let Ok(code) = extract_code_with_layout(&detection, &image, ORDER, layout) else {
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
    (correct, all.len())
}

fn main() {
    println!("Illumination ramp vs its direction. Matched range and bits in view:");
    println!("  lattice axes square {SQUARE_AXES:.2} px, diagonals square {SQUARE_DIAG:.2} px");
    println!("  {POSES} poses per cell (theta spread +/-36 deg), correct decodes / total\n");

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
