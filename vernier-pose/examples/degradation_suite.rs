//! Extended robustness suite: three degradation families, three variants each,
//! on fine level grids, for both code layouts.
//!
//! Every condition decodes the same 100 random poses for three patterns:
//! lattice axes at 8 px, diagonals at 8 px (equal square size), and diagonals at
//! 8·√2 px (equal range and bits in view). The lattice images are identical in
//! both comparisons, so they are decoded once. Degradations are seeded from the
//! condition and pose, never the pattern, so the comparison is paired.
//!
//! Output is CSV on stdout (progress on stderr):
//!   family,variant,level,pattern,correct,total,mean_check_bits
//!
//! Run with
//!   cargo run --release --example degradation_suite -p vernier-pose > suite.csv

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

// ------------------------------------------------------------------ randomness

struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        // Mix the seed so neighbouring seeds do not start correlated.
        let mut z = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        Lcg(z ^ (z >> 31))
    }
    fn next_u32(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }
    fn unit(&mut self) -> f64 {
        (self.next_u32() as f64 + 0.5) / (u32::MAX as f64 + 1.0)
    }
    fn normal(&mut self) -> f64 {
        let (u1, u2) = (self.unit(), self.unit());
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
    /// Poisson sample: Knuth for small means, normal approximation above 30.
    fn poisson(&mut self, mean: f64) -> f64 {
        if mean <= 0.0 {
            return 0.0;
        }
        if mean > 30.0 {
            return (mean + mean.sqrt() * self.normal()).round().max(0.0);
        }
        let limit = (-mean).exp();
        let (mut k, mut p) = (0.0, 1.0);
        loop {
            p *= self.unit();
            if p <= limit {
                return k;
            }
            k += 1.0;
        }
    }
}

// ------------------------------------------------------------------ degradations

#[derive(Clone, Copy)]
enum Variant {
    Ramp,
    Vignette,
    Shading,
    Defocus,
    Motion0,
    Motion45,
    Gaussian,
    Shot,
    SaltPepper,
}

impl Variant {
    fn family(self) -> &'static str {
        match self {
            Variant::Ramp | Variant::Vignette | Variant::Shading => "illumination",
            Variant::Defocus | Variant::Motion0 | Variant::Motion45 => "blur",
            Variant::Gaussian | Variant::Shot | Variant::SaltPepper => "noise",
        }
    }
    fn name(self) -> &'static str {
        match self {
            Variant::Ramp => "linear ramp",
            Variant::Vignette => "vignetting",
            Variant::Shading => "low-frequency shading",
            Variant::Defocus => "gaussian defocus",
            Variant::Motion0 => "motion blur 0deg",
            Variant::Motion45 => "motion blur 45deg",
            Variant::Gaussian => "gaussian noise",
            Variant::Shot => "shot noise",
            Variant::SaltPepper => "salt and pepper",
        }
    }
    fn levels(self) -> Vec<f64> {
        let grid = |hi: f64, step: f64| {
            let n = (hi / step).round() as usize;
            (0..=n).map(|k| k as f64 * step).collect::<Vec<_>>()
        };
        match self {
            Variant::Ramp => grid(1.2, 0.1),
            Variant::Vignette | Variant::Shading => grid(1.0, 0.1),
            Variant::Defocus => grid(4.0, 0.25),
            Variant::Motion0 | Variant::Motion45 => grid(24.0, 2.0),
            Variant::Gaussian => grid(2.0, 0.1),
            // Photons collected by a fully white pixel: fewer is worse.
            Variant::Shot => vec![1000.0, 300.0, 100.0, 30.0, 10.0, 5.0, 3.0, 2.0, 1.0],
            Variant::SaltPepper => grid(0.5, 0.05),
        }
    }
    fn id(self) -> u64 {
        self as u64
    }
}

const ALL: [Variant; 9] = [
    Variant::Ramp,
    Variant::Vignette,
    Variant::Shading,
    Variant::Defocus,
    Variant::Motion0,
    Variant::Motion45,
    Variant::Gaussian,
    Variant::Shot,
    Variant::SaltPepper,
];

fn sample_bilinear(image: &[f32], x: f64, y: f64) -> f64 {
    let max = (SIZE - 1) as f64;
    let (x, y) = (x.clamp(0.0, max), y.clamp(0.0, max));
    let (x0, y0) = (x.floor() as usize, y.floor() as usize);
    let (x1, y1) = ((x0 + 1).min(SIZE - 1), (y0 + 1).min(SIZE - 1));
    let (fx, fy) = (x - x0 as f64, y - y0 as f64);
    let at = |c: usize, r: usize| image[r * SIZE + c] as f64;
    (at(x0, y0) * (1.0 - fx) + at(x1, y0) * fx) * (1.0 - fy)
        + (at(x0, y1) * (1.0 - fx) + at(x1, y1) * fx) * fy
}

fn gaussian_blur(image: &mut [f32], sigma: f64) {
    if sigma <= 0.0 {
        return;
    }
    let radius = (3.0 * sigma).ceil() as isize;
    let kernel: Vec<f64> = (-radius..=radius)
        .map(|d| (-(d * d) as f64 / (2.0 * sigma * sigma)).exp())
        .collect();
    let norm: f64 = kernel.iter().sum();
    let clamp = |v: isize| v.clamp(0, SIZE as isize - 1) as usize;
    let mut pass = vec![0.0f32; image.len()];
    for r in 0..SIZE {
        for c in 0..SIZE {
            let acc: f64 = kernel
                .iter()
                .enumerate()
                .map(|(k, w)| w * image[r * SIZE + clamp(c as isize + k as isize - radius)] as f64)
                .sum();
            pass[r * SIZE + c] = (acc / norm) as f32;
        }
    }
    for c in 0..SIZE {
        for r in 0..SIZE {
            let acc: f64 = kernel
                .iter()
                .enumerate()
                .map(|(k, w)| w * pass[clamp(r as isize + k as isize - radius) * SIZE + c] as f64)
                .sum();
            image[r * SIZE + c] = (acc / norm) as f32;
        }
    }
}

fn motion_blur(image: &mut [f32], length: f64, degrees: f64) {
    if length <= 0.0 {
        return;
    }
    let (dy, dx) = degrees.to_radians().sin_cos();
    let taps = length.round() as i64 + 1;
    let source = image.to_vec();
    for r in 0..SIZE {
        for c in 0..SIZE {
            let mut acc = 0.0;
            for k in 0..taps {
                let t = k as f64 - (taps - 1) as f64 / 2.0;
                acc += sample_bilinear(&source, c as f64 + t * dx, r as f64 + t * dy);
            }
            image[r * SIZE + c] = (acc / taps as f64) as f32;
        }
    }
}

fn degrade(image: &mut [f32], variant: Variant, level: f64, seed: u64) {
    let half = SIZE as f64 / 2.0;
    let gain_map = |image: &mut [f32], gain: &dyn Fn(f64, f64) -> f64| {
        for r in 0..SIZE {
            for c in 0..SIZE {
                let p = &mut image[r * SIZE + c];
                *p = (*p as f64 * gain(c as f64 - half, r as f64 - half)) as f32;
            }
        }
    };
    match variant {
        Variant::Ramp => gain_map(image, &|x, y| {
            // Along the image diagonal, spanning [1 - a, 1 + a].
            1.0 + level * (x + y) / (2.0 * half)
        }),
        Variant::Vignette => {
            let rmax2 = 2.0 * half * half;
            gain_map(image, &|x, y| (1.0 - level * (x * x + y * y) / rmax2).max(0.0))
        }
        Variant::Shading => {
            // Blotches about 256 px across, a phase offset so the centre is not
            // special.
            let k = std::f64::consts::TAU / 256.0;
            gain_map(image, &|x, y| 1.0 + level * (k * x + 0.7).sin() * (k * y + 1.9).sin())
        }
        Variant::Defocus => gaussian_blur(image, level),
        Variant::Motion0 => motion_blur(image, level, 0.0),
        Variant::Motion45 => motion_blur(image, level, 45.0),
        Variant::Gaussian => {
            let mut rng = Lcg::new(seed);
            for p in image.iter_mut() {
                *p = (*p as f64 + level * rng.normal()) as f32;
            }
        }
        Variant::Shot => {
            let mut rng = Lcg::new(seed);
            for p in image.iter_mut() {
                *p = (rng.poisson(level * (*p as f64).max(0.0)) / level) as f32;
            }
        }
        Variant::SaltPepper => {
            let mut rng = Lcg::new(seed);
            for p in image.iter_mut() {
                if rng.unit() < level {
                    *p = if rng.unit() < 0.5 { 0.0 } else { 1.0 };
                }
            }
        }
    }
    // A sensor clips; that is part of every one of these.
    for p in image.iter_mut() {
        *p = p.clamp(0.0, 1.0);
    }
}

// ------------------------------------------------------------------ experiment

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

struct Subject {
    name: &'static str,
    pattern: Checkerboard,
    square: f64,
    /// Clean renders, one per pose, reused by every condition.
    renders: Vec<Vec<f32>>,
}

fn threads() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4)
}

fn evaluate(subject: &Subject, all: &[PatternPose], variant: Variant, level: f64) -> (usize, f64) {
    let layout = subject.pattern.code_layout();
    let period = 3 * subject.pattern.code().len() as i64;
    let chunk = all.len().div_ceil(threads());
    let results: Vec<(usize, f64)> = std::thread::scope(|scope| {
        let handles: Vec<_> = all
            .chunks(chunk)
            .enumerate()
            .map(|(c, slice)| {
                scope.spawn(move || {
                    let backend = CpuBackend::new();
                    let buffer = BufferLayout::packed(SIZE, SIZE);
                    let (mut correct, mut checks) = (0usize, 0f64);
                    for (offset, pose) in slice.iter().enumerate() {
                        let index = c * chunk + offset;
                        let mut image = subject.renders[index].clone();
                        let seed = (variant.id() << 56) ^ (level.to_bits() >> 8) ^ index as u64;
                        degrade(&mut image, variant, level, seed);
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
                            (-pose.x / subject.square - 0.5).round() as i64,
                            (-pose.y / subject.square - 0.5).round() as i64,
                        );
                        if (code.centre_square.0 - want.0).rem_euclid(period) == 0
                            && (code.centre_square.1 - want.1).rem_euclid(period) == 0
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
    let correct = results.iter().map(|r| r.0).sum();
    let checks: f64 = results.iter().map(|r| r.1).sum();
    (correct, if correct > 0 { checks / correct as f64 } else { 0.0 })
}

fn render_all(pattern: &Checkerboard, all: &[PatternPose]) -> Vec<Vec<f32>> {
    let chunk = all.len().div_ceil(threads());
    std::thread::scope(|scope| {
        let handles: Vec<_> = all
            .chunks(chunk)
            .map(|slice| {
                scope.spawn(move || {
                    slice
                        .iter()
                        .map(|pose| pattern.render(SIZE, SIZE, pose).as_slice().to_vec())
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        handles.into_iter().flat_map(|h| h.join().unwrap()).collect()
    })
}

fn main() {
    let all = poses();
    let make = |name, square: f64, layout| {
        let pattern = Checkerboard::new(square, ORDER).unwrap().with_code_layout(layout);
        let renders = render_all(&pattern, &all);
        Subject { name, pattern, square, renders }
    };
    eprintln!("rendering clean images...");
    let subjects = [
        make("lattice8", 8.0, CodeLayout::LatticeAxes),
        make("diag8", 8.0, CodeLayout::Diagonals),
        make("diag11", 8.0 * std::f64::consts::SQRT_2, CodeLayout::Diagonals),
    ];

    println!("family,variant,level,pattern,correct,total,mean_check_bits");
    let total: usize = ALL.iter().map(|v| v.levels().len()).sum();
    let mut done = 0;
    for variant in ALL {
        for level in variant.levels() {
            for subject in &subjects {
                let (correct, bits) = evaluate(subject, &all, variant, level);
                println!(
                    "{},{},{level},{},{correct},{POSES},{bits:.2}",
                    variant.family(),
                    variant.name(),
                    subject.name
                );
            }
            done += 1;
            eprintln!("[{done}/{total}] {} {level}", variant.name());
        }
    }
}
