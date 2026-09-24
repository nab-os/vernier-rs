use vernier_core::Complex32;
use vernier_core::buffer::BufferLayout;
use vernier_cpu::CpuBackend;
use vernier_patterns::PatternPose;
use vernier_patterns::checkerboard::{Checkerboard, CodeLayout, CodePacking};
use vernier_pose::checkerboard::{
    CheckerboardError, detect_checkerboard_with_packing, extract_code,
    solve_checkerboard_with_packing,
};
use vernier_spectral::spectrum::analyze_two;

const SIZE: usize = 512;
const SQUARE: f64 = 8.0;
const ORDER: u32 = 8;
const LAYOUTS: [CodeLayout; 2] = [CodeLayout::Squares, CodeLayout::Diamonds];

fn random_poses(seed: u64, n: usize) -> Vec<PatternPose> {
    let mut state = seed;
    let mut unit = move || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((state >> 32) as f64 + 0.5) / (u32::MAX as f64 + 1.0)
    };
    (0..n)
        .map(|_| {
            let (x, y) = ((unit() - 0.5) * 4000.0, (unit() - 0.5) * 4000.0);
            PatternPose::new(x, y, (unit() - 0.5) * 1.26)
        })
        .collect()
}

fn solve(pattern: &Checkerboard, pose: &PatternPose) -> vernier_core::Pose {
    let image = pattern.render(SIZE, SIZE, pose);
    let buffer = BufferLayout::packed(SIZE, SIZE);
    let packing = pattern.code_packing();
    let detection = detect_checkerboard_with_packing(
        &CpuBackend::new(), image.as_slice(), buffer, 4.0, 10, 0, 0.0, packing,
    )
    .unwrap();
    solve_checkerboard_with_packing(
        &detection, image.as_slice(), SQUARE, ORDER, pattern.code_layout(), packing,
    )
    .unwrap_or_else(|e| panic!("{:?}/{packing:?} at {pose:?}: {e}", pattern.code_layout()))
    .0
}

fn position_error(pattern: &Checkerboard, pose: &PatternPose) -> f64 {
    let found = solve(pattern, pose);
    let (dx, dy) = pattern.wrap_offset(found.x + pose.x, found.y + pose.y);
    dx.abs().max(dy.abs())
}

#[test]
fn decodes_random_poses() {
    for layout in LAYOUTS {
        let pattern = Checkerboard::new(SQUARE, ORDER).unwrap().with_code_layout(layout);
        for pose in random_poses(0x2545_f491_4f6c_dd1d, 24) {
            let error = position_error(&pattern, &pose);
            assert!(error < 0.25, "{layout:?} at {pose:?}: {error:.3} px");
        }
    }
}

#[test]
fn reports_the_orientation() {
    for layout in LAYOUTS {
        let pattern = Checkerboard::new(SQUARE, ORDER).unwrap().with_code_layout(layout);
        for degrees in [-170.0f64, -80.0, 0.0, 45.0, 100.0, 150.0] {
            let theta = degrees.to_radians();
            let found = solve(&pattern, &PatternPose::new(123.4, -567.8, theta));
            let error = (found.theta - theta + std::f64::consts::PI).rem_euclid(std::f64::consts::TAU)
                - std::f64::consts::PI;
            assert!(error.to_degrees().abs() < 0.05, "{layout:?} at {degrees}: {}", found.theta.to_degrees());
        }
    }
}

// These poses make the plain peak search lock onto the code's 1/3 line.
const LOCKING_POSES: [usize; 5] = [1, 27, 38, 52, 97];

#[test]
fn recovers_from_a_subharmonic_lock() {
    let poses = random_poses(0x9e37_79b9_7f4a_7c15, 100);
    for layout in LAYOUTS {
        let pattern = Checkerboard::new(SQUARE, ORDER).unwrap().with_code_layout(layout);
        for index in LOCKING_POSES {
            assert!(position_error(&pattern, &poses[index]) < 0.5 * SQUARE, "{layout:?} pose {index}");
        }
    }
}

#[test]
fn refuses_a_subharmonic_lock() {
    let poses = random_poses(0x9e37_79b9_7f4a_7c15, 100);
    let pattern = Checkerboard::new(SQUARE, ORDER).unwrap();
    for index in [1, 97] {
        let image = pattern.render(SIZE, SIZE, &poses[index]);
        let complex: Vec<Complex32> = image.as_slice().iter().map(|&v| Complex32::new(v, 0.0)).collect();
        let buffer = BufferLayout::packed(SIZE, SIZE);
        let detection = analyze_two(&CpuBackend::new(), &complex, buffer, 4.0, 10, 0, 0.0).unwrap();
        assert!(matches!(
            extract_code(&detection, image.as_slice(), ORDER),
            Err(CheckerboardError::SubharmonicLock)
        ));
    }
}

/// The same round trip on the denser packing: two bits per supercell means the
/// decoder has to lift each run's position by its slot parity, not just by the
/// LFSR index, or it lands half a supercell out.
#[test]
fn decodes_random_poses_with_two_bit_packing() {
    for layout in LAYOUTS {
        let pattern = Checkerboard::new(SQUARE, ORDER)
            .unwrap()
            .with_code_layout(layout)
            .with_code_packing(CodePacking::TwoBits);
        for pose in random_poses(0x2545_f491_4f6c_dd1d, 24) {
            let error = position_error(&pattern, &pose);
            assert!(error < 0.25, "{layout:?} at {pose:?}: {error:.3} px");
        }
    }
}

/// The denser packing must reach further for the same order: its supercell is
/// bigger, so one sequence spans more squares.
#[test]
fn two_bit_packing_reaches_further() {
    let one = Checkerboard::new(SQUARE, ORDER).unwrap();
    let two = one.clone().with_code_packing(CodePacking::TwoBits);
    assert!(two.range_squares() > one.range_squares());
}
