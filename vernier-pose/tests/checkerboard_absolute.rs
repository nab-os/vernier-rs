//! Render at a known pose, detect, decode, compare with the pose.

use vernier_core::Complex32;
use vernier_core::buffer::BufferLayout;
use vernier_cpu::CpuBackend;
use vernier_patterns::PatternPose;
use vernier_patterns::checkerboard::{Checkerboard, CodeLayout};
use vernier_pose::checkerboard::{
    CheckerboardError, detect_checkerboard, extract_code, extract_code_with_layout,
    solve_checkerboard, solve_checkerboard_with_layout,
};
use vernier_spectral::spectrum::{Detection, analyze_two};

const SIZE: usize = 512;
const SQUARE: f64 = 8.0;
const ORDER: u32 = 8;

fn detect(image: &vernier_core::GrayImage) -> Detection {
    let backend = CpuBackend::new();
    let layout = BufferLayout::packed(SIZE, SIZE);
    detect_checkerboard(&backend, image.as_slice(), layout, 4.0, 10, 0, 0.0).expect("two carrier peaks")
}

/// Plain peak search, without the sub-harmonic retry.
fn detect_unguarded(image: &vernier_core::GrayImage) -> Detection {
    let backend = CpuBackend::new();
    let layout = BufferLayout::packed(SIZE, SIZE);
    let complex: Vec<Complex32> = image
        .as_slice()
        .iter()
        .map(|&v| Complex32::new(v, 0.0))
        .collect();
    analyze_two(&backend, &complex, layout, 4.0, 10, 0, 0.0).expect("two carrier peaks")
}

/// Square under the image centre, which sits at pattern point `(-x, -y)`.
fn expected_square(pattern: &Checkerboard, pose: &PatternPose) -> (i64, i64) {
    let (x, y) = pattern.code_layout().to_lattice(-pose.x, -pose.y);
    ((x / SQUARE - 0.5).round() as i64, (y / SQUARE - 0.5).round() as i64)
}

fn assert_decodes_to_pose(pattern: &Checkerboard, pose: PatternPose) {
    let image = pattern.render(SIZE, SIZE, &pose);
    let detection = detect(&image);
    let code = extract_code_with_layout(
        &detection,
        image.as_slice(),
        ORDER,
        pattern.code_layout(),
    )
    .unwrap_or_else(|e| panic!("decode failed at {pose:?} [{:?}]: {e}", pattern.code_layout()));

    let period = 3 * pattern.code().len() as i64;
    let (want_i, want_j) = expected_square(pattern, &pose);
    let (got_i, got_j) = code.centre_square;
    assert_eq!(
        (
            (got_i - want_i).rem_euclid(period),
            (got_j - want_j).rem_euclid(period)
        ),
        (0, 0),
        "at {pose:?}: decoded square ({got_i},{got_j}), expected ({want_i},{want_j}) \
         [transform {}, {} check bits]",
        code.transform,
        code.check_bits,
    );
}

#[test]
fn decodes_the_absolute_square_at_many_translations() {
    let pattern = Checkerboard::new(SQUARE, ORDER).unwrap();
    for step in 0..12 {
        let pose = PatternPose::new(step as f64 * 137.0 + 3.3, step as f64 * -83.0 - 5.7, 0.0);
        assert_decodes_to_pose(&pattern, pose);
    }
}

#[test]
fn decodes_through_rotation() {
    let pattern = Checkerboard::new(SQUARE, ORDER).unwrap();
    for (index, theta) in [0.12, 0.55, -0.31, 0.79].into_iter().enumerate() {
        let pose = PatternPose::new(index as f64 * 61.0 + 2.5, index as f64 * 47.0 - 1.5, theta);
        assert_decodes_to_pose(&pattern, pose);
    }
}

#[test]
fn recovers_sub_square_position_not_just_the_square() {
    let pattern = Checkerboard::new(SQUARE, ORDER).unwrap();
    let mut worst = 0.0f64;
    for step in 0..8 {
        let pose = PatternPose::new(step as f64 * 29.0 + 1.7, step as f64 * 17.0 + 0.9, 0.0);
        let image = pattern.render(SIZE, SIZE, &pose);
        let detection = detect(&image);
        let (recovered, _) = solve_checkerboard(&detection, image.as_slice(), SQUARE, ORDER)
            .unwrap_or_else(|e| panic!("solve failed at {pose:?}: {e}"));

        let (error_x, error_y) = pattern.wrap_offset(recovered.x + pose.x, recovered.y + pose.y);
        worst = worst.max(error_x.abs()).max(error_y.abs());
    }
    assert!(
        worst < 0.25,
        "worst absolute position error {worst:.4} px — the fine phase is not \
         being combined with the code correctly"
    );
}

#[test]
fn rejects_an_image_with_no_code_in_view() {
    // Too few squares in view: must error, not guess.
    let pattern = Checkerboard::new(28.0, ORDER).unwrap();
    let image = pattern.render(128, 128, &PatternPose::IDENTITY);
    let backend = CpuBackend::new();
    let layout = BufferLayout::packed(128, 128);
    let complex: Vec<Complex32> = image
        .as_slice()
        .iter()
        .map(|&v| Complex32::new(v, 0.0))
        .collect();
    let Ok(detection) = analyze_two(&backend, &complex, layout, 4.0, 4, 0, 0.0) else {
        return;
    };
    assert!(
        extract_code(&detection, image.as_slice(), ORDER).is_err(),
        "a 4-square-wide view must not produce a confident decode"
    );
}

// --- diamonds ---

fn diamonds() -> Checkerboard {
    Checkerboard::new(SQUARE, ORDER)
        .unwrap()
        .with_code_layout(CodeLayout::Diamonds)
}

#[test]
fn diamonds_decode_the_absolute_square_at_many_translations() {
    let pattern = diamonds();
    for step in 0..12 {
        let pose = PatternPose::new(step as f64 * 137.0 + 3.3, step as f64 * -83.0 - 5.7, 0.0);
        assert_decodes_to_pose(&pattern, pose);
    }
}

#[test]
fn diamonds_decode_through_rotation() {
    let pattern = diamonds();
    for (index, theta) in [0.12, 0.55, -0.31, 0.79].into_iter().enumerate() {
        let pose = PatternPose::new(index as f64 * 61.0 + 2.5, index as f64 * 47.0 - 1.5, theta);
        assert_decodes_to_pose(&pattern, pose);
    }
}

#[test]
fn the_layouts_do_not_decode_each_other() {
    let pose = PatternPose::new(311.0 + 3.3, -177.0 - 5.7, 0.0);
    let pattern = diamonds();
    let image = pattern.render(SIZE, SIZE, &pose);
    let detection = detect(&image);

    let right = extract_code_with_layout(&detection, image.as_slice(), ORDER, CodeLayout::Diamonds)
        .expect("the matching layout must decode");
    let (want_i, want_j) = expected_square(&pattern, &pose);
    let period = 3 * pattern.code().len() as i64;
    assert_eq!(
        (
            (right.centre_square.0 - want_i).rem_euclid(period),
            (right.centre_square.1 - want_j).rem_euclid(period)
        ),
        (0, 0),
        "matching layout decoded the wrong square"
    );

    if let Ok(wrong) = extract_code(&detection, image.as_slice(), ORDER) {
        assert_ne!(
            wrong.centre_square, right.centre_square,
            "the squares rule reproduced the diamonds decode"
        );
    }
}

#[test]
fn diamonds_recover_sub_square_position() {
    // Catches a half-period slip in the (u, v) -> (i, j) conversion.
    let pattern = diamonds();
    let mut worst = 0.0f64;
    for step in 0..8 {
        let pose = PatternPose::new(step as f64 * 29.0 + 1.7, step as f64 * 17.0 + 0.9, 0.0);
        let image = pattern.render(SIZE, SIZE, &pose);
        let detection = detect(&image);
        let (recovered, _) = solve_checkerboard_with_layout(
            &detection,
            image.as_slice(),
            SQUARE,
            ORDER,
            CodeLayout::Diamonds,
        )
        .unwrap_or_else(|e| panic!("solve failed at {pose:?}: {e}"));

        let (error_x, error_y) = pattern.wrap_offset(recovered.x + pose.x, recovered.y + pose.y);
        worst = worst.max(error_x.abs()).max(error_y.abs());
    }
    assert!(
        worst < 0.25,
        "worst absolute position error {worst:.4} px -- the fine phase is not \
         being combined with the diamond code correctly"
    );
}

#[test]
fn diamonds_decode_random_poses() {
    // Hits both parities of Δu + Δv; the fixed poses above only hit even.
    let pattern = diamonds();
    let mut state: u64 = 0x2545_f491_4f6c_dd1d;
    let mut unit = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 32) as f64 + 0.5) / (u32::MAX as f64 + 1.0)
    };
    for _ in 0..24 {
        let pose = PatternPose::new(
            (unit() - 0.5) * 4000.0,
            (unit() - 0.5) * 4000.0,
            (unit() - 0.5) * 1.26,
        );
        assert_decodes_to_pose(&pattern, pose);
    }
}

// --- detection locked onto the code's 1/3 line ---

fn random_poses() -> Vec<PatternPose> {
    let mut state: u64 = 0x9e37_79b9_7f4a_7c15;
    let mut unit = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 32) as f64 + 0.5) / (u32::MAX as f64 + 1.0)
    };
    (0..100)
        .map(|_| {
            let x = (unit() - 0.5) * 4000.0;
            let y = (unit() - 0.5) * 4000.0;
            let theta = (unit() - 0.5) * 2.0 * 0.63;
            PatternPose::new(x, y, theta)
        })
        .collect()
}

// Poses from `random_poses` that used to lock onto the 1/3 line.
const SUBHARMONIC_POSES: [usize; 5] = [1, 27, 38, 52, 97];

#[test]
fn guarded_detection_decodes_poses_that_lock_onto_the_code_line() {
    let poses = random_poses();
    for layout in [CodeLayout::Squares, CodeLayout::Diamonds] {
        let pattern = Checkerboard::new(SQUARE, ORDER).unwrap().with_code_layout(layout);
        for &index in &SUBHARMONIC_POSES {
            assert_decodes_to_pose(&pattern, poses[index]);
        }
    }
}

#[test]
fn decoder_refuses_a_lock_onto_the_code_line() {
    let poses = random_poses();
    let pattern = Checkerboard::new(SQUARE, ORDER).unwrap();
    // 97 used to decode to the wrong square.
    for &index in &[1usize, 97] {
        let image = pattern.render(SIZE, SIZE, &poses[index]);
        let detection = detect_unguarded(&image);
        match extract_code(&detection, image.as_slice(), ORDER) {
            Err(CheckerboardError::SubharmonicLock) => {}
            other => panic!("pose {index}: expected SubharmonicLock, got {other:?}"),
        }
    }
}

#[test]
fn reports_the_pattern_orientation_at_every_quarter() {
    // Was once 45° off everywhere.
    for layout in [CodeLayout::Squares, CodeLayout::Diamonds] {
        let pattern = Checkerboard::new(SQUARE, ORDER).unwrap().with_code_layout(layout);
        for degrees in [-170.0f64, -80.0, -30.0, 0.0, 3.0, 45.0, 70.0, 100.0, 150.0] {
            let theta = degrees.to_radians();
            let pose = PatternPose::new(123.4, -567.8, theta);
            let image = pattern.render(SIZE, SIZE, &pose);
            let detection = detect(&image);
            let (recovered, _) =
                solve_checkerboard_with_layout(&detection, image.as_slice(), SQUARE, ORDER, layout)
                    .unwrap_or_else(|e| panic!("{layout:?} at {degrees} deg: {e}"));
            let error = (recovered.theta - theta + std::f64::consts::PI)
                .rem_euclid(std::f64::consts::TAU)
                - std::f64::consts::PI;
            assert!(
                error.to_degrees().abs() < 0.05,
                "{layout:?} at {degrees} deg: reported {:.3} deg",
                recovered.theta.to_degrees()
            );
            assert!(
                (-std::f64::consts::PI..=std::f64::consts::PI).contains(&recovered.theta),
                "orientation not wrapped: {}",
                recovered.theta
            );
        }
    }
}
