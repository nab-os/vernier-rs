//! Full absolute round trip for the coded checkerboard: render at a known pose,
//! run two-direction detection, decode the code, and check the recovered
//! absolute square index against the pose that was rendered.
//!
//! Unlike a self-consistency check, this compares against ground truth. The
//! image centre of a pattern rendered at `pose` sits at pattern coordinates
//! `(−pose.x, −pose.y)`, so the square it lands on is known exactly, and the
//! decode has to name that square out of the `3·(2⁸−1) = 765` in the sequence.

use vernier_core::Complex32;
use vernier_core::buffer::BufferLayout;
use vernier_cpu::CpuBackend;
use vernier_patterns::PatternPose;
use vernier_patterns::checkerboard::{Checkerboard, CodeLayout};
use vernier_pose::checkerboard::{
    extract_code, extract_code_with_layout, solve_checkerboard, solve_checkerboard_with_layout,
};
use vernier_spectral::spectrum::{Detection, analyze_two};

const SIZE: usize = 512;
const SQUARE: f64 = 8.0;
const ORDER: u32 = 8;

fn detect(image: &vernier_core::GrayImage) -> Detection {
    let backend = CpuBackend::new();
    let layout = BufferLayout::packed(SIZE, SIZE);
    let complex: Vec<Complex32> = image
        .as_slice()
        .iter()
        .map(|&v| Complex32::new(v, 0.0))
        .collect();
    analyze_two(&backend, &complex, layout, 4.0, 10, 0, 0.0).expect("two carrier peaks")
}

/// The square the image centre lands on, from the rendering pose alone.
fn expected_square(pose: &PatternPose) -> (i64, i64) {
    (
        (-pose.x / SQUARE - 0.5).round() as i64,
        (-pose.y / SQUARE - 0.5).round() as i64,
    )
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
    let (want_i, want_j) = expected_square(&pose);
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
    // Translations spread over hundreds of squares, deliberately not on square
    // boundaries, so the sub-square phase is exercised too.
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
    // The whole point of coarse+fine: the absolute position must be continuous,
    // not quantized to the square grid.
    let pattern = Checkerboard::new(SQUARE, ORDER).unwrap();
    let mut worst = 0.0f64;
    for step in 0..8 {
        let pose = PatternPose::new(step as f64 * 29.0 + 1.7, step as f64 * 17.0 + 0.9, 0.0);
        let image = pattern.render(SIZE, SIZE, &pose);
        let detection = detect(&image);
        let (recovered, _) = solve_checkerboard(&detection, image.as_slice(), SQUARE, ORDER)
            .unwrap_or_else(|e| panic!("solve failed at {pose:?}: {e}"));

        let period = 3.0 * pattern.code().len() as f64 * SQUARE;
        let error_x = (recovered.x - -pose.x).rem_euclid(period);
        let error_y = (recovered.y - -pose.y).rem_euclid(period);
        let error_x = error_x.min(period - error_x);
        let error_y = error_y.min(period - error_y);
        worst = worst.max(error_x).max(error_y);
    }
    assert!(
        worst < 0.25,
        "worst absolute position error {worst:.4} px — the fine phase is not \
         being combined with the code correctly"
    );
}

#[test]
fn rejects_an_image_with_no_code_in_view() {
    // A field of view too small to hold order + spare bits must fail loudly
    // rather than return a confident wrong answer.
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
        return; // No peaks at all is an acceptable failure for this case.
    };
    assert!(
        extract_code(&detection, image.as_slice(), ORDER).is_err(),
        "a 4-square-wide view must not produce a confident decode"
    );
}

// --- diagonal code layout ---------------------------------------------------
//
// Same pattern geometry, code indexed along the carriers instead of the square
// edges. Ground truth is identical: the layout changes which square carries
// which bit, not where the squares are.

fn diagonal() -> Checkerboard {
    Checkerboard::new(SQUARE, ORDER)
        .unwrap()
        .with_code_layout(CodeLayout::Diagonals)
}

#[test]
fn diagonal_layout_decodes_the_absolute_square_at_many_translations() {
    let pattern = diagonal();
    for step in 0..12 {
        let pose = PatternPose::new(step as f64 * 137.0 + 3.3, step as f64 * -83.0 - 5.7, 0.0);
        assert_decodes_to_pose(&pattern, pose);
    }
}

#[test]
fn diagonal_layout_decodes_through_rotation() {
    let pattern = diagonal();
    for (index, theta) in [0.12, 0.55, -0.31, 0.79].into_iter().enumerate() {
        let pose = PatternPose::new(index as f64 * 61.0 + 2.5, index as f64 * 47.0 - 1.5, theta);
        assert_decodes_to_pose(&pattern, pose);
    }
}

#[test]
fn diagonal_layout_decodes_at_45_degrees() {
    // The pose this layout exists for: diamond squares, upright code grid.
    let pattern = diagonal();
    let quarter = std::f64::consts::FRAC_PI_4;
    for step in 0..4 {
        let pose = PatternPose::new(step as f64 * 53.0 + 1.9, step as f64 * -37.0 + 2.7, quarter);
        assert_decodes_to_pose(&pattern, pose);
    }
}

#[test]
fn the_layouts_do_not_decode_each_other() {
    // The layouts put their coding sites in different places, so reading one
    // with the other's rule must fail or land somewhere else -- never quietly
    // agree, which would mean the layout parameter was doing nothing.
    let pose = PatternPose::new(311.0 + 3.3, -177.0 - 5.7, 0.0);
    let pattern = diagonal();
    let image = pattern.render(SIZE, SIZE, &pose);
    let detection = detect(&image);

    let right = extract_code_with_layout(&detection, image.as_slice(), ORDER, CodeLayout::Diagonals)
        .expect("the matching layout must decode");
    let (want_i, want_j) = expected_square(&pose);
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
            "the lattice-axis rule reproduced the diagonal decode, so the layout \
             parameter is not actually being used"
        );
    }
}

#[test]
fn diagonal_layout_recovers_sub_square_position() {
    // Coarse code plus fine phase, for the diagonal layout: the absolute
    // position must be continuous, not quantized to the square grid. This is
    // the test that would catch a half-period slip in the (u, v) -> (i, j)
    // conversion, which a square-index check alone can absorb.
    let pattern = diagonal();
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
            CodeLayout::Diagonals,
        )
        .unwrap_or_else(|e| panic!("solve failed at {pose:?}: {e}"));

        let period = 3.0 * pattern.code().len() as f64 * SQUARE;
        let error_x = (recovered.x - -pose.x).rem_euclid(period);
        let error_y = (recovered.y - -pose.y).rem_euclid(period);
        let error_x = error_x.min(period - error_x);
        let error_y = error_y.min(period - error_y);
        worst = worst.max(error_x).max(error_y);
    }
    assert!(
        worst < 0.25,
        "worst absolute position error {worst:.4} px -- the fine phase is not \
         being combined with the diagonal code correctly"
    );
}
