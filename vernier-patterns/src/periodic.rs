//! Periodic (sinusoidal grid) pattern generation at a known pose.
//!
//! This is the validation workhorse. A periodic pattern is a cosine of a fixed
//! spatial period; rendering it at a known translation and orientation gives a
//! ground-truth image whose pose the detector must recover. The math here is the
//! deliberate inverse of `vernier-pose::periodic::estimate`:
//!
//! - **orientation** `theta` rotates the pattern in the image plane, so the
//!   fundamental peak appears at angle `theta` in the frequency domain.
//! - **translation** shifts the cosine's argument; a shift of a fraction of the
//!   period advances the fundamental's phase by that fraction of 2π — which is
//!   exactly what the estimator reads back.
//!
//! Keeping these two consistent is what makes a render -> detect -> pose round
//! trip close to within numerical tolerance.

use vernier_core::scalar::consts::TAU;
use vernier_core::{GrayImage, Real};

use crate::PatternPose;
use crate::render::{into_pattern_frame, render_with};

/// Parameters of a periodic pattern.
#[derive(Clone, Copy, Debug)]
pub struct Periodic {
    /// Spatial period in pixels (distance between bright stripes).
    pub period_px: Real,
}

impl Periodic {
    /// Creates a periodic pattern with the given pixel period.
    pub fn new(period_px: Real) -> Self {
        Self { period_px }
    }

    /// Renders this pattern at `pose` into a `width x height` image.
    ///
    /// The pattern stripes run perpendicular to the pattern-frame X axis, so the
    /// intensity is `0.5 + 0.5·cos(2π·(x' - x_shift)/period)`, where `x'` is the
    /// pixel mapped into the rotated pattern frame and `x_shift` is the
    /// translation. The `0.5 +` offset keeps intensities in `[0, 1]` like a real
    /// camera image (the detector works on the AC part either way).
    pub fn render(&self, width: usize, height: usize, pose: &PatternPose) -> GrayImage {
        let center_x = width as Real / 2.0;
        let center_y = height as Real / 2.0;
        let period = self.period_px;

        render_with(width, height, |px, py| {
            let (x_pattern, _y_pattern) = into_pattern_frame(px, py, center_x, center_y, pose.theta);
            let phase = TAU * (x_pattern - pose.x) / period;
            0.5 + 0.5 * phase.cos()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_produces_expected_dimensions() {
        let pat = Periodic::new(8.0);
        let img = pat.render(32, 16, &PatternPose::IDENTITY);
        assert_eq!(img.width(), 32);
        assert_eq!(img.height(), 16);
    }

    #[test]
    fn intensities_stay_in_unit_range() {
        let pat = Periodic::new(8.0);
        let img = pat.render(32, 32, &PatternPose::new(1.5, 0.0, 0.3));
        for &v in img.as_slice() {
            assert!((0.0..=1.0).contains(&v), "intensity {v} out of [0,1]");
        }
    }

    #[test]
    fn period_is_respected_along_x() {
        // At identity pose, points one period apart on the center row should
        // have (nearly) equal intensity.
        let period = 8.0;
        let pat = Periodic::new(period);
        let img = pat.render(64, 64, &PatternPose::IDENTITY);
        let row = 32;
        let a = img.get(row, 20);
        let b = img.get(row, 20 + period as usize);
        assert!(
            (a - b).abs() < 1e-3,
            "a={a} b={b} should match one period apart"
        );
    }
}
