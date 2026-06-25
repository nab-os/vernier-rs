//! Periodic (sinusoidal grid) pattern generation at a known pose.
//!
//! The validation workhorse: a cosine of fixed spatial period, rendered at a
//! known translation and orientation to give a ground-truth image the detector
//! must recover. It's the deliberate inverse of
//! `vernier-pose::periodic::estimate` — `theta` rotates the pattern so the
//! fundamental peak appears at that angle in the frequency domain, and a
//! translation shifts the cosine's phase by the matching fraction of 2π.

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

    /// Renders this pattern on the GPU at `pose` (requires the `vulkan` feature).
    /// Rasterises through `vernier-render`, evaluating `(1+cos_x)(1+cos_y)/4` in
    /// the fragment shader. `camera` supplies the µm/pixel used to convert the
    /// pose's pixel units into µm.
    #[cfg(feature = "vulkan")]
    pub fn render_gpu(
        &self,
        renderer: &vernier_render::PatternRenderer,
        camera: &vernier_render::CameraModel,
        width: usize,
        height: usize,
        pose: &PatternPose,
    ) -> GrayImage {
        let period_um = self.period_px as f32 * camera.pixel_size;
        let pose_x_um = pose.x as f32 * camera.pixel_size;
        let pose_y_um = pose.y as f32 * camera.pixel_size;

        // Conservative bounding box of the visible area in period-cell units.
        // The half-diagonal (in µm) covers any rotation.
        let half_diag_um =
            ((width * width + height * height) as f32).sqrt() * 0.5 * camera.pixel_size + period_um;

        let col_min = ((pose_x_um - half_diag_um) / period_um).floor() as i64;
        let col_max = ((pose_x_um + half_diag_um) / period_um).ceil() as i64;
        let row_min = ((pose_y_um - half_diag_um) / period_um).floor() as i64;
        let row_max = ((pose_y_um + half_diag_um) / period_um).ceil() as i64;

        let mut cell_origins = Vec::new();
        for col in col_min..=col_max {
            for row in row_min..=row_max {
                cell_origins.push([col as f32 * period_um, row as f32 * period_um]);
            }
        }

        renderer.render_quads(
            &cell_origins,
            &vernier_render::RenderParams {
                width,
                height,
                period_um,
                pixel_size: camera.pixel_size,
                pose_x_um,
                pose_y_um,
                alpha: pose.theta as f32,
            },
        )
    }

    /// Renders this pattern at `pose` into a `width × height` image. Stripes run
    /// perpendicular to the pattern-frame X axis; the `0.5 +` offset keeps
    /// intensities in `[0, 1]` like a real camera image.
    pub fn render(&self, width: usize, height: usize, pose: &PatternPose) -> GrayImage {
        let center_x = width as Real / 2.0;
        let center_y = height as Real / 2.0;

        render_with(width, height, |px, py| {
            let (x_pattern, y_pattern) = into_pattern_frame(px, py, center_x, center_y, pose.theta);
            self.intensity_at(x_pattern - pose.x, y_pattern - pose.y)
        })
    }

    /// Intensity at continuous pattern-frame coordinates `(x, y)` (no pose
    /// applied). This CPU model is the 1-D stripe carrier
    /// `0.5 + 0.5·cos(2π·x/period)` (independent of `y`); the C++ reference uses
    /// the 2-D grid product `(1+cos_x)(1+cos_y)/4`. The divergence is inherited
    /// from the existing renderer.
    pub fn intensity_at(&self, x: Real, _y: Real) -> Real {
        0.5 + 0.5 * (TAU * x / self.period_px).cos()
    }

    /// Carrier phase along the X axis (radians), matching C++ `getPhase1`.
    pub fn phase1_at(&self, x: Real, _y: Real) -> Real {
        TAU * x / self.period_px
    }

    /// Carrier phase along the Y axis (radians), matching C++ `getPhase2`.
    pub fn phase2_at(&self, _x: Real, y: Real) -> Real {
        TAU * y / self.period_px
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
