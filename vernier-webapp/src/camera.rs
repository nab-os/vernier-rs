//! The virtual camera the explorer looks through.
//!
//! `PatternPose` carries x, y and θ, so out-of-plane motion has no path through
//! it. The explorer keeps its own six-freedom pose and does the projection here,
//! sampling the pattern through `PatternSettings::sampler`. Nothing about the
//! pattern is reimplemented: this only decides which point of the plane each
//! pixel is looking at.

use nalgebra::{Matrix3, Matrix4, Vector3};
use vernier_core::{GrayImage, Real};

use crate::pattern::Sampler;

/// Focal length in pixels. Fixed, with the distance doing the work: one number
/// to move rather than two that trade off against each other.
pub const FOCAL: Real = 1000.0;

/// Carrier fringes per image pixel at the home distance. Eight samples per
/// period puts the peaks at `size / 8` bins from the centre — comfortably past
/// the rejected low-frequency disc and well short of Nyquist.
pub const TARGET_SAMPLES_PER_PERIOD: Real = 8.0;

/// Camera distance putting [`TARGET_SAMPLES_PER_PERIOD`] pixels on one carrier
/// period. The magnification is `FOCAL / z`, so this inverts it.
pub fn home_distance(carrier_period_px: Real) -> Real {
    FOCAL * carrier_period_px / TARGET_SAMPLES_PER_PERIOD
}

/// A full six-degree-of-freedom pose, in the convention the C++ library uses:
/// `cTp = transl(0,0,z) · rotz(α) · roty(β) · rotx(γ) · transl(x,y,0)`.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Pose6 {
    /// Translation in the pattern plane, in pattern pixels.
    pub x: Real,
    pub y: Real,
    /// Camera distance, in the same units as [`FOCAL`].
    pub z: Real,
    /// Rotation about Z — the in-plane one — in radians.
    pub alpha: Real,
    /// Rotation about the intrinsic Y axis, in radians.
    pub beta: Real,
    /// Rotation about the intrinsic X axis, in radians.
    pub gamma: Real,
}

impl Pose6 {
    pub fn at_distance(z: Real) -> Self {
        Self { x: 0.0, y: 0.0, z, alpha: 0.0, beta: 0.0, gamma: 0.0 }
    }

    /// Image pixels per pattern pixel.
    pub fn magnification(&self) -> Real {
        FOCAL / self.z
    }

    /// Moves the pattern by a drag given in image pixels.
    ///
    /// The pose translates the pattern before the rotations are applied, so the
    /// screen delta has to be turned back by −α for the pattern to follow the
    /// cursor rather than sliding off at an angle.
    pub fn translate_by_drag(&mut self, dx: Real, dy: Real) {
        let scale = 1.0 / self.magnification();
        let (dx, dy) = (dx * scale, dy * scale);
        let (sa, ca) = self.alpha.sin_cos();
        self.x += ca * dx + sa * dy;
        self.y += -sa * dx + ca * dy;
    }

    fn camera_to_pattern(&self) -> Matrix4<Real> {
        let (sa, ca) = self.alpha.sin_cos();
        let (sb, cb) = self.beta.sin_cos();
        let (sg, cg) = self.gamma.sin_cos();

        #[rustfmt::skip]
        let tz = Matrix4::new(
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, self.z,
            0.0, 0.0, 0.0, 1.0,
        );
        #[rustfmt::skip]
        let rz = Matrix4::new(
            ca, -sa, 0.0, 0.0,
            sa,  ca, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        );
        #[rustfmt::skip]
        let ry = Matrix4::new(
             cb, 0.0, sb, 0.0,
            0.0, 1.0, 0.0, 0.0,
            -sb, 0.0, cb, 0.0,
            0.0, 0.0, 0.0, 1.0,
        );
        #[rustfmt::skip]
        let rx = Matrix4::new(
            1.0, 0.0, 0.0, 0.0,
            0.0,  cg, -sg, 0.0,
            0.0,  sg,  cg, 0.0,
            0.0, 0.0, 0.0, 1.0,
        );
        #[rustfmt::skip]
        let txy = Matrix4::new(
            1.0, 0.0, 0.0, self.x,
            0.0, 1.0, 0.0, self.y,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        );
        tz * rz * ry * rx * txy
    }

    /// Inverse of the homography taking the pattern plane to the image, or
    /// `None` when the pose degenerates — the plane seen exactly edge-on.
    ///
    /// The pattern lives at z = 0, so only columns 0, 1 and 3 of the pose
    /// matrix survive the projection; dropping column 2 is what turns the 3×4
    /// camera matrix into something square enough to invert.
    fn inverse_homography(&self, principal: Real) -> Option<Matrix3<Real>> {
        let m = self.camera_to_pattern();
        let mut h = Matrix3::zeros();
        for (col, src) in [0usize, 1, 3].into_iter().enumerate() {
            h[(0, col)] = FOCAL * m[(0, src)] + principal * m[(2, src)];
            h[(1, col)] = FOCAL * m[(1, src)] + principal * m[(2, src)];
            h[(2, col)] = m[(2, src)];
        }
        h.try_inverse()
    }
}

/// Renders what the camera sees into a `size × size` image.
///
/// `supersample` sub-samples each pixel edge: the coded patterns are binary, and
/// point-sampling their edges aliases into a shifted carrier phase.
pub fn render(
    sampler: &Sampler,
    pose: &Pose6,
    size: usize,
    supersample: u32,
) -> GrayImage {
    let mut out = vec![0.0f32; size * size];

    let Some(inverse) = pose.inverse_homography(size as Real / 2.0) else {
        // Edge-on: nothing projects, and a blank frame says so more plainly
        // than a panic would.
        return GrayImage::from_vec(size, size, out).expect("size × size buffer");
    };

    let steps = supersample.max(1) as usize;
    let step = 1.0 / steps as Real;
    let offset = step / 2.0;
    let weight = 1.0 / (steps * steps) as Real;

    for row in 0..size {
        for col in 0..size {
            let mut sum = 0.0;
            for sy in 0..steps {
                let py = row as Real + offset + sy as Real * step;
                for sx in 0..steps {
                    let px = col as Real + offset + sx as Real * step;
                    let p = inverse * Vector3::new(px, py, 1.0);
                    // A non-positive third coordinate is a point behind the
                    // camera: it has no image, so it contributes nothing.
                    if p.z <= 0.0 {
                        continue;
                    }
                    sum += sampler(p.x / p.z, p.y / p.z);
                }
            }
            out[row * size + col] = (sum * weight) as f32;
        }
    }

    GrayImage::from_vec(size, size, out).expect("size × size buffer")
}
