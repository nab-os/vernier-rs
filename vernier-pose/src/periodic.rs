//! Phase-only pose estimation from fitted phase planes (André et al. 2020/2021,
//! 2022 IJCV Eq. 1): the fine, sub-period regime.
//!
//! Given the two fitted phase planes (one per pattern direction), the in-plane
//! relations are:
//!
//! - `x = (φ1 / 2π)·λ + k1·λ`   (sub-period part + integer period order)
//! - `y = (φ2 / 2π)·λ + k2·λ`
//! - `α = atan2(b1, a1) + k3·(π/2)`
//!
//! where `φi = ci` is the plane constant (center phase) of direction `i`, `λ` is
//! the physical period, `(ai, bi)` are the plane gradients, and `k1, k2, k3` are
//! the unknown integer orders/quadrant resolved by the absolute decode. This
//! module computes the **fine** part (the `(φ/2π)·λ` terms and the raw
//! orientation); the `k·λ` and `k3·π/2` terms come from
//! [`absolute`](crate::absolute).
//!
//! ## What changed from the earlier version
//!
//! Orientation now comes from the *plane gradients* `atan2(b1, a1)`, not from a
//! bare `atan2` of integer bin indices — this is the actual high-resolution
//! angle (André et al. 2021, Eq. 3). And position comes from the fitted center
//! phase `c`, which averages the redundant phase over the whole image, rather
//! than one bin's value.

use vernier_core::Pose;
use vernier_core::scalar::consts::TAU;
use vernier_detection::PhasePlane;

use crate::Calibration;

/// Estimates the fine (sub-period) pose from the two direction phase planes.
///
/// `plane1` and `plane2` are the fitted planes for the two perpendicular
/// pattern directions. Returns a [`Pose`] whose translation is correct **modulo
/// `calib.period`** and whose orientation is the raw in-image angle (before
/// quadrant disambiguation). The integer period orders and quadrant are applied
/// by the absolute path.
pub fn estimate(plane1: &PhasePlane, plane2: &PhasePlane, calib: &Calibration) -> Pose {
    // Sub-period displacements from the center phases (mod one period).
    let x = (plane1.c / TAU) * calib.period;
    let y = (plane2.c / TAU) * calib.period;

    // Orientation from the first plane's gradients (high-resolution angle).
    let theta = plane1.orientation();

    Pose::new(x, y, theta)
}

/// Single-direction fine estimate, when only one phase plane is available.
///
/// Recovers one translational component and the orientation; the orthogonal
/// component is left at zero. Useful for 1D periodic patterns or partial
/// detection.
pub fn estimate_single(plane: &PhasePlane, calib: &Calibration) -> Pose {
    let theta = plane.orientation();
    let disp = (plane.c / TAU) * calib.period;
    Pose::new(disp * theta.cos(), disp * theta.sin(), theta)
}

#[cfg(test)]
mod tests {
    use super::*;
    use vernier_core::Real;
    use vernier_core::scalar::consts::PI;
    use vernier_detection::PhasePlane;

    fn plane(a: Real, b: Real, c: Real) -> PhasePlane {
        PhasePlane { a, b, c }
    }

    #[test]
    fn zero_phase_zero_position() {
        let calib = Calibration::new(10.0, 32, 32);
        // Horizontal direction (gradient along x), zero center phase.
        let p1 = plane(0.5, 0.0, 0.0);
        let p2 = plane(0.0, 0.5, 0.0);
        let pose = estimate(&p1, &p2, &calib);
        assert!(pose.x.abs() < 1e-6 && pose.y.abs() < 1e-6);
        assert!(pose.theta.abs() < 1e-6);
    }

    #[test]
    fn half_period_phase_gives_half_period_shift() {
        let calib = Calibration::new(10.0, 32, 32);
        // c = π is half of 2π -> x = period/2 = 5.0.
        let p1 = plane(0.5, 0.0, PI);
        let p2 = plane(0.0, 0.5, 0.0);
        let pose = estimate(&p1, &p2, &calib);
        assert!((pose.x - 5.0).abs() < 1e-4, "x={}", pose.x);
    }

    #[test]
    fn orientation_from_gradients() {
        let calib = Calibration::new(10.0, 32, 32);
        // Equal a and b -> 45 degrees.
        let p1 = plane(0.3, 0.3, 0.0);
        let p2 = plane(-0.3, 0.3, 0.0);
        let pose = estimate(&p1, &p2, &calib);
        assert!((pose.theta - PI / 4.0).abs() < 1e-4, "theta={}", pose.theta);
    }
}
