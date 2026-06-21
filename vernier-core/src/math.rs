pub type Scalar = f64;
pub type Vec2 = nalgebra::Vector2<Scalar>;
pub type Vec3 = nalgebra::Vector3<Scalar>;
pub type Mat3 = nalgebra::Matrix3<Scalar>;
pub type Mat4 = nalgebra::Matrix4<Scalar>;

/// Normalizes an angle (in radians) to the half-open interval `(-pi, pi]`,
/// matching the C++ `vernier::angleInPiPi` helper used by [`crate::Pose`].
pub fn angle_in_pi_pi(mut angle: Scalar) -> Scalar {
    use std::f64::consts::{PI, TAU};
    while angle > PI {
        angle -= TAU;
    }
    while angle <= -PI {
        angle += TAU;
    }
    angle
}
