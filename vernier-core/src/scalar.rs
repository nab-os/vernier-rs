/// Host-side computation scalar. `f64` everywhere the C++ reference uses
/// `double`: phase maps, plane fits, pose assembly. Device buffers and the FFT
/// stay `f32` ([`Complex32`](crate::Complex32)) — per-pixel carrier phase only
/// needs ~1e-6 rad — but everything downstream of `arg()` runs in `f64` so the
/// absolute position keeps the method's 10^8 range-to-resolution ratio (an f32
/// pose at 3.5 cm quantizes at ~4 nm).
pub type Real = f64;

pub mod consts {
    use super::Real;

    pub const PI: Real = std::f64::consts::PI;
    pub const TAU: Real = std::f64::consts::TAU;
    pub const SQRT_2: Real = std::f64::consts::SQRT_2;
}
