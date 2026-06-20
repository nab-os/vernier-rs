use bytemuck::{Pod, Zeroable};

use crate::scalar::Real;

#[repr(C)]
#[derive(Clone, Copy, PartialEq, Pod, Zeroable, Debug, Default)]
pub struct Complex32 {
    pub re: f32,
    pub im: f32,
}

impl Complex32 {
    pub const ZERO: Self = Self { re: 0.0, im: 0.0 };
    pub const ONE: Self = Self { re: 1.0, im: 0.0 };

    #[inline]
    pub const fn new(re: f32, im: f32) -> Self {
        Self { re, im }
    }

    #[inline]
    pub const fn from_real(re: f32) -> Self {
        Self { re, im: 0.0 }
    }

    #[inline]
    pub fn from_phase(theta: Real) -> Self {
        let (s, c) = theta.sin_cos();
        Self {
            re: c as f32,
            im: s as f32,
        }
    }

    #[inline]
    pub fn norm_sqr(self) -> Real {
        let re = self.re as Real;
        let im = self.im as Real;
        re * re + im * im
    }

    #[inline]
    pub fn norm(self) -> Real {
        self.norm_sqr().sqrt()
    }

    #[inline]
    pub fn arg(self) -> Real {
        (self.im as Real).atan2(self.re as Real)
    }

    #[inline]
    pub const fn conj(self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }
}

impl core::ops::Add for Complex32 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self::new(self.re + rhs.re, self.im + rhs.im)
    }
}

impl core::ops::Sub for Complex32 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.re - rhs.re, self.im - rhs.im)
    }
}

impl core::ops::Mul for Complex32 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self::new(
            self.re * rhs.re - self.im * rhs.im,
            self.re * rhs.im + self.im * rhs.re,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_is_two_contiguous_f32() {
        assert_eq!(core::mem::size_of::<Complex32>(), 8);
        assert_eq!(core::mem::align_of::<Complex32>(), 4);
    }

    #[test]
    fn slice_casts_to_bytes() {
        let data = [Complex32::new(1.0, 2.0), Complex32::new(3.0, 4.0)];
        let bytes: &[u8] = bytemuck::cast_slice(&data);
        assert_eq!(bytes.len(), 16);
    }

    #[test]
    fn arg_of_i_is_half_pi() {
        let z = Complex32::new(0.0, 1.0);
        assert!((z.arg() - crate::scalar::consts::PI / 2.0).abs() < 1e-6);
    }
}
