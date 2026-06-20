//! # vernier-pose
//!
//! Turns detected phase planes ([`PhasePlane`](vernier_detection::PhasePlane)) into a
//! [`Pose`](vernier_core::Pose). This is the *algorithm* layer; the `Pose`
//! *type* lives in `vernier-core`. Same word, two crates, deliberately.
//!
//! Two estimation paths mirror the two regimes in the Vernier papers:
//!
//! - [`periodic`] — phase-only. Fine (sub-period) translation and orientation
//!   from a single fundamental peak, but the absolute position is **ambiguous**
//!   modulo the pattern period (you know where you are *within* a cell, not
//!   *which* cell).
//! - [`absolute`] — coarse code decode + fine phase. Resolves which period you
//!   are in to give an unambiguous absolute `(x, y, θ)`. The coarse decode is
//!   pattern-specific (megarena) and is stubbed at its boundary here.
//!
//! Both are plain functions over already-detected features — no backend
//! generics needed at this layer, because by the time we have phase planes the
//! device work is done and we are doing small host-side arithmetic.

pub mod absolute;
pub mod periodic;

use vernier_core::Real;

/// Pattern calibration: the physical scale that converts image-domain
/// measurements into real-world units.
///
/// Without this, detection yields only pixels and radians; calibration is what
/// earns the "nanometer over centimeters" claim.
#[derive(Clone, Copy, Debug)]
pub struct Calibration {
    /// Spatial period of the pattern in physical units (e.g. micrometres per
    /// pattern cell).
    pub period: Real,
    /// Image width in pixels (needed to convert a frequency bin into cycles per
    /// pixel, then into the observed period).
    pub image_width: usize,
    /// Image height in pixels.
    pub image_height: usize,
}

impl Calibration {
    /// Creates a calibration.
    pub fn new(period: Real, image_width: usize, image_height: usize) -> Self {
        Self {
            period,
            image_width,
            image_height,
        }
    }
}
