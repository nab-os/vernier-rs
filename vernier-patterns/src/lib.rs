//! # vernier-patterns
//!
//! Generation and rendering of the calibrated patterns the library measures.
//! Produces [`GrayImage`](vernier_core::GrayImage)s, which is what makes this
//! crate the source of *ground-truth test data*: render a pattern at a known
//! pose, run detection + pose estimation, and assert the recovered pose matches.
//! Without this, the whole pipeline can only be tested on hand-rolled signals.
//!
//! ## What is implemented vs bounded
//!
//! - [`periodic`] — fully implemented. A sinusoidal/grid pattern at a known
//!   `(translation, orientation)` with well-defined math, the workhorse for
//!   validating the detection/pose pipeline end to end.
//! - [`megarena`] — fully implemented. The absolute LFSR-encoded pattern behind
//!   the "one nanometer over ten centimeters" result: a 3-period-per-bit carrier
//!   gated by a maximal [`lfsr`] sequence, with one corner removed to fix the
//!   π/2 rotation ambiguity. This is what gives the absolute path something real
//!   to decode.
//! - [`lfsr`] — the maximal-length sequence generator shared by the megarena
//!   renderer and (eventually) the decoder.
//! - [`render`] — shared rasterization helpers used by the generators.
//! - [`qrcode`], [`stamp`] — still bounded: these embed pattern-specific layouts
//!   whose exact construction this crate fixes the interface for and leaves the
//!   encoding as a documented boundary.

pub mod lfsr;
pub mod megarena;
pub mod periodic;
pub mod qrcode;
pub mod render;
pub mod stamp;

use vernier_core::Real;

/// The pose at which to *render* a pattern — the ground truth a detector should
/// recover. Distinct from [`vernier_core::Pose`] only in intent (input to
/// generation vs output of estimation); kept as its own type so the direction
/// of data flow is legible.
#[derive(Clone, Copy, Debug)]
pub struct PatternPose {
    /// Translation along X in physical units.
    pub x: Real,
    /// Translation along Y in physical units.
    pub y: Real,
    /// Orientation in radians.
    pub theta: Real,
}

impl PatternPose {
    /// The untranslated, unrotated reference pose.
    pub const IDENTITY: Self = Self {
        x: 0.0,
        y: 0.0,
        theta: 0.0,
    };

    /// Constructs a pattern pose.
    pub fn new(x: Real, y: Real, theta: Real) -> Self {
        Self { x, y, theta }
    }
}
