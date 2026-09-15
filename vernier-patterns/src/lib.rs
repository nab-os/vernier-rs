//! # vernier-patterns
//!
//! Generates and renders the calibrated patterns the library measures. Because
//! it produces [`GrayImage`](vernier_core::GrayImage)s, it's the source of
//! ground-truth test data: render at a known pose, run detection + pose
//! estimation, and check the recovered pose matches.
//!
//! - [`periodic`] — a sinusoidal grid at a known translation/orientation; the
//!   workhorse for validating the pipeline end to end.
//! - [`megarena`] — the absolute LFSR-encoded pattern: a 3-period-per-bit
//!   carrier gated by a maximal [`lfsr`] sequence, with one corner removed to
//!   fix the π/2 rotation ambiguity.
//! - [`checkerboard`] — the same [`lfsr`] code on a 50/50 black-and-white
//!   carrier: bits are written by inverting one square per 3×3 supercell
//!   instead of removing dots, which keeps the fill balanced and buys ~1.6× the
//!   carrier amplitude of a megarena.
//! - [`lfsr`] — the maximal-length sequence generator shared by the renderer and
//!   decoder.
//! - [`render`] — shared rasterization helpers.
//! - [`qrcode`], [`stamp`] — stubs; the interface is fixed, the encoding isn't.

pub mod checkerboard;
pub mod lfsr;
pub mod megarena;
pub mod periodic;
pub mod qrcode;
pub mod render;
pub mod stamp;

#[cfg(feature = "vulkan")]
pub use vernier_render::{CameraModel, PatternRenderer, RenderParams};

use vernier_core::Real;

/// The pose at which to render a pattern — the ground truth a detector should
/// recover. Distinct from [`vernier_core::Pose`] only in intent (generation
/// input vs estimation output), kept separate so data flow stays legible.
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
