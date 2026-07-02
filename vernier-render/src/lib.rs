//! # vernier-render
//!
//! Vulkan rasterisation renderer for the calibrated patterns in
//! `vernier-patterns`.
//!
//! The renderer expresses each pattern as a list of period-cell origins in µm
//! and draws one quad per cell through a full graphics pipeline (vertex +
//! fragment shader, rasterisation, offscreen framebuffer).  The fragment
//! shader evaluates the same `(1+cos_x)(1+cos_y)/4` cosine-carrier formula as
//! the C++ reference `getIntensity`, so the produced images are directly
//! comparable.
//!
//! ## Camera model
//!
//! [`CameraModel`] carries the `pixel_size` (µm/pixel) that converts between
//! the pattern's physical coordinate system and the image's pixel grid.
//! Together with the pattern's [`PatternPose`](vernier_patterns::PatternPose)
//! (translation in pixels, rotation in radians), it defines the full
//! orthographic projection:
//!
//! ```text
//! col = (cos θ · (x − pose_x_µm) − sin θ · (y − pose_y_µm)) / pixel_size + width/2
//! row = (sin θ · (x − pose_x_µm) + cos θ · (y − pose_y_µm)) / pixel_size + height/2
//! ```
//!
//! where `pose_x_µm = pose.x * pixel_size` etc.

mod renderer;

pub use renderer::PatternRenderer;

/// Physical camera parameters for converting between µm and pixels.
#[derive(Clone, Copy, Debug)]
pub struct CameraModel {
    /// Pixel size in µm/pixel (i.e. one pixel covers this many micrometres).
    pub pixel_size: f32,
}

/// All per-render parameters passed to [`PatternRenderer::render_quads`].
#[derive(Clone, Copy, Debug)]
pub struct RenderParams {
    /// Output image width in pixels.
    pub width: usize,
    /// Output image height in pixels.
    pub height: usize,
    /// Spatial period in µm.
    pub period_um: f32,
    /// Pixel size in µm/pixel.
    pub pixel_size: f32,
    /// X component of the image centre in pattern space (µm).
    pub pose_x_um: f32,
    /// Y component of the image centre in pattern space (µm).
    pub pose_y_um: f32,
    /// Pattern orientation in radians.
    pub alpha: f32,
}
