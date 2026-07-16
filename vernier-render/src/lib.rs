//! # vernier-render
//!
//! Vulkan rasterisation renderer for the patterns in `vernier-patterns`. Each
//! pattern becomes a list of period-cell origins (µm), drawn one quad per cell
//! through a vertex + fragment pipeline into an offscreen framebuffer. The
//! fragment shader uses the same `(1+cos_x)(1+cos_y)/4` carrier as the C++
//! reference `getIntensity`, so the output is directly comparable.
//!
//! [`CameraModel`] carries the `pixel_size` (µm/pixel) that, together with a
//! [`PatternPose`](vernier_patterns::PatternPose), defines the orthographic
//! projection:
//!
//! ```text
//! col = (cos θ · (x − pose_x_µm) − sin θ · (y − pose_y_µm)) / pixel_size + width/2
//! row = (sin θ · (x − pose_x_µm) + cos θ · (y − pose_y_µm)) / pixel_size + height/2
//! ```

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
