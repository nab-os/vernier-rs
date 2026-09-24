//! The browser side: painting an intensity field onto a `<canvas>` and handing
//! it back as a PNG.
//!
//! The pattern is drawn at its true pixel size and scaled by CSS, so a 2048 px
//! render stays a 2048 px PNG however small the preview is on screen.

use wasm_bindgen::{Clamped, JsCast};
use web_sys::{CanvasRenderingContext2d, HtmlAnchorElement, HtmlCanvasElement, ImageData};

/// DOM id of the preview canvas, shared by the component and the painters.
pub const CANVAS_ID: &str = "vernier-pattern-canvas";

fn document() -> Result<web_sys::Document, String> {
    web_sys::window()
        .and_then(|window| window.document())
        .ok_or_else(|| "no document — not running in a browser".to_string())
}

fn canvas(id: &str) -> Result<HtmlCanvasElement, String> {
    document()?
        .get_element_by_id(id)
        .ok_or_else(|| format!("canvas #{id} is not in the DOM"))?
        .dyn_into::<HtmlCanvasElement>()
        .map_err(|_| format!("#{id} is not a <canvas>"))
}

/// Milliseconds from the page's monotonic clock, for the render timing readout.
pub fn now_ms() -> f64 {
    web_sys::window()
        .and_then(|window| window.performance())
        .map(|performance| performance.now())
        .unwrap_or(0.0)
}

fn context_of(canvas: &HtmlCanvasElement) -> Result<CanvasRenderingContext2d, String> {
    canvas
        .get_context("2d")
        .map_err(|_| "canvas 2d context unavailable".to_string())?
        .ok_or_else(|| "canvas 2d context unavailable".to_string())?
        .dyn_into::<CanvasRenderingContext2d>()
        .map_err(|_| "canvas 2d context has an unexpected type".to_string())
}

/// Resizes the canvas to the image and blits `rgba` into it. `rgba` must hold
/// `width * height * 4` bytes.
pub fn paint(width: usize, height: usize, rgba: &[u8]) -> Result<(), String> {
    paint_to(CANVAS_ID, width, height, rgba)
}

/// Same, for any canvas on the page — the explorer has six of them.
pub fn paint_to(id: &str, width: usize, height: usize, rgba: &[u8]) -> Result<(), String> {
    let canvas = canvas(id)?;
    canvas.set_width(width as u32);
    canvas.set_height(height as u32);

    let context = context_of(&canvas)?;

    let image = ImageData::new_with_u8_clamped_array_and_sh(
        Clamped(rgba),
        width as u32,
        height as u32,
    )
    .map_err(|_| "could not build ImageData for the render".to_string())?;

    context
        .put_image_data(&image, 0.0, 0.0)
        .map_err(|_| "could not paint the render onto the canvas".to_string())
}

/// Saves the current canvas contents as a PNG through a synthetic download.
pub fn download_png(file_stem: &str) -> Result<(), String> {
    let canvas = canvas(CANVAS_ID)?;
    let url = canvas
        .to_data_url_with_type("image/png")
        .map_err(|_| "could not encode the canvas as PNG".to_string())?;

    let anchor = document()?
        .create_element("a")
        .map_err(|_| "could not create the download link".to_string())?
        .dyn_into::<HtmlAnchorElement>()
        .map_err(|_| "could not create the download link".to_string())?;
    anchor.set_href(&url);
    anchor.set_download(&format!("{file_stem}.png"));
    anchor.click();
    Ok(())
}

// ---------------------------------------------------------------------------
// Explorer stage panels
// ---------------------------------------------------------------------------

/// Maps values to grey levels, stretched between the extremes actually present
/// so every stage uses the full range whatever its units are.
///
/// `log` compresses with `ln(1 + v)` first. On a coded pattern the carrier
/// peaks stand orders of magnitude above the sidebands, so the difference is
/// between seeing the spectrum and seeing two white dots on black.
pub fn grey_levels(data: &[f32], log: bool) -> Vec<u8> {
    let compress = |v: f32| if log { (1.0 + v.max(0.0)).ln() } else { v };

    let (mut low, mut high) = (f32::INFINITY, f32::NEG_INFINITY);
    for &v in data {
        let v = compress(v);
        if v.is_finite() {
            low = low.min(v);
            high = high.max(v);
        }
    }
    let span = if (high - low).abs() < 1e-12 { 1.0 } else { high - low };

    data.iter()
        .map(|&v| (((compress(v) - low) / span) * 255.0).clamp(0.0, 255.0) as u8)
        .collect()
}

/// One wrapped phase as a level, `−π` to `π` over the full range.
fn phase_level(radians: f32) -> u8 {
    let turn = (radians as f64 / std::f64::consts::TAU).rem_euclid(1.0);
    (turn * 255.0).clamp(0.0, 255.0) as u8
}

/// Paints a square scalar stage onto the canvas with the given id.
pub fn paint_grey(id: &str, size: usize, data: &[f32], log: bool) -> Result<(), String> {
    let levels = grey_levels(data, log);
    let mut rgba = Vec::with_capacity(size * size * 4);
    for level in levels {
        rgba.extend_from_slice(&[level, level, level, 255]);
    }
    paint_to(id, size, size, &rgba)
}

/// Paints both wrapped phases into one full-resolution image, one per colour
/// channel: direction 1 in red, direction 2 in green.
///
/// The two phases live on the same pixels, so splitting them by channel shows
/// them where they actually are — each channel keeps its own sawtooth, and the
/// lattice their sum makes (yellow corners, red and green fringes crossing)
/// is the pattern's own grid, drawn by the measurement rather than over it.
pub fn paint_phase_channels(
    id: &str,
    size: usize,
    first: &[f32],
    second: &[f32],
) -> Result<(), String> {
    let mut rgba = Vec::with_capacity(size * size * 4);
    for pixel in 0..size * size {
        rgba.extend_from_slice(&[
            phase_level(first[pixel]),
            phase_level(second[pixel]),
            0,
            255,
        ]);
    }
    paint_to(id, size, size, &rgba)
}

/// Colour of a thumbnail cell no square landed in. Distinct from a black
/// square, which is a measurement and not padding.
const THUMBNAIL_PADDING: [u8; 4] = [20, 24, 33, 255];

/// Paints an extracted thumbnail one pixel per pattern square. The CSS scales
/// it up with `image-rendering: pixelated`, so a square stays a square.
pub fn paint_thumbnail(
    id: &str,
    side: usize,
    levels: &[u8],
    present: &[bool],
) -> Result<(), String> {
    let mut rgba = Vec::with_capacity(side * side * 4);
    for cell in 0..side * side {
        if present[cell] {
            let level = levels[cell];
            rgba.extend_from_slice(&[level, level, level, 255]);
        } else {
            rgba.extend_from_slice(&THUMBNAIL_PADDING);
        }
    }
    paint_to(id, side, side, &rgba)
}

/// Blanks a stage, for when there is nothing to show in it.
pub fn clear(id: &str) -> Result<(), String> {
    let canvas = canvas(id)?;
    let context = context_of(&canvas)?;
    context.clear_rect(0.0, 0.0, canvas.width() as f64, canvas.height() as f64);
    Ok(())
}
