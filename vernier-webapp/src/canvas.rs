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

/// Resizes the canvas to the image and blits `rgba` into it. `rgba` must hold
/// `width * height * 4` bytes.
pub fn paint(width: usize, height: usize, rgba: &[u8]) -> Result<(), String> {
    let canvas = canvas(CANVAS_ID)?;
    canvas.set_width(width as u32);
    canvas.set_height(height as u32);

    let context = canvas
        .get_context("2d")
        .map_err(|_| "canvas 2d context unavailable".to_string())?
        .ok_or_else(|| "canvas 2d context unavailable".to_string())?
        .dyn_into::<CanvasRenderingContext2d>()
        .map_err(|_| "canvas 2d context has an unexpected type".to_string())?;

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
