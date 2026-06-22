//! Minimal RGB drawing for debug overlays (peaks on the spectrum, decoded cells
//! and bits on the image). Hand-rolled primitives on an `Rgb` buffer — no
//! drawing-library dependency; the `image` crate is used only to encode the PNG.

use std::path::Path;

/// A mutable RGB canvas, row-major, 3 bytes per pixel.
pub struct Canvas {
    pub width: usize,
    pub height: usize,
    pub pixels: Vec<u8>,
}

impl Canvas {
    /// Builds a canvas from a grayscale `0.0..=1.0` buffer (the background image).
    pub fn from_gray(width: usize, height: usize, gray: &[f64]) -> Self {
        let mut pixels = Vec::with_capacity(width * height * 3);
        for &v in gray {
            let g = (v.clamp(0.0, 1.0) * 255.0).round() as u8;
            pixels.push(g);
            pixels.push(g);
            pixels.push(g);
        }
        Self {
            width,
            height,
            pixels,
        }
    }

    #[inline]
    fn put(&mut self, x: isize, y: isize, rgb: [u8; 3]) {
        if x < 0 || y < 0 || x as usize >= self.width || y as usize >= self.height {
            return;
        }
        let i = (y as usize * self.width + x as usize) * 3;
        self.pixels[i] = rgb[0];
        self.pixels[i + 1] = rgb[1];
        self.pixels[i + 2] = rgb[2];
    }

    /// Blends a color over a pixel with alpha `a` in 0..=1 (for translucent fills).
    #[inline]
    fn blend(&mut self, x: isize, y: isize, rgb: [u8; 3], a: f64) {
        if x < 0 || y < 0 || x as usize >= self.width || y as usize >= self.height {
            return;
        }
        let i = (y as usize * self.width + x as usize) * 3;
        for k in 0..3 {
            let bg = self.pixels[i + k] as f64;
            let fg = rgb[k] as f64;
            self.pixels[i + k] = (bg * (1.0 - a) + fg * a).round().clamp(0.0, 255.0) as u8;
        }
    }

    /// Draws a hollow circle of the given radius (Bresenham-ish, thick by 1px).
    pub fn circle(&mut self, cx: isize, cy: isize, r: isize, rgb: [u8; 3]) {
        let mut x = r;
        let mut y = 0isize;
        let mut err = 0isize;
        while x >= y {
            for (dx, dy) in [
                (x, y),
                (y, x),
                (-x, y),
                (-y, x),
                (x, -y),
                (y, -x),
                (-x, -y),
                (-y, -x),
            ] {
                self.put(cx + dx, cy + dy, rgb);
            }
            y += 1;
            if err <= 0 {
                err += 2 * y + 1;
            }
            if err > 0 {
                x -= 1;
                err -= 2 * x + 1;
            }
        }
    }

    /// Draws a crosshair centered at (cx, cy) with arm length `len`.
    pub fn cross(&mut self, cx: isize, cy: isize, len: isize, rgb: [u8; 3]) {
        for d in -len..=len {
            self.put(cx + d, cy, rgb);
            self.put(cx, cy + d, rgb);
        }
    }

    /// Draws a line (Bresenham).
    pub fn line(&mut self, x0: isize, y0: isize, x1: isize, y1: isize, rgb: [u8; 3]) {
        let dx = (x1 - x0).abs();
        let dy = -(y1 - y0).abs();
        let sx = if x0 < x1 { 1 } else { -1 };
        let sy = if y0 < y1 { 1 } else { -1 };
        let mut err = dx + dy;
        let (mut x, mut y) = (x0, y0);
        loop {
            self.put(x, y, rgb);
            if x == x1 && y == y1 {
                break;
            }
            let e2 = 2 * err;
            if e2 >= dy {
                err += dy;
                x += sx;
            }
            if e2 <= dx {
                err += dx;
                y += sy;
            }
        }
    }

    /// Fills a small square centered at (cx, cy), half-size `h`, with alpha blend.
    pub fn fill_square(&mut self, cx: isize, cy: isize, h: isize, rgb: [u8; 3], alpha: f64) {
        for dy in -h..=h {
            for dx in -h..=h {
                self.blend(cx + dx, cy + dy, rgb, alpha);
            }
        }
    }

    /// Saves the canvas as an RGB PNG.
    pub fn save_png(&self, path: &Path) -> Result<(), String> {
        let buf =
            image::RgbImage::from_raw(self.width as u32, self.height as u32, self.pixels.clone())
                .ok_or_else(|| "canvas size mismatch".to_string())?;
        buf.save(path)
            .map_err(|e| format!("failed to save {}: {e}", path.display()))
    }
}

/// A few named colors for overlays.
pub mod color {
    pub const RED: [u8; 3] = [255, 40, 40];
    pub const GREEN: [u8; 3] = [40, 220, 40];
    pub const BLUE: [u8; 3] = [60, 120, 255];
    pub const YELLOW: [u8; 3] = [255, 220, 40];
    pub const CYAN: [u8; 3] = [40, 220, 220];
}
