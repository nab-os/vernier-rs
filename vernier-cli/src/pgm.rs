//! Dependency-free grayscale image output (binary PGM, "P5").
//!
//! Used by the `inspect` command to dump each pipeline stage as an image with no
//! external image-crate dependency — PGM is a trivial header plus one byte per
//! pixel, writable with only `std`. Any viewer opens it, and
//! `convert stage.pgm stage.png` turns it into a PNG if wanted.
//!
//! All writers normalize their input to the 0..=255 range, because the data at
//! each stage lives on wildly different scales (intensities in [0,1], FFT
//! magnitudes in the thousands, phases in [-π, π]). Several normalization modes
//! are provided to match what is actually informative for each stage.

use std::fs::File;
use std::io::{BufWriter, Result, Write};
use std::path::Path;

/// Writes a `width × height` grayscale image (row-major `0..=255` bytes) as a
/// binary PGM file.
fn write_pgm(path: &Path, width: usize, height: usize, pixels: &[u8]) -> Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    // PGM binary header: magic, dimensions, max value.
    write!(writer, "P5\n{} {}\n255\n", width, height)?;
    writer.write_all(pixels)?;
    writer.flush()
}

/// Normalizes `data` to bytes by linear min→max scaling. Good for anything where
/// you want to see the full dynamic range (intensities, phase maps).
pub fn save_linear(path: &Path, width: usize, height: usize, data: &[f64]) -> Result<()> {
    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    for &v in data {
        if v.is_finite() {
            lo = lo.min(v);
            hi = hi.max(v);
        }
    }
    let span = if hi > lo { hi - lo } else { 1.0 };
    let bytes: Vec<u8> = data
        .iter()
        .map(|&v| {
            let n = ((v - lo) / span * 255.0).round();
            n.clamp(0.0, 255.0) as u8
        })
        .collect();
    write_pgm(path, width, height, &bytes)
}

/// Normalizes by `log(1 + |v|)` then min→max. Essential for FFT magnitudes: the
/// DC and peak bins are orders of magnitude brighter than everything else, so a
/// linear scale shows only a few white dots on black. Log scale reveals the lobe
/// structure and the spectral skirts.
pub fn save_log_magnitude(path: &Path, width: usize, height: usize, mag: &[f64]) -> Result<()> {
    let logged: Vec<f64> = mag.iter().map(|&m| (1.0 + m.abs()).ln()).collect();
    save_linear(path, width, height, &logged)
}

/// Saves data already known to be in `0.0..=1.0` directly (no per-image
/// rescaling), so absolute brightness is comparable across images. For the
/// generated pattern and the band-pass mask.
pub fn save_unit(path: &Path, width: usize, height: usize, data: &[f64]) -> Result<()> {
    let bytes: Vec<u8> = data
        .iter()
        .map(|&v| (v.clamp(0.0, 1.0) * 255.0).round() as u8)
        .collect();
    write_pgm(path, width, height, &bytes)
}

/// fftshift: moves the zero-frequency (DC) bin from the corner to the center, so
/// the spectrum is displayed the way it is usually drawn — lobes arranged around
/// a central DC. Operates on a row-major `width × height` buffer.
pub fn fftshift(width: usize, height: usize, data: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; data.len()];
    let (half_width, half_height) = (width / 2, height / 2);
    for y in 0..height {
        for x in 0..width {
            // Swap quadrants diagonally.
            let sx = (x + half_width) % width;
            let sy = (y + half_height) % height;
            out[sy * width + sx] = data[y * width + x];
        }
    }
    out
}
