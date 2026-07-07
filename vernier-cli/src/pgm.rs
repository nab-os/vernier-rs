//! Small dependency-free image helpers shared by the debug renderers.

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
