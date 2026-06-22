//! [`CpuBackend`]: the `rustfft`/`ndarray` implementation of
//! [`ComputeBackend`](vernier_core::ComputeBackend).
//!
//! The compute primitives are deliberately simple here — `extract_phase` is a
//! `map`, `argmax_magnitude` is a scan. On the GPU these become a kernel and a
//! reduction; on the CPU they are the obviously-correct reference the GPU is
//! measured against.

use std::cell::RefCell;

use vernier_core::buffer::{Buffer2D, BufferLayout};
use vernier_core::{Complex32, ComputeBackend, Real, Result, VernierError};

use crate::buffer::CpuBuffer;
use crate::fft::Fft2dPlanner;

/// CPU reference backend.
///
/// The FFT planner is cached in a `RefCell` because planning mutates internal
/// state but the trait methods take `&self` (a backend is shared across pipeline
/// stages). Single-threaded interior mutability is sufficient; if the pipeline
/// ever shares a backend across threads, swap this for a `Mutex`.
pub struct CpuBackend {
    planner: RefCell<Fft2dPlanner>,
}

impl CpuBackend {
    /// Creates a new CPU backend.
    pub fn new() -> Self {
        Self {
            planner: RefCell::new(Fft2dPlanner::new()),
        }
    }
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl ComputeBackend for CpuBackend {
    type Buffer2D = CpuBuffer;

    fn upload(&self, data: &[Complex32], layout: BufferLayout) -> Result<Self::Buffer2D> {
        if !layout.is_contiguous() {
            return Err(VernierError::NonContiguous {
                stride: layout.row_stride,
                width: layout.width,
            });
        }
        CpuBuffer::from_slice(data, layout).ok_or(VernierError::ShapeMismatch {
            lhs: layout,
            rhs: BufferLayout::packed(layout.width, layout.height),
        })
    }

    fn download(&self, buffer: &Self::Buffer2D) -> Result<Vec<Complex32>> {
        Ok(buffer.as_slice().to_vec())
    }

    fn hann_window(&self, buffer: &mut Self::Buffer2D) -> Result<()> {
        let layout = buffer.layout();
        let (w, h) = (layout.width, layout.height);

        // Precompute the separable 1D Hann factors for each axis.
        let hann = |k: usize, n: usize| -> f32 {
            if n <= 1 {
                return 1.0;
            }
            use std::f32::consts::TAU;
            0.5 * (1.0 - (TAU * k as f32 / (n as f32 - 1.0)).cos())
        };
        let wx: Vec<f32> = (0..w).map(|i| hann(i, w)).collect();
        let wy: Vec<f32> = (0..h).map(|j| hann(j, h)).collect();

        let data = buffer.as_mut_slice();
        for r in 0..h {
            let gy = wy[r];
            for c in 0..w {
                let g = wx[c] * gy;
                let idx = r * w + c;
                data[idx].re *= g;
                data[idx].im *= g;
            }
        }
        Ok(())
    }

    fn fft2d(&self, buffer: &mut Self::Buffer2D) -> Result<()> {
        self.planner.borrow_mut().forward(buffer);
        Ok(())
    }

    fn ifft2d(&self, buffer: &mut Self::Buffer2D) -> Result<()> {
        self.planner.borrow_mut().inverse(buffer);
        Ok(())
    }

    fn extract_phase(&self, buffer: &Self::Buffer2D) -> Result<Self::Buffer2D> {
        // Wrapped phase atan2(im, re) into the re lane; im zeroed. The parallel
        // per-pixel kernel on the GPU; a plain map here.
        let phases: Vec<Complex32> = buffer
            .as_slice()
            .iter()
            .map(|c| Complex32::new(c.arg() as f32, 0.0))
            .collect();
        CpuBuffer::from_slice(&phases, buffer.layout()).ok_or(VernierError::ShapeMismatch {
            lhs: buffer.layout(),
            rhs: buffer.layout(),
        })
    }

    fn argmax_magnitude(&self, buffer: &Self::Buffer2D) -> Result<(usize, Real)> {
        // Reduction: largest |z|^2 (cheaper than |z|, same ordering).
        let mut best_idx = 0usize;
        let mut best = Real::NEG_INFINITY;
        for (i, c) in buffer.as_slice().iter().enumerate() {
            let m = c.norm_sqr();
            if m > best {
                best = m;
                best_idx = i;
            }
        }
        Ok((best_idx, best.sqrt()))
    }

    fn argmax_magnitude_halfplane(
        &self,
        buffer: &Self::Buffer2D,
        min_radius: usize,
    ) -> Result<(usize, Real)> {
        let layout = buffer.layout();
        let (w, h) = (layout.width, layout.height);

        // Signed frequency of a bin: bins past N/2 wrap to negative.
        let signed = |f: usize, n: usize| -> isize {
            let f = f as isize;
            let n = n as isize;
            if f > n / 2 { f - n } else { f }
        };

        let data = buffer.as_slice();
        let min_r2 = (min_radius * min_radius) as isize;
        let mut best_idx = 0usize;
        let mut best = Real::NEG_INFINITY;
        for fy in 0..h {
            let sfy = signed(fy, h);
            for fx in 0..w {
                if fx == 0 && fy == 0 {
                    continue; // exclude DC
                }
                let sfx = signed(fx, w);
                // Exclude the low-frequency disk (lighting/vignette content).
                if sfx * sfx + sfy * sfy < min_r2 {
                    continue;
                }
                // Canonical half-plane: positive x frequency, or the +y axis.
                let in_half = sfx > 0 || (sfx == 0 && sfy > 0);
                if !in_half {
                    continue;
                }
                let m = data[fy * w + fx].norm_sqr();
                if m > best {
                    best = m;
                    best_idx = fy * w + fx;
                }
            }
        }
        Ok((best_idx, best.sqrt()))
    }

    fn argmax_magnitude_halfplane_excluding(
        &self,
        buffer: &Self::Buffer2D,
        exclude_x: usize,
        exclude_y: usize,
        radius: usize,
        min_radius: usize,
    ) -> Result<(usize, Real)> {
        let layout = buffer.layout();
        let (w, h) = (layout.width, layout.height);

        let signed = |f: usize, n: usize| -> isize {
            let f = f as isize;
            let n = n as isize;
            if f > n / 2 { f - n } else { f }
        };
        let ex = signed(exclude_x, w);
        let ey = signed(exclude_y, h);
        let r = radius as isize;
        let min_r2 = (min_radius * min_radius) as isize;

        let data = buffer.as_slice();
        let mut best_idx = 0usize;
        let mut best = Real::NEG_INFINITY;
        for fy in 0..h {
            let sfy = signed(fy, h);
            for fx in 0..w {
                if fx == 0 && fy == 0 {
                    continue;
                }
                let sfx = signed(fx, w);
                if sfx * sfx + sfy * sfy < min_r2 {
                    continue; // low-frequency lighting content
                }
                let in_half = sfx > 0 || (sfx == 0 && sfy > 0);
                if !in_half {
                    continue;
                }
                // Skip the excluded neighborhood (Chebyshev distance in signed
                // frequency). Also exclude the mirror of the excluded point,
                // since a sideband of the first lobe can appear on either side.
                let near = (sfx - ex).abs() <= r && (sfy - ey).abs() <= r;
                let near_mirror = (sfx + ex).abs() <= r && (sfy + ey).abs() <= r;
                if near || near_mirror {
                    continue;
                }
                let m = data[fy * w + fx].norm_sqr();
                if m > best {
                    best = m;
                    best_idx = fy * w + fx;
                }
            }
        }
        Ok((best_idx, best.sqrt()))
    }

    fn bandpass_filter(
        &self,
        buffer: &mut Self::Buffer2D,
        center_x: usize,
        center_y: usize,
        sigma: Real,
    ) -> Result<()> {
        let layout = buffer.layout();
        let (w, h) = (layout.width, layout.height);
        let two_sigma_sq = 2.0 * sigma * sigma;

        // Frequency-bin distance must wrap: bin 0 and bin w-1 are adjacent in a
        // DFT (the spectrum is periodic). Without wraparound, a lobe near the
        // Nyquist edge would be clipped. `circular_delta` gives the shorter
        // signed distance around the ring of length `n`.
        let circular_delta = |a: usize, c: usize, n: usize| -> Real {
            let d = a as isize - c as isize;
            let n = n as isize;
            // Reduce into (-n/2, n/2].
            let d = ((d % n) + n) % n;
            let d = if d > n / 2 { d - n } else { d };
            d as Real
        };

        let data = buffer.as_mut_slice();
        for fy in 0..h {
            let dy = circular_delta(fy, center_y, h);
            for fx in 0..w {
                let dx = circular_delta(fx, center_x, w);
                let r2 = dx * dx + dy * dy;
                let gain = (-r2 / two_sigma_sq).exp() as f32;
                let idx = fy * w + fx;
                data[idx].re *= gain;
                data[idx].im *= gain;
            }
        }
        Ok(())
    }

    fn name(&self) -> &str {
        "cpu-rustfft"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use vernier_core::buffer::BufferLayout;

    fn checkerboard(w: usize, h: usize) -> (Vec<Complex32>, BufferLayout) {
        let layout = BufferLayout::packed(w, h);
        let mut v = Vec::with_capacity(layout.len());
        for r in 0..h {
            for c in 0..w {
                let val = if (r + c) % 2 == 0 { 1.0 } else { -1.0 };
                v.push(Complex32::new(val, 0.0));
            }
        }
        (v, layout)
    }

    #[test]
    fn fft_then_ifft_is_identity() {
        let backend = CpuBackend::new();
        let (data, layout) = checkerboard(8, 8);
        let mut buf = backend.upload(&data, layout).unwrap();

        backend.fft2d(&mut buf).unwrap();
        backend.ifft2d(&mut buf).unwrap();

        let out = backend.download(&buf).unwrap();
        for (orig, got) in data.iter().zip(out.iter()) {
            assert_abs_diff_eq!(orig.re, got.re, epsilon = 1e-4);
            assert_abs_diff_eq!(orig.im, got.im, epsilon = 1e-4);
        }
    }

    #[test]
    fn argmax_finds_the_dc_spike() {
        // A constant image has all energy at the DC bin (index 0) after FFT.
        let backend = CpuBackend::new();
        let layout = BufferLayout::packed(8, 8);
        let data = vec![Complex32::new(1.0, 0.0); layout.len()];
        let mut buf = backend.upload(&data, layout).unwrap();

        backend.fft2d(&mut buf).unwrap();
        let (idx, _mag) = backend.argmax_magnitude(&buf).unwrap();
        assert_eq!(idx, 0, "all energy should sit at the DC bin");
    }

    #[test]
    fn non_square_round_trips() {
        // Guards the row/column length bookkeeping (w != h).
        let backend = CpuBackend::new();
        let (data, layout) = checkerboard(16, 4);
        let mut buf = backend.upload(&data, layout).unwrap();
        backend.fft2d(&mut buf).unwrap();
        backend.ifft2d(&mut buf).unwrap();
        let out = backend.download(&buf).unwrap();
        for (orig, got) in data.iter().zip(out.iter()) {
            assert_abs_diff_eq!(orig.re, got.re, epsilon = 1e-4);
        }
    }
}
