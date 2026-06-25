//! [`CpuBackend`]: the `rustfft`/`ndarray` implementation of
//! [`ComputeBackend`](vernier_core::ComputeBackend).
//!
//! The compute primitives are deliberately simple here — `extract_phase` is a
//! `map`, `argmax_magnitude` is a scan. On the GPU these become a kernel and a
//! reduction; on the CPU they are the obviously-correct reference the GPU is
//! measured against.

use std::cell::RefCell;

use vernier_core::buffer::{Buffer2D, BufferLayout};
use vernier_core::scalar::consts::{PI, TAU};
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

    fn annulus_mask(&self, spectrum: &mut CpuBuffer, min_frequency: usize, max_frequency: usize) {
        let layout = spectrum.layout();
        let (width, height) = (layout.width, layout.height);
        let min_r2 = (min_frequency * min_frequency) as Real;
        let max_r2 = if max_frequency > 0 {
            (max_frequency * max_frequency) as Real
        } else {
            Real::INFINITY
        };

        let signed = |f: usize, n: usize| -> isize {
            let f = f as isize;
            let n = n as isize;
            if f > n / 2 { f - n } else { f }
        };

        for fy in 0..height {
            let sfy = signed(fy, height) as Real;
            for fx in 0..width {
                let sfx = signed(fx, width) as Real;
                let r2 = sfx * sfx + sfy * sfy;
                if r2 < min_r2 || r2 > max_r2 {
                    let v = &mut spectrum.as_mut_slice()[fy * width + fx];
                    v.re = 0.0;
                    v.im = 0.0;
                }
            }
        }
    }
    fn argmax_magnitude(&self, buffer: &CpuBuffer) -> Result<(usize, Real)> {
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
        buffer: &CpuBuffer,
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
        buffer: &CpuBuffer,
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

    fn halfplane_argmax(
        &self,
        buffer: &CpuBuffer,
        width: usize,
        height: usize,
    ) -> Option<(usize, usize)> {
        let signed = |f: usize, n: usize| -> isize {
            let f = f as isize;
            let n = n as isize;
            if f > n / 2 { f - n } else { f }
        };
        let mut best = Real::NEG_INFINITY;
        let mut result = None;
        for fy in 0..height {
            let sfy = signed(fy, height);
            if sfy < 0 {
                continue;
            }
            for fx in 0..width {
                let m = buffer.as_slice()[fy * width + fx].re;
                if m > best {
                    best = m;
                    result = Some((fx, fy));
                }
            }
        }
        result
    }

    fn halfplane_argmax_angular_excl(
        &self,
        buffer: &CpuBuffer,
        width: usize,
        height: usize,
        center_angle: Real,
        half_width: Real,
    ) -> Option<(usize, usize)> {
        let signed = |f: usize, n: usize| -> isize {
            let f = f as isize;
            let n = n as isize;
            if f > n / 2 { f - n } else { f }
        };
        let mut best = Real::NEG_INFINITY;
        let mut result = None;
        for fy in 0..height {
            let sfy_i = signed(fy, height);
            if sfy_i < 0 {
                continue;
            }
            let sfy = sfy_i as Real;
            for fx in 0..width {
                let sfx_i = signed(fx, width);
                let sfx = sfx_i as Real;
                // Angular difference from center (shortest arc, in (-π, π]).
                let angle = sfy.atan2(sfx);
                let diff = ((angle - center_angle + PI).rem_euclid(TAU)) - PI;
                if diff.abs() < half_width {
                    continue;
                }
                let m = buffer.as_slice()[fy * width + fx].re;
                if m > best {
                    best = m;
                    result = Some((fx, fy));
                }
            }
        }
        result
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

    fn fft2d(&self, buffer: &mut Self::Buffer2D) -> Result<()> {
        self.planner.borrow_mut().forward(buffer);
        Ok(())
    }

    fn ifft2d(&self, buffer: &mut Self::Buffer2D) -> Result<()> {
        self.planner.borrow_mut().inverse(buffer);
        Ok(())
    }

    fn extract_phase(&self, buffer: &CpuBuffer) -> Result<CpuBuffer> {
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

    fn peak_search(
        &self,
        buffer: &mut Self::Buffer2D,
        min_frequency: usize,
        max_frequency: usize,
        smoothing_sigma: Real,
        sigma: Real,
    ) -> Result<Option<Self::Buffer2D>> {
        let mut spectrum = buffer.clone();
        let layout = spectrum.layout(); // must read before mutable borrows below
        let (width, height) = (layout.width, layout.height);

        // Convert complex spectrum to magnitude in-place (re = |z|, im = 0).
        // All subsequent operations (annulus mask, blur, argmax) work on real
        // magnitudes stored in the .re field.
        for v in spectrum.as_mut_slice().iter_mut() {
            v.re = (v.re * v.re + v.im * v.im).sqrt();
            v.im = 0.0;
        }

        self.annulus_mask(&mut spectrum, min_frequency, max_frequency);

        let signed = |f: usize, n: usize| -> isize {
            let f = f as isize;
            let n = n as isize;
            if f > n / 2 { f - n } else { f }
        };

        // Gaussian blur on magnitude (C++ cv::GaussianBlur).
        if smoothing_sigma > 0.0 {
            self.gaussian_blur_2d(&mut spectrum, smoothing_sigma)
                .unwrap();
        }

        // Peak 1: largest magnitude in canonical half-plane.
        let (cx1, cy1) = {
            if let Some(a) = self.halfplane_argmax(&spectrum, width, height) {
                a
            } else {
                return Ok(None);
            }
        };
        let sfx1 = signed(cx1, width) as Real;
        let sfy1 = signed(cy1, height) as Real;

        // Angular cone exclusion around peak 1's direction (C++ applyAngularCut).
        let distance = (sfx1 * sfx1 + sfy1 * sfy1).sqrt();
        let center_angle = sfy1.atan2(sfx1);
        // C++: widthAngle = 2 * atan2(3*sigma, distance); we use half that as the
        // exclusion threshold so a bin is excluded when |angle_diff| < half_width.
        let half_width = (3.0 * sigma).atan2(distance);

        // Restrict peak 2 search to a frequency band around peak 1's radius so that
        // sub-harmonics or harmonics at very different frequencies don't win over the
        // true perpendicular carrier (which should be at the same spatial frequency for
        // a square grid).
        let r_min_sq = (distance * 0.5) * (distance * 0.5);
        let r_max_sq = (distance * 2.0) * (distance * 2.0);
        for fy in 0..height {
            let sfy_i = signed(fy, height) as Real;
            for fx in 0..width {
                let sfx_i = signed(fx, width) as Real;
                let r_sq = sfx_i * sfx_i + sfy_i * sfy_i;
                if r_sq < r_min_sq || r_sq > r_max_sq {
                    spectrum.as_mut_slice()[fy * width + fx].re = 0.0;
                }
            }
        }

        // Peak 2: largest magnitude outside the angular cone.
        let (cx2, cy2) = {
            if let Some(a) = self.halfplane_argmax_angular_excl(
                &spectrum,
                width,
                height,
                center_angle,
                half_width,
            ) {
                a
            } else {
                return Ok(None);
            }
        };

        // Order so the larger signed column frequency is direction 1 (C++ convention:
        // swap if mainPeak1.x < mainPeak2.x in the shifted spectrum, equiv. to
        // sfx1 < sfx2 in unshifted).
        let sfx2 = signed(cx2, width) as Real;
        if sfx1 >= sfx2 {
            Ok(CpuBuffer::from_slice(
                &[
                    Complex32 {
                        re: cx1 as f32,
                        im: 0.0,
                    },
                    Complex32 {
                        re: cy1 as f32,
                        im: 0.0,
                    },
                    Complex32 {
                        re: cx2 as f32,
                        im: 0.0,
                    },
                    Complex32 {
                        re: cy2 as f32,
                        im: 0.0,
                    },
                ],
                BufferLayout {
                    width: 2,
                    height: 2,
                    row_stride: 2,
                },
            ))
        } else {
            Ok(CpuBuffer::from_slice(
                &[
                    Complex32 {
                        re: cx2 as f32,
                        im: 0.0,
                    },
                    Complex32 {
                        re: cy2 as f32,
                        im: 0.0,
                    },
                    Complex32 {
                        re: cx1 as f32,
                        im: 0.0,
                    },
                    Complex32 {
                        re: cy1 as f32,
                        im: 0.0,
                    },
                ],
                BufferLayout {
                    width: 2,
                    height: 2,
                    row_stride: 2,
                },
            ))
        }
    }

    fn filter(
        &self,
        buffer: &mut Self::Buffer2D,
        min_frequency: usize,
        max_frequency: usize,
    ) -> Result<()> {
        self.annulus_mask(buffer, min_frequency, max_frequency);
        Ok(())
    }

    /// Separable 2D Gaussian blur on a real-valued array, in place.
    fn gaussian_blur_2d(&self, buffer: &mut Self::Buffer2D, sigma: Real) -> Result<()> {
        let width = buffer.layout().width;
        let height = buffer.layout().height;
        let radius = (3.0 * sigma).ceil() as usize;
        let n = 2 * radius + 1;
        let kernel: Vec<Real> = (0..n)
            .map(|i| {
                let x = i as Real - radius as Real;
                (-x * x / (2.0 * sigma * sigma)).exp()
            })
            .collect();
        let ksum: Real = kernel.iter().sum();
        let kernel: Vec<Real> = kernel.iter().map(|&k| k / ksum).collect();

        let mut tmp = vec![0.0_f32; width * height];

        // Blur along rows.
        for r in 0..height {
            for c in 0..width {
                let (mut v, mut w) = (0.0_f32, 0.0_f32);
                for (ki, &kv) in kernel.iter().enumerate() {
                    let sc = c as isize + ki as isize - radius as isize;
                    if sc >= 0 && (sc as usize) < width {
                        v += buffer.as_slice()[r * width + sc as usize].re * kv;
                        w += kv;
                    }
                }
                tmp[r * width + c] = if w > 0.0 { v / w } else { 0.0 };
            }
        }

        // Blur along columns.
        for r in 0..height {
            for c in 0..width {
                let (mut v, mut w) = (0.0_f32, 0.0_f32);
                for (ki, &kv) in kernel.iter().enumerate() {
                    let sr = r as isize + ki as isize - radius as isize;
                    if sr >= 0 && (sr as usize) < height {
                        v += tmp[sr as usize * width + c] * kv;
                        w += kv;
                    }
                }
                buffer.as_mut_slice()[r * width + c].re = if w > 0.0 { v / w } else { 0.0 };
            }
        }

        Ok(())
    }

    fn bandpass_filter(&self, buffer: &mut CpuBuffer, sigma: Real) -> Result<()> {
        let layout = buffer.layout();
        let (w, h) = (layout.width, layout.height);
        let two_sigma_sq = 2.0 * sigma * sigma;

        let center_x = w / 2;
        let center_y = h / 2;

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
