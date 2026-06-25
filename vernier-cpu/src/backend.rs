//! [`CpuBackend`] and [`CpuJob`]: the `rustfft`/`ndarray` implementations of
//! [`ComputeBackend`](vernier_core::ComputeBackend) and [`ComputeJob`](vernier_core::ComputeJob).

use std::cell::RefCell;

use vernier_core::buffer::{Buffer2D, BufferLayout};
use vernier_core::scalar::consts::{PI, TAU};
use vernier_core::{Complex32, ComputeBackend, ComputeJob, Real, Result, VernierError};

use crate::buffer::CpuBuffer;
use crate::fft::Fft2dPlanner;

/// CPU reference backend.
pub struct CpuBackend {
    planner: RefCell<Fft2dPlanner>,
}

impl CpuBackend {
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
    type Job<'a> = CpuJob<'a>;

    fn begin(&self) -> Result<CpuJob<'_>> {
        Ok(CpuJob { backend: self })
    }

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

    fn download(&self, buffer: &CpuBuffer) -> Result<Vec<Complex32>> {
        Ok(buffer.as_slice().to_vec())
    }

    fn name(&self) -> &str {
        "cpu-rustfft"
    }
}

// ---------------------------------------------------------------------------
// CpuJob — synchronous; submit() is a no-op
// ---------------------------------------------------------------------------

pub struct CpuJob<'a> {
    backend: &'a CpuBackend,
}

impl CpuJob<'_> {
    fn annulus_mask(spectrum: &mut CpuBuffer, min_frequency: usize, max_frequency: usize) {
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

    fn halfplane_argmax(buffer: &CpuBuffer, width: usize, height: usize) -> Option<(usize, usize)> {
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

impl ComputeJob for CpuJob<'_> {
    type Buffer2D = CpuBuffer;

    fn copy_buffer(&mut self, src: &CpuBuffer) -> Result<CpuBuffer> {
        Ok(src.clone())
    }

    fn fft2d(&mut self, buf: &mut CpuBuffer) -> Result<()> {
        self.backend.planner.borrow_mut().forward(buf);
        Ok(())
    }

    fn ifft2d(&mut self, buf: &mut CpuBuffer) -> Result<()> {
        self.backend.planner.borrow_mut().inverse(buf);
        Ok(())
    }

    fn extract_phase(&mut self, buf: &CpuBuffer) -> Result<CpuBuffer> {
        let phases: Vec<Complex32> = buf
            .as_slice()
            .iter()
            .map(|c| Complex32::new(c.arg() as f32, 0.0))
            .collect();
        CpuBuffer::from_slice(&phases, buf.layout()).ok_or(VernierError::ShapeMismatch {
            lhs: buf.layout(),
            rhs: buf.layout(),
        })
    }

    fn filter(
        &mut self,
        buf: &mut CpuBuffer,
        min_frequency: usize,
        max_frequency: usize,
    ) -> Result<()> {
        Self::annulus_mask(buf, min_frequency, max_frequency);
        Ok(())
    }

    fn gaussian_blur_2d(&mut self, buf: &mut CpuBuffer, sigma: Real) -> Result<()> {
        let width = buf.layout().width;
        let height = buf.layout().height;
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

        for r in 0..height {
            for c in 0..width {
                let (mut v, mut w) = (0.0_f32, 0.0_f32);
                for (ki, &kv) in kernel.iter().enumerate() {
                    let sc = c as isize + ki as isize - radius as isize;
                    if sc >= 0 && (sc as usize) < width {
                        v += buf.as_slice()[r * width + sc as usize].re * kv;
                        w += kv;
                    }
                }
                tmp[r * width + c] = if w > 0.0 { v / w } else { 0.0 };
            }
        }

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
                buf.as_mut_slice()[r * width + c].re = if w > 0.0 { v / w } else { 0.0 };
            }
        }

        Ok(())
    }

    fn bandpass_filter(
        &mut self,
        buf: &mut CpuBuffer,
        cx: usize,
        cy: usize,
        sigma: Real,
    ) -> Result<()> {
        let layout = buf.layout();
        let (w, h) = (layout.width, layout.height);
        let two_sigma_sq = 2.0 * sigma * sigma;

        let circular_delta = |a: usize, c: usize, n: usize| -> Real {
            let d = a as isize - c as isize;
            let n = n as isize;
            let d = ((d % n) + n) % n;
            let d = if d > n / 2 { d - n } else { d };
            d as Real
        };

        let data = buf.as_mut_slice();
        for fy in 0..h {
            let dy = circular_delta(fy, cy, h);
            for fx in 0..w {
                let dx = circular_delta(fx, cx, w);
                let r2 = dx * dx + dy * dy;
                let gain = (-r2 / two_sigma_sq).exp() as f32;
                let idx = fy * w + fx;
                data[idx].re *= gain;
                data[idx].im *= gain;
            }
        }
        Ok(())
    }

    fn peak_search(
        &mut self,
        buffer: &mut CpuBuffer,
        min_frequency: usize,
        max_frequency: usize,
        smoothing_sigma: Real,
        sigma: Real,
    ) -> Result<Option<CpuBuffer>> {
        let mut spectrum = buffer.clone();
        let layout = spectrum.layout();
        let (width, height) = (layout.width, layout.height);

        for v in spectrum.as_mut_slice().iter_mut() {
            v.re = (v.re * v.re + v.im * v.im).sqrt();
            v.im = 0.0;
        }

        Self::annulus_mask(&mut spectrum, min_frequency, max_frequency);

        let signed = |f: usize, n: usize| -> isize {
            let f = f as isize;
            let n = n as isize;
            if f > n / 2 { f - n } else { f }
        };

        if smoothing_sigma > 0.0 {
            self.gaussian_blur_2d(&mut spectrum, smoothing_sigma).unwrap();
        }

        let (cx1, cy1) = {
            if let Some(a) = Self::halfplane_argmax(&spectrum, width, height) {
                a
            } else {
                return Ok(None);
            }
        };
        let sfx1 = signed(cx1, width) as Real;
        let sfy1 = signed(cy1, height) as Real;

        let distance = (sfx1 * sfx1 + sfy1 * sfy1).sqrt();
        let center_angle = sfy1.atan2(sfx1);
        let half_width = (3.0 * sigma).atan2(distance);

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

        let (cx2, cy2) = {
            if let Some(a) =
                Self::halfplane_argmax_angular_excl(&spectrum, width, height, center_angle, half_width)
            {
                a
            } else {
                return Ok(None);
            }
        };

        let sfx2 = signed(cx2, width) as Real;
        let (d1x, d1y, d2x, d2y) = if sfx1 >= sfx2 {
            (cx1, cy1, cx2, cy2)
        } else {
            (cx2, cy2, cx1, cy1)
        };

        Ok(CpuBuffer::from_slice(
            &[
                Complex32 { re: d1x as f32, im: 0.0 },
                Complex32 { re: d1y as f32, im: 0.0 },
                Complex32 { re: d2x as f32, im: 0.0 },
                Complex32 { re: d2y as f32, im: 0.0 },
            ],
            BufferLayout { width: 2, height: 2, row_stride: 2 },
        ))
    }

    fn submit(self) -> Result<()> {
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Keep argmax_magnitude as a standalone method on CpuBackend for tests
// ---------------------------------------------------------------------------
impl CpuBackend {
    pub fn argmax_magnitude(&self, buffer: &CpuBuffer) -> Result<(usize, Real)> {
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
        let mut job = backend.begin().unwrap();
        job.fft2d(&mut buf).unwrap();
        job.ifft2d(&mut buf).unwrap();
        job.submit().unwrap();
        let out = backend.download(&buf).unwrap();
        for (orig, got) in data.iter().zip(out.iter()) {
            assert_abs_diff_eq!(orig.re, got.re, epsilon = 1e-4);
            assert_abs_diff_eq!(orig.im, got.im, epsilon = 1e-4);
        }
    }

    #[test]
    fn argmax_finds_the_dc_spike() {
        let backend = CpuBackend::new();
        let layout = BufferLayout::packed(8, 8);
        let data = vec![Complex32::new(1.0, 0.0); layout.len()];
        let mut buf = backend.upload(&data, layout).unwrap();
        let mut job = backend.begin().unwrap();
        job.fft2d(&mut buf).unwrap();
        job.submit().unwrap();
        let (idx, _mag) = backend.argmax_magnitude(&buf).unwrap();
        assert_eq!(idx, 0, "all energy should sit at the DC bin");
    }

    #[test]
    fn non_square_round_trips() {
        let backend = CpuBackend::new();
        let (data, layout) = checkerboard(16, 4);
        let mut buf = backend.upload(&data, layout).unwrap();
        let mut job = backend.begin().unwrap();
        job.fft2d(&mut buf).unwrap();
        job.ifft2d(&mut buf).unwrap();
        job.submit().unwrap();
        let out = backend.download(&buf).unwrap();
        for (orig, got) in data.iter().zip(out.iter()) {
            assert_abs_diff_eq!(orig.re, got.re, epsilon = 1e-4);
        }
    }
}
