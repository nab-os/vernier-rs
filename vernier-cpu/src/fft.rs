use std::sync::Arc;

use rustfft::num_complex::Complex as RfComplex;
use rustfft::{Fft, FftDirection, FftPlanner};
use vernier_core::Complex32;
use vernier_core::buffer::Buffer2D;

use crate::buffer::CpuBuffer;

#[inline]
fn as_rustfft_mut(slice: &mut [Complex32]) -> &mut [RfComplex<f32>] {
    unsafe {
        std::slice::from_raw_parts_mut(slice.as_mut_ptr() as *mut RfComplex<f32>, slice.len())
    }
}

pub(crate) struct Fft2dPlanner {
    planner: FftPlanner<f32>,
}

impl Fft2dPlanner {
    pub(crate) fn new() -> Self {
        Self {
            planner: FftPlanner::new(),
        }
    }

    fn plan(&mut self, len: usize, dir: FftDirection) -> Arc<dyn Fft<f32>> {
        self.planner.plan_fft(len, dir)
    }

    pub(crate) fn forward(&mut self, buf: &mut CpuBuffer) {
        self.transform(buf, FftDirection::Forward);
    }

    pub(crate) fn inverse(&mut self, buf: &mut CpuBuffer) {
        self.transform(buf, FftDirection::Inverse);
        let element_count = buf.layout().len() as f32;
        for element in buf.as_mut_slice() {
            element.re /= element_count;
            element.im /= element_count;
        }
    }

    fn transform(&mut self, buf: &mut CpuBuffer, dir: FftDirection) {
        let layout = buf.layout();
        let (width, height) = (layout.width, layout.height);

        let row_fft = self.plan(width, dir);
        {
            let data = as_rustfft_mut(buf.as_mut_slice());
            for row in data.chunks_exact_mut(width) {
                row_fft.process(row);
            }
        }

        transpose(buf);
        let col_fft = self.plan(height, dir);
        {
            let data = as_rustfft_mut(buf.as_mut_slice());
            for row in data.chunks_exact_mut(height) {
                col_fft.process(row);
            }
        }
        transpose(buf);
    }
}

fn transpose(buf: &mut CpuBuffer) {
    let transposed = buf.data.t().as_standard_layout().into_owned();
    buf.data = transposed;
}
