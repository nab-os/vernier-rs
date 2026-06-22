use crate::buffer::BufferLayout;
use crate::complex::Complex32;
use crate::scalar::Real;

#[derive(Clone, Debug)]
pub struct GrayImage {
    layout: BufferLayout,
    data: Vec<f32>,
}

impl GrayImage {
    pub fn from_vec(width: usize, height: usize, data: Vec<f32>) -> Option<Self> {
        if data.len() != width * height {
            return None;
        }
        Some(Self {
            layout: BufferLayout::packed(width, height),
            data,
        })
    }

    pub fn zeros(width: usize, height: usize) -> Self {
        Self {
            layout: BufferLayout::packed(width, height),
            data: vec![0.0; width * height],
        }
    }

    #[inline]
    pub fn layout(&self) -> BufferLayout {
        self.layout
    }

    #[inline]
    pub fn width(&self) -> usize {
        self.layout.width
    }

    #[inline]
    pub fn height(&self) -> usize {
        self.layout.height
    }

    #[inline]
    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        &mut self.data
    }

    pub fn to_complex(&self) -> Vec<Complex32> {
        self.data.iter().map(|&v| Complex32::from_real(v)).collect()
    }

    #[inline]
    pub fn get(&self, row: usize, col: usize) -> Real {
        self.data[self.layout.flat_index(row, col)] as Real
    }
}
