use ndarray::Array2;
use vernier_core::Complex32;
use vernier_core::buffer::{Buffer2D, BufferLayout};

#[derive(Clone, Debug)]
pub struct CpuBuffer {
    pub(crate) data: Array2<Complex32>,
}

impl CpuBuffer {
    pub(crate) fn from_slice(values: &[Complex32], layout: BufferLayout) -> Option<Self> {
        if !layout.is_contiguous() || values.len() != layout.len() {
            return None;
        }
        let data = Array2::from_shape_vec((layout.height, layout.width), values.to_vec()).ok()?;
        Some(Self { data })
    }

    pub(crate) fn as_slice(&self) -> &[Complex32] {
        self.data
            .as_slice()
            .expect("CpuBuffer is always standard-layout/contiguous")
    }

    pub(crate) fn as_mut_slice(&mut self) -> &mut [Complex32] {
        self.data
            .as_slice_mut()
            .expect("CpuBuffer is always standard-layout/contiguous")
    }
}

impl Buffer2D for CpuBuffer {
    fn layout(&self) -> BufferLayout {
        let (height, width) = self.data.dim();
        BufferLayout::packed(width, height)
    }
}
