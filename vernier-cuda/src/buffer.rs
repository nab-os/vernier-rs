use cudarc::driver::CudaSlice;
use vernier_core::buffer::{Buffer2D, BufferLayout};

pub struct CudaBuffer {
    pub data: CudaSlice<f32>,
    pub width: usize,
    pub height: usize,
}

impl CudaBuffer {
    pub fn n_complex(&self) -> usize {
        self.width * self.height
    }

    pub fn n_floats(&self) -> usize {
        2 * self.width * self.height
    }
}

impl Buffer2D for CudaBuffer {
    fn layout(&self) -> BufferLayout {
        BufferLayout::packed(self.width, self.height)
    }
}
