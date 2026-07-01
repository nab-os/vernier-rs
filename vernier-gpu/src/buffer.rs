use vernier_core::Complex32;
use vernier_core::buffer::{Buffer2D, BufferLayout};
use vulkano::buffer::Subbuffer;

#[derive(Clone, Debug)]
pub struct GpuBuffer {
    pub buffer: Subbuffer<[Complex32]>,
    pub width: usize,
    pub height: usize,
}

impl GpuBuffer {
    pub fn size(&self) -> usize {
        self.width * self.height
    }
}

impl Buffer2D for GpuBuffer {
    fn layout(&self) -> BufferLayout {
        BufferLayout::packed(self.width, self.height)
    }
}
