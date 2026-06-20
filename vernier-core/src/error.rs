use crate::buffer::BufferLayout;

pub type Result<T> = core::result::Result<T, VernierError>;

#[derive(Debug, thiserror::Error)]
pub enum VernierError {
    #[error("non-contiguous buffer: row_stride {stride} != width {width}")]
    NonContiguous { stride: usize, width: usize },

    #[error("shape mismatch: {lhs:?} vs {rhs:?}")]
    ShapeMismatch {
        lhs: BufferLayout,
        rhs: BufferLayout,
    },

    #[error("unsupported transform size: {0}x{1}")]
    UnsupportedSize(usize, usize),

    #[error("backend error: {0}")]
    Backend(String),
}
