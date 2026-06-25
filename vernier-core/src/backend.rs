//! The [`ComputeBackend`] trait — the one swappable axis of the architecture.
//!
//! `vernier-cpu` implements this with `rustfft`/`ndarray`; `vernier-gpu`
//! implements the *same* trait with Vulkano compute. The pipeline crates
//! (`vernier-detection`, `vernier-pose`) are generic over `B: ComputeBackend`
//! and never name a concrete backend, so choosing one is a single line in
//! `vernier-cli`.
//!
//! ## Scope discipline
//!
//! This trait abstracts only operations that genuinely have two
//! implementations — the FFT, its inverse, the per-pixel phase stage, and the
//! on-device reductions that become kernels. Orchestration that runs on
//! already-downloaded results (megarena decode bookkeeping, peak selection
//! logic) stays as plain functions in the pipeline crates. A backend trait that
//! grows to twenty methods is a smell; keep new methods out unless a GPU version
//! would actually differ from the CPU one.
//!
//! ## The on-device principle, encoded in the types
//!
//! [`upload`](ComputeBackend::upload) and [`download`](ComputeBackend::download)
//! are the *only* host<->device crossings. Everything else takes and returns
//! `Self::Buffer2D`, so a correct pipeline uploads once, chains
//! [`fft2d`](ComputeBackend::fft2d) -> [`extract_phase`](ComputeBackend::extract_phase)
//! -> reductions on the device, and downloads only the handful of values it
//! actually needs. If you find yourself calling `download` between two compute
//! steps, that is the bug the API is shaped to make visible.

use crate::buffer::{Buffer2D, BufferLayout};
use crate::complex::Complex32;
use crate::error::Result;
use crate::scalar::Real;

/// A compute backend: owns a device/context and executes the spectral pipeline
/// primitives on its own [`Buffer2D`](ComputeBackend::Buffer2D) type.
pub trait ComputeBackend {
    /// The backend's concrete 2D buffer.
    ///
    /// CPU: a wrapper over `ndarray::Array2<Complex32>`.
    /// GPU: a wrapper over a Vulkano `Subbuffer<[Complex32]>`.
    ///
    /// `Clone` is part of the contract because the real detection chain must
    /// duplicate a spectrum before the destructive band-pass + inverse FFT (one
    /// copy per pattern direction). On CPU this is an array copy; on GPU a
    /// buffer-to-buffer copy command. A backend that cannot duplicate a buffer
    /// cannot implement the two-direction pipeline, so the requirement belongs
    /// here rather than leaking into every caller as a `where` clause.
    type Buffer2D: Buffer2D + Clone;

    // --- Host <-> device crossings (explicit and rare) ----------------------

    /// Initializes the backend before queueing operations
    fn init(&mut self) -> Result<()> {
        Ok(())
    }

    /// Exec queued operations
    fn exec(&mut self) -> Result<()> {
        Ok(())
    }

    /// Uploads contiguous, row-major complex data into a device buffer.
    ///
    /// `data.len()` must equal `layout.len()` and `layout` must be contiguous;
    /// otherwise the implementation returns
    /// [`VernierError::NonContiguous`](crate::error::VernierError::NonContiguous)
    /// or a shape error. On the GPU path this is a zero-copy byte cast plus a
    /// staging transfer; do not call it per stage.
    fn upload(&self, data: &[Complex32], layout: BufferLayout) -> Result<Self::Buffer2D>;

    /// Downloads a device buffer back to host memory, row-major.
    ///
    /// The deliberate device->host crossing. In a tuned pipeline this returns a
    /// small result (an isolated peak neighborhood, or the final phases), not a
    /// full image.
    fn download(&self, buffer: &Self::Buffer2D) -> Result<Vec<Complex32>>;

    /// Convenience: upload a real image as complex (imaginary parts zeroed).
    ///
    /// Provided so callers do not hand-roll the promotion; backends may override
    /// it with an R2C fast path that never materializes the zeros.
    fn upload_real(&self, image: &crate::image::GrayImage) -> Result<Self::Buffer2D> {
        let complex = image.to_complex();
        self.upload(&complex, image.layout())
    }

    // --- On-device compute primitives ---------------------------------------

    /// In-place 2D forward FFT of `buffer`.
    ///
    /// Implementations may require power-of-two dimensions and should return
    /// [`VernierError::UnsupportedSize`](crate::error::VernierError::UnsupportedSize)
    /// otherwise. The GPU path is the project's reason for existing: a batched,
    /// on-device transform (hand-written Stockham passes, or VkFFT via interop).
    fn fft2d(&self, buffer: &mut Self::Buffer2D) -> Result<()>;

    /// In-place 2D inverse FFT of `buffer`.
    fn ifft2d(&self, buffer: &mut Self::Buffer2D) -> Result<()>;

    fn extract_phase(&self, buffer: &Self::Buffer2D) -> Result<Self::Buffer2D>;

    fn peak_search(
        &self,
        buffer: &mut Self::Buffer2D,
        min_frequency: usize,
        max_frequency: usize,
        smoothing_sigma: Real,
        sigma: Real,
    ) -> Result<Option<Self::Buffer2D>>;

    fn filter(
        &self,
        buffer: &mut Self::Buffer2D,
        min_frequency: usize,
        max_frequency: usize,
    ) -> Result<()>;

    /// Separable 2D Gaussian blur on a real-valued array, in place.
    fn gaussian_blur_2d(&self, buffer: &mut Self::Buffer2D, sigma: Real) -> Result<()>;

    /// Applies a Gaussian band-pass filter centered on a single frequency lobe,
    /// in place on a frequency-domain buffer.
    fn bandpass_filter(&self, buffer: &mut Self::Buffer2D, sigma: Real) -> Result<()>;

    /// A short human-readable name for the active backend (e.g. `"cpu-rustfft"`,
    /// `"gpu-vulkano"`). For logs and the benchmark harness.
    fn name(&self) -> &str;
}
