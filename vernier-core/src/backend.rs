//! The [`ComputeBackend`] and [`ComputeJob`] traits.
//!
//! `vernier-cpu` implements these with `rustfft`/`ndarray`; `vernier-gpu`
//! implements the same traits with Vulkano compute.
//!
//! ## Lifecycle
//!
//! ```text
//! backend.upload(...)         → Buffer2D       (immediate; staging transfer)
//! let mut job = backend.begin()?
//! job.fft2d(&mut buf)?        → queues work
//! job.peak_search(...)        → queues work, returns a Buffer2D handle
//! job.submit()?               → executes everything queued so far
//! backend.download(&buf)?     → reads result back to host (immediate)
//! ```
//!
//! A job is a unit of work that may span multiple operations. On the CPU all
//! operations are synchronous and `submit()` is a no-op. On the GPU operations
//! are recorded into a command buffer and `submit()` dispatches and waits.
//!
//! ## The `copy_buffer` method
//!
//! When the same spectrum must be processed in two independent ways (e.g. two
//! carrier directions), use `job.copy_buffer(src)` to obtain a distinct GPU
//! buffer before applying destructive in-place operations. On the CPU this is
//! a plain `Vec` copy; on the GPU it queues a `copy_buffer` command.

use crate::buffer::{Buffer2D, BufferLayout};
use crate::complex::Complex32;
use crate::error::Result;
use crate::scalar::Real;

/// A unit of queued work produced by [`ComputeBackend::begin`].
///
/// Each method records an operation; [`submit`](ComputeJob::submit) executes them.
/// On the CPU every method executes immediately and `submit` is a no-op.
pub trait ComputeJob {
    /// The buffer type this job operates on (must match the backend's `Buffer2D`).
    type Buffer2D;

    /// Uploads `data` into a new device buffer as part of this job's command
    /// stream, queuing the staging copy alongside subsequent compute dispatches
    /// rather than in a separate blocking submission.
    fn upload(&mut self, data: &[Complex32], layout: BufferLayout) -> Result<Self::Buffer2D>;

    /// Produces a deep copy of `src` as a new, independently-writable buffer.
    ///
    /// On GPU this queues a `copy_buffer` command; on CPU it clones the underlying
    /// storage. Use this before any destructive in-place operation when the same
    /// source buffer must be processed in multiple independent ways.
    fn copy_buffer(&mut self, src: &Self::Buffer2D) -> Result<Self::Buffer2D>;

    /// In-place 2D forward FFT.
    fn fft2d(&mut self, buf: &mut Self::Buffer2D) -> Result<()>;

    /// In-place 2D inverse FFT.
    fn ifft2d(&mut self, buf: &mut Self::Buffer2D) -> Result<()>;

    /// Gaussian band-pass filter in-place, centred on frequency bin `(cx, cy)`.
    fn bandpass_filter(
        &mut self,
        buf: &mut Self::Buffer2D,
        cx: usize,
        cy: usize,
        sigma: Real,
    ) -> Result<()>;

    /// Gaussian band-pass filter reading the carrier bin from a GPU-resident
    /// peaks buffer (the output of `peak_search`).  `direction` selects which
    /// pair: 0 → `peaks[0..1]` = (cx1, cy1), 1 → `peaks[2..3]` = (cx2, cy2).
    ///
    /// On the CPU this reads the coordinates synchronously from the slice and
    /// delegates to `bandpass_filter`.  On the GPU it binds the peaks buffer
    /// directly and avoids a host round-trip.
    fn bandpass_from_peaks(
        &mut self,
        buf: &mut Self::Buffer2D,
        peaks: &Self::Buffer2D,
        direction: u32,
        sigma: Real,
    ) -> Result<()>;

    /// Annulus mask: zeroes bins outside `[min_frequency, max_frequency]` from DC.
    fn filter(
        &mut self,
        buf: &mut Self::Buffer2D,
        min_frequency: usize,
        max_frequency: usize,
    ) -> Result<()>;

    /// Separable 2D Gaussian blur on the real component, in-place.
    fn gaussian_blur_2d(&mut self, buf: &mut Self::Buffer2D, sigma: Real) -> Result<()>;

    /// Per-pixel `atan2(im, re)` → new buffer with phase in the `.re` lane.
    fn extract_phase(&mut self, buf: &Self::Buffer2D) -> Result<Self::Buffer2D>;

    /// Finds the two carrier peaks and returns a 2×2 buffer holding
    /// `[cx1, cy1, cx2, cy2]` (ordered so direction 1 has the larger signed
    /// column frequency). Returns `None` when no valid peaks are found.
    fn peak_search(
        &mut self,
        buf: &mut Self::Buffer2D,
        min_frequency: usize,
        max_frequency: usize,
        smoothing_sigma: Real,
        sigma: Real,
    ) -> Result<Option<Self::Buffer2D>>;

    /// Computes the phase-plane coefficients (a, b, c) from the complex IFFT
    /// output `buf` using per-pixel phase gradients, without unwrapping.
    ///
    /// Returns a 3-element buffer: `[Complex32(a,0), Complex32(b,0), Complex32(c,0)]`.
    /// `crop_factor ∈ [0,1)` trims edges before accumulation (0.5 = center half).
    fn plane_fit_from_ifft(&mut self, buf: &Self::Buffer2D, crop_factor: Real) -> Result<Self::Buffer2D>;

    /// Executes all queued operations and waits for completion.
    ///
    /// Consumes the job; create a new one via [`ComputeBackend::begin`] for
    /// subsequent work.
    fn submit(self) -> Result<()>;
}

/// A compute backend: creates jobs, uploads data, and downloads results.
pub trait ComputeBackend {
    /// The backend's concrete 2D buffer type.
    type Buffer2D: Buffer2D;

    /// The type of job returned by [`begin`](ComputeBackend::begin).
    type Job<'a>: ComputeJob<Buffer2D = Self::Buffer2D>
    where
        Self: 'a;

    /// Begins a new unit of queued work.
    fn begin(&self) -> Result<Self::Job<'_>>;

    /// Uploads contiguous, row-major complex data into a device buffer.
    fn upload(&self, data: &[Complex32], layout: BufferLayout) -> Result<Self::Buffer2D>;

    /// Downloads a device buffer back to host memory, row-major.
    ///
    /// Must be called after the job that produced `buffer` has been submitted.
    fn download(&self, buffer: &Self::Buffer2D) -> Result<Vec<Complex32>>;

    /// Uploads a real (greyscale) image as complex (imaginary parts zeroed).
    fn upload_real(&self, image: &crate::image::GrayImage) -> Result<Self::Buffer2D> {
        let complex = image.to_complex();
        self.upload(&complex, image.layout())
    }

    /// A short human-readable name for the backend, e.g. `"cpu-rustfft"`.
    fn name(&self) -> &str;
}
