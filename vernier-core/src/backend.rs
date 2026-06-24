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

    /// Extracts the wrapped phase `atan2(im, re)` of each element into a new
    /// buffer (stored in the `re` lane; `im` set to zero), staying on-device.
    ///
    /// In the real pipeline this runs on the **spatial-domain complex field**
    /// produced by [`ifft2d`](ComputeBackend::ifft2d) after a single lobe has
    /// been isolated by [`bandpass_filter`](ComputeBackend::bandpass_filter) —
    /// its argument is the wrapped phase map of one pattern direction (André et
    /// al. 2021, Fig. 2c). It is the embarrassingly-parallel per-pixel stage: a
    /// one-line kernel on the GPU, a `map` on the CPU. Phase *unwrapping* and the
    /// least-squares *plane fit* that follow are data-dependent and live in
    /// `vernier-detection`, not here.
    fn extract_phase(&self, buffer: &Self::Buffer2D) -> Result<Self::Buffer2D>;

    /// Applies an annulus mask to put aside low and high frequencies
    fn annulus_mask(&self, buffer: &mut Self::Buffer2D, min_frequency: usize, max_frequency: usize);

    /// Ports C++ `PatternPhase::peaksSearch` to Rust.
    /// Applies an annulus mask + Gaussian blur, finds two peaks via half-plane search and
    /// angular cone isolation, then orders them by signed column frequency.
    ///
    /// `buffer` is not modified — implementations work on an internal clone so the
    /// original complex FFT spectrum is preserved for the subsequent `analyze_at` calls.
    ///
    /// Returns `None` if fewer than two valid bins exist (should not happen on any
    /// non-trivial image).
    fn peak_search(
        &self,
        buffer: &Self::Buffer2D,
        min_frequency: usize,
        max_frequency: usize,
        smoothing_sigma: Real,
        sigma: Real,
    ) -> Option<((usize, usize), (usize, usize))>;

    /// Returns the flat index and magnitude of the largest-magnitude element.
    ///
    /// A reduction — a parallel `reduce` on the GPU, an `iter` scan on the CPU.
    /// Used to locate the fundamental-frequency peak after [`fft2d`].
    /// The index is row-major per the buffer's [`BufferLayout`].
    ///
    /// [`fft2d`]: ComputeBackend::fft2d
    fn argmax_magnitude(&self, buffer: &Self::Buffer2D) -> Result<(usize, Real)>;

    /// Like [`argmax_magnitude`](ComputeBackend::argmax_magnitude), but restricts
    /// the search to one half of frequency space so it deterministically picks
    /// one lobe of each conjugate pair.
    ///
    /// A real image's spectrum is Hermitian: every peak at signed frequency
    /// `(fx, fy)` has an equal-magnitude conjugate at `(-fx, -fy)`. A plain
    /// global argmax breaks that tie arbitrarily, and landing on the negative
    /// twin negates the recovered phase gradient — reflecting the measured
    /// orientation to `π − θ`. To avoid this, the search is confined to the
    /// canonical half-plane `sfx > 0, or (sfx == 0 and sfy > 0)`, where `sfx`,
    /// `sfy` are the *signed* frequencies (bins above N/2 are negative). DC is
    /// excluded.
    ///
    /// On the GPU this is the same reduction as `argmax_magnitude` with a
    /// per-bin predicate — still a trivial masked reduction kernel.
    /// Finds the largest-magnitude bin in the canonical half-plane, **excluding
    /// a low-frequency disk** of radius `min_radius` bins around DC.
    ///
    /// The low-frequency exclusion is essential on real images: lighting falloff,
    /// vignetting, and overall brightness put enormous energy in the bins
    /// immediately around DC — often far exceeding the carrier peak. Excluding
    /// only the single DC bin (the naive approach) makes the search lock onto
    /// this lighting content (e.g. bin (0,1)) instead of the pattern carrier. A
    /// `min_radius` that clears the lighting skirt but stays well below the
    /// carrier radius (the carrier sits at `image_size / period_px` bins)
    /// recovers the true peak. `min_radius = 0` reduces to DC-only exclusion.
    ///
    /// See [`argmax_magnitude_halfplane`](ComputeBackend::argmax_magnitude_halfplane)
    /// for the half-plane / conjugate-disambiguation rationale.
    fn argmax_magnitude_halfplane(
        &self,
        buffer: &Self::Buffer2D,
        min_radius: usize,
    ) -> Result<(usize, Real)>;

    /// Like [`argmax_magnitude_halfplane`](ComputeBackend::argmax_magnitude_halfplane),
    /// but ignores bins within `radius` (in signed-frequency bins, Chebyshev
    /// distance) of `(exclude_x, exclude_y)`.
    ///
    /// Used to find the *second* carrier peak of a 2D grid: after locating the
    /// first direction's peak, its neighborhood is masked so the search returns
    /// the perpendicular direction's peak instead of a sideband of the first.
    /// `exclude_x`/`exclude_y` are bin indices (not signed); the comparison is
    /// done in signed-frequency space so the mask follows the lobe correctly
    /// near the Nyquist edge.
    fn argmax_magnitude_halfplane_excluding(
        &self,
        buffer: &Self::Buffer2D,
        exclude_x: usize,
        exclude_y: usize,
        radius: usize,
        min_radius: usize,
    ) -> Result<(usize, Real)>;

    /// Separable 2D Gaussian blur on a real-valued array, in place.
    fn gaussian_blur_2d(
        &self,
        buffer: &mut Self::Buffer2D,
        width: usize,
        height: usize,
        sigma: Real,
    );

    /// Argmax in the positive-fy half-plane (`sfy >= 0`), mirroring C++
    /// `PatternPhase::peaksSearch` which zeros the top half of the shifted spectrum
    /// before calling `maxCoeff`.  The C++ bottom half is `sfy >= 0` (rows ≥ height/2
    /// in the shifted spectrum).  This matches image 2's case where the near-vertical
    /// carrier at sfx=1,sfy=-220 is a stronger stray than the true horizontal carrier
    /// at sfx=73,sfy=2 — excluding sfy<0 prevents that stray from winning.
    fn halfplane_argmax(
        &self,
        buffer: &Self::Buffer2D,
        width: usize,
        height: usize,
    ) -> Option<(usize, usize)>;

    /// Argmax in the positive-fy half-plane (`sfy >= 0`), excluding bins whose
    /// direction from DC is within `half_width` radians of `center_angle`.
    ///
    fn halfplane_argmax_angular_excl(
        &self,
        buffer: &Self::Buffer2D,
        width: usize,
        height: usize,
        center_angle: Real,
        half_width: Real,
    ) -> Option<(usize, usize)>;

    /// Applies a Gaussian band-pass filter centered on a single frequency lobe,
    /// in place on a frequency-domain buffer.
    ///
    /// This is the step that isolates one spectral lobe before the inverse FFT
    /// (André et al. 2020/2021). The filter is a Gaussian in frequency space:
    /// each bin `(fx, fy)` is multiplied by
    /// `exp(-((fx-cx)² + (fy-cy)²) / (2σ²))`, where `(cx, cy)` is the lobe center
    /// and `sigma` its width in bins.
    ///
    /// Critically, only the lobe around `(center_x, center_y)` is kept — its
    /// complex-conjugate mirror is *not* re-added. Excluding the conjugate is
    /// what breaks the Hermitian symmetry of a real image's spectrum, so the
    /// subsequent [`ifft2d`](ComputeBackend::ifft2d) yields a genuinely complex
    /// field whose argument is the wrapped phase. Keeping both lobes would give a
    /// real-valued (cosine) result with no usable phase.
    ///
    /// On the GPU this is a per-bin multiply — a trivial kernel. On the CPU it is
    /// a `map` over the buffer.
    fn bandpass_filter(
        &self,
        buffer: &mut Self::Buffer2D,
        center_x: usize,
        center_y: usize,
        sigma: Real,
    ) -> Result<()>;

    /// A short human-readable name for the active backend (e.g. `"cpu-rustfft"`,
    /// `"gpu-vulkano"`). For logs and the benchmark harness.
    fn name(&self) -> &str;
}
