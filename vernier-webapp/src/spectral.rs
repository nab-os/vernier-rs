//! One pass of the spectral pipeline, keeping every stage it passes through.
//!
//! This is the whole measurement chain the library runs, compiled to
//! WebAssembly and executed in the page: there is no server behind it.
//!
//! `vernier_spectral::analyze_two` does the same work but hands back only the
//! fitted planes and the phase maps: the spectrum and the band-passed lobes are
//! intermediate buffers it drops. Those are most of what there is to look at
//! here, so this runs the same sequence of backend operations and copies the
//! two spectra out on the way past. The extra cost is two buffer copies.

use vernier_core::buffer::BufferLayout;
use vernier_core::{Complex32, ComputeBackend, ComputeJob, Real, Result};
use vernier_spectral::planefit::{PhasePlane, fit_plane_to_unwrapped};
use vernier_spectral::spectrum::{Detection, DirectionResult, REGRESSION_CROP_FACTOR};
use vernier_spectral::unwrap::quarters_unwrap_phase;

/// Everything one pass produces, in the order the pipeline produces it.
pub struct Stages {
    /// Magnitude of the raw spectrum, centred, on a linear scale. The display
    /// decides whether to compress it: keeping the honest magnitudes here means
    /// switching scales costs a texture upload rather than another transform.
    pub spectrum: Vec<f32>,
    /// Magnitude of the two band-passed lobes together, centred and linear.
    pub filtered: Vec<f32>,
    /// The pattern rebuilt from the two carriers alone.
    pub reconstruction: Vec<f32>,
    /// Wrapped phase of each direction, as the planes saw it.
    pub phase1: Vec<f32>,
    pub phase2: Vec<f32>,
    /// The same maps unwrapped: what the plane fit was given, and what the
    /// square indexing downstream needs to know which period it is in.
    pub unwrapped1: Vec<Real>,
    pub unwrapped2: Vec<Real>,
    /// Peak positions in centred spectrum coordinates, in bins from the middle.
    pub peaks: [(Real, Real); 2],
    /// The same peaks as raw FFT bins, which is what a [`Detection`] carries.
    pub peak_bins: [(usize, usize); 2],
    pub planes: [PhasePlane; 2],
}

impl Stages {
    /// Packs this pass into the [`Detection`] the decoders take, so the code
    /// extraction can be handed the very same object `vernier_spectral::
    /// spectrum::analyze_two` would give it — this module being a
    /// stage-keeping copy of that function, the two agree by construction.
    pub fn detection(&self, size: usize) -> Detection {
        let direction = |plane: PhasePlane, peak_bin: (usize, usize)| DirectionResult {
            plane,
            peak: plane.peak_location(size, size),
            peak_bin,
        };
        Detection {
            dir1: direction(self.planes[0], self.peak_bins[0]),
            dir2: direction(self.planes[1], self.peak_bins[1]),
            phase1: self.unwrapped1.clone(),
            phase2: self.unwrapped2.clone(),
            width: size,
            height: size,
        }
    }
}

/// Wraps a frequency index into the signed offset from DC that display and
/// geometry both want: bins past the halfway point are negative frequencies.
fn signed(index: usize, n: usize) -> isize {
    let (index, n) = (index as isize, n as isize);
    if index > n / 2 { index - n } else { index }
}

/// Moves DC from the corner to the middle, so a spectrum reads the way it is
/// always drawn, and takes the magnitude. The scale stays linear here; see
/// [`Stages::spectrum`].
fn centre_spectrum(source: &[Complex32], size: usize) -> Vec<f32> {
    let half = size / 2;
    let mut out = vec![0.0f32; size * size];
    for fy in 0..size {
        for fx in 0..size {
            let value = source[fy * size + fx];
            let magnitude = (value.re * value.re + value.im * value.im).sqrt();
            let dx = (fx + half) % size;
            let dy = (fy + half) % size;
            out[dy * size + dx] = magnitude;
        }
    }
    out
}

/// Runs the pipeline over one image.
///
/// Returns `Ok(None)` when the peak search finds nothing to work with, which is
/// an ordinary outcome — the pattern can be zoomed until its carriers fall
/// inside the rejected low-frequency disc — and not a failure.
pub fn run<B: ComputeBackend>(
    backend: &B,
    image: &[f32],
    size: usize,
    sigma: Real,
    min_frequency: usize,
    max_frequency: usize,
    smoothing_sigma: Real,
) -> Result<Option<Stages>> {
    let layout = BufferLayout::packed(size, size);
    let data: Vec<Complex32> = image.iter().map(|&v| Complex32::new(v, 0.0)).collect();

    let (raw, filtered1, filtered2, field1, field2, peak_bins) = {
        let mut job = backend.begin()?;
        let mut buffer = job.upload(&data, layout)?;
        job.fft2d(&mut buffer)?;

        let peaks = match job.peak_search(
            &mut buffer,
            min_frequency,
            max_frequency,
            smoothing_sigma,
            sigma,
        )? {
            Some(peaks) => peaks,
            None => return Ok(None),
        };

        let mut field1 = job.copy_buffer(&buffer)?;
        let mut field2 = job.copy_buffer(&buffer)?;
        job.bandpass_from_peaks(&mut field1, &peaks, 0, sigma)?;
        job.bandpass_from_peaks(&mut field2, &peaks, 1, sigma)?;

        // Kept before the inverse transform: afterwards these buffers hold the
        // spatial field, and the filtered spectrum is gone.
        let filtered1 = job.copy_buffer(&field1)?;
        let filtered2 = job.copy_buffer(&field2)?;

        job.ifft2d(&mut field1)?;
        job.ifft2d(&mut field2)?;
        job.submit()?;
        (buffer, filtered1, filtered2, field1, field2, peaks)
    };

    let peaks_data = backend.download(&peak_bins)?;
    let bins = [
        (peaks_data[0].re as usize, peaks_data[1].re as usize),
        (peaks_data[2].re as usize, peaks_data[3].re as usize),
    ];
    let peaks = [
        (signed(bins[0].0, size) as Real, signed(bins[0].1, size) as Real),
        (signed(bins[1].0, size) as Real, signed(bins[1].1, size) as Real),
    ];

    let spectrum = centre_spectrum(&backend.download(&raw)?, size);

    let lobes1 = backend.download(&filtered1)?;
    let lobes2 = backend.download(&filtered2)?;
    let combined: Vec<Complex32> =
        lobes1.iter().zip(&lobes2).map(|(&a, &b)| a + b).collect();
    let filtered = centre_spectrum(&combined, size);

    let spatial1 = backend.download(&field1)?;
    let spatial2 = backend.download(&field2)?;

    let wrapped1: Vec<Real> = spatial1.iter().map(|c| c.arg() as Real).collect();
    let wrapped2: Vec<Real> = spatial2.iter().map(|c| c.arg() as Real).collect();

    let mut unwrapped1 = wrapped1.clone();
    let mut unwrapped2 = wrapped2.clone();
    quarters_unwrap_phase(&mut unwrapped1, size, size);
    quarters_unwrap_phase(&mut unwrapped2, size, size);
    let planes = [
        fit_plane_to_unwrapped(&unwrapped1, size, size, REGRESSION_CROP_FACTOR),
        fit_plane_to_unwrapped(&unwrapped2, size, size, REGRESSION_CROP_FACTOR),
    ];

    // The same (1+cos)(1+cos)/4 model the patterns are drawn with, rebuilt from
    // the two recovered phases: what the two carriers alone can say, with the
    // code and every harmonic left behind in the discarded spectrum.
    let reconstruction: Vec<f32> = wrapped1
        .iter()
        .zip(&wrapped2)
        .map(|(&p1, &p2)| (((1.0 + p1.cos()) * (1.0 + p2.cos())) / 4.0) as f32)
        .collect();

    Ok(Some(Stages {
        spectrum,
        filtered,
        reconstruction,
        phase1: wrapped1.iter().map(|&v| v as f32).collect(),
        phase2: wrapped2.iter().map(|&v| v as f32).collect(),
        unwrapped1,
        unwrapped2,
        peaks,
        peak_bins: bins,
        planes,
    }))
}
