use vernier_core::buffer::{Buffer2D, BufferLayout};
use vernier_core::{ComputeBackend, ComputeJob, Real, Result, VernierError, Complex32};

use crate::planefit::PhasePlane;

#[derive(Clone, Copy, Debug)]
pub struct DirectionResult {
    pub plane: PhasePlane,
    pub peak: (Real, Real),
    pub peak_bin: (usize, usize),
}

#[derive(Clone, Debug)]
pub struct Detection {
    pub dir1: DirectionResult,
    pub dir2: DirectionResult,
    pub phase1: Vec<Real>,
    pub phase2: Vec<Real>,
    pub width: usize,
    pub height: usize,
}

pub fn forward<B: ComputeBackend>(backend: &B, buffer: &mut B::Buffer2D) -> Result<()> {
    let mut job = backend.begin()?;
    job.fft2d(buffer)?;
    job.submit()
}

fn direction_from_plane_data(
    plane_data: Vec<Complex32>,
    width: usize,
    height: usize,
    peak_x: usize,
    peak_y: usize,
) -> (DirectionResult, Vec<Real>) {
    let a = plane_data[0].re as Real;
    let b = plane_data[1].re as Real;
    let c = plane_data[2].re as Real;
    let plane = PhasePlane { a, b, c };
    let peak = plane.peak_location(width, height);
    let half_width = width as Real / 2.0;
    let half_height = height as Real / 2.0;
    let phase: Vec<Real> = (0..height)
        .flat_map(|row| {
            (0..width)
                .map(move |col| a * (col as Real - half_width) + b * (row as Real - half_height) + c)
        })
        .collect();
    (DirectionResult { plane, peak, peak_bin: (peak_x, peak_y) }, phase)
}

pub fn analyze_direction<B: ComputeBackend>(
    backend: &B,
    mut spectrum: B::Buffer2D,
    sigma: Real,
) -> Result<DirectionResult> {
    let layout = spectrum.layout();
    let (width, height) = (layout.width, layout.height);

    let (peaks_buffer, planes_buffer) = {
        let mut job = backend.begin()?;
        let peaks_buffer = job
            .peak_search(&mut spectrum, 0, 0, 0.5, sigma)?
            .ok_or_else(|| VernierError::Backend("no carrier peaks found".into()))?;
        let planes_buffer = job.spectral_plane_fit_two(&spectrum, &peaks_buffer, sigma)?;
        job.submit()?;
        (peaks_buffer, planes_buffer)
    };

    let peaks_data = backend.download(&peaks_buffer)?;
    let (peak_x, peak_y) = (peaks_data[0].re as usize, peaks_data[1].re as usize);

    let planes_data = backend.download(&planes_buffer)?;
    let plane = PhasePlane {
        a: planes_data[0].re as Real,
        b: planes_data[1].re as Real,
        c: planes_data[2].re as Real,
    };
    let peak = plane.peak_location(width, height);
    Ok(DirectionResult { plane, peak, peak_bin: (peak_x, peak_y) })
}

pub fn analyze_two<B: ComputeBackend>(
    backend: &B,
    data: &[Complex32],
    layout: BufferLayout,
    sigma: Real,
    min_frequency: usize,
    max_frequency: usize,
    smoothing_sigma: Real,
) -> Result<Detection> {
    let (width, height) = (layout.width, layout.height);

    let (peaks_buffer, plane_buffer1, plane_buffer2) = {
        let mut job = backend.begin()?;
        let mut buffer = job.upload(data, layout)?;
        job.fft2d(&mut buffer)?;
        let peaks_buffer = job
            .peak_search(&mut buffer, min_frequency, max_frequency, smoothing_sigma, sigma)?
            .ok_or_else(|| VernierError::Backend("no carrier peaks found in spectrum".into()))?;
        let mut spec1 = job.copy_buffer(&buffer)?;
        let mut spec2 = job.copy_buffer(&buffer)?;
        job.bandpass_from_peaks(&mut spec1, &peaks_buffer, 0, sigma)?;
        job.bandpass_from_peaks(&mut spec2, &peaks_buffer, 1, sigma)?;
        job.ifft2d(&mut spec1)?;
        job.ifft2d(&mut spec2)?;
        let plane_fit1 = job.plane_fit_from_ifft(&spec1, 0.5)?;
        let plane_fit2 = job.plane_fit_from_ifft(&spec2, 0.5)?;
        job.submit()?;
        (peaks_buffer, plane_fit1, plane_fit2)
    };

    let peaks_data = backend.download(&peaks_buffer)?;
    let (peak_x1, peak_y1) = (peaks_data[0].re as usize, peaks_data[1].re as usize);
    let (peak_x2, peak_y2) = (peaks_data[2].re as usize, peaks_data[3].re as usize);

    let (dir1, phase1) =
        direction_from_plane_data(backend.download(&plane_buffer1)?, width, height, peak_x1, peak_y1);
    let (dir2, phase2) =
        direction_from_plane_data(backend.download(&plane_buffer2)?, width, height, peak_x2, peak_y2);

    Ok(Detection { dir1, dir2, phase1, phase2, width, height })
}
