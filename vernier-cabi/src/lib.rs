//! C ABI for vernier-rs.
//!
//! Create a handle with `vernier_detector_new` (CPU) or
//! `vernier_detector_new_cuda` (GPU), call the detection functions, then
//! release the handle with `vernier_detector_free`. Each function is
//! thread-safe in the sense that separate handles may be used concurrently;
//! a single handle must not be used from multiple threads simultaneously.

use std::cell::RefCell;
use std::ffi::{CString, c_char};

use vernier_core::buffer::BufferLayout;
use vernier_core::image::GrayImage;
use vernier_core::scalar::consts::TAU;
use vernier_core::{Complex32, Real};
use vernier_cpu::CpuBackend;
use vernier_detection::spectrum::{Detection, analyze_two};
use vernier_pose::absolute::{CoarseDecoder, MegarenaDecoder, extract_code};
use vernier_pose::{Calibration, periodic};

// ─── Thread-local error storage ──────────────────────────────────────────────

thread_local! {
    static LAST_ERROR: RefCell<Option<CString>> = RefCell::new(None);
}

fn set_last_error(msg: impl std::fmt::Display) {
    let s = msg.to_string();
    LAST_ERROR.with(|e| {
        *e.borrow_mut() = CString::new(s).ok();
    });
}

fn clear_last_error() {
    LAST_ERROR.with(|e| *e.borrow_mut() = None);
}

/// Returns the last error message on this thread, or NULL if the last call
/// succeeded.
///
/// The pointer is valid until the next vernier call on this thread.
#[unsafe(no_mangle)]
pub extern "C" fn vernier_last_error() -> *const c_char {
    LAST_ERROR.with(|e| {
        e.borrow()
            .as_ref()
            .map_or(std::ptr::null(), |s| s.as_ptr())
    })
}

// ─── Backend enum ─────────────────────────────────────────────────────────────

enum BackendInner {
    Cpu(CpuBackend),
    #[cfg(feature = "cuda")]
    Cuda(vernier_cuda::CudaBackend),
}

impl BackendInner {
    fn analyze_two(
        &self,
        data: &[Complex32],
        layout: BufferLayout,
        sigma: Real,
        min_frequency: usize,
        max_frequency: usize,
        smoothing_sigma: Real,
    ) -> vernier_core::Result<Detection> {
        match self {
            BackendInner::Cpu(b) => {
                analyze_two(b, data, layout, sigma, min_frequency, max_frequency, smoothing_sigma)
            }
            #[cfg(feature = "cuda")]
            BackendInner::Cuda(b) => {
                analyze_two(b, data, layout, sigma, min_frequency, max_frequency, smoothing_sigma)
            }
        }
    }
}

// ─── Detector handle ─────────────────────────────────────────────────────────

/// Opaque handle to a Vernier detector. Create with `vernier_detector_new`
/// (CPU) or `vernier_detector_new_cuda` (GPU); free with
/// `vernier_detector_free`.
pub struct VernierDetector {
    backend: BackendInner,
}

/// Creates a CPU-backed detector. Returns NULL on allocation failure.
///
/// Must be freed with `vernier_detector_free`.
#[unsafe(no_mangle)]
pub extern "C" fn vernier_detector_new() -> *mut VernierDetector {
    Box::into_raw(Box::new(VernierDetector {
        backend: BackendInner::Cpu(CpuBackend::new()),
    }))
}

/// Creates a CUDA-backed detector. Returns NULL if no CUDA device is
/// available or if the library was not compiled with CUDA support (check
/// `vernier_last_error` for details).
///
/// Must be freed with `vernier_detector_free`.
#[unsafe(no_mangle)]
pub extern "C" fn vernier_detector_new_cuda() -> *mut VernierDetector {
    #[cfg(feature = "cuda")]
    {
        match vernier_cuda::CudaBackend::new() {
            Ok(b) => Box::into_raw(Box::new(VernierDetector {
                backend: BackendInner::Cuda(b),
            })),
            Err(e) => {
                set_last_error(e);
                std::ptr::null_mut()
            }
        }
    }
    #[cfg(not(feature = "cuda"))]
    {
        set_last_error("CUDA support not compiled in (rebuild with --features cuda)");
        std::ptr::null_mut()
    }
}

/// Frees a detector. Passing NULL is a no-op.
#[unsafe(no_mangle)]
pub extern "C" fn vernier_detector_free(det: *mut VernierDetector) {
    if !det.is_null() {
        unsafe { drop(Box::from_raw(det)) };
    }
}

// ─── Result type ─────────────────────────────────────────────────────────────

/// Pose returned by all detection functions.
///
/// Check `found` before reading `x`, `y`, `theta`. On failure (`found == 0`)
/// call `vernier_last_error()` for a description.
#[repr(C)]
pub struct VernierPose {
    pub x: f32,
    pub y: f32,
    pub theta: f32,
    /// 1 on success, 0 on failure.
    pub found: i32,
}

impl VernierPose {
    fn not_found() -> Self {
        Self { x: 0.0, y: 0.0, theta: 0.0, found: 0 }
    }
}

// ─── Shared image setup ───────────────────────────────────────────────────────

fn make_image(pixels: *const f32, width: usize, height: usize) -> Option<GrayImage> {
    if pixels.is_null() {
        return None;
    }
    let slice = unsafe { std::slice::from_raw_parts(pixels, width * height) };
    GrayImage::from_vec(width, height, slice.to_vec())
}

// ─── Periodic (relative / fine) detection ────────────────────────────────────

/// Periodic (relative) detection: recovers `x`, `y` modulo the pattern period
/// and the in-image orientation `theta`.
///
/// - `det`              — handle from `vernier_detector_new[_cuda]` (must not be NULL).
/// - `pixels`           — row-major f32 image, `width × height` elements in [0, 1].
/// - `period`           — pattern spatial period in physical units.
/// - `sigma`            — bandpass filter half-width in frequency bins.
/// - `min_frequency`    — inner annulus radius for peak search (0 = no limit).
/// - `max_frequency`    — outer annulus radius for peak search (0 = no limit).
/// - `smoothing_sigma`  — Gaussian blur on the magnitude spectrum before peak
///                        search; 0 disables blurring.
///
/// Returns a pose with `found == 0` on failure.
#[unsafe(no_mangle)]
pub extern "C" fn vernier_detect_periodic(
    det: *mut VernierDetector,
    pixels: *const f32,
    width: usize,
    height: usize,
    period: f32,
    sigma: f32,
    min_frequency: usize,
    max_frequency: usize,
    smoothing_sigma: f32,
) -> VernierPose {
    clear_last_error();

    let det = match unsafe { det.as_ref() } {
        Some(d) => d,
        None => {
            set_last_error("null detector pointer");
            return VernierPose::not_found();
        }
    };

    let image = match make_image(pixels, width, height) {
        Some(img) => img,
        None => {
            set_last_error("null or mismatched pixels pointer");
            return VernierPose::not_found();
        }
    };

    let detection = match det.backend.analyze_two(
        &image.to_complex(),
        image.layout(),
        sigma as Real,
        min_frequency,
        max_frequency,
        smoothing_sigma as Real,
    ) {
        Ok(d) => d,
        Err(e) => {
            set_last_error(e);
            return VernierPose::not_found();
        }
    };

    let calib = Calibration::new(period as Real, width, height);
    let pose = periodic::estimate(&detection.dir1.plane, &detection.dir2.plane, &calib);

    VernierPose { x: pose.x, y: pose.y, theta: pose.theta, found: 1 }
}

// ─── Megarena absolute detection ─────────────────────────────────────────────

/// Megarena absolute detection: recovers an unambiguous `(x, y, theta)` using
/// the LFSR binary code embedded in the pattern.
///
/// - `det`              — handle from `vernier_detector_new[_cuda]` (must not be NULL).
/// - `pixels`           — row-major f32 image, `width × height` elements in [0, 1].
/// - `physical_period`  — pattern spatial period in micrometres (9 µm for the
///                        reference pattern).
/// - `code_size`        — LFSR order in bits (12 for the reference pattern).
/// - `sigma`            — bandpass filter half-width in frequency bins.
/// - `min_frequency`    — inner annulus radius for peak search (0 = no limit).
/// - `max_frequency`    — outer annulus radius for peak search (0 = no limit).
/// - `smoothing_sigma`  — Gaussian blur on the magnitude spectrum before peak
///                        search; 0 disables blurring.
///
/// Returns a pose with `found == 0` on failure.
#[unsafe(no_mangle)]
pub extern "C" fn vernier_detect_megarena(
    det: *mut VernierDetector,
    pixels: *const f32,
    width: usize,
    height: usize,
    physical_period: f32,
    code_size: u32,
    sigma: f32,
    min_frequency: usize,
    max_frequency: usize,
    smoothing_sigma: f32,
) -> VernierPose {
    clear_last_error();

    let det = match unsafe { det.as_ref() } {
        Some(d) => d,
        None => {
            set_last_error("null detector pointer");
            return VernierPose::not_found();
        }
    };

    let image = match make_image(pixels, width, height) {
        Some(img) => img,
        None => {
            set_last_error("null or mismatched pixels pointer");
            return VernierPose::not_found();
        }
    };

    let complex = image.to_complex();
    let layout = image.layout();

    let detection = match det.backend.analyze_two(
        &complex,
        layout,
        sigma as Real,
        min_frequency,
        max_frequency,
        smoothing_sigma as Real,
    ) {
        Ok(d) => d,
        Err(e) => {
            set_last_error(e);
            return VernierPose::not_found();
        }
    };

    let calib = Calibration::new(physical_period as Real, width, height);
    let fine = periodic::estimate(&detection.dir1.plane, &detection.dir2.plane, &calib);

    let intensity: Vec<Real> = image.as_slice().iter().copied().collect();

    let code = match extract_code(&detection, &intensity, code_size) {
        Some(c) => c,
        None => {
            set_last_error("code extraction failed: pattern may be occluded or too small");
            return VernierPose::not_found();
        }
    };

    let decoder = match MegarenaDecoder::new(
        code_size,
        code.x_window.clone(),
        code.y_window.clone(),
        code.k3,
    ) {
        Some(d) => d,
        None => {
            set_last_error(format!("unsupported LFSR code size {code_size}"));
            return VernierPose::not_found();
        }
    };

    if decoder.decode().is_none() {
        set_last_error("LFSR decode failed: windows did not localize in the sequence");
        return VernierPose::not_found();
    }

    let period = physical_period as Real;
    let swap = code.msb1 != code.msb2;

    let (x_c, x_ps, x_msb) = if swap {
        (detection.dir2.plane.c, code.y_periodshift, code.msb2)
    } else {
        (detection.dir1.plane.c, code.x_periodshift, code.msb1)
    };
    let (y_c, y_ps, y_msb) = if swap {
        (detection.dir1.plane.c, code.x_periodshift, code.msb1)
    } else {
        (detection.dir2.plane.c, code.y_periodshift, code.msb2)
    };

    let flip_c = |c: Real, msb: bool| -> Real { if msb { c } else { -c } };
    let x = -(period * (flip_c(x_c, x_msb) / TAU + x_ps as Real));
    let y = -(period * (flip_c(y_c, y_msb) / TAU + y_ps as Real));

    VernierPose { x, y, theta: fine.theta, found: 1 }
}
