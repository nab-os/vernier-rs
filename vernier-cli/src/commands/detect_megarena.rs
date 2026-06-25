//! Port of the C++ `detectingMegarenaPattern.cpp` example.
//!
//! The original constructs `MegarenaPatternDetector(physicalPeriod, codeSize)`,
//! runs `compute(image)`, and if a pattern is found prints the estimated 2D
//! pose. This is the full absolute path on a real photo: two-direction
//! detection → binary-code extraction → LFSR localization → assembled absolute
//! `(x, y, θ)`.
//!
//! `physical_period` (µm) and `code_size` (LFSR order in bits) mirror the C++
//! constructor arguments — for the paper's target, 9 µm and 12 bits.

use std::path::PathBuf;

use vernier_core::buffer::BufferLayout;
use vernier_core::{Complex32, ComputeBackend, Real};
use vernier_detection::spectrum::analyze_two;
use vernier_pose::absolute::{CoarseDecoder, MegarenaDecoder, detect_orientation, extract_code};
use vernier_pose::{Calibration, periodic};

use crate::backend_select::BackendTask;
use crate::imageio::load_grayscale;

/// Megarena detection parameters.
pub struct DetectMegarena {
    /// Path to the image file.
    pub image_path: PathBuf,
    /// Physical period of the pattern in micrometres (the paper uses 9 µm).
    pub physical_period: f32,
    /// LFSR code size in bits (the paper uses 12).
    pub code_size: u32,
    /// Band-pass filter width in bins (C++ default 3.0).
    pub sigma: f32,
    /// Inner annulus radius for peak search in bins; 0 = no lower limit (C++ default 20).
    pub min_frequency: usize,
    /// Outer annulus radius for peak search in bins; 0 = no upper limit (C++ default 500).
    pub max_frequency: usize,
    /// Gaussian blur sigma applied to magnitude before peak search (C++ default 0.5).
    pub smoothing_sigma: f32,
    /// Optional path prefix for debug overlay images (spectrum + decoded cells).
    pub debug_image: Option<std::path::PathBuf>,
    /// Print intermediate detection details (carriers, planes, orientation).
    pub verbose: bool,
}

/// What a DetectMegarena run reports.
pub struct DetectMegarenaReport {
    /// Backend name.
    pub backend: String,
    /// Pose x
    pub x: Real,
    /// Pose y
    pub y: Real,
    /// Theta
    pub theta: Real,
    /// K3
    pub k3: u8,
}

impl BackendTask for DetectMegarena {
    type Output = DetectMegarenaReport;

    fn run<B: ComputeBackend>(&self, backend: &mut B) -> DetectMegarenaReport {
        let img = load_grayscale(&self.image_path).unwrap();
        let (width, height) = (img.width, img.height);

        let layout = BufferLayout::packed(width, height);
        let complex: Vec<Complex32> = img.data.iter().map(|&v| Complex32::new(v, 0.0)).collect();
        let mut buf = backend
            .upload(&complex, layout)
            .map_err(|e| format!("upload failed: {e:?}"))
            .unwrap();

        // Two-direction detection -> phase planes + phase maps.
        let detection = analyze_two(
            backend,
            &mut buf,
            self.sigma as vernier_core::Real,
            self.min_frequency,
            self.max_frequency,
            self.smoothing_sigma as vernier_core::Real,
        )
        .map_err(|e| format!("detection failed: {e:?}"))
        .unwrap();

        let calib = Calibration::new(self.physical_period as vernier_core::Real, width, height);
        let fine = periodic::estimate(&detection.dir1.plane, &detection.dir2.plane, &calib);

        let intensity: Vec<vernier_core::Real> =
            img.data.iter().map(|&v| v as vernier_core::Real).collect();
        let code = extract_code(&detection, &intensity, self.code_size).unwrap();

        let decoder = MegarenaDecoder::new(
            self.code_size,
            code.x_window.clone(),
            code.y_window.clone(),
            code.k3,
        )
        .ok_or_else(|| format!("unsupported code size {}", self.code_size))
        .unwrap();

        let orders = decoder.decode().unwrap();
        // C++ absolute position formula (MegarenaPatternDetector::draw):
        //   periodShift = bitSequence DOT-period index of image centre
        //               = 3*(K_center + order)
        //   x = −physicalPeriod × (c/(2π) + periodShift)
        //
        // Swap/flip follows C++ computeAbsolutePose rotation logic:
        //   swap x↔y when exactly one MSB is 0 (rotate90 or rotate270)
        //   negate c for the direction whose MSB=0 (C++ plane.flip() negates a,b,c)
        use vernier_core::scalar::consts::TAU;
        let period = self.physical_period as vernier_core::Real;
        let swap = code.msb1 != code.msb2;
        let (ps_x, ps_y, c_x, c_y, msb_x, msb_y) = if swap {
            (
                code.y_periodshift,
                code.x_periodshift,
                detection.dir2.plane.c,
                detection.dir1.plane.c,
                code.msb2,
                code.msb1,
            )
        } else {
            (
                code.x_periodshift,
                code.y_periodshift,
                detection.dir1.plane.c,
                detection.dir2.plane.c,
                code.msb1,
                code.msb2,
            )
        };
        let c_x_eff = if msb_x { c_x } else { -c_x };
        let c_y_eff = if msb_y { c_y } else { -c_y };
        let abs_x = -(period * (c_x_eff / TAU + ps_x as vernier_core::Real));
        let abs_y = -(period * (c_y_eff / TAU + ps_y as vernier_core::Real));

        DetectMegarenaReport {
            backend: backend.name().to_string(),
            x: abs_x,
            y: abs_y,
            theta: fine.theta,
            k3: orders.k3,
        }
    }
}

/// Builds a sibling path by appending a suffix to the prefix's file stem.
/// e.g. prefix "out/dbg" + "_spectrum.png" -> "out/dbg_spectrum.png".
fn with_suffix(prefix: &std::path::Path, suffix: &str) -> std::path::PathBuf {
    let mut s = prefix.as_os_str().to_os_string();
    s.push(suffix);
    std::path::PathBuf::from(s)
}
