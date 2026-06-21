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
use vernier_core::{Complex32, ComputeBackend};
use vernier_cpu::CpuBackend;
use vernier_detection::spectrum::analyze_two;
use vernier_pose::absolute::{extract_code, CoarseDecoder, MegarenaDecoder};
use vernier_pose::{periodic, Calibration};

use crate::imageio::load_grayscale;

/// Megarena detection parameters (mirrors the C++ constructor).
pub struct DetectMegarena {
    /// Path to the image file.
    pub image_path: PathBuf,
    /// Physical period of the pattern in micrometres (the paper uses 9 µm).
    pub physical_period: f32,
    /// LFSR code size in bits (the paper uses 12).
    pub code_size: u32,
    /// Band-pass filter width in bins.
    pub sigma: f32,
    /// Peak-exclusion radius when finding the second carrier.
    pub exclude_radius: usize,
}

impl DetectMegarena {
    /// Runs the full absolute detection and prints the pose, or reports that no
    /// decodable pattern was found.
    pub fn run(&self) -> Result<(), String> {
        let img = load_grayscale(&self.image_path)?;
        let (w, h) = (img.width, img.height);
        println!(
            "loaded {} ({}x{}), physical period {} µm, code size {} bits",
            self.image_path.display(),
            w,
            h,
            self.physical_period,
            self.code_size
        );

        let backend = CpuBackend::new();
        let layout = BufferLayout::packed(w, h);
        let complex: Vec<Complex32> = img.data.iter().map(|&v| Complex32::new(v, 0.0)).collect();
        let mut buf = backend
            .upload(&complex, layout)
            .map_err(|e| format!("upload failed: {e:?}"))?;

        // Two-direction detection -> phase planes + phase maps.
        let detection = analyze_two(
            &backend,
            &mut buf,
            self.sigma as vernier_core::Real,
            self.exclude_radius,
        )
        .map_err(|e| format!("detection failed: {e:?}"))?;

        // Fine (sub-period) pose from the phase planes.
        // The pixel period is recovered from the carrier peak: period_px =
        // image_size / |peak frequency in cycles across the image|. We use the
        // plane gradient magnitude, which gives cycles-per-pixel directly.
        let calib = Calibration::new(self.physical_period as vernier_core::Real, w, h);
        let fine = periodic::estimate(&detection.dir1.plane, &detection.dir2.plane, &calib);

        // Absolute code extraction + decode.
        let intensity: Vec<vernier_core::Real> =
            img.data.iter().map(|&v| v as vernier_core::Real).collect();
        let code = match extract_code(&detection, &intensity, self.code_size) {
            Some(c) => c,
            None => {
                println!("Pattern not found... (no full {}-bit code window visible; \
                          the image may be too small — need ~{} periods across the field, \
                          or the pattern is occluded/low-contrast)",
                    self.code_size, 3 * self.code_size);
                return Ok(());
            }
        };

        println!(
            "  dir1 peak_bin={:?} plane=(a={:.4} b={:.4} c={:.4})",
            detection.dir1.peak_bin, detection.dir1.plane.a, detection.dir1.plane.b, detection.dir1.plane.c
        );
        println!(
            "  dir2 peak_bin={:?} plane=(a={:.4} b={:.4} c={:.4})",
            detection.dir2.peak_bin, detection.dir2.plane.a, detection.dir2.plane.b, detection.dir2.plane.c
        );
        println!(
            "  extracted: x_window={:?} (first_triple={})",
            code.x_window, code.x_first_triple
        );
        println!(
            "  extracted: y_window={:?} (first_triple={})",
            code.y_window, code.y_first_triple
        );

        let decoder = MegarenaDecoder::new(
            self.code_size,
            code.x_window.clone(),
            code.y_window.clone(),
            0, // quadrant supplied; corner-sync recovery is a documented extension
        )
        .ok_or_else(|| format!("unsupported code size {}", self.code_size))?;

        match decoder.decode() {
            Some(orders) => {
                // Absolute position = cell order * period + fine sub-period part.
                let period = self.physical_period as vernier_core::Real;
                let abs_x = orders.k1 as vernier_core::Real * period + fine.x;
                let abs_y = orders.k2 as vernier_core::Real * period + fine.y;
                println!("Pattern found.");
                println!(
                    "Estimated pose: x={:.4} µm, y={:.4} µm, θ={:.6} rad (quadrant k3={})",
                    abs_x, abs_y, fine.theta, orders.k3
                );
                println!(
                    "  (fine: x={:.4} y={:.4} within cell; orders k1={} k2={})",
                    fine.x, fine.y, orders.k1, orders.k2
                );
            }
            None => {
                println!(
                    "Pattern not found... (code window did not localize in the LFSR \
                     sequence — likely bit-extraction errors from noise/contrast)"
                );
            }
        }
        Ok(())
    }
}
