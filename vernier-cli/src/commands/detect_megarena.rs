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
use vernier_pose::absolute::{CoarseDecoder, MegarenaDecoder, extract_code};
use vernier_pose::{Calibration, periodic};

use crate::imageio::load_grayscale;

/// Megarena detection parameters (mirrors the C++ constructor).
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
    /// Apply a Hann window before the FFT (recommended for real images).
    pub window: bool,
    /// Optional path prefix for debug overlay images (spectrum + decoded cells).
    pub debug_image: Option<std::path::PathBuf>,
}

impl DetectMegarena {
    /// Runs the full absolute detection and prints the pose, or reports that no
    /// decodable pattern was found.
    pub fn run(&self) -> Result<(), String> {
        let img = load_grayscale(&self.image_path)?;
        let (width, height) = (img.width, img.height);
        println!(
            "loaded {} ({}x{}), physical period {} µm, code size {} bits",
            self.image_path.display(),
            width,
            height,
            self.physical_period,
            self.code_size
        );

        let backend = CpuBackend::new();
        let layout = BufferLayout::packed(width, height);
        let complex: Vec<Complex32> = img.data.iter().map(|&v| Complex32::new(v, 0.0)).collect();
        let mut buf = backend
            .upload(&complex, layout)
            .map_err(|e| format!("upload failed: {e:?}"))?;

        // Two-direction detection -> phase planes + phase maps.
        let detection = analyze_two(
            &backend,
            &mut buf,
            self.sigma as vernier_core::Real,
            self.min_frequency,
            self.max_frequency,
            self.smoothing_sigma as vernier_core::Real,
            self.window,
        )
        .map_err(|e| format!("detection failed: {e:?}"))?;

        // Debug overlays: spectrum with detected carriers, image with decoded
        // cells/bits. Generated before the (possibly-failing) decode so you can
        // see what was found even when the code doesn't fully resolve.
        if let Some(prefix) = &self.debug_image {
            let gray: Vec<f64> = img.data.iter().map(|&v| v as f64).collect();
            let spectrum_path = with_suffix(prefix, "_spectrum.png");
            crate::commands::debug_render::render_spectrum_debug(
                &gray,
                width,
                height,
                &detection,
                self.min_frequency,
                &spectrum_path,
            )?;

            let intensity_r: Vec<vernier_core::Real> =
                img.data.iter().map(|&v| v as vernier_core::Real).collect();
            let (xb, yb) = vernier_pose::absolute::decode_bit_maps(&detection, &intensity_r);
            let decode_path = with_suffix(prefix, "_decoded.png");
            crate::commands::debug_render::render_decode_debug(
                &gray,
                width,
                height,
                &detection,
                &xb,
                &yb,
                &decode_path,
            )?;
            println!(
                "debug overlays: {} (carriers on spectrum), {} (decoded cells/bits)",
                spectrum_path.display(),
                decode_path.display()
            );
        }

        // Fine (sub-period) pose from the phase planes.
        // The pixel period is recovered from the carrier peak: period_px =
        // image_size / |peak frequency in cycles across the image|. We use the
        // plane gradient magnitude, which gives cycles-per-pixel directly.
        let calib = Calibration::new(self.physical_period as vernier_core::Real, width, height);
        let fine = periodic::estimate(&detection.dir1.plane, &detection.dir2.plane, &calib);

        // Absolute code extraction + decode.
        let intensity: Vec<vernier_core::Real> =
            img.data.iter().map(|&v| v as vernier_core::Real).collect();
        let code = match extract_code(&detection, &intensity, self.code_size) {
            Some(c) => c,
            None => {
                println!(
                    "Pattern not found... (no full {}-bit code window visible; \
                          the image may be too small — need ~{} periods across the field, \
                          or the pattern is occluded/low-contrast)",
                    self.code_size,
                    3 * self.code_size
                );
                return Ok(());
            }
        };

        let decoder = MegarenaDecoder::new(
            self.code_size,
            code.x_window.clone(),
            code.y_window.clone(),
            code.k3,
        )
        .ok_or_else(|| format!("unsupported code size {}", self.code_size))?;

        match decoder.decode() {
            Some(orders) => {
                // C++ absolute position formula (MegarenaPatternDetector::draw):
                //   periodShift = bitSequence DOT-period index of image centre
                //               = 3*(K_center + order − 1) + 1
                //   x = −physicalPeriod × (c/(2π) + periodShift)
                //
                // Swap/flip follows C++ computeAbsolutePose rotation logic:
                //   swap x↔y when exactly one MSB is 0 (rotate90 or rotate270)
                //   negate c for the direction whose MSB=0 (C++ plane.flip() negates a,b,c)
                use vernier_core::scalar::consts::TAU;
                let period = self.physical_period as vernier_core::Real;
                let order = self.code_size as i64;
                let swap = code.msb1 != code.msb2;
                let (k_cx, k_cy, c_x, c_y, msb_x, msb_y) = if swap {
                    (
                        code.y_k_center,
                        code.x_k_center,
                        detection.dir2.plane.c,
                        detection.dir1.plane.c,
                        code.msb2,
                        code.msb1,
                    )
                } else {
                    (
                        code.x_k_center,
                        code.y_k_center,
                        detection.dir1.plane.c,
                        detection.dir2.plane.c,
                        code.msb1,
                        code.msb2,
                    )
                };
                let c_x_eff = if msb_x { c_x } else { -c_x };
                let c_y_eff = if msb_y { c_y } else { -c_y };
                let maxcol_x = 3 * (k_cx + order - 1) + 1;
                let maxcol_y = 3 * (k_cy + order - 1) + 1;
                let abs_x = -(period * (c_x_eff / TAU + maxcol_x as vernier_core::Real));
                let abs_y = -(period * (c_y_eff / TAU + maxcol_y as vernier_core::Real));
                println!("Pattern found.");
                println!(
                    "Estimated pose: x={:.4} µm, y={:.4} µm, θ={:.6} rad (quadrant k3={})",
                    abs_x, abs_y, fine.theta, orders.k3
                );
                println!(
                    "  (K_center x={} y={}; maxcol x={} y={})",
                    k_cx, k_cy, maxcol_x, maxcol_y
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

/// Builds a sibling path by appending a suffix to the prefix's file stem.
/// e.g. prefix "out/dbg" + "_spectrum.png" -> "out/dbg_spectrum.png".
fn with_suffix(prefix: &std::path::Path, suffix: &str) -> std::path::PathBuf {
    let mut s = prefix.as_os_str().to_os_string();
    s.push(suffix);
    std::path::PathBuf::from(s)
}
