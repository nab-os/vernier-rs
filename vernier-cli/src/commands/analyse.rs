//! Port of the C++ `analysingImage.cpp` example.
//!
//! The original loads an image, runs `PatternPhase.compute`, and — if two
//! spectral peaks are found — prints the two phase planes, then shows control
//! images (spectrum + fringes). This is the direct analogue: load a real photo,
//! run two-direction detection, report both fitted phase planes, and optionally
//! dump the stage images.
//!
//! This is the "analyse a real picture from the command line" tool: point it at
//! a JPEG/PNG of a pattern and see what the detector extracts.

use std::path::{Path, PathBuf};

use vernier_core::buffer::BufferLayout;
use vernier_core::{Complex32, ComputeBackend};
use vernier_cpu::CpuBackend;
use vernier_detection::spectrum::analyze_two;

use crate::imageio::load_grayscale;

/// Analyse parameters.
pub struct Analyse {
    /// Path to the image file to analyse.
    pub image_path: PathBuf,
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
    /// Optional directory to also dump pipeline-stage images into.
    pub stages_dir: Option<PathBuf>,
}

impl Analyse {
    /// Runs the analysis, printing results. Returns Ok even when no peaks are
    /// found (that is a valid outcome to report, not an error); Err is for I/O
    /// or shape problems.
    pub fn run(&self) -> Result<(), String> {
        let img = load_grayscale(&self.image_path)?;
        let (width, height) = (img.width, img.height);
        println!(
            "loaded {} ({}x{} grayscale)",
            self.image_path.display(),
            width,
            height
        );

        let backend = CpuBackend::new();
        let layout = BufferLayout::packed(width, height);
        let complex: Vec<Complex32> = img.data.iter().map(|&v| Complex32::new(v, 0.0)).collect();
        let mut buf = backend
            .upload(&complex, layout)
            .map_err(|e| format!("upload failed: {e:?}"))?;

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

        // Report the two phase planes (mirrors the C++ getPlane1/getPlane2).
        let p1 = &detection.dir1.plane;
        let p2 = &detection.dir2.plane;
        println!("Phase plane 1: a={:.6} b={:.6} c={:.6}", p1.a, p1.b, p1.c);
        println!("Phase plane 2: a={:.6} b={:.6} c={:.6}", p2.a, p2.b, p2.c);
        println!(
            "  direction 1: peak bin {:?}, orientation {:.4} rad",
            detection.dir1.peak_bin,
            p1.orientation()
        );
        println!(
            "  direction 2: peak bin {:?}, orientation {:.4} rad",
            detection.dir2.peak_bin,
            p2.orientation()
        );

        // Sanity check the two carriers are roughly perpendicular — if not, the
        // detection likely locked onto a sideband, not the second direction.
        let dtheta = (p1.orientation() - p2.orientation()).abs();
        let perp = (dtheta - std::f64::consts::FRAC_PI_2 as vernier_core::Real).abs();
        if perp > 0.3 {
            println!(
                "  warning: directions are {:.1}° apart (expected ~90°); \
                 the pattern may not be a clean 2D grid, or sigma/exclude-radius \
                 need tuning",
                dtheta.to_degrees()
            );
        }

        // Optionally dump the stage images (the C++ showControlImages analogue).
        if let Some(dir) = &self.stages_dir {
            self.dump_stages(&img, width, height, dir)?;
            println!("control/stage images written to '{}/'", dir.display());
        }

        Ok(())
    }

    /// Dumps the spectrum + fringe stage images, reusing the inspect machinery
    /// path conceptually. Kept minimal here: writes the input and FFT magnitude
    /// so the user has the "control images" the C++ example shows.
    fn dump_stages(&self, img: &crate::imageio::LoadedImage, width: usize, height: usize, dir: &Path) -> Result<(), String> {
        use crate::pgm;
        std::fs::create_dir_all(dir).map_err(|e| format!("mkdir failed: {e}"))?;

        let intensities: Vec<f64> = img.data.iter().map(|&v| v as f64).collect();
        pgm::save_unit(&dir.join("00_input.pgm"), width, height, &intensities)
            .map_err(|e| format!("write failed: {e}"))?;

        // Recompute FFT magnitude for the control image.
        let backend = CpuBackend::new();
        let layout = BufferLayout::packed(width, height);
        let complex: Vec<Complex32> = img.data.iter().map(|&v| Complex32::new(v, 0.0)).collect();
        let mut buf = backend.upload(&complex, layout).unwrap();
        backend.fft2d(&mut buf).unwrap();
        let spec = backend.download(&buf).unwrap();
        let mag: Vec<f64> = spec.iter().map(|c| (c.norm_sqr() as f64).sqrt()).collect();
        pgm::save_log_magnitude(
            &dir.join("01_fft_magnitude.pgm"),
            width,
            height,
            &pgm::fftshift(width, height, &mag),
        )
        .map_err(|e| format!("write failed: {e}"))?;
        Ok(())
    }
}
