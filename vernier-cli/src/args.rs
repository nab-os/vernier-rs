//! Command-line argument definitions (argh).

use argh::FromArgs;

/// vernier-rs: GPU-accelerated pose measurement of calibrated patterns.
#[derive(FromArgs)]
pub struct TopLevel {
    #[argh(subcommand)]
    pub command: Command,
}

/// The available subcommands.
#[derive(FromArgs)]
#[argh(subcommand)]
pub enum Command {
    Detect(DetectArgs),
    Bench(BenchArgs),
    Roundtrip(RoundtripArgs),
    Inspect(InspectArgs),
    Analyse(AnalyseArgs),
    DetectMegarena(DetectMegarenaArgs),
}

/// Run detection on a synthetic pattern and print the recovered pose.
#[derive(FromArgs)]
#[argh(subcommand, name = "detect")]
pub struct DetectArgs {
    /// backend to use: cpu (or gpu with the gpu feature)
    #[argh(option, default = "String::from(\"cpu\")")]
    pub backend: String,

    /// square image side length (default 64)
    #[argh(option, default = "64")]
    pub size: usize,

    /// pattern frequency in cycles across the width (default 5)
    #[argh(option, default = "5")]
    pub cycles: usize,

    /// pattern period in physical units (default 10.0)
    #[argh(option, default = "10.0")]
    pub period: f32,

    /// band-pass filter width in bins (default 2.0)
    #[argh(option, default = "2.0")]
    pub sigma: f32,
}

/// Time the detection pipeline; compare backends by running this twice.
#[derive(FromArgs)]
#[argh(subcommand, name = "bench")]
pub struct BenchArgs {
    /// backend to use: cpu (or gpu with the gpu feature)
    #[argh(option, default = "String::from(\"cpu\")")]
    pub backend: String,

    /// square image side length (default 256)
    #[argh(option, default = "256")]
    pub size: usize,

    /// number of timed iterations (default 50)
    #[argh(option, default = "50")]
    pub iterations: usize,

    /// band-pass filter width in bins (default 2.0)
    #[argh(option, default = "2.0")]
    pub sigma: f32,
}

/// Full validation: generate a pattern at a known pose, detect it, report error.
#[derive(FromArgs)]
#[argh(subcommand, name = "roundtrip")]
pub struct RoundtripArgs {
    /// backend to use: cpu (or gpu with the gpu feature)
    #[argh(option, default = "String::from(\"cpu\")")]
    pub backend: String,

    /// square image side length (default 128)
    #[argh(option, default = "128")]
    pub size: usize,

    /// pattern period in pixels (default 16.0)
    #[argh(option, default = "16.0")]
    pub period: f32,

    /// true orientation to render at, in radians (default 0.2)
    #[argh(option, default = "0.2")]
    pub theta: f32,

    /// band-pass filter width in bins (default 2.0)
    #[argh(option, default = "2.0")]
    pub sigma: f32,
}

/// Run the pipeline and save each stage as a PGM image for inspection.
#[derive(FromArgs)]
#[argh(subcommand, name = "inspect")]
pub struct InspectArgs {
    /// square image side length (default 128)
    #[argh(option, default = "128")]
    pub size: usize,

    /// pattern period in pixels (default 16.0)
    #[argh(option, default = "16.0")]
    pub period: f32,

    /// orientation to render at, in radians (default 0.2)
    #[argh(option, default = "0.2")]
    pub theta: f32,

    /// band-pass filter width in bins (default 4.0)
    #[argh(option, default = "4.0")]
    pub sigma: f32,

    /// output directory for stage images (default "vernier_stages")
    #[argh(option, default = "String::from(\"vernier_stages\")")]
    pub out: String,
}

/// Analyse a real image: compute and print the two phase planes (port of
/// analysingImage.cpp).
#[derive(FromArgs)]
#[argh(subcommand, name = "analyse")]
pub struct AnalyseArgs {
    /// path to the image file (JPEG/PNG/BMP/TIFF)
    #[argh(option)]
    pub image: String,

    /// band-pass filter width in bins (default 3.0, C++ PatternPhase default)
    #[argh(option, default = "3.0")]
    pub sigma: f32,

    /// inner annulus radius for peak search in bins; 0 = no lower limit (default 20)
    #[argh(option, default = "20")]
    pub min_frequency: usize,

    /// outer annulus radius for peak search in bins; 0 = no upper limit (default 500)
    #[argh(option, default = "500")]
    pub max_frequency: usize,

    /// gaussian blur sigma applied to magnitude before peak search (default 0.5)
    #[argh(option, default = "0.5")]
    pub smoothing_sigma: f32,

    /// disable the Hann window (windowing is on by default for real images)
    #[argh(switch)]
    pub no_window: bool,

    /// optional directory to also write control/stage images into
    #[argh(option)]
    pub stages: Option<String>,
}

/// Detect a megarena pattern in a real image and print its absolute pose (port
/// of detectingMegarenaPattern.cpp).
#[derive(FromArgs)]
#[argh(subcommand, name = "detect-megarena")]
pub struct DetectMegarenaArgs {
    /// path to the image file (JPEG/PNG/BMP/TIFF)
    #[argh(option)]
    pub image: String,

    /// physical period of the pattern in micrometres (default 9.0)
    #[argh(option, default = "9.0")]
    pub period: f32,

    /// LFSR code size in bits (default 12)
    #[argh(option, default = "12")]
    pub code_size: u32,

    /// band-pass filter width in bins (default 3.0, C++ PatternPhase default)
    #[argh(option, default = "3.0")]
    pub sigma: f32,

    /// inner annulus radius for peak search in bins; 0 = no lower limit (default 20)
    #[argh(option, default = "20")]
    pub min_frequency: usize,

    /// outer annulus radius for peak search in bins; 0 = no upper limit (default 500)
    #[argh(option, default = "500")]
    pub max_frequency: usize,

    /// gaussian blur sigma applied to magnitude before peak search (default 0.5)
    #[argh(option, default = "0.5")]
    pub smoothing_sigma: f32,

    /// disable the Hann window (windowing is on by default for real images)
    #[argh(switch)]
    pub no_window: bool,

    /// path prefix for debug overlay images (writes <prefix>_spectrum.png and _decoded.png)
    #[argh(option)]
    pub debug_image: Option<String>,
}
