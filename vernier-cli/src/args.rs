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
    Bench(BenchArgs),
    CheckerboardFigures(CheckerboardFiguresArgs),
    DetectMegarena(DetectMegarenaArgs),
    RenderCheckerboard(RenderCheckerboardArgs),
    RenderMegarena(RenderMegarenaArgs),
    RoundtripMegarena(RoundtripMegarenaArgs),
}

/// Time the full two-direction detection pipeline on a synthetic image.
#[derive(FromArgs)]
#[argh(subcommand, name = "bench")]
pub struct BenchArgs {
    /// backend to use: cpu or gpu (default: cpu)
    #[argh(option, default = "String::from(\"cpu\")")]
    pub backend: String,

    /// square image side length (default: 512)
    #[argh(option, default = "512")]
    pub size: usize,

    /// number of timed iterations (default: 20)
    #[argh(option, default = "20")]
    pub iters: usize,

    /// band-pass filter width in bins (default: 3.0)
    #[argh(option, default = "3.0")]
    pub sigma: f32,

    /// inner annulus radius for peak search in bins; 0 = no lower limit (default: 20)
    #[argh(option, default = "20")]
    pub min_frequency: usize,

    /// outer annulus radius for peak search in bins; 0 = no upper limit (default: 500)
    #[argh(option, default = "500")]
    pub max_frequency: usize,

    /// gaussian blur sigma applied to magnitude before peak search (default: 0.5)
    #[argh(option, default = "0.5")]
    pub smoothing_sigma: f32,
}

/// Detect a megarena pattern in a real image and print its absolute pose (port
/// of detectingMegarenaPattern.cpp).
#[derive(FromArgs)]
#[argh(subcommand, name = "detect-megarena")]
pub struct DetectMegarenaArgs {
    /// backend to use: cpu (or gpu with the gpu feature)
    #[argh(option, default = "String::from(\"cpu\")")]
    pub backend: String,

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

    /// path prefix for debug overlay images (writes <prefix>_spectrum.png and _decoded.png)
    #[argh(option)]
    pub debug_image: Option<String>,

    /// print intermediate detection details (carriers, planes, orientation)
    #[argh(switch)]
    pub verbose: bool,
}

/// Render a megarena pattern at given coordinates and save it as PNG.
#[derive(FromArgs)]
#[argh(subcommand, name = "render-megarena")]
pub struct RenderMegarenaArgs {
    /// output PNG file path
    #[argh(option)]
    pub output: String,

    /// image width in pixels (default: 512)
    #[argh(option, default = "512")]
    pub width: usize,

    /// image height in pixels (default: 512)
    #[argh(option, default = "512")]
    pub height: usize,

    /// pattern X offset in pixels (default: 0.0)
    #[argh(option, default = "0.0")]
    pub x: f64,

    /// pattern Y offset in pixels (default: 0.0)
    #[argh(option, default = "0.0")]
    pub y: f64,

    /// pattern orientation in radians (default: 0.0)
    #[argh(option, default = "0.0")]
    pub theta: f64,

    /// dot period in pixels (default: 20.0)
    #[argh(option, default = "20.0")]
    pub period: f64,

    /// LFSR code size in bits, 3..=16 (default: 8)
    #[argh(option, default = "8")]
    pub code_size: u32,
}

/// Render a megarena at a known pose, run the full detection pipeline, and
/// report the error between the recovered pose and the ground truth.
#[derive(FromArgs)]
#[argh(subcommand, name = "roundtrip-megarena")]
pub struct RoundtripMegarenaArgs {
    /// backend to use: cpu or gpu (default: cpu)
    #[argh(option, default = "String::from(\"cpu\")")]
    pub backend: String,

    /// image width in pixels (default: 512)
    #[argh(option, default = "512")]
    pub width: usize,

    /// image height in pixels (default: 512)
    #[argh(option, default = "512")]
    pub height: usize,

    /// ground-truth X position in pixels (default: 0.0)
    #[argh(option, default = "0.0")]
    pub x: f64,

    /// ground-truth Y position in pixels (default: 0.0)
    #[argh(option, default = "0.0")]
    pub y: f64,

    /// ground-truth orientation in radians (default: 0.0)
    #[argh(option, default = "0.0")]
    pub theta: f64,

    /// dot period in pixels (default: 20.0)
    #[argh(option, default = "20.0")]
    pub period: f64,

    /// LFSR code size in bits, 3..=16 (default: 8)
    #[argh(option, default = "8")]
    pub code_size: u32,

    /// band-pass filter width in bins (default: 3.0)
    #[argh(option, default = "3.0")]
    pub sigma: f64,

    /// inner annulus radius for peak search in bins; 0 = no lower limit (default: 20)
    #[argh(option, default = "20")]
    pub min_frequency: usize,

    /// outer annulus radius for peak search in bins; 0 = no upper limit (default: 500)
    #[argh(option, default = "500")]
    pub max_frequency: usize,

    /// gaussian blur sigma applied to magnitude before peak search (default: 0.5)
    #[argh(option, default = "0.5")]
    pub smoothing_sigma: f64,

    /// render the pattern on the GPU via Vulkan instead of the CPU path (requires --features vulkan)
    #[argh(switch)]
    pub render_gpu: bool,

    /// camera pixel size in µm/pixel, used when --render-gpu is set (default: 1.0)
    #[argh(option, default = "1.0")]
    pub pixel_size: f64,
}

/// Render a coded checkerboard pattern at given coordinates and save it as PNG.
#[derive(FromArgs)]
#[argh(subcommand, name = "render-checkerboard")]
pub struct RenderCheckerboardArgs {
    /// output PNG file path
    #[argh(option)]
    pub output: String,

    /// image width in pixels (default: 512)
    #[argh(option, default = "512")]
    pub width: usize,

    /// image height in pixels (default: 512)
    #[argh(option, default = "512")]
    pub height: usize,

    /// pattern X offset in pixels (default: 0.0)
    #[argh(option, default = "0.0")]
    pub x: f64,

    /// pattern Y offset in pixels (default: 0.0)
    #[argh(option, default = "0.0")]
    pub y: f64,

    /// pattern orientation in radians (default: 0.0)
    #[argh(option, default = "0.0")]
    pub theta: f64,

    /// checkerboard square side in pixels; the carrier period is this times
    /// sqrt(2) (default: 12.0)
    #[argh(option, default = "12.0")]
    pub square: f64,

    /// LFSR code size in bits, 4..=12 (default: 8)
    #[argh(option, default = "8")]
    pub code_size: u32,

    /// render the uncoded checkerboard instead (the carrier with no code)
    #[argh(switch)]
    pub plain: bool,

    /// index the code along the lattice diagonals (parallel to the carriers)
    /// instead of the square edges; with --theta pi/4 this gives diamond
    /// squares with an upright code grid
    #[argh(switch)]
    pub diagonal_code: bool,
}

/// Generate the explainer figures and measurements for the coded checkerboard.
#[derive(FromArgs)]
#[argh(subcommand, name = "checkerboard-figures")]
pub struct CheckerboardFiguresArgs {
    /// directory the PNGs are written to
    #[argh(option)]
    pub out_dir: String,

    /// checkerboard square side in pixels (default: 8.0)
    #[argh(option, default = "8.0")]
    pub square: f64,

    /// LFSR code size in bits, 4..=12 (default: 8)
    #[argh(option, default = "8")]
    pub code_size: u32,

    /// side of the square figures in pixels (default: 512)
    #[argh(option, default = "512")]
    pub size: usize,

    /// number of poses used for the phase-bias measurement (default: 24)
    #[argh(option, default = "24")]
    pub poses: usize,
}
