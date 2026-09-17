use std::path::Path;

use vernier_patterns::{PatternPose, checkerboard::{Checkerboard, CodeLayout}};

use crate::imageio;

pub struct RenderCheckerboardArgs {
    pub width: usize,
    pub height: usize,
    pub x: f64,
    pub y: f64,
    pub theta: f64,
    pub square_px: f64,
    pub code_size: u32,
    /// Render the uncoded carrier instead of the coded pattern.
    pub plain: bool,
    /// Diamond layout: code along the diagonals.
    pub diamonds: bool,
    pub output: std::path::PathBuf,
}

pub fn run(args: &RenderCheckerboardArgs) -> Result<(), String> {
    let layout = if args.diamonds {
        CodeLayout::Diamonds
    } else {
        CodeLayout::Squares
    };
    let pattern = Checkerboard::new(args.square_px, args.code_size)
        .ok_or_else(|| format!("unsupported code size {}; must be 4..=12", args.code_size))?
        .with_code_layout(layout);

    let pose = PatternPose::new(args.x, args.y, args.theta);
    let image = if args.plain {
        pattern.render_plain(args.width, args.height, &pose)
    } else {
        pattern.render(args.width, args.height, &pose)
    };

    imageio::save_grayscale_png(
        Path::new(&args.output),
        args.width,
        args.height,
        image.as_slice(),
    )
}
