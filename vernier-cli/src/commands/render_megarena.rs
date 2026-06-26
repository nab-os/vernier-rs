use std::path::Path;

use vernier_patterns::{PatternPose, megarena::Megarena};

use crate::imageio;

pub struct RenderMegarenaArgs {
    pub width: usize,
    pub height: usize,
    pub x: f64,
    pub y: f64,
    pub theta: f64,
    pub period_px: f64,
    pub code_size: u32,
    pub output: std::path::PathBuf,
}

pub fn run(args: &RenderMegarenaArgs) -> Result<(), String> {
    let pattern = Megarena::new(args.period_px as f32, args.code_size)
        .ok_or_else(|| format!("unsupported code size {}; must be 3..=16", args.code_size))?;

    let pose = PatternPose::new(args.x as f32, args.y as f32, args.theta as f32);
    let image = pattern.render(args.width, args.height, &pose);

    imageio::save_grayscale_png(
        Path::new(&args.output),
        args.width,
        args.height,
        image.as_slice(),
    )
}
