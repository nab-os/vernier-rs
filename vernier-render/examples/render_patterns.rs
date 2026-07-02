use std::path::Path;

use vernier_patterns::{
    CameraModel, PatternPose, PatternRenderer, megarena::Megarena, periodic::Periodic,
};

fn save_png(path: &str, width: usize, height: usize, data: &[f32]) {
    let bytes: Vec<u8> = data
        .iter()
        .map(|&v| (v.clamp(0.0, 1.0) * 255.0).round() as u8)
        .collect();
    let img = image::GrayImage::from_raw(width as u32, height as u32, bytes)
        .expect("buffer size mismatch");
    img.save(Path::new(path)).expect("failed to save PNG");
    println!("saved {path}");
}

fn main() {
    let renderer = PatternRenderer::new();
    let camera = CameraModel { pixel_size: 2.0 }; // 2 µm/pixel

    let width = 512;
    let height = 512;

    // --- Periodic pattern at a slight rotation ---
    let periodic = Periodic::new(20.0); // 20 px period
    let pose = PatternPose::new(0.0, 0.0, 0.15); // 0.15 rad rotation
    let img = periodic.render_gpu(&renderer, &camera, width, height, &pose);
    save_png("/tmp/periodic_gpu.png", width, height, img.as_slice());

    // --- Megarena pattern (8-bit code, 20 px period) ---
    let megarena = Megarena::new(9.0, 12).expect("valid order");
    let pose = PatternPose::new(0.0, 0.0, 0.15);
    let img = megarena.render_gpu(&renderer, &camera, width, height, &pose);
    save_png("/tmp/megarena_gpu.png", width, height, img.as_slice());
}
