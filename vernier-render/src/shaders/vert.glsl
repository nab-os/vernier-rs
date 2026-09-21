#version 450

// Per-vertex: which corner of the unit cell [0,1]×[0,1]
layout(location = 0) in vec2 local_corner;
// Per-instance: centre of the dot in pattern space (µm). The carrier peaks at
// integer multiples of the period, so a dot's quad is centred on its lattice
// point, not anchored at the cell's bottom-left corner.
layout(location = 1) in vec2 dot_center;

layout(push_constant) uniform PushConstantData {
    float cos_alpha;  // cos(pose.theta)
    float sin_alpha;  // sin(pose.theta)
    float pose_x;     // image-center x in pattern space (µm)
    float pose_y;     // image-center y in pattern space (µm)
    float pixel_size; // µm / pixel
    float period;     // period in µm
    float img_width;  // image width  in pixels
    float img_height; // image height in pixels
} pc;

layout(location = 0) out vec2 out_pattern_pos;

void main() {
    // Fragment position in pattern space (µm)
    vec2 pat = dot_center + (local_corner - 0.5) * pc.period;
    out_pattern_pos = pat;

    // Displacement from the image-center point in pattern space
    float dx = pat.x - pc.pose_x;
    float dy = pat.y - pc.pose_y;

    // Rotate into camera frame and scale to pixels.
    // This is the inverse of into_pattern_frame: image ← pattern.
    float col = (pc.cos_alpha * dx - pc.sin_alpha * dy) / pc.pixel_size
                + pc.img_width  * 0.5;
    float row = (pc.sin_alpha * dx + pc.cos_alpha * dy) / pc.pixel_size
                + pc.img_height * 0.5;

    // Pixel → NDC (Vulkan: Y increases downward, same as pixel row)
    float ndc_x =  2.0 * col / pc.img_width  - 1.0;
    float ndc_y =  2.0 * row / pc.img_height - 1.0;

    gl_Position = vec4(ndc_x, ndc_y, 0.0, 1.0);
}
