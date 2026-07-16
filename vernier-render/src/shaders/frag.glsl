#version 450

const float TAU = 6.283185307179586;

layout(location = 0) in vec2 in_pattern_pos;

layout(push_constant) uniform PushConstantData {
    float cos_alpha;
    float sin_alpha;
    float pose_x;
    float pose_y;
    float pixel_size;
    float period;
    float img_width;
    float img_height;
} pc;

layout(location = 0) out float out_intensity;

void main() {
    // (1 + cos(2π·x/p)) / 2 · (1 + cos(2π·y/p)) / 2
    // = (1 + cos_x)(1 + cos_y) / 4  — matches C++ getIntensity formula
    float cx = (1.0 + cos(TAU * in_pattern_pos.x / pc.period)) * 0.5;
    float cy = (1.0 + cos(TAU * in_pattern_pos.y / pc.period)) * 0.5;
    out_intensity = cx * cy;
}
