#version 450

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly buffer InputBuf  { vec2 src[]; };
layout(set = 0, binding = 1)          buffer OutputBuf { vec2 dst[]; };

layout(push_constant) uniform PushConstantData {
    uint width;
    uint height;
} pc;

void main() {
    uint x = gl_GlobalInvocationID.x;
    uint y = gl_GlobalInvocationID.y;
    if (x >= pc.width || y >= pc.height) return;

    uint flat_index = y * pc.width + x;
    vec2 z = src[flat_index];
    dst[flat_index] = vec2(atan(z.y, z.x), 0.0);
}
