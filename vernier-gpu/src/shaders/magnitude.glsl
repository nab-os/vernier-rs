#version 450

// Per-element in-place: re = sqrt(re² + im²), im = 0.

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) buffer DataBuf { vec2 data[]; };

layout(push_constant) uniform PushConstantData {
    uint n;
} pc;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= pc.n) return;
    data[idx] = vec2(length(data[idx]), 0.0);
}
