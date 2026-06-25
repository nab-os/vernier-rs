#version 450

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(push_constant) uniform PushConstantData {
    uint width;
    uint height;
} pc;

void main() {
}
