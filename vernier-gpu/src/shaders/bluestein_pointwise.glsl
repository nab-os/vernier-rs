#version 450

// Pointwise multiply work_a[i] *= work_b[k(i)].
//
// pass=0 (row):  work_a is (height × M), k(i) = i % M
// pass=1 (col):  work_a is (M × width),  k(i) = i / width

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) buffer WorkA { vec2 work_a[]; };
layout(set = 0, binding = 1) buffer WorkB { vec2 work_b[]; };

layout(push_constant) uniform PushConstantData {
    uint n_elements;
    uint period;   // M for pass=0 (k = i % M), width for pass=1 (k = i / width)
    uint pass;
} pc;

vec2 cmul(vec2 a, vec2 b) {
    return vec2(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
}

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= pc.n_elements) return;

    uint k = (pc.pass == 0u) ? (i % pc.period) : (i / pc.period);
    work_a[i] = cmul(work_a[i], work_b[k]);
}
