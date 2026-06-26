#version 450

// Gaussian band-pass centered at a carrier whose coordinates are read from a
// GPU-resident peaks buffer — no CPU round-trip needed between peak_search and
// filtering.  `direction` selects which peak: 0 → peaks[0..1], 1 → peaks[2..3].

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0) buffer DataBuf { vec2 data[]; };
layout(set = 0, binding = 1) readonly buffer PeaksBuf { vec2 peaks[]; };

layout(push_constant) uniform PushConstantData {
    uint width;
    uint height;
    uint direction;
    float sigma;
} pc;

float circular_delta(uint a, uint c, uint n) {
    int d = int(a) - int(c);
    int ni = int(n);
    d = ((d % ni) + ni) % ni;
    if (d > ni / 2) d -= ni;
    return float(d);
}

void main() {
    uint x = gl_GlobalInvocationID.x;
    uint y = gl_GlobalInvocationID.y;
    if (x >= pc.width || y >= pc.height) return;

    uint base = pc.direction * 2u;
    uint cx = uint(peaks[base].x);
    uint cy = uint(peaks[base + 1u].x);

    float dx = circular_delta(x, cx, pc.width);
    float dy = circular_delta(y, cy, pc.height);
    float r2 = dx * dx + dy * dy;
    float two_sigma_sq = 2.0 * pc.sigma * pc.sigma;
    float gain = exp(-r2 / two_sigma_sq);

    uint flat_index = y * pc.width + x;
    data[flat_index] *= gain;
}
