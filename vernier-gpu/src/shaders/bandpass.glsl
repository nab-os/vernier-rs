#version 450

// Gaussian band-pass filter centered at carrier bin (cx, cy) in the
// unshifted DFT layout (DC at origin). Distance uses circular wrap so
// a lobe near the Nyquist edge is not clipped.

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0) buffer DataBuf { vec2 data[]; };

layout(push_constant) uniform PushConstantData {
    uint width;
    uint height;
    uint cx;
    uint cy;
    float sigma;
} pc;

// Shortest signed distance from bin `a` to center `c` with period `n`.
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

    float dx = circular_delta(x, pc.cx, pc.width);
    float dy = circular_delta(y, pc.cy, pc.height);
    float r2 = dx * dx + dy * dy;
    float two_sigma_sq = 2.0 * pc.sigma * pc.sigma;
    float gain = exp(-r2 / two_sigma_sq);

    uint idx = y * pc.width + x;
    data[idx] *= gain;
}
