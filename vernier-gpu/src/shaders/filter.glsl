#version 450

// Annulus mask on an unshifted DFT buffer (DC at origin).
// Zeros every bin whose signed-frequency radius falls outside
// [min_frequency, max_frequency]. When max_frequency == 0 the outer
// bound is disabled (no high-frequency cut).

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0) buffer DataBuf { vec2 data[]; };

layout(push_constant) uniform PushConstantData {
    uint width;
    uint height;
    uint min_frequency;
    uint max_frequency;
} pc;

// Signed frequency of an unshifted DFT bin: bins past N/2 wrap to negative.
int signed_freq(uint f, uint n) {
    int fi = int(f);
    int ni = int(n);
    return (fi > ni / 2) ? (fi - ni) : fi;
}

void main() {
    uint x = gl_GlobalInvocationID.x;
    uint y = gl_GlobalInvocationID.y;
    if (x >= pc.width || y >= pc.height) return;

    int sfx = signed_freq(x, pc.width);
    int sfy = signed_freq(y, pc.height);
    float r2 = float(sfx * sfx + sfy * sfy);

    float min_r2 = float(pc.min_frequency * pc.min_frequency);
    float max_r2 = (pc.max_frequency > 0u)
        ? float(pc.max_frequency * pc.max_frequency)
        : 1.0e30;

    if (r2 < min_r2 || r2 > max_r2) {
        data[y * pc.width + x] = vec2(0.0);
    }
}
