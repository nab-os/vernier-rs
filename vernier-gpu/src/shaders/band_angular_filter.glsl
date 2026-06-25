#version 450

// Per-element filter applied between the two carrier-peak searches.
// Zeros every bin that falls outside the band [0.5·r₁, 2·r₁] (where r₁ is
// the distance from DC to peak 1) OR inside the angular cone of half-width
// atan(3·σ / r₁) centred on peak 1's direction.
//
// Mirrors the CPU block:
//   "Restrict peak 2 search to a frequency band around peak 1's radius …"
//   plus halfplane_argmax_angular_excl.
//
// Dispatched as [ceil(width/8), ceil(height/8), 1].

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0) buffer         MagBuf   { vec2 mag[]; };
layout(set = 0, binding = 1) readonly buffer Peak1Buf { vec2 peak1[]; };  // 1 element {cx1, cy1}

layout(push_constant) uniform PushConstantData {
    uint  width;
    uint  height;
    float sigma;
} pc;

const float PI  = 3.14159265358979323846;
const float TAU = 6.28318530717958647692;

int signed_freq(uint f, uint sz) {
    int fi = int(f);
    int ni = int(sz);
    return (fi > ni / 2) ? (fi - ni) : fi;
}

void main() {
    uint x = gl_GlobalInvocationID.x;
    uint y = gl_GlobalInvocationID.y;
    if (x >= pc.width || y >= pc.height) return;

    // Read peak1 coordinates written by argmax_global.
    float cx1 = peak1[0].x;
    float cy1 = peak1[0].y;

    int sfx1i = signed_freq(uint(cx1), pc.width);
    int sfy1i = signed_freq(uint(cy1), pc.height);
    float sfx1 = float(sfx1i);
    float sfy1 = float(sfy1i);
    float r1   = sqrt(sfx1 * sfx1 + sfy1 * sfy1);

    float band_min_r2 = (r1 * 0.5) * (r1 * 0.5);
    float band_max_r2 = (r1 * 2.0) * (r1 * 2.0);
    float center_angle = atan(sfy1, sfx1);
    float half_width   = atan(3.0 * pc.sigma, max(r1, 1.0));

    int sfx = signed_freq(x, pc.width);
    int sfy = signed_freq(y, pc.height);
    float r2 = float(sfx * sfx + sfy * sfy);

    uint idx = y * pc.width + x;

    // Frequency-band gate.
    if (r2 < band_min_r2 || r2 > band_max_r2) {
        mag[idx].x = 0.0;
        return;
    }

    // Angular-cone exclusion around peak 1.
    float angle = atan(float(sfy), float(sfx));
    float diff  = mod(angle - center_angle + PI, TAU) - PI;
    if (abs(diff) < half_width) {
        mag[idx].x = 0.0;
    }
}
