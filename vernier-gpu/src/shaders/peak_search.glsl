#version 450

// Single-thread ordering pass: reads the two peak positions produced by the
// two argmax passes and writes [cx1, cy1, cx2, cy2] to the output buffer,
// ordered so that direction 1 has the larger signed column frequency (matching
// the CPU convention from PatternPhase::peaksSearch).
//
// Dispatched as [1, 1, 1].

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly buffer Peak1Buf  { vec2 p1[]; };
layout(set = 0, binding = 1) readonly buffer Peak2Buf  { vec2 p2[]; };
layout(set = 0, binding = 2)          buffer OutputBuf { vec2 out_data[]; };

layout(push_constant) uniform PushConstantData {
    uint width;
    uint height;
} pc;

int signed_freq(uint f, uint sz) {
    int fi = int(f);
    int ni = int(sz);
    return (fi > ni / 2) ? (fi - ni) : fi;
}

void main() {
    vec2 a = p1[0];   // {cx_a, cy_a}
    vec2 b = p2[0];   // {cx_b, cy_b}

    int sfx_a = signed_freq(uint(a.x), pc.width);
    int sfx_b = signed_freq(uint(b.x), pc.width);

    // Direction 1 = larger signed column frequency.
    if (sfx_a < sfx_b) {
        vec2 tmp = a; a = b; b = tmp;
    }

    out_data[0] = vec2(a.x, 0.0);   // cx1
    out_data[1] = vec2(a.y, 0.0);   // cy1
    out_data[2] = vec2(b.x, 0.0);   // cx2
    out_data[3] = vec2(b.y, 0.0);   // cy2
}
