#version 450

// Stage-2 reduction: a single workgroup of 256 threads reduces the intermediate
// buffer produced by argmax_local to a single best (x, y) pair.
//
// Dispatched as [1, 1, 1].

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly buffer Intermediate { vec2 entries[]; };
layout(set = 0, binding = 1)          buffer PeakBuf      { vec2 peak[]; };  // 1 element {x, y}

layout(push_constant) uniform PushConstantData {
    uint n_groups;   // number of entries in the intermediate buffer
    uint width;      // needed to decode linear index → (x, y)
} pc;

shared float s_mag[256];
shared float s_gidx[256];

void main() {
    uint tid = gl_LocalInvocationID.x;

    float best_mag  = -1.0;
    float best_gidx = 0.0;

    // Each thread may cover multiple intermediate entries.
    for (uint i = tid; i < pc.n_groups; i += 256u) {
        vec2 e = entries[i];
        if (e.x > best_mag) {
            best_mag  = e.x;
            best_gidx = e.y;
        }
    }

    s_mag[tid]  = best_mag;
    s_gidx[tid] = best_gidx;
    barrier();

    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride && s_mag[tid + stride] > s_mag[tid]) {
            s_mag[tid]  = s_mag[tid + stride];
            s_gidx[tid] = s_gidx[tid + stride];
        }
        barrier();
    }

    if (tid == 0u) {
        uint idx = uint(s_gidx[0]);
        peak[0] = vec2(float(idx % pc.width), float(idx / pc.width));
    }
}
