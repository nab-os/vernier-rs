#version 450

// Stage-1 parallel reduction: each workgroup of 256 threads finds the local
// maximum .re element in the upper DFT half-plane (signed fy >= 0), excluding
// DC. Writes {max_mag, float(best_linear_index)} to the intermediate buffer,
// one vec2 per workgroup.
//
// Dispatched as [ceil(n / 256), 1, 1]. The number of workgroups must fit in the
// intermediate buffer (allocated by the host to that size).

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly buffer MagBuf     { vec2 src[]; };
layout(set = 0, binding = 1)          buffer Intermediate { vec2 result[]; };

layout(push_constant) uniform PushConstantData {
    uint width;
    uint height;
    uint n;       // = width * height
} pc;

shared float s_mag[256];
shared float s_gidx[256];   // best global linear index, stored as float

int signed_freq(uint f, uint sz) {
    int fi = int(f);
    int ni = int(sz);
    return (fi > ni / 2) ? (fi - ni) : fi;
}

void main() {
    uint tid   = gl_LocalInvocationID.x;
    uint gidx  = gl_WorkGroupID.x * 256u + tid;

    float best_mag  = -1.0;
    float best_gidx = 0.0;

    if (gidx < pc.n) {
        uint x = gidx % pc.width;
        uint y = gidx / pc.width;
        int sfx = signed_freq(x, pc.width);
        int sfy = signed_freq(y, pc.height);

        // Upper half-plane (signed fy >= 0), excluding DC.
        bool valid = (sfy >= 0) && !(sfx == 0 && sfy == 0);
        if (valid) {
            best_mag  = src[gidx].x;
            best_gidx = float(gidx);
        }
    }

    s_mag[tid]  = best_mag;
    s_gidx[tid] = best_gidx;
    barrier();

    // Parallel tree-reduction (log₂ 256 = 8 steps).
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride && s_mag[tid + stride] > s_mag[tid]) {
            s_mag[tid]  = s_mag[tid + stride];
            s_gidx[tid] = s_gidx[tid + stride];
        }
        barrier();
    }

    if (tid == 0u) {
        result[gl_WorkGroupID.x] = vec2(s_mag[0], s_gidx[0]);
    }
}
