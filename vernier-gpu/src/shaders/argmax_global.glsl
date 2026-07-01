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
shared float s_global_index[256];

void main() {
    uint thread_id = gl_LocalInvocationID.x;

    float best_mag          = -1.0;
    float best_global_index = 0.0;

    // Each thread may cover multiple intermediate entries.
    for (uint i = thread_id; i < pc.n_groups; i += 256u) {
        vec2 entry = entries[i];
        if (entry.x > best_mag) {
            best_mag          = entry.x;
            best_global_index = entry.y;
        }
    }

    s_mag[thread_id]          = best_mag;
    s_global_index[thread_id] = best_global_index;
    barrier();

    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (thread_id < stride && s_mag[thread_id + stride] > s_mag[thread_id]) {
            s_mag[thread_id]          = s_mag[thread_id + stride];
            s_global_index[thread_id] = s_global_index[thread_id + stride];
        }
        barrier();
    }

    if (thread_id == 0u) {
        uint linear_index = uint(s_global_index[0]);
        peak[0] = vec2(float(linear_index % pc.width), float(linear_index / pc.width));
    }
}
