#version 450

// Partial-sum pass for GPU plane fitting.
//
// Each thread computes the wrapped phase gradient at its pixel (x, y):
//   da = arg(z[x+1,y] * conj(z[x,y]))   — x-gradient ≈ a
//   db = arg(z[x,y+1] * conj(z[x,y]))   — y-gradient ≈ b
//
// This works without unwrapping because the per-pixel phase step is always
// < π for our patterns (period > 2 px in both directions). Gradients are
// averaged in the reduction; the mean converges to the true plane gradient.
//
// Outputs 2 vec2 elements per workgroup into `parts`:
//   parts[gid*2+0] = vec2(da_sum, da_count)
//   parts[gid*2+1] = vec2(db_sum, db_count)

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly buffer ZField { vec2 z[]; };
layout(set = 0, binding = 1) buffer Partials { vec2 parts[]; };

layout(push_constant) uniform PushConstantData {
    uint width;
    uint height;
    uint x0;   // crop left  (inclusive)
    uint y0;   // crop top   (inclusive)
    uint x1;   // crop right (exclusive)
    uint y1;   // crop bottom (exclusive)
} pc;

shared float s_da[64];
shared float s_da_cnt[64];
shared float s_db[64];
shared float s_db_cnt[64];

float phase_diff(vec2 z1, vec2 z0) {
    // arg(z1 * conj(z0)) without computing the full product magnitude
    float cross = z1.y * z0.x - z1.x * z0.y;
    float dot   = z1.x * z0.x + z1.y * z0.y;
    return atan(cross, dot);
}

void main() {
    uint x        = gl_GlobalInvocationID.x;
    uint y        = gl_GlobalInvocationID.y;
    uint local_id = gl_LocalInvocationIndex;

    float da = 0.0, da_count = 0.0;
    float db = 0.0, db_count = 0.0;

    if (x >= pc.x0 && x + 1u < pc.x1 && y >= pc.y0 && y < pc.y1) {
        vec2 z0 = z[y * pc.width + x];
        vec2 z1 = z[y * pc.width + x + 1u];
        da       = phase_diff(z1, z0);
        da_count = 1.0;
    }
    if (x >= pc.x0 && x < pc.x1 && y >= pc.y0 && y + 1u < pc.y1) {
        vec2 z0 = z[y * pc.width + x];
        vec2 z1 = z[(y + 1u) * pc.width + x];
        db       = phase_diff(z1, z0);
        db_count = 1.0;
    }

    s_da[local_id]     = da;
    s_da_cnt[local_id] = da_count;
    s_db[local_id]     = db;
    s_db_cnt[local_id] = db_count;
    barrier();

    for (uint stride = 32u; stride > 0u; stride >>= 1u) {
        if (local_id < stride) {
            s_da[local_id]     += s_da[local_id + stride];
            s_da_cnt[local_id] += s_da_cnt[local_id + stride];
            s_db[local_id]     += s_db[local_id + stride];
            s_db_cnt[local_id] += s_db_cnt[local_id + stride];
        }
        barrier();
    }

    if (local_id == 0u) {
        uint workgroup_id = gl_WorkGroupID.y * gl_NumWorkGroups.x + gl_WorkGroupID.x;
        parts[workgroup_id * 2u + 0u] = vec2(s_da[0], s_da_cnt[0]);
        parts[workgroup_id * 2u + 1u] = vec2(s_db[0], s_db_cnt[0]);
    }
}
