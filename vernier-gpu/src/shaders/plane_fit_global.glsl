#version 450

// Global reduction pass: sum all partial groups, compute (a, b, c).
//
// Outputs result[0..2] as vec2 with only .x used:
//   result[0].x = a  (x phase gradient, rad/px)
//   result[1].x = b  (y phase gradient, rad/px)
//   result[2].x = c  (center pixel phase, rad)

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly buffer ZField   { vec2 z[]; };
layout(set = 0, binding = 1) readonly buffer Partials { vec2 parts[]; };
layout(set = 0, binding = 2) buffer Result            { vec2 result[]; };

layout(push_constant) uniform PushConstantData {
    uint n_groups;    // number of partial entries (one per workgroup in pass 1)
    uint center_idx;  // flat index of center pixel: (H/2)*W + W/2
} pc;

shared float s_da[256];
shared float s_da_cnt[256];
shared float s_db[256];
shared float s_db_cnt[256];

void main() {
    uint local_id = gl_LocalInvocationID.x;

    float da = 0.0, da_count = 0.0;
    float db = 0.0, db_count = 0.0;

    for (uint i = local_id; i < pc.n_groups; i += 256u) {
        da       += parts[i * 2u + 0u].x;
        da_count += parts[i * 2u + 0u].y;
        db       += parts[i * 2u + 1u].x;
        db_count += parts[i * 2u + 1u].y;
    }

    s_da[local_id]     = da;
    s_da_cnt[local_id] = da_count;
    s_db[local_id]     = db;
    s_db_cnt[local_id] = db_count;
    barrier();

    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (local_id < stride) {
            s_da[local_id]     += s_da[local_id + stride];
            s_da_cnt[local_id] += s_da_cnt[local_id + stride];
            s_db[local_id]     += s_db[local_id + stride];
            s_db_cnt[local_id] += s_db_cnt[local_id + stride];
        }
        barrier();
    }

    if (local_id == 0u) {
        float a = (s_da_cnt[0] > 0.0) ? s_da[0] / s_da_cnt[0] : 0.0;
        float b = (s_db_cnt[0] > 0.0) ? s_db[0] / s_db_cnt[0] : 0.0;
        vec2 center_sample = z[pc.center_idx];
        float c = atan(center_sample.y, center_sample.x);
        result[0] = vec2(a, 0.0);
        result[1] = vec2(b, 0.0);
        result[2] = vec2(c, 0.0);
    }
}
