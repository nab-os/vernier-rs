#version 450

// 2D inverse FFT — in-place, row-major Complex32 buffer.
//
// Dispatched twice per 2D transform:
//   pass=0  →  row pass,    dispatch [1,      height, 1]
//   pass=1  →  column pass, dispatch [width,  1,      1]
//
// For power-of-two N ≤ 2048: Cooley-Tukey DIT (positive twiddle, scale 1/N per pass).
// For other N             : direct IDFT (O(N²), same normalization).
//
// Combined normalization across both passes = 1/(W × H), matching rustfft.

layout(local_size_x = 1024, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) buffer DataBuf { vec2 data[]; };

layout(push_constant) uniform PushConstantData {
    uint width;
    uint height;
    uint pass;
} pc;

shared vec2 s[2048];

const float TAU = 6.28318530717958647692;

vec2 cmul(vec2 a, vec2 b) {
    return vec2(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
}

uint bit_reverse(uint v, uint bits) {
    uint r = 0u;
    for (uint i = 0u; i < bits; i++) { r = (r << 1u) | (v & 1u); v >>= 1u; }
    return r;
}

bool is_pow2(uint n) { return n > 0u && (n & (n - 1u)) == 0u; }

void main() {
    uint tid = gl_LocalInvocationID.x;
    uint n;

    // ---- Load one row or column into shared memory -------------------------
    if (pc.pass == 0u) {
        n = pc.width;
        uint base = gl_WorkGroupID.y * pc.width;
        if (tid < n)          s[tid]          = data[base + tid];
        if (tid + 1024u < n)  s[tid + 1024u]  = data[base + tid + 1024u];
    } else {
        n = pc.height;
        uint col = gl_WorkGroupID.x;
        if (tid < n)          s[tid]          = data[tid         * pc.width + col];
        if (tid + 1024u < n)  s[tid + 1024u]  = data[(tid+1024u) * pc.width + col];
    }
    barrier();

    float scale = 1.0 / float(n);

    // ---- Choose algorithm based on whether N is a power of two ------------
    if (is_pow2(n) && n <= 2048u) {
        // ---- Cooley-Tukey IDIT (in-place on s[]) --------------------------
        uint bits = 0u;
        for (uint t = n; t > 1u; t >>= 1u) bits++;

        if (tid < n)         { uint r=bit_reverse(tid,bits);       if(r>tid)        {vec2 t=s[tid];      s[tid]=s[r];      s[r]=t;} }
        if (tid+1024u < n)   { uint r=bit_reverse(tid+1024u,bits); if(r>tid+1024u)  {vec2 t=s[tid+1024u];s[tid+1024u]=s[r];s[r]=t;} }
        barrier();

        for (uint size = 2u; size <= n; size <<= 1u) {
            uint hstep = size >> 1u;
            for (uint k = tid; k < n / 2u; k += 1024u) {
                uint grp = k / hstep, pos = k % hstep;
                uint i = grp * size + pos, j = i + hstep;
                float angle = +TAU * float(pos) / float(size);
                vec2 w = vec2(cos(angle), sin(angle));
                vec2 u = s[i], v = cmul(w, s[j]);
                s[i] = u + v;  s[j] = u - v;
            }
            barrier();
        }

        // Scale by 1/N and write back
        if (pc.pass == 0u) {
            uint base = gl_WorkGroupID.y * pc.width;
            if (tid < n)          data[base + tid]         = s[tid]         * scale;
            if (tid+1024u < n)    data[base + tid+1024u]   = s[tid+1024u]   * scale;
        } else {
            uint col = gl_WorkGroupID.x;
            if (tid < n)          data[tid         * pc.width + col] = s[tid]         * scale;
            if (tid+1024u < n)    data[(tid+1024u) * pc.width + col] = s[tid+1024u]   * scale;
        }

    } else {
        // ---- Direct IDFT: read from s[], write straight to global memory --
        if (pc.pass == 0u) {
            uint base = gl_WorkGroupID.y * pc.width;
            for (uint k = tid; k < n; k += 1024u) {
                vec2 sum = vec2(0.0);
                for (uint m = 0u; m < n; m++) {
                    float angle = +TAU * float(k) * float(m) / float(n);
                    sum += cmul(s[m], vec2(cos(angle), sin(angle)));
                }
                data[base + k] = sum * scale;
            }
        } else {
            uint col = gl_WorkGroupID.x;
            for (uint k = tid; k < n; k += 1024u) {
                vec2 sum = vec2(0.0);
                for (uint m = 0u; m < n; m++) {
                    float angle = +TAU * float(k) * float(m) / float(n);
                    sum += cmul(s[m], vec2(cos(angle), sin(angle)));
                }
                data[k * pc.width + col] = sum * scale;
            }
        }
    }
}
