#version 450

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly buffer Spectrum { vec2 S[];        };
layout(set = 0, binding = 1) readonly buffer PeaksBuf { vec2 peaks[];    };
layout(set = 0, binding = 2)          buffer Partials { float partial[];  };

layout(push_constant) uniform PushConstantData {
    uint  W;
    uint  H;
    uint  N;
    float sigma;
} pc;

shared float sh_c_re1[64]; shared float sh_c_im1[64];
shared float sh_sfx_num1[64]; shared float sh_sfy_num1[64]; shared float sh_den1[64];
shared float sh_c_re2[64]; shared float sh_c_im2[64];
shared float sh_sfx_num2[64]; shared float sh_sfy_num2[64]; shared float sh_den2[64];

int signed_freq(uint f, uint sz) {
    int fi = int(f);
    int ni = int(sz);
    return (fi > ni / 2) ? (fi - ni) : fi;
}

void main() {
    uint thread_id   = gl_LocalInvocationID.x;
    uint global_index = gl_WorkGroupID.x * 64u + thread_id;

    float neg_inv_two_sigma_sq = -1.0 / (2.0 * pc.sigma * pc.sigma);

    float sfx1 = float(signed_freq(uint(peaks[0].x), pc.W));
    float sfy1 = float(signed_freq(uint(peaks[1].x), pc.H));
    float sfx2 = float(signed_freq(uint(peaks[2].x), pc.W));
    float sfy2 = float(signed_freq(uint(peaks[3].x), pc.H));

    float c_re1 = 0.0, c_im1 = 0.0, sfx_num1 = 0.0, sfy_num1 = 0.0, den1 = 0.0;
    float c_re2 = 0.0, c_im2 = 0.0, sfx_num2 = 0.0, sfy_num2 = 0.0, den2 = 0.0;

    if (global_index < pc.N) {
        uint  fx           = global_index % pc.W;
        uint  fy           = global_index / pc.W;
        float sfx          = float(signed_freq(fx, pc.W));
        float sfy          = float(signed_freq(fy, pc.H));
        float sign         = ((fx + fy) % 2u == 0u) ? 1.0 : -1.0;

        vec2  spectrum_val = S[global_index];
        float magnitude_sq = dot(spectrum_val, spectrum_val);
        float s_re         = spectrum_val.x * sign;
        float s_im         = spectrum_val.y * sign;

        float dx1 = sfx - sfx1, dy1 = sfy - sfy1;
        float weight1           = exp(neg_inv_two_sigma_sq * (dx1 * dx1 + dy1 * dy1));
        float weighted_magnitude1 = weight1 * magnitude_sq;
        c_re1     = weight1 * s_re;  c_im1     = weight1 * s_im;
        sfx_num1  = weighted_magnitude1 * sfx;  sfy_num1 = weighted_magnitude1 * sfy;  den1 = weighted_magnitude1;

        float dx2 = sfx - sfx2, dy2 = sfy - sfy2;
        float weight2           = exp(neg_inv_two_sigma_sq * (dx2 * dx2 + dy2 * dy2));
        float weighted_magnitude2 = weight2 * magnitude_sq;
        c_re2     = weight2 * s_re;  c_im2     = weight2 * s_im;
        sfx_num2  = weighted_magnitude2 * sfx;  sfy_num2 = weighted_magnitude2 * sfy;  den2 = weighted_magnitude2;
    }

    sh_c_re1[thread_id]=c_re1; sh_c_im1[thread_id]=c_im1;
    sh_sfx_num1[thread_id]=sfx_num1; sh_sfy_num1[thread_id]=sfy_num1; sh_den1[thread_id]=den1;
    sh_c_re2[thread_id]=c_re2; sh_c_im2[thread_id]=c_im2;
    sh_sfx_num2[thread_id]=sfx_num2; sh_sfy_num2[thread_id]=sfy_num2; sh_den2[thread_id]=den2;
    barrier();

    for (uint stride = 32u; stride > 0u; stride >>= 1u) {
        if (thread_id < stride) {
            sh_c_re1[thread_id]+=sh_c_re1[thread_id+stride]; sh_c_im1[thread_id]+=sh_c_im1[thread_id+stride];
            sh_sfx_num1[thread_id]+=sh_sfx_num1[thread_id+stride]; sh_sfy_num1[thread_id]+=sh_sfy_num1[thread_id+stride]; sh_den1[thread_id]+=sh_den1[thread_id+stride];
            sh_c_re2[thread_id]+=sh_c_re2[thread_id+stride]; sh_c_im2[thread_id]+=sh_c_im2[thread_id+stride];
            sh_sfx_num2[thread_id]+=sh_sfx_num2[thread_id+stride]; sh_sfy_num2[thread_id]+=sh_sfy_num2[thread_id+stride]; sh_den2[thread_id]+=sh_den2[thread_id+stride];
        }
        barrier();
    }

    if (thread_id == 0u) {
        uint base = gl_WorkGroupID.x * 10u;
        partial[base+0u]=sh_c_re1[0]; partial[base+1u]=sh_c_im1[0];
        partial[base+2u]=sh_sfx_num1[0]; partial[base+3u]=sh_sfy_num1[0]; partial[base+4u]=sh_den1[0];
        partial[base+5u]=sh_c_re2[0]; partial[base+6u]=sh_c_im2[0];
        partial[base+7u]=sh_sfx_num2[0]; partial[base+8u]=sh_sfy_num2[0]; partial[base+9u]=sh_den2[0];
    }
}
