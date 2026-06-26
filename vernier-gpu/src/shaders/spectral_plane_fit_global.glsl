#version 450

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly buffer Partials { float partial[]; };
layout(set = 0, binding = 1)          buffer Result   { vec2  result[];  };

layout(push_constant) uniform PushConstantData {
    uint  n_groups;
    float W;
    float H;
} pc;

shared float sh_c_re1[256]; shared float sh_c_im1[256];
shared float sh_sfx_num1[256]; shared float sh_sfy_num1[256]; shared float sh_den1[256];
shared float sh_c_re2[256]; shared float sh_c_im2[256];
shared float sh_sfx_num2[256]; shared float sh_sfy_num2[256]; shared float sh_den2[256];

void main() {
    uint local_id = gl_LocalInvocationID.x;

    float c_re1=0.0, c_im1=0.0, sfx_num1=0.0, sfy_num1=0.0, den1=0.0;
    float c_re2=0.0, c_im2=0.0, sfx_num2=0.0, sfy_num2=0.0, den2=0.0;

    for (uint g = local_id; g < pc.n_groups; g += 256u) {
        uint base_index = g * 10u;
        c_re1+=partial[base_index+0u]; c_im1+=partial[base_index+1u];
        sfx_num1+=partial[base_index+2u]; sfy_num1+=partial[base_index+3u]; den1+=partial[base_index+4u];
        c_re2+=partial[base_index+5u]; c_im2+=partial[base_index+6u];
        sfx_num2+=partial[base_index+7u]; sfy_num2+=partial[base_index+8u]; den2+=partial[base_index+9u];
    }

    sh_c_re1[local_id]=c_re1; sh_c_im1[local_id]=c_im1;
    sh_sfx_num1[local_id]=sfx_num1; sh_sfy_num1[local_id]=sfy_num1; sh_den1[local_id]=den1;
    sh_c_re2[local_id]=c_re2; sh_c_im2[local_id]=c_im2;
    sh_sfx_num2[local_id]=sfx_num2; sh_sfy_num2[local_id]=sfy_num2; sh_den2[local_id]=den2;
    barrier();

    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (local_id < stride) {
            sh_c_re1[local_id]+=sh_c_re1[local_id+stride]; sh_c_im1[local_id]+=sh_c_im1[local_id+stride];
            sh_sfx_num1[local_id]+=sh_sfx_num1[local_id+stride]; sh_sfy_num1[local_id]+=sh_sfy_num1[local_id+stride]; sh_den1[local_id]+=sh_den1[local_id+stride];
            sh_c_re2[local_id]+=sh_c_re2[local_id+stride]; sh_c_im2[local_id]+=sh_c_im2[local_id+stride];
            sh_sfx_num2[local_id]+=sh_sfx_num2[local_id+stride]; sh_sfy_num2[local_id]+=sh_sfy_num2[local_id+stride]; sh_den2[local_id]+=sh_den2[local_id+stride];
        }
        barrier();
    }

    if (local_id == 0u) {
        float two_pi = 6.283185307179586;
        float a1 = (sh_den1[0] > 0.0) ? (two_pi * sh_sfx_num1[0] / sh_den1[0] / pc.W) : 0.0;
        float b1 = (sh_den1[0] > 0.0) ? (two_pi * sh_sfy_num1[0] / sh_den1[0] / pc.H) : 0.0;
        float c1 = atan(sh_c_im1[0], sh_c_re1[0]);
        float a2 = (sh_den2[0] > 0.0) ? (two_pi * sh_sfx_num2[0] / sh_den2[0] / pc.W) : 0.0;
        float b2 = (sh_den2[0] > 0.0) ? (two_pi * sh_sfy_num2[0] / sh_den2[0] / pc.H) : 0.0;
        float c2 = atan(sh_c_im2[0], sh_c_re2[0]);
        result[0]=vec2(a1,0.0); result[1]=vec2(b1,0.0); result[2]=vec2(c1,0.0);
        result[3]=vec2(a2,0.0); result[4]=vec2(b2,0.0); result[5]=vec2(c2,0.0);
    }
}
