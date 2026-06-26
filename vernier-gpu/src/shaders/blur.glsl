#version 450

// Separable Gaussian blur on the .re component (vec2.x) of each element.
// Dispatched twice from the host: pass=0 (horizontal) reads from binding 0
// and writes to binding 1; pass=1 (vertical) reads from binding 0 and writes
// to binding 1.  The host swaps the buffer bindings between passes so that
// the horizontal output becomes the vertical input.
//
// Kernel radius = ceil(3 * sigma); values outside the image are clamped
// to the nearest border (matching the CPU reference).

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly buffer SrcBuf { vec2 src[]; };
layout(set = 0, binding = 1)          buffer DstBuf { vec2 dst[]; };

layout(push_constant) uniform PushConstantData {
    uint  width;
    uint  height;
    float sigma;
    uint  pass;   // 0 = horizontal, 1 = vertical
} pc;

void main() {
    uint x = gl_GlobalInvocationID.x;
    uint y = gl_GlobalInvocationID.y;
    if (x >= pc.width || y >= pc.height) return;

    int radius = int(ceil(3.0 * pc.sigma));

    float weighted_value = 0.0;
    float weight_sum     = 0.0;

    if (pc.pass == 0u) {
        // Horizontal: convolve along x for fixed y.
        for (int k = -radius; k <= radius; k++) {
            int sx = int(x) + k;
            sx = clamp(sx, 0, int(pc.width) - 1);
            float kernel_weight = exp(-float(k * k) / (2.0 * pc.sigma * pc.sigma));
            weighted_value += src[y * pc.width + uint(sx)].x * kernel_weight;
            weight_sum     += kernel_weight;
        }
    } else {
        // Vertical: convolve along y for fixed x.
        for (int k = -radius; k <= radius; k++) {
            int sy = int(y) + k;
            sy = clamp(sy, 0, int(pc.height) - 1);
            float kernel_weight = exp(-float(k * k) / (2.0 * pc.sigma * pc.sigma));
            weighted_value += src[uint(sy) * pc.width + x].x * kernel_weight;
            weight_sum     += kernel_weight;
        }
    }

    uint flat_index = y * pc.width + x;
    dst[flat_index].x = (weight_sum > 0.0) ? weighted_value / weight_sum : 0.0;
    dst[flat_index].y = 0.0;
}
