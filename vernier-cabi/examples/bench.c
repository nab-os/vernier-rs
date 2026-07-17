/* Need POSIX.1-2001 for clock_gettime / CLOCK_MONOTONIC under -std=c11. */
#define _POSIX_C_SOURCE 200112L

/*
 * bench.c — vernier-cabi binding benchmark.
 *
 * Generates a synthetic sinusoidal image and times vernier_detect_periodic()
 * across N iterations.  Prints one line to stdout:
 *
 *     binding=c backend=cpu size=512x512 iters=N mean=X.XXms best=X.XXms
 *
 * When --backend cuda is requested but CUDA is unavailable the binary exits
 * with code 2 and prints a message to stderr; the orchestrator treats this as
 * "n/a".
 *
 * Build (or use "make bench" from this directory):
 *
 *   cc -std=c11 -O2 -I../include -L../../target/release \
 *       bench.c -lvernier_cabi -Wl,-rpath,../../target/release \
 *       -lm -o bench
 *   ./bench [--size 512] [--iters 50] [--backend cpu|cuda]
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "vernier.h"

#define TWO_PI 6.28318530717958f

static float *synthetic_image(int size)
{
    float *img = malloc((size_t)size * size * sizeof(float));
    if (!img) return NULL;

    float period = (float)size / 16.0f;
    float theta  = 0.1f;
    float cx = size / 2.0f, cy = size / 2.0f;
    float c = cosf(-theta), s = sinf(-theta);

    for (int row = 0; row < size; ++row) {
        for (int col = 0; col < size; ++col) {
            float dx = (float)col - cx;
            float dy = (float)row - cy;
            float xp = c * dx - s * dy;
            img[row * size + col] = 0.5f + 0.5f * cosf(TWO_PI * xp / period);
        }
    }
    return img;
}

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec * 1e-6;
}

int main(int argc, char *argv[])
{
    int  size    = 512;
    int  iters   = 50;
    int  use_cuda = 0;

    for (int i = 1; i < argc; ++i) {
        if (i + 1 < argc) {
            if (strcmp(argv[i], "--size")    == 0) { size  = atoi(argv[++i]); continue; }
            if (strcmp(argv[i], "--iters")   == 0) { iters = atoi(argv[++i]); continue; }
            if (strcmp(argv[i], "--backend") == 0) {
                use_cuda = (strcmp(argv[++i], "cuda") == 0);
                continue;
            }
        }
    }

    float *img = synthetic_image(size);
    if (!img) { fputs("out of memory\n", stderr); return 1; }

    VernierDetector *det = use_cuda ? vernier_detector_new_cuda()
                                    : vernier_detector_new();
    if (!det) {
        const char *msg = vernier_last_error();
        fprintf(stderr, "%s unavailable: %s\n", use_cuda ? "cuda" : "cpu",
                msg ? msg : "unknown error");
        free(img);
        return use_cuda ? 2 : 1;
    }

    const char *backend = use_cuda ? "cuda" : "cpu";
    float period = (float)size / 16.0f;

    /* Warm-up: prime FFT/GPU pipeline cache. */
    vernier_detect_periodic(det, img, (size_t)size, (size_t)size,
                            period, 3.0f, 5, 0, 0.5f);

    double best = 1e18, total = 0.0;
    for (int i = 0; i < iters; ++i) {
        double t0 = now_ms();
        VernierPose p = vernier_detect_periodic(
            det, img, (size_t)size, (size_t)size,
            period, 3.0f, 5, 0, 0.5f);
        double ms = now_ms() - t0;
        (void)p;
        total += ms;
        if (ms < best) best = ms;
    }

    double mean = total / iters;
    printf("binding=c backend=%s size=%dx%d iters=%d mean=%.2fms best=%.2fms\n",
           backend, size, size, iters, mean, best);

    vernier_detector_free(det);
    free(img);
    return 0;
}
