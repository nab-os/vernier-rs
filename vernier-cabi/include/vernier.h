#ifndef VERNIER_H
#define VERNIER_H

#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

// Opaque handle to a Vernier detector. Create with `vernier_detector_new`
// (CPU) or `vernier_detector_new_cuda` (GPU); free with
// `vernier_detector_free`.
typedef struct VernierDetector VernierDetector;

// Pose returned by all detection functions.
//
// Check `found` before reading `x`, `y`, `theta`. On failure (`found == 0`)
// call `vernier_last_error()` for a description.
typedef struct VernierPose {
    float x;
    float y;
    float theta;
    // 1 on success, 0 on failure.
    int32_t found;
} VernierPose;

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

// Returns the last error message on this thread, or NULL if the last call
// succeeded.
//
// The pointer is valid until the next vernier call on this thread.
const char *vernier_last_error(void);

// Creates a CPU-backed detector. Returns NULL on allocation failure.
//
// Must be freed with `vernier_detector_free`.
struct VernierDetector *vernier_detector_new(void);

// Creates a CUDA-backed detector. Returns NULL if no CUDA device is
// available or if the library was not compiled with CUDA support (check
// `vernier_last_error` for details).
//
// Must be freed with `vernier_detector_free`.
struct VernierDetector *vernier_detector_new_cuda(void);

// Frees a detector. Passing NULL is a no-op.
void vernier_detector_free(struct VernierDetector *det);

// Periodic (relative) detection: recovers `x`, `y` modulo the pattern period
// and the in-image orientation `theta`.
//
// - `det`              — handle from `vernier_detector_new[_cuda]` (must not be NULL).
// - `pixels`           — row-major f32 image, `width × height` elements in [0, 1].
// - `period`           — pattern spatial period in physical units.
// - `sigma`            — bandpass filter half-width in frequency bins.
// - `min_frequency`    — inner annulus radius for peak search (0 = no limit).
// - `max_frequency`    — outer annulus radius for peak search (0 = no limit).
// - `smoothing_sigma`  — Gaussian blur on the magnitude spectrum before peak
//                        search; 0 disables blurring.
//
// Returns a pose with `found == 0` on failure.
struct VernierPose vernier_detect_periodic(struct VernierDetector *det,
                                           const float *pixels,
                                           size_t width,
                                           size_t height,
                                           float period,
                                           float sigma,
                                           size_t min_frequency,
                                           size_t max_frequency,
                                           float smoothing_sigma);

// Megarena absolute detection: recovers an unambiguous `(x, y, theta)` using
// the LFSR binary code embedded in the pattern.
//
// - `det`              — handle from `vernier_detector_new[_cuda]` (must not be NULL).
// - `pixels`           — row-major f32 image, `width × height` elements in [0, 1].
// - `physical_period`  — pattern spatial period in micrometres (9 µm for the
//                        reference pattern).
// - `code_size`        — LFSR order in bits (12 for the reference pattern).
// - `sigma`            — bandpass filter half-width in frequency bins.
// - `min_frequency`    — inner annulus radius for peak search (0 = no limit).
// - `max_frequency`    — outer annulus radius for peak search (0 = no limit).
// - `smoothing_sigma`  — Gaussian blur on the magnitude spectrum before peak
//                        search; 0 disables blurring.
//
// Returns a pose with `found == 0` on failure.
struct VernierPose vernier_detect_megarena(struct VernierDetector *det,
                                           const float *pixels,
                                           size_t width,
                                           size_t height,
                                           float physical_period,
                                           uint32_t code_size,
                                           float sigma,
                                           size_t min_frequency,
                                           size_t max_frequency,
                                           float smoothing_sigma);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  /* VERNIER_H */
