#pragma once
#include "vernier.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

/// C++ RAII wrapper around the vernier C ABI.
namespace vernier {

/// Detected in-plane pose.
/// `x` and `y` are in the same physical units as the `period` argument.
/// `theta` is in radians.
struct Pose {
    float x;
    float y;
    float theta;
};

/// CPU- or CUDA-backed pose detector.
///
/// Reuse across frames — the internal FFT planner caches its plan, so
/// repeated calls on same-size images are cheaper than constructing a new
/// Detector each time.
class Detector {
public:
    /// Creates a CPU-backed detector.
    /// Throws `std::runtime_error` on allocation failure.
    Detector() {
        handle_ = vernier_detector_new();
        if (!handle_)
            throw std::runtime_error("vernier: failed to create CPU detector");
    }

    /// Creates a CUDA-backed detector.
    /// Throws `std::runtime_error` if no CUDA device is available or if the
    /// library was not built with CUDA support (rebuild with `--features cuda`).
    static Detector cuda() {
        VernierDetector* h = vernier_detector_new_cuda();
        if (!h) {
            const char* err = vernier_last_error();
            throw std::runtime_error(err ? std::string(err)
                                        : "vernier: CUDA detector creation failed");
        }
        return Detector(h);
    }

    Detector(Detector&& other) noexcept : handle_(other.handle_) {
        other.handle_ = nullptr;
    }

    Detector& operator=(Detector&& other) noexcept {
        if (this != &other) {
            reset();
            handle_ = other.handle_;
            other.handle_ = nullptr;
        }
        return *this;
    }

    Detector(const Detector&)            = delete;
    Detector& operator=(const Detector&) = delete;

    ~Detector() { reset(); }

    /// Periodic (relative) detection.
    ///
    /// Recovers `x` and `y` modulo `period` and the in-image orientation
    /// `theta`. Throws `std::runtime_error` if no carrier peaks are found.
    ///
    /// @param pixels          Row-major float array, `width * height` elements in [0, 1].
    /// @param width           Image width in pixels.
    /// @param height          Image height in pixels.
    /// @param period          Pattern spatial period in physical units.
    /// @param sigma           Bandpass filter half-width in frequency bins.
    /// @param min_frequency   Inner spectral annulus radius; 0 = no limit.
    /// @param max_frequency   Outer spectral annulus radius; 0 = no limit.
    /// @param smoothing_sigma Gaussian blur sigma on magnitude spectrum; 0 disables.
    Pose detect_periodic(
        const float* pixels,
        std::size_t  width,
        std::size_t  height,
        float        period,
        float        sigma           = 3.0f,
        std::size_t  min_frequency   = 0,
        std::size_t  max_frequency   = 0,
        float        smoothing_sigma = 0.5f)
    {
        VernierPose p = vernier_detect_periodic(
            handle_, pixels, width, height,
            period, sigma, min_frequency, max_frequency, smoothing_sigma);
        if (!p.found)
            throw_last_error("periodic detection failed");
        return {p.x, p.y, p.theta};
    }

    /// Megarena absolute detection.
    ///
    /// Recovers an unambiguous `(x, y, theta)` by combining the fine phase
    /// measurement with the LFSR binary code embedded in the Megarena pattern.
    /// Throws `std::runtime_error` if detection or LFSR decode fails.
    ///
    /// @param pixels          Row-major float array, `width * height` elements in [0, 1].
    /// @param width           Image width in pixels.
    /// @param height          Image height in pixels.
    /// @param physical_period Pattern spatial period in micrometres (9.0 for the reference pattern).
    /// @param code_size       LFSR order in bits (12 for the reference pattern).
    /// @param sigma           Bandpass filter half-width in frequency bins.
    /// @param min_frequency   Inner spectral annulus radius; 0 = no limit.
    /// @param max_frequency   Outer spectral annulus radius; 0 = no limit.
    /// @param smoothing_sigma Gaussian blur sigma on magnitude spectrum.
    Pose detect_megarena(
        const float*  pixels,
        std::size_t   width,
        std::size_t   height,
        float         physical_period,
        std::uint32_t code_size,
        float         sigma           = 3.0f,
        std::size_t   min_frequency   = 20,
        std::size_t   max_frequency   = 500,
        float         smoothing_sigma = 0.5f)
    {
        VernierPose p = vernier_detect_megarena(
            handle_, pixels, width, height,
            physical_period, code_size,
            sigma, min_frequency, max_frequency, smoothing_sigma);
        if (!p.found)
            throw_last_error("megarena detection failed");
        return {p.x, p.y, p.theta};
    }

private:
    VernierDetector* handle_;

    explicit Detector(VernierDetector* h) : handle_(h) {}

    void reset() noexcept {
        if (handle_) {
            vernier_detector_free(handle_);
            handle_ = nullptr;
        }
    }

    [[noreturn]] static void throw_last_error(const char* fallback) {
        const char* err = vernier_last_error();
        throw std::runtime_error(std::string("vernier: ") +
                                 (err ? err : fallback));
    }
};

}  // namespace vernier
