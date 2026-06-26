//! Phase unwrapping — the data-dependent stage.
//!
//! Wrapped phase lives in `(-π, π]`; physical displacement is continuous and can
//! exceed one period. Unwrapping removes the 2π jumps to recover a continuous
//! phase. Unlike the per-pixel `extract_phase`, this is inherently *sequential*
//! (each sample's correction depends on the running total), which is exactly why
//! it does **not** belong on the backend trait as a parallel kernel — it runs
//! host-side on the small extracted result, not on the full image.
//!
//! This is the simple 1D unwrap. 2D unwrapping (path-following / quality-guided)
//! is a harder problem reserved for when the pipeline needs a full unwrapped
//! field; for fundamental-peak phase tracking the 1D version is sufficient.

use vernier_core::Real;
use vernier_core::scalar::consts::{PI, TAU};

/// Unwraps a 1D sequence of wrapped phases in place.
///
/// Walks the sequence, adding/subtracting 2π whenever the step between
/// consecutive samples exceeds π, so the output is continuous.
pub fn unwrap_1d(phases: &mut [Real]) {
    if phases.len() < 2 {
        return;
    }
    let mut offset: Real = 0.0;
    let mut prev = phases[0];
    for p in phases.iter_mut().skip(1) {
        let raw = *p;
        let mut delta = raw - prev;
        // Reduce the step into (-π, π].
        while delta > PI {
            delta -= TAU;
            offset -= TAU;
        }
        while delta <= -PI {
            delta += TAU;
            offset += TAU;
        }
        prev = raw;
        *p = raw + offset;
    }
}

/// Unwraps a 2D wrapped phase map outward from the center in four quadrants,
/// matching the C++ `quartersUnwrapPhase` in `Spatial.cpp`.
///
/// The algorithm propagates along the center row left and right from the center
/// pixel, then sweeps each column vertically using the accumulated row offset as
/// the starting point. This outward-from-center propagation is more robust than
/// row-by-row separable unwrapping for phase maps with radial continuity.
pub fn quarters_unwrap_phase(phase: &mut [Real], width: usize, height: usize) {
    if width == 0 || height == 0 {
        return;
    }
    let origin_x = width / 2;
    let origin_y = height / 2;

    let flat_index = |row: usize, col: usize| row * width + col;

    let step = |iter: &mut isize, prev: Real, next: Real| {
        let diff = next - prev;
        if diff > PI {
            *iter -= 1;
        } else if diff <= -PI {
            *iter += 1;
        }
    };

    // ---- Left half ----
    // Center row: origin_x-1 down to 0 (outer loop uses C++ `col` from origin_x to 1,
    // writing col-1 each iteration; vertical strips lag at column `col`).
    let mut iter_x: isize = 0;
    let mut next_x = phase[flat_index(origin_y, origin_x)];
    for col in (1..=origin_x).rev() {
        // Advance one step left along center row.
        let prev_x = next_x;
        next_x = phase[flat_index(origin_y, col - 1)];
        step(&mut iter_x, prev_x, next_x);
        phase[flat_index(origin_y, col - 1)] = next_x + iter_x as Real * TAU;

        // Quarter 3: column `col`, upward from center row.
        let mut iter_y = iter_x;
        let mut next_y = next_x; // seed = raw value at (origin_y, col-1), matches C++
        for row in (1..=origin_y).rev() {
            let prev_y = next_y;
            next_y = phase[flat_index(row - 1, col)];
            step(&mut iter_y, prev_y, next_y);
            phase[flat_index(row - 1, col)] = next_y + iter_y as Real * TAU;
        }

        // Quarter 2: column `col`, downward from center row.
        let mut iter_y = iter_x;
        let mut next_y = next_x;
        for row in origin_y..height - 1 {
            let prev_y = next_y;
            next_y = phase[flat_index(row + 1, col)];
            step(&mut iter_y, prev_y, next_y);
            phase[flat_index(row + 1, col)] = next_y + iter_y as Real * TAU;
        }
    }
    // Column 0 vertical strips (handled after the left-half loop in C++).
    {
        let mut iter_y = iter_x;
        let mut next_y = next_x;
        for row in (1..=origin_y).rev() {
            let prev_y = next_y;
            next_y = phase[flat_index(row - 1, 0)];
            step(&mut iter_y, prev_y, next_y);
            phase[flat_index(row - 1, 0)] = next_y + iter_y as Real * TAU;
        }
    }
    {
        let mut iter_y = iter_x;
        let mut next_y = next_x;
        for row in origin_y..height - 1 {
            let prev_y = next_y;
            next_y = phase[flat_index(row + 1, 0)];
            step(&mut iter_y, prev_y, next_y);
            phase[flat_index(row + 1, 0)] = next_y + iter_y as Real * TAU;
        }
    }

    // ---- Right half ----
    let mut iter_x: isize = 0;
    let mut next_x = phase[flat_index(origin_y, origin_x)];
    for col in origin_x..width - 1 {
        let prev_x = next_x;
        next_x = phase[flat_index(origin_y, col + 1)];
        step(&mut iter_x, prev_x, next_x);
        phase[flat_index(origin_y, col + 1)] = next_x + iter_x as Real * TAU;

        // Quarter 4: column `col`, upward from center row.
        let mut iter_y = iter_x;
        let mut next_y = next_x;
        for row in (1..=origin_y).rev() {
            let prev_y = next_y;
            next_y = phase[flat_index(row - 1, col)];
            step(&mut iter_y, prev_y, next_y);
            phase[flat_index(row - 1, col)] = next_y + iter_y as Real * TAU;
        }

        // Quarter 1: column `col`, downward from center row.
        let mut iter_y = iter_x;
        let mut next_y = next_x;
        for row in origin_y..height - 1 {
            let prev_y = next_y;
            next_y = phase[flat_index(row + 1, col)];
            step(&mut iter_y, prev_y, next_y);
            phase[flat_index(row + 1, col)] = next_y + iter_y as Real * TAU;
        }
    }
    // Last column vertical strips.
    {
        let mut iter_y = iter_x;
        let mut next_y = next_x;
        for row in (1..=origin_y).rev() {
            let prev_y = next_y;
            next_y = phase[flat_index(row - 1, width - 1)];
            step(&mut iter_y, prev_y, next_y);
            phase[flat_index(row - 1, width - 1)] = next_y + iter_y as Real * TAU;
        }
    }
    {
        let mut iter_y = iter_x;
        let mut next_y = next_x;
        for row in origin_y..height - 1 {
            let prev_y = next_y;
            next_y = phase[flat_index(row + 1, width - 1)];
            step(&mut iter_y, prev_y, next_y);
            phase[flat_index(row + 1, width - 1)] = next_y + iter_y as Real * TAU;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vernier_core::scalar::consts::TAU;

    #[test]
    fn unwraps_a_simple_ramp() {
        // A phase that increases past π and wraps: 0, 2, -2.0 (wrapped), ...
        // Construct a ramp of 0, 2, 4, 6 rad then wrap it, and check we recover
        // a monotonic sequence.
        let true_ramp: Vec<Real> = (0..6).map(|i| i as Real * 2.0).collect();
        let wrapped: Vec<Real> = true_ramp
            .iter()
            .map(|&x| {
                let mut v = x % TAU;
                if v > PI {
                    v -= TAU;
                }
                v
            })
            .collect();
        let mut work = wrapped.clone();
        unwrap_1d(&mut work);
        // Differences should match the true ramp's differences (constant 2.0).
        for w in work.windows(2) {
            assert!((w[1] - w[0] - 2.0).abs() < 1e-4);
        }
    }
}
