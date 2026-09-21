//! Numeric checks on the explorer's chain: render a pattern through the camera,
//! run the detector, and assert the spectrum landed where the geometry says it
//! must. Nothing here needs a browser — the same code that runs in the page runs
//! natively under `cargo test`.

use vernier_core::Real;
use vernier_cpu::CpuBackend;

use crate::camera;
use crate::canvas;
use crate::explorer::ExplorerSettings;
use crate::pattern::{PatternKind, PatternSettings};
use crate::spectral::{self, Stages};

/// A pattern whose carrier is 16 px whichever kind it is, so every case is
/// compared against the same prediction.
fn settings_for(kind: PatternKind) -> PatternSettings {
    PatternSettings {
        kind,
        period_px: 16.0,
        square_px: 16.0 / std::f64::consts::SQRT_2,
        order: 10,
        lfsr_offset: 0,
        ..Default::default()
    }
}

fn view_for(pattern: &PatternSettings, size: usize) -> ExplorerSettings {
    ExplorerSettings {
        size,
        supersample: 3,
        min_frequency: 8,
        ..ExplorerSettings::for_period(pattern.explorer_period_px())
    }
}

fn analyse_with(pattern: &PatternSettings, view: &ExplorerSettings) -> Option<Stages> {
    let sampler = pattern.sampler().expect("this kind has a point sampler");
    let image = camera::render(&sampler, &view.pose, view.size, view.supersample);
    spectral::run(
        &CpuBackend::new(),
        image.as_slice(),
        view.size,
        view.sigma,
        view.min_frequency,
        view.max_frequency(),
        view.smoothing_sigma,
    )
    .expect("the chain runs")
}

fn analyse(kind: PatternKind, size: usize) -> Stages {
    let pattern = settings_for(kind);
    let view = view_for(&pattern, size);
    analyse_with(&pattern, &view).expect("two carrier peaks are found")
}

fn polar(peak: (Real, Real)) -> (Real, Real) {
    let (x, y) = peak;
    ((x * x + y * y).sqrt(), y.atan2(x).to_degrees())
}

fn separation_deg(stages: &Stages) -> Real {
    let (_, a1) = polar(stages.peaks[0]);
    let (_, a2) = polar(stages.peaks[1]);
    (a1 - a2).abs().rem_euclid(180.0)
}

/// The home distance is defined to put `TARGET_SAMPLES_PER_PERIOD` pixels on one
/// carrier period, which fixes the peak radius at `size / 8` bins. Every pattern
/// has to land there, whatever it wraps around its carrier — including the
/// checkerboard, whose period is the diagonal and not the square side.
#[test]
fn carriers_land_at_the_expected_radius() {
    let size = 128;
    let expected = size as Real / camera::TARGET_SAMPLES_PER_PERIOD;
    for kind in PatternKind::ALL {
        if !kind.has_point_sampler() {
            continue;
        }
        for peak in analyse(kind, size).peaks {
            let (radius, _) = polar(peak);
            assert!(
                (radius - expected).abs() <= 1.5,
                "{}: peak radius {radius:.2} bins, expected {expected:.2}",
                kind.label()
            );
        }
    }
}

/// The two carriers of each of these patterns are orthogonal, so their peaks sit
/// a right angle apart.
#[test]
fn the_two_carriers_are_orthogonal() {
    for kind in PatternKind::ALL {
        if !kind.has_point_sampler() {
            continue;
        }
        let separation = separation_deg(&analyse(kind, 128));
        assert!(
            (separation - 90.0).abs() <= 4.0,
            "{}: peaks {separation:.2} deg apart, expected 90",
            kind.label()
        );
    }
}

/// A checkerboard's carriers run along its diagonals — that is what makes its
/// period `a·√2` — so its peaks sit at ±45°, where the grid patterns put theirs
/// on the axes.
#[test]
fn checkerboard_carriers_run_diagonally() {
    for peak in analyse(PatternKind::Checkerboard, 128).peaks {
        let (_, angle) = polar(peak);
        let offset = (angle.rem_euclid(90.0) - 45.0).abs();
        assert!(offset <= 5.0, "checkerboard peak at {angle:.2} deg is not diagonal");
    }
}

#[test]
fn grid_carriers_run_along_the_axes() {
    for kind in [PatternKind::Periodic, PatternKind::Megarena] {
        for peak in analyse(kind, 128).peaks {
            let (_, angle) = polar(peak);
            let offset = angle.rem_euclid(90.0).min(90.0 - angle.rem_euclid(90.0));
            assert!(offset <= 5.0, "{}: peak at {angle:.2} deg is off-axis", kind.label());
        }
    }
}

/// Out-of-plane tilt is the freedom `PatternPose` cannot express, and the reason
/// this crate carries its own camera at all: under perspective the two carriers
/// stop being orthogonal in the image, which is exactly what the explorer is for
/// showing.
#[test]
fn tilting_out_of_plane_skews_the_peak_pair() {
    let pattern = settings_for(PatternKind::Checkerboard);
    let mut view = view_for(&pattern, 128);
    view.pose.beta = 35_f64.to_radians();
    view.pose.gamma = (-20_f64).to_radians();

    let stages = analyse_with(&pattern, &view).expect("the tilted pattern still has carriers");
    let separation = separation_deg(&stages);
    assert!(
        (separation - 90.0).abs() > 2.0,
        "a 35/-20 deg tilt should skew the pair, but they are {separation:.2} deg apart"
    );
}

/// The stubs have no layout to sample, and the explorer has to say so rather
/// than drawing a blank frame and leaving it a mystery.
#[test]
fn stub_generators_are_refused_with_a_reason() {
    for kind in [PatternKind::Stamp, PatternKind::QrLike] {
        let message = settings_for(kind).sampler().err().expect("stub has no sampler");
        assert!(message.contains(kind.label()), "{message}");
        assert!(message.contains("vernier-patterns/src/"), "{message}");
    }
}

/// Why the log toggle exists: a coded pattern's carrier peaks stand so far above
/// its sidebands that a linear ramp crushes most of the spectrum to black, while
/// the log spreads it across the range.
#[test]
fn the_log_scale_lifts_the_spectrum_off_the_floor() {
    let stages = analyse(PatternKind::Megarena, 128);
    let median = |log: bool| {
        let mut levels = canvas::grey_levels(&stages.spectrum, log);
        levels.sort_unstable();
        levels[levels.len() / 2]
    };

    let (linear, log) = (median(false), median(true));
    assert!(linear <= 2, "a linear ramp should leave most of it black, got {linear}");
    // Where the log puts the bulk depends on how much sideband energy the code
    // spreads, so this is a floor on the order of magnitude, not a fit.
    assert!(log >= 20, "the log should lift the bulk clear of black, got {log}");
    assert!(log > linear + 15, "log {log} and linear {linear} are too close");
}
