//! Numeric checks on the explorer's chain, run natively: render a pattern
//! through the camera, run the detector, and assert the spectrum landed where
//! the geometry says it must.

use vernier_core::Real;
use vernier_cpu::CpuBackend;
use vernier_patterns::checkerboard::CodeLayout;

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

/// The home distance puts `TARGET_SAMPLES_PER_PERIOD` pixels on one carrier
/// period, fixing the peak radius at `size / 8` bins. Every pattern has to land
/// there — including the checkerboard, whose period is the diagonal.
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

/// The two carriers of each of these patterns are orthogonal.
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

/// Squares leaves the carriers on the diagonals; Diamonds turns the lattice 45°
/// and puts them on the axes.
#[test]
fn the_code_layout_decides_where_the_carriers_point() {
    for (layout, offset_deg) in [(CodeLayout::Squares, 45.0), (CodeLayout::Diamonds, 0.0)] {
        let pattern =
            PatternSettings { code_layout: layout, ..settings_for(PatternKind::Checkerboard) };
        let view = view_for(&pattern, 128);
        let stages = analyse_with(&pattern, &view).expect("carriers are found");

        for peak in stages.peaks {
            let (_, angle) = polar(peak);
            let within = angle.rem_euclid(90.0);
            let offset = (within - offset_deg).abs().min((90.0 - offset_deg - within).abs());
            assert!(
                offset <= 5.0,
                "{layout:?}: peak at {angle:.2} deg is not {offset_deg} deg off an axis"
            );
        }
    }
}

/// Out-of-plane tilt is the freedom `PatternPose` cannot express, and the reason
/// this crate carries its own camera: under perspective the two carriers stop
/// being orthogonal in the image.
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

/// A coded pattern's peaks stand so far above its sidebands that a linear ramp
/// crushes most of the spectrum to black.
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
    assert!(log >= 20, "the log should lift the bulk clear of black, got {log}");
}
