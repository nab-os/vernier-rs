//! The spectrum explorer view.
//!
//! One pattern, one camera, and every stage the detector passes through laid
//! out beside each other:
//!
//!   camera image → FFT → peak selection → band-pass → reconstruction → phases
//!
//! Drag the camera image and watch the spectrum answer. Translation leaves the
//! peaks where they are and only turns the phase; rotation about Z swings them
//! around the origin; distance moves them radially; the out-of-plane angles
//! pull the pair off its right angle until the search band or the Gaussian
//! window stops tracking them. Changing the source pattern changes what sits
//! around those peaks — a bare harmonic lattice for the periodic grid, a
//! thicket of code sidebands for the megarena and the checkerboard.

use dioxus::prelude::*;
use vernier_core::Real;
use vernier_cpu::CpuBackend;

use crate::camera::{self, Pose6};
use crate::canvas;
use crate::controls::{NumberField, Section, Toggle};
use crate::pattern::PatternSettings;
use crate::spectral::{self, Stages};

/// Sides the transform is run at. Every stage is `size × size`, and the cost is
/// dominated by the per-pixel render and the transform, so this is the one knob
/// that decides whether dragging keeps up.
const SIZE_PRESETS: [usize; 3] = [128, 256, 512];

/// DOM ids of the six stage canvases, in the order they are shown.
const PANELS: [(&str, &str); 6] = [
    ("explorer-image", "1. Camera image"),
    ("explorer-spectrum", "2. FFT · magnitude"),
    ("explorer-peaks", "3. Peak selection"),
    ("explorer-filtered", "4. Band-passed lobes"),
    ("explorer-reconstruction", "5. Reconstruction"),
    ("explorer-phases", "6. Wrapped phases 1 / 2"),
];

const MAX_TILT_DEG: Real = 63.0;
const DRAG_ALPHA_DEG: Real = 0.25;
const DRAG_TILT_DEG: Real = 0.2;
const DRAG_ZOOM: Real = 0.005;
const WHEEL_ZOOM: Real = 0.0015;

/// How the explorer is set up, apart from the pattern itself.
#[derive(Clone, PartialEq, Debug)]
pub struct ExplorerSettings {
    pub pose: Pose6,
    pub size: usize,
    pub supersample: u32,
    /// Gaussian band-pass width, in bins.
    pub sigma: Real,
    /// Inner radius of the search band, in bins. Carriers inside it are
    /// rejected as low-frequency background.
    pub min_frequency: usize,
    /// Blur applied before the peak search, in bins.
    pub smoothing_sigma: Real,
    /// Compress the FFT panels logarithmically. On a coded pattern the carrier
    /// peaks stand orders of magnitude above the sidebands, so a linear ramp
    /// shows two white dots on black and nothing else; the log is what makes
    /// the rest of the spectrum visible. Linear is the honest view of the
    /// magnitudes, and the one that shows how completely the peaks dominate.
    pub log_spectra: bool,
}

impl ExplorerSettings {
    pub fn for_period(carrier_period_px: Real) -> Self {
        Self {
            pose: Pose6::at_distance(camera::home_distance(carrier_period_px)),
            size: 256,
            supersample: 2,
            sigma: 3.0,
            min_frequency: 10,
            smoothing_sigma: 1.0,
            log_spectra: true,
        }
    }

    /// Nyquist is the ceiling: no bin beyond it carries a frequency the image
    /// can represent.
    pub fn max_frequency(&self) -> usize {
        self.size / 2 - 2
    }
}

/// What one pass produced, for the readouts.
#[derive(Clone, PartialEq, Debug, Default)]
struct Report {
    render_ms: f64,
    analyse_ms: f64,
    error: Option<String>,
    peaks: Option<[(Real, Real); 2]>,
    plane_deg: Option<[Real; 2]>,
    measured_period_px: Option<Real>,
    /// Carrier period the camera should be producing, in image pixels: the
    /// pattern's own fringe spacing times the magnification. Shown beside the
    /// measured one, because the two agreeing is the whole claim.
    expected_period_px: Real,
}

#[component]
pub fn Explorer(settings: Signal<PatternSettings>, explorer: Signal<ExplorerSettings>) -> Element {
    let mut report = use_signal(Report::default);

    // One effect over both signals: any edit to the pattern or the camera
    // invalidates the whole chain, and nothing here is cheap enough to be worth
    // splitting into finer dependencies.
    use_effect(move || {
        let pattern = settings.read().clone();
        let view = explorer.read().clone();

        let sampler = match pattern.sampler() {
            Ok(sampler) => sampler,
            Err(message) => {
                report.set(Report {
                    error: Some(message),
                    expected_period_px: pattern.explorer_period_px()
                        * view.pose.magnification(),
                    ..Default::default()
                });
                for (id, _) in PANELS {
                    let _ = canvas::clear(id);
                }
                return;
            }
        };

        let started = canvas::now_ms();
        let image = camera::render(&sampler, &view.pose, view.size, view.supersample);
        let rendered = canvas::now_ms();

        let stages = spectral::run(
            &CpuBackend::new(),
            image.as_slice(),
            view.size,
            view.sigma,
            view.min_frequency,
            view.max_frequency(),
            view.smoothing_sigma,
        );
        let analysed = canvas::now_ms();

        let mut outcome = Report {
            render_ms: rendered - started,
            analyse_ms: analysed - rendered,
            expected_period_px: pattern.explorer_period_px() * view.pose.magnification(),
            ..Default::default()
        };

        if let Err(message) = canvas::paint_grey(PANELS[0].0, view.size, image.as_slice(), false) {
            outcome.error = Some(message);
        }

        match stages {
            Ok(Some(stages)) => {
                if let Err(message) = paint_stages(&stages, &view) {
                    outcome.error = Some(message);
                }
                outcome.peaks = Some(stages.peaks);
                outcome.plane_deg = Some([
                    stages.planes[0].orientation().to_degrees(),
                    stages.planes[1].orientation().to_degrees(),
                ]);
                let (px, py) = stages.peaks[0];
                let radius = (px * px + py * py).sqrt();
                outcome.measured_period_px =
                    (radius > 0.0).then(|| view.size as Real / radius);
            }
            Ok(None) => {
                outcome.error = Some(
                    "No carrier peaks inside the search band. Lower the band floor, \
                     or back the camera off so the fringes are finer."
                        .to_string(),
                );
                for (id, _) in &PANELS[1..] {
                    let _ = canvas::clear(id);
                }
            }
            Err(error) => outcome.error = Some(format!("{error}")),
        }

        report.set(outcome);
    });

    let view = explorer.read().clone();
    let pattern = settings.read().clone();
    let current = report.read().clone();

    rsx! {
        main { class: "layout",
            aside { class: "panel",
                Section { title: "Source pattern".to_string(),
                    div { class: "type-grid",
                        for kind in crate::pattern::PatternKind::ALL {
                            button {
                                key: "{kind.label()}",
                                r#type: "button",
                                class: "type-button {crate::selected(pattern.kind == kind)} {availability(kind)}",
                                onclick: move |_| {
                                    let mut write = settings.write();
                                    write.kind = kind;
                                },
                                "{kind.label()}"
                            }
                        }
                    }
                    p { class: "blurb", "{pattern.kind.blurb()}" }
                    if !pattern.kind.has_point_sampler() {
                        p { class: "notice",
                            "This generator is a stub upstream and ignores the pose, so there is \
                             nothing for a camera to look at. The other three work."
                        }
                    }
                    {crate::pattern_fields(settings, &pattern)}
                }

                Section { title: "Camera".to_string(),
                    p { class: "blurb",
                        "Drag the camera image: left to translate, right to rotate and change \
                         distance, middle to tilt out of plane. Hold shift for finer motion."
                    }
                    NumberField {
                        label: "Distance".to_string(),
                        value: view.pose.z,
                        min: 100.0,
                        max: 40000.0,
                        step: 1.0,
                        unit: "px".to_string(),
                        hint: format!("{:.3} image px per pattern px", view.pose.magnification()),
                        on_change: move |value: f64| explorer.write().pose.z = value,
                    }
                    NumberField {
                        label: "Rotation about Z".to_string(),
                        value: view.pose.alpha.to_degrees(),
                        min: -180.0,
                        max: 180.0,
                        step: 0.1,
                        unit: "°".to_string(),
                        hint: "Swings both peaks around the origin.".to_string(),
                        on_change: move |value: f64| {
                            explorer.write().pose.alpha = value.to_radians()
                        },
                    }
                    NumberField {
                        label: "Tilt about Y".to_string(),
                        value: view.pose.beta.to_degrees(),
                        min: -MAX_TILT_DEG,
                        max: MAX_TILT_DEG,
                        step: 0.1,
                        unit: "°".to_string(),
                        hint: "Out of plane: pulls the peak pair off its right angle.".to_string(),
                        on_change: move |value: f64| {
                            explorer.write().pose.beta = value.to_radians()
                        },
                    }
                    NumberField {
                        label: "Tilt about X".to_string(),
                        value: view.pose.gamma.to_degrees(),
                        min: -MAX_TILT_DEG,
                        max: MAX_TILT_DEG,
                        step: 0.1,
                        unit: "°".to_string(),
                        hint: String::new(),
                        on_change: move |value: f64| {
                            explorer.write().pose.gamma = value.to_radians()
                        },
                    }
                }

                Section { title: "Analysis".to_string(),
                    div { class: "preset-row",
                        for side in SIZE_PRESETS {
                            button {
                                key: "{side}",
                                r#type: "button",
                                class: "preset {crate::selected(view.size == side)}",
                                onclick: move |_| explorer.write().size = side,
                                "{side}²"
                            }
                        }
                    }
                    NumberField {
                        label: "Window sigma".to_string(),
                        value: view.sigma,
                        min: 0.5,
                        max: 20.0,
                        step: 0.1,
                        unit: "bins".to_string(),
                        hint: "Width of the Gaussian kept around each peak.".to_string(),
                        on_change: move |value: f64| explorer.write().sigma = value,
                    }
                    NumberField {
                        label: "Band floor".to_string(),
                        value: view.min_frequency as f64,
                        min: 2.0,
                        max: 60.0,
                        step: 1.0,
                        unit: "bins".to_string(),
                        hint: "Carriers closer to DC than this are rejected.".to_string(),
                        on_change: move |value: f64| {
                            explorer.write().min_frequency = value.round() as usize
                        },
                    }
                    NumberField {
                        label: "Supersampling".to_string(),
                        value: view.supersample as f64,
                        min: 1.0,
                        max: 4.0,
                        step: 1.0,
                        unit: "×/edge".to_string(),
                        hint: "The coded patterns have hard edges; 1× aliases them.".to_string(),
                        on_change: move |value: f64| {
                            explorer.write().supersample = value.round() as u32
                        },
                    }
                }

                Section { title: "Display".to_string(),
                    Toggle {
                        label: "Log scale on the FFT panels".to_string(),
                        checked: view.log_spectra,
                        hint: if view.log_spectra {
                            "On: sidebands and the noise floor are visible beside the peaks."
                                .to_string()
                        } else {
                            "Off: true magnitudes, and the carrier peaks swamp everything else."
                                .to_string()
                        },
                        on_change: move |value: bool| explorer.write().log_spectra = value,
                    }
                }

                div { class: "actions",
                    button {
                        r#type: "button",
                        class: "action",
                        onclick: move |_| {
                            let period = settings.read().explorer_period_px();
                            explorer.write().pose =
                                Pose6::at_distance(camera::home_distance(period));
                        },
                        "Reset camera"
                    }
                }
            }

            section { class: "stage",
                div { class: "stage-head",
                    div { class: "readouts",
                        for (name, value) in readouts(&current) {
                            div { key: "{name}", class: "readout",
                                span { class: "readout-name", "{name}" }
                                span { class: "readout-value", "{value}" }
                            }
                        }
                    }
                    p { class: "timing",
                        "{view.size}×{view.size} px · render {current.render_ms:.0} ms · \
                         analyse {current.analyse_ms:.0} ms"
                    }
                }

                if let Some(message) = current.error.clone() {
                    p { class: "error", "{message}" }
                }

                div { class: "stage-grid",
                    for (index, (id, title)) in PANELS.into_iter().enumerate() {
                        StagePanel {
                            key: "{id}",
                            id: id.to_string(),
                            title: title.to_string(),
                            draggable: index == 0,
                            overlay: index == 2,
                            explorer,
                            settings,
                            size: view.size,
                            peaks: current.peaks,
                        }
                    }
                }
            }
        }
    }
}

/// One captioned stage canvas. The camera image is the one that takes the
/// pointer; the peak-selection panel carries the overlay.
#[component]
#[allow(clippy::too_many_arguments)]
fn StagePanel(
    id: String,
    title: String,
    draggable: bool,
    overlay: bool,
    explorer: Signal<ExplorerSettings>,
    settings: Signal<PatternSettings>,
    size: usize,
    peaks: Option<[(Real, Real); 2]>,
) -> Element {
    // Where the pointer was last seen, so a move can be turned into a delta.
    let mut last = use_signal(|| None::<(f64, f64)>);

    rsx! {
        figure { class: "stage-panel",
            figcaption { class: "panel-title", "{title}" }
            div { class: "panel-canvas {draggable_class(draggable)}",
                canvas { id: "{id}", class: "preview" }

                if overlay {
                    {peak_overlay(size, peaks)}
                }

                if draggable {
                    div {
                        class: "drag-surface",
                        // The right button has to be claimed here or the browser
                        // menu opens on top of the drag.
                        oncontextmenu: move |event| event.prevent_default(),
                        onpointerdown: move |event| {
                            let point = event.data().client_coordinates();
                            last.set(Some((point.x, point.y)));
                        },
                        onpointerup: move |_| last.set(None),
                        onpointerleave: move |_| last.set(None),
                        onpointermove: move |event| {
                            let Some((px, py)) = last() else { return };
                            let point = event.data().client_coordinates();
                            let (dx, dy) = (point.x - px, point.y - py);
                            last.set(Some((point.x, point.y)));

                            let buttons = event.data().held_buttons();
                            let fine = if event.data().modifiers().shift() { 0.1 } else { 1.0 };
                            let mut view = explorer.write();

                            if buttons.contains(dioxus::html::input_data::MouseButton::Primary) {
                                view.pose.translate_by_drag(-dx * fine, -dy * fine);
                            }
                            if buttons.contains(dioxus::html::input_data::MouseButton::Secondary) {
                                view.pose.alpha += (dx * DRAG_ALPHA_DEG * fine).to_radians();
                                view.pose.z = clamp_distance(
                                    view.pose.z * (dy * DRAG_ZOOM * fine).exp(),
                                );
                            }
                            if buttons.contains(dioxus::html::input_data::MouseButton::Auxiliary) {
                                let limit = MAX_TILT_DEG.to_radians();
                                view.pose.beta = (view.pose.beta
                                    + (dx * DRAG_TILT_DEG * fine).to_radians())
                                    .clamp(-limit, limit);
                                view.pose.gamma = (view.pose.gamma
                                    + (dy * DRAG_TILT_DEG * fine).to_radians())
                                    .clamp(-limit, limit);
                            }
                        },
                        onwheel: move |event| {
                            // Claimed so the page does not scroll under the drag.
                            event.prevent_default();
                            let delta = event.data().delta().strip_units().y;
                            let mut view = explorer.write();
                            view.pose.z = clamp_distance(view.pose.z * (delta * WHEEL_ZOOM).exp());
                        },
                    }
                }
            }
        }
    }
}

/// Class fragment dimming a generator the camera cannot sample.
fn availability(kind: crate::pattern::PatternKind) -> &'static str {
    if kind.has_point_sampler() { "" } else { "is-unavailable" }
}

/// Class fragment marking the panel that takes the pointer.
fn draggable_class(draggable: bool) -> &'static str {
    if draggable { "is-draggable" } else { "" }
}

fn clamp_distance(z: Real) -> Real {
    z.clamp(100.0, 40000.0)
}

/// The band-pass rings, the two peaks and the Gaussian windows sitting on them,
/// drawn as SVG over the panel so the rings stay crisp however the canvas is
/// scaled to fit.
fn peak_overlay(size: usize, peaks: Option<[(Real, Real); 2]>) -> Element {
    let Some(peaks) = peaks else {
        return rsx! {};
    };
    let side = size as f64;
    let centre = side / 2.0;
    let colours = ["#ff5f5f", "#6dff6d"];

    rsx! {
        svg {
            class: "overlay",
            view_box: "0 0 {side} {side}",
            preserve_aspect_ratio: "none",
            for (index, (px, py)) in peaks.into_iter().enumerate() {
                g { key: "{index}",
                    line {
                        x1: "{centre}",
                        y1: "{centre}",
                        x2: "{centre + px}",
                        y2: "{centre + py}",
                        stroke: "{colours[index]}",
                        stroke_width: "1",
                    }
                    circle {
                        cx: "{centre + px}",
                        cy: "{centre + py}",
                        r: "6",
                        fill: "none",
                        stroke: "{colours[index]}",
                        stroke_width: "1",
                    }
                }
            }
        }
    }
}

/// Paints stages 2 to 6. Stage 1 is painted by the caller, which is the only
/// one that exists even when the peak search comes up empty.
fn paint_stages(stages: &Stages, view: &ExplorerSettings) -> Result<(), String> {
    let log = view.log_spectra;
    canvas::paint_grey(PANELS[1].0, view.size, &stages.spectrum, log)?;
    canvas::paint_grey(PANELS[2].0, view.size, &stages.spectrum, log)?;
    canvas::paint_grey(PANELS[3].0, view.size, &stages.filtered, log)?;
    canvas::paint_grey(PANELS[4].0, view.size, &stages.reconstruction, false)?;
    canvas::paint_phases(PANELS[5].0, view.size, &stages.phase1, &stages.phase2)
}

/// The numbers under the stage head: where the peaks are and what the planes
/// made of them.
fn readouts(report: &Report) -> Vec<(String, String)> {
    let mut out = Vec::new();

    if let Some(peaks) = report.peaks {
        for (index, (px, py)) in peaks.into_iter().enumerate() {
            out.push((
                format!("Peak {}", index + 1),
                format!(
                    "r {:.1} bins · {:.2}°",
                    (px * px + py * py).sqrt(),
                    py.atan2(px).to_degrees()
                ),
            ));
        }
        let (x1, y1) = peaks[0];
        let (x2, y2) = peaks[1];
        let separation = (y1.atan2(x1) - y2.atan2(x2)).to_degrees().abs().rem_euclid(180.0);
        out.push(("Separation".to_string(), format!("{separation:.2}°")));
    }

    if let Some([a, b]) = report.plane_deg {
        out.push(("Planes".to_string(), format!("{a:.3}° · {b:.3}°")));
    }
    if let Some(period) = report.measured_period_px {
        out.push(("Measured carrier".to_string(), format!("{period:.2} px")));
    }
    out.push((
        "Expected carrier".to_string(),
        format!("{:.2} px", report.expected_period_px),
    ));
    out
}
