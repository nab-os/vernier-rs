//! The spectrum explorer view: one pattern, one camera, and every stage the
//! detector passes through laid out beside each other.
//!
//!   camera image → FFT and peak selection → reconstruction → phases
//!     → thumbnail and code
//!
//! Drag the camera image and watch the spectrum answer.

use dioxus::prelude::*;
use vernier_core::Real;
use vernier_cpu::CpuBackend;

use crate::camera::{self, Pose6};
use crate::canvas;
use crate::coding::{CENTRE_COLOUR, Source, Thumbnail};
use crate::controls::{NumberField, Section, Toggle};
use crate::pattern::{PatternKind, PatternSettings};
use crate::spectral::{self, Stages};

/// Sides the transform is run at. Every stage is `size × size`, and the cost is
/// dominated by the per-pixel render and the transform, so this is the one knob
/// that decides whether dragging keeps up.
const SIZE_PRESETS: [usize; 3] = [128, 256, 512];

/// DOM ids of the five stage canvases, in the order they are shown.
///
/// The spectrum appears once, with the peak selection drawn on it: two panels
/// showed the same pixels, and the picked peaks say more sitting on the
/// spectrum they were picked from than beside a copy of it. The band-passed
/// lobes are gone with them — two dots where the peak markers already are.
const PANELS: [(&str, &str); 5] = [
    ("explorer-image", "1. Camera image"),
    ("explorer-spectrum", "2. FFT · magnitude and peak selection"),
    ("explorer-reconstruction", "3. Reconstruction"),
    ("explorer-phases", "4. Wrapped phases · red 1, green 2"),
    ("explorer-thumbnail", "5. Extracted thumbnail · coding sites"),
];

/// Index of the panel the peak overlay belongs to.
const PEAKS_PANEL: usize = 1;
/// Index of the panel the coding overlay belongs to.
const THUMBNAIL_PANEL: usize = 4;

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
    /// Inner radius of the search band, in bins.
    pub min_frequency: usize,
    /// Blur applied before the peak search, in bins.
    pub smoothing_sigma: Real,
    /// Compress the FFT panel logarithmically. Without it a coded pattern's
    /// peaks leave the rest of the spectrum at black.
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

    /// Nyquist is the ceiling.
    pub fn max_frequency(&self) -> usize {
        self.size / 2 - 2
    }
}

/// What one pass produced, for the readouts.
#[derive(Clone, PartialEq, Debug, Default)]
struct Report {
    render_ms: f64,
    analyse_ms: f64,
    /// Time in the square sampling and the code search, which only the coded
    /// patterns pay.
    decode_ms: f64,
    error: Option<String>,
    peaks: Option<[(Real, Real); 2]>,
    plane_deg: Option<[Real; 2]>,
    measured_period_px: Option<Real>,
    /// Carrier period the camera should be producing, in image pixels, shown
    /// beside the measured one.
    expected_period_px: Real,
    /// The lattice-level extraction, for the patterns that carry a code.
    thumbnail: Option<Thumbnail>,
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

                // Only the coded patterns have a lattice to extract; the
                // rest of the chain is the same for every kind.
                if let Some(source) = coding_source(&pattern) {
                    let detection = stages.detection(view.size);
                    outcome.thumbnail =
                        Thumbnail::extract(&detection, image.as_slice(), source);
                }
                outcome.decode_ms = canvas::now_ms() - analysed;

                match &outcome.thumbnail {
                    Some(thumbnail) => {
                        if let Err(message) = canvas::paint_thumbnail(
                            PANELS[THUMBNAIL_PANEL].0,
                            thumbnail.side,
                            &thumbnail.levels,
                            &thumbnail.present,
                        ) {
                            outcome.error = Some(message);
                        }
                    }
                    None => {
                        let _ = canvas::clear(PANELS[THUMBNAIL_PANEL].0);
                    }
                }
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
                        p { class: "notice", "A stub upstream: nothing to look at." }
                    }
                    {crate::pattern_fields(settings, &pattern)}
                }

                Section { title: "Camera".to_string(),
                    p { class: "blurb",
                        "Drag: left translates, right rotates and changes distance, middle tilts. \
                         Shift for finer motion."
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
                        hint: String::new(),
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
                        hint: "Out of plane.".to_string(),
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
                        hint: String::new(),
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
                        hint: String::new(),
                        on_change: move |value: f64| {
                            explorer.write().supersample = value.round() as u32
                        },
                    }
                }

                Section { title: "Display".to_string(),
                    Toggle {
                        label: "Log scale on the FFT panel".to_string(),
                        checked: view.log_spectra,
                        hint: String::new(),
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
                         analyse {current.analyse_ms:.0} ms · decode {current.decode_ms:.0} ms"
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
                            overlay: match index {
                                PEAKS_PANEL => Overlay::Peaks,
                                THUMBNAIL_PANEL => Overlay::Coding,
                                _ => Overlay::None,
                            },
                            explorer,
                            size: view.size,
                            peaks: current.peaks,
                            thumbnail: current.thumbnail.clone(),
                            note: (index == THUMBNAIL_PANEL
                                && coding_source(&pattern).is_none())
                                .then(|| {
                                    "Only the coded patterns have a lattice to \
                                     extract. Pick Checkerboard or Megarena."
                                        .to_string()
                                }),
                        }
                    }
                }
            }
        }
    }
}

/// What is drawn on top of a stage canvas.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Overlay {
    None,
    /// The two carrier peaks, on the spectrum they were picked from.
    Peaks,
    /// The squares the code inverted, on the thumbnail they were read from.
    Coding,
}

/// One captioned stage canvas.
#[component]
#[allow(clippy::too_many_arguments)]
fn StagePanel(
    id: String,
    title: String,
    draggable: bool,
    overlay: Overlay,
    explorer: Signal<ExplorerSettings>,
    size: usize,
    peaks: Option<[(Real, Real); 2]>,
    thumbnail: Option<Thumbnail>,
    note: Option<String>,
) -> Element {
    // Where the pointer was last seen, so a move can be turned into a delta.
    let mut last = use_signal(|| None::<(f64, f64)>);

    rsx! {
        figure { class: "stage-panel",
            figcaption { class: "panel-title", "{title}" }
            div { class: "panel-canvas {draggable_class(draggable)}",
                canvas { id: "{id}", class: "preview" }

                match overlay {
                    Overlay::Peaks => peak_overlay(size, peaks),
                    Overlay::Coding => coding_overlay(thumbnail.as_ref()),
                    Overlay::None => rsx! {},
                }

                if let Some(note) = note {
                    p { class: "panel-note", "{note}" }
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

            // Four marker colours say nothing without this.
            if let Some(entries) = legend_of(overlay, thumbnail.as_ref()) {
                ul { class: "panel-legend",
                    for (colour, meaning) in entries {
                        li { key: "{meaning}",
                            span { class: "swatch", style: "background: {colour}" }
                            "{meaning}"
                        }
                    }
                }
            }
        }
    }
}

/// The colour key for a panel that has one.
fn legend_of(
    overlay: Overlay,
    thumbnail: Option<&Thumbnail>,
) -> Option<Vec<(&'static str, &'static str)>> {
    match overlay {
        Overlay::Coding => thumbnail.map(|t| t.legend.clone()),
        _ => None,
    }
}

/// Which decoder, if any, the selected pattern has a lattice for. The stubs
/// and the plain periodic carrier have no code to read, so their thumbnail
/// panel stays empty rather than showing a lattice that means nothing.
fn coding_source(pattern: &PatternSettings) -> Option<Source> {
    match pattern.kind {
        PatternKind::Checkerboard => Some(Source::Checkerboard {
            order: pattern.order,
            layout: pattern.code_layout,
            packing: pattern.code_packing,
        }),
        PatternKind::Megarena => Some(Source::Megarena { order: pattern.order }),
        PatternKind::Periodic | PatternKind::Stamp | PatternKind::QrLike => None,
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

/// The two peaks, drawn as SVG over the panel so they stay crisp however the
/// canvas is scaled to fit.
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

/// The coding sites, marked on the thumbnail they were read from, with a cross
/// on the node the image centre falls in — the one the decoded position is
/// reported for.
///
/// Drawn in cell coordinates, the same space the thumbnail canvas is painted
/// in, so a mark lands on its node however far the panel scales it up.
fn coding_overlay(thumbnail: Option<&Thumbnail>) -> Element {
    let Some(thumbnail) = thumbnail else {
        return rsx! {};
    };
    let side = thumbnail.side as f64;
    let (cx, cy) = thumbnail.centre;
    // A cell is one canvas pixel, so every length here is a fraction of one.
    // Hundreds of sites on a busy lattice need a filled mark to read at all:
    // an unfilled ring of the same size disappears into the pattern.
    let stroke = 0.09;
    let arm = 1.6;

    rsx! {
        svg {
            class: "overlay",
            view_box: "0 0 {side} {side}",
            preserve_aspect_ratio: "none",
            for site in thumbnail.sites.iter() {
                circle {
                    key: "{site.col}-{site.row}",
                    cx: "{site.col as f64 + 0.5}",
                    cy: "{site.row as f64 + 0.5}",
                    r: "0.34",
                    fill: site.mark.colour(),
                    fill_opacity: "0.8",
                    // Dark rim so a mark stays visible on a bright node too.
                    stroke: "#10151c",
                    stroke_width: "{stroke}",
                }
            }
            line {
                x1: "{cx - arm}", y1: "{cy}", x2: "{cx + arm}", y2: "{cy}",
                stroke: CENTRE_COLOUR, stroke_width: "{stroke * 2.0}",
            }
            line {
                x1: "{cx}", y1: "{cy - arm}", x2: "{cx}", y2: "{cy + arm}",
                stroke: CENTRE_COLOUR, stroke_width: "{stroke * 2.0}",
            }
        }
    }
}

/// Paints stages 2 to 4; stage 1 is painted by the caller, and stage 5 needs
/// the decode, which not every pattern gets.
fn paint_stages(stages: &Stages, view: &ExplorerSettings) -> Result<(), String> {
    canvas::paint_grey(PANELS[1].0, view.size, &stages.spectrum, view.log_spectra)?;
    canvas::paint_grey(PANELS[2].0, view.size, &stages.reconstruction, false)?;
    canvas::paint_phase_channels(PANELS[3].0, view.size, &stages.phase1, &stages.phase2)
}

/// The numbers under the stage head.
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

    if let Some(thumbnail) = &report.thumbnail {
        out.push((
            capitalized(thumbnail.unit),
            format!(
                "{} sampled · {} read · {} coding",
                thumbnail.sampled,
                thumbnail.judged,
                thumbnail.sites.len()
            ),
        ));
        if let Some((x_window, y_window)) = &thumbnail.windows {
            out.push(("Code window".to_string(), format!("{x_window} · {y_window}")));
        }
        out.push(("Decoded".to_string(), thumbnail.verdict.clone()));
    }
    out
}

/// The node name as a readout label.
fn capitalized(unit: &str) -> String {
    let mut chars = unit.chars();
    match chars.next() {
        Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
        None => String::new(),
    }
}
