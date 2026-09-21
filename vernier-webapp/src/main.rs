//! A browser front-end for `vernier-patterns`.
//!
//! Pick a generator, tune every parameter it exposes, and watch the render
//! update. The pattern itself is produced by the same `vernier-patterns` code
//! the CLI and the test suite call — this crate only compiles it to wasm, wraps
//! its parameters in widgets, and blits the resulting intensity field onto a
//! canvas. Nothing about the pattern is reimplemented here, so what you see is
//! what the library generates.

mod camera;
mod canvas;
mod controls;
mod explorer;
#[cfg(test)]
mod explorer_tests;
mod pattern;
mod spectral;

use dioxus::prelude::*;

use controls::{NumberField, Section, Toggle};
use explorer::{Explorer, ExplorerSettings};
use pattern::{PatternKind, PatternSettings, MAX_SIDE, MIN_SIDE, ORDER_RANGE};

const MAIN_CSS: Asset = asset!("/assets/main.css");

/// Image side presets, the sizes the CLI and benchmarks use.
const SIZE_PRESETS: [usize; 4] = [256, 512, 1024, 2048];

fn main() {
    dioxus::launch(App);
}

/// Outcome of the last render: how long it took, and why it failed if it did.
#[derive(Clone, PartialEq, Debug, Default)]
struct RenderReport {
    elapsed_ms: f64,
    error: Option<String>,
}

/// Which of the two tools is on screen. They share the pattern selection, so
/// switching views keeps whatever pattern was being looked at.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum View {
    Generator,
    Explorer,
}

#[component]
fn App() -> Element {
    let settings = use_signal(PatternSettings::default);
    let mut view = use_signal(|| View::Generator);
    let explorer = use_signal(|| {
        ExplorerSettings::for_period(PatternSettings::default().explorer_period_px())
    });

    rsx! {
        document::Title { "Vernier patterns" }
        document::Link { rel: "stylesheet", href: MAIN_CSS }

        header { class: "masthead",
            div { class: "masthead-head",
                h1 {
                    match view() {
                        View::Generator => "Vernier pattern generator",
                        View::Explorer => "Vernier spectrum explorer",
                    }
                }
                nav { class: "view-switch",
                    button {
                        r#type: "button",
                        class: "view-button {selected(view() == View::Generator)}",
                        onclick: move |_| view.set(View::Generator),
                        "Generator"
                    }
                    button {
                        r#type: "button",
                        class: "view-button {selected(view() == View::Explorer)}",
                        onclick: move |_| view.set(View::Explorer),
                        "Spectrum explorer"
                    }
                }
            }
            p { class: "subtitle",
                match view() {
                    View::Generator => rsx! {
                        "Every pattern below is rendered by "
                        code { "vernier-patterns" }
                        " compiled to WebAssembly — the same generators the CLI and the round-trip tests use."
                    },
                    View::Explorer => rsx! {
                        "The detector runs in this page: "
                        code { "vernier-spectral" }
                        " compiled to WebAssembly, transforming each frame as you move the camera."
                    },
                }
            }
        }

        match view() {
            View::Generator => rsx! { GeneratorView { settings } },
            View::Explorer => rsx! { Explorer { settings, explorer } },
        }
    }
}

#[component]
fn GeneratorView(settings: Signal<PatternSettings>) -> Element {
    let mut settings = settings;
    let mut report = use_signal(RenderReport::default);
    let mut actual_size = use_signal(|| true);

    // Re-render whenever any parameter changes. `settings` is one signal holding
    // the whole parameter set, so every edit invalidates this effect and nothing
    // else has to be wired up per-field.
    use_effect(move || {
        let current = settings.read().clone();
        let started = canvas::now_ms();

        let outcome = current.render().and_then(|image| {
            let rgba = pattern::to_rgba(&image, current.invert);
            canvas::paint(image.width(), image.height(), &rgba)
        });

        report.set(RenderReport {
            elapsed_ms: canvas::now_ms() - started,
            error: outcome.err(),
        });
    });

    let current = settings.read().clone();

    rsx! {
        main { class: "layout",
            aside { class: "panel",

                Section { title: "Pattern type".to_string(),
                    div { class: "type-grid",
                        for kind in PatternKind::ALL {
                            button {
                                key: "{kind.label()}",
                                r#type: "button",
                                class: "type-button {selected(current.kind == kind)}",
                                onclick: move |_| settings.write().kind = kind,
                                "{kind.label()}"
                            }
                        }
                    }
                    p { class: "blurb", "{current.kind.blurb()}" }
                    if current.kind.is_stub() {
                        p { class: "notice",
                            "This generator is a stub upstream: "
                            code { "{current.kind.source_path()}" }
                            " returns a blank field until the layout is implemented. \
                             The controls below are the real parameters, but the render will be black."
                        }
                    }
                }

                Section { title: "Image".to_string(),
                    div { class: "preset-row",
                        for side in SIZE_PRESETS {
                            button {
                                key: "{side}",
                                r#type: "button",
                                class: "preset {selected(current.width == side && current.height == side)}",
                                onclick: move |_| {
                                    let mut write = settings.write();
                                    write.width = side;
                                    write.height = side;
                                },
                                "{side}²"
                            }
                        }
                    }
                    NumberField {
                        label: "Width".to_string(),
                        value: current.width as f64,
                        min: MIN_SIDE as f64,
                        max: MAX_SIDE as f64,
                        step: 1.0,
                        unit: "px".to_string(),
                        hint: String::new(),
                        on_change: move |value: f64| settings.write().width = value.round() as usize,
                    }
                    NumberField {
                        label: "Height".to_string(),
                        value: current.height as f64,
                        min: MIN_SIDE as f64,
                        max: MAX_SIDE as f64,
                        step: 1.0,
                        unit: "px".to_string(),
                        hint: String::new(),
                        on_change: move |value: f64| settings.write().height = value.round() as usize,
                    }
                }

                Section { title: "Pattern parameters".to_string(),
                    {pattern_fields(settings, &current)}
                }

                Section { title: "Pose".to_string(),
                    NumberField {
                        label: "X translation".to_string(),
                        value: current.pose_x,
                        min: -1000.0,
                        max: 1000.0,
                        step: 0.1,
                        unit: "px".to_string(),
                        hint: String::new(),
                        on_change: move |value: f64| settings.write().pose_x = value,
                    }
                    NumberField {
                        label: "Y translation".to_string(),
                        value: current.pose_y,
                        min: -1000.0,
                        max: 1000.0,
                        step: 0.1,
                        unit: "px".to_string(),
                        hint: String::new(),
                        on_change: move |value: f64| settings.write().pose_y = value,
                    }
                    NumberField {
                        label: "Orientation".to_string(),
                        value: current.theta_deg,
                        min: -180.0,
                        max: 180.0,
                        step: 0.1,
                        unit: "°".to_string(),
                        hint: format!("{:.5} rad", current.theta_deg.to_radians()),
                        on_change: move |value: f64| settings.write().theta_deg = value,
                    }
                }

                Section { title: "Display".to_string(),
                    Toggle {
                        label: "Invert".to_string(),
                        checked: current.invert,
                        hint: "Swaps black and white in the preview and the PNG. Does not touch the generator."
                            .to_string(),
                        on_change: move |value: bool| settings.write().invert = value,
                    }
                    Toggle {
                        label: "Show at actual size".to_string(),
                        checked: actual_size(),
                        hint: "Off scales the preview to fit the panel. The PNG is always full resolution."
                            .to_string(),
                        on_change: move |value: bool| actual_size.set(value),
                    }
                }

                div { class: "actions",
                    button {
                        r#type: "button",
                        class: "action primary",
                        onclick: move |_| {
                            let stem = settings.read().file_stem();
                            if let Err(message) = canvas::download_png(&stem) {
                                report.write().error = Some(message);
                            }
                        },
                        "Download PNG"
                    }
                    button {
                        r#type: "button",
                        class: "action",
                        onclick: move |_| settings.set(PatternSettings::default()),
                        "Reset"
                    }
                }
            }

            section { class: "stage",
                div { class: "stage-head",
                    div { class: "readouts",
                        for (name, value) in current.derived() {
                            div { key: "{name}", class: "readout",
                                span { class: "readout-name", "{name}" }
                                span { class: "readout-value", "{value}" }
                            }
                        }
                    }
                    p { class: "timing",
                        "{current.width}×{current.height} px · {report.read().elapsed_ms:.1} ms"
                    }
                }

                if let Some(message) = report.read().error.clone() {
                    p { class: "error", "{message}" }
                }

                div { class: "canvas-frame {fit_class(actual_size())}",
                    canvas { id: canvas::CANVAS_ID, class: "preview" }
                }

                details { class: "snippet",
                    summary { "Equivalent vernier-patterns call" }
                    pre { code { "{current.equivalent_rust()}" } }
                }
            }
        }
    }
}

/// The fields specific to the selected generator. Split out of [`App`] so the
/// per-kind parameter list stays readable next to the shared sections.
pub(crate) fn pattern_fields(mut settings: Signal<PatternSettings>, current: &PatternSettings) -> Element {
    match current.kind {
        PatternKind::Periodic => rsx! {
            NumberField {
                label: "Spatial period".to_string(),
                value: current.period_px,
                min: 2.0,
                max: 200.0,
                step: 0.1,
                unit: "px".to_string(),
                hint: "Distance between bright stripes.".to_string(),
                on_change: move |value: f64| settings.write().period_px = value,
            }
        },
        PatternKind::Megarena => rsx! {
            NumberField {
                label: "Dot period".to_string(),
                value: current.period_px,
                min: 2.0,
                max: 200.0,
                step: 0.1,
                unit: "px".to_string(),
                hint: "One carrier period. Three of these carry one code bit.".to_string(),
                on_change: move |value: f64| settings.write().period_px = value,
            }
            NumberField {
                label: "LFSR order".to_string(),
                value: current.order as f64,
                min: *ORDER_RANGE.start() as f64,
                max: *ORDER_RANGE.end() as f64,
                step: 1.0,
                unit: "bits".to_string(),
                hint: "Bits per unique window — sets the absolute range.".to_string(),
                on_change: move |value: f64| settings.write().order = value.round() as u32,
            }
            NumberField {
                label: "LFSR offset".to_string(),
                value: current.lfsr_offset as f64,
                min: 0.0,
                max: 512.0,
                step: 1.0,
                unit: String::new(),
                hint: "Code index placed at the centre triple. Upstream suggests setting it to the order."
                    .to_string(),
                on_change: move |value: f64| settings.write().lfsr_offset = value.round() as i64,
            }
        },
        PatternKind::Checkerboard => rsx! {
            NumberField {
                label: "Square side".to_string(),
                value: current.square_px,
                min: 2.0,
                max: 200.0,
                step: 0.1,
                unit: "px".to_string(),
                hint: format!(
                    "One checkerboard square. The carriers run diagonally, so a fringe is {:.2} px \
                     (a√2) apart — that, not this, is the period a detector is given.",
                    current.carrier_period_px()
                ),
                on_change: move |value: f64| settings.write().square_px = value,
            }
            NumberField {
                label: "LFSR order".to_string(),
                value: current.order as f64,
                min: *ORDER_RANGE.start() as f64,
                max: *ORDER_RANGE.end() as f64,
                step: 1.0,
                unit: "bits".to_string(),
                hint: "Bits per unique window — sets the absolute range.".to_string(),
                on_change: move |value: f64| settings.write().order = value.round() as u32,
            }
            NumberField {
                label: "LFSR offset".to_string(),
                value: current.lfsr_offset as f64,
                min: 0.0,
                max: 512.0,
                step: 1.0,
                unit: String::new(),
                hint: "Code index placed at supercell 0.".to_string(),
                on_change: move |value: f64| settings.write().lfsr_offset = value.round() as i64,
            }
            NumberField {
                label: "Supersampling".to_string(),
                value: current.supersample as f64,
                min: 1.0,
                max: 8.0,
                step: 1.0,
                unit: "×/edge".to_string(),
                hint: "Sub-samples per pixel edge. The pattern has hard edges; 1× point-samples \
                       them and aliases, which shifts the measured carrier phase."
                    .to_string(),
                on_change: move |value: f64| settings.write().supersample = value.round() as u32,
            }
            Toggle {
                label: "Uncoded reference".to_string(),
                checked: current.plain_checkerboard,
                hint: "Renders the plain checkerboard with every coding site left at its parity \
                       colour, so you can see exactly which squares the code inverts."
                    .to_string(),
                on_change: move |value: bool| settings.write().plain_checkerboard = value,
            }
        },
        PatternKind::Stamp => rsx! {
            NumberField {
                label: "Tile size".to_string(),
                value: current.tile_px as f64,
                min: 4.0,
                max: 512.0,
                step: 1.0,
                unit: "px".to_string(),
                hint: "Side of one stamp tile.".to_string(),
                on_change: move |value: f64| settings.write().tile_px = value.round() as usize,
            }
        },
        PatternKind::QrLike => rsx! {
            NumberField {
                label: "Modules".to_string(),
                value: current.modules as f64,
                min: 5.0,
                max: 177.0,
                step: 1.0,
                unit: "per axis".to_string(),
                hint: "Cells along each axis.".to_string(),
                on_change: move |value: f64| settings.write().modules = value.round() as usize,
            }
            NumberField {
                label: "Module size".to_string(),
                value: current.module_px as f64,
                min: 1.0,
                max: 64.0,
                step: 1.0,
                unit: "px".to_string(),
                hint: "Pixels per cell.".to_string(),
                on_change: move |value: f64| settings.write().module_px = value.round() as usize,
            }
        },
    }
}

/// Class fragment marking the active choice in a button group.
pub(crate) fn selected(active: bool) -> &'static str {
    if active { "is-active" } else { "" }
}

/// Class fragment switching the preview between 1:1 and scale-to-fit.
fn fit_class(actual_size: bool) -> &'static str {
    if actual_size { "is-actual" } else { "is-fit" }
}
