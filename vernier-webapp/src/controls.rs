//! Reusable parameter widgets.
//!
//! Every numeric parameter gets both a slider and a number box: the slider to
//! sweep a value and watch the pattern move, the box to type the exact figure a
//! test or a datasheet calls for. The slider commits on every input event, the
//! box on blur or Enter — clamping mid-keystroke would fight anyone typing a
//! two-digit number into a field whose minimum is two digits.

use dioxus::prelude::*;

/// A labelled numeric parameter. Integer parameters pass `step: 1.0` and round
/// at the call site; everything here is `f64` so one widget covers both.
#[component]
pub fn NumberField(
    label: String,
    value: f64,
    min: f64,
    max: f64,
    step: f64,
    unit: String,
    hint: String,
    on_change: EventHandler<f64>,
) -> Element {
    let display = if step >= 1.0 {
        format!("{value:.0}")
    } else {
        format!("{value}")
    };

    rsx! {
        div { class: "field",
            div { class: "field-head",
                label { class: "field-label", "{label}" }
                div { class: "field-entry",
                    input {
                        class: "number",
                        r#type: "number",
                        min: "{min}",
                        max: "{max}",
                        step: "{step}",
                        value: "{display}",
                        onchange: move |event| {
                            if let Ok(parsed) = event.value().trim().parse::<f64>() {
                                on_change.call(parsed.clamp(min, max));
                            }
                        },
                    }
                    if !unit.is_empty() {
                        span { class: "unit", "{unit}" }
                    }
                }
            }
            input {
                class: "slider",
                r#type: "range",
                min: "{min}",
                max: "{max}",
                step: "{step}",
                value: "{value}",
                oninput: move |event| {
                    if let Ok(parsed) = event.value().parse::<f64>() {
                        on_change.call(parsed);
                    }
                },
            }
            if !hint.is_empty() {
                p { class: "hint", "{hint}" }
            }
        }
    }
}

/// A labelled on/off parameter.
#[component]
pub fn Toggle(label: String, checked: bool, hint: String, on_change: EventHandler<bool>) -> Element {
    rsx! {
        div { class: "field field-toggle",
            label { class: "toggle",
                input {
                    r#type: "checkbox",
                    checked: "{checked}",
                    onchange: move |event| on_change.call(event.checked()),
                }
                span { class: "field-label", "{label}" }
            }
            if !hint.is_empty() {
                p { class: "hint", "{hint}" }
            }
        }
    }
}

/// A titled group of fields.
#[component]
pub fn Section(title: String, children: Element) -> Element {
    rsx! {
        section { class: "panel-section",
            h2 { class: "section-title", "{title}" }
            {children}
        }
    }
}
