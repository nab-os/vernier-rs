//! The parameter model the UI edits, and its translation into `vernier-patterns`
//! calls.
//!
//! Everything the generators accept is represented here as one flat, cloneable
//! [`PatternSettings`] — the widgets edit fields, [`PatternSettings::render`]
//! turns the whole thing into a [`GrayImage`]. Keeping the mapping in one place
//! means a new generator parameter needs a field and one match arm, not a tour
//! of the component tree.

use vernier_core::{GrayImage, Real};
use vernier_patterns::megarena::Megarena;
use vernier_patterns::periodic::Periodic;
use vernier_patterns::qrcode::QrLike;
use vernier_patterns::stamp::Stamp;
use vernier_patterns::PatternPose;

/// LFSR orders `vernier_patterns::lfsr::Lfsr::maximal` accepts.
pub const ORDER_RANGE: std::ops::RangeInclusive<u32> = 4..=12;

/// Largest image side the UI will render. The generators are per-pixel CPU
/// loops, so this is a responsiveness limit, not a correctness one.
pub const MAX_SIDE: usize = 2048;
/// Smallest image side worth rendering.
pub const MIN_SIDE: usize = 16;

/// Which generator to run.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PatternKind {
    /// Sinusoidal carrier — `vernier_patterns::periodic::Periodic`.
    Periodic,
    /// Absolute LFSR-coded dot grid — `vernier_patterns::megarena::Megarena`.
    Megarena,
    /// `vernier_patterns::stamp::Stamp` (layout not implemented upstream).
    Stamp,
    /// `vernier_patterns::qrcode::QrLike` (encoding not implemented upstream).
    QrLike,
}

impl PatternKind {
    /// Every kind, in the order the type selector shows them.
    pub const ALL: [PatternKind; 4] = [
        PatternKind::Periodic,
        PatternKind::Megarena,
        PatternKind::Stamp,
        PatternKind::QrLike,
    ];

    /// Short name for the type selector.
    pub fn label(self) -> &'static str {
        match self {
            PatternKind::Periodic => "Periodic",
            PatternKind::Megarena => "Megarena",
            PatternKind::Stamp => "Stamp",
            PatternKind::QrLike => "QR-like",
        }
    }

    /// One-line description of what the generator produces.
    pub fn blurb(self) -> &'static str {
        match self {
            PatternKind::Periodic => {
                "Sinusoidal stripe carrier of fixed spatial period. The relative-pose workhorse: \
                 its fundamental peak sits at the pattern orientation in the frequency domain."
            }
            PatternKind::Megarena => {
                "Absolute dot grid. Three periods per bit, the central one gated by a maximal LFSR, \
                 with one corner of each 3×3 cell dropped to break the π/2 rotation ambiguity."
            }
            PatternKind::Stamp => {
                "Stamp tile layout. The interface is fixed upstream but the rasterizer is a stub, \
                 so it renders a blank field."
            }
            PatternKind::QrLike => {
                "Module grid with finder patterns. The interface is fixed upstream but the cell \
                 encoding is a stub, so it renders a blank field."
            }
        }
    }

    /// Whether `vernier-patterns` still returns a blank image for this kind.
    /// The UI says so out loud rather than showing an unexplained black square.
    pub fn is_stub(self) -> bool {
        matches!(self, PatternKind::Stamp | PatternKind::QrLike)
    }

    /// Upstream module implementing this kind, shown next to the stub warning.
    pub fn source_path(self) -> &'static str {
        match self {
            PatternKind::Periodic => "vernier-patterns/src/periodic.rs",
            PatternKind::Megarena => "vernier-patterns/src/megarena.rs",
            PatternKind::Stamp => "vernier-patterns/src/stamp.rs",
            PatternKind::QrLike => "vernier-patterns/src/qrcode.rs",
        }
    }
}

/// Every knob the generators expose, plus the display-only ones.
#[derive(Clone, PartialEq, Debug)]
pub struct PatternSettings {
    /// Which generator to run.
    pub kind: PatternKind,

    /// Output image width in pixels.
    pub width: usize,
    /// Output image height in pixels.
    pub height: usize,

    /// Pattern X translation in pixels.
    pub pose_x: Real,
    /// Pattern Y translation in pixels.
    pub pose_y: Real,
    /// Pattern orientation. Degrees here, radians at the generator boundary —
    /// a slider in radians is unreadable.
    pub theta_deg: Real,

    /// Spatial period in pixels. Shared by [`PatternKind::Periodic`] and
    /// [`PatternKind::Megarena`], which both build their carrier from it.
    pub period_px: Real,

    /// LFSR order — bits per unique window, so it sets the absolute range.
    pub order: u32,
    /// LFSR index placed at triple 0. Upstream suggests `order` to keep the
    /// decode window off the sequence boundary.
    pub lfsr_offset: i64,

    /// Stamp tile side in pixels.
    pub tile_px: usize,

    /// Modules per axis in the QR-like grid.
    pub modules: usize,
    /// Pixels per module.
    pub module_px: usize,

    /// Display only: swap black and white before painting.
    pub invert: bool,
}

impl Default for PatternSettings {
    fn default() -> Self {
        Self {
            kind: PatternKind::Megarena,
            width: 512,
            height: 512,
            pose_x: 0.0,
            pose_y: 0.0,
            theta_deg: 0.0,
            period_px: 20.0,
            order: 8,
            lfsr_offset: 8,
            tile_px: 32,
            modules: 21,
            module_px: 8,
            invert: false,
        }
    }
}

impl PatternSettings {
    /// The pose the generators take, with the UI's degrees converted to radians.
    pub fn pose(&self) -> PatternPose {
        PatternPose::new(self.pose_x, self.pose_y, self.theta_deg.to_radians())
    }

    /// Runs the selected generator. The error case is a parameter combination
    /// upstream rejects — currently only an unsupported LFSR order.
    pub fn render(&self) -> Result<GrayImage, String> {
        let pose = self.pose();
        let (w, h) = (self.width, self.height);

        match self.kind {
            PatternKind::Periodic => Ok(Periodic::new(self.period_px).render(w, h, &pose)),
            PatternKind::Megarena => Megarena::new(self.period_px, self.order)
                .map(|pattern| pattern.with_lfsr_offset(self.lfsr_offset).render(w, h, &pose))
                .ok_or_else(|| {
                    format!(
                        "LFSR order {} is unsupported — vernier-patterns generates maximal \
                         sequences of order {}..={} only.",
                        self.order,
                        ORDER_RANGE.start(),
                        ORDER_RANGE.end()
                    )
                }),
            PatternKind::Stamp => Ok(Stamp::new(self.tile_px).render(w, h, &pose)),
            PatternKind::QrLike => {
                Ok(QrLike::new(self.modules, self.module_px).render(w, h, &pose))
            }
        }
    }

    /// Quantities derived from the current parameters that are worth reading off
    /// directly — what the code length and absolute range actually work out to.
    pub fn derived(&self) -> Vec<(String, String)> {
        match self.kind {
            PatternKind::Periodic => vec![
                ("Periods across width".into(), format!("{:.2}", self.width as Real / self.period_px)),
                ("Periods across height".into(), format!("{:.2}", self.height as Real / self.period_px)),
            ],
            PatternKind::Megarena => {
                let code_len = (1u64 << self.order) - 1;
                // Three periods per bit: the whole sequence spans this many pixels
                // before the absolute code repeats.
                let range_px = code_len as Real * 3.0 * self.period_px;
                vec![
                    ("Code length".into(), format!("{code_len} bits (2^{} − 1)", self.order)),
                    ("Absolute range".into(), format!("{range_px:.0} px")),
                    ("Bits across width".into(), format!("{:.2}", self.width as Real / (3.0 * self.period_px))),
                ]
            }
            PatternKind::Stamp => vec![
                ("Tiles across width".into(), format!("{:.2}", self.width as Real / self.tile_px.max(1) as Real)),
            ],
            PatternKind::QrLike => vec![
                ("Grid extent".into(), format!("{} px", self.modules * self.module_px)),
            ],
        }
    }

    /// The `vernier-patterns` call this UI state corresponds to, so a parameter
    /// set found here can be carried straight into Rust code or a test.
    pub fn equivalent_rust(&self) -> String {
        let pose = format!(
            "PatternPose::new({:.3}, {:.3}, {:.5}_f64.to_radians())",
            self.pose_x, self.pose_y, self.theta_deg
        );
        let constructor = match self.kind {
            PatternKind::Periodic => format!("Periodic::new({:.3})", self.period_px),
            PatternKind::Megarena => format!(
                "Megarena::new({:.3}, {})\n    .unwrap()\n    .with_lfsr_offset({})",
                self.period_px, self.order, self.lfsr_offset
            ),
            PatternKind::Stamp => format!("Stamp::new({})", self.tile_px),
            PatternKind::QrLike => format!("QrLike::new({}, {})", self.modules, self.module_px),
        };
        format!(
            "let pose = {pose};\nlet image = {constructor}\n    .render({}, {}, &pose);",
            self.width, self.height
        )
    }

    /// Filename for the PNG download, tagged with the parameters that produced it.
    pub fn file_stem(&self) -> String {
        let base = match self.kind {
            PatternKind::Periodic => format!("periodic_p{:.0}", self.period_px),
            PatternKind::Megarena => format!("megarena_p{:.0}_n{}", self.period_px, self.order),
            PatternKind::Stamp => format!("stamp_t{}", self.tile_px),
            PatternKind::QrLike => format!("qrlike_{}x{}", self.modules, self.module_px),
        };
        format!("{base}_{}x{}", self.width, self.height)
    }
}

/// Expands a `[0, 1]` intensity field into the RGBA bytes a canvas `ImageData`
/// wants. Intensities are clamped rather than rescaled: the generators promise
/// `[0, 1]`, and rescaling would hide a generator that broke that promise.
pub fn to_rgba(image: &GrayImage, invert: bool) -> Vec<u8> {
    let mut rgba = vec![0u8; image.width() * image.height() * 4];
    for (pixel, &intensity) in rgba.chunks_exact_mut(4).zip(image.as_slice()) {
        let level = intensity.clamp(0.0, 1.0);
        let level = if invert { 1.0 - level } else { level };
        let byte = (level * 255.0).round() as u8;
        pixel[0] = byte;
        pixel[1] = byte;
        pixel[2] = byte;
        pixel[3] = 255;
    }
    rgba
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_kind_renders_at_the_requested_size() {
        for kind in PatternKind::ALL {
            let settings = PatternSettings {
                kind,
                width: 48,
                height: 32,
                ..Default::default()
            };
            let image = settings.render().expect("default parameters should render");
            assert_eq!(image.width(), 48, "{kind:?} width");
            assert_eq!(image.height(), 32, "{kind:?} height");
        }
    }

    #[test]
    fn unsupported_lfsr_order_reports_the_supported_range() {
        let settings = PatternSettings {
            kind: PatternKind::Megarena,
            order: 3,
            ..Default::default()
        };
        let message = settings.render().expect_err("order 3 is below the supported range");
        assert!(message.contains("4..=12"), "unhelpful message: {message}");
    }

    #[test]
    fn whole_supported_order_range_builds() {
        for order in ORDER_RANGE {
            let settings = PatternSettings {
                kind: PatternKind::Megarena,
                order,
                width: 32,
                height: 32,
                ..Default::default()
            };
            assert!(settings.render().is_ok(), "order {order} should build");
        }
    }

    #[test]
    fn orientation_reaches_the_generator_in_radians() {
        let settings = PatternSettings {
            theta_deg: 90.0,
            ..Default::default()
        };
        let expected = std::f64::consts::FRAC_PI_2;
        assert!((settings.pose().theta - expected).abs() < 1e-12);
    }

    #[test]
    fn rgba_is_opaque_grey_and_inverts() {
        let image = GrayImage::from_vec(2, 1, vec![0.0, 1.0]).unwrap();

        let plain = to_rgba(&image, false);
        assert_eq!(plain, vec![0, 0, 0, 255, 255, 255, 255, 255]);

        let inverted = to_rgba(&image, true);
        assert_eq!(inverted, vec![255, 255, 255, 255, 0, 0, 0, 255]);
    }

    #[test]
    fn rgba_clamps_rather_than_rescaling_out_of_range_intensities() {
        let image = GrayImage::from_vec(2, 1, vec![-0.5, 1.5]).unwrap();
        let rgba = to_rgba(&image, false);
        assert_eq!(rgba[0], 0);
        assert_eq!(rgba[4], 255);
    }

    #[test]
    fn megarena_readouts_match_the_encoding() {
        let settings = PatternSettings {
            kind: PatternKind::Megarena,
            order: 8,
            period_px: 20.0,
            ..Default::default()
        };
        let derived = settings.derived();
        let code_length = &derived.iter().find(|(name, _)| name == "Code length").unwrap().1;
        assert!(code_length.starts_with("255 bits"), "got {code_length}");

        // 255 bits × 3 periods per bit × 20 px.
        let range = &derived.iter().find(|(name, _)| name == "Absolute range").unwrap().1;
        assert_eq!(range, "15300 px");
    }

    #[test]
    fn snippet_carries_the_parameters_that_produced_the_render() {
        let settings = PatternSettings {
            kind: PatternKind::Megarena,
            order: 10,
            period_px: 12.5,
            width: 640,
            height: 480,
            ..Default::default()
        };
        let snippet = settings.equivalent_rust();
        assert!(snippet.contains("Megarena::new(12.500, 10)"), "got:\n{snippet}");
        assert!(snippet.contains(".render(640, 480, &pose)"), "got:\n{snippet}");
    }
}
