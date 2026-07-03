//! Placeholder detectors for pattern families whose detection algorithms are
//! not yet ported.
//!
//! The C++ `Detector::newInstance` recognizes `"StampPattern"` and
//! `"HPCodePattern"`; these stubs keep the factory's class-name surface in
//! parity while making the missing implementation explicit — construction and
//! configuration succeed, but [`compute`](PatternDetector::compute) returns a
//! "not implemented" error.
//!
//! Porting them for real needs computer-vision primitives the workspace does
//! not have yet: `StampPattern` requires quadrilateral detection
//! (C++ `SquareDetector`), `HPCodePattern` requires Canny-based fiducial
//! detection (C++ `QRFiducialDetector`).

use vernier_core::{GrayImage, Pose, Result, VernierError};

use crate::{Metadata, PatternDetector, SpectralConfig};

/// A detector whose family is recognized by the factory but whose detection is
/// not implemented. `compute` fails with a descriptive error.
pub struct UnimplementedDetector {
    classname: &'static str,
    config: SpectralConfig,
    meta: Metadata,
}

impl UnimplementedDetector {
    /// Creates a stub for the given class name.
    pub fn new(classname: &'static str) -> Self {
        Self {
            classname,
            config: SpectralConfig::default(),
            meta: Metadata::default(),
        }
    }
}

impl PatternDetector for UnimplementedDetector {
    fn compute(&mut self, _image: &GrayImage) -> Result<()> {
        Err(VernierError::Message(format!(
            "{} detection is not implemented in vernier-rs yet",
            self.classname
        )))
    }

    fn pattern_found(&self, _id: i32) -> bool {
        false
    }
    fn pattern_count(&self) -> i32 {
        0
    }
    fn get_2d_pose(&self, _id: i32) -> Pose {
        Pose::ORIGIN
    }
    fn get_3d_pose(&self, _id: i32) -> Pose {
        Pose::ORIGIN
    }
    fn get_all_3d_poses(&self, _id: i32) -> Vec<Pose> {
        Vec::new()
    }

    fn classname(&self) -> &str {
        self.classname
    }

    fn description(&self) -> &str {
        &self.meta.description
    }
    fn set_description(&mut self, value: String) {
        self.meta.description = value;
    }
    fn author(&self) -> &str {
        &self.meta.author
    }
    fn set_author(&mut self, value: String) {
        self.meta.author = value;
    }
    fn date(&self) -> &str {
        &self.meta.date
    }
    fn set_date(&mut self, value: String) {
        self.meta.date = value;
    }
    fn unit(&self) -> &str {
        &self.meta.unit
    }
    fn set_unit(&mut self, value: String) {
        self.meta.unit = value;
    }

    fn get_double(&self, attribute: &str) -> Option<f64> {
        self.config.get_double(attribute)
    }
    fn set_double(&mut self, attribute: &str, value: f64) -> bool {
        self.config.set_double(attribute, value)
    }
    fn get_int(&self, attribute: &str) -> Option<i64> {
        match attribute {
            "minFrequency" => Some(self.config.min_frequency as i64),
            "maxFrequency" => Some(self.config.max_frequency as i64),
            _ => None,
        }
    }
    fn set_int(&mut self, attribute: &str, value: i64) -> bool {
        match attribute {
            "minFrequency" => self.config.min_frequency = value.max(0) as usize,
            "maxFrequency" => self.config.max_frequency = value.max(0) as usize,
            _ => return false,
        }
        true
    }
    fn get_bool(&self, _attribute: &str) -> Option<bool> {
        None
    }
    fn set_bool(&mut self, _attribute: &str, _value: bool) -> bool {
        false
    }

    fn describe(&self) -> String {
        format!("{} [ unimplemented ]", self.classname)
    }
}
