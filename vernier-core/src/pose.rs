use crate::scalar::Real;

#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub struct Pose {
    pub x: Real,
    pub y: Real,
    pub theta: Real,
}

impl Pose {
    pub const ORIGIN: Self = Self {
        x: 0.0,
        y: 0.0,
        theta: 0.0,
    };

    #[inline]
    pub const fn new(x: Real, y: Real, theta: Real) -> Self {
        Self { x, y, theta }
    }

    #[inline]
    pub fn translation_distance(&self, other: &Pose) -> Real {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        (dx * dx + dy * dy).sqrt()
    }

    #[inline]
    pub fn angle_difference(&self, other: &Pose) -> Real {
        use crate::scalar::consts::TAU;
        let mut d = self.theta - other.theta;
        while d > crate::scalar::consts::PI {
            d -= TAU;
        }
        while d <= -crate::scalar::consts::PI {
            d += TAU;
        }
        d
    }
}
