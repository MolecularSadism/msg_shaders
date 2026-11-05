//! Injectable colour palette for pixel planets.
//!
//! Upstream `PixelPlanets` ships fixed colour schemes; this fork instead takes an
//! ordered colour ramp from the caller and derives every layer's `vec4[]`
//! palette uniform from it, so a project can feed in its own art-direction
//! colours.

use bevy::prelude::*;

/// An ordered light → dark colour ramp used to colour a planet's layers.
///
/// The crate samples this ramp to fill the discrete `colors[]` / `dark_colors[]`
/// uniform arrays each shader expects.
#[derive(Clone, Debug)]
pub struct PlanetPalette {
    /// Linear-space RGBA ramp, ordered brightest → darkest. Always non-empty.
    ramp: Vec<Vec4>,
}

impl PlanetPalette {
    /// Builds a palette from an ordered light → dark sequence of colours.
    ///
    /// An empty input falls back to a neutral grey ramp so the planet always
    /// renders something.
    pub fn new(colors: impl IntoIterator<Item = Color>) -> Self {
        let mut ramp: Vec<Vec4> = colors.into_iter().map(color_to_linear_vec4).collect();
        if ramp.is_empty() {
            ramp = vec![
                Vec4::new(0.60, 0.60, 0.62, 1.0),
                Vec4::new(0.30, 0.30, 0.34, 1.0),
                Vec4::new(0.10, 0.10, 0.14, 1.0),
            ];
        }
        Self { ramp }
    }

    /// Samples the ramp at `t` in `[0, 1]` (0 = brightest, 1 = darkest).
    fn sample(&self, t: f32) -> Vec4 {
        let len = self.ramp.len();
        if len == 1 {
            return self.ramp[0];
        }
        let pos = t.clamp(0.0, 1.0) * (len - 1) as f32;
        // `pos` is non-negative, so the cast floors it.
        let lo = pos as usize;
        let hi = (lo + 1).min(len - 1);
        let frac = pos - lo as f32;
        self.ramp[lo].lerp(self.ramp[hi], frac)
    }

    /// Evenly samples `N` colours across the ramp.
    fn even<const N: usize>(&self) -> [Vec4; N] {
        std::array::from_fn(|i| {
            let t = if N == 1 {
                0.0
            } else {
                i as f32 / (N - 1) as f32
            };
            self.sample(t)
        })
    }

    /// Three-stop body palette (light → dark).
    #[must_use]
    pub fn colors3(&self) -> [Vec4; 3] {
        self.even()
    }

    /// Four-stop body palette (light → dark).
    #[must_use]
    pub fn colors4(&self) -> [Vec4; 4] {
        self.even()
    }

    /// Darkened three-stop palette for shadow-side / ring bands.
    #[must_use]
    pub fn dark3(&self) -> [Vec4; 3] {
        darken(self.colors3(), 0.45)
    }

    /// Scales every colour toward black by `factor` — `1.0` leaves the palette unchanged, lower
    /// values darken it, `0.0` is black. This caps the brightest stop and shrinks the absolute
    /// contrast, so the planet reads as dimmer and more distant.
    #[must_use]
    pub fn dimmed(mut self, factor: f32) -> Self {
        let f = factor.max(0.0);
        for c in &mut self.ramp {
            c.x *= f;
            c.y *= f;
            c.z *= f;
        }
        self
    }

    /// Light, lightly tinted cloud palette derived from the body ramp.
    #[must_use]
    pub fn cloud4(&self) -> [Vec4; 4] {
        let white = Vec4::new(1.0, 1.0, 1.0, 1.0);
        [
            white.lerp(self.sample(0.0), 0.15),
            self.sample(0.1),
            self.sample(0.5),
            self.sample(0.85),
        ]
    }
}

fn color_to_linear_vec4(color: Color) -> Vec4 {
    let l = color.to_linear();
    Vec4::new(l.red, l.green, l.blue, l.alpha)
}

fn darken<const N: usize>(mut colors: [Vec4; N], factor: f32) -> [Vec4; N] {
    for c in &mut colors {
        c.x *= factor;
        c.y *= factor;
        c.z *= factor;
    }
    colors
}

#[cfg(test)]
mod tests {
    use super::*;

    fn luma(v: Vec4) -> f32 {
        v.x + v.y + v.z
    }

    #[test]
    fn empty_palette_falls_back_to_opaque_ramp() {
        let palette = PlanetPalette::new([]);
        let colors = palette.colors3();
        assert!(colors.iter().all(|v| (v.w - 1.0).abs() <= f32::EPSILON));
    }

    #[test]
    fn ramp_samples_light_to_dark() {
        let palette = PlanetPalette::new([Color::WHITE, Color::BLACK]);
        let colors = palette.colors3();
        assert!(luma(colors[0]) > luma(colors[1]));
        assert!(luma(colors[1]) > luma(colors[2]));
    }

    #[test]
    fn dark_ramp_is_not_brighter_than_body() {
        let palette = PlanetPalette::new([Color::srgb(0.8, 0.8, 0.8), Color::srgb(0.2, 0.2, 0.2)]);
        let body = palette.colors3();
        let dark = palette.dark3();
        for i in 0..3 {
            assert!(luma(dark[i]) <= luma(body[i]) + 1e-6);
        }
    }

    #[test]
    fn dimmed_darkens_every_stop() {
        let palette = PlanetPalette::new([Color::srgb(0.8, 0.6, 0.4), Color::srgb(0.2, 0.1, 0.05)]);
        let bright = palette.clone().colors3();
        let dim = palette.dimmed(0.5).colors3();
        for i in 0..3 {
            assert!(luma(dim[i]) < luma(bright[i]));
        }
    }

    #[test]
    fn single_colour_ramp_is_uniform() {
        let palette = PlanetPalette::new([Color::srgb(0.3, 0.5, 0.7)]);
        let colors = palette.colors4();
        assert!(
            colors
                .iter()
                .all(|v| v.abs_diff_eq(colors[0], f32::EPSILON))
        );
    }
}
