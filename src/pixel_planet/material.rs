//! `Material2d` definitions for each pixel-planet layer.
//!
//! Each material's uniform struct mirrors the `struct U { .. }` field order of
//! its WGSL shader exactly — `ShaderType` and the WGSL std140 layout must agree.

use bevy::{
    prelude::*,
    render::render_resource::{AsBindGroup, ShaderType},
    shader::ShaderRef,
    sprite_render::{AlphaMode2d, Material2d},
};

use super::palette::PlanetPalette;

/// Parameters shared by every layer of a single planet.
#[derive(Clone, Copy)]
pub struct CommonParams {
    /// Highlight location in `[0, 1]` UV space (bright side of the planet).
    pub light_origin: Vec2,
    /// Pixelation resolution (cells across the planet).
    pub pixels: f32,
    /// Surface tilt in radians, applied inside the shader.
    pub rotation: f32,
    /// Noise frequency.
    pub size: f32,
    /// Noise seed (`1.0..=10.0`).
    pub seed: f32,
    /// Number of fbm octaves.
    pub octaves: i32,
    /// Dithered colour banding and silhouette/ring-rim strength. `None` keeps
    /// hard edges and solid colour bands; `Some(strength)` enables both, with
    /// `strength` widening the stippled edge fade (a pixel-cell rim width for
    /// the planet body).
    pub dither: Option<f32>,
}

/// A pixel-planet layer material whose surface animates over time.
///
/// The `time` uniform offsets the noise sampling (and the ring rotation) inside
/// the shader, so advancing it each frame scrolls the surface as if the planet
/// were spinning. Every layer material exposes it through this trait so a single
/// generic system can drive the rotation for all of them.
pub trait PlanetMaterial {
    /// Mutable access to the shader `time` uniform driving the rotation.
    fn time_mut(&mut self) -> &mut f32;
}

macro_rules! impl_material2d {
    ($mat:ty, $handle:path) => {
        impl Material2d for $mat {
            fn fragment_shader() -> ShaderRef {
                $handle.into()
            }
            fn alpha_mode(&self) -> AlphaMode2d {
                AlphaMode2d::Blend
            }
        }
        impl PlanetMaterial for $mat {
            fn time_mut(&mut self) -> &mut f32 {
                &mut self.u.time
            }
        }
    };
}

// ---------------------------------------------------------------------------
// Barren / airless rocky planet
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, ShaderType)]
pub struct NoAtmosphereUniform {
    pub colors: [Vec4; 3],
    pub light_origin: Vec2,
    pub pixels: f32,
    pub rotation: f32,
    pub size: f32,
    pub seed: f32,
    pub time: f32,
    pub light_border_1: f32,
    pub light_border_2: f32,
    pub dither_size: f32,
    pub edge_dither: f32,
    pub octaves: i32,
    pub should_dither: u32,
}

#[derive(Asset, TypePath, AsBindGroup, Clone)]
pub struct NoAtmosphereMaterial {
    #[uniform(0)]
    pub u: NoAtmosphereUniform,
}

impl NoAtmosphereMaterial {
    pub fn new(p: CommonParams, palette: &PlanetPalette) -> Self {
        Self {
            u: NoAtmosphereUniform {
                colors: palette.colors3(),
                light_origin: p.light_origin,
                pixels: p.pixels,
                rotation: p.rotation,
                size: p.size,
                seed: p.seed,
                time: 0.0,
                light_border_1: 0.4,
                light_border_2: 0.6,
                dither_size: 2.0,
                edge_dither: p.dither.unwrap_or(0.0),
                octaves: p.octaves,
                should_dither: u32::from(p.dither.is_some()),
            },
        }
    }
}
impl_material2d!(NoAtmosphereMaterial, super::NO_ATMOSPHERE_SHADER);

// ---------------------------------------------------------------------------
// Terran: base / ocean
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, ShaderType)]
pub struct PlanetUnderUniform {
    pub colors: [Vec4; 3],
    pub light_origin: Vec2,
    pub pixels: f32,
    pub rotation: f32,
    pub size: f32,
    pub seed: f32,
    pub time: f32,
    pub light_border_1: f32,
    pub light_border_2: f32,
    pub dither_size: f32,
    pub edge_dither: f32,
    pub octaves: i32,
    pub should_dither: u32,
}

#[derive(Asset, TypePath, AsBindGroup, Clone)]
pub struct PlanetUnderMaterial {
    #[uniform(0)]
    pub u: PlanetUnderUniform,
}

impl PlanetUnderMaterial {
    pub fn new(p: CommonParams, palette: &PlanetPalette) -> Self {
        Self {
            u: PlanetUnderUniform {
                colors: palette.colors3(),
                light_origin: p.light_origin,
                pixels: p.pixels,
                rotation: p.rotation,
                size: p.size,
                seed: p.seed,
                time: 0.0,
                light_border_1: 0.4,
                light_border_2: 0.6,
                dither_size: 2.0,
                edge_dither: p.dither.unwrap_or(0.0),
                octaves: p.octaves,
                should_dither: u32::from(p.dither.is_some()),
            },
        }
    }
}
impl_material2d!(PlanetUnderMaterial, super::PLANET_UNDER_SHADER);

// ---------------------------------------------------------------------------
// Terran: continents
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, ShaderType)]
pub struct PlanetLandmassUniform {
    pub colors: [Vec4; 4],
    pub light_origin: Vec2,
    pub pixels: f32,
    pub rotation: f32,
    pub size: f32,
    pub seed: f32,
    pub time: f32,
    pub light_border_1: f32,
    pub light_border_2: f32,
    pub land_cutoff: f32,
    pub edge_dither: f32,
    pub octaves: i32,
}

#[derive(Asset, TypePath, AsBindGroup, Clone)]
pub struct PlanetLandmassMaterial {
    #[uniform(0)]
    pub u: PlanetLandmassUniform,
}

impl PlanetLandmassMaterial {
    pub fn new(p: CommonParams, palette: &PlanetPalette) -> Self {
        Self {
            u: PlanetLandmassUniform {
                colors: palette.colors4(),
                light_origin: p.light_origin,
                pixels: p.pixels,
                rotation: p.rotation,
                size: p.size,
                seed: p.seed,
                time: 0.0,
                light_border_1: 0.4,
                light_border_2: 0.5,
                land_cutoff: 0.5,
                edge_dither: p.dither.unwrap_or(0.0),
                octaves: p.octaves,
            },
        }
    }
}
impl_material2d!(PlanetLandmassMaterial, super::PLANET_LANDMASS_SHADER);

// ---------------------------------------------------------------------------
// Cloud layer
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, ShaderType)]
pub struct CloudsUniform {
    pub colors: [Vec4; 4],
    pub light_origin: Vec2,
    pub pixels: f32,
    pub rotation: f32,
    pub size: f32,
    pub seed: f32,
    pub time: f32,
    pub light_border_1: f32,
    pub light_border_2: f32,
    pub cloud_cover: f32,
    pub stretch: f32,
    pub cloud_curve: f32,
    pub edge_dither: f32,
    pub octaves: i32,
}

#[derive(Asset, TypePath, AsBindGroup, Clone)]
pub struct CloudsMaterial {
    #[uniform(0)]
    pub u: CloudsUniform,
}

impl CloudsMaterial {
    pub fn new(p: CommonParams, palette: &PlanetPalette, cloud_cover: f32) -> Self {
        Self {
            u: CloudsUniform {
                colors: palette.cloud4(),
                light_origin: p.light_origin,
                pixels: p.pixels,
                rotation: p.rotation,
                size: p.size,
                seed: p.seed,
                time: 0.0,
                light_border_1: 0.52,
                light_border_2: 0.62,
                cloud_cover,
                stretch: 2.0,
                cloud_curve: 1.3,
                edge_dither: p.dither.unwrap_or(0.0),
                octaves: p.octaves,
            },
        }
    }
}
impl_material2d!(CloudsMaterial, super::CLOUDS_SHADER);

// ---------------------------------------------------------------------------
// Gas giant body
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, ShaderType)]
pub struct GasLayersUniform {
    pub colors: [Vec4; 3],
    pub dark_colors: [Vec4; 3],
    pub light_origin: Vec2,
    pub pixels: f32,
    pub rotation: f32,
    pub size: f32,
    pub seed: f32,
    pub time: f32,
    pub light_border_1: f32,
    pub light_border_2: f32,
    pub bands: f32,
    pub edge_dither: f32,
    pub octaves: i32,
    pub n_colors: i32,
    pub should_dither: u32,
}

#[derive(Asset, TypePath, AsBindGroup, Clone)]
pub struct GasLayersMaterial {
    #[uniform(0)]
    pub u: GasLayersUniform,
}

impl GasLayersMaterial {
    pub fn new(p: CommonParams, palette: &PlanetPalette, bands: f32) -> Self {
        Self {
            u: GasLayersUniform {
                colors: palette.colors3(),
                dark_colors: palette.dark3(),
                light_origin: p.light_origin,
                pixels: p.pixels,
                rotation: p.rotation,
                size: p.size,
                seed: p.seed,
                time: 0.0,
                light_border_1: 0.52,
                light_border_2: 0.62,
                bands,
                edge_dither: p.dither.unwrap_or(0.0),
                octaves: p.octaves,
                n_colors: 3,
                should_dither: u32::from(p.dither.is_some()),
            },
        }
    }
}
impl_material2d!(GasLayersMaterial, super::GAS_LAYERS_SHADER);

// ---------------------------------------------------------------------------
// Ring
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, ShaderType)]
pub struct RingUniform {
    pub colors: [Vec4; 3],
    pub dark_colors: [Vec4; 3],
    pub light_origin: Vec2,
    pub pixels: f32,
    pub rotation: f32,
    pub size: f32,
    pub seed: f32,
    pub time: f32,
    pub ring_width: f32,
    pub ring_perspective: f32,
    pub scale_rel_to_planet: f32,
    pub edge_dither: f32,
    pub octaves: i32,
    pub n_colors: i32,
}

#[derive(Asset, TypePath, AsBindGroup, Clone)]
pub struct RingMaterial {
    #[uniform(0)]
    pub u: RingUniform,
}

impl RingMaterial {
    pub fn new(p: CommonParams, palette: &PlanetPalette, scale_rel_to_planet: f32) -> Self {
        Self {
            u: RingUniform {
                colors: palette.colors3(),
                dark_colors: palette.dark3(),
                light_origin: p.light_origin,
                pixels: p.pixels,
                rotation: p.rotation,
                size: p.size,
                seed: p.seed,
                time: 0.0,
                ring_width: 0.1,
                ring_perspective: 4.0,
                scale_rel_to_planet,
                edge_dither: p.dither.unwrap_or(0.0),
                octaves: p.octaves,
                n_colors: 3,
            },
        }
    }
}
impl_material2d!(RingMaterial, super::RING_SHADER);
