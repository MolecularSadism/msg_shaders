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
    /// Roll about the view axis in radians, applied inside the shader.
    pub rotation: f32,
    /// Pole tilt toward the viewer in radians; `0` keeps the pole in the screen plane.
    pub tilt: f32,
    /// Sine latitude past which the polar caps cover the surface; `None` leaves the poles bare.
    pub pole_cap: Option<f32>,
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

impl CommonParams {
    /// `pole_cap` as the shader reads it: a sine latitude above `1` covers nothing.
    fn pole_cap_uniform(&self) -> f32 {
        self.pole_cap.map_or(NO_POLE_CAP, |cap| cap.clamp(0.0, 1.0))
    }
}

/// Sine latitude no surface point reaches, so the shader draws no polar cap.
const NO_POLE_CAP: f32 = 2.0;

/// The view a planet layer is drawn with: everything that changes after spawn.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LayerView {
    /// Spin about the pole, in radians.
    pub spin: f32,
    /// Pole tilt toward the viewer, in radians.
    pub tilt: f32,
    /// Highlight location in `[0, 1]` UV space.
    pub light_origin: Vec2,
}

impl LayerView {
    /// The tilt as the shader reads it: `(cos, sin)`.
    #[must_use]
    pub fn tilt_cs(&self) -> Vec2 {
        Vec2::from_angle(self.tilt)
    }
}

/// A pixel-planet layer material whose view changes after spawn.
///
/// The `time` uniform is the spin about the pole in radians, `tilt_cs` the pole tilt, and
/// `light_origin` the highlight. Every layer material exposes them through this trait so a single
/// generic system can drive all of them.
pub trait PlanetMaterial {
    /// Writes `view` into the layer's uniforms.
    fn set_view(&mut self, view: LayerView);
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
            fn set_view(&mut self, view: LayerView) {
                self.u.time = view.spin;
                self.u.tilt_cs = view.tilt_cs();
                self.u.light_origin = view.light_origin;
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
    pub tilt_cs: Vec2,
    pub pixels: f32,
    pub rotation: f32,
    pub size: f32,
    pub seed: f32,
    pub time: f32,
    pub light_border_1: f32,
    pub light_border_2: f32,
    pub dither_size: f32,
    pub edge_dither: f32,
    pub pole_cap: f32,
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
                tilt_cs: Vec2::from_angle(p.tilt),
                pixels: p.pixels,
                rotation: p.rotation,
                size: p.size,
                seed: p.seed,
                time: 0.0,
                light_border_1: 0.4,
                light_border_2: 0.6,
                dither_size: 2.0,
                edge_dither: p.dither.unwrap_or(0.0),
                pole_cap: p.pole_cap_uniform(),
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
    pub tilt_cs: Vec2,
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
                tilt_cs: Vec2::from_angle(p.tilt),
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
    pub tilt_cs: Vec2,
    pub pixels: f32,
    pub rotation: f32,
    pub size: f32,
    pub seed: f32,
    pub time: f32,
    pub light_border_1: f32,
    pub light_border_2: f32,
    pub land_cutoff: f32,
    pub edge_dither: f32,
    pub pole_cap: f32,
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
                tilt_cs: Vec2::from_angle(p.tilt),
                pixels: p.pixels,
                rotation: p.rotation,
                size: p.size,
                seed: p.seed,
                time: 0.0,
                light_border_1: 0.4,
                light_border_2: 0.5,
                land_cutoff: 0.5,
                edge_dither: p.dither.unwrap_or(0.0),
                pole_cap: p.pole_cap_uniform(),
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
    pub tilt_cs: Vec2,
    pub pixels: f32,
    pub rotation: f32,
    pub size: f32,
    pub seed: f32,
    pub time: f32,
    pub light_border_1: f32,
    pub light_border_2: f32,
    pub cloud_cover: f32,
    pub stretch: f32,
    pub edge_dither: f32,
    pub pole_cap: f32,
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
                tilt_cs: Vec2::from_angle(p.tilt),
                pixels: p.pixels,
                rotation: p.rotation,
                size: p.size,
                seed: p.seed,
                time: 0.0,
                light_border_1: 0.52,
                light_border_2: 0.62,
                cloud_cover,
                stretch: 2.0,
                edge_dither: p.dither.unwrap_or(0.0),
                pole_cap: p.pole_cap_uniform(),
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
    pub tilt_cs: Vec2,
    pub pixels: f32,
    pub rotation: f32,
    pub size: f32,
    pub seed: f32,
    pub time: f32,
    pub light_border_1: f32,
    pub light_border_2: f32,
    pub bands: f32,
    pub edge_dither: f32,
    pub pole_cap: f32,
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
                tilt_cs: Vec2::from_angle(p.tilt),
                pixels: p.pixels,
                rotation: p.rotation,
                size: p.size,
                seed: p.seed,
                time: 0.0,
                light_border_1: 0.52,
                light_border_2: 0.62,
                bands,
                edge_dither: p.dither.unwrap_or(0.0),
                pole_cap: p.pole_cap_uniform(),
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
    pub tilt_cs: Vec2,
    pub pixels: f32,
    pub rotation: f32,
    pub size: f32,
    pub seed: f32,
    pub time: f32,
    pub ring_width: f32,
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
                tilt_cs: Vec2::from_angle(p.tilt),
                pixels: p.pixels,
                rotation: p.rotation,
                size: p.size,
                seed: p.seed,
                time: 0.0,
                ring_width: 0.1,
                scale_rel_to_planet,
                edge_dither: p.dither.unwrap_or(0.0),
                octaves: p.octaves,
                n_colors: 3,
            },
        }
    }
}
impl_material2d!(RingMaterial, super::RING_SHADER);

#[cfg(test)]
mod tests {
    use bevy::render::render_resource::encase::UniformBuffer;

    use super::*;

    /// Gives each uniform field a value no other field holds, so a swap shows.
    trait Distinct {
        fn distinct(next: &mut f32) -> Self;
        fn bytes(&self) -> Vec<u8>;
    }

    impl Distinct for f32 {
        fn distinct(next: &mut f32) -> Self {
            *next += 1.0;
            *next
        }
        fn bytes(&self) -> Vec<u8> {
            self.to_le_bytes().to_vec()
        }
    }

    impl Distinct for i32 {
        fn distinct(next: &mut f32) -> Self {
            f32::distinct(next) as i32
        }
        fn bytes(&self) -> Vec<u8> {
            self.to_le_bytes().to_vec()
        }
    }

    impl Distinct for u32 {
        fn distinct(next: &mut f32) -> Self {
            f32::distinct(next) as u32
        }
        fn bytes(&self) -> Vec<u8> {
            self.to_le_bytes().to_vec()
        }
    }

    impl Distinct for Vec2 {
        fn distinct(next: &mut f32) -> Self {
            Vec2::new(f32::distinct(next), f32::distinct(next))
        }
        fn bytes(&self) -> Vec<u8> {
            self.to_array().iter().flat_map(|v| v.bytes()).collect()
        }
    }

    impl<const N: usize> Distinct for [Vec4; N] {
        fn distinct(next: &mut f32) -> Self {
            std::array::from_fn(|_| {
                Vec4::new(
                    f32::distinct(next),
                    f32::distinct(next),
                    f32::distinct(next),
                    f32::distinct(next),
                )
            })
        }
        fn bytes(&self) -> Vec<u8> {
            self.iter()
                .flat_map(|v| v.to_array())
                .flat_map(|v| v.to_le_bytes())
                .collect()
        }
    }

    /// The members of the shader's `struct U`, by name and byte offset, and its size.
    fn shader_layout(shader: &str) -> (Vec<(String, usize)>, usize) {
        let mut src = String::from(
            "struct VertexOutput { @builtin(position) clip_position: vec4<f32>, \
             @location(0) world_position: vec4<f32>, @location(1) uv: vec2<f32> };\n",
        );
        for source in [
            include_str!("../shaders/pixel_planet_functions.wgsl"),
            shader,
        ] {
            for line in source.lines() {
                if !line.trim_start().starts_with('#') {
                    src.push_str(line);
                    src.push('\n');
                }
            }
        }
        let module = naga::front::wgsl::parse_str(&src)
            .unwrap_or_else(|e| panic!("WGSL parse error\n{}", e.emit_to_string(&src)));
        let (_, ty) = module
            .types
            .iter()
            .find(|(_, ty)| ty.name.as_deref() == Some("U"))
            .expect("the shader declares `struct U`");
        let naga::TypeInner::Struct { members, span } = &ty.inner else {
            panic!("`U` is a struct");
        };
        let members = members
            .iter()
            .map(|member| {
                (
                    member.name.clone().expect("named member"),
                    member.offset as usize,
                )
            })
            .collect();
        (members, *span as usize)
    }

    /// Fills every field of `$uniform` with a distinct value, encodes it as the GPU sees it and
    /// checks each field lands at the offset the shader's `struct U` declares, in the same order.
    macro_rules! assert_layout {
        ($shader:literal, $uniform:expr, [$($field:ident),* $(,)?]) => {{
            let mut uniform = $uniform;
            let mut next = 0.0;
            $(uniform.$field = Distinct::distinct(&mut next);)*
            let mut buffer = UniformBuffer::new(Vec::<u8>::new());
            buffer.write(&uniform).expect("the uniform encodes");
            let bytes = buffer.into_inner();

            let (members, size) = shader_layout(include_str!($shader));
            let names: Vec<&str> = members.iter().map(|(name, _)| name.as_str()).collect();
            assert_eq!(names, [$(stringify!($field)),*], "{} member order", $shader);
            assert!(bytes.len() >= size, "{}: {} bytes for a {size}-byte struct", $shader, bytes.len());
            $(
                let (_, offset) = members
                    .iter()
                    .find(|(name, _)| name == stringify!($field))
                    .expect("member");
                let expected = Distinct::bytes(&uniform.$field);
                assert_eq!(
                    &bytes[*offset..*offset + expected.len()],
                    expected.as_slice(),
                    "{}: `{}` at offset {offset}",
                    $shader,
                    stringify!($field),
                );
            )*
        }};
    }

    fn params() -> CommonParams {
        CommonParams {
            light_origin: Vec2::splat(0.3),
            pixels: 64.0,
            rotation: 0.0,
            tilt: 0.4,
            pole_cap: Some(0.8),
            size: 6.0,
            seed: 2.0,
            octaves: 3,
            dither: Some(1.0),
        }
    }

    fn palette() -> PlanetPalette {
        PlanetPalette::new([Color::WHITE, Color::BLACK])
    }

    #[test]
    fn no_atmosphere_uniform_matches_its_shader() {
        assert_layout!(
            "../shaders/pixel_planet_no_atmosphere.wgsl",
            NoAtmosphereMaterial::new(params(), &palette()).u,
            [
                colors,
                light_origin,
                tilt_cs,
                pixels,
                rotation,
                size,
                seed,
                time,
                light_border_1,
                light_border_2,
                dither_size,
                edge_dither,
                pole_cap,
                octaves,
                should_dither,
            ]
        );
    }

    #[test]
    fn under_uniform_matches_its_shader() {
        assert_layout!(
            "../shaders/pixel_planet_under.wgsl",
            PlanetUnderMaterial::new(params(), &palette()).u,
            [
                colors,
                light_origin,
                tilt_cs,
                pixels,
                rotation,
                size,
                seed,
                time,
                light_border_1,
                light_border_2,
                dither_size,
                edge_dither,
                octaves,
                should_dither,
            ]
        );
    }

    #[test]
    fn landmass_uniform_matches_its_shader() {
        assert_layout!(
            "../shaders/pixel_planet_landmass.wgsl",
            PlanetLandmassMaterial::new(params(), &palette()).u,
            [
                colors,
                light_origin,
                tilt_cs,
                pixels,
                rotation,
                size,
                seed,
                time,
                light_border_1,
                light_border_2,
                land_cutoff,
                edge_dither,
                pole_cap,
                octaves,
            ]
        );
    }

    #[test]
    fn clouds_uniform_matches_its_shader() {
        assert_layout!(
            "../shaders/pixel_planet_clouds.wgsl",
            CloudsMaterial::new(params(), &palette(), 0.5).u,
            [
                colors,
                light_origin,
                tilt_cs,
                pixels,
                rotation,
                size,
                seed,
                time,
                light_border_1,
                light_border_2,
                cloud_cover,
                stretch,
                edge_dither,
                pole_cap,
                octaves,
            ]
        );
    }

    #[test]
    fn gas_layers_uniform_matches_its_shader() {
        assert_layout!(
            "../shaders/pixel_planet_gas_layers.wgsl",
            GasLayersMaterial::new(params(), &palette(), 1.0).u,
            [
                colors,
                dark_colors,
                light_origin,
                tilt_cs,
                pixels,
                rotation,
                size,
                seed,
                time,
                light_border_1,
                light_border_2,
                bands,
                edge_dither,
                pole_cap,
                octaves,
                n_colors,
                should_dither,
            ]
        );
    }

    #[test]
    fn ring_uniform_matches_its_shader() {
        assert_layout!(
            "../shaders/pixel_planet_ring.wgsl",
            RingMaterial::new(params(), &palette(), 4.0).u,
            [
                colors,
                dark_colors,
                light_origin,
                tilt_cs,
                pixels,
                rotation,
                size,
                seed,
                time,
                ring_width,
                scale_rel_to_planet,
                edge_dither,
                octaves,
                n_colors,
            ]
        );
    }

    #[test]
    fn a_view_reaches_every_view_uniform() {
        let view = LayerView {
            spin: 1.25,
            tilt: 0.5,
            light_origin: Vec2::new(0.2, 0.7),
        };
        let mut material = GasLayersMaterial::new(params(), &palette(), 1.0);
        material.set_view(view);
        assert_eq!(material.u.time, view.spin);
        assert_eq!(material.u.tilt_cs, Vec2::new(0.5_f32.cos(), 0.5_f32.sin()));
        assert_eq!(material.u.light_origin, view.light_origin);
    }

    #[test]
    fn a_missing_pole_cap_lies_past_every_latitude() {
        let bare = CommonParams {
            pole_cap: None,
            ..params()
        };
        assert!(NoAtmosphereMaterial::new(bare, &palette()).u.pole_cap > 1.0);
        assert_eq!(
            NoAtmosphereMaterial::new(params(), &palette()).u.pole_cap,
            0.8
        );
    }
}
