//! Layered procedural nebula + twinkling starfield background.
//!
//! [`NebulaPlugin`] registers a [`Nebula`] component: attach it to an entity
//! (with a `Transform`) and the plugin builds a quad that composites a stack of
//! nebula [`NebulaLayer`]s over a base color, then adds a stack of
//! [`StarLayer`]s. The output is snapped to the game's art-pixel grid
//! (pixel-perfect from screen-space derivatives) and reduced to a palette with
//! ordered dithering, reusing [`crate::ColorQuantizationPlugin`] and
//! [`crate::PixelationPlugin`].
//!
//! Each nebula layer has its own noise settings, a three-stop color ramp
//! (`c1` edge -> `c2` body -> `c3` core), and a parallax factor; layers
//! composite back-to-front, painting over ([`NebulaBlend::Over`]) or adding
//! ([`NebulaBlend::Additive`]). Star layers likewise carry their own spacing,
//! density, and parallax. Set [`Nebula::scroll`] to the camera's world position
//! each frame; each layer multiplies it by its own parallax factor for depth.
//!
//! ```
//! use bevy::prelude::*;
//! use msg_shaders::{Nebula, NebulaPlugin};
//!
//! fn register(app: &mut App) {
//!     app.add_plugins(NebulaPlugin);
//! }
//!
//! fn setup(mut commands: Commands) {
//!     commands.spawn((Nebula::default(), Transform::default()));
//! }
//! ```

use bevy::{
    asset::{embedded_asset, load_internal_asset, uuid_handle},
    prelude::*,
    render::render_resource::{AsBindGroup, ShaderType},
    shader::{Shader, ShaderRef},
    sprite_render::{AlphaMode2d, Material2d, Material2dPlugin},
};

use crate::quantize::{MAX_PALETTE_COLORS, linear_rgb_to_oklab};
use crate::quantize_material::ColorQuantizeUniforms;

/// Handle for the importable nebula functions shader.
pub const NEBULA_FUNCTIONS_SHADER_HANDLE: Handle<Shader> =
    uuid_handle!("3c9d1e77-2b64-4a05-9f18-6d7c8a1b2e34");

/// Maximum nebula cloud layers uploaded to the shader (fixed uniform array).
/// Extra layers on the [`Nebula`] are dropped by [`Nebula::to_uniforms`].
pub const MAX_NEBULA_LAYERS: usize = 6;

/// Maximum star layers uploaded to the shader (fixed uniform array).
pub const MAX_STAR_LAYERS: usize = 4;

/// How nebula layers combine with what is behind them.
#[derive(Debug, Clone, Copy, Reflect, PartialEq, Eq, Default)]
pub enum NebulaBlend {
    /// Painter's over: dense areas paint the layer's color over the layers
    /// below. Keeps colors distinct — the right choice for limited palettes.
    #[default]
    Over,
    /// Emissive add: layers brighten what is behind them.
    Additive,
}

impl NebulaBlend {
    fn as_u32(self) -> u32 {
        match self {
            NebulaBlend::Over => 0,
            NebulaBlend::Additive => 1,
        }
    }
}

/// One nebula cloud layer: a noise field mapped through a three-stop color ramp
/// and composited with its own parallax and decorrelation.
#[derive(Debug, Clone, Reflect)]
pub struct NebulaLayer {
    /// Ramp colors, linear RGBA: `[edge, body, core]`, keyed by cloud density.
    pub colors: [[f32; 4]; 3],
    /// Feature frequency (cycles per world unit). Smaller = larger clouds.
    pub scale: f32,
    /// Cloud fbm octaves (clamped to `1..=8`).
    pub octaves: u32,
    /// Domain-warp fbm octaves (clamped to `1..=8`).
    pub warp_octaves: u32,
    /// Domain-warp displacement (wispiness).
    pub warp_strength: f32,
    /// Lower edge of the cloud smoothstep band.
    pub cloud_low: f32,
    /// Upper edge of the cloud smoothstep band.
    pub cloud_high: f32,
    /// Coverage / brightness multiplier for this layer.
    pub intensity: f32,
    /// Fraction of [`Nebula::scroll`] this layer drifts by (parallax depth).
    pub parallax: f32,
    /// Sampling offset in world units, decorrelating layers that share settings.
    pub offset: Vec2,
    /// Sampling rotation in radians, likewise for decorrelation.
    pub rotation: f32,
}

impl Default for NebulaLayer {
    fn default() -> Self {
        Self {
            colors: [
                [0.10, 0.05, 0.25, 1.0],
                [0.35, 0.12, 0.55, 1.0],
                [0.60, 0.80, 1.00, 1.0],
            ],
            scale: 0.005,
            octaves: 5,
            warp_octaves: 4,
            warp_strength: 3.0,
            cloud_low: 0.30,
            cloud_high: 0.85,
            intensity: 1.0,
            parallax: 0.12,
            offset: Vec2::ZERO,
            rotation: 0.0,
        }
    }
}

/// One star layer: a hashed field of single-pixel twinkling stars, drifting at
/// its own parallax rate.
#[derive(Debug, Clone, Reflect)]
pub struct StarLayer {
    /// Art-pixels between star slots (one candidate star per `spacing`² block).
    pub spacing: f32,
    /// Fraction of blocks that hold a star (0-1).
    pub density: f32,
    /// Emission multiplier; above 1 pushes peaks toward white.
    pub brightness: f32,
    /// Fraction of [`Nebula::scroll`] this layer drifts by (parallax depth).
    pub parallax: f32,
    /// Per-layer seed offset, decorrelating layers from one another.
    pub seed: f32,
}

impl Default for StarLayer {
    fn default() -> Self {
        Self {
            spacing: 9.0,
            density: 0.55,
            brightness: 1.6,
            parallax: 0.15,
            seed: 0.0,
        }
    }
}

/// Layered nebula + starfield background.
///
/// Attach with a `Transform`; the plugin inserts the quad mesh and material.
/// Leave [`Nebula::palette`] empty for a smooth, un-quantized look; supply
/// linear-RGB colors to snap to a palette with dithering.
#[derive(Component, Debug, Clone, Reflect)]
#[reflect(Component)]
pub struct Nebula {
    /// World-space size of the quad.
    pub size: Vec2,
    /// Deep-space base color behind every layer (linear RGBA).
    pub background: [f32; 4],
    /// Cloud layers, composited back to front. Capped at [`MAX_NEBULA_LAYERS`].
    pub layers: Vec<NebulaLayer>,
    /// Star layers, added on top. Capped at [`MAX_STAR_LAYERS`].
    pub stars: Vec<StarLayer>,
    /// Cool end of the star color shift (linear RGBA).
    pub star_cool: [f32; 4],
    /// Warm end of the star color shift (linear RGBA).
    pub star_warm: [f32; 4],
    /// Twinkle rate (radians per second, scaled per star).
    pub twinkle_speed: f32,
    /// Twinkle pulse exponent; higher = briefer, sparklier flashes.
    pub twinkle_sharpness: f32,
    /// Oscillation floor below which a star winks fully dark (0 disables).
    pub star_blackout: f32,
    /// How layers combine with what is behind them.
    pub blend: NebulaBlend,
    /// Quantization palette (max 64, linear RGB). Empty disables quantization.
    pub palette: Vec<[f32; 4]>,
    /// Dither pattern: 0 = none, 1 = Bayer 4x4, 2 = Bayer 8x8.
    pub dither_pattern: u32,
    /// World units per art pixel. `0.0` derives it from screen derivatives
    /// (pixel-perfect, follows zoom); a positive value forces a size.
    pub pixel_size: f32,
    /// Parallax reference — set to the camera's world position each frame. Each
    /// layer multiplies this by its own parallax factor.
    pub scroll: Vec2,
    /// Global pattern rotation in radians (e.g. a day/night sky angle).
    pub rotation: f32,
    /// Global pattern seed, added to each star layer's seed.
    pub seed: f32,
}

impl Default for Nebula {
    fn default() -> Self {
        Self {
            size: Vec2::splat(2000.0),
            background: [0.02, 0.01, 0.05, 1.0],
            layers: vec![
                NebulaLayer {
                    colors: [
                        [0.06, 0.03, 0.16, 1.0],
                        [0.30, 0.10, 0.50, 1.0],
                        [0.45, 0.55, 0.95, 1.0],
                    ],
                    scale: 0.0035,
                    intensity: 0.9,
                    parallax: 0.10,
                    ..Default::default()
                },
                NebulaLayer {
                    colors: [
                        [0.04, 0.10, 0.28, 1.0],
                        [0.10, 0.35, 0.65, 1.0],
                        [0.70, 0.95, 1.00, 1.0],
                    ],
                    scale: 0.009,
                    cloud_low: 0.45,
                    intensity: 0.6,
                    parallax: 0.18,
                    ..Default::default()
                },
            ],
            stars: vec![
                StarLayer {
                    spacing: 9.0,
                    parallax: 0.14,
                    ..Default::default()
                },
                StarLayer {
                    spacing: 15.0,
                    density: 0.4,
                    brightness: 1.3,
                    parallax: 0.22,
                    seed: 5.0,
                },
            ],
            star_cool: [0.60, 0.80, 1.00, 1.0],
            star_warm: [1.00, 0.85, 0.50, 1.0],
            twinkle_speed: 2.0,
            twinkle_sharpness: 3.0,
            star_blackout: 0.12,
            blend: NebulaBlend::Over,
            palette: Vec::new(),
            dither_pattern: 1,
            pixel_size: 0.0,
            scroll: Vec2::ZERO,
            rotation: 0.0,
            seed: 0.0,
        }
    }
}

impl Nebula {
    /// Pack the settings into shader uniforms at time `time`. Layers and star
    /// layers beyond the fixed caps are dropped.
    pub fn to_uniforms(&self, time: f32) -> NebulaUniforms {
        let mut layers = [NebulaLayerUniform::default(); MAX_NEBULA_LAYERS];
        for (slot, layer) in layers.iter_mut().zip(self.layers.iter()) {
            *slot = NebulaLayerUniform {
                c1: Vec4::from_array(layer.colors[0]),
                c2: Vec4::from_array(layer.colors[1]),
                c3: Vec4::from_array(layer.colors[2]),
                offset: layer.offset,
                scale: layer.scale,
                warp_strength: layer.warp_strength,
                cloud_low: layer.cloud_low,
                cloud_high: layer.cloud_high,
                intensity: layer.intensity,
                parallax: layer.parallax,
                rotation: layer.rotation,
                octaves: layer.octaves.clamp(1, 8),
                warp_octaves: layer.warp_octaves.clamp(1, 8),
                _pad0: 0.0,
            };
        }

        let mut stars = [StarLayerUniform::default(); MAX_STAR_LAYERS];
        for (slot, star) in stars.iter_mut().zip(self.stars.iter()) {
            *slot = StarLayerUniform {
                params: Vec4::new(
                    star.spacing.max(1.0),
                    star.density,
                    star.brightness,
                    star.parallax,
                ),
                seed_pad: Vec4::new(star.seed, 0.0, 0.0, 0.0),
            };
        }

        NebulaUniforms {
            layers,
            stars,
            background: Vec4::from_array(self.background),
            star_cool: Vec4::from_array(self.star_cool),
            star_warm: Vec4::from_array(self.star_warm),
            world_size: self.size,
            scroll: self.scroll,
            num_layers: self.layers.len().min(MAX_NEBULA_LAYERS) as u32,
            num_stars: self.stars.len().min(MAX_STAR_LAYERS) as u32,
            blend: self.blend.as_u32(),
            twinkle_speed: self.twinkle_speed,
            twinkle_sharpness: self.twinkle_sharpness,
            star_blackout: self.star_blackout,
            rotation: self.rotation,
            time,
            pixel_size: self.pixel_size,
            seed: self.seed,
            _pad0: 0.0,
            _pad1: 0.0,
        }
    }

    /// Build the quantization uniforms from the configured palette.
    pub fn quantization_uniforms(&self) -> ColorQuantizeUniforms {
        let mut palette = [Vec4::ZERO; MAX_PALETTE_COLORS];
        let mut palette_oklab = [Vec4::ZERO; MAX_PALETTE_COLORS];
        for (i, color) in self.palette.iter().take(MAX_PALETTE_COLORS).enumerate() {
            palette[i] = Vec4::from_array(*color);
            let (l, a, b) = linear_rgb_to_oklab(color[0], color[1], color[2]);
            palette_oklab[i] = Vec4::new(l, a, b, 0.0);
        }
        ColorQuantizeUniforms {
            palette,
            palette_oklab,
            palette_size: self.palette.len().min(MAX_PALETTE_COLORS) as u32,
            alpha_cutoff: 0.03,
            dither_pattern: self.dither_pattern,
            transparency_floor: 0.06,
        }
    }
}

/// One nebula layer packed for the shader. Mirrors the WGSL `NebulaLayer`.
#[derive(Clone, Copy, ShaderType, Default)]
pub struct NebulaLayerUniform {
    pub c1: Vec4,
    pub c2: Vec4,
    pub c3: Vec4,
    pub offset: Vec2,
    pub scale: f32,
    pub warp_strength: f32,
    pub cloud_low: f32,
    pub cloud_high: f32,
    pub intensity: f32,
    pub parallax: f32,
    pub rotation: f32,
    pub octaves: u32,
    pub warp_octaves: u32,
    pub _pad0: f32,
}

/// One star layer packed for the shader. Mirrors the WGSL `StarLayer`.
#[derive(Clone, Copy, ShaderType, Default)]
pub struct StarLayerUniform {
    /// `(spacing, density, brightness, parallax)`.
    pub params: Vec4,
    /// `(seed, _, _, _)`.
    pub seed_pad: Vec4,
}

/// Shader uniforms for the nebula material.
///
/// Field order matches the WGSL `NebulaSettings` struct layout exactly.
#[derive(Clone, Copy, ShaderType)]
pub struct NebulaUniforms {
    pub layers: [NebulaLayerUniform; MAX_NEBULA_LAYERS],
    pub stars: [StarLayerUniform; MAX_STAR_LAYERS],
    pub background: Vec4,
    pub star_cool: Vec4,
    pub star_warm: Vec4,
    pub world_size: Vec2,
    pub scroll: Vec2,
    pub num_layers: u32,
    pub num_stars: u32,
    pub blend: u32,
    pub twinkle_speed: f32,
    pub twinkle_sharpness: f32,
    pub star_blackout: f32,
    pub rotation: f32,
    pub time: f32,
    pub pixel_size: f32,
    pub seed: f32,
    pub _pad0: f32,
    pub _pad1: f32,
}

/// Material that renders the layered nebula + starfield.
#[derive(Asset, TypePath, AsBindGroup, Clone)]
pub struct NebulaMaterial {
    #[uniform(0)]
    pub uniforms: NebulaUniforms,
    #[uniform(1)]
    pub quantization: ColorQuantizeUniforms,
}

impl Material2d for NebulaMaterial {
    fn fragment_shader() -> ShaderRef {
        "embedded://msg_shaders/shaders/nebula_material.wgsl".into()
    }

    fn alpha_mode(&self) -> AlphaMode2d {
        AlphaMode2d::Opaque
    }
}

/// Marks a [`Nebula`] entity whose mesh and material have been inserted.
#[derive(Component)]
struct NebulaMesh;

/// Plugin that registers the nebula material and drives it.
///
/// ```
/// use bevy::prelude::*;
/// use msg_shaders::NebulaPlugin;
///
/// fn register(app: &mut App) {
///     app.add_plugins(NebulaPlugin);
/// }
/// ```
pub struct NebulaPlugin;

impl Plugin for NebulaPlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<crate::ColorQuantizationPlugin>() {
            app.add_plugins(crate::ColorQuantizationPlugin);
        }
        if !app.is_plugin_added::<crate::PixelationPlugin>() {
            app.add_plugins(crate::PixelationPlugin);
        }

        load_internal_asset!(
            app,
            NEBULA_FUNCTIONS_SHADER_HANDLE,
            "shaders/nebula_functions.wgsl",
            Shader::from_wgsl
        );
        embedded_asset!(app, "shaders/nebula_material.wgsl");

        app.register_type::<Nebula>();
        app.register_type::<NebulaLayer>();
        app.register_type::<StarLayer>();
        app.register_type::<NebulaBlend>();

        if !app.is_plugin_added::<Material2dPlugin<NebulaMaterial>>() {
            app.add_plugins(Material2dPlugin::<NebulaMaterial>::default());
        }

        app.add_systems(Update, (spawn_nebula_meshes, animate_nebula));
    }
}

/// Inserts the quad mesh and material for newly added [`Nebula`] entities,
/// preserving the entity's existing `Transform`.
fn spawn_nebula_meshes(
    mut commands: Commands,
    mut materials: ResMut<Assets<NebulaMaterial>>,
    mut meshes: ResMut<Assets<Mesh>>,
    query: Query<(Entity, &Nebula), Without<NebulaMesh>>,
) {
    for (entity, nebula) in &query {
        let mesh = meshes.add(Mesh::from(Rectangle::new(nebula.size.x, nebula.size.y)));
        let material = materials.add(NebulaMaterial {
            uniforms: nebula.to_uniforms(0.0),
            quantization: nebula.quantization_uniforms(),
        });
        commands
            .entity(entity)
            .insert((NebulaMesh, Mesh2d(mesh), MeshMaterial2d(material)));
    }
}

/// Advances the nebula time and re-syncs animatable settings from the component,
/// so runtime edits take effect. The palette is rebuilt only when the component
/// changes.
fn animate_nebula(
    time: Res<Time>,
    mut materials: ResMut<Assets<NebulaMaterial>>,
    query: Query<(Ref<Nebula>, &MeshMaterial2d<NebulaMaterial>)>,
) {
    let elapsed = time.elapsed_secs();
    for (nebula, handle) in &query {
        let Some(material) = materials.get_mut(&handle.0) else {
            continue;
        };
        material.uniforms = nebula.to_uniforms(elapsed);
        if nebula.is_changed() {
            material.quantization = nebula.quantization_uniforms();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_palette_disables_quantization() {
        let nebula = Nebula {
            palette: Vec::new(),
            ..Default::default()
        };
        assert_eq!(nebula.quantization_uniforms().palette_size, 0);
    }

    #[test]
    fn palette_size_is_clamped() {
        let nebula = Nebula {
            palette: vec![[1.0, 1.0, 1.0, 1.0]; MAX_PALETTE_COLORS + 8],
            ..Default::default()
        };
        assert_eq!(
            nebula.quantization_uniforms().palette_size as usize,
            MAX_PALETTE_COLORS
        );
    }

    #[test]
    fn layer_and_star_counts_are_capped() {
        let nebula = Nebula {
            layers: vec![NebulaLayer::default(); MAX_NEBULA_LAYERS + 3],
            stars: vec![StarLayer::default(); MAX_STAR_LAYERS + 2],
            ..Default::default()
        };
        let u = nebula.to_uniforms(0.0);
        assert_eq!(u.num_layers as usize, MAX_NEBULA_LAYERS);
        assert_eq!(u.num_stars as usize, MAX_STAR_LAYERS);
    }

    #[test]
    fn octaves_are_clamped() {
        let nebula = Nebula {
            layers: vec![NebulaLayer {
                octaves: 99,
                warp_octaves: 0,
                ..Default::default()
            }],
            ..Default::default()
        };
        let u = nebula.to_uniforms(0.0);
        assert_eq!(u.layers[0].octaves, 8);
        assert_eq!(u.layers[0].warp_octaves, 1);
    }

    #[test]
    fn star_spacing_never_below_one() {
        let nebula = Nebula {
            stars: vec![StarLayer {
                spacing: 0.0,
                ..Default::default()
            }],
            ..Default::default()
        };
        assert!(nebula.to_uniforms(0.0).stars[0].params.x >= 1.0);
    }

    #[test]
    fn blend_maps_to_flag() {
        assert_eq!(NebulaBlend::Over.as_u32(), 0);
        assert_eq!(NebulaBlend::Additive.as_u32(), 1);
    }
}
