//! Procedural nebula + twinkling starfield background.
//!
//! [`NebulaPlugin`] registers a [`Nebula`] component: attach it to an entity
//! (with a `Transform`) and the plugin builds a quad that renders fractal-noise
//! nebula clouds and a hashed, twinkling star field. The output is snapped to
//! the game's art-pixel grid (pixel-perfect, derived from screen-space
//! derivatives so it tracks camera zoom) and reduced to a palette with ordered
//! dithering, reusing [`crate::ColorQuantizationPlugin`] and
//! [`crate::PixelationPlugin`].
//!
//! Stars sparkle: each pulses in brightness (occasionally winking to full black)
//! and shifts color between a cool and a warm tint over time.
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

/// Emission colors for the nebula and its stars (linear RGBA).
#[derive(Debug, Clone, Reflect)]
pub struct NebulaColors {
    /// Dominant nebula tint.
    pub color_a: [f32; 4],
    /// Secondary nebula tint, mixed in by a low-frequency hue field.
    pub color_b: [f32; 4],
    /// Deep-space base color behind the clouds.
    pub background: [f32; 4],
    /// Cool end of the star color shift.
    pub star_cool: [f32; 4],
    /// Warm end of the star color shift.
    pub star_warm: [f32; 4],
}

impl Default for NebulaColors {
    fn default() -> Self {
        Self {
            color_a: [0.35, 0.12, 0.55, 1.0],
            color_b: [0.06, 0.20, 0.45, 1.0],
            background: [0.02, 0.01, 0.05, 1.0],
            star_cool: [0.60, 0.80, 1.00, 1.0],
            star_warm: [1.00, 0.85, 0.50, 1.0],
        }
    }
}

/// Procedural nebula + starfield background.
///
/// Attach with a `Transform`; the plugin inserts the quad mesh and material.
/// The pattern is drawn in a coordinate local to the quad, so [`Nebula::scroll`]
/// (drift/parallax) and [`Nebula::rotation`] (day/night sky) stay precise far
/// from the world origin. Leave [`Nebula::palette`] empty for a smooth,
/// un-quantized look; supply linear-RGB colors to snap to a palette.
#[derive(Component, Debug, Clone, Reflect)]
#[reflect(Component)]
pub struct Nebula {
    /// World-space size of the quad.
    pub size: Vec2,
    /// Emission colors.
    pub colors: NebulaColors,
    /// Quantization palette (max 64, linear RGB). Empty disables quantization.
    pub palette: Vec<[f32; 4]>,
    /// Dither pattern: 0 = none, 1 = Bayer 4x4, 2 = Bayer 8x8.
    pub dither_pattern: u32,
    /// Nebula feature frequency (cycles per world unit). Smaller = larger clouds.
    pub nebula_scale: f32,
    /// Overall cloud density / contrast.
    pub density: f32,
    /// fbm octaves for the cloud field. Clamped to `1..=8`.
    pub octaves: u32,
    /// fbm octaves for the domain-warp and hue fields. Clamped to `1..=8`.
    pub warp_octaves: u32,
    /// How far the domain warp displaces the cloud lookup (wispiness).
    pub warp_strength: f32,
    /// Lower edge of the cloud smoothstep band. Raising it thins the clouds.
    pub cloud_low: f32,
    /// Upper edge of the cloud smoothstep band. Widening the band softens them.
    pub cloud_high: f32,
    /// Fraction of star blocks that contain a star (0-1).
    pub star_density: f32,
    /// Art-pixels between star slots on the densest layer.
    pub star_spacing: f32,
    /// Star emission multiplier; values above 1 push peaks toward white.
    pub star_brightness: f32,
    /// Twinkle rate (radians per second, scaled per star).
    pub twinkle_speed: f32,
    /// Exponent sharpening the twinkle pulse; higher = brief, sparkly flashes.
    pub twinkle_sharpness: f32,
    /// Oscillation value below which a star winks fully dark (0 disables).
    pub star_blackout: f32,
    /// Pattern drift offset in world units (parallax / slow motion).
    pub scroll: Vec2,
    /// Pattern rotation in radians (e.g. a day/night sky angle).
    pub rotation: f32,
    /// World units per art pixel. `0.0` derives it from screen derivatives
    /// (pixel-perfect, follows camera zoom); a positive value forces a size.
    pub pixel_size: f32,
    /// Pattern seed.
    pub seed: f32,
}

impl Default for Nebula {
    fn default() -> Self {
        Self {
            size: Vec2::splat(2000.0),
            colors: NebulaColors::default(),
            palette: Vec::new(),
            dither_pattern: 1,
            nebula_scale: 0.006,
            density: 1.0,
            octaves: 5,
            warp_octaves: 4,
            warp_strength: 3.0,
            cloud_low: 0.30,
            cloud_high: 0.85,
            star_density: 0.55,
            star_spacing: 9.0,
            star_brightness: 1.6,
            twinkle_speed: 2.0,
            twinkle_sharpness: 3.0,
            star_blackout: 0.12,
            scroll: Vec2::ZERO,
            rotation: 0.0,
            pixel_size: 0.0,
            seed: 0.0,
        }
    }
}

impl Nebula {
    /// Pack the animatable settings into shader uniforms at time `time`.
    pub fn to_uniforms(&self, time: f32) -> NebulaUniforms {
        NebulaUniforms {
            color_a: Vec4::from_array(self.colors.color_a),
            color_b: Vec4::from_array(self.colors.color_b),
            background: Vec4::from_array(self.colors.background),
            star_cool: Vec4::from_array(self.colors.star_cool),
            star_warm: Vec4::from_array(self.colors.star_warm),
            world_size: self.size,
            scroll: self.scroll,
            nebula_scale: self.nebula_scale,
            density: self.density,
            star_density: self.star_density,
            twinkle_speed: self.twinkle_speed,
            star_spacing: self.star_spacing.max(1.0),
            star_brightness: self.star_brightness,
            rotation: self.rotation,
            time,
            pixel_size: self.pixel_size,
            seed: self.seed,
            octaves: self.octaves.clamp(1, 8),
            warp_octaves: self.warp_octaves.clamp(1, 8),
            warp_strength: self.warp_strength,
            cloud_low: self.cloud_low,
            cloud_high: self.cloud_high,
            twinkle_sharpness: self.twinkle_sharpness,
            star_blackout: self.star_blackout,
            _pad0: 0.0,
            _pad1: 0.0,
            _pad2: 0.0,
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

/// Shader uniforms for the nebula material.
///
/// Field order matches the WGSL `NebulaSettings` struct layout exactly.
#[derive(Clone, Copy, ShaderType)]
pub struct NebulaUniforms {
    pub color_a: Vec4,
    pub color_b: Vec4,
    pub background: Vec4,
    pub star_cool: Vec4,
    pub star_warm: Vec4,
    pub world_size: Vec2,
    pub scroll: Vec2,
    pub nebula_scale: f32,
    pub density: f32,
    pub star_density: f32,
    pub twinkle_speed: f32,
    pub star_spacing: f32,
    pub star_brightness: f32,
    pub rotation: f32,
    pub time: f32,
    pub pixel_size: f32,
    pub seed: f32,
    pub octaves: u32,
    pub warp_octaves: u32,
    pub warp_strength: f32,
    pub cloud_low: f32,
    pub cloud_high: f32,
    pub twinkle_sharpness: f32,
    pub star_blackout: f32,
    pub _pad0: f32,
    pub _pad1: f32,
    pub _pad2: f32,
}

/// Material that renders the procedural nebula + starfield.
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
        // The material imports the pixelation and quantization function modules.
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
        app.register_type::<NebulaColors>();

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
/// so runtime edits to [`Nebula`] (scroll, rotation, colors) take effect. The
/// palette is rebuilt only when the component changes.
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
        let nebula = Nebula::default();
        assert!(nebula.palette.is_empty());
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
    fn uniforms_carry_time_and_size() {
        let nebula = Nebula {
            size: Vec2::new(1234.0, 5678.0),
            ..Default::default()
        };
        let u = nebula.to_uniforms(3.5);
        assert_eq!(u.time, 3.5);
        assert_eq!(u.world_size, Vec2::new(1234.0, 5678.0));
    }

    #[test]
    fn star_spacing_never_below_one() {
        let nebula = Nebula {
            star_spacing: 0.0,
            ..Default::default()
        };
        assert!(nebula.to_uniforms(0.0).star_spacing >= 1.0);
    }

    #[test]
    fn octaves_are_clamped() {
        let nebula = Nebula {
            octaves: 99,
            warp_octaves: 0,
            ..Default::default()
        };
        let u = nebula.to_uniforms(0.0);
        assert_eq!(u.octaves, 8);
        assert_eq!(u.warp_octaves, 1);
    }
}
