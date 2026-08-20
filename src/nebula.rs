//! Layered procedural nebula + twinkling starfield background.
//!
//! [`NebulaPlugin`] registers a [`Nebula`] component: attach it to an entity
//! (with a `Transform`) and the plugin builds a quad that composites a stack of
//! nebula [`NebulaLayer`]s over a base color, then a stack of [`StarLayer`]s.
//! The output is snapped to the game's art-pixel grid (pixel-perfect from
//! screen-space derivatives) with ordered dithering, reusing
//! [`crate::PixelationPlugin`].
//!
//! Each nebula layer has its own noise settings, a three-stop color ramp
//! (`c1` edge -> `c2` body -> `c3` core), and a parallax factor. Star layers
//! likewise carry their own spacing, density, and parallax. Set
//! [`Nebula::scroll`] to the camera's world position each frame; each layer
//! multiplies it by its own parallax factor for depth.
//!
//! **Layers never share colors.** Each layer finishes its own math: its density
//! decides where it paints and which of its own three stops it paints with,
//! resolved by one ordered-dither decision rather than an alpha. The result is written
//! over what is behind rather than blended into it, so every pixel on screen is
//! exactly one authored color — the background, one of a cloud layer's three
//! stops, or one of the two star colors. Nothing downstream has to match a
//! blended color back to a shared palette, which is what would otherwise let
//! one layer's color choices move another layer's visible edges.
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

use crate::DitherPattern;
use smallvec::{SmallVec, smallvec};

use bevy::{
    asset::{embedded_asset, load_internal_asset, uuid_handle},
    prelude::*,
    render::render_resource::{AsBindGroup, ShaderType},
    shader::{Shader, ShaderRef},
    sprite_render::{AlphaMode2d, Material2d, Material2dPlugin},
};

/// Handle for the importable nebula functions shader.
pub const NEBULA_FUNCTIONS_SHADER_HANDLE: Handle<Shader> =
    uuid_handle!("3c9d1e77-2b64-4a05-9f18-6d7c8a1b2e34");

/// Maximum nebula cloud layers uploaded to the shader (fixed uniform array).
/// Extra layers on the [`Nebula`] are dropped by [`Nebula::to_uniforms`].
pub const MAX_NEBULA_LAYERS: usize = 6;

/// Maximum star layers uploaded to the shader (fixed uniform array).
pub const MAX_STAR_LAYERS: usize = 4;

/// Maximum star colors uploaded to the shader, summed over every star layer
/// (fixed uniform array). [`Nebula::to_uniforms`] fills the array layer by
/// layer and drops the colors that no longer fit.
pub const MAX_STAR_COLORS: usize = 8;

/// One nebula cloud layer: a noise field mapped through a three-stop color ramp
/// and composited with its own parallax and decorrelation offset.
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
    /// How far up its own ramp this layer climbs where it is densest, as a
    /// fraction: `1.0` reaches `c3`, `0.5` peaks halfway between `c1` and `c2`.
    /// Lower values thin the layer and hold it to its darker stops, never
    /// dimming it into a blend with the layer below.
    pub intensity: f32,
    /// How much this layer sticks to the world as [`Nebula::scroll`] moves: `0`
    /// pins it to the camera, `1` moves it with the world. Lower reads as further
    /// away. The drift is quantized to whole art pixels.
    pub parallax: f32,
    /// Sampling offset in world units, decorrelating layers that share settings.
    /// The noise is hashed per lattice cell, so a shift alone yields an
    /// unrelated field — no second decorrelation axis is needed.
    pub offset: Vec2,
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
    /// Colors this layer's stars are drawn from (linear RGBA). Each star picks
    /// one by hash and keeps it, so a layer shows exactly these colors and
    /// never a mixture of them. An empty list draws no stars.
    ///
    /// Give near layers the brighter end of the palette and far layers the
    /// dimmer end: with stars drawn at full color, this split is what reads as
    /// depth. Inline capacity is [`MAX_STAR_COLORS`], the shader's cap on the
    /// colors of every layer put together, so a list that spills to the heap is
    /// already one the shader cannot take whole.
    pub colors: SmallVec<[[f32; 4]; MAX_STAR_COLORS]>,
    /// How much this layer sticks to the world as [`Nebula::scroll`] moves: `0`
    /// pins it to the camera, `1` moves it with the world. Lower reads as further
    /// away. The drift is quantized to whole star cells.
    pub parallax: f32,
    /// Per-layer seed offset, decorrelating layers from one another.
    pub seed: f32,
}

impl Default for StarLayer {
    fn default() -> Self {
        Self {
            spacing: 9.0,
            density: 0.55,
            colors: smallvec![[0.60, 0.80, 1.00, 1.0]],
            parallax: 0.15,
            seed: 0.0,
        }
    }
}

/// Layered nebula + starfield background.
///
/// Attach with a `Transform`; the plugin inserts the quad mesh and material.
/// There is no separate quantization palette: the colors authored on the
/// background, the layers, and the star ends *are* the palette, and every
/// rendered pixel is one of them exactly.
#[derive(Component, Debug, Clone, Reflect)]
#[reflect(Component)]
pub struct Nebula {
    /// World-space size of the quad.
    pub size: Vec2,
    /// Deep-space base color behind every layer (linear RGBA).
    pub background: [f32; 4],
    /// Cloud layers, composited back to front. Capped at [`MAX_NEBULA_LAYERS`].
    pub layers: Vec<NebulaLayer>,
    /// Star layers, drawn on top. Capped at [`MAX_STAR_LAYERS`].
    pub stars: Vec<StarLayer>,
    /// Blink rate (radians per second, scaled per star).
    pub twinkle_speed: f32,
    /// Fraction of stars that twinkle at all (0-1). Those that do blink between
    /// lit and dark; the rest hold steady, so `0.0` gives a still sky and `1.0`
    /// puts every star on a blink.
    pub twinkle_chance: f32,
    /// Ordered-dither matrix shifting each layer's level across its stop
    /// boundaries. [`DitherPattern::None`] gives hard edges and hard bands.
    pub dither_pattern: DitherPattern,
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
                    intensity: 0.85,
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
                    colors: smallvec![[0.45, 0.60, 0.85, 1.0], [1.00, 0.85, 0.50, 1.0]],
                    parallax: 0.22,
                    seed: 5.0,
                },
            ],
            twinkle_speed: 2.0,
            twinkle_chance: 0.15,
            dither_pattern: DitherPattern::default(),
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
                octaves: layer.octaves.clamp(1, 8),
                warp_octaves: layer.warp_octaves.clamp(1, 8),
                _pad0: 0.0,
                _pad1: 0.0,
            };
        }

        // Every layer's colors go into one flat array; a layer addresses its own
        // by the `(first, count)` slice it was packed into. Colors past the cap
        // are dropped from the tail, so the layers that fit keep the palette
        // they were authored with.
        let mut star_colors = [Vec4::ZERO; MAX_STAR_COLORS];
        let mut used = 0usize;

        let mut stars = [StarLayerUniform::default(); MAX_STAR_LAYERS];
        for (slot, star) in stars.iter_mut().zip(self.stars.iter()) {
            let first = used;
            for color in star.colors.iter().take(MAX_STAR_COLORS - first) {
                star_colors[used] = Vec4::from_array(*color);
                used += 1;
            }
            *slot = StarLayerUniform {
                params: Vec4::new(
                    star.spacing.max(1.0),
                    star.density,
                    first as f32,
                    star.parallax,
                ),
                seed_pad: Vec4::new(star.seed, (used - first) as f32, 0.0, 0.0),
            };
        }

        NebulaUniforms {
            layers,
            stars,
            background: Vec4::from_array(self.background),
            star_colors,
            world_size: self.size,
            scroll: self.scroll,
            num_layers: self.layers.len().min(MAX_NEBULA_LAYERS) as u32,
            num_stars: self.stars.len().min(MAX_STAR_LAYERS) as u32,
            dither_pattern: self.dither_pattern.as_u32(),
            twinkle_speed: self.twinkle_speed,
            twinkle_chance: self.twinkle_chance.clamp(0.0, 1.0),
            _pad2: 0.0,
            rotation: self.rotation,
            time,
            pixel_size: self.pixel_size,
            seed: self.seed,
            _pad0: 0.0,
            _pad1: 0.0,
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
    pub octaves: u32,
    pub warp_octaves: u32,
    pub _pad0: f32,
    pub _pad1: f32,
}

/// One star layer packed for the shader. Mirrors the WGSL `StarLayer`.
#[derive(Clone, Copy, ShaderType, Default)]
pub struct StarLayerUniform {
    /// `(spacing, density, first_color, parallax)`, where `first_color` indexes
    /// [`NebulaUniforms::star_colors`].
    pub params: Vec4,
    /// `(seed, color_count, _, _)`.
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
    /// Every star layer's colors, concatenated; each layer reads the slice its
    /// `params.z`/`seed_pad.y` name.
    pub star_colors: [Vec4; MAX_STAR_COLORS],
    pub world_size: Vec2,
    pub scroll: Vec2,
    pub num_layers: u32,
    pub num_stars: u32,
    pub dither_pattern: u32,
    pub twinkle_speed: f32,
    pub twinkle_chance: f32,
    pub _pad2: f32,
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
}

impl Material2d for NebulaMaterial {
    fn fragment_shader() -> ShaderRef {
        "embedded://msg_shaders/shaders/nebula_material.wgsl".into()
    }

    fn alpha_mode(&self) -> AlphaMode2d {
        AlphaMode2d::Opaque
    }
}

/// Marks a [`Nebula`] entity whose mesh and material have been inserted, and
/// records the [`Nebula::size`] its quad was built from. The shader maps UVs
/// back to world units through the `world_size` uniform, so a stale extent here
/// scales every world-space quantity it derives, the art-pixel grid included.
#[derive(Component)]
struct NebulaMesh(Vec2);

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

        if !app.is_plugin_added::<Material2dPlugin<NebulaMaterial>>() {
            app.add_plugins(Material2dPlugin::<NebulaMaterial>::default());
        }

        app.add_systems(
            Update,
            (spawn_nebula_meshes, resize_nebula_meshes, animate_nebula).chain(),
        );
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
        });
        commands.entity(entity).insert((
            NebulaMesh(nebula.size),
            Mesh2d(mesh),
            MeshMaterial2d(material),
        ));
    }
}

/// Rebuilds the quad whenever [`Nebula::size`] changes, keeping the geometry
/// and the `world_size` uniform describing the same rectangle.
fn resize_nebula_meshes(
    mut meshes: ResMut<Assets<Mesh>>,
    mut query: Query<(&Nebula, &mut NebulaMesh, &Mesh2d), Changed<Nebula>>,
) {
    for (nebula, mut built, handle) in &mut query {
        if built.0 == nebula.size {
            continue;
        }
        let Some(mesh) = meshes.get_mut(&handle.0) else {
            continue;
        };
        *mesh = Mesh::from(Rectangle::new(nebula.size.x, nebula.size.y));
        built.0 = nebula.size;
    }
}

/// Advances the nebula time and re-syncs animatable settings from the component,
/// so runtime edits take effect.
fn animate_nebula(
    time: Res<Time>,
    mut materials: ResMut<Assets<NebulaMaterial>>,
    query: Query<(&Nebula, &MeshMaterial2d<NebulaMaterial>)>,
) {
    let elapsed = time.elapsed_secs();
    for (nebula, handle) in &query {
        let Some(material) = materials.get_mut(&handle.0) else {
            continue;
        };
        material.uniforms = nebula.to_uniforms(elapsed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dither_pattern_reaches_the_shader() {
        let nebula = Nebula {
            dither_pattern: DitherPattern::Bayer8x8,
            ..Default::default()
        };
        assert_eq!(nebula.to_uniforms(0.0).dither_pattern, 2);
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

    /// Both chances are probabilities the shader tests a hash against, so a
    /// `twinkle_chance` is a probability the shader tests a hash against, so a
    /// value outside `0..=1` would make every star take one branch.
    #[test]
    fn twinkle_chance_is_clamped_to_a_probability() {
        let nebula = Nebula {
            twinkle_chance: -1.0,
            ..Default::default()
        };
        assert_eq!(nebula.to_uniforms(0.0).twinkle_chance, 0.0);
    }

    /// Each layer must address its own colors and no one else's, whatever the
    /// layers before it contributed.
    #[test]
    fn star_layers_get_disjoint_color_slices() {
        let nebula = Nebula {
            stars: vec![
                StarLayer {
                    colors: smallvec![[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]],
                    ..Default::default()
                },
                StarLayer {
                    colors: smallvec![[0.0, 0.0, 1.0, 1.0]],
                    ..Default::default()
                },
            ],
            ..Default::default()
        };
        let u = nebula.to_uniforms(0.0);

        assert_eq!((u.stars[0].params.z, u.stars[0].seed_pad.y), (0.0, 2.0));
        assert_eq!((u.stars[1].params.z, u.stars[1].seed_pad.y), (2.0, 1.0));
        assert_eq!(u.star_colors[0], Vec4::new(1.0, 0.0, 0.0, 1.0));
        assert_eq!(u.star_colors[2], Vec4::new(0.0, 0.0, 1.0, 1.0));
    }

    /// Past the cap the colors that fit must keep their slices, so the layers
    /// that still have a palette are drawn with the one they were authored
    /// with rather than a shifted window into someone else's.
    #[test]
    fn star_colors_past_the_cap_are_dropped_from_the_tail() {
        let nebula = Nebula {
            stars: vec![
                StarLayer {
                    colors: smallvec![[1.0, 0.0, 0.0, 1.0]; MAX_STAR_COLORS],
                    ..Default::default()
                },
                StarLayer {
                    colors: smallvec![[0.0, 1.0, 0.0, 1.0]],
                    ..Default::default()
                },
            ],
            ..Default::default()
        };
        let u = nebula.to_uniforms(0.0);

        assert_eq!(u.stars[0].seed_pad.y, MAX_STAR_COLORS as f32);
        // Nothing left for the second layer, which then draws no stars.
        assert_eq!(u.stars[1].seed_pad.y, 0.0);
    }

    /// Recoloring one layer must leave every layer's shape untouched. The
    /// colors only ever reach `c1`/`c2`/`c3`; nothing the density field is
    /// built from may be derived from them.
    #[test]
    fn layer_colors_never_reach_shape_uniforms() {
        let base = Nebula::default();
        let mut recolored = base.clone();
        recolored.layers[0].colors = [
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
        ];

        let a = base.to_uniforms(0.0);
        let b = recolored.to_uniforms(0.0);
        assert_eq!(a.num_layers, b.num_layers);
        for i in 0..a.num_layers as usize {
            let (x, y) = (a.layers[i], b.layers[i]);
            assert_eq!(x.offset, y.offset, "layer {i} offset");
            assert_eq!(x.scale, y.scale, "layer {i} scale");
            assert_eq!(x.warp_strength, y.warp_strength, "layer {i} warp_strength");
            assert_eq!(x.cloud_low, y.cloud_low, "layer {i} cloud_low");
            assert_eq!(x.cloud_high, y.cloud_high, "layer {i} cloud_high");
            assert_eq!(x.intensity, y.intensity, "layer {i} intensity");
            assert_eq!(x.parallax, y.parallax, "layer {i} parallax");
            assert_eq!(x.octaves, y.octaves, "layer {i} octaves");
            assert_eq!(x.warp_octaves, y.warp_octaves, "layer {i} warp_octaves");
        }
        assert_eq!(a.seed, b.seed);
    }

    /// A layer's colors reach the shader as its own three stops and nowhere
    /// else, so no other layer can see them.
    #[test]
    fn layer_colors_stay_on_their_own_layer() {
        let mut nebula = Nebula::default();
        let marker = [1.0, 0.0, 1.0, 1.0];
        nebula.layers[0].colors[1] = marker;

        let u = nebula.to_uniforms(0.0);
        assert_eq!(u.layers[0].c2, Vec4::from_array(marker));
        for i in 1..u.num_layers as usize {
            let l = u.layers[i];
            let m = Vec4::from_array(marker);
            assert!(l.c1 != m && l.c2 != m && l.c3 != m, "layer {i} saw layer 0");
        }
    }
}
