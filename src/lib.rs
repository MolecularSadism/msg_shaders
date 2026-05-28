// msg_shaders - MSG studio shader effects for Bevy.
// Black hole based on Eric Bruneton's black_hole_shader (BSD-3-Clause)
// https://github.com/ebruneton/black_hole_shader

mod material;
mod pixelate;
mod quantize;
mod quantize_material;

use bevy::prelude::*;
pub use material::{
    BlackHoleMaterial, BlackHoleUniforms, LensingHoleMaterial, LensingHoleUniforms,
};

// Color quantization (reusable across materials).
pub use quantize::{
    COLOR_QUANTIZE_FUNCTIONS_SHADER_HANDLE, ColorQuantizationPlugin, DitherPattern,
    MAX_PALETTE_COLORS, linear_rgb_to_oklab,
};
pub use quantize_material::{ColorQuantizeMaterial, ColorQuantizeUniforms, QuantizationConfig};

// Pixelation (reusable across materials).
pub use pixelate::{
    PIXELATE_FUNCTIONS_SHADER_HANDLE, PixelateConfig, PixelateMaterial, PixelateUniforms,
    PixelationPlugin, QuantizePixelateMaterial,
};

pub mod prelude {
    pub use super::{
        BlackHole, BlackHoleColors, BlackHoleGeometry, BlackHolePlugin, BlackHoleQuantization,
        ColorQuantizationPlugin, ColorQuantizeMaterial, ColorQuantizeUniforms, DitherPattern,
        HoleQuantization, LensingHole, LensingHolePlugin, MAX_PALETTE_COLORS, PixelateConfig,
        PixelateMaterial, PixelationPlugin, QuantizationConfig, QuantizePixelateMaterial,
    };
}

// ============================================================================
// BLACK HOLE
// ============================================================================

pub struct BlackHolePlugin;

impl Plugin for BlackHolePlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(material::MaterialsPlugin);
        app.register_type::<BlackHole>();
        app.register_type::<BlackHoleColors>();
        app.register_type::<BlackHoleGeometry>();
        app.add_systems(Update, (spawn_blackhole_meshes, update_blackhole_time));
    }
}

/// Extra margin (as a multiplier) around the disc outer radius when sizing the
/// quad. Captures the photon ring and Gaussian falloff just past the disc edge.
const QUAD_DISC_PADDING: f32 = 1.05;

/// Computes the world-space quad coverage and the shader's centered-UV scale.
///
/// The shader is authored as if the centered UV ranges in `[-1, 1]` and the
/// visible disc edge lands at `centered = disk_outer = shadow_radius *
/// disk_outer_ratio` (typically ~0.6). The corners of that unit quad
/// (`centered = ±1`, distance up to `√2`) are wasted: never sampled by the
/// disc geometry.
///
/// We shrink the quad to fit the disc plus a small margin, and proportionally
/// expand the shader's UV mapping so the disc still appears at the same world
/// size and the rest of the shader math is unchanged.
///
/// Returns `(coverage, outer_scale)`:
///   - `coverage`: factor to multiply into the world-space quad size.
///   - `outer_scale`: replaces the `* 2.0` factor in `uv = (in.uv - 0.5) * outer_scale`.
///
/// The invariant `coverage = outer_scale / 2` keeps the visual disc size
/// identical to the unshrunk quad case.
fn quad_scale_factors(geo: &BlackHoleGeometry) -> (f32, f32) {
    let disc_extent = geo.shadow_radius * geo.disk_outer_ratio * QUAD_DISC_PADDING;
    // Don't grow the quad past its original 1.0-unit size, but allow shrinkage.
    let outer_scale = (2.0 * disc_extent).clamp(0.0, 2.0);
    let coverage = outer_scale * 0.5;
    (coverage, outer_scale)
}

/// Accretion disc color configuration for different radial zones.
#[derive(Debug, Clone, Reflect)]
pub struct BlackHoleColors {
    /// Innermost disc: white-hot plasma near the ISCO (Inner Stable Circular Orbit).
    pub disk_inner: [f32; 4],
    /// Mid-disc: transition zone with intermediate temperature.
    pub disk_mid: [f32; 4],
    /// Outer disc: cooler material at the disc periphery.
    pub disk_outer: [f32; 4],
    /// Photon ring glow: light trapped at the photon sphere.
    pub glow: [f32; 4],
    /// Event horizon black: the color rendered inside the Schwarzschild radius.
    pub black: [f32; 4],
}

impl Default for BlackHoleColors {
    fn default() -> Self {
        Self {
            disk_inner: [1.0, 1.0, 0.8, 1.0],
            disk_mid: [1.0, 0.6, 0.2, 1.0],
            disk_outer: [0.8, 0.2, 0.1, 1.0],
            glow: [1.0, 0.9, 0.7, 1.0],
            black: [0.0, 0.0, 0.0, 1.0],
        }
    }
}

/// Geometric parameters controlling the black hole's spatial structure.
#[derive(Debug, Clone, Reflect)]
pub struct BlackHoleGeometry {
    /// Schwarzschild radius - the event horizon boundary (normalized, 0.0-1.0).
    /// This defines the size of the black central region.
    pub shadow_radius: f32,
    /// Inner disc radius as a multiple of shadow_radius.
    /// Represents the ISCO (Innermost Stable Circular Orbit).
    /// For a Schwarzschild black hole, theoretically 3.0; visually ~1.5.
    pub disk_inner_ratio: f32,
    /// Outer disc radius as a multiple of shadow_radius.
    /// Defines the extent of the visible accretion disc.
    pub disk_outer_ratio: f32,
    /// Gaussian width of the photon ring glow effect.
    pub photon_ring_width: f32,
    /// Peak brightness of the photon ring (0.0-1.0+).
    pub photon_ring_intensity: f32,
    /// Relativistic Doppler beaming strength (0.0 = off, 1.0 = full effect).
    /// Controls how much brighter the approaching side appears.
    pub doppler_strength: f32,
    /// Accretion disc matter density multiplier.
    /// Higher values create denser, more opaque disc clouds.
    pub cloud_density: f32,
    /// Secondary lensed disc axial inner boundary as multiple of shadow_radius.
    pub axial_inner_ratio: f32,
    /// Secondary lensed disc axial outer boundary as multiple of shadow_radius.
    pub axial_outer_ratio: f32,
    /// Brightness multiplier for secondary rings.
    pub secondary_brightness: f32,
}

impl Default for BlackHoleGeometry {
    fn default() -> Self {
        Self {
            shadow_radius: 0.15,
            disk_inner_ratio: 1.5,
            disk_outer_ratio: 4.0,
            photon_ring_width: 0.003,
            photon_ring_intensity: 0.8,
            doppler_strength: 1.0,
            cloud_density: 1.0,
            axial_inner_ratio: 1.01,
            axial_outer_ratio: 1.25,
            secondary_brightness: 0.7,
        }
    }
}

/// Black hole component based on Bruneton's Schwarzschild spacetime model.
///
/// Simulates a non-rotating (Schwarzschild) black hole with:
/// - Event horizon (shadow): The boundary beyond which light cannot escape
/// - Photon sphere: Where light orbits the black hole
/// - Accretion disc: Orbiting matter heated to incandescence
/// - Gravitational lensing: Light bending around the singularity creating lensed top/bottom views
/// - Doppler beaming: Relativistic brightening of approaching matter
#[derive(Component, Debug, Clone, Reflect)]
#[reflect(Component)]
pub struct BlackHole {
    /// Visual size in world units.
    pub size: f32,
    /// Accretion disc angular velocity (radians per second).
    pub spin: f32,
    /// Observer inclination angle in radians.
    /// 0.0 = edge-on view, π/2 = face-on view.
    pub inclination: f32,
    /// Color configuration for disc zones.
    pub colors: BlackHoleColors,
    /// Geometric parameters for the black hole structure.
    pub geometry: BlackHoleGeometry,
    /// Pixelation grid: number of cells across the quad. `0.0` disables
    /// pixelation; higher values produce chunkier blocks.
    pub pixel_grid: f32,
}

impl Default for BlackHole {
    fn default() -> Self {
        Self {
            size: 100.0,
            spin: 0.5,
            inclination: 0.3,
            colors: BlackHoleColors::default(),
            geometry: BlackHoleGeometry::default(),
            pixel_grid: 0.0,
        }
    }
}

/// Shared color quantization settings used by both `BlackHole` and `LensingHole`.
///
/// When present, applies retro-style palette quantization with dithering.
/// The palette colors should be in linear RGB space.
#[derive(Component, Debug, Clone, Reflect, Default)]
#[reflect(Component)]
pub struct HoleQuantization {
    /// Color palette for quantization (max 64 colors, linear RGB)
    pub palette: Vec<[f32; 4]>,
    /// Alpha threshold below which pixels become transparent (0.0-1.0)
    pub alpha_cutoff: f32,
    /// Dither pattern: 0=none, 1=bayer4x4, 2=bayer8x8
    pub dither_pattern: u32,
    /// Minimum normalized alpha for dithering (0.0-1.0)
    pub transparency_floor: f32,
}

impl HoleQuantization {
    /// Create quantization settings with a palette.
    pub fn new(palette: Vec<[f32; 4]>) -> Self {
        Self {
            palette,
            alpha_cutoff: 0.03,
            dither_pattern: 1, // Bayer4x4
            transparency_floor: 0.06,
        }
    }

    /// Convert to shader uniforms.
    pub fn to_uniforms(&self) -> ColorQuantizeUniforms {
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
            alpha_cutoff: self.alpha_cutoff,
            dither_pattern: self.dither_pattern,
            transparency_floor: self.transparency_floor,
        }
    }
}

/// Type alias preserving backward compatibility with `BlackHoleQuantization`.
pub type BlackHoleQuantization = HoleQuantization;

#[derive(Component)]
struct BlackHoleMesh;

/// Shared 1×1 quad mesh used by both the black hole and lensing hole meshes.
#[derive(Resource)]
struct HoleQuadMesh(Handle<Mesh>);

/// Returns the shared unit-quad mesh, inserting the resource on first use so a
/// single mesh handle is reused across both hole effects.
fn hole_quad_mesh(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    existing: Option<&HoleQuadMesh>,
) -> Handle<Mesh> {
    if let Some(quad) = existing {
        return quad.0.clone();
    }
    let handle = meshes.add(Mesh::from(Rectangle::new(1.0, 1.0)));
    commands.insert_resource(HoleQuadMesh(handle.clone()));
    handle
}

fn spawn_blackhole_meshes(
    mut commands: Commands,
    mut materials: ResMut<Assets<BlackHoleMaterial>>,
    mut meshes: ResMut<Assets<Mesh>>,
    quad_mesh: Option<Res<HoleQuadMesh>>,
    query: Query<
        (Entity, &BlackHole, &Transform, Option<&HoleQuantization>),
        Without<BlackHoleMesh>,
    >,
) {
    if query.is_empty() {
        return;
    }
    let mesh_handle = hole_quad_mesh(&mut commands, &mut meshes, quad_mesh.as_deref());
    for (entity, blackhole, transform, quantization) in &query {
        let geo = &blackhole.geometry;
        let (coverage, outer_scale) = quad_scale_factors(geo);
        let uniforms = BlackHoleUniforms {
            spin: blackhole.spin,
            inclination: blackhole.inclination,
            time: 0.0,
            shadow_radius: geo.shadow_radius,
            disk_inner_ratio: geo.disk_inner_ratio,
            disk_outer_ratio: geo.disk_outer_ratio,
            photon_ring_width: geo.photon_ring_width,
            photon_ring_intensity: geo.photon_ring_intensity,
            doppler_strength: geo.doppler_strength,
            cloud_density: geo.cloud_density,
            axial_inner_ratio: geo.axial_inner_ratio,
            axial_outer_ratio: geo.axial_outer_ratio,
            secondary_brightness: geo.secondary_brightness,
            outer_scale,
            pixel_grid: blackhole.pixel_grid,
            _pad2: 0.0,
            disk_inner_color: Vec4::from_array(blackhole.colors.disk_inner),
            disk_mid_color: Vec4::from_array(blackhole.colors.disk_mid),
            disk_outer_color: Vec4::from_array(blackhole.colors.disk_outer),
            glow_color: Vec4::from_array(blackhole.colors.glow),
            black_color: Vec4::from_array(blackhole.colors.black),
        };

        let quantization_uniforms = quantization.map(|q| q.to_uniforms()).unwrap_or_default();

        let material = materials.add(BlackHoleMaterial {
            uniforms,
            quantization: quantization_uniforms,
        });

        #[cfg(feature = "render_3d")]
        commands.entity(entity).insert((
            BlackHoleMesh,
            Mesh3d(mesh_handle.clone()),
            MeshMaterial3d(material),
            Transform {
                translation: transform.translation,
                rotation: transform.rotation,
                scale: transform.scale * blackhole.size * coverage,
            },
        ));

        #[cfg(feature = "render_2d")]
        commands.entity(entity).insert((
            BlackHoleMesh,
            Mesh2d(mesh_handle.clone()),
            MeshMaterial2d(material),
            Transform {
                translation: transform.translation,
                rotation: transform.rotation,
                scale: transform.scale * blackhole.size * coverage,
            },
        ));
    }
}

#[cfg(feature = "render_3d")]
fn update_blackhole_time(
    time: Res<Time>,
    mut materials: ResMut<Assets<BlackHoleMaterial>>,
    query: Query<&MeshMaterial3d<BlackHoleMaterial>>,
) {
    let elapsed = time.elapsed_secs();

    for material_handle in &query {
        if let Some(material) = materials.get_mut(&material_handle.0) {
            material.uniforms.time = elapsed;
        }
    }
}

#[cfg(feature = "render_2d")]
fn update_blackhole_time(
    time: Res<Time>,
    mut materials: ResMut<Assets<BlackHoleMaterial>>,
    query: Query<&MeshMaterial2d<BlackHoleMaterial>>,
) {
    let elapsed = time.elapsed_secs();

    for material_handle in &query {
        if let Some(material) = materials.get_mut(&material_handle.0) {
            material.uniforms.time = elapsed;
        }
    }
}

// ============================================================================
// LENSING HOLE
// ============================================================================

pub struct LensingHolePlugin;

impl Plugin for LensingHolePlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(material::MaterialsPlugin);
        app.register_type::<LensingHole>();
        app.add_systems(
            Update,
            (spawn_lensing_hole_meshes, update_lensing_hole_time),
        );
    }
}

/// Gravitational lensing hole component for the level-ending cinematic.
///
/// Renders a Schwarzschild-style lens that distorts a sampled background
/// (typically the pixel-perfect canvas) plus a photon-ring glow and a solid
/// black event horizon disc. The `size` field (0..1) scales the whole effect
/// and is animated by the CPU each frame during Growing/Collapsing.
#[derive(Component, Debug, Clone, Reflect)]
#[reflect(Component)]
pub struct LensingHole {
    /// Schwarzschild radius in aspect-corrected screen units. Controls the inner black disc.
    pub shadow_radius: f32,
    /// Deflection strength. Higher = more visible UV warping at the horizon edge.
    pub lensing_strength: f32,
    /// Current normalized visual size (0 = invisible, 1 = full quad). Animated by CPU.
    pub size: f32,
    /// Gaussian width of the photon ring glow.
    pub photon_ring_width: f32,
    /// Peak brightness of the photon ring.
    pub photon_ring_intensity: f32,
    /// Photon ring color (linear RGBA).
    pub photon_ring_color: [f32; 4],
    /// Event horizon color (linear RGBA).
    pub black_color: [f32; 4],
    /// Background texture sampled by the lens. In game use, set this to the
    /// pixel-perfect canvas image handle so the hole distorts the actual world.
    pub background: Option<Handle<Image>>,
    /// Pixelation grid: number of cells across the screen. `0.0` disables
    /// pixelation; higher values produce chunkier blocks.
    pub pixel_grid: f32,
    /// Quad-to-viewport UV scale, per axis (the lens quad's world size divided
    /// by the visible viewport's world size). The lens samples its background
    /// using the mesh UV, so this is what keeps the hole centered on the camera
    /// and the distortion aligned with the captured scene independent of window
    /// resolution or camera viewport offset. Set this each frame from the
    /// camera projection; defaults to `Vec2::ONE` (quad == viewport).
    pub uv_scale: Vec2,
    /// Lens center in viewport UV space. `(0.5, 0.5)` (the default) centers the
    /// hole on screen; shifting it moves the event horizon, photon ring, and
    /// sampled distortion together. Content the lens deflects off the captured
    /// viewport stays transparent, so only on-screen pixels are distorted.
    pub hole_center: Vec2,
}

impl Default for LensingHole {
    fn default() -> Self {
        Self {
            shadow_radius: 0.12,
            lensing_strength: 0.05,
            size: 0.0,
            photon_ring_width: 0.02,
            photon_ring_intensity: 1.2,
            photon_ring_color: [0.6, 0.8, 1.0, 1.0],
            black_color: [0.0, 0.0, 0.0, 1.0],
            background: None,
            pixel_grid: 0.0,
            uv_scale: Vec2::ONE,
            hole_center: Vec2::splat(0.5),
        }
    }
}

#[cfg(feature = "render_2d")]
#[derive(Component)]
struct LensingHoleMesh;

#[cfg(feature = "render_2d")]
fn spawn_lensing_hole_meshes(
    mut commands: Commands,
    mut materials: ResMut<Assets<LensingHoleMaterial>>,
    quad_mesh: Option<Res<HoleQuadMesh>>,
    mut meshes: ResMut<Assets<Mesh>>,
    query: Query<
        (Entity, &LensingHole, &Transform, Option<&HoleQuantization>),
        Without<LensingHoleMesh>,
    >,
) {
    if query.is_empty() {
        return;
    }
    let mesh_handle = hole_quad_mesh(&mut commands, &mut meshes, quad_mesh.as_deref());
    for (entity, hole, transform, quantization) in &query {
        let uniforms = LensingHoleUniforms {
            shadow_radius: hole.shadow_radius,
            lensing_strength: hole.lensing_strength,
            size: hole.size,
            time: 0.0,
            photon_ring_width: hole.photon_ring_width,
            photon_ring_intensity: hole.photon_ring_intensity,
            pixel_grid: hole.pixel_grid,
            _pad1: 0.0,
            uv_scale: hole.uv_scale,
            hole_center: hole.hole_center,
            photon_ring_color: Vec4::from_array(hole.photon_ring_color),
            black_color: Vec4::from_array(hole.black_color),
        };

        let quantization_uniforms = quantization.map(|q| q.to_uniforms()).unwrap_or_default();

        let material = materials.add(LensingHoleMaterial {
            uniforms,
            quantization: quantization_uniforms,
            background: hole.background.clone(),
        });

        commands.entity(entity).insert((
            LensingHoleMesh,
            Mesh2d(mesh_handle.clone()),
            MeshMaterial2d(material),
            Transform {
                translation: transform.translation,
                rotation: transform.rotation,
                scale: transform.scale,
            },
        ));
    }
}

#[cfg(not(feature = "render_2d"))]
fn spawn_lensing_hole_meshes() {}

#[cfg(feature = "render_2d")]
fn update_lensing_hole_time(
    time: Res<Time>,
    mut materials: ResMut<Assets<LensingHoleMaterial>>,
    query: Query<(&LensingHole, &MeshMaterial2d<LensingHoleMaterial>)>,
) {
    let elapsed = time.elapsed_secs();

    for (hole, material_handle) in &query {
        if let Some(material) = materials.get_mut(&material_handle.0) {
            material.uniforms.time = elapsed;
            material.uniforms.size = hole.size;
            material.uniforms.uv_scale = hole.uv_scale;
            material.uniforms.hole_center = hole.hole_center;
        }
    }
}

#[cfg(not(feature = "render_2d"))]
fn update_lensing_hole_time() {}
