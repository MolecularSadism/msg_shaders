// msg_shaders - MSG studio shader effects for Bevy.
// Black hole based on Eric Bruneton's black_hole_shader (BSD-3-Clause)
// https://github.com/ebruneton/black_hole_shader

#[cfg(feature = "render_2d")]
mod lensing_field;
mod material;
mod pixelate;
mod quantize;
mod quantize_material;

use bevy::prelude::*;
use bevy::render::extract_component::ExtractComponent;
#[cfg(feature = "render_2d")]
pub use lensing_field::{
    LENSING_FIELD_RES, LensingFieldPlugin, LensingFieldSettings, LensingFieldTextures,
    display::LensingDisplayLabel,
    sources::{
        DeflectionShape, DeflectionSource, LightDeflectionRequest, LightDeflector,
        MAX_DEFLECTION_SOURCES,
    },
};
pub use material::{BlackHoleMaterial, BlackHoleUniforms, LensData, MAX_LENSES};

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
        BlackHole, BlackHoleColors, BlackHoleGeometry, BlackHoleOverlay, BlackHolePlugin,
        BlackHoleQuantization, ColorQuantizationPlugin, ColorQuantizeMaterial,
        ColorQuantizeUniforms, DitherPattern, HoleQuantization, LensingHoleCamera,
        LensingHolePlugin, MAX_PALETTE_COLORS, PixelateConfig, PixelateMaterial, PixelationPlugin,
        QuantizationConfig, QuantizePixelateMaterial, lens_capture_extent,
    };
    #[cfg(feature = "render_2d")]
    pub use super::{
        DeflectionShape, DeflectionSource, LENSING_FIELD_RES, LensingFieldPlugin,
        LensingFieldSettings, LightDeflectionRequest, LightDeflector, MAX_DEFLECTION_SOURCES,
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

/// Shared color quantization settings used by both `BlackHole` and `BlackHoleOverlay`.
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
        #[cfg(feature = "render_2d")]
        app.add_plugins(LensingFieldPlugin);
        app.register_type::<BlackHoleOverlay>();
        app.register_type::<LensingHoleCamera>();
        // `drive_lensing` seeds the deflection-source array with the black-hole
        // `Lens` sources; `pack_deflection_sources` appends the generic shapes
        // (components + one-shot requests) after it.
        #[cfg(feature = "render_2d")]
        app.add_systems(
            Update,
            (
                drive_lensing,
                lensing_field::sources::pack_deflection_sources,
            )
                .chain(),
        );
        #[cfg(not(feature = "render_2d"))]
        app.add_systems(Update, drive_lensing);
    }
}

/// Marker for the orthographic camera the lensing display renders against.
///
/// Add this to the 2D camera whose view the player sees. The deflection field
/// and the full-screen display pass read its projection and transform to map
/// world positions into the viewport, so callers only need to place the
/// black-hole entities in the world.
#[derive(Component, Debug, Clone, Copy, Reflect, Default, ExtractComponent)]
#[reflect(Component)]
pub struct LensingHoleCamera;

/// Visual photon-sphere and event-horizon hole rendered around a world point.
///
/// Draws the photon-ring glow and the solid black event-horizon disc over the
/// lit scene (the full-screen display pass), in world units around `size`. It
/// supplies the *appearance* only — the gravitational deflection that warps the
/// surrounding scene is a separate [`LightDeflector`] (a `Lens` shape) on the
/// same entity. The effect tracks the camera's translation, zoom, and rotation
/// implicitly, because the display pass works in world space.
#[derive(Component, Debug, Clone, Reflect)]
#[reflect(Component)]
pub struct BlackHoleOverlay {
    /// Event-horizon radius as a fraction of the visible halo (`size`).
    pub shadow_radius: f32,
    /// Visible halo radius, in world units (0 = invisible). Animated by the CPU.
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
    /// scene-capture image handle so the hole distorts the actual world.
    pub background: Option<Handle<Image>>,
    /// World-space center of the square scene-capture canvas (the camera's world
    /// position). Computed each frame by [`sync_lensing_hole_to_camera`].
    pub canvas_center: Vec2,
    /// World-space side lengths of the square scene-capture canvas (the viewport
    /// diagonal). Computed each frame; see [`lens_capture_extent`].
    pub canvas_extent: Vec2,
}

impl Default for BlackHoleOverlay {
    fn default() -> Self {
        Self {
            shadow_radius: 0.12,
            size: 0.0,
            photon_ring_width: 0.02,
            photon_ring_intensity: 1.2,
            photon_ring_color: [0.6, 0.8, 1.0, 1.0],
            black_color: [0.0, 0.0, 0.0, 1.0],
            background: None,
            canvas_center: Vec2::ZERO,
            canvas_extent: Vec2::ONE,
        }
    }
}

/// World-space side length of the square scene-capture canvas for a viewport of
/// the given world size. The canvas is axis-aligned and must cover the viewport
/// at every camera rotation, so it spans the viewport diagonal on both axes.
///
/// The scene-capture camera must be sized to the same square (centered on the
/// camera, no rotation) so the canvas the lens samples matches what was rendered.
pub fn lens_capture_extent(viewport_world_size: Vec2) -> Vec2 {
    Vec2::splat(viewport_world_size.length())
}

/// Gathers every visible [`BlackHoleOverlay`] into the visual lens array — the
/// photon-ring / event-horizon discs the display pass draws — and computes the
/// canvas geometry the field is sampled over.
///
/// It reads the [`LensingHoleCamera`]'s orthographic projection to derive the
/// canvas center (the camera position) and the square canvas extent (the
/// viewport diagonal, see [`lens_capture_extent`]). The deflection that warps
/// the scene is gathered separately from [`LightDeflector`] sources by
/// `pack_deflection_sources`; this only feeds the display pass.
///
/// To keep the per-fragment loop cheap with many overlays, the active set is
/// culled on the CPU: overlays with negligible size are dropped, and those whose
/// halo AABB does not intersect the canvas square are frustum-culled. Survivors
/// are sorted by influence and capped to [`MAX_LENSES`], then written into
/// `LensingFieldExtractSource` for extraction to the render world.
#[cfg(feature = "render_2d")]
fn drive_lensing(
    q_camera: Query<(&Projection, &GlobalTransform), With<LensingHoleCamera>>,
    q_holes: Query<(
        &BlackHoleOverlay,
        &GlobalTransform,
        Option<&HoleQuantization>,
    )>,
    field_textures: Option<Res<lensing_field::LensingFieldTextures>>,
    field_settings: Option<Res<lensing_field::LensingFieldSettings>>,
    mut field_source: Option<ResMut<lensing_field::extract::LensingFieldExtractSource>>,
    mut images: ResMut<Assets<Image>>,
    // Cached (palette key, LUT handle). The strong handle keeps the baked image
    // alive; it is rebuilt only when the ring palette changes.
    mut lut_cache: Local<Option<(u64, Handle<Image>)>>,
) {
    let Ok((Projection::Orthographic(ortho), camera_gt)) = q_camera.single() else {
        return;
    };

    let canvas_center = camera_gt.translation().truncate();
    let canvas_extent = lens_capture_extent(ortho.area.size());

    // Canvas AABB for frustum culling. A lens contributes only if its halo
    // (`center ± size`) overlaps this square.
    let half = canvas_extent * 0.5;
    let canvas_min = canvas_center - half;
    let canvas_max = canvas_center + half;

    // Survivors carry an influence weight for sorting when capping to MAX_LENSES.
    let mut survivors: Vec<(f32, LensData)> = Vec::new();
    // The photon ring / shadow edge can be palette-quantized; the first lens
    // carrying a `HoleQuantization` provides the palette for the whole pass.
    let mut quantization: Option<ColorQuantizeUniforms> = None;

    for (hole, gt, quant) in &q_holes {
        if hole.size <= 1e-3 {
            continue;
        }
        let center = gt.translation().truncate();
        // Halo AABB vs canvas AABB.
        if center.x + hole.size < canvas_min.x
            || center.x - hole.size > canvas_max.x
            || center.y + hole.size < canvas_min.y
            || center.y - hole.size > canvas_max.y
        {
            continue;
        }

        if quantization.is_none()
            && let Some(q) = quant
        {
            quantization = Some(q.to_uniforms());
        }

        // Z-rotation of the hole entity, used by the shader to rotate the pixel
        // grid into the hole's own frame so the pixelation spins with it.
        let rotation = gt.rotation().to_euler(EulerRot::ZYX).0;

        // Larger, brighter holes win a slot first when the cap is exceeded.
        let influence = hole.size * hole.photon_ring_intensity;
        survivors.push((
            influence,
            LensData {
                center_size_shadow: Vec4::new(center.x, center.y, hole.size, hole.shadow_radius),
                // `strength_ring.x` (deflection) is unused by the display shader;
                // the deflection comes from the entity's `LightDeflector`.
                strength_ring: Vec4::new(
                    0.0,
                    hole.photon_ring_width,
                    hole.photon_ring_intensity,
                    rotation,
                ),
                photon_ring_color: Vec4::from_array(hole.photon_ring_color),
                black_color: Vec4::from_array(hole.black_color),
            },
        ));
    }

    if survivors.len() > MAX_LENSES {
        survivors.sort_by(|a, b| b.0.total_cmp(&a.0));
        survivors.truncate(MAX_LENSES);
    }

    let survivor_lens_data: Vec<LensData> = survivors.into_iter().map(|(_, d)| d).collect();

    // Feed the visual lens array, canvas geometry, and ring palette into the
    // extract source for the display pass. The deflection field itself is driven
    // separately from `LightDeflector` sources by `pack_deflection_sources`.
    if let Some(ref mut source) = field_source {
        let quant = quantization.unwrap_or_default();

        // Rebuild the baked nearest-palette LUT only when the palette changes.
        // An empty palette is never sampled (the display shader guards on
        // `palette_size == 0`), so a tiny placeholder avoids baking the full cube.
        let key = quant.palette_key();
        let lut_handle = match lut_cache.as_ref() {
            Some((cached_key, handle)) if *cached_key == key => handle.clone(),
            _ => {
                let resolution = if quant.palette_size == 0 {
                    2
                } else {
                    quantize_material::PALETTE_LUT_RESOLUTION
                };
                let handle = images.add(quant.build_lut(resolution));
                *lut_cache = Some((key, handle.clone()));
                handle
            }
        };

        lensing_field::extract::update_lensing_field_source(
            source,
            field_settings
                .as_deref()
                .unwrap_or(&lensing_field::LensingFieldSettings::default()),
            field_textures.as_deref(),
            &survivor_lens_data,
            canvas_center,
            canvas_extent,
            quant,
            lut_handle,
        );
    }
}

#[cfg(not(feature = "render_2d"))]
fn drive_lensing() {}
