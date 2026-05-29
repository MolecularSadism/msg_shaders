// Black Hole Material
// Based on Eric Bruneton's black_hole_shader (BSD-3-Clause)
// https://github.com/ebruneton/black_hole_shader

use bevy::{
    asset::embedded_asset,
    prelude::*,
    render::render_resource::{AsBindGroup, ShaderType},
    shader::ShaderRef,
};

#[cfg(feature = "render_2d")]
use bevy::sprite_render::{AlphaMode2d, Material2d, Material2dPlugin};

use crate::quantize_material::ColorQuantizeUniforms;

#[derive(Asset, TypePath, AsBindGroup, Clone)]
pub struct BlackHoleMaterial {
    #[uniform(0)]
    pub uniforms: BlackHoleUniforms,
    #[uniform(1)]
    pub quantization: ColorQuantizeUniforms,
}

/// Shader uniforms - 16-byte aligned for GPU compatibility.
///
/// # Geometry Parameters (normalized UV space)
/// - `shadow_radius`: Schwarzschild radius - the event horizon boundary
/// - `disk_inner_ratio`: Inner Stable Circular Orbit (ISCO) as multiple of shadow_radius
/// - `disk_outer_ratio`: Outer accretion disc edge as multiple of shadow_radius
///
/// # Photon Sphere Parameters
/// - `photon_ring_width`: Gaussian width of the photon ring glow
/// - `photon_ring_intensity`: Peak brightness of photon ring
///
/// # Relativistic Effects
/// - `doppler_strength`: Intensity of relativistic Doppler beaming (0.0 = none, 1.0 = full)
/// - `cloud_density`: Accretion disc matter density (affects particle visibility threshold)
#[derive(Clone, Copy, ShaderType)]
pub struct BlackHoleUniforms {
    // Row 1: Core dynamics (16 bytes)
    pub spin: f32,
    pub inclination: f32,
    pub time: f32,
    pub shadow_radius: f32,

    // Row 2: Disk geometry (16 bytes)
    pub disk_inner_ratio: f32,
    pub disk_outer_ratio: f32,
    pub photon_ring_width: f32,
    pub photon_ring_intensity: f32,

    // Row 3: Effects (16 bytes)
    pub doppler_strength: f32,
    pub cloud_density: f32,
    pub axial_inner_ratio: f32,
    pub axial_outer_ratio: f32,

    // Row 4: Secondary ring parameters (16 bytes)
    pub secondary_brightness: f32,
    /// Centered-UV scale factor. Applied as `uv = (in.uv - 0.5) * outer_scale`
    /// so the visible disc fills the quad even when the quad has been
    /// shrunk on the CPU to match the disc's bounding extent. Equal to
    /// `2.0 * shadow_radius * disk_outer_ratio * disc_padding`.
    pub outer_scale: f32,
    /// Pixelation grid: number of cells across the quad's UV `[0, 1]` range.
    /// `0.0` disables pixelation. Snaps the centered UV before any disc math
    /// so the whole effect renders as discrete blocks.
    pub pixel_grid: f32,
    pub _pad2: f32,

    // Row 5-9: Colors (80 bytes)
    pub disk_inner_color: Vec4,
    pub disk_mid_color: Vec4,
    pub disk_outer_color: Vec4,
    pub glow_color: Vec4,
    pub black_color: Vec4,
}

impl Default for BlackHoleUniforms {
    fn default() -> Self {
        Self {
            spin: 0.5,
            inclination: 0.3,
            time: 0.0,
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
            outer_scale: 2.0,
            pixel_grid: 0.0,
            _pad2: 0.0,
            disk_inner_color: Vec4::new(1.0, 1.0, 0.8, 1.0),
            disk_mid_color: Vec4::new(1.0, 0.6, 0.2, 1.0),
            disk_outer_color: Vec4::new(0.8, 0.2, 0.1, 1.0),
            glow_color: Vec4::new(1.0, 0.9, 0.7, 1.0),
            black_color: Vec4::new(0.0, 0.0, 0.0, 1.0),
        }
    }
}

#[cfg(feature = "render_3d")]
impl Material for BlackHoleMaterial {
    fn vertex_shader() -> ShaderRef {
        "embedded://msg_shaders/shaders/blackhole.vert".into()
    }

    fn fragment_shader() -> ShaderRef {
        "embedded://msg_shaders/shaders/blackhole.frag".into()
    }

    fn alpha_mode(&self) -> AlphaMode {
        AlphaMode::Blend
    }
}

#[cfg(feature = "render_2d")]
impl Material2d for BlackHoleMaterial {
    fn fragment_shader() -> ShaderRef {
        "embedded://msg_shaders/shaders/blackhole_2d.wgsl".into()
    }

    fn alpha_mode(&self) -> AlphaMode2d {
        AlphaMode2d::Blend
    }
}

/// Maximum number of lenses combined in a single [`LensingMaterial`] pass.
///
/// The CPU gather culls (frustum + size threshold) and caps the active set to
/// this many; the shader loops only over `count <= MAX_LENSES`, so empty slots
/// cost nothing. At 64 B/lens this is a 4 KB uniform, well under the 64 KB limit.
/// Kept in sync with the `array<LensData, …>` size in `lensing_2d.wgsl`.
pub const MAX_LENSES: usize = 64;

/// One lens packed for the combined-field shader. Four 16-byte rows (64 B).
///
/// Components are packed into `vec4`s to keep a tight, alignment-friendly layout
/// for the uniform array; the shader unpacks them by field index.
#[derive(Clone, Copy, ShaderType)]
pub struct LensData {
    /// `xy` = world center, `z` = halo radius (`size`), `w` = `shadow_radius`
    /// (event-horizon radius as a fraction of the halo).
    pub center_size_shadow: Vec4,
    /// `x` = lensing strength, `y` = photon ring width, `z` = photon ring
    /// intensity, `w` = padding.
    pub strength_ring: Vec4,
    pub photon_ring_color: Vec4,
    pub black_color: Vec4,
}

impl Default for LensData {
    fn default() -> Self {
        Self {
            center_size_shadow: Vec4::new(0.0, 0.0, 0.0, 0.12),
            strength_ring: Vec4::ZERO,
            photon_ring_color: Vec4::ZERO,
            black_color: Vec4::new(0.0, 0.0, 0.0, 1.0),
        }
    }
}

/// Uniforms for the combined multi-lens shader.
///
/// All lenses share one scene-capture canvas, so the canvas geometry is stored
/// once. Each fragment sums every active lens's Schwarzschild deflection into a
/// single flow field, then samples the canvas once — so overlapping lenses warp
/// the space between them mutually rather than stacking independent distortions.
#[derive(Clone, Copy, ShaderType)]
pub struct LensingUniforms {
    /// World-space center of the square scene-capture canvas (camera position).
    pub canvas_center: Vec2,
    /// World-space side lengths of the square canvas (viewport diagonal).
    pub canvas_extent: Vec2,
    /// Number of populated entries in `lenses`. The shader loops `0..count`.
    pub count: u32,
    pub lenses: [LensData; MAX_LENSES],
}

impl Default for LensingUniforms {
    fn default() -> Self {
        Self {
            canvas_center: Vec2::ZERO,
            canvas_extent: Vec2::ONE,
            count: 0,
            lenses: [LensData::default(); MAX_LENSES],
        }
    }
}

/// Material for the combined gravitational-lensing pass.
///
/// A single canvas-covering quad carries this material; its uniforms hold an
/// array of lenses gathered from all `LensingHole` entities each frame.
/// The velocity field texture drives background deflection; see
/// `LensingFieldPlugin` for how it is produced.
#[derive(Asset, TypePath, AsBindGroup, Clone, Default)]
pub struct LensingMaterial {
    #[uniform(0)]
    pub uniforms: LensingUniforms,
    #[uniform(1)]
    pub quantization: ColorQuantizeUniforms,
    /// Background to distort (the scene-capture image). Nearest-filtered to
    /// preserve the canvas's pixel grid.
    #[texture(2)]
    #[sampler(3)]
    pub background: Option<Handle<Image>>,
    /// Velocity (deflection) field produced by the GPU fluid simulation.
    /// Each texel stores a 2D world-space deflection offset (RG channels).
    /// When `None` (e.g. first frame before the simulation initialises),
    /// the fragment shader falls back to zero deflection.
    #[texture(4)]
    #[sampler(5)]
    pub velocity_field: Option<Handle<Image>>,
}

#[cfg(feature = "render_2d")]
impl Material2d for LensingMaterial {
    fn fragment_shader() -> ShaderRef {
        "embedded://msg_shaders/shaders/lensing_2d.wgsl".into()
    }

    // The single canvas-covering quad writes every canvas pixel exactly once, so
    // no in-canvas blending is required. Writing the fragment's straight (un-
    // premultiplied) color and alpha through keeps the alpha composite for the
    // canvas sprite's own blend over the world. Blending here instead would
    // premultiply the stored color, which the sprite then multiplies by alpha a
    // second time — darkening soft edges into a grey halo.
    fn alpha_mode(&self) -> AlphaMode2d {
        AlphaMode2d::Opaque
    }
}

/// Registers the embedded shaders and material plugins used by both
/// `BlackHole` and `LensingHole`. Safe to add from multiple parent plugins:
/// `is_unique` is false and the inner `Material*Plugin` adds are guarded by
/// `is_plugin_added` checks.
pub(crate) struct MaterialsPlugin;

impl Plugin for MaterialsPlugin {
    fn build(&self, app: &mut App) {
        app.register_type::<crate::HoleQuantization>();

        // The hole shaders `#import` both function modules; make sure the
        // shaders that define them are registered. Both plugins guard their
        // own internal adds, so this is safe alongside an app that also adds
        // them directly.
        if !app.is_plugin_added::<crate::ColorQuantizationPlugin>() {
            app.add_plugins(crate::ColorQuantizationPlugin);
        }
        if !app.is_plugin_added::<crate::PixelationPlugin>() {
            app.add_plugins(crate::PixelationPlugin);
        }

        #[cfg(feature = "render_3d")]
        {
            embedded_asset!(app, "shaders/blackhole.vert");
            embedded_asset!(app, "shaders/blackhole.frag");
            if !app.is_plugin_added::<MaterialPlugin<BlackHoleMaterial>>() {
                app.add_plugins(MaterialPlugin::<BlackHoleMaterial>::default());
            }
        }

        #[cfg(feature = "render_2d")]
        {
            embedded_asset!(app, "shaders/blackhole_2d.wgsl");
            if !app.is_plugin_added::<Material2dPlugin<BlackHoleMaterial>>() {
                app.add_plugins(Material2dPlugin::<BlackHoleMaterial>::default());
            }
            embedded_asset!(app, "shaders/lensing_2d.wgsl");
            if !app.is_plugin_added::<Material2dPlugin<LensingMaterial>>() {
                app.add_plugins(Material2dPlugin::<LensingMaterial>::default());
            }
            // Compute shaders for the lensing-field fluid simulation.
            embedded_asset!(app, "shaders/lensing_field_inject.wgsl");
            embedded_asset!(app, "shaders/lensing_field_advect.wgsl");
            embedded_asset!(app, "shaders/lensing_field_divergence.wgsl");
            embedded_asset!(app, "shaders/lensing_field_pressure_jacobi.wgsl");
            embedded_asset!(app, "shaders/lensing_field_gradient_subtract.wgsl");
        }
    }

    fn is_unique(&self) -> bool {
        false
    }
}
