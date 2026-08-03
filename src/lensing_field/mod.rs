// Lensing-field flow simulation plugin.
//
// Runs a per-frame compute pipeline on the GPU that produces a 2-channel
// velocity texture (the "vector map") sampled by the gravitational-lensing
// fragment shader for its deflection offsets.
//
// Pipeline (per frame, 512×512 grid):
//   1. inject — decay previous velocity, add per-cell Schwarzschild force
//   2. advect — semi-Lagrangian self-advection
//
// Pressure projection is intentionally omitted: lens deflection is a gradient
// field, and projecting it to be divergence-free cancels the radial signal.

pub(crate) mod display;
pub(crate) mod extract;
pub(crate) mod node;
pub(crate) mod pipelines;
pub mod sources;
pub(crate) mod textures;

pub use textures::LensingFieldTextures;

use bevy::prelude::*;

use crate::lensing_field::{
    display::LensingDisplayPlugin, extract::LensingFieldExtractPlugin,
    node::LensingFieldNodePlugin, pipelines::LensingFieldPipelinesPlugin,
};

/// Grid resolution of the velocity field (one axis). The grid is always
/// square: the total texture size is `LENSING_FIELD_RES × LENSING_FIELD_RES`.
pub const LENSING_FIELD_RES: u32 = 512;

/// Runtime-tunable parameters for the lensing-field simulation.
///
/// Insert this resource to override defaults. Changes take effect on the
/// next frame; the GPU-side dispatch reads the extracted copy.
#[derive(Resource, Clone, Reflect)]
#[reflect(Resource)]
pub struct LensingFieldSettings {
    /// Multiplier on the Schwarzschild force injected each frame.
    pub force_scale: f32,
    /// Per-frame velocity decay multiplier applied during inject.
    /// `0.0` = no temporal accumulation (fresh field every frame).
    /// `1.0` would accumulate without bound — keep well below.
    pub decay: f32,
    /// Semi-Lagrangian advection time step.
    pub dt: f32,
    /// Falloff exponent shared by every `Lens` deflection source: how fast the
    /// bend decays from a lens's inner radius out to its reach.
    ///
    /// `1.0` is the physical inverse-distance law; below `1.0` the bend carries
    /// further out, above it concentrates near the inner radius. This is one
    /// global shape knob rather than a per-lens field, because the profile is
    /// normalised against it — every lens reads `1.0` at its inner radius and
    /// `0.0` at its reach whatever this is set to. Retuning it therefore
    /// reshapes every lens at once without recalibrating any of them, and an
    /// individual effect only ever authors its own reach.
    pub lens_falloff: f32,
    /// Width, in screen pixels, of the mirrored guard band outside the rendered
    /// viewport. A deflected sample that lands off-screen reads the scene
    /// mirrored back across the nearest edge instead of the clamped border texel
    /// (which otherwise smears into a band). `0.0` disables mirroring and leaves
    /// off-screen samples clamped.
    pub edge_mirror_px: f32,
    /// Linear-RGBA color blended in beyond the mirrored guard band, where a
    /// sample lands too far off-screen for mirroring to stay plausible. The
    /// alpha is the blend strength at full overshoot; `0.0` alpha keeps the
    /// mirror (or clamp, when `edge_mirror_px` is `0.0`) with no color fallback.
    pub edge_fallback_color: Vec4,
}

impl Default for LensingFieldSettings {
    fn default() -> Self {
        Self {
            force_scale: 1.0,
            decay: 0.0,
            dt: 0.016,
            lens_falloff: 1.0,
            edge_mirror_px: 16.0,
            edge_fallback_color: Vec4::ZERO,
        }
    }
}

/// Plugin that manages the GPU lensing-field simulation.
///
/// Add this alongside [`crate::LensingHolePlugin`]; it inserts the compute
/// render-graph node and manages the velocity texture pair.
pub struct LensingFieldPlugin;

impl Plugin for LensingFieldPlugin {
    fn build(&self, app: &mut App) {
        app.register_type::<LensingFieldSettings>();
        app.init_resource::<LensingFieldSettings>();
        app.register_type::<sources::LightDeflector>();
        app.register_type::<sources::DeflectionShape>();
        app.add_message::<sources::LightDeflectionRequest>();
        app.add_plugins((
            LensingFieldExtractPlugin,
            LensingFieldNodePlugin,
            LensingFieldPipelinesPlugin,
            LensingDisplayPlugin,
        ));

        app.add_systems(Startup, allocate_field_textures);
    }
}

fn allocate_field_textures(mut commands: Commands, mut images: ResMut<Assets<Image>>) {
    let tex = textures::allocate_lensing_textures(&mut images);
    commands.insert_resource(tex);
}
