// Lensing-field fluid simulation plugin.
//
// Runs a lightweight semi-Lagrangian fluid simulation on the GPU each frame to
// produce a per-frame velocity (deflection) texture for the gravitational-
// lensing effect.  The fragment shader then samples that texture instead of
// summing analytic Schwarzschild deflections per-fragment.
//
// Pipeline (per frame, 512×512 grid):
//   1. inject      – write Schwarzschild forces as velocity, applying decay
//   2. advect      – semi-Lagrangian self-advection
//   3. divergence  – central-difference divergence of velocity
//   4. pressure    – N Jacobi iterations (ping-pong)
//   5. gradient    – subtract pressure gradient from velocity

pub(crate) mod extract;
pub(crate) mod node;
pub(crate) mod pipelines;
pub(crate) mod textures;

pub use textures::LensingFieldTextures;

use bevy::prelude::*;

use crate::lensing_field::{
    extract::LensingFieldExtractPlugin, node::LensingFieldNodePlugin,
    pipelines::LensingFieldPipelinesPlugin,
};

/// Grid resolution of the velocity field (one axis).  The grid is always
/// square: the total texture size is `LENSING_FIELD_RES × LENSING_FIELD_RES`.
///
/// Exposed as a constant so downstream code can reference it without reaching
/// into this module.
pub const LENSING_FIELD_RES: u32 = 512;

/// Runtime-tunable parameters for the lensing-field fluid simulation.
///
/// Insert this resource to override defaults.  Changes take effect on the
/// next frame; the GPU-side dispatch reads the extracted copy.
#[derive(Resource, Clone, Reflect)]
#[reflect(Resource)]
pub struct LensingFieldSettings {
    /// Number of Jacobi pressure-solve iterations per frame.
    /// More iterations → more divergence-free flow, higher GPU cost.
    pub jacobi_iters: u32,
    /// Scale applied to the Schwarzschild force when injecting it as velocity.
    pub force_scale: f32,
    /// Per-frame velocity decay multiplier (applied during the inject pass).
    /// Values close to 1.0 make the field persist longer.
    pub decay: f32,
    /// Semi-Lagrangian advection time step.
    pub dt: f32,
}

impl Default for LensingFieldSettings {
    fn default() -> Self {
        Self {
            jacobi_iters: 8,
            force_scale: 1.0,
            decay: 0.98,
            dt: 0.016,
        }
    }
}

/// Plugin that manages the GPU lensing-field fluid simulation.
///
/// Add this alongside [`crate::LensingHolePlugin`]; it inserts the compute
/// render-graph node and manages the ping-pong velocity textures.
pub struct LensingFieldPlugin;

impl Plugin for LensingFieldPlugin {
    fn build(&self, app: &mut App) {
        app.register_type::<LensingFieldSettings>();
        app.init_resource::<LensingFieldSettings>();
        app.add_plugins((
            LensingFieldExtractPlugin,
            LensingFieldNodePlugin,
            LensingFieldPipelinesPlugin,
        ));

        // Allocate the field textures on startup.
        app.add_systems(Startup, allocate_field_textures);
    }
}

fn allocate_field_textures(mut commands: Commands, mut images: ResMut<Assets<Image>>) {
    let tex = textures::allocate_lensing_textures(&mut images);
    commands.insert_resource(tex);
}
