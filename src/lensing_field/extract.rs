// Extraction of lensing-field data from the main world into the render world.
//
// `ExtractedLensingField` is a render-world resource rebuilt each frame from
// the main-world `LensingFieldTextures` and `LensingFieldSettings`.  The render
// node reads it to decide how many Jacobi iterations to run and which texture
// handles to bind.

use bevy::{
    prelude::*,
    render::extract_resource::{ExtractResource, ExtractResourcePlugin},
};

use crate::{
    LensData, MAX_LENSES,
    lensing_field::{LensingFieldSettings, textures::LensingFieldTextures},
};

/// Render-world snapshot of the field settings, lens data, and texture handles,
/// extracted once per frame from the main world.
#[derive(Resource, Clone)]
pub struct ExtractedLensingField {
    /// Jacobi pressure-solve iterations this frame.
    pub jacobi_iters: u32,
    /// Force injection scale applied in the inject pass.
    pub force_scale: f32,
    /// Per-frame velocity decay multiplier (applied during inject).
    pub decay: f32,
    /// Semi-Lagrangian advection time step.
    pub dt: f32,
    /// Per-lens packed data: `xy` = world center, `z` = halo radius, `w` = shadow_radius.
    pub lens_center_size_shadow: [Vec4; MAX_LENSES],
    /// Per-lens packed data: `x` = lensing_strength, `y` = ring_width, `z` = ring_intensity, `w` =
    /// pad.
    pub lens_strength_ring: [Vec4; MAX_LENSES],
    /// Canvas center (xy) and extent (zw) in world space.
    pub canvas_center_extent: Vec4,
    /// Velocity texture the fragment shader reads this frame.
    pub velocity_read: Handle<Image>,
    /// Velocity texture the compute shader writes this frame.
    pub velocity_write: Handle<Image>,
    /// Pressure ping texture.
    pub pressure_ping: Handle<Image>,
    /// Pressure pong texture.
    pub pressure_pong: Handle<Image>,
    /// Divergence scratch texture.
    pub divergence: Handle<Image>,
    /// Number of active lenses (0 = skip dispatch).
    pub lens_count: u32,
}

impl Default for ExtractedLensingField {
    fn default() -> Self {
        Self {
            jacobi_iters: 8,
            force_scale: 1.0,
            decay: 0.98,
            dt: 0.016,
            lens_center_size_shadow: [Vec4::ZERO; MAX_LENSES],
            lens_strength_ring: [Vec4::ZERO; MAX_LENSES],
            canvas_center_extent: Vec4::new(0.0, 0.0, 1.0, 1.0),
            velocity_read: Handle::default(),
            velocity_write: Handle::default(),
            pressure_ping: Handle::default(),
            pressure_pong: Handle::default(),
            divergence: Handle::default(),
            lens_count: 0,
        }
    }
}

/// Wrapper so `LensingFieldExtractSource` can be used with `ExtractResourcePlugin`.
///
/// Pairs `LensingFieldTextures` + `LensingFieldSettings` plus the per-frame lens
/// array from `drive_lensing` in the main world, writing them together into one
/// render-world resource.
#[derive(Resource, Clone)]
pub struct LensingFieldExtractSource {
    pub settings: LensingFieldSettings,
    pub textures: Option<LensingFieldTextures>,
    pub lens_count: u32,
    pub lens_center_size_shadow: [Vec4; MAX_LENSES],
    pub lens_strength_ring: [Vec4; MAX_LENSES],
    pub canvas_center_extent: Vec4,
}

impl Default for LensingFieldExtractSource {
    fn default() -> Self {
        Self {
            settings: LensingFieldSettings::default(),
            textures: None,
            lens_count: 0,
            lens_center_size_shadow: [Vec4::ZERO; MAX_LENSES],
            lens_strength_ring: [Vec4::ZERO; MAX_LENSES],
            canvas_center_extent: Vec4::new(0.0, 0.0, 1.0, 1.0),
        }
    }
}

impl ExtractResource for LensingFieldExtractSource {
    type Source = LensingFieldExtractSource;

    fn extract_resource(source: &Self::Source) -> Self {
        source.clone()
    }
}

/// Plugin that registers the extraction pipeline.
pub(crate) struct LensingFieldExtractPlugin;

impl Plugin for LensingFieldExtractPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<LensingFieldExtractSource>();
        app.add_plugins(ExtractResourcePlugin::<LensingFieldExtractSource>::default());

        // Populate `ExtractedLensingField` in the render world from the
        // extracted source.  This system runs in `ExtractSchedule`, so both the
        // render-world resource and the GPU textures are ready before Prepare.
        let Some(render_app) = app.get_sub_app_mut(bevy::render::RenderApp) else {
            return;
        };

        render_app.init_resource::<ExtractedLensingField>();
        render_app.add_systems(
            bevy::render::ExtractSchedule,
            rebuild_extracted_lensing_field,
        );
    }
}

fn rebuild_extracted_lensing_field(
    source: Res<LensingFieldExtractSource>,
    mut extracted: ResMut<ExtractedLensingField>,
) {
    let Some(ref textures) = source.textures else {
        extracted.lens_count = 0;
        return;
    };

    *extracted = ExtractedLensingField {
        jacobi_iters: source.settings.jacobi_iters,
        force_scale: source.settings.force_scale,
        decay: source.settings.decay,
        dt: source.settings.dt,
        lens_center_size_shadow: source.lens_center_size_shadow,
        lens_strength_ring: source.lens_strength_ring,
        canvas_center_extent: source.canvas_center_extent,
        velocity_read: textures.velocity_read().clone(),
        velocity_write: textures.velocity_write().clone(),
        pressure_ping: textures.pressure_ping.clone(),
        pressure_pong: textures.pressure_pong.clone(),
        divergence: textures.divergence.clone(),
        lens_count: source.lens_count,
    };
}

// ── Helper called by drive_lensing ───────────────────────────────────────────

/// Populates the extract source from the gathered lens array.
///
/// Called from `drive_lensing` each frame after culling and sorting survivors.
/// `canvas_center` and `canvas_extent` are in world units.
pub fn update_lensing_field_source(
    source: &mut LensingFieldExtractSource,
    settings: &LensingFieldSettings,
    textures: Option<&LensingFieldTextures>,
    lenses: &[LensData],
    canvas_center: Vec2,
    canvas_extent: Vec2,
) {
    source.settings = settings.clone();
    source.textures = textures.cloned();
    source.lens_count = lenses.len().min(MAX_LENSES) as u32;
    source.canvas_center_extent = Vec4::new(
        canvas_center.x,
        canvas_center.y,
        canvas_extent.x,
        canvas_extent.y,
    );

    let mut css = [Vec4::ZERO; MAX_LENSES];
    let mut sr = [Vec4::ZERO; MAX_LENSES];
    for (i, lens) in lenses.iter().take(MAX_LENSES).enumerate() {
        css[i] = lens.center_size_shadow;
        sr[i] = lens.strength_ring;
    }
    source.lens_center_size_shadow = css;
    source.lens_strength_ring = sr;
}
