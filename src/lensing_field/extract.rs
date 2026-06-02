// Extraction of lensing-field data from the main world into the render world.
//
// `ExtractedLensingField` is a render-world resource rebuilt each frame from
// the main-world `LensingFieldTextures` and `LensingFieldSettings`. The render
// node reads it to bind the velocity textures and dispatch the compute passes.

use bevy::{
    prelude::*,
    render::extract_resource::{ExtractResource, ExtractResourcePlugin},
};

use crate::{
    ColorQuantizeUniforms, LensData, MAX_LENSES,
    lensing_field::{
        LensingFieldSettings,
        sources::{DeflectionSource, MAX_DEFLECTION_SOURCES},
        textures::LensingFieldTextures,
    },
};

/// Render-world snapshot of the field settings, source data, visual lens data,
/// and texture handles, extracted once per frame from the main world.
///
/// Two parallel data sets ride along: the `sources` array feeds the inject
/// compute pass (the deflection field); the `lens_*` arrays feed the display
/// pass (the photon ring / event-horizon disc). The black hole appears in both;
/// shapes that draw no disc (rings, lines) appear only in `sources`.
#[derive(Resource, Clone)]
pub struct ExtractedLensingField {
    pub force_scale: f32,
    pub decay: f32,
    pub dt: f32,
    /// Shape-tagged deflection sources for the inject pass.
    pub sources: [DeflectionSource; MAX_DEFLECTION_SOURCES],
    /// Number of populated entries in `sources`.
    pub source_count: u32,
    pub lens_center_size_shadow: [Vec4; MAX_LENSES],
    pub lens_strength_ring: [Vec4; MAX_LENSES],
    pub lens_photon_ring_color: [Vec4; MAX_LENSES],
    pub lens_black_color: [Vec4; MAX_LENSES],
    pub canvas_center_extent: Vec4,
    /// Palette for quantizing the photon-ring / shadow-edge region of the
    /// display pass. `palette_size == 0` disables quantization.
    pub ring_quantization: ColorQuantizeUniforms,
    pub velocity_read: Handle<Image>,
    pub velocity_write: Handle<Image>,
    /// Number of populated entries in the visual `lens_*` arrays.
    pub lens_count: u32,
}

impl Default for ExtractedLensingField {
    fn default() -> Self {
        Self {
            force_scale: 1.0,
            decay: 0.0,
            dt: 0.016,
            sources: [DeflectionSource::default(); MAX_DEFLECTION_SOURCES],
            source_count: 0,
            lens_center_size_shadow: [Vec4::ZERO; MAX_LENSES],
            lens_strength_ring: [Vec4::ZERO; MAX_LENSES],
            lens_photon_ring_color: [Vec4::ZERO; MAX_LENSES],
            lens_black_color: [Vec4::ZERO; MAX_LENSES],
            canvas_center_extent: Vec4::new(0.0, 0.0, 1.0, 1.0),
            ring_quantization: ColorQuantizeUniforms::default(),
            velocity_read: Handle::default(),
            velocity_write: Handle::default(),
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
    /// Shape-tagged deflection sources for the inject pass. Filled by
    /// `drive_lensing` (black-hole `Lens` sources) then `pack_deflection_sources`
    /// (generic shapes).
    pub sources: [DeflectionSource; MAX_DEFLECTION_SOURCES],
    pub source_count: u32,
    pub lens_count: u32,
    pub lens_center_size_shadow: [Vec4; MAX_LENSES],
    pub lens_strength_ring: [Vec4; MAX_LENSES],
    pub lens_photon_ring_color: [Vec4; MAX_LENSES],
    pub lens_black_color: [Vec4; MAX_LENSES],
    pub canvas_center_extent: Vec4,
    pub ring_quantization: ColorQuantizeUniforms,
}

impl Default for LensingFieldExtractSource {
    fn default() -> Self {
        Self {
            settings: LensingFieldSettings::default(),
            textures: None,
            sources: [DeflectionSource::default(); MAX_DEFLECTION_SOURCES],
            source_count: 0,
            lens_count: 0,
            lens_center_size_shadow: [Vec4::ZERO; MAX_LENSES],
            lens_strength_ring: [Vec4::ZERO; MAX_LENSES],
            lens_photon_ring_color: [Vec4::ZERO; MAX_LENSES],
            lens_black_color: [Vec4::ZERO; MAX_LENSES],
            canvas_center_extent: Vec4::new(0.0, 0.0, 1.0, 1.0),
            ring_quantization: ColorQuantizeUniforms::default(),
        }
    }
}

impl LensingFieldExtractSource {
    /// Overwrites the deflection-source array with `sources`, capping at
    /// [`MAX_DEFLECTION_SOURCES`] and `warn!`-ing on any drop so a full array
    /// never silently reads as complete coverage.
    pub fn set_sources(&mut self, sources: &[DeflectionSource]) {
        let count = sources.len().min(MAX_DEFLECTION_SOURCES);
        if sources.len() > MAX_DEFLECTION_SOURCES {
            warn!(
                "deflection source array full ({MAX_DEFLECTION_SOURCES}); dropped {} source(s)",
                sources.len() - MAX_DEFLECTION_SOURCES
            );
        }
        self.sources = [DeflectionSource::default(); MAX_DEFLECTION_SOURCES];
        self.sources[..count].copy_from_slice(&sources[..count]);
        self.source_count = count as u32;
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
            rebuild_extracted_lensing_field.run_if(resource_exists::<LensingFieldExtractSource>),
        );
    }
}

fn rebuild_extracted_lensing_field(
    source: Res<LensingFieldExtractSource>,
    mut extracted: ResMut<ExtractedLensingField>,
) {
    let Some(ref textures) = source.textures else {
        extracted.lens_count = 0;
        extracted.source_count = 0;
        return;
    };

    *extracted = ExtractedLensingField {
        force_scale: source.settings.force_scale,
        decay: source.settings.decay,
        dt: source.settings.dt,
        sources: source.sources,
        source_count: source.source_count,
        lens_center_size_shadow: source.lens_center_size_shadow,
        lens_strength_ring: source.lens_strength_ring,
        lens_photon_ring_color: source.lens_photon_ring_color,
        lens_black_color: source.lens_black_color,
        canvas_center_extent: source.canvas_center_extent,
        ring_quantization: source.ring_quantization,
        velocity_read: textures.velocity_read().clone(),
        velocity_write: textures.velocity_write().clone(),
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
    ring_quantization: ColorQuantizeUniforms,
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
    source.ring_quantization = ring_quantization;

    let mut css = [Vec4::ZERO; MAX_LENSES];
    let mut sr = [Vec4::ZERO; MAX_LENSES];
    let mut ring = [Vec4::ZERO; MAX_LENSES];
    let mut black = [Vec4::ZERO; MAX_LENSES];
    for (i, lens) in lenses.iter().take(MAX_LENSES).enumerate() {
        css[i] = lens.center_size_shadow;
        sr[i] = lens.strength_ring;
        ring[i] = lens.photon_ring_color;
        // Pre-snap the solid shadow fill into the ring palette here, so the
        // event-horizon interior writes a constant color in the display shader
        // rather than running the per-pixel palette match across the whole
        // screen when the hole grows to full coverage.
        let snapped = ring_quantization.nearest_palette_color(lens.black_color.truncate());
        black[i] = snapped.extend(lens.black_color.w);
    }
    source.lens_center_size_shadow = css;
    source.lens_strength_ring = sr;
    source.lens_photon_ring_color = ring;
    source.lens_black_color = black;
}
