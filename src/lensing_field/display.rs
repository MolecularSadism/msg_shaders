// Full-screen lensing display pass.
//
// Distorts the lit scene by the GPU-simulated deflection field as a
// post-process over the camera's `ViewTarget`, instead of sampling an
// offscreen scene capture onto a world-space quad. Running after the scene
// (and any lighting composited into the view) means the lens warps the final
// lit pixels at full screen resolution.
//
// Per screen pixel: reconstruct the world position from the camera's inverse
// view-projection, sample the deflection field at that world position, project
// the deflected world position back to screen space, and read the lit scene
// there. The per-lens photon-ring and event-horizon discs are drawn on top in
// world space, matching the field's source lenses.

use bevy::core_pipeline::{Core2d, Core2dSystems, FullscreenShader};
use bevy::prelude::*;
use bevy::render::camera::ExtractedCamera;
use bevy::render::extract_component::ExtractComponentPlugin;
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_resource::binding_types::{
    sampler, texture_2d, texture_3d, uniform_buffer,
};
use bevy::render::render_resource::{
    BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries, CachedPipelineState,
    CachedRenderPipelineId, ColorTargetState, ColorWrites, FilterMode, FragmentState,
    MultisampleState, Operations, PipelineCache, PrimitiveState, RenderPassColorAttachment,
    RenderPassDescriptor, RenderPipelineDescriptor, Sampler, SamplerBindingType, SamplerDescriptor,
    ShaderStages, ShaderType, SpecializedRenderPipeline, SpecializedRenderPipelines, TextureFormat,
    TextureSampleType, UniformBuffer,
};
use bevy::render::renderer::{RenderContext, RenderDevice, RenderQueue, ViewQuery};
use bevy::render::texture::GpuImage;
use bevy::render::view::{ExtractedView, ViewTarget};
use bevy::render::{Render, RenderApp, RenderSystems};
use bevy::shader::Shader;
use msg_post_process::PostProcessAppExt;

use crate::lensing_field::extract::ExtractedLensingField;
use crate::{ColorQuantizeUniforms, DiscData, LensData, LensingHoleCamera, MAX_DISCS, MAX_LENSES};

const DISPLAY_SHADER: &str = "embedded://msg_shaders/shaders/lensing_display.wgsl";

/// Per-frame uniform for the display fragment shader.
///
/// Carries the camera transforms (to map between screen and world) plus the
/// gathered lens array and canvas geometry the field was simulated over.
#[derive(Clone, ShaderType)]
struct LensingDisplayUniform {
    /// Maps world space to clip space; used to project a deflected world
    /// position back onto the screen.
    clip_from_world: Mat4,
    /// Inverse of `clip_from_world`; reconstructs the world position of a
    /// screen pixel.
    world_from_clip: Mat4,
    /// `xy` = canvas world-space center, `zw` = canvas world-space extent. The
    /// field is sampled in this canvas's UV space.
    canvas_center_extent: Vec4,
    /// Camera viewport within the render target, normalized by target size.
    /// `xy` = viewport origin / target size, `zw` = viewport size / target size.
    /// `(0, 0, 1, 1)` when no viewport is set (full-target render).
    viewport_rect: Vec4,
    /// Mirrored guard band for off-screen deflected samples. `xy` = band width
    /// per axis in full-target UV (screen pixels normalized by target size),
    /// `z` = mirror enable (`1.0` on, `0.0` off), `w` unused.
    edge_mirror: Vec4,
    /// Linear-RGBA fallback color blended in beyond the mirrored guard band.
    /// The alpha scales the blend at full overshoot.
    edge_fallback_color: Vec4,
    /// Palette quantization applied only to the photon-ring / shadow-edge band.
    /// `palette_size == 0` disables it.
    quantization: ColorQuantizeUniforms,
    /// Number of populated entries in `lenses`.
    count: u32,
    /// Physical pixels per art pixel, rounded from the camera projection.
    /// Passed as a uniform so the pixel grid keeps a fixed screen-space block
    /// size even when the projection shifts mid-animation.
    art_pixel_scale: f32,
    /// Number of populated entries in `discs`.
    disc_count: u32,
    lenses: [LensData; MAX_LENSES],
    /// `DrawAfterLensing` black holes, drawn over the warp.
    discs: [DiscData; MAX_DISCS],
}

impl Default for LensingDisplayUniform {
    fn default() -> Self {
        Self {
            clip_from_world: Mat4::IDENTITY,
            world_from_clip: Mat4::IDENTITY,
            canvas_center_extent: Vec4::new(0.0, 0.0, 1.0, 1.0),
            viewport_rect: Vec4::new(0.0, 0.0, 1.0, 1.0),
            edge_mirror: Vec4::ZERO,
            edge_fallback_color: Vec4::ZERO,
            quantization: ColorQuantizeUniforms::default(),
            count: 0,
            art_pixel_scale: 4.0,
            disc_count: 0,
            lenses: [LensData::default(); MAX_LENSES],
            discs: [DiscData::default(); MAX_DISCS],
        }
    }
}

/// Persistent GPU buffer for [`LensingDisplayUniform`], written once per frame.
#[derive(Resource, Default)]
struct LensingDisplayUniforms {
    buffer: UniformBuffer<LensingDisplayUniform>,
}

/// Builds the per-frame display uniform from the extracted lens field and the
/// player camera's view-projection.
fn prepare_lensing_display_uniforms(
    mut uniforms: ResMut<LensingDisplayUniforms>,
    extracted: Res<ExtractedLensingField>,
    views: Query<(&ExtractedView, &ExtractedCamera), With<LensingHoleCamera>>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
) {
    let (clip_from_world, viewport_rect, edge_mirror, art_pixel_scale) = views
        .iter()
        .next()
        .map(|(ev, cam)| {
            let cfworld = ev
                .clip_from_world
                .unwrap_or_else(|| ev.clip_from_view * ev.world_from_view.to_matrix().inverse());

            // The display pass renders to the full target texture (no GPU viewport
            // set on the render pass). When the camera has a sub-window viewport
            // (e.g. the dev inspector clips the game area), the projection matrices
            // are sized to the viewport, not the full target. The shader must know
            // the viewport's position and extent within the full target so it can
            // correctly map full-target UVs to viewport-relative NDC.
            let vp_rect = match (cam.physical_viewport_size, cam.physical_target_size) {
                (Some(vp_size), Some(target_size)) if target_size.x > 0 && target_size.y > 0 => {
                    let origin = cam
                        .viewport
                        .as_ref()
                        .map_or(UVec2::ZERO, |v| v.physical_position);
                    Vec4::new(
                        origin.x as f32 / target_size.x as f32,
                        origin.y as f32 / target_size.y as f32,
                        vp_size.x as f32 / target_size.x as f32,
                        vp_size.y as f32 / target_size.y as f32,
                    )
                }
                _ => Vec4::new(0.0, 0.0, 1.0, 1.0),
            };

            // Mirror guard-band width, converted from screen pixels to per-axis
            // full-target UV so the shader can measure how far a deflected
            // sample lands past the rendered region. `z` flags mirroring on.
            let mirror = match cam.physical_target_size {
                Some(target_size) if target_size.x > 0 && target_size.y > 0 => {
                    let enabled = if extracted.edge_mirror_px > 0.0 {
                        1.0
                    } else {
                        0.0
                    };
                    Vec4::new(
                        extracted.edge_mirror_px / target_size.x as f32,
                        extracted.edge_mirror_px / target_size.y as f32,
                        enabled,
                        0.0,
                    )
                }
                _ => Vec4::ZERO,
            };

            // Art-pixel scale: physical pixels per art pixel, derived from the
            // orthographic projection width and the physical viewport. Rounding
            // to an integer gives the same N that `art_pixel_size(wpp)` would
            // compute on the GPU, but pinned to the CPU-side projection so the
            // pixel grid keeps a fixed screen-space block size even if the
            // GPU-derived wpp drifts (e.g. when the projection shifts
            // mid-animation).
            let proj_scale_x = ev.clip_from_view.x_axis.x;
            let art_px = if proj_scale_x.abs() > f32::EPSILON {
                let area_width = 2.0 / proj_scale_x;
                let physical_width = cam
                    .physical_viewport_size
                    .map_or(area_width, |s| s.x as f32);
                (physical_width / area_width).round().max(1.0)
            } else {
                4.0
            };

            (cfworld, vp_rect, mirror, art_px)
        })
        .unwrap_or((
            Mat4::IDENTITY,
            Vec4::new(0.0, 0.0, 1.0, 1.0),
            Vec4::ZERO,
            4.0,
        ));

    let count = extracted.lens_count.min(MAX_LENSES as u32);
    let mut lenses = [LensData::default(); MAX_LENSES];
    for (i, slot) in lenses.iter_mut().enumerate().take(count as usize) {
        *slot = LensData {
            center_size_shadow: extracted.lens_center_size_shadow[i],
            strength_ring: extracted.lens_strength_ring[i],
            photon_ring_color: extracted.lens_photon_ring_color[i],
            black_color: extracted.lens_black_color[i],
            ring_colors: extracted.lens_ring_colors[i],
            ring_pattern: extracted.lens_ring_pattern[i],
            ring_edge: extracted.lens_ring_edge[i],
        };
    }

    uniforms.buffer.set(LensingDisplayUniform {
        clip_from_world,
        world_from_clip: clip_from_world.inverse(),
        canvas_center_extent: extracted.canvas_center_extent,
        viewport_rect,
        edge_mirror,
        edge_fallback_color: extracted.edge_fallback_color,
        quantization: extracted.ring_quantization,
        count,
        art_pixel_scale,
        disc_count: extracted.disc_count.min(MAX_DISCS as u32),
        lenses,
        discs: extracted.discs,
    });
    uniforms.buffer.write_buffer(&render_device, &render_queue);
}

/// Cached pipeline id for a view, specialized on its target format.
#[derive(Component, Clone, Copy)]
struct LensingDisplayPipelineId(CachedRenderPipelineId);

fn queue_lensing_display_pipelines(
    mut commands: Commands,
    pipeline_cache: Res<PipelineCache>,
    pipeline: Res<LensingDisplayPipeline>,
    mut specialized: ResMut<SpecializedRenderPipelines<LensingDisplayPipeline>>,
    views: Query<(Entity, &ExtractedView), With<LensingHoleCamera>>,
) {
    for (entity, view) in &views {
        let id = specialized.specialize(
            &pipeline_cache,
            &pipeline,
            LensingDisplayPipelineKey {
                target_format: view.target_format,
            },
        );
        commands.entity(entity).insert(LensingDisplayPipelineId(id));
    }
}

/// Warps the current view's target by the deflection field.
fn lensing_display_pass(
    view: ViewQuery<(&ViewTarget, &LensingDisplayPipelineId)>,
    extracted: Res<ExtractedLensingField>,
    pipeline_cache: Res<PipelineCache>,
    pipeline: Option<Res<LensingDisplayPipeline>>,
    gpu_images: Res<RenderAssets<GpuImage>>,
    uniforms: Res<LensingDisplayUniforms>,
    mut render_context: RenderContext,
) {
    let (view_target, pipeline_id) = view.into_inner();
    // No deflection sources and no hole to draw over the warp: leave the view
    // untouched (the scene passes straight through to the downstream
    // post-processing stack). Sources without a visual lens (rings, lines)
    // still warp here; they just draw no disc, because the per-lens loop is
    // keyed on the visual `lens_count`.
    if extracted.source_count == 0 && extracted.disc_count == 0 {
        return;
    }

    let Some(pipeline) = pipeline else {
        return;
    };
    // `get_render_pipeline` returns `None` both while the pipeline is still
    // compiling and after a compile error, so surface the error state — a
    // WGSL fault would otherwise make this node a silent no-op (the scene
    // passes through unwarped with no log).
    if let CachedPipelineState::Err(e) = pipeline_cache.get_render_pipeline_state(pipeline_id.0) {
        bevy::log::error!("lensing_display pipeline error: {e}");
        return;
    }
    let Some(render_pipeline) = pipeline_cache.get_render_pipeline(pipeline_id.0) else {
        return;
    };

    let Some(field) = gpu_images.get(&extracted.velocity_write) else {
        return;
    };
    // Baked palette LUT for the ring-band quantization. Missing only on the
    // first frame before the image uploads; skip the pass until it's ready
    // rather than binding a mismatched texture.
    let Some(lut) = gpu_images.get(&extracted.ring_quantization_lut) else {
        return;
    };

    let Some(uniform_binding) = uniforms.buffer.binding() else {
        return;
    };

    let post_process = view_target.post_process_write();
    let layout = pipeline_cache.get_bind_group_layout(&pipeline.layout);
    let bind_group = render_context.render_device().create_bind_group(
        "lensing_display_bind_group",
        &layout,
        &BindGroupEntries::sequential((
            post_process.source,
            &pipeline.scene_sampler,
            &field.texture_view,
            &pipeline.field_sampler,
            uniform_binding,
            &lut.texture_view,
        )),
    );

    let mut render_pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
        label: Some("lensing_display_pass"),
        color_attachments: &[Some(RenderPassColorAttachment {
            view: post_process.destination,
            resolve_target: None,
            ops: Operations::default(),
            depth_slice: None,
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    render_pass.set_render_pipeline(render_pipeline);
    render_pass.set_bind_group(0, &bind_group, &[]);
    render_pass.draw(0..3, 0..1);
}

#[derive(PartialEq, Eq, Hash, Clone, Copy)]
struct LensingDisplayPipelineKey {
    target_format: TextureFormat,
}

#[derive(Resource)]
struct LensingDisplayPipeline {
    layout: BindGroupLayoutDescriptor,
    scene_sampler: Sampler,
    field_sampler: Sampler,
    shader: Handle<Shader>,
    fullscreen_shader: FullscreenShader,
}

impl FromWorld for LensingDisplayPipeline {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();

        let layout = BindGroupLayoutDescriptor::new(
            "lensing_display_bind_group_layout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::FRAGMENT,
                (
                    // Lit scene (post-process source).
                    texture_2d(TextureSampleType::Float { filterable: true }),
                    sampler(SamplerBindingType::Filtering),
                    // Deflection field (world-space offsets per texel).
                    texture_2d(TextureSampleType::Float { filterable: true }),
                    sampler(SamplerBindingType::Filtering),
                    uniform_buffer::<LensingDisplayUniform>(false),
                    // Baked nearest-palette LUT (Rgba32Float, unfilterable),
                    // point-loaded in the shader so no sampler is bound.
                    texture_3d(TextureSampleType::Float { filterable: false }),
                ),
            ),
        );

        // Clamp to edge so deflected samples off the screen read the border
        // rather than wrapping.
        let make_sampler = |label: &'static str| {
            render_device.create_sampler(&SamplerDescriptor {
                label: Some(label),
                mag_filter: FilterMode::Linear,
                min_filter: FilterMode::Linear,
                ..Default::default()
            })
        };
        let scene_sampler = make_sampler("lensing_display_scene_sampler");
        let field_sampler = make_sampler("lensing_display_field_sampler");

        let shader = world.load_asset(DISPLAY_SHADER);
        let fullscreen_shader = FullscreenShader::from_world(world);

        Self {
            layout,
            scene_sampler,
            field_sampler,
            shader,
            fullscreen_shader,
        }
    }
}

impl SpecializedRenderPipeline for LensingDisplayPipeline {
    type Key = LensingDisplayPipelineKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        RenderPipelineDescriptor {
            label: Some("lensing_display_pipeline".into()),
            layout: vec![self.layout.clone()],
            vertex: self.fullscreen_shader.to_vertex_state(),
            fragment: Some(FragmentState {
                shader: self.shader.clone(),
                shader_defs: vec![],
                entry_point: Some("fragment".into()),
                targets: vec![Some(ColorTargetState {
                    format: key.target_format,
                    blend: None,
                    write_mask: ColorWrites::ALL,
                })],
            }),
            primitive: PrimitiveState::default(),
            depth_stencil: None,
            multisample: MultisampleState::default(),
            immediate_size: 0,
            zero_initialize_workgroup_memory: false,
        }
    }
}

/// System set holding the lensing display pass in the [`Core2d`] schedule.
///
/// Exposed so the host can order it relative to other post-process passes, e.g.
/// `app.render_after(Core2d, LensingDisplaySystems, RcCompositeSystems)` to warp the
/// lit scene.
#[derive(Debug, Hash, PartialEq, Eq, Clone, SystemSet)]
pub struct LensingDisplaySystems;

/// Registers the full-screen lensing display pass as a post-process pass.
///
/// By default it runs between the main pass and the post-processing stack. The
/// host can refine the ordering — e.g. place it after a lighting composite so it
/// warps lit pixels — via [`LensingDisplaySystems`].
pub(crate) struct LensingDisplayPlugin;

impl Plugin for LensingDisplayPlugin {
    fn build(&self, app: &mut App) {
        // The shader is embedded by `material::MaterialsPlugin`, which this
        // plugin is added alongside.
        app.add_plugins(ExtractComponentPlugin::<LensingHoleCamera>::default());
        // Default placement: a full-screen pass between the main pass and the
        // post-processing stack. The host can additionally order it after a
        // lighting pass so it warps lit pixels — see [`LensingDisplaySystems`].
        app.add_post_process_pass(Core2d, LensingDisplaySystems, lensing_display_pass);
        app.render_between(
            Core2d,
            LensingDisplaySystems,
            Core2dSystems::MainPass,
            Core2dSystems::PostProcess,
        );

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .init_resource::<LensingDisplayUniforms>()
            .init_resource::<SpecializedRenderPipelines<LensingDisplayPipeline>>()
            .add_systems(
                Render,
                (
                    prepare_lensing_display_uniforms.in_set(RenderSystems::Prepare),
                    queue_lensing_display_pipelines
                        .in_set(RenderSystems::Queue)
                        .run_if(resource_exists::<LensingDisplayPipeline>),
                ),
            );
    }

    fn finish(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app.init_resource::<LensingDisplayPipeline>();
    }
}
