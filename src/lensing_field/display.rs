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

use bevy::core_pipeline::FullscreenShader;
use bevy::ecs::query::QueryItem;
use bevy::prelude::*;
use bevy::render::extract_component::ExtractComponentPlugin;
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_graph::{NodeRunError, RenderGraphContext, RenderLabel, ViewNode};
use bevy::render::render_resource::binding_types::{sampler, texture_2d, uniform_buffer};
use bevy::render::render_resource::{
    BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries, CachedRenderPipelineId,
    ColorTargetState, ColorWrites, FilterMode, FragmentState, MultisampleState, Operations,
    PipelineCache, PrimitiveState, RenderPassColorAttachment, RenderPassDescriptor,
    RenderPipelineDescriptor, Sampler, SamplerBindingType, SamplerDescriptor, ShaderStages,
    ShaderType, SpecializedRenderPipeline, SpecializedRenderPipelines, TextureFormat,
    TextureSampleType, UniformBuffer,
};
use bevy::render::renderer::{RenderContext, RenderDevice, RenderQueue};
use bevy::render::texture::GpuImage;
use bevy::render::view::{ExtractedView, ViewTarget};
use bevy::render::{Render, RenderApp, RenderSystems};
use bevy::shader::Shader;
use bevy_post_process_2d::{PostProcess2dAppExt, PostProcessOrder};

use crate::lensing_field::extract::ExtractedLensingField;
use crate::{LensData, LensingHoleCamera, MAX_LENSES};

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
    /// Number of populated entries in `lenses`.
    count: u32,
    lenses: [LensData; MAX_LENSES],
}

impl Default for LensingDisplayUniform {
    fn default() -> Self {
        Self {
            clip_from_world: Mat4::IDENTITY,
            world_from_clip: Mat4::IDENTITY,
            canvas_center_extent: Vec4::new(0.0, 0.0, 1.0, 1.0),
            count: 0,
            lenses: [LensData::default(); MAX_LENSES],
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
    views: Query<&ExtractedView, With<LensingHoleCamera>>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
) {
    let clip_from_world = views
        .iter()
        .next()
        .map(|ev| {
            ev.clip_from_world
                .unwrap_or_else(|| ev.clip_from_view * ev.world_from_view.to_matrix().inverse())
        })
        .unwrap_or(Mat4::IDENTITY);

    let count = extracted.lens_count.min(MAX_LENSES as u32);
    let mut lenses = [LensData::default(); MAX_LENSES];
    for (i, slot) in lenses.iter_mut().enumerate().take(count as usize) {
        *slot = LensData {
            center_size_shadow: extracted.lens_center_size_shadow[i],
            strength_ring: extracted.lens_strength_ring[i],
            photon_ring_color: extracted.lens_photon_ring_color[i],
            black_color: extracted.lens_black_color[i],
        };
    }

    uniforms.buffer.set(LensingDisplayUniform {
        clip_from_world,
        world_from_clip: clip_from_world.inverse(),
        canvas_center_extent: extracted.canvas_center_extent,
        count,
        lenses,
    });
    uniforms.buffer.write_buffer(&render_device, &render_queue);
}

/// Cached pipeline id for a view, specialized on its `ViewTarget` format.
#[derive(Component, Clone, Copy)]
struct LensingDisplayPipelineId(CachedRenderPipelineId);

fn queue_lensing_display_pipelines(
    mut commands: Commands,
    pipeline_cache: Res<PipelineCache>,
    pipeline: Res<LensingDisplayPipeline>,
    mut specialized: ResMut<SpecializedRenderPipelines<LensingDisplayPipeline>>,
    views: Query<(Entity, &ViewTarget), With<LensingHoleCamera>>,
) {
    for (entity, view_target) in &views {
        let id = specialized.specialize(
            &pipeline_cache,
            &pipeline,
            LensingDisplayPipelineKey {
                target_format: view_target.main_texture_format(),
            },
        );
        commands.entity(entity).insert(LensingDisplayPipelineId(id));
    }
}

#[derive(Default)]
struct LensingDisplayNode;

impl ViewNode for LensingDisplayNode {
    type ViewQuery = (&'static ViewTarget, &'static LensingDisplayPipelineId);

    fn run<'w>(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext<'w>,
        (view_target, pipeline_id): QueryItem<'w, '_, Self::ViewQuery>,
        world: &'w World,
    ) -> Result<(), NodeRunError> {
        let extracted = world.resource::<ExtractedLensingField>();
        // No lenses: leave the view untouched (the scene passes straight through
        // to the downstream post-processing stack).
        if extracted.lens_count == 0 {
            return Ok(());
        }

        let pipeline_cache = world.resource::<PipelineCache>();
        let pipeline = world.resource::<LensingDisplayPipeline>();
        let Some(render_pipeline) = pipeline_cache.get_render_pipeline(pipeline_id.0) else {
            return Ok(());
        };

        let gpu_images = world.resource::<RenderAssets<GpuImage>>();
        let Some(field) = gpu_images.get(&extracted.velocity_write) else {
            return Ok(());
        };

        let uniforms = world.resource::<LensingDisplayUniforms>();
        let Some(uniform_binding) = uniforms.buffer.binding() else {
            return Ok(());
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
        });
        render_pass.set_render_pipeline(render_pipeline);
        render_pass.set_bind_group(0, &bind_group, &[]);
        render_pass.draw(0..3, 0..1);

        Ok(())
    }
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
            push_constant_ranges: vec![],
            zero_initialize_workgroup_memory: false,
        }
    }
}

/// Render-graph label for the lensing display pass.
#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
struct LensingDisplayLabel;

/// Registers the full-screen lensing display pass as a post-process node,
/// ordered after lighting in the shared 2D post-process chain.
pub(crate) struct LensingDisplayPlugin;

impl Plugin for LensingDisplayPlugin {
    fn build(&self, app: &mut App) {
        // The shader is embedded by `material::MaterialsPlugin`, which this
        // plugin is added alongside.
        app.add_plugins(ExtractComponentPlugin::<LensingHoleCamera>::default());
        app.add_post_process_2d_node::<LensingDisplayNode>(
            LensingDisplayLabel,
            PostProcessOrder::DISTORTION,
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
