// Compute pipelines for the lensing-field flow simulation.
//
// Two compute shaders handle one frame:
//   inject — force injection + velocity decay
//   advect — semi-Lagrangian self-advection
//
// Pressure projection is intentionally omitted: gravitational deflection is a
// gradient field, and projecting it to be divergence-free would cancel the
// signal and leave only the 4-point stencil's cardinal-axis residual.

use bevy::prelude::*;
use bevy::render::render_resource::binding_types::{texture_storage_2d, uniform_buffer};
use bevy::render::render_resource::{
    BindGroupLayoutDescriptor, BindGroupLayoutEntries, CachedComputePipelineId,
    ComputePipelineDescriptor, PipelineCache, ShaderStages, ShaderType, StorageTextureAccess,
    TextureFormat, UniformBuffer,
};
use bevy::render::renderer::{RenderDevice, RenderQueue};
use bevy::render::{Render, RenderApp, RenderSystems};

use crate::lensing_field::extract::ExtractedLensingField;
use crate::lensing_field::sources::{DeflectionSource, MAX_DEFLECTION_SOURCES};

// ── Uniform sent to every lensing-field compute pass ─────────────────────────

/// Mirrors `LensingFieldUniform` in the compute shaders.
#[derive(ShaderType, Clone, Copy)]
pub(crate) struct LensingFieldUniform {
    pub sources: [DeflectionSource; MAX_DEFLECTION_SOURCES],
    /// Canvas center (xy) and extent (zw) in world space.
    pub canvas_center_extent: Vec4,
    pub source_count: u32,
    /// Per-frame velocity decay multiplier applied during inject.
    pub decay: f32,
    /// Force injection scale.
    pub force_scale: f32,
    /// Advection time step.
    pub dt: f32,
}

impl Default for LensingFieldUniform {
    fn default() -> Self {
        Self {
            sources: [DeflectionSource::default(); MAX_DEFLECTION_SOURCES],
            canvas_center_extent: Vec4::new(0.0, 0.0, 1.0, 1.0),
            source_count: 0,
            decay: 0.0,
            force_scale: 1.0,
            dt: 0.016,
        }
    }
}

// ── Pipeline resource ─────────────────────────────────────────────────────────

#[derive(Resource)]
pub struct LensingFieldPipelines {
    pub(crate) layout_uniforms: BindGroupLayoutDescriptor,
    /// Binding 1 layout for inject and advect: velocity_in + velocity_out.
    pub(crate) layout_vel_ping_pong: BindGroupLayoutDescriptor,

    pub(crate) uniforms: UniformBuffer<LensingFieldUniform>,

    pub(crate) pipeline_inject: CachedComputePipelineId,
    pub(crate) pipeline_advect: CachedComputePipelineId,
}

impl FromWorld for LensingFieldPipelines {
    fn from_world(world: &mut World) -> Self {
        let pipeline_cache = world.resource::<PipelineCache>();

        let layout_uniforms = BindGroupLayoutDescriptor::new(
            "lensing_field_g0",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (uniform_buffer::<LensingFieldUniform>(false),),
            ),
        );

        let layout_vel_ping_pong = BindGroupLayoutDescriptor::new(
            "lensing_field_g1_vel",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (
                    texture_storage_2d(TextureFormat::Rg16Float, StorageTextureAccess::ReadOnly),
                    texture_storage_2d(TextureFormat::Rg16Float, StorageTextureAccess::WriteOnly),
                ),
            ),
        );

        let shader_inject =
            world.load_asset("embedded://msg_shaders/shaders/lensing_field_inject.wgsl");
        let shader_advect =
            world.load_asset("embedded://msg_shaders/shaders/lensing_field_advect.wgsl");

        let pipeline_inject = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("lensing_field_inject".into()),
            entry_point: Some("main".into()),
            layout: vec![layout_uniforms.clone(), layout_vel_ping_pong.clone()],
            push_constant_ranges: vec![],
            shader: shader_inject,
            shader_defs: vec![],
            zero_initialize_workgroup_memory: false,
        });

        let pipeline_advect = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("lensing_field_advect".into()),
            entry_point: Some("main".into()),
            layout: vec![layout_uniforms.clone(), layout_vel_ping_pong.clone()],
            push_constant_ranges: vec![],
            shader: shader_advect,
            shader_defs: vec![],
            zero_initialize_workgroup_memory: false,
        });

        Self {
            layout_uniforms,
            layout_vel_ping_pong,
            uniforms: UniformBuffer::default(),
            pipeline_inject,
            pipeline_advect,
        }
    }
}

// ── Prepare system: upload the uniform buffer once per frame ──────────────────

pub(crate) fn prepare_lensing_field_uniforms(
    extracted: Res<ExtractedLensingField>,
    mut pipelines: ResMut<LensingFieldPipelines>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
) {
    let uniform = LensingFieldUniform {
        sources: extracted.sources,
        canvas_center_extent: extracted.canvas_center_extent,
        source_count: extracted.source_count,
        decay: extracted.decay,
        force_scale: extracted.force_scale,
        dt: extracted.dt,
    };
    pipelines.uniforms.set(uniform);
    pipelines
        .uniforms
        .write_buffer(&render_device, &render_queue);
}

// ── Plugin ────────────────────────────────────────────────────────────────────

pub(crate) struct LensingFieldPipelinesPlugin;

impl Plugin for LensingFieldPipelinesPlugin {
    fn build(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app.add_systems(
            Render,
            prepare_lensing_field_uniforms.in_set(RenderSystems::Prepare),
        );
    }

    fn finish(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app.init_resource::<LensingFieldPipelines>();
    }
}
