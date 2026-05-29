// Compute pipelines for the lensing-field fluid simulation.
//
// Five compute shaders handle one fluid frame:
//   inject    – force injection + velocity decay
//   advect    – semi-Lagrangian self-advection
//   divergence – central-difference divergence
//   pressure  – one Jacobi iteration (ping-pong N times)
//   gradient  – pressure-gradient subtraction from velocity

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

// ── Uniform sent to every lensing-field compute pass ─────────────────────────

/// Mirrors `LensingFieldUniform` in all five compute shaders.
///
/// Contains the full lens array (matching `LensData` from `material.rs`),
/// simulation parameters, and per-lens count. Kept at one struct so every
/// shader uses the same bind group layout at binding 0.
#[derive(ShaderType, Clone, Copy)]
pub(crate) struct LensingFieldUniform {
    /// World-space center (xy), halo radius (z), shadow radius (w) for each lens.
    pub lens_center_size_shadow: [Vec4; crate::MAX_LENSES],
    /// lensing_strength (x), photon_ring_width (y), photon_ring_intensity (z), pad (w).
    pub lens_strength_ring: [Vec4; crate::MAX_LENSES],
    /// World-space canvas center (xy) and extent (zw). Used to map grid cells
    /// to world coordinates in the inject pass.
    pub canvas_center_extent: Vec4,
    /// Number of active lenses (0 = no-op dispatch).
    pub lens_count: u32,
    /// Per-frame velocity decay multiplier.
    pub decay: f32,
    /// Schwarzschild force scale.
    pub force_scale: f32,
    /// Advection time step.
    pub dt: f32,
}

impl Default for LensingFieldUniform {
    fn default() -> Self {
        Self {
            lens_center_size_shadow: [Vec4::ZERO; crate::MAX_LENSES],
            lens_strength_ring: [Vec4::ZERO; crate::MAX_LENSES],
            canvas_center_extent: Vec4::new(0.0, 0.0, 1.0, 1.0),
            lens_count: 0,
            decay: 0.98,
            force_scale: 1.0,
            dt: 0.016,
        }
    }
}

// ── Pipeline resource ─────────────────────────────────────────────────────────

/// Cached compute pipeline IDs and bind group layout descriptors for the five
/// fluid passes.  Created once in [`FromWorld`] and reused every frame.
#[derive(Resource)]
pub struct LensingFieldPipelines {
    /// Layout for binding 0 in all passes: one uniform buffer.
    pub(crate) layout_uniforms: BindGroupLayoutDescriptor,
    /// Layout for binding 1 in the inject and advect passes:
    /// velocity_in (read storage) and velocity_out (write storage).
    pub(crate) layout_vel_ping_pong: BindGroupLayoutDescriptor,
    /// Layout for binding 1 in the divergence pass:
    /// velocity_in (read), divergence_out (write).
    pub(crate) layout_divergence: BindGroupLayoutDescriptor,
    /// Layout for binding 1 in the pressure Jacobi pass:
    /// divergence (read), pressure_in (read), pressure_out (write).
    pub(crate) layout_pressure: BindGroupLayoutDescriptor,
    /// Layout for binding 1 in the gradient subtract pass:
    /// velocity_in (read), pressure_in (read), velocity_out (write).
    pub(crate) layout_gradient: BindGroupLayoutDescriptor,

    /// Persistent uniform buffer, uploaded once per frame.
    pub(crate) uniforms: UniformBuffer<LensingFieldUniform>,

    pub(crate) pipeline_inject: CachedComputePipelineId,
    pub(crate) pipeline_advect: CachedComputePipelineId,
    pub(crate) pipeline_divergence: CachedComputePipelineId,
    pub(crate) pipeline_pressure: CachedComputePipelineId,
    pub(crate) pipeline_gradient: CachedComputePipelineId,
}

impl FromWorld for LensingFieldPipelines {
    fn from_world(world: &mut World) -> Self {
        let pipeline_cache = world.resource::<PipelineCache>();

        // Binding 0 in every pass: one non-dynamic uniform.
        let layout_uniforms = BindGroupLayoutDescriptor::new(
            "lensing_field_g0",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (uniform_buffer::<LensingFieldUniform>(false),),
            ),
        );

        // velocity_in (Rg16Float read-storage) + velocity_out (write-storage).
        // Used by inject, advect, and gradient passes.
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

        // velocity_in (read) + divergence_out (write).
        let layout_divergence = BindGroupLayoutDescriptor::new(
            "lensing_field_g1_divergence",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (
                    texture_storage_2d(TextureFormat::Rg16Float, StorageTextureAccess::ReadOnly),
                    texture_storage_2d(TextureFormat::R16Float, StorageTextureAccess::WriteOnly),
                ),
            ),
        );

        // divergence_in (read, R16) + pressure_in (read, R16) + pressure_out (write, R16).
        let layout_pressure = BindGroupLayoutDescriptor::new(
            "lensing_field_g1_pressure",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (
                    texture_storage_2d(TextureFormat::R16Float, StorageTextureAccess::ReadOnly),
                    texture_storage_2d(TextureFormat::R16Float, StorageTextureAccess::ReadOnly),
                    texture_storage_2d(TextureFormat::R16Float, StorageTextureAccess::WriteOnly),
                ),
            ),
        );

        // velocity_in (read, Rg16) + pressure_in (read, R16) + velocity_out (write, Rg16).
        let layout_gradient = BindGroupLayoutDescriptor::new(
            "lensing_field_g1_gradient",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (
                    texture_storage_2d(TextureFormat::Rg16Float, StorageTextureAccess::ReadOnly),
                    texture_storage_2d(TextureFormat::R16Float, StorageTextureAccess::ReadOnly),
                    texture_storage_2d(TextureFormat::Rg16Float, StorageTextureAccess::WriteOnly),
                ),
            ),
        );

        let shader_inject =
            world.load_asset("embedded://msg_shaders/shaders/lensing_field_inject.wgsl");
        let shader_advect =
            world.load_asset("embedded://msg_shaders/shaders/lensing_field_advect.wgsl");
        let shader_divergence =
            world.load_asset("embedded://msg_shaders/shaders/lensing_field_divergence.wgsl");
        let shader_pressure =
            world.load_asset("embedded://msg_shaders/shaders/lensing_field_pressure_jacobi.wgsl");
        let shader_gradient =
            world.load_asset("embedded://msg_shaders/shaders/lensing_field_gradient_subtract.wgsl");

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

        let pipeline_divergence =
            pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some("lensing_field_divergence".into()),
                entry_point: Some("main".into()),
                layout: vec![layout_uniforms.clone(), layout_divergence.clone()],
                push_constant_ranges: vec![],
                shader: shader_divergence,
                shader_defs: vec![],
                zero_initialize_workgroup_memory: false,
            });

        let pipeline_pressure = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("lensing_field_pressure_jacobi".into()),
            entry_point: Some("main".into()),
            layout: vec![layout_uniforms.clone(), layout_pressure.clone()],
            push_constant_ranges: vec![],
            shader: shader_pressure,
            shader_defs: vec![],
            zero_initialize_workgroup_memory: false,
        });

        let pipeline_gradient = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("lensing_field_gradient_subtract".into()),
            entry_point: Some("main".into()),
            layout: vec![layout_uniforms.clone(), layout_gradient.clone()],
            push_constant_ranges: vec![],
            shader: shader_gradient,
            shader_defs: vec![],
            zero_initialize_workgroup_memory: false,
        });

        Self {
            layout_uniforms,
            layout_vel_ping_pong,
            layout_divergence,
            layout_pressure,
            layout_gradient,
            uniforms: UniformBuffer::default(),
            pipeline_inject,
            pipeline_advect,
            pipeline_divergence,
            pipeline_pressure,
            pipeline_gradient,
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
        lens_center_size_shadow: extracted.lens_center_size_shadow,
        lens_strength_ring: extracted.lens_strength_ring,
        canvas_center_extent: extracted.canvas_center_extent,
        lens_count: extracted.lens_count,
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
        // Shader embedding is handled in `material::MaterialsPlugin` alongside
        // the other msg_shaders embedded assets (from `src/material.rs` where
        // the path `"shaders/..."` resolves to `src/shaders/...` correctly).
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
