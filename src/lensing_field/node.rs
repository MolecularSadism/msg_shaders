// Render-graph node for the lensing-field fluid simulation.
//
// Dispatches five compute passes each frame (inject → advect → divergence →
// pressure Jacobi × N → gradient subtract). The final divergence-free velocity
// lands in `velocity_write` (i.e. `velocity_pong`), which `LensingMaterial`
// samples in the same-frame main pass thanks to the render-graph edge.
//
// The node is a `ViewNode` targeting `LensingHoleCamera` cameras so the
// dispatch runs exactly once per lens-overlay render target, before the 2D
// main pass that renders the `LensingMaterial` quad.

use bevy::core_pipeline::core_2d::graph::{Core2d, Node2d};
use bevy::prelude::*;
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_graph::{
    NodeRunError, RenderGraphContext, RenderGraphExt, RenderLabel, ViewNode, ViewNodeRunner,
};
use bevy::render::render_resource::{
    BindGroup, BindGroupEntries, CachedPipelineState, ComputePassDescriptor, PipelineCache,
};
use bevy::render::renderer::{RenderContext, RenderDevice};
use bevy::render::texture::GpuImage;
use bevy::render::{Render, RenderApp, RenderSystems};

use crate::lensing_field::{extract::ExtractedLensingField, pipelines::LensingFieldPipelines};

const WORKGROUP_SIZE: u32 = 8;
const FIELD_DISPATCH: u32 = crate::lensing_field::LENSING_FIELD_RES / WORKGROUP_SIZE;

// ── Render label ──────────────────────────────────────────────────────────────

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub(crate) struct LensingFieldLabel;

// ── Per-frame bind groups (built in Prepare) ──────────────────────────────────

/// Bind groups built each frame once the GPU images are ready.
///
/// Velocity bind-group permutations cover inject/advect (vel_read → vel_write),
/// the pressure Jacobi iterations (pressure_ping ↔ pressure_pong), and the
/// gradient subtract pass (vel_read + pressure_result → vel_write).
#[derive(Resource, Default)]
pub(crate) struct LensingFieldBindGroups {
    /// Group 0: the shared uniform.
    pub g0: Option<BindGroup>,
    /// inject: vel_read (storage read) → vel_write (storage write).
    pub g1_vel_read_to_write: Option<BindGroup>,
    /// advect: vel_write (storage read) → vel_read (storage write).
    /// After inject the injected velocity lives in vel_write; advecting it back
    /// into vel_read keeps vel_read as the canonical "current frame" field.
    pub g1_vel_write_to_read: Option<BindGroup>,
    /// divergence pass: vel_read → divergence_out.
    pub g1_divergence: Option<BindGroup>,
    /// Jacobi pass A: divergence + pressure_ping → pressure_pong.
    pub g1_pressure_ping_to_pong: Option<BindGroup>,
    /// Jacobi pass B: divergence + pressure_pong → pressure_ping.
    pub g1_pressure_pong_to_ping: Option<BindGroup>,
    /// gradient subtract (Jacobi even iterations end in pong): vel_read + pressure_pong →
    /// vel_write.
    pub g1_gradient_from_pong: Option<BindGroup>,
    /// gradient subtract (Jacobi odd iterations end in ping): vel_read + pressure_ping →
    /// vel_write.
    pub g1_gradient_from_ping: Option<BindGroup>,
}

// ── Prepare systems ───────────────────────────────────────────────────────────

pub(crate) fn prepare_lensing_field_bind_groups(
    mut bind_groups: ResMut<LensingFieldBindGroups>,
    pipelines: Res<LensingFieldPipelines>,
    extracted: Res<ExtractedLensingField>,
    gpu_images: Res<RenderAssets<GpuImage>>,
    render_device: Res<RenderDevice>,
    pipeline_cache: Res<PipelineCache>,
) {
    // Nothing to do if no lenses are active or the buffer is unavailable.
    if extracted.lens_count == 0 {
        *bind_groups = LensingFieldBindGroups::default();
        return;
    }

    let Some(buf) = pipelines.uniforms.binding() else {
        return;
    };

    // Build group 0 (uniforms) — always the same layout.
    let g0_layout = pipeline_cache.get_bind_group_layout(&pipelines.layout_uniforms);
    bind_groups.g0 = Some(render_device.create_bind_group(
        "lensing_field_g0",
        &g0_layout,
        &BindGroupEntries::sequential((buf,)),
    ));

    // Gather GPU texture views.
    let Some(vel_r) = gpu_images.get(&extracted.velocity_read) else {
        return;
    };
    let Some(vel_w) = gpu_images.get(&extracted.velocity_write) else {
        return;
    };
    let Some(div) = gpu_images.get(&extracted.divergence) else {
        return;
    };
    let Some(p_ping) = gpu_images.get(&extracted.pressure_ping) else {
        return;
    };
    let Some(p_pong) = gpu_images.get(&extracted.pressure_pong) else {
        return;
    };

    let g1_vel = pipeline_cache.get_bind_group_layout(&pipelines.layout_vel_ping_pong);
    let g1_div = pipeline_cache.get_bind_group_layout(&pipelines.layout_divergence);
    let g1_prs = pipeline_cache.get_bind_group_layout(&pipelines.layout_pressure);

    bind_groups.g1_vel_read_to_write = Some(render_device.create_bind_group(
        "lensing_field_g1_vel_r2w",
        &g1_vel,
        &BindGroupEntries::sequential((&vel_r.texture_view, &vel_w.texture_view)),
    ));
    bind_groups.g1_vel_write_to_read = Some(render_device.create_bind_group(
        "lensing_field_g1_vel_w2r",
        &g1_vel,
        &BindGroupEntries::sequential((&vel_w.texture_view, &vel_r.texture_view)),
    ));
    bind_groups.g1_divergence = Some(render_device.create_bind_group(
        "lensing_field_g1_divergence",
        &g1_div,
        &BindGroupEntries::sequential((&vel_r.texture_view, &div.texture_view)),
    ));
    bind_groups.g1_pressure_ping_to_pong = Some(render_device.create_bind_group(
        "lensing_field_g1_pressure_p2p",
        &g1_prs,
        &BindGroupEntries::sequential((
            &div.texture_view,
            &p_ping.texture_view,
            &p_pong.texture_view,
        )),
    ));
    bind_groups.g1_pressure_pong_to_ping = Some(render_device.create_bind_group(
        "lensing_field_g1_pressure_pp2p",
        &g1_prs,
        &BindGroupEntries::sequential((
            &div.texture_view,
            &p_pong.texture_view,
            &p_ping.texture_view,
        )),
    ));

    let g1_grad = pipeline_cache.get_bind_group_layout(&pipelines.layout_gradient);
    // When Jacobi ends in pong (even iterations), the final pressure is in pong.
    bind_groups.g1_gradient_from_pong = Some(render_device.create_bind_group(
        "lensing_field_g1_gradient_from_pong",
        &g1_grad,
        &BindGroupEntries::sequential((
            &vel_r.texture_view,
            &p_pong.texture_view,
            &vel_w.texture_view,
        )),
    ));
    // When Jacobi ends in ping (odd iterations), the final pressure is in ping.
    bind_groups.g1_gradient_from_ping = Some(render_device.create_bind_group(
        "lensing_field_g1_gradient_from_ping",
        &g1_grad,
        &BindGroupEntries::sequential((
            &vel_r.texture_view,
            &p_ping.texture_view,
            &vel_w.texture_view,
        )),
    ));
}

// ── Render node ───────────────────────────────────────────────────────────────

#[derive(Default)]
pub(crate) struct LensingFieldNode;

impl ViewNode for LensingFieldNode {
    // Target cameras that own a lensing overlay: only one such camera exists
    // per scene, so the dispatch runs once.  The `LensingHoleCamera` marker
    // is a filter; fetch nothing from the entity itself — all data lives in
    // world resources.
    type ViewQuery = ();

    fn run<'w>(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext<'w>,
        (): (),
        world: &'w World,
    ) -> Result<(), NodeRunError> {
        let extracted = world.resource::<ExtractedLensingField>();
        if extracted.lens_count == 0 {
            return Ok(());
        }

        let bind_groups = world.resource::<LensingFieldBindGroups>();
        let (
            Some(g0),
            Some(g1_r2w),
            Some(g1_w2r),
            Some(g1_div),
            Some(g1_pp),
            Some(g1_ppp),
            Some(g1_grad_ping),
            Some(g1_grad_pong),
        ) = (
            bind_groups.g0.as_ref(),
            bind_groups.g1_vel_read_to_write.as_ref(),
            bind_groups.g1_vel_write_to_read.as_ref(),
            bind_groups.g1_divergence.as_ref(),
            bind_groups.g1_pressure_ping_to_pong.as_ref(),
            bind_groups.g1_pressure_pong_to_ping.as_ref(),
            bind_groups.g1_gradient_from_ping.as_ref(),
            bind_groups.g1_gradient_from_pong.as_ref(),
        )
        else {
            return Ok(());
        };

        let pipelines = world.resource::<LensingFieldPipelines>();
        let pipeline_cache = world.resource::<PipelineCache>();

        // Verify all pipelines are compiled before touching the encoder.
        for &id in &[
            pipelines.pipeline_inject,
            pipelines.pipeline_advect,
            pipelines.pipeline_divergence,
            pipelines.pipeline_pressure,
            pipelines.pipeline_gradient,
        ] {
            match pipeline_cache.get_compute_pipeline_state(id) {
                CachedPipelineState::Ok(_) => {}
                CachedPipelineState::Err(e) => {
                    bevy::log::error!("lensing_field pipeline error: {e}");
                    return Ok(());
                }
                _ => return Ok(()),
            }
        }

        let pipe_inject = pipeline_cache
            .get_compute_pipeline(pipelines.pipeline_inject)
            .unwrap();
        let pipe_advect = pipeline_cache
            .get_compute_pipeline(pipelines.pipeline_advect)
            .unwrap();
        let pipe_divergence = pipeline_cache
            .get_compute_pipeline(pipelines.pipeline_divergence)
            .unwrap();
        let pipe_pressure = pipeline_cache
            .get_compute_pipeline(pipelines.pipeline_pressure)
            .unwrap();
        let pipe_gradient = pipeline_cache
            .get_compute_pipeline(pipelines.pipeline_gradient)
            .unwrap();

        let jacobi_iters = extracted.jacobi_iters.max(1);

        let mut pass =
            render_context
                .command_encoder()
                .begin_compute_pass(&ComputePassDescriptor {
                    label: Some("lensing_field"),
                    timestamp_writes: None,
                });

        // Pass 1: inject — force + decay into vel_read → vel_write.
        pass.set_pipeline(pipe_inject);
        pass.set_bind_group(0, g0, &[]);
        pass.set_bind_group(1, g1_r2w, &[]);
        pass.dispatch_workgroups(FIELD_DISPATCH, FIELD_DISPATCH, 1);

        // Pass 2: advect — semi-Lagrangian advect vel_write → vel_read.
        // After inject the new velocity is in vel_write; we advect it back
        // into vel_read so the downstream passes always read from vel_read.
        pass.set_pipeline(pipe_advect);
        pass.set_bind_group(0, g0, &[]);
        pass.set_bind_group(1, g1_w2r, &[]);
        pass.dispatch_workgroups(FIELD_DISPATCH, FIELD_DISPATCH, 1);

        // Pass 3: divergence — central differences on vel_read → divergence.
        pass.set_pipeline(pipe_divergence);
        pass.set_bind_group(0, g0, &[]);
        pass.set_bind_group(1, g1_div, &[]);
        pass.dispatch_workgroups(FIELD_DISPATCH, FIELD_DISPATCH, 1);

        // Pass 4: pressure Jacobi (ping-pong).
        // First iteration: ping → pong.  Track which buffer holds the final pressure.
        pass.set_pipeline(pipe_pressure);
        pass.set_bind_group(0, g0, &[]);
        let mut pong_is_current = true; // after first iter pong holds the result
        for _ in 0..jacobi_iters {
            let g1 = if pong_is_current { g1_pp } else { g1_ppp };
            pass.set_bind_group(1, g1, &[]);
            pass.dispatch_workgroups(FIELD_DISPATCH, FIELD_DISPATCH, 1);
            pong_is_current = !pong_is_current;
        }

        // Pass 5: gradient subtract — subtract grad(pressure) from velocity.
        // Choose the bind group matching whichever pressure buffer holds the
        // final solved pressure (the one that was last written).
        // After N iterations starting with ping→pong:
        //   odd N  → last write was pong  → pong_is_current is false (it was flipped)
        //   even N → last write was ping  → pong_is_current is true
        // So the final pressure is in pong when !pong_is_current.
        let g1_grad = if pong_is_current {
            // Last write landed in ping (even iterations).
            g1_grad_ping
        } else {
            // Last write landed in pong (odd iterations).
            g1_grad_pong
        };
        pass.set_pipeline(pipe_gradient);
        pass.set_bind_group(0, g0, &[]);
        pass.set_bind_group(1, g1_grad, &[]);
        pass.dispatch_workgroups(FIELD_DISPATCH, FIELD_DISPATCH, 1);

        drop(pass);

        Ok(())
    }
}

// ── Plugin ────────────────────────────────────────────────────────────────────

pub(crate) struct LensingFieldNodePlugin;

impl Plugin for LensingFieldNodePlugin {
    fn build(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .init_resource::<LensingFieldBindGroups>()
            .add_systems(
                Render,
                prepare_lensing_field_bind_groups.in_set(RenderSystems::PrepareBindGroups),
            )
            .add_render_graph_node::<ViewNodeRunner<LensingFieldNode>>(Core2d, LensingFieldLabel)
            // Run the field simulation before the 2D main pass starts drawing
            // the LensingMaterial quad.
            // Run the field simulation before the 2D main pass so the velocity
            // texture is ready when LensingMaterial renders.
            .add_render_graph_edges(Core2d, (LensingFieldLabel, Node2d::StartMainPass));
    }
}
