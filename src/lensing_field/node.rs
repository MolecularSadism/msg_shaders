// Render-graph node for the lensing-field flow simulation.
//
// Per frame: inject (write decayed previous + per-cell Schwarzschild force
// into `velocity_read`), then semi-Lagrangian advect (read `velocity_read`,
// write `velocity_write`). The lensing display pass samples `velocity_write`.

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

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub(crate) struct LensingFieldLabel;

/// Bind groups built each frame once the GPU images are ready.
///
/// Inject reads `velocity_write` (last frame's advect output, the field's
/// persistent state) and writes the decayed+force-injected result into
/// `velocity_read` as the intermediate buffer. Advect then reads that
/// intermediate and writes the advected field back into `velocity_write`,
/// which is the handle the material samples.
#[derive(Resource, Default)]
pub(crate) struct LensingFieldBindGroups {
    pub g0: Option<BindGroup>,
    /// inject: read = velocity_write (persistent state) → write = velocity_read.
    pub g1_inject: Option<BindGroup>,
    /// advect: read = velocity_read (intermediate) → write = velocity_write.
    pub g1_advect: Option<BindGroup>,
}

pub(crate) fn prepare_lensing_field_bind_groups(
    mut bind_groups: ResMut<LensingFieldBindGroups>,
    pipelines: Res<LensingFieldPipelines>,
    extracted: Res<ExtractedLensingField>,
    gpu_images: Res<RenderAssets<GpuImage>>,
    render_device: Res<RenderDevice>,
    pipeline_cache: Res<PipelineCache>,
) {
    if extracted.source_count == 0 {
        *bind_groups = LensingFieldBindGroups::default();
        return;
    }

    let Some(buf) = pipelines.uniforms.binding() else {
        return;
    };

    let g0_layout = pipeline_cache.get_bind_group_layout(&pipelines.layout_uniforms);
    bind_groups.g0 = Some(render_device.create_bind_group(
        "lensing_field_g0",
        &g0_layout,
        &BindGroupEntries::sequential((buf,)),
    ));

    let Some(vel_r) = gpu_images.get(&extracted.velocity_read) else {
        return;
    };
    let Some(vel_w) = gpu_images.get(&extracted.velocity_write) else {
        return;
    };

    let g1_layout = pipeline_cache.get_bind_group_layout(&pipelines.layout_vel_ping_pong);

    bind_groups.g1_inject = Some(render_device.create_bind_group(
        "lensing_field_g1_inject",
        &g1_layout,
        &BindGroupEntries::sequential((&vel_w.texture_view, &vel_r.texture_view)),
    ));
    bind_groups.g1_advect = Some(render_device.create_bind_group(
        "lensing_field_g1_advect",
        &g1_layout,
        &BindGroupEntries::sequential((&vel_r.texture_view, &vel_w.texture_view)),
    ));
}

#[derive(Default)]
pub(crate) struct LensingFieldNode;

impl ViewNode for LensingFieldNode {
    type ViewQuery = ();

    fn run<'w>(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext<'w>,
        (): (),
        world: &'w World,
    ) -> Result<(), NodeRunError> {
        let extracted = world.resource::<ExtractedLensingField>();
        if extracted.source_count == 0 {
            return Ok(());
        }

        let bind_groups = world.resource::<LensingFieldBindGroups>();
        let (Some(g0), Some(g1_inject), Some(g1_advect)) = (
            bind_groups.g0.as_ref(),
            bind_groups.g1_inject.as_ref(),
            bind_groups.g1_advect.as_ref(),
        ) else {
            return Ok(());
        };

        let pipelines = world.resource::<LensingFieldPipelines>();
        let pipeline_cache = world.resource::<PipelineCache>();

        for &id in &[pipelines.pipeline_inject, pipelines.pipeline_advect] {
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

        // Two separate compute passes so wgpu's resource tracker can insert a
        // Vulkan pipeline barrier between them. inject writes velocity_read;
        // advect reads it — a single pass cannot barrier between its own
        // dispatches, which produces a write→read hazard on Vulkan.
        {
            let mut pass =
                render_context
                    .command_encoder()
                    .begin_compute_pass(&ComputePassDescriptor {
                        label: Some("lensing_field_inject"),
                        timestamp_writes: None,
                    });
            pass.set_pipeline(pipe_inject);
            pass.set_bind_group(0, g0, &[]);
            pass.set_bind_group(1, g1_inject, &[]);
            pass.dispatch_workgroups(FIELD_DISPATCH, FIELD_DISPATCH, 1);
        }

        {
            let mut pass =
                render_context
                    .command_encoder()
                    .begin_compute_pass(&ComputePassDescriptor {
                        label: Some("lensing_field_advect"),
                        timestamp_writes: None,
                    });
            pass.set_pipeline(pipe_advect);
            pass.set_bind_group(0, g0, &[]);
            pass.set_bind_group(1, g1_advect, &[]);
            pass.dispatch_workgroups(FIELD_DISPATCH, FIELD_DISPATCH, 1);
        }

        Ok(())
    }
}

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
            // The simulation must finish before the 2D main pass samples the
            // velocity texture in the lensing display pass.
            .add_render_graph_edges(Core2d, (LensingFieldLabel, Node2d::StartMainPass));
    }
}
