// Vertex stage of the pixel-planet layers in `PixelGrid::World`.
//
// Replaces Bevy's `Mesh2d` vertex shader for those pipelines: besides placing
// the quad it derives the world -> cell map the fragment stage snaps with (see
// `pixel_planet_world_grid.wgsl`). The map comes from the mesh transform the
// renderer uploads anyway, so the mode costs nothing per frame on the CPU.

#import bevy_sprite::mesh2d_functions as mesh_functions
#import msg_shaders::pixel_planet_world_grid::{WorldGridVertexOutput, WORLD_GRID_QUAD_MARGIN}

struct Vertex {
    @builtin(instance_index) instance_index: u32,
    @location(0) position: vec3<f32>,
    @location(2) uv: vec2<f32>,
};

@vertex
fn vertex(vertex: Vertex) -> WorldGridVertexOutput {
    let world_from_local = mesh_functions::get_world_from_local(vertex.instance_index);
    let axis_x = world_from_local[0].xy;
    let axis_y = world_from_local[1].xy;

    // The layer mesh is a centred rectangle whose UVs span it once, so every
    // corner yields the same signed local extent per unit of UV.
    let size = vertex.position.xy / (vertex.uv - vec2<f32>(0.5));
    let cells = abs(size.x) * length(axis_x);

    // local_from_world: inverse of the transform's 2x2 linear part.
    let det = axis_x.x * axis_y.y - axis_y.x * axis_x.y;
    let local_from_world_x = vec2<f32>(axis_y.y, -axis_x.y) / det;
    let local_from_world_y = vec2<f32>(-axis_y.x, axis_x.x) / det;
    let cell_from_local = vec2<f32>(cells) / size;

    // Grow the quad so it covers every world pixel that looks up a point on it.
    let margin = sign(vertex.position.xy) * WORLD_GRID_QUAD_MARGIN
        / vec2<f32>(length(axis_x), length(axis_y));

    var out: WorldGridVertexOutput;
    out.world_position = mesh_functions::mesh2d_position_local_to_world(
        world_from_local,
        vec4<f32>(vertex.position.xy + margin, vertex.position.z, 1.0)
    );
    out.position = mesh_functions::mesh2d_position_world_to_clip(out.world_position);
    out.cell_from_world_x = cell_from_local * local_from_world_x;
    out.cell_from_world_y = cell_from_local * local_from_world_y;
    out.origin = world_from_local[3].xy;
    out.cells = cells;
    return out;
}
