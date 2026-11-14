// Terran base / ocean layer (Deep-Fold/PixelPlanets `PlanetUnder`, MIT).

#ifdef PIXEL_GRID_WORLD
#import msg_shaders::pixel_planet_world_grid::{WorldGridVertexOutput, world_grid_cell, world_grid_checker}
#else
#import bevy_sprite::mesh2d_vertex_output::VertexOutput
#endif
#import msg_shaders::pixel_planet_functions::{fbm_v, spherify, rotate_uv, dither_bool, dither_coverage}

struct U {
    colors: array<vec4<f32>, 3>,
    light_origin: vec2<f32>,
    pixels: f32,
    rotation: f32,
    size: f32,
    seed: f32,
    time: f32,
    light_border_1: f32,
    light_border_2: f32,
    dither_size: f32,
    edge_dither: f32,
    octaves: i32,
    should_dither: u32,
}

@group(2) @binding(0) var<uniform> u: U;

// The pixel cell this fragment belongs to: `uv` is its lookup point, `cell` its
// integer coordinate, `pixels` the cell count across the quad. `PixelGrid`
// picks the space the cells live in.
#ifdef PIXEL_GRID_WORLD
@fragment
fn fragment(mesh: WorldGridVertexOutput) -> @location(0) vec4<f32> {
    let grid = world_grid_cell(mesh);
    let pixels = grid.pixels;
    var uv = grid.uv;
    let cell = grid.cell;
    let dith = world_grid_checker(cell);
#else
@fragment
fn fragment(mesh: VertexOutput) -> @location(0) vec4<f32> {
    let pixels = u.pixels;
    var uv = floor(mesh.uv * u.pixels) / u.pixels;
    let cell = floor(mesh.uv * u.pixels);
    let dith = dither_bool(uv, mesh.uv, u.pixels);
#endif
    let tile = vec2<f32>(2.0, 1.0);

    var d_light = distance(uv, u.light_origin);

    let d_circle = distance(uv, vec2<f32>(0.5));
    let edge_w = (1.0 / pixels) * u.edge_dither;
    let coverage = clamp((0.49999 - d_circle) / max(edge_w, 1e-6), 0.0, 1.0);
    let a = dither_coverage(coverage, cell);

    uv = spherify(uv);
    uv = rotate_uv(uv, u.rotation);

    d_light += fbm_v(uv * u.size + vec2<f32>(u.time, 0.0), u.octaves, u.size, u.seed, tile) * 0.3;

    let dither_border = (1.0 / pixels) * u.dither_size;

    var col = u.colors[0];
    if (d_light > u.light_border_1) {
        col = u.colors[1];
        if (d_light < u.light_border_1 + dither_border && (dith || u.should_dither == 0u)) {
            col = u.colors[0];
        }
    }
    if (d_light > u.light_border_2) {
        col = u.colors[2];
        if (d_light < u.light_border_2 + dither_border && (dith || u.should_dither == 0u)) {
            col = u.colors[1];
        }
    }

    return vec4<f32>(col.rgb, a * col.a);
}
