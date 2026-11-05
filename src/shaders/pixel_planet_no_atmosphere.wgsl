// Barren / airless rocky planet (Deep-Fold/PixelPlanets `NoAtmosphere`, MIT).

#import bevy_sprite::mesh2d_vertex_output::VertexOutput
#import msg_shaders::pixel_planet_functions::{fbm_v, rotate_uv, dither_bool, dither_coverage}

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

@fragment
fn fragment(mesh: VertexOutput) -> @location(0) vec4<f32> {
    let tile = vec2<f32>(1.0, 1.0);

    var uv = floor(mesh.uv * u.pixels) / u.pixels;

    let d_circle = distance(uv, vec2<f32>(0.5));
    var d_light = distance(uv, u.light_origin);
    let cell = floor(mesh.uv * u.pixels);
    let edge_w = (1.0 / u.pixels) * u.edge_dither;
    let coverage = clamp((0.49999 - d_circle) / max(edge_w, 1e-6), 0.0, 1.0);
    let a = dither_coverage(coverage, cell);

    let dith = dither_bool(uv, mesh.uv, u.pixels);
    uv = rotate_uv(uv, u.rotation);

    let fbm1 = fbm_v(uv, u.octaves, u.size, u.seed, tile);
    d_light += fbm_v(uv * u.size + vec2<f32>(fbm1) + vec2<f32>(u.time, 0.0), u.octaves, u.size, u.seed, tile) * 0.3;

    let dither_border = (1.0 / u.pixels) * u.dither_size;

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
