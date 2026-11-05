// Terran continents layer (Deep-Fold/PixelPlanets `PlanetLandmass`, MIT).

#import bevy_sprite::mesh2d_vertex_output::VertexOutput
#import msg_shaders::pixel_planet_functions::{fbm_v, spherify, rotate_uv, dither_coverage}

struct U {
    colors: array<vec4<f32>, 4>,
    light_origin: vec2<f32>,
    pixels: f32,
    rotation: f32,
    size: f32,
    seed: f32,
    time: f32,
    light_border_1: f32,
    light_border_2: f32,
    land_cutoff: f32,
    edge_dither: f32,
    octaves: i32,
}

@group(2) @binding(0) var<uniform> u: U;

@fragment
fn fragment(mesh: VertexOutput) -> @location(0) vec4<f32> {
    let tile = vec2<f32>(2.0, 1.0);

    var uv = floor(mesh.uv * u.pixels) / u.pixels;

    var d_light = distance(uv, u.light_origin);
    let d_circle = distance(uv, vec2<f32>(0.5));
    let cell = floor(mesh.uv * u.pixels);
    let edge_w = (1.0 / u.pixels) * u.edge_dither;
    let coverage = clamp((0.49999 - d_circle) / max(edge_w, 1e-6), 0.0, 1.0);
    let a = dither_coverage(coverage, cell);

    uv = rotate_uv(uv, u.rotation);
    uv = spherify(uv);

    let base = uv * u.size + vec2<f32>(u.time, 0.0);

    let fbm1 = fbm_v(base, u.octaves, u.size, u.seed, tile);
    var fbm2 = fbm_v(base - u.light_origin * fbm1, u.octaves, u.size, u.seed, tile);
    var fbm3 = fbm_v(base - u.light_origin * 1.5 * fbm1, u.octaves, u.size, u.seed, tile);
    var fbm4 = fbm_v(base - u.light_origin * 2.0 * fbm1, u.octaves, u.size, u.seed, tile);

    if (d_light < u.light_border_1) {
        fbm4 *= 0.9;
    }
    if (d_light > u.light_border_1) {
        fbm2 *= 1.05;
        fbm3 *= 1.05;
        fbm4 *= 1.05;
    }
    if (d_light > u.light_border_2) {
        fbm2 *= 1.3;
        fbm3 *= 1.4;
        fbm4 *= 1.8;
    }

    let dl = pow(d_light, 2.0) * 0.1;

    var col = u.colors[3];
    if (fbm4 + dl < fbm1) {
        col = u.colors[2];
    }
    if (fbm3 + dl < fbm1) {
        col = u.colors[1];
    }
    if (fbm2 + dl < fbm1) {
        col = u.colors[0];
    }

    return vec4<f32>(col.rgb, step(u.land_cutoff, fbm1) * a * col.a);
}
