// Barren / airless rocky planet (Deep-Fold/PixelPlanets `NoAtmosphere`, MIT).

#import bevy_sprite::mesh2d_vertex_output::VertexOutput
#import msg_shaders::pixel_planet_functions::{fbm_v, sphere_lonlat, lonlat_noise, pole_coverage, rotate_uv, dither_bool, dither_coverage}

struct U {
    colors: array<vec4<f32>, 3>,
    light_origin: vec2<f32>,
    tilt_cs: vec2<f32>,
    pixels: f32,
    rotation: f32,
    size: f32,
    seed: f32,
    time: f32,
    light_border_1: f32,
    light_border_2: f32,
    dither_size: f32,
    edge_dither: f32,
    pole_cap: f32,
    octaves: i32,
    should_dither: u32,
}

@group(2) @binding(0) var<uniform> u: U;

// Noise size of the low-frequency domain warp: two cells per revolution.
const WARP_SIZE: f32 = 2.0;

@fragment
fn fragment(mesh: VertexOutput) -> @location(0) vec4<f32> {
    let tile = vec2<f32>(1.0, 1.0);

    let uv = floor(mesh.uv * u.pixels) / u.pixels;

    let d_circle = distance(uv, vec2<f32>(0.5));
    var d_light = distance(uv, u.light_origin);
    let cell = floor(mesh.uv * u.pixels);
    let edge_w = (1.0 / u.pixels) * u.edge_dither;
    let coverage = clamp((0.49999 - d_circle) / max(edge_w, 1e-6), 0.0, 1.0);
    let a = dither_coverage(coverage, cell);

    let dith = dither_bool(uv, mesh.uv, u.pixels);
    let ll = sphere_lonlat(rotate_uv(uv, u.rotation), u.tilt_cs);

    let fbm1 = fbm_v(lonlat_noise(ll, u.time, WARP_SIZE, tile), u.octaves, WARP_SIZE, u.seed, tile);
    d_light += fbm_v(lonlat_noise(ll, u.time, u.size, tile) + vec2<f32>(fbm1), u.octaves, u.size, u.seed, tile) * 0.3;

    let dither_border = (1.0 / u.pixels) * u.dither_size;

    var band = 0;
    if (d_light > u.light_border_1) {
        band = 1;
        if (d_light < u.light_border_1 + dither_border && (dith || u.should_dither == 0u)) {
            band = 0;
        }
    }
    if (d_light > u.light_border_2) {
        band = 2;
        if (d_light < u.light_border_2 + dither_border && (dith || u.should_dither == 0u)) {
            band = 1;
        }
    }

    // Frost caps: one light band brighter than the ground around them.
    let cap = dither_coverage(pole_coverage(ll.y, u.pole_cap, edge_w * 2.0), cell);
    if (cap > 0.5) {
        band = max(band - 1, 0);
    }

    let col = u.colors[band];
    return vec4<f32>(col.rgb, a * col.a);
}
