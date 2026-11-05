// Banded gas giant body (Deep-Fold/PixelPlanets `GasLayers`, MIT).

#import bevy_sprite::mesh2d_vertex_output::VertexOutput
#import msg_shaders::pixel_planet_functions::{fbm_v, spherify, rotate_uv, dither_bool, circle_noise, dither_coverage}

struct U {
    colors: array<vec4<f32>, 3>,
    dark_colors: array<vec4<f32>, 3>,
    light_origin: vec2<f32>,
    pixels: f32,
    rotation: f32,
    size: f32,
    seed: f32,
    time: f32,
    light_border_1: f32,
    light_border_2: f32,
    bands: f32,
    edge_dither: f32,
    octaves: i32,
    n_colors: i32,
    should_dither: u32,
}

@group(2) @binding(0) var<uniform> u: U;

fn turbulence(uv: vec2<f32>) -> f32 {
    let tile = vec2<f32>(2.0, 1.0);
    var c_noise = 0.0;
    for (var i = 0; i < 10; i = i + 1) {
        c_noise += circle_noise(
            (uv * u.size * 0.3) + vec2<f32>(f32(i + 1) + 10.0) + vec2<f32>(u.time, 0.0),
            u.size, u.seed, tile,
        );
    }
    return c_noise;
}

@fragment
fn fragment(mesh: VertexOutput) -> @location(0) vec4<f32> {
    let tile = vec2<f32>(2.0, 1.0);

    var uv = floor(mesh.uv * u.pixels) / u.pixels;
    let light_d = distance(uv, u.light_origin);

    let dith = dither_bool(uv, mesh.uv, u.pixels);
    let d_circle = distance(uv, vec2<f32>(0.5));
    let cell = floor(mesh.uv * u.pixels);
    let edge_w = (1.0 / u.pixels) * u.edge_dither;
    let coverage = clamp((0.49999 - d_circle) / max(edge_w, 1e-6), 0.0, 1.0);
    let a = dither_coverage(coverage, cell);

    uv = rotate_uv(uv, u.rotation);
    uv = spherify(uv);

    // a band is just one dimensional noise
    let band = fbm_v(vec2<f32>(0.0, uv.y * u.size * u.bands), u.octaves, u.size, u.seed, tile);
    let turb = turbulence(uv);

    let fbm1 = fbm_v(uv * u.size, u.octaves, u.size, u.seed, tile);
    var fbm2 = fbm_v(
        uv * vec2<f32>(1.0, 2.0) * u.size + vec2<f32>(fbm1) + vec2<f32>(-u.time, 0.0) + vec2<f32>(turb),
        u.octaves, u.size, u.seed, tile,
    );

    fbm2 *= pow(band, 2.0) * 7.0;
    let light = fbm2 + light_d * 1.8;
    fbm2 += pow(light_d, 1.0) - 0.3;
    fbm2 = smoothstep(-0.2, 4.0 - fbm2, light);

    if (dith && u.should_dither != 0u) {
        fbm2 *= 1.1;
    }

    let posterized = floor(fbm2 * 4.0) / 2.0;
    var col = vec4<f32>(vec3<f32>(0.0), 1.0);
    if (fbm2 < 0.625) {
        col = u.colors[clamp(i32(posterized * f32(u.n_colors - 1)), 0, 2)];
    } else {
        col = u.dark_colors[clamp(i32((posterized - 1.0) * f32(u.n_colors - 1)), 0, 2)];
    }

    return vec4<f32>(col.rgb, a * col.a);
}
