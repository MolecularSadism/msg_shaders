// Cloud layer (Deep-Fold/PixelPlanets `Clouds`, MIT).

#import bevy_sprite::mesh2d_vertex_output::VertexOutput
#import msg_shaders::pixel_planet_functions::{fbm_v, spherify, rotate_uv, circle_noise, dither_coverage}

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
    cloud_cover: f32,
    stretch: f32,
    cloud_curve: f32,
    edge_dither: f32,
    octaves: i32,
}

@group(2) @binding(0) var<uniform> u: U;

fn cloud_alpha(uv: vec2<f32>) -> f32 {
    let tile = vec2<f32>(1.0, 1.0);
    var c_noise = 0.0;
    // more iterations for more turbulence
    for (var i = 0; i < 9; i = i + 1) {
        c_noise += circle_noise(
            (uv * u.size * 0.3) + vec2<f32>(f32(i + 1) + 10.0) + vec2<f32>(u.time, 0.0),
            u.size, u.seed, tile,
        );
    }
    return fbm_v(uv * u.size + vec2<f32>(c_noise) + vec2<f32>(u.time, 0.0), u.octaves, u.size, u.seed, tile);
}

@fragment
fn fragment(mesh: VertexOutput) -> @location(0) vec4<f32> {
    var uv = floor(mesh.uv * u.pixels) / u.pixels;

    let d_light = distance(uv, u.light_origin);
    let d_to_center = distance(uv, vec2<f32>(0.5));
    let cell = floor(mesh.uv * u.pixels);
    let edge_w = (1.0 / u.pixels) * u.edge_dither;
    let coverage = clamp((0.49999 - d_to_center) / max(edge_w, 1e-6), 0.0, 1.0);
    let a = dither_coverage(coverage, cell);

    uv = rotate_uv(uv, u.rotation);
    uv = spherify(uv);
    // slightly make uv go down on the right, and up on the left
    uv.y += smoothstep(0.0, u.cloud_curve, abs(uv.x - 0.4));

    let c = cloud_alpha(uv * vec2<f32>(1.0, u.stretch));

    var col = u.colors[0];
    if (c < u.cloud_cover + 0.03) {
        col = u.colors[1];
    }
    if (d_light + c * 0.2 > u.light_border_1) {
        col = u.colors[2];
    }
    if (d_light + c * 0.2 > u.light_border_2) {
        col = u.colors[3];
    }

    let c2 = c * step(d_to_center, 0.5);
    return vec4<f32>(col.rgb, step(u.cloud_cover, c2) * a * col.a);
}
