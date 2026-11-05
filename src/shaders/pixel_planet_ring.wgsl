// Planetary ring (Deep-Fold/PixelPlanets `GasPlanetLayers/Ring`, MIT).
//
// Drawn on a quad larger than the planet body; the body occupies a fraction
// `1.0 / scale_rel_to_planet` of this quad, and the ring's back half is masked
// where it would pass behind that disc.

#import bevy_sprite::mesh2d_vertex_output::VertexOutput
#import msg_shaders::pixel_planet_functions::{fbm_v, rotate_uv, dither_coverage}

struct U {
    colors: array<vec4<f32>, 3>,
    dark_colors: array<vec4<f32>, 3>,
    light_origin: vec2<f32>,
    pixels: f32,
    rotation: f32,
    size: f32,
    seed: f32,
    time: f32,
    ring_width: f32,
    ring_perspective: f32,
    scale_rel_to_planet: f32,
    edge_dither: f32,
    octaves: i32,
    n_colors: i32,
}

@group(2) @binding(0) var<uniform> u: U;

// Maps the shared edge-dither strength (a pixel-cell rim width for the planet
// body) into the ring's noise-value space, where the visible edge is the
// crossing of `ring` past its alpha threshold.
const EDGE_DITHER_RING_SCALE: f32 = 0.04;

@fragment
fn fragment(mesh: VertexOutput) -> @location(0) vec4<f32> {
    let tile = vec2<f32>(2.0, 1.0);

    var uv = floor(mesh.uv * u.pixels) / u.pixels;
    let cell = floor(mesh.uv * u.pixels);

    let light_d = distance(uv, u.light_origin);
    uv = rotate_uv(uv, u.rotation);

    // center is used to determine ring position
    var uv_center = uv - vec2<f32>(0.0, 0.5);

    // tilt ring
    uv_center *= vec2<f32>(1.0, u.ring_perspective);
    let center_d = distance(uv_center, vec2<f32>(0.5, 0.0));

    // cut out 2 circles of different sizes and keep only their intersection
    var ring = smoothstep(0.5 - u.ring_width * 2.0, 0.5 - u.ring_width, center_d);
    ring *= smoothstep(center_d - u.ring_width, center_d, 0.4);

    // pretend the ring goes behind the planet by removing it in the upper half
    if (uv.y < 0.5) {
        ring *= step(1.0 / u.scale_rel_to_planet, distance(uv, vec2<f32>(0.5)));
    }

    // rotate material in the ring
    uv_center = rotate_uv(uv_center + vec2<f32>(0.0, 0.5), u.time);
    ring *= fbm_v(uv_center * u.size, u.octaves, u.size, u.seed, tile);

    // apply colors based on final value
    var posterized = floor((ring + pow(light_d, 2.0) * 2.0) * 4.0) / 4.0;
    posterized = min(posterized, 2.0);
    var col: vec4<f32>;
    if (posterized <= 1.0) {
        col = u.colors[clamp(i32(posterized * f32(u.n_colors - 1)), 0, 2)];
    } else {
        col = u.dark_colors[clamp(i32((posterized - 1.0) * f32(u.n_colors - 1)), 0, 2)];
    }

    let ring_w = u.edge_dither * EDGE_DITHER_RING_SCALE;
    let coverage = clamp((ring - 0.28 + ring_w) / max(2.0 * ring_w, 1e-6), 0.0, 1.0);
    let ring_a = dither_coverage(coverage, cell);
    return vec4<f32>(col.rgb, ring_a * col.a);
}
