// Nebula + starfield background material.
//
// Draws procedural nebula clouds and a twinkling star field, snapped to the
// game's art-pixel grid (pixel-perfect from screen-space derivatives) and
// reduced to a palette with ordered dithering. The pattern is authored in a
// bounded, rotatable coordinate local to the quad, so it stays precise far from
// the world origin and can be spun for a day/night sky.

#import bevy_sprite::mesh2d_vertex_output::VertexOutput
#import msg_shaders::pixelate_functions as px
#import msg_shaders::color_quantize_functions as cq
#import msg_shaders::nebula_functions as nb

struct NebulaSettings {
    color_a: vec4<f32>,
    color_b: vec4<f32>,
    background: vec4<f32>,
    star_cool: vec4<f32>,
    star_warm: vec4<f32>,
    world_size: vec2<f32>,
    scroll: vec2<f32>,
    nebula_scale: f32,
    density: f32,
    star_density: f32,
    twinkle_speed: f32,
    star_spacing: f32,
    star_brightness: f32,
    rotation: f32,
    time: f32,
    pixel_size: f32,
    seed: f32,
    octaves: u32,
    warp_octaves: u32,
    warp_strength: f32,
    cloud_low: f32,
    cloud_high: f32,
    twinkle_sharpness: f32,
    star_blackout: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};

@group(2) @binding(0)
var<uniform> nebula: NebulaSettings;

// Mirrors `ColorQuantizeUniforms` (palette pre-converted to Oklab on the CPU).
struct QuantizationSettings {
    palette: array<vec4<f32>, 64>,
    palette_oklab: array<vec4<f32>, 64>,
    palette_size: u32,
    alpha_cutoff: f32,
    dither_pattern: u32,
    transparency_floor: f32,
};

@group(2) @binding(1)
var<uniform> quantization: QuantizationSettings;

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let world = in.world_position.xy;

    // World units per art pixel: pixel-perfect from screen derivatives (tracks
    // camera zoom), or an explicit override. The branch is on a uniform, so the
    // derivative call stays under uniform control flow.
    var aps = nebula.pixel_size;
    if aps <= 0.0 {
        aps = px::art_pixel_size(px::world_units_per_pixel(world));
    }

    // Bounded pattern coordinate local to the quad, rotated for the day/night
    // sky and offset by parallax/drift scroll.
    let local = (in.uv - vec2<f32>(0.5, 0.5)) * nebula.world_size;
    let base = px::rotate2d(local, nebula.rotation) + nebula.scroll;

    // Snap to the art-pixel grid so both nebula and stars render at art
    // resolution, and index the grid cell for per-block dithering and stars.
    let snapped = px::pixelate_world(base, aps);
    let cell = px::pixel_cell_index(base, aps);

    var color = nb::nebula_color(
        snapped * nebula.nebula_scale,
        nebula.color_a.rgb,
        nebula.color_b.rgb,
        nebula.background.rgb,
        nebula.density,
        nebula.octaves,
        nebula.warp_octaves,
        nebula.warp_strength,
        nebula.cloud_low,
        nebula.cloud_high,
    );

    // Two star layers of differing spacing for a hint of depth.
    color += nb::star_layer(
        cell, nebula.star_spacing, nebula.time, nebula.seed,
        nebula.twinkle_speed, nebula.star_cool.rgb, nebula.star_warm.rgb,
        nebula.star_density, nebula.twinkle_sharpness, nebula.star_blackout,
    ) * nebula.star_brightness;
    color += nb::star_layer(
        cell + vec2<f32>(131.0, 71.0), nebula.star_spacing * 1.9, nebula.time,
        nebula.seed + 17.0, nebula.twinkle_speed * 0.8, nebula.star_cool.rgb,
        nebula.star_warm.rgb, nebula.star_density * 0.7,
        nebula.twinkle_sharpness, nebula.star_blackout,
    ) * nebula.star_brightness * 0.75;

    if quantization.palette_size == 0u {
        return vec4<f32>(color, 1.0);
    }

    var palette = quantization.palette;
    var palette_oklab = quantization.palette_oklab;
    let quantized = cq::quantize_color_opaque(
        color, cell, &palette, &palette_oklab,
        quantization.palette_size, quantization.dither_pattern,
    );
    return vec4<f32>(quantized, 1.0);
}
