// Layered nebula + starfield background material.
//
// Composites a stack of procedural nebula layers (each its own noise settings,
// three-stop color ramp, and parallax) over a base color, then adds a stack of
// twinkling star layers. Everything is snapped to the game's art-pixel grid
// (pixel-perfect from screen-space derivatives) and reduced to a palette with
// ordered dithering. The pattern is authored in a bounded coordinate local to
// the quad, so it stays precise far from the world origin.

#import bevy_sprite::mesh2d_vertex_output::VertexOutput
#import msg_shaders::pixelate_functions as px
#import msg_shaders::color_quantize_functions as cq
#import msg_shaders::nebula_functions as nb

// Keep in sync with MAX_NEBULA_LAYERS / MAX_STAR_LAYERS in nebula.rs.
const MAX_NEBULA_LAYERS: u32 = 6u;
const MAX_STAR_LAYERS: u32 = 4u;

struct NebulaLayer {
    c1: vec4<f32>,
    c2: vec4<f32>,
    c3: vec4<f32>,
    offset: vec2<f32>,
    scale: f32,
    warp_strength: f32,
    cloud_low: f32,
    cloud_high: f32,
    intensity: f32,
    parallax: f32,
    rotation: f32,
    octaves: u32,
    warp_octaves: u32,
    _pad0: f32,
};

struct StarLayer {
    // params: (spacing, density, brightness, parallax)
    params: vec4<f32>,
    // seed_pad: (seed, _, _, _)
    seed_pad: vec4<f32>,
};

struct NebulaSettings {
    layers: array<NebulaLayer, 6>,
    stars: array<StarLayer, 4>,
    background: vec4<f32>,
    star_cool: vec4<f32>,
    star_warm: vec4<f32>,
    world_size: vec2<f32>,
    scroll: vec2<f32>,
    num_layers: u32,
    num_stars: u32,
    blend: u32,
    twinkle_speed: f32,
    twinkle_sharpness: f32,
    star_blackout: f32,
    rotation: f32,
    time: f32,
    pixel_size: f32,
    seed: f32,
    _pad0: f32,
    _pad1: f32,
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
    // zoom), or an explicit override. The branch is on a uniform.
    var aps = nebula.pixel_size;
    if aps <= 0.0 {
        aps = px::art_pixel_size(px::world_units_per_pixel(world));
    }

    // Bounded pattern coordinate local to the quad, with the global rotation
    // (e.g. a day/night sky) applied once.
    let local = (in.uv - vec2<f32>(0.5, 0.5)) * nebula.world_size;
    let base = px::rotate2d(local, nebula.rotation);

    // One dither grid for the whole image (parallax-independent), so blocks
    // quantize consistently.
    let dither_cell = px::pixel_cell_index(base, aps);

    var color = nebula.background.rgb;

    // Nebula layers, back to front. Each has its own parallax drift, a
    // decorrelation offset/rotation, and a three-stop ramp keyed by density.
    for (var i = 0u; i < min(nebula.num_layers, MAX_NEBULA_LAYERS); i = i + 1u) {
        let layer = nebula.layers[i];
        let lp = px::rotate2d(base, layer.rotation) + nebula.scroll * layer.parallax + layer.offset;
        let snapped = px::pixelate_world(lp, aps);
        let d = nb::nebula_density(
            snapped * layer.scale,
            layer.octaves,
            layer.warp_octaves,
            layer.warp_strength,
            layer.cloud_low,
            layer.cloud_high,
        );
        let tint = nb::ramp3(d, layer.c1.rgb, layer.c2.rgb, layer.c3.rgb);
        let coverage = clamp(d * layer.intensity, 0.0, 1.0);
        if nebula.blend == 1u {
            color += tint * d * layer.intensity;
        } else {
            color = mix(color, tint, coverage);
        }
    }

    // Star layers on top, each drifting at its own parallax rate.
    for (var j = 0u; j < min(nebula.num_stars, MAX_STAR_LAYERS); j = j + 1u) {
        let star = nebula.stars[j];
        let sp = base + nebula.scroll * star.params.w;
        let cell = px::pixel_cell_index(sp, aps);
        color += nb::star_layer(
            cell,
            star.params.x,
            nebula.time,
            nebula.seed + star.seed_pad.x,
            nebula.twinkle_speed,
            nebula.star_cool.rgb,
            nebula.star_warm.rgb,
            star.params.y,
            nebula.twinkle_sharpness,
            nebula.star_blackout,
        ) * star.params.z;
    }

    if quantization.palette_size == 0u {
        return vec4<f32>(color, 1.0);
    }

    var palette = quantization.palette;
    var palette_oklab = quantization.palette_oklab;
    let quantized = cq::quantize_color_opaque(
        color, dither_cell, &palette, &palette_oklab,
        quantization.palette_size, quantization.dither_pattern,
    );
    return vec4<f32>(quantized, 1.0);
}
