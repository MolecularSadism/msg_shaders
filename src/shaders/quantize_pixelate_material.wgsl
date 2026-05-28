// Combined pixelation + color quantization material shader.
// Snaps the sample to a grid, then reduces the sampled color to a palette
// with optional dithering. The two effects compose in a single pass.

#import bevy_sprite::mesh2d_vertex_output::VertexOutput
#import msg_shaders::pixelate_functions as px
#import msg_shaders::color_quantize_functions as cq

@group(2) @binding(0)
var texture: texture_2d<f32>;
@group(2) @binding(1)
var texture_sampler: sampler;

struct QuantizationSettings {
    palette: array<vec4<f32>, 64>,
    palette_oklab: array<vec4<f32>, 64>,
    palette_size: u32,
    alpha_cutoff: f32,
    dither_pattern: u32,
    transparency_floor: f32,
};

@group(2) @binding(2)
var<uniform> quantization: QuantizationSettings;

struct PixelateSettings {
    grid: vec2<f32>,
    _pad: vec2<f32>,
};

@group(2) @binding(3)
var<uniform> pixelate: PixelateSettings;

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let uv = px::pixelate_uv(in.uv, pixelate.grid);
    let color = textureSample(texture, texture_sampler, uv);

    if quantization.palette_size == 0u {
        return color;
    }

    let screen_pos = in.position.xy;
    var palette = quantization.palette;
    var palette_oklab = quantization.palette_oklab;

    return cq::quantize_color(
        color,
        screen_pos,
        &palette,
        &palette_oklab,
        quantization.palette_size,
        quantization.alpha_cutoff,
        quantization.transparency_floor,
        quantization.dither_pattern
    );
}
