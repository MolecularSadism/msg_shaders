// Standalone color quantization material shader.
// Applies color quantization to a textured mesh/sprite.

#import bevy_sprite::mesh2d_vertex_output::VertexOutput
#import msg_shaders::color_quantize_functions as cq

@group(2) @binding(0)
var texture: texture_2d<f32>;
@group(2) @binding(1)
var texture_sampler: sampler;

struct QuantizationSettings {
    palette: array<vec4<f32>, 64>,
    // Palette colors pre-converted to Oklab on the CPU, eliminating per-pixel
    // cbrt/pow calls inside the palette search loop.
    palette_oklab: array<vec4<f32>, 64>,
    palette_size: u32,
    alpha_cutoff: f32,
    dither_pattern: u32,
    transparency_floor: f32,
};

@group(2) @binding(2)
var<uniform> quantization: QuantizationSettings;

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    // Sample the texture
    let color = textureSample(texture, texture_sampler, in.uv);

    // If no palette is configured, pass through unchanged
    if quantization.palette_size == 0u {
        return color;
    }

    // Use fragment position for dithering pattern
    let screen_pos = in.position.xy;

    // Copy palette arrays to local variables for pointer passing
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
