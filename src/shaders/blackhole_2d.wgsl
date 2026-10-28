// ============================================================================
// BLACK HOLE 2D MATERIAL
// ============================================================================
// Mesh entry point for `BlackHoleMaterial`; the black hole itself is drawn by
// `msg_shaders::blackhole_functions::shade`, shared with the lensing display
// pass.
//
// Based on Eric Bruneton's black_hole_shader (BSD-3-Clause)
// ============================================================================

#import bevy_sprite::mesh2d_vertex_output::VertexOutput
#import msg_shaders::pixelate_functions as px
#import msg_shaders::blackhole_functions as bh

// Color quantization settings (optional - palette_size=0 disables)
struct QuantizationSettings {
    palette: array<vec4<f32>, 64>,
    // Palette pre-converted to Oklab on the CPU so the shader skips per-pixel cbrt/pow.
    palette_oklab: array<vec4<f32>, 64>,
    palette_size: u32,
    alpha_cutoff: f32,
    dither_pattern: u32,
    transparency_floor: f32,
};

@group(2) @binding(0)
var<uniform> material: bh::BlackHoleMaterial;

@group(2) @binding(1)
var<uniform> quantization: QuantizationSettings;

// Baked nearest-palette LUT (Rgba32Float). Point-loaded (no sampler) to replace
// the per-pixel Oklab palette loop with one fetch at each quantization site.
@group(2) @binding(2)
var palette_lut: texture_3d<f32>;

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    // Optional pixelation: snap the quad UV to a grid so the whole effect
    // renders as discrete blocks. A grid of 0 passes the UV through unchanged.
    let src_uv = px::pixelate_uv(in.uv, vec2<f32>(material.pixel_grid, material.pixel_grid));
    let quantize = bh::QuantizeParams(
        quantization.palette_size,
        quantization.alpha_cutoff,
        quantization.dither_pattern,
        quantization.transparency_floor,
    );
    return bh::shade(src_uv, in.position.xy, material, quantize, palette_lut);
}
