// Standalone pixelation material shader.
// Samples a texture through a snapped grid so the result looks blocky.

#import bevy_sprite::mesh2d_vertex_output::VertexOutput
#import msg_shaders::pixelate_functions as px

@group(2) @binding(0)
var texture: texture_2d<f32>;
@group(2) @binding(1)
var texture_sampler: sampler;

struct PixelateSettings {
    // Number of cells across each axis. A component of 0 disables snapping.
    grid: vec2<f32>,
    _pad: vec2<f32>,
};

@group(2) @binding(2)
var<uniform> pixelate: PixelateSettings;

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let uv = px::pixelate_uv(in.uv, pixelate.grid);
    return textureSample(texture, texture_sampler, uv);
}
