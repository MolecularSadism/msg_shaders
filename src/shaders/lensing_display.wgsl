// ============================================================================
// LENSING DISPLAY - Full-screen gravitational-lensing post-process
// ============================================================================
// Distorts the lit scene already rendered into the view target. For each screen
// pixel it reconstructs the world position, samples the GPU-simulated deflection
// field at that world position, projects the deflected world position back to
// screen space, and reads the lit scene there. The photon ring and event-horizon
// disc are drawn per-lens on top, in world space.
//
// The photon ring / shadow edge is the only region that pays for world-pixel
// snapping and palette quantization: a fragment outside the ring band returns
// the smoothly-sampled lit scene and never runs the snap or the palette match.
//
// Unlike the canvas-strategy material, nothing is sampled from an offscreen
// scene capture: the source is the live, lit view target, so the lens warps the
// final image at full screen resolution.
// ============================================================================

#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput
#import msg_shaders::color_quantize_functions as cq
#import msg_shaders::pixelate_functions as px

// One lens, packed into four vec4 rows. Must match `LensData` in material.rs.
struct LensData {
    // xy = world center, z = halo radius (`size`), w = shadow_radius.
    center_size_shadow: vec4<f32>,
    // x = lensing_strength, y = photon_ring_width, z = photon_ring_intensity.
    strength_ring: vec4<f32>,
    photon_ring_color: vec4<f32>,
    black_color: vec4<f32>,
};

const MAX_LENSES: u32 = 64u;

// Must match `ColorQuantizeUniforms` field order in quantize_material.rs.
struct QuantizationSettings {
    palette: array<vec4<f32>, 64>,
    palette_oklab: array<vec4<f32>, 64>,
    palette_size: u32,
    alpha_cutoff: f32,
    dither_pattern: u32,
    transparency_floor: f32,
};

struct LensingDisplay {
    clip_from_world: mat4x4<f32>,
    world_from_clip: mat4x4<f32>,
    // xy = canvas world-space center, zw = canvas world-space extent.
    canvas_center_extent: vec4<f32>,
    quantization: QuantizationSettings,
    count: u32,
    lenses: array<LensData, 64>,
};

@group(0) @binding(0) var scene_tex: texture_2d<f32>;
@group(0) @binding(1) var scene_sampler: sampler;
@group(0) @binding(2) var velocity_field_tex: texture_2d<f32>;
@group(0) @binding(3) var velocity_field_sampler: sampler;
@group(0) @binding(4) var<uniform> u: LensingDisplay;

// Map a world position to the field's canvas UV. The canvas is a square,
// world-axis-aligned region centered on `canvas_center_extent.xy`; V is flipped
// because world +y is up while texture V points down.
fn canvas_uv(world: vec2<f32>) -> vec2<f32> {
    let center = u.canvas_center_extent.xy;
    let extent = max(u.canvas_center_extent.zw, vec2<f32>(1e-5, 1e-5));
    let uv = (world - center) / extent + vec2<f32>(0.5);
    return vec2<f32>(uv.x, 1.0 - uv.y);
}

// Reconstruct the world-space position of a screen pixel from its UV.
fn world_from_uv(uv: vec2<f32>) -> vec2<f32> {
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let world = u.world_from_clip * vec4<f32>(ndc, 0.0, 1.0);
    return world.xy / world.w;
}

// Project a world-space position to screen UV.
fn uv_from_world(world: vec2<f32>) -> vec2<f32> {
    let clip = u.clip_from_world * vec4<f32>(world, 0.0, 1.0);
    let ndc = clip.xy / clip.w;
    return vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
}

// Lit scene at a world position: deflect by the field, project back to screen,
// sample the view target (clamped to the visible region).
fn lensed_scene(world: vec2<f32>) -> vec3<f32> {
    let deflect = textureSample(velocity_field_tex, velocity_field_sampler, canvas_uv(world)).rg;
    let sample_uv = clamp(uv_from_world(world - deflect), vec2<f32>(0.0), vec2<f32>(1.0));
    return textureSample(scene_tex, scene_sampler, sample_uv).rgb;
}

// Palette-quantize a color; pass-through when no palette is configured.
fn maybe_quantize(color: vec4<f32>, dither_pos: vec2<f32>) -> vec4<f32> {
    if u.quantization.palette_size == 0u {
        return color;
    }
    var palette = u.quantization.palette;
    var palette_oklab = u.quantization.palette_oklab;
    return cq::quantize_color(
        color,
        dither_pos,
        &palette,
        &palette_oklab,
        u.quantization.palette_size,
        u.quantization.alpha_cutoff,
        u.quantization.transparency_floor,
        u.quantization.dither_pattern,
    );
}

@fragment
fn fragment(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let world = world_from_uv(in.uv);

    // Snap the per-lens photon ring and shadow edge to the displayed art-pixel
    // grid so they read as crisp pixel-art blocks, matching the rest of the scene
    // rather than the finer full-screen display resolution. The lensed background
    // below keeps the un-snapped world position, so only the ring/shadow band is
    // pixelated. Cell size and dither phase come from the live world-to-pixel
    // conversion, never a hardcoded ratio.
    let cell = px::art_pixel_size(px::world_units_per_pixel(world));
    let world_px = px::pixelate_world(world, cell);
    let dither_pos = px::pixel_cell_index(world, cell);

    var ring_accum = vec3<f32>(0.0);
    var ring_strength = 0.0;
    var horizon = false;
    var horizon_color = vec3<f32>(0.0);

    let count = min(u.count, MAX_LENSES);
    for (var i = 0u; i < count; i = i + 1u) {
        let lens = u.lenses[i];
        let center = lens.center_size_shadow.xy;
        let size = max(lens.center_size_shadow.z, 1e-4);
        let rs = lens.center_size_shadow.w;

        let r_raw = length(world - center) / size; // rs at horizon, 1 at outer rim.
        if r_raw > 1.0 {
            continue;
        }

        // Snapped radius: a whole art-pixel cell shares one radius, so the horizon
        // disc and ring band step in pixel blocks instead of smooth circles.
        let r = length(world_px - center) / size;

        // EVENT HORIZON: solid black inside rs (any lens wins).
        if r < rs {
            horizon = true;
            horizon_color = lens.black_color.rgb;
        }

        // PHOTON RING: Gaussian glow just outside rs via mul-chain.
        let ring_t = clamp(1.0 - (r - rs) / (lens.strength_ring.y * 6.0), 0.0, 1.0);
        let ring_t2 = ring_t * ring_t;
        let ring_i = ring_t2 * ring_t2 * lens.strength_ring.z;
        ring_accum += lens.photon_ring_color.rgb * ring_i;
        ring_strength = max(ring_strength, ring_i);
    }

    // Interior of the event horizon is a solid color, palette-quantized so the
    // shadow and its pixel-snapped edge sit in the same palette as the ring.
    if horizon {
        return maybe_quantize(vec4<f32>(horizon_color, 1.0), dither_pos);
    }

    // The lit scene plus any ring glow. Fragments outside the ring band stop
    // here, so the palette match below runs only on the ring.
    let ring_mask = clamp(ring_strength, 0.0, 1.0);
    let scene = vec4<f32>(lensed_scene(world) + ring_accum, 1.0);
    if ring_mask <= 0.0 {
        return scene;
    }

    // Photon ring / shadow edge only: palette-quantize with one Bayer threshold
    // per art-pixel cell, mixed in by the ring mask.
    let quantized = maybe_quantize(scene, dither_pos);
    return mix(scene, quantized, ring_mask);
}
