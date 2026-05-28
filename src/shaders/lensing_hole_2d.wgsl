// ============================================================================
// LENSING HOLE 2D SHADER - Gravitational Lensing Screen Effect
// ============================================================================
// Renders a Schwarzschild-style gravitational lens centered on `hole_center`
// (in viewport UV space; defaults to the screen center).
// Samples a background texture (the pixel-perfect canvas) with lensed
// screen-space UVs so the actual game world appears distorted around the hole.
//
// Math is single-pass and cheap:
//   - Lensing displacement: dir * (strength / (r - rs)) — single divide, no loop.
//   - Photon ring: mul-chain Gaussian (no exp/pow).
//   - Sampling clamped so off-screen lensed UVs return black instead of edge
//     repeating, which keeps the math closed and avoids artifacts.
// ============================================================================

#import bevy_sprite::mesh2d_vertex_output::VertexOutput
#import msg_shaders::color_quantize_functions as cq

struct LensingHoleMaterial {
    shadow_radius: f32,
    lensing_strength: f32,
    size: f32,
    time: f32,

    photon_ring_width: f32,
    photon_ring_intensity: f32,
    world_pixel: f32,
    _pad1: f32,

    uv_scale: vec2<f32>,
    hole_center: vec2<f32>,

    cells_per_uv: vec2<f32>,
    cell_phase: vec2<f32>,

    photon_ring_color: vec4<f32>,
    black_color: vec4<f32>,
};

struct QuantizationSettings {
    palette: array<vec4<f32>, 64>,
    palette_oklab: array<vec4<f32>, 64>,
    palette_size: u32,
    alpha_cutoff: f32,
    dither_pattern: u32,
    transparency_floor: f32,
};

@group(2) @binding(0) var<uniform> material: LensingHoleMaterial;
@group(2) @binding(1) var<uniform> quantization: QuantizationSettings;
@group(2) @binding(2) var background_tex: texture_2d<f32>;
@group(2) @binding(3) var background_sampler: sampler;

// World-pixel cell index containing a viewport-UV position. `cells_per_uv` and
// `cell_phase` describe the world grid in viewport-UV terms (computed on the CPU
// from the camera so large world coordinates never reach the shader). The
// integer result steps once per world pixel.
fn world_cell(uv: vec2<f32>) -> vec2<f32> {
    return floor(material.cell_phase + uv * material.cells_per_uv);
}

// Snap a viewport-UV position to the center of its world-pixel cell. With
// `world_pixel == 0` snapping is disabled and the position passes through.
fn snap_uv(uv: vec2<f32>) -> vec2<f32> {
    if material.world_pixel <= 0.0 {
        return uv;
    }
    let cell = world_cell(uv);
    return (cell + vec2<f32>(0.5) - material.cell_phase) / material.cells_per_uv;
}

// Quantize the result if a palette is provided, otherwise pass-through.
fn maybe_quantize(color: vec4<f32>, screen_pos: vec2<f32>) -> vec4<f32> {
    if quantization.palette_size == 0u {
        return color;
    }
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
        quantization.dither_pattern,
    );
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    // Map the quad's own UV onto the captured viewport. The quad is centered on
    // the hole's world position, which projects to `hole_center` in viewport UV;
    // `uv_scale` (= quad size / viewport size, per axis) rescales its [0,1] UV so
    // the quad center lands on `hole_center` and the edges line up with the
    // captured scene. Keying off the mesh UV instead of the framebuffer position
    // keeps the sampled scene aligned regardless of window size, camera viewport
    // offset, or compositing overlays.
    let uv_scale = max(material.uv_scale, vec2<f32>(1e-5, 1e-5));
    let viewport_uv = material.hole_center + (in.uv - vec2<f32>(0.5)) * uv_scale;

    // Snap the fragment's sampled position to the world-pixel grid so the disc,
    // ring, and sampled scene all read as discrete world-size squares rather
    // than fine framebuffer pixels.
    let screen_uv = snap_uv(viewport_uv);

    // Dither coordinate in world-pixel cells (one Bayer threshold per world
    // pixel, not per framebuffer pixel). Biased positive so the integer modulo
    // in the dither lookup never sees a negative; the bias is a multiple of the
    // 4x4/8x8 pattern sizes so it doesn't shift the pattern.
    let dither_pos = world_cell(viewport_uv) + vec2<f32>(1048576.0);

    // Aspect-correct centered coords so the disc stays round. `uv_scale.y /
    // uv_scale.x` equals the viewport's width/height ratio. `hole_center` is the
    // lens center in viewport UV space (0.5,0.5 = screen center); shifting it
    // moves the whole effect on screen while keeping the disc round.
    let aspect = uv_scale.y / uv_scale.x;
    let centered = vec2<f32>((screen_uv.x - material.hole_center.x) * aspect, screen_uv.y - material.hole_center.y);

    // `size` (0..1) scales the whole effect: when 0 the hole is invisible.
    let size = max(material.size, 0.0001);
    let scaled = centered / size;
    let r = length(scaled);

    let rs = material.shadow_radius;

    // ------------------------------------------------------------------
    // EVENT HORIZON: solid black inside rs.
    // ------------------------------------------------------------------
    if r < rs {
        return maybe_quantize(vec4<f32>(material.black_color.rgb, 1.0), dither_pos);
    }

    // ------------------------------------------------------------------
    // PHOTON RING: Gaussian glow just outside rs via mul-chain.
    // ------------------------------------------------------------------
    let ring_dist = r - rs;
    let ring_w = material.photon_ring_width;
    let ring_t = clamp(1.0 - ring_dist / (ring_w * 6.0), 0.0, 1.0);
    let ring_t2 = ring_t * ring_t;
    let ring_t4 = ring_t2 * ring_t2;
    let ring_intensity = ring_t4 * material.photon_ring_intensity;

    // ------------------------------------------------------------------
    // GRAVITATIONAL LENSING: deflect screen-space sample toward center.
    // ------------------------------------------------------------------
    // Smooth Schwarzschild-style deflection: magnitude ~ rs * rs / (r - rs).
    // The extra `rs` factor keeps the deflection proportional to the event-
    // horizon radius, so the deflection scale, the photon ring, and the black
    // disc all sit at the same multiple of `rs` and track each other at every
    // `size`. The 1/dr falloff (with a `max` clamp at the rim) stays well-
    // behaved as r approaches the event horizon, unlike 1/dr² which spikes to
    // infinity and pushes the sample uv far off-screen.
    let dr = max(r - rs, rs * 0.1);
    let deflect = material.lensing_strength * rs * rs / dr;

    // Direction from center, applied to the *unscaled* centered coords so
    // the displacement magnitude in screen UV matches `size`.
    let dir = centered / max(length(centered), 1e-5);
    let lensed_centered = centered - dir * deflect * size;

    // Map back to screen UV [0,1], then snap to the world-pixel grid so the
    // distorted scene is sampled in the same world-size squares as the disc.
    let sample_uv = snap_uv(vec2<f32>(lensed_centered.x / aspect, lensed_centered.y) + material.hole_center);

    // Track whether the deflected sample lands inside the captured viewport.
    // Where it falls off-screen the lens has nothing to show — those fragments
    // should be fully transparent so the un-warped scene shows through, rather
    // than stamping a black ring around the hole.
    let in_bounds = step(0.0, sample_uv.x) * step(sample_uv.x, 1.0)
                  * step(0.0, sample_uv.y) * step(sample_uv.y, 1.0);
    let clamped_uv = clamp(sample_uv, vec2<f32>(0.0), vec2<f32>(1.0));

    // Sample the captured scene. `in_bounds` zeroes the contribution when the
    // sample is off-screen so clamping doesn't produce a colored border.
    let bg = textureSample(background_tex, background_sampler, clamped_uv).rgb * in_bounds;

    // Edge fade so the hole blends out at its outer rim instead of stamping a
    // hard circle. `r` is in scaled centered space (post-`size` divide).
    let edge_fade = clamp((1.0 - r) * 4.0, 0.0, 1.0);

    let col = bg + material.photon_ring_color.rgb * ring_intensity;
    // Outer-disc alpha: opaque where the lens samples the scene, transparent
    // where it deflects off-screen, with a soft edge near the outer rim.
    // The photon ring contributes its own alpha so the ring stays visible
    // even where the deflected sample lands off the captured viewport.
    let scene_alpha = in_bounds * edge_fade;
    let ring_alpha = clamp(ring_intensity, 0.0, 1.0) * edge_fade;
    let out_alpha = max(scene_alpha, ring_alpha);
    let output = vec4<f32>(clamp(col, vec3<f32>(0.0), vec3<f32>(1.0)), out_alpha);

    return maybe_quantize(output, dither_pos);
}
