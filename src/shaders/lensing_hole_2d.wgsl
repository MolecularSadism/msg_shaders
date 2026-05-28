// ============================================================================
// LENSING HOLE 2D SHADER - Gravitational Lensing Screen Effect
// ============================================================================
// Renders a Schwarzschild-style gravitational lens. The quad is a square sized
// in *world units* and centered on the hole's world position; the captured
// scene (the pixel-perfect canvas) is sampled by mapping each fragment's world
// position back into the capture's viewport UV.
//
// Working in world units (rather than normalized viewport UV) means:
//   - No aspect correction: the quad is square in world space, so the disc is
//     round automatically.
//   - Pixel-perfect sampling: an undeflected fragment maps to the canvas texel
//     at its own world position, so the lens lines up 1:1 with the game world.
//   - Cheap math: single divide for the deflection, mul-chain Gaussian ring.
// ============================================================================

#import bevy_sprite::mesh2d_vertex_output::VertexOutput
#import msg_shaders::color_quantize_functions as cq
#import msg_shaders::pixelate_functions as px

struct LensingHoleMaterial {
    shadow_ratio: f32,
    lensing_strength: f32,
    world_radius: f32,
    time: f32,

    photon_ring_width: f32,
    photon_ring_intensity: f32,
    pixel_grid: f32,
    _pad1: f32,

    world_to_uv: vec2<f32>,
    hole_center: vec2<f32>,

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
    let screen_pos = in.position.xy;

    // Centered, isotropic quad coordinates in [-1, 1]. The quad is square in
    // world units, so this needs no aspect correction — the disc stays round.
    let frac = (in.uv - vec2<f32>(0.5)) * 2.0;
    let r = length(frac);

    // Event horizon as a fraction of the quad's half-extent.
    let rs = material.shadow_ratio;

    // ------------------------------------------------------------------
    // EVENT HORIZON: solid black inside rs.
    // ------------------------------------------------------------------
    if r < rs {
        return maybe_quantize(vec4<f32>(material.black_color.rgb, 1.0), screen_pos);
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
    // GRAVITATIONAL LENSING: deflect the sample toward the center.
    // ------------------------------------------------------------------
    // Schwarzschild-style deflection in fraction units: magnitude ~ rs²/(r-rs).
    // The 1/dr falloff (clamped at the rim) stays well-behaved near the horizon,
    // unlike 1/dr² which spikes and pushes the sample far off-screen.
    let dr = max(r - rs, rs * 0.1);
    let deflect = material.lensing_strength * rs * rs / dr;
    let dir = frac / max(length(frac), 1e-5);
    let lensed_frac = frac - dir * deflect;

    // Map the lensed fraction back to a true world offset (`world_radius` is the
    // quad's half-extent in world units), then to the capture's viewport UV.
    // `world_to_uv` is the per-axis inverse viewport size (y negated because
    // world +y is up while the texture's V points down). Because both the quad
    // and the capture share the camera's view, an undeflected fragment maps to
    // the canvas texel at its own world position — pixel-perfect alignment.
    let lensed_world = lensed_frac * material.world_radius;
    var sample_uv = material.hole_center + lensed_world * material.world_to_uv;
    // Optional pixelation: snap the sample UV to a grid. A grid of 0 is a no-op.
    sample_uv = px::pixelate_uv(sample_uv, vec2<f32>(material.pixel_grid, material.pixel_grid));

    // Track whether the deflected sample lands inside the captured viewport.
    // Where it falls off-screen the lens has nothing to show — those fragments
    // stay transparent so the un-warped scene shows through, rather than
    // stamping a black ring around the hole.
    let in_bounds = step(0.0, sample_uv.x) * step(sample_uv.x, 1.0)
                  * step(0.0, sample_uv.y) * step(sample_uv.y, 1.0);
    let clamped_uv = clamp(sample_uv, vec2<f32>(0.0), vec2<f32>(1.0));

    // Sample the captured scene. `in_bounds` zeroes the contribution when the
    // sample is off-screen so clamping doesn't produce a colored border.
    let bg = textureSample(background_tex, background_sampler, clamped_uv).rgb * in_bounds;

    // Edge fade so the hole blends out at its outer rim instead of stamping a
    // hard circle. `r` is the fraction of the quad half-extent.
    let edge_fade = clamp((1.0 - r) * 4.0, 0.0, 1.0);

    let col = bg + material.photon_ring_color.rgb * ring_intensity;
    // Outer-disc alpha: opaque where the lens samples the scene, transparent
    // where it deflects off-screen, with a soft edge near the outer rim. The
    // photon ring contributes its own alpha so it stays visible even where the
    // deflected sample lands off the captured viewport.
    let scene_alpha = in_bounds * edge_fade;
    let ring_alpha = clamp(ring_intensity, 0.0, 1.0) * edge_fade;
    let out_alpha = max(scene_alpha, ring_alpha);
    let output = vec4<f32>(clamp(col, vec3<f32>(0.0), vec3<f32>(1.0)), out_alpha);

    return maybe_quantize(output, screen_pos);
}
