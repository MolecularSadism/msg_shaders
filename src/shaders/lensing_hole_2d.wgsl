// ============================================================================
// LENSING HOLE 2D SHADER - Gravitational Lensing Screen Effect
// ============================================================================
// Renders a Schwarzschild-style gravitational lens around a world-space point
// (`hole_center`). The lens works entirely in world space: each fragment reads
// its own world position from the mesh, deflects it toward the hole, and samples
// the scene-capture canvas — a world-axis-aligned, non-rotating texture centered
// on the camera (`canvas_center`) and spanning the viewport diagonal
// (`canvas_extent`). Because everything is expressed in world coordinates, the
// effect tracks the main camera's full transform (translation, zoom, AND
// rotation) implicitly: the rotation lives in the mesh-to-screen projection, not
// in this shader.
//
// Math is single-pass and cheap:
//   - Lensing displacement: dir * (strength * rs * rs / (r - rs)) — single divide.
//   - Photon ring: mul-chain Gaussian (no exp/pow).
//   - Sampling clamped so off-canvas lensed samples stay transparent, which
//     keeps the math closed and avoids edge-repeat artifacts.
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

    canvas_center: vec2<f32>,
    canvas_extent: vec2<f32>,

    hole_center: vec2<f32>,
    _pad2: vec2<f32>,

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

// Snap a world position to the center of its world-pixel cell so the disc, ring,
// and sampled scene read as discrete world-size squares. With `world_pixel == 0`
// snapping is disabled and the position passes through.
fn snap_world(p: vec2<f32>) -> vec2<f32> {
    if material.world_pixel <= 0.0 {
        return p;
    }
    let wp = material.world_pixel;
    return (floor(p / wp) + vec2<f32>(0.5)) * wp;
}

// Map a world position to the scene-capture canvas UV. The canvas is a square
// world-axis-aligned region centered on `canvas_center`; V is flipped because
// world +y is up while texture V points down.
fn canvas_uv(world: vec2<f32>) -> vec2<f32> {
    let extent = max(material.canvas_extent, vec2<f32>(1e-5, 1e-5));
    let uv = (world - material.canvas_center) / extent + vec2<f32>(0.5);
    return vec2<f32>(uv.x, 1.0 - uv.y);
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
    // The fragment's true world position, recovered from the mesh. This already
    // accounts for the camera's translation, zoom, and rotation, so all lens
    // math below stays in world space and never re-derives the camera.
    let world = snap_world(in.world_position.xy);

    // Dither coordinate in world-pixel cells (one Bayer threshold per world
    // pixel). Biased positive so the integer modulo in the dither lookup never
    // sees a negative; the bias is a multiple of the 4x4/8x8 pattern sizes so it
    // doesn't shift the pattern.
    let wp = max(material.world_pixel, 1.0);
    let dither_pos = floor(in.world_position.xy / wp) + vec2<f32>(1048576.0);

    // Position relative to the hole, normalized by the halo radius (`size`):
    // r == shadow_radius at the event horizon, r == 1 at the outer rim. World
    // space is isotropic, so the disc is round without aspect correction.
    let centered = world - material.hole_center;
    let size = max(material.size, 0.0001);
    let r = length(centered) / size;

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
    // GRAVITATIONAL LENSING: deflect the sampled world position toward center.
    // ------------------------------------------------------------------
    // Smooth Schwarzschild-style deflection: magnitude ~ rs * rs / (r - rs).
    // The extra `rs` factor keeps the deflection proportional to the event-
    // horizon radius. The 1/dr falloff (with a `max` clamp at the rim) stays
    // well-behaved as r approaches the event horizon, unlike 1/dr² which spikes
    // to infinity and pushes the sample far off the canvas.
    let dr = max(r - rs, rs * 0.1);
    let deflect = material.lensing_strength * rs * rs / dr;

    // Direction from center; deflection is scaled back to world units by `size`.
    let dir = centered / max(length(centered), 1e-5);
    let lensed_world = world - dir * deflect * size;

    let sample_uv = canvas_uv(snap_world(lensed_world));

    // Track whether the deflected sample lands inside the captured canvas. Where
    // it falls outside, the lens has nothing to show — those fragments stay
    // transparent so the un-warped scene shows through instead of stamping a
    // black ring around the hole.
    let in_bounds = step(0.0, sample_uv.x) * step(sample_uv.x, 1.0)
                  * step(0.0, sample_uv.y) * step(sample_uv.y, 1.0);
    let clamped_uv = clamp(sample_uv, vec2<f32>(0.0), vec2<f32>(1.0));

    let bg = textureSample(background_tex, background_sampler, clamped_uv).rgb * in_bounds;

    // Edge fade so the hole blends out at its outer rim instead of stamping a
    // hard circle. `r` is in halo-normalized space.
    let edge_fade = clamp((1.0 - r) * 4.0, 0.0, 1.0);

    let col = bg + material.photon_ring_color.rgb * ring_intensity;
    // Outer-disc alpha: opaque where the lens samples the scene, transparent
    // where it deflects off the canvas, with a soft edge near the outer rim. The
    // photon ring contributes its own alpha so it stays visible even where the
    // deflected sample lands off the captured canvas.
    let scene_alpha = in_bounds * edge_fade;
    let ring_alpha = clamp(ring_intensity, 0.0, 1.0) * edge_fade;
    let out_alpha = max(scene_alpha, ring_alpha);
    let output = vec4<f32>(clamp(col, vec3<f32>(0.0), vec3<f32>(1.0)), out_alpha);

    return maybe_quantize(output, dither_pos);
}
