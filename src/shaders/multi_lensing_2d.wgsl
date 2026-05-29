// ============================================================================
// MULTI-LENSING 2D SHADER - Combined Gravitational Lensing Flow Field
// ============================================================================
// Renders many Schwarzschild-style lenses in a single pass over one canvas-
// covering quad. Each fragment sums every active lens's deflection into one
// flow field, then samples the scene-capture canvas ONCE — so overlapping
// lenses mutually warp the space between them instead of stacking independent
// distortions.
//
// Like the single-lens variant, everything is world space: the fragment reads
// its own world position from the mesh and samples a world-axis-aligned, non-
// rotating capture texture centered on the camera (`canvas_center`) spanning
// the viewport diagonal (`canvas_extent`). Camera translation/zoom/rotation are
// handled by the mesh→screen projection, never re-derived here.
//
// Cost is kept low by two culling layers: the CPU gather caps the active set and
// drops off-screen/negligible lenses, and the per-fragment `r > 1.0` early-out
// skips any lens whose halo does not cover this fragment. The loop bound is the
// dynamic `count`, so unused array slots are free.
// ============================================================================

#import bevy_sprite::mesh2d_vertex_output::VertexOutput
#import msg_shaders::color_quantize_functions as cq

// One lens, packed into four vec4 rows. Must match `LensData` in material.rs.
struct LensData {
    // xy = world center, z = halo radius (`size`), w = shadow_radius.
    center_size_shadow: vec4<f32>,
    // x = lensing_strength, y = photon_ring_width, z = photon_ring_intensity.
    strength_ring: vec4<f32>,
    photon_ring_color: vec4<f32>,
    black_color: vec4<f32>,
};

// Must match `MAX_LENSES` in material.rs.
const MAX_LENSES: u32 = 64u;

struct MultiLensing {
    canvas_center: vec2<f32>,
    canvas_extent: vec2<f32>,
    count: u32,
    lenses: array<LensData, 64>,
};

struct QuantizationSettings {
    palette: array<vec4<f32>, 64>,
    palette_oklab: array<vec4<f32>, 64>,
    palette_size: u32,
    alpha_cutoff: f32,
    dither_pattern: u32,
    transparency_floor: f32,
};

@group(2) @binding(0) var<uniform> material: MultiLensing;
@group(2) @binding(1) var<uniform> quantization: QuantizationSettings;
@group(2) @binding(2) var background_tex: texture_2d<f32>;
@group(2) @binding(3) var background_sampler: sampler;

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
    // True world position from the mesh; all lens math stays in world space.
    let world = in.world_position.xy;
    // One Bayer threshold per world pixel (the canvas is world-pixel aligned).
    let dither_pos = floor(in.position.xy);

    var total_deflect = vec2<f32>(0.0);
    var ring_accum = vec3<f32>(0.0);
    var ring_strength = 0.0;
    var coverage = 0.0;
    var horizon = false;
    var horizon_color = vec3<f32>(0.0);

    let count = min(material.count, MAX_LENSES);
    for (var i = 0u; i < count; i = i + 1u) {
        let lens = material.lenses[i];
        let center = lens.center_size_shadow.xy;
        let size = max(lens.center_size_shadow.z, 1e-4);
        let rs = lens.center_size_shadow.w;

        let centered = world - center;
        let dist = length(centered);
        let r = dist / size; // halo-normalized: rs at horizon, 1 at outer rim.

        // Per-fragment cull: this lens's halo does not cover the fragment.
        if r > 1.0 {
            continue;
        }

        // EVENT HORIZON: solid black inside rs (any lens wins).
        if r < rs {
            horizon = true;
            horizon_color = lens.black_color.rgb;
        }

        // PHOTON RING: Gaussian glow just outside rs via mul-chain.
        let ring_w = lens.strength_ring.y;
        let ring_dist = r - rs;
        let ring_t = clamp(1.0 - ring_dist / (ring_w * 6.0), 0.0, 1.0);
        let ring_t2 = ring_t * ring_t;
        let ring_t4 = ring_t2 * ring_t2;
        let ring_i = ring_t4 * lens.strength_ring.z;
        ring_accum += lens.photon_ring_color.rgb * ring_i;
        ring_strength = max(ring_strength, ring_i);

        // GRAVITATIONAL LENSING: deflection contribution (the flow field — sums).
        // Smooth Schwarzschild-style magnitude ~ rs * rs / (r - rs), clamped at
        // the rim so it stays well-behaved near the horizon.
        let dr = max(r - rs, rs * 0.1);
        let deflect = lens.strength_ring.x * rs * rs / dr;
        let dir = centered / max(dist, 1e-5);
        total_deflect += dir * deflect * size;

        // Soft edge fade toward the outer rim; the union covers all lenses.
        coverage = max(coverage, clamp((1.0 - r) * 4.0, 0.0, 1.0));
    }

    if horizon {
        return maybe_quantize(vec4<f32>(horizon_color, 1.0), dither_pos);
    }

    let lensed_world = world - total_deflect;
    let sample_uv = canvas_uv(lensed_world);
    let in_bounds = step(0.0, sample_uv.x) * step(sample_uv.x, 1.0)
                  * step(0.0, sample_uv.y) * step(sample_uv.y, 1.0);
    let clamped_uv = clamp(sample_uv, vec2<f32>(0.0), vec2<f32>(1.0));
    let bg = textureSample(background_tex, background_sampler, clamped_uv).rgb * in_bounds;

    let col = bg + ring_accum;
    // Opaque where the field samples the scene, transparent where it deflects
    // off-canvas, with a soft outer-rim edge. The ring contributes its own alpha
    // so it stays visible even where the deflected sample lands off-canvas.
    let scene_alpha = in_bounds * coverage;
    let ring_alpha = clamp(ring_strength, 0.0, 1.0) * coverage;
    let out_alpha = max(scene_alpha, ring_alpha);
    let output = vec4<f32>(clamp(col, vec3<f32>(0.0), vec3<f32>(1.0)), out_alpha);

    return maybe_quantize(output, dither_pos);
}
