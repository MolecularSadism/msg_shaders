// ============================================================================
// LENSING 2D SHADER - Combined Gravitational Lensing Flow Field
// ============================================================================
// Renders any number of Schwarzschild-style lenses in a single pass over one
// canvas-covering quad.  Background deflection comes from a GPU-simulated
// velocity (flow-field) texture written by compute shaders each frame.  The
// per-lens loop is still executed for the photon ring and event horizon disc,
// which must remain sharp and per-lens-colored.
//
// If the velocity field texture is not yet available (first frame or no
// simulation running), the background is sampled straight (zero deflection).
//
// Everything is world space: the fragment reads its own world position from the
// mesh and samples a world-axis-aligned, non-rotating capture texture centered
// on the camera (`canvas_center`) spanning the viewport diagonal
// (`canvas_extent`).
// ============================================================================

#import bevy_sprite::mesh2d_vertex_output::VertexOutput
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

// Must match `MAX_LENSES` in material.rs.
const MAX_LENSES: u32 = 64u;

struct Lensing {
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

@group(2) @binding(0) var<uniform> material: Lensing;
@group(2) @binding(1) var<uniform> quantization: QuantizationSettings;
@group(2) @binding(2) var background_tex: texture_2d<f32>;
@group(2) @binding(3) var background_sampler: sampler;
@group(2) @binding(4) var velocity_field_tex: texture_2d<f32>;
@group(2) @binding(5) var velocity_field_sampler: sampler;

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
    // Pixel size for the photon ring and shadow edge: the world span of one
    // displayed art pixel, from the live world-to-pixel conversion (not a hardcoded
    // ratio). The grid is anchored per lens (below), not to the world, so the hole
    // keeps smooth sub-pixel movement while its edge reads as blocks.
    let cell = px::art_pixel_size(px::world_units_per_pixel(world));

    var ring_accum = vec3<f32>(0.0);
    var ring_strength = 0.0;
    var coverage = 0.0;
    var horizon = false;
    var horizon_color = vec3<f32>(0.0);
    // Center of the lens owning this fragment, so the dither cell is measured from
    // the hole's center — the quantized blocks ride with the moving hole. The
    // dither is deliberately NOT rotated: rotation belongs to the ring/shadow
    // geometry only, so the lensed background quantized here never spins.
    var active_center = vec2<f32>(0.0);
    var active_weight = -1.0;

    let count = min(material.count, MAX_LENSES);
    for (var i = 0u; i < count; i = i + 1u) {
        let lens = material.lenses[i];
        let center = lens.center_size_shadow.xy;
        let size = max(lens.center_size_shadow.z, 1e-4);
        let rs = lens.center_size_shadow.w;

        let angle = lens.strength_ring.w; // hole z-rotation in radians
        let centered = world - center;
        let r_raw = length(centered) / size; // halo-normalized: rs at horizon, 1 at rim.

        // Per-fragment cull: this lens's halo does not cover the fragment.
        if r_raw > 1.0 {
            continue;
        }

        // Snapped radius: pixelate the offset in the hole's own rotated frame.
        // Rotating the centered offset by -angle aligns the block grid to (and
        // spins it with) the hole's rotation; the grid rides with the hole and
        // stays smooth under motion. A whole art-pixel cell shares one radius,
        // stepping the horizon disc and photon ring in pixel blocks, not smooth
        // circles.
        let r = length(px::pixelate_world(px::rotate2d(centered, -angle), cell)) / size;

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

        // Horizon dominates; otherwise the strongest ring owns the fragment's center.
        let weight = select(ring_i, 1e9, r < rs);
        if weight > active_weight {
            active_weight = weight;
            active_center = center;
        }

        // Soft edge fade toward the outer rim uses the un-snapped radius so the
        // halo alpha stays smooth; only the ring + shadow edge are pixel-snapped.
        coverage = max(coverage, clamp((1.0 - r_raw) * 4.0, 0.0, 1.0));
    }

    // Dither phase per art-pixel cell, in the owning lens's centered (unrotated)
    // frame so the quantized blocks stay locked to the hole as it moves. Rotation
    // is confined to the ring/shadow geometry above, never the lensed background.
    let dither_pos = px::pixel_cell_index(world - active_center, cell);

    if horizon {
        return maybe_quantize(vec4<f32>(horizon_color, 1.0), dither_pos);
    }

    // Sample the velocity (deflection) field produced by the GPU fluid simulation.
    // The field is stored in the same canvas-UV space as the background texture.
    // When the texture is not yet ready (no lenses or first frame) it defaults
    // to zero, so the background samples straight through.
    let field_uv = canvas_uv(world);
    let total_deflect = textureSample(velocity_field_tex, velocity_field_sampler, field_uv).rg;

    let lensed_world = world - total_deflect;
    let sample_uv = canvas_uv(lensed_world);
    let in_bounds = step(0.0, sample_uv.x) * step(sample_uv.x, 1.0)
                  * step(0.0, sample_uv.y) * step(sample_uv.y, 1.0);
    let clamped_uv = clamp(sample_uv, vec2<f32>(0.0), vec2<f32>(1.0));
    let bg = textureSample(background_tex, background_sampler, clamped_uv).rgb * in_bounds;

    // Lensed background: opaque where the field samples the scene, transparent
    // where it deflects off-canvas, with a soft outer-rim edge.
    let scene_alpha = in_bounds * coverage;
    let bg_clamped = clamp(bg, vec3<f32>(0.0), vec3<f32>(1.0));

    // Photon sphere: a hard cutoff replaces the soft glow blend. Below the
    // cutoff the ring contributes nothing and the lensed background passes
    // through untouched — at full fidelity, never palette-quantized. There is no
    // partial transparency band in between. `alpha_cutoff` is the ring-strength
    // threshold.
    let ring_mask = clamp(ring_strength, 0.0, 1.0);
    if ring_mask <= 0.0 || ring_mask < quantization.alpha_cutoff {
        return vec4<f32>(bg_clamped, scene_alpha);
    }

    // Above the cutoff the band is a solid, opaque ring color. Divide out the
    // Gaussian falloff so it reads as one flat photon-ring color rather than a
    // glow, then palette-quantize that color. Only the photon sphere is
    // quantized — the lensed background above is returned untouched.
    let ring_color = ring_accum / max(ring_strength, 1e-5);
    let quantized = maybe_quantize(vec4<f32>(ring_color, 1.0), dither_pos);
    return vec4<f32>(quantized.rgb, coverage);
}
