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
    // xy = viewport origin / target size, zw = viewport size / target size.
    // (0, 0, 1, 1) when no viewport is set (full-target render).
    viewport_rect: vec4<f32>,
    // xy = mirror guard-band width per axis in full-target UV, z = mirror enable
    // (1 on / 0 off), w unused.
    edge_mirror: vec4<f32>,
    // Linear-RGBA fallback color blended beyond the mirrored band; a = strength.
    edge_fallback_color: vec4<f32>,
    quantization: QuantizationSettings,
    count: u32,
    lenses: array<LensData, 64>,
};

@group(0) @binding(0) var scene_tex: texture_2d<f32>;
@group(0) @binding(1) var scene_sampler: sampler;
@group(0) @binding(2) var velocity_field_tex: texture_2d<f32>;
@group(0) @binding(3) var velocity_field_sampler: sampler;
@group(0) @binding(4) var<uniform> u: LensingDisplay;
// Palette lookup table: nearest palette color baked over the linear-RGB cube,
// point-loaded (no sampler) in place of the per-pixel Oklab palette loop.
// Unused when `quantization.palette_size == 0`.
@group(0) @binding(5) var palette_lut: texture_3d<f32>;

// Map a world position to the field's canvas UV. The canvas is a square,
// world-axis-aligned region centered on `canvas_center_extent.xy`; V is flipped
// because world +y is up while texture V points down.
fn canvas_uv(world: vec2<f32>) -> vec2<f32> {
    let center = u.canvas_center_extent.xy;
    let extent = max(u.canvas_center_extent.zw, vec2<f32>(1e-5, 1e-5));
    let uv = (world - center) / extent + vec2<f32>(0.5);
    return vec2<f32>(uv.x, 1.0 - uv.y);
}

// Reconstruct the world-space position of a screen pixel from its full-target UV.
//
// The display pass renders to the full target texture without a GPU viewport,
// so `in.uv` spans the entire target. The camera's clip_from_world is sized to
// the sub-window viewport (e.g. the dev inspector game area). Correct by
// converting the full-target UV to a viewport-relative UV first, then to NDC.
fn world_from_uv(uv: vec2<f32>) -> vec2<f32> {
    let vp_uv = (uv - u.viewport_rect.xy) / max(u.viewport_rect.zw, vec2<f32>(1e-5));
    let ndc = vec2<f32>(vp_uv.x * 2.0 - 1.0, 1.0 - vp_uv.y * 2.0);
    let world = u.world_from_clip * vec4<f32>(ndc, 0.0, 1.0);
    return world.xy / world.w;
}

// Project a world-space position to full-target UV.
//
// Produces a UV suitable for sampling the full-target scene texture, accounting
// for the viewport sub-region that clip_from_world maps to.
fn uv_from_world(world: vec2<f32>) -> vec2<f32> {
    let clip = u.clip_from_world * vec4<f32>(world, 0.0, 1.0);
    let ndc = clip.xy / clip.w;
    let vp_uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    return u.viewport_rect.xy + vp_uv * u.viewport_rect.zw;
}

// Reflect a coordinate that falls outside `[lo, hi]` back inside, across the
// nearest edge (a single fold). Coordinates already inside pass through. Used to
// fill the guard band: the region just off-screen mirrors the interior instead
// of repeating the clamped border texel.
fn reflect_coord(s: f32, lo: f32, hi: f32) -> f32 {
    if s < lo {
        return lo + (lo - s);
    }
    if s > hi {
        return hi - (s - hi);
    }
    return s;
}

// Read the lit scene at a projected sample UV, resolving samples that land
// outside the rendered viewport. The valid region is `viewport_rect`; a sample
// past it would otherwise clamp to the border texel and smear into a band. It is
// instead mirrored back across the crossed edge(s), then — for samples too far
// out for mirroring to stay plausible — blended toward the configured fallback
// color. `textureSampleLevel` keeps the read in per-fragment control flow
// (implicit-derivative sampling demands uniformity).
fn scene_at(sample_uv: vec2<f32>) -> vec3<f32> {
    let lo = u.viewport_rect.xy;
    let hi = u.viewport_rect.xy + u.viewport_rect.zw;
    // Per-axis distance the sample lands outside the rendered region (0 inside).
    let over = max(max(lo - sample_uv, sample_uv - hi), vec2<f32>(0.0));

    // Inside the rendered region: straight read, no edge handling.
    if over.x <= 0.0 && over.y <= 0.0 {
        return textureSampleLevel(scene_tex, scene_sampler, sample_uv, 0.0).rgb;
    }

    // Mirror the off-screen sample back across the edge(s) it crossed. When
    // mirroring is disabled (`edge_mirror.z == 0`) the sample is left as-is and
    // the clamp-to-edge sampler carries it.
    let mirrored = vec2<f32>(
        reflect_coord(sample_uv.x, lo.x, hi.x),
        reflect_coord(sample_uv.y, lo.y, hi.y),
    );
    let src_uv = mix(sample_uv, mirrored, u.edge_mirror.z);
    let color = textureSampleLevel(scene_tex, scene_sampler, src_uv, 0.0).rgb;

    // Beyond the mirrored guard band, ramp toward the fallback color. `t` is 1.0
    // at the band's outer edge and grows with overshoot; the ramp spans one more
    // band width. With zero fallback alpha this is a no-op and mirroring stands.
    let margin = max(u.edge_mirror.xy, vec2<f32>(1e-6));
    let t = max(over.x / margin.x, over.y / margin.y);
    let blend = clamp(t - 1.0, 0.0, 1.0) * u.edge_fallback_color.a;
    return mix(color, u.edge_fallback_color.rgb, blend);
}

// Lit scene at a world position: deflect by the field, project the deflected
// world position back to screen, sample the view target.
fn lensed_scene(world: vec2<f32>) -> vec3<f32> {
    let deflect = textureSample(velocity_field_tex, velocity_field_sampler, canvas_uv(world)).rg;
    let sample_uv = uv_from_world(world - deflect);
    return scene_at(sample_uv);
}

// Palette-quantize a color; pass-through when no palette is configured. The
// nearest-palette match comes from the baked LUT (`palette_lut`) instead of a
// per-pixel loop, so the photon-ring band no longer pays a `palette_size`-long
// Oklab search per fragment when the hole grows to near-full coverage.
fn maybe_quantize(color: vec4<f32>, dither_pos: vec2<f32>) -> vec4<f32> {
    if u.quantization.palette_size == 0u {
        return color;
    }
    return cq::quantize_color_lut(
        color,
        dither_pos,
        palette_lut,
        u.quantization.palette_size,
        u.quantization.alpha_cutoff,
        u.quantization.transparency_floor,
        u.quantization.dither_pattern,
    );
}

@fragment
fn fragment(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let world = world_from_uv(in.uv);

    // Pixel size for the photon ring and shadow edge: the world span of one
    // displayed art pixel, from the live world-to-pixel conversion (never a
    // hardcoded ratio). The grid is anchored per lens (below), not to the world,
    // so the hole keeps smooth sub-pixel movement while its edge reads as blocks.
    let cell = px::art_pixel_size(px::world_units_per_pixel(world));

    var ring_accum = vec3<f32>(0.0);
    var ring_strength = 0.0;
    var horizon = false;
    var horizon_color = vec3<f32>(0.0);
    // Center of the lens owning this fragment, so the dither cell is measured from
    // the hole's center — the quantized blocks ride with the moving hole. The
    // dither is deliberately NOT rotated: rotation belongs to the ring/shadow
    // geometry only, so the lensed background quantized here never spins.
    var active_center = vec2<f32>(0.0);
    var active_weight = -1.0;

    let count = min(u.count, MAX_LENSES);
    for (var i = 0u; i < count; i = i + 1u) {
        let lens = u.lenses[i];
        let center = lens.center_size_shadow.xy;
        let size = max(lens.center_size_shadow.z, 1e-4);
        let rs = lens.center_size_shadow.w;

        let angle = lens.strength_ring.w; // hole z-rotation in radians
        let centered = world - center;
        let r_raw = length(centered) / size; // rs at horizon, 1 at outer rim.
        if r_raw > 1.0 {
            continue;
        }

        // Snap the centered offset to the art-pixel grid in the hole's own
        // rotated frame. Rotating by -angle aligns the block grid to (and spins
        // it with) the hole's rotation; the grid rides with the hole and stays
        // smooth under motion. A whole art-pixel cell shares one snapped offset,
        // so the horizon disc and ring band step in pixel blocks, not smooth
        // circles.
        let snapped = px::pixelate_world(px::rotate2d(centered, -angle), cell);
        let r = length(snapped) / size; // ring band, measured from the cell center

        // EVENT HORIZON: solid black wherever a cell's *center* lies inside the
        // shadow radius (any lens wins). Measuring to the snapped cell center —
        // the same distance the photon ring uses below — instead of the cell's
        // nearest edge drops the corner cells, so the disc reads as a pixel
        // circle at every size. The nearest-edge test instead fills any cell the
        // circle merely grazes, which at a few-pixel radius squares the shadow
        // off into a block (corner cells sit only ~half a cell farther than edge
        // cells). The floor at the cell's half-diagonal keeps the central block
        // lit so the shadow steps down to a 2×2 and never blinks out at the
        // bottom of a size pulse rather than rounding away one pixel at a time.
        let shadow_world = max(rs * size, cell * 0.708);
        let in_horizon = length(snapped) < shadow_world;
        if in_horizon {
            horizon = true;
            horizon_color = lens.black_color.rgb;
        }

        // PHOTON RING: Gaussian glow just outside rs via mul-chain.
        let ring_t = clamp(1.0 - (r - rs) / (lens.strength_ring.y * 6.0), 0.0, 1.0);
        let ring_t2 = ring_t * ring_t;
        let ring_i = ring_t2 * ring_t2 * lens.strength_ring.z;
        ring_accum += lens.photon_ring_color.rgb * ring_i;
        ring_strength = max(ring_strength, ring_i);

        // Horizon dominates; otherwise the strongest ring owns the fragment's center.
        let weight = select(ring_i, 1e9, in_horizon);
        if weight > active_weight {
            active_weight = weight;
            active_center = center;
        }
    }

    // Interior of the event horizon is a solid color, pre-snapped to the ring
    // palette on the CPU (`LensData.black_color`). Returning it directly is the
    // early-out for the full-coverage cinematic: every covered fragment writes a
    // constant color and skips both the deflected scene read and the per-pixel
    // palette match. The pixel-snapped edge is carried by the snapped radius `r`,
    // so the shadow boundary still reads as blocks.
    if horizon {
        return vec4<f32>(horizon_color, 1.0);
    }

    // Dither phase per art-pixel cell, in the owning lens's centered (unrotated)
    // frame so the quantized blocks stay locked to the hole as it moves. Rotation
    // is confined to the ring/shadow geometry above, never the lensed background.
    let dither_pos = px::pixel_cell_index(world - active_center, cell);

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
