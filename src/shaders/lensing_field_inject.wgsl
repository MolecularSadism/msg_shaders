// lensing_field_inject.wgsl
//
// Inject pass: for each grid cell, sum the force contributed by every active
// deflection source and write the result into velocity_out, first multiplying
// the existing velocity in velocity_in by `decay`. This replaces the previous
// frame's velocity with a decayed + force-injected field.
//
// Each source carries a shape tag (tag_strength.x); the per-cell loop dispatches
// on it. Case 0 (Lens) is the black-hole Schwarzschild force, identical to the
// pre-migration loop. Cases 1/2 (Ring/Line) are edge-tapered and uniform push
// shapes respectively.
// Adding a deflection type is one more `fn` + one more `case`.
//
// Group 0: shared uniform (settings + source array).
// Group 1 binding 0: velocity_in  (Rg16Float, read-only storage)
// Group 1 binding 1: velocity_out (Rg16Float, write-only storage)

const TAU: f32 = 6.283185307179586;

struct DeflectionSource {
    // x = shape tag, y = strength, zw reserved.
    tag_strength: vec4<f32>,
    // Geometry row 0 (shape-dependent).
    geom_a: vec4<f32>,
    // Geometry row 1 (shape-dependent).
    geom_b: vec4<f32>,
};

struct LensingFieldUniform {
    sources: array<DeflectionSource, 64>,
    canvas_center_extent: vec4<f32>,    // xy=center, zw=extent
    source_count: u32,
    decay:        f32,
    force_scale:  f32,
    dt:           f32,
    lens_falloff: f32,
};

@group(0) @binding(0) var<uniform> u: LensingFieldUniform;
@group(1) @binding(0) var velocity_in:  texture_storage_2d<rg16float, read>;
@group(1) @binding(1) var velocity_out: texture_storage_2d<rg16float, write>;

// Grid resolution (must match LENSING_FIELD_RES on the Rust side).
const RES: u32 = 512u;

// Map a grid coordinate to world space. Matches the canvas_uv convention used
// in the fragment shader: V=0 is world top, V=1 is world bottom, so Y is
// negated relative to raw uv.y.
fn grid_to_world(coord: vec2<u32>) -> vec2<f32> {
    let uv = (vec2<f32>(coord) + vec2<f32>(0.5)) / f32(RES);
    let center = u.canvas_center_extent.xy;
    let extent = u.canvas_center_extent.zw;
    return center + vec2<f32>(uv.x - 0.5, 0.5 - uv.y) * extent;
}

// Radial outward lens deflection, falling monotonically from the inner radius
// to the rim. geom_a = (center.xy, size, unused), geom_b.x = inner_radius as a
// fraction of `size`; strength = tag_strength.y is the peak deflection as a
// fraction of `size`.
//
// Two radii bound it: the deflection peaks at `inner_radius` — normally the
// event horizon of whatever the lens sits behind — and reaches zero at `size`.
//
//   profile(r) = ((inner/r)^k - (inner/1)^k) / (1 - (inner/1)^k)
//
// `k` is `u.lens_falloff`, one global shape knob (`k = 1` is the physical
// inverse-distance law, `k < 1` stretches the bend further out). The subtract
// and divide normalise it: the profile is 1 at `inner_radius` and 0 at the rim
// for *every* k, so retuning k reshapes the curve without rescaling any lens
// authored against it.
//
// Nothing is computed inside `inner_radius`. That region sits behind an opaque
// event-horizon disc wherever a lens has a visual, so it can only ever be
// painted over — and for a hole grown to cover the screen it is nearly the whole
// grid, which the early-out skips outright.
fn lens_force(world: vec2<f32>, g_a: vec4<f32>, g_b: vec4<f32>, strength: f32) -> vec2<f32> {
    let center = g_a.xy;
    let size   = max(g_a.z, 1e-4);
    let inner  = clamp(g_b.x, 1e-4, 0.99);
    let delta  = world - center;
    let dist   = length(delta);
    let r      = dist / size;
    if r >= 1.0 || r < inner {
        return vec2<f32>(0.0);
    }
    let k       = max(u.lens_falloff, 1e-3);
    let at_rim  = pow(inner, k);
    let profile = (pow(inner / r, k) - at_rim) / max(1.0 - at_rim, 1e-5);
    return (delta / max(dist, 1e-5)) * strength * size * profile;
}

// Share of a ring's extent each of its edges eases across, as a fraction of
// the band's `thickness` radially and of its `arc` angularly. The push holds
// full strength across the middle and reaches zero at the edge with a matching
// slope, so it meets the undeflected field without a step — a step there
// advects into a hard-edged disc the eye reads as a seam against the rest of
// the scene.
const RING_EDGE_FADE: f32 = 0.45;

// Outward push inside an annular sector, easing to zero across the outer
// `RING_EDGE_FADE` of the band. geom_a = (center.xy, inner, thickness);
// geom_b = (start_angle, arc). arc = TAU is a closed ring.
//
// A band with an inner radius eases at that edge too; a disc (`inner == 0`)
// has no inner edge to meet the field across, so it carries full strength out
// of its centre.
fn ring_force(world: vec2<f32>, g_a: vec4<f32>, g_b: vec4<f32>, strength: f32) -> vec2<f32> {
    let center = g_a.xy;
    let inner  = g_a.z;
    let thick  = g_a.w;
    let arc    = g_b.y;
    let delta  = world - center;
    let dist   = length(delta);
    if dist < inner || dist > inner + thick {
        return vec2<f32>(0.0);
    }
    let ang = atan2(delta.y, delta.x);
    // CCW angle from the sector start, wrapped to [0, TAU). The + 1.0 keeps the
    // argument positive before fract so a sector spanning the 0-angle seam works.
    let from_start = fract((ang - g_b.x) / TAU + 1.0) * TAU;
    if from_start > arc {
        return vec2<f32>(0.0);
    }

    // Radial taper: 0 at the inner edge of the band, 1 at its rim.
    let across = (dist - inner) / max(thick, 1e-5);
    var fade = smoothstep(0.0, RING_EDGE_FADE, 1.0 - across);
    if inner > 0.0 {
        fade = fade * smoothstep(0.0, RING_EDGE_FADE, across);
    }

    // Angular taper across the two ends of a sector. A closed ring has no ends.
    if arc < TAU {
        let ends = max(RING_EDGE_FADE * arc, 1e-5);
        fade = fade
            * smoothstep(0.0, ends, from_start)
            * smoothstep(0.0, ends, arc - from_start);
    }

    return (delta / max(dist, 1e-5)) * strength * fade;
}

// Uniform directional push inside a band. geom_a = (center.xy, half_length,
// thickness); geom_b.x = rotation (push direction). The band extends
// perpendicular to the push for +-half_length, with depth `thickness` along the
// push axis, centered on the line.
fn line_force(world: vec2<f32>, g_a: vec4<f32>, g_b: vec4<f32>, strength: f32) -> vec2<f32> {
    let center   = g_a.xy;
    let half_len = g_a.z;
    let thick    = g_a.w;
    let rot      = g_b.x;
    let dir  = vec2<f32>(cos(rot), sin(rot));
    let perp = vec2<f32>(-dir.y, dir.x);
    let delta  = world - center;
    let along  = dot(delta, dir);
    let across = dot(delta, perp);
    if abs(across) > half_len || abs(along) > thick * 0.5 {
        return vec2<f32>(0.0);
    }
    return dir * strength;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let coord = gid.xy;
    if coord.x >= RES || coord.y >= RES {
        return;
    }

    // Decay the existing velocity.
    let prev = textureLoad(velocity_in, vec2<i32>(coord)).rg;
    var vel = prev * u.decay;

    if u.source_count == 0u {
        textureStore(velocity_out, vec2<i32>(coord), vec4<f32>(vel, 0.0, 0.0));
        return;
    }

    let world = grid_to_world(coord);

    // Accumulate the force from every active source, dispatching on shape.
    var total_force = vec2<f32>(0.0);
    for (var i = 0u; i < u.source_count; i = i + 1u) {
        let s = u.sources[i];
        let tag = u32(s.tag_strength.x);
        let strength = s.tag_strength.y;
        switch tag {
            case 0u: {
                total_force += lens_force(world, s.geom_a, s.geom_b, strength);
            }
            case 1u: {
                total_force += ring_force(world, s.geom_a, s.geom_b, strength);
            }
            case 2u: {
                total_force += line_force(world, s.geom_a, s.geom_b, strength);
            }
            default: {}
        }
    }

    vel = vel + total_force * u.force_scale;

    textureStore(velocity_out, vec2<i32>(coord), vec4<f32>(vel, 0.0, 0.0));
}
