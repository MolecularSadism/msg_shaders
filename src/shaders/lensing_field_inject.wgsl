// lensing_field_inject.wgsl
//
// Inject pass: for each grid cell, sum the force contributed by every active
// deflection source and write the result into velocity_out, first multiplying
// the existing velocity in velocity_in by `decay`. This replaces the previous
// frame's velocity with a decayed + force-injected field.
//
// Each source carries a shape tag (tag_strength.x); the per-cell loop dispatches
// on it. Case 0 (Lens) is the black-hole Schwarzschild force, identical to the
// pre-migration loop. Cases 1/2 (Ring/Line) are uniform-push deflection shapes.
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

// Radial outward lens deflection, peaking across the core and fading to nothing
// at the rim. geom_a = (center.xy, size, core_radius); strength = tag_strength.y
// is the peak deflection as a fraction of `size`.
//
// The profile is normalised in `core_radius`: it peaks at exactly
// `strength * size` whatever the core is, so the core only shapes how the warp
// tapers, never how strong it is. A pinhead core therefore keeps the full reach
// and simply concentrates the bend nearer the centre — sizing the visual hole
// no longer scales the lensing away with it.
fn lens_force(world: vec2<f32>, g_a: vec4<f32>, strength: f32) -> vec2<f32> {
    let center = g_a.xy;
    let size   = max(g_a.z, 1e-4);
    let core   = clamp(g_a.w, 1e-3, 0.99);
    let delta  = world - center;
    let dist   = length(delta);
    let r      = dist / size;
    if r >= 1.0 {
        return vec2<f32>(0.0);
    }
    // Inside the core the magnitude ramps up from zero at the exact centre, so
    // the centre carries no direction singularity. Outside it falls off as 1/r,
    // reaching zero at the rim.
    let inner = clamp(r / core, 0.0, 1.0);
    let outer = (core / max(r, core) - core) / (1.0 - core);
    return (delta / max(dist, 1e-5)) * strength * size * inner * outer;
}

// Uniform outward push inside an annular sector. geom_a = (center.xy, inner,
// thickness); geom_b = (start_angle, arc). arc = TAU is a closed ring.
fn ring_force(world: vec2<f32>, g_a: vec4<f32>, g_b: vec4<f32>, strength: f32) -> vec2<f32> {
    let center = g_a.xy;
    let inner  = g_a.z;
    let thick  = g_a.w;
    let delta  = world - center;
    let dist   = length(delta);
    if dist < inner || dist > inner + thick {
        return vec2<f32>(0.0);
    }
    let ang = atan2(delta.y, delta.x);
    // CCW angle from the sector start, wrapped to [0, TAU). The + 1.0 keeps the
    // argument positive before fract so a sector spanning the 0-angle seam works.
    let from_start = fract((ang - g_b.x) / TAU + 1.0) * TAU;
    if from_start > g_b.y {
        return vec2<f32>(0.0);
    }
    return (delta / max(dist, 1e-5)) * strength;
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
                total_force += lens_force(world, s.geom_a, strength);
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
