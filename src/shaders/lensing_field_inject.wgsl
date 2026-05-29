// lensing_field_inject.wgsl
//
// Inject pass: for each grid cell, sum Schwarzschild force contributions from
// all active lenses and write the result into velocity_out, first multiplying
// the existing velocity in velocity_in by `decay`.  This replaces the previous
// frame's velocity with a smoothly-decayed + force-injected field.
//
// Group 0: shared uniform (settings + lens array).
// Group 1 binding 0: velocity_in  (Rg16Float, read-only storage)
// Group 1 binding 1: velocity_out (Rg16Float, write-only storage)

struct LensingFieldUniform {
    lens_center_size_shadow: array<vec4<f32>, 64>,
    lens_strength_ring:      array<vec4<f32>, 64>,
    canvas_center_extent:    vec4<f32>,    // xy=center, zw=extent
    lens_count:  u32,
    decay:       f32,
    force_scale: f32,
    dt:          f32,
};

@group(0) @binding(0) var<uniform> u: LensingFieldUniform;
@group(1) @binding(0) var velocity_in:  texture_storage_2d<rg16float, read>;
@group(1) @binding(1) var velocity_out: texture_storage_2d<rg16float, write>;

// Grid resolution (must match LENSING_FIELD_RES on the Rust side).
const RES: u32 = 512u;

// Map a grid coordinate to world space.
fn grid_to_world(coord: vec2<u32>) -> vec2<f32> {
    let uv = (vec2<f32>(coord) + vec2<f32>(0.5)) / f32(RES);
    // canvas_center_extent.xy = world-space center; .zw = extent (side lengths).
    let center = u.canvas_center_extent.xy;
    let extent = u.canvas_center_extent.zw;
    // uv (0→1) maps to center ± extent/2.
    return center + (uv - vec2<f32>(0.5)) * extent;
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

    if u.lens_count == 0u {
        textureStore(velocity_out, vec2<i32>(coord), vec4<f32>(vel, 0.0, 0.0));
        return;
    }

    let world = grid_to_world(coord);

    // Accumulate Schwarzschild deflection from every active lens.
    var total_force = vec2<f32>(0.0);
    for (var i = 0u; i < u.lens_count; i = i + 1u) {
        let css  = u.lens_center_size_shadow[i];
        let sr_v = u.lens_strength_ring[i];

        let center   = css.xy;
        let size     = max(css.z, 1e-4);
        let rs       = css.w;
        let strength = sr_v.x;

        let delta = world - center;
        let dist  = length(delta);
        let r     = dist / size;

        // Only inject within the halo; skip the horizon interior to avoid
        // degeneracy at r ~ rs.
        if r >= 1.0 || r < rs {
            continue;
        }

        let dr      = max(r - rs, rs * 0.1);
        let mag     = strength * rs * rs / dr;
        let dir     = delta / max(dist, 1e-5);
        total_force = total_force + dir * mag * size;
    }

    vel = vel + total_force * u.force_scale;

    textureStore(velocity_out, vec2<i32>(coord), vec4<f32>(vel, 0.0, 0.0));
}
