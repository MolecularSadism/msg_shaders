// lensing_field_advect.wgsl
//
// Semi-Lagrangian advection pass.  For each grid cell, traces the
// backward-characteristic pos_prev = pos - dt * vel(pos), samples
// velocity_in at pos_prev with bilinear clamping, and writes the result to
// velocity_out.
//
// Group 0: shared uniform.
// Group 1 binding 0: velocity_in  (Rg16Float, read-only storage)
// Group 1 binding 1: velocity_out (Rg16Float, write-only storage)

struct LensingFieldUniform {
    lens_center_size_shadow: array<vec4<f32>, 64>,
    lens_strength_ring:      array<vec4<f32>, 64>,
    canvas_center_extent:    vec4<f32>,
    lens_count:  u32,
    decay:       f32,
    force_scale: f32,
    dt:          f32,
};

@group(0) @binding(0) var<uniform> u: LensingFieldUniform;
@group(1) @binding(0) var velocity_in:  texture_storage_2d<rg16float, read>;
@group(1) @binding(1) var velocity_out: texture_storage_2d<rg16float, write>;

const RES: u32 = 512u;
const FRES: f32 = 512.0;

// Bilinear sample of velocity_in at a floating-point grid coordinate,
// clamped to the grid boundaries (free-slip boundary condition).
fn sample_bilinear(pos: vec2<f32>) -> vec2<f32> {
    let p = pos - vec2<f32>(0.5);
    let i = vec2<i32>(floor(p));
    let f = fract(p);

    let i00 = clamp(i,              vec2<i32>(0), vec2<i32>(i32(RES) - 1));
    let i10 = clamp(i + vec2<i32>(1, 0), vec2<i32>(0), vec2<i32>(i32(RES) - 1));
    let i01 = clamp(i + vec2<i32>(0, 1), vec2<i32>(0), vec2<i32>(i32(RES) - 1));
    let i11 = clamp(i + vec2<i32>(1, 1), vec2<i32>(0), vec2<i32>(i32(RES) - 1));

    let v00 = textureLoad(velocity_in, i00).rg;
    let v10 = textureLoad(velocity_in, i10).rg;
    let v01 = textureLoad(velocity_in, i01).rg;
    let v11 = textureLoad(velocity_in, i11).rg;

    let lerp_x0 = mix(v00, v10, f.x);
    let lerp_x1 = mix(v01, v11, f.x);
    return mix(lerp_x0, lerp_x1, f.y);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let coord = gid.xy;
    if coord.x >= RES || coord.y >= RES {
        return;
    }

    let pos = vec2<f32>(coord) + vec2<f32>(0.5);

    // Current velocity at this cell.
    let vel_here = textureLoad(velocity_in, vec2<i32>(coord)).rg;

    // Backward-trace: where did this fluid parcel come from?
    // Grid Y increases downward while world Y increases upward (V=0 = world
    // top), so the Y component of the grid displacement is negated relative to
    // world-space velocity.
    let extent = u.canvas_center_extent.zw;
    let world_to_grid = FRES / max(extent, vec2<f32>(1e-5, 1e-5));
    let vel_grid = vel_here * vec2<f32>(world_to_grid.x, -world_to_grid.y);
    let pos_prev = pos - u.dt * vel_grid;

    let advected = sample_bilinear(pos_prev);

    textureStore(velocity_out, vec2<i32>(coord), vec4<f32>(advected, 0.0, 0.0));
}
