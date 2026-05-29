// lensing_field_divergence.wgsl
//
// Divergence pass.  Computes the central-difference divergence of velocity_in
// and writes a scalar result to divergence_out.
//
//   div(v)[i,j] = 0.5 * (v_x[i+1,j] - v_x[i-1,j] + v_y[i,j+1] - v_y[i,j-1])
//
// Boundary cells clamp to the nearest interior cell (free-slip).
//
// Group 0: shared uniform.
// Group 1 binding 0: velocity_in   (Rg16Float, read-only storage)
// Group 1 binding 1: divergence_out (R16Float, write-only storage)

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
@group(1) @binding(0) var velocity_in:    texture_storage_2d<rg16float, read>;
@group(1) @binding(1) var divergence_out: texture_storage_2d<r16float,  write>;

const RES: u32 = 512u;

fn load_vel(coord: vec2<i32>) -> vec2<f32> {
    let c = clamp(coord, vec2<i32>(0), vec2<i32>(i32(RES) - 1));
    return textureLoad(velocity_in, c).rg;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let coord = vec2<i32>(gid.xy);
    if u32(coord.x) >= RES || u32(coord.y) >= RES {
        return;
    }

    let vr = load_vel(coord + vec2<i32>(1, 0)).x;
    let vl = load_vel(coord - vec2<i32>(1, 0)).x;
    let vt = load_vel(coord + vec2<i32>(0, 1)).y;
    let vb = load_vel(coord - vec2<i32>(0, 1)).y;

    let div = 0.5 * ((vr - vl) + (vt - vb));

    textureStore(divergence_out, coord, vec4<f32>(div, 0.0, 0.0, 0.0));
}
