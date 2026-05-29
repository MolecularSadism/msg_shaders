// lensing_field_gradient_subtract.wgsl
//
// Gradient-subtract pass.  Subtracts the central-difference gradient of the
// solved pressure field from the velocity, making the field approximately
// divergence-free.
//
//   v_out[i,j] = v_in[i,j] - 0.5 * (p[i+1,j] - p[i-1,j], p[i,j+1] - p[i,j-1])
//
// Boundary cells clamp (free-slip).
//
// Group 0: shared uniform.
// Group 1 binding 0: velocity_in  (Rg16Float, read-only storage)
// Group 1 binding 1: pressure_in  (R16Float,  read-only storage) — final solved pressure
// Group 1 binding 2: velocity_out (Rg16Float, write-only storage)

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
@group(1) @binding(1) var pressure_in:  texture_storage_2d<r16float,  read>;
@group(1) @binding(2) var velocity_out: texture_storage_2d<rg16float, write>;

const RES: u32 = 512u;

fn load_p(coord: vec2<i32>) -> f32 {
    let c = clamp(coord, vec2<i32>(0), vec2<i32>(i32(RES) - 1));
    return textureLoad(pressure_in, c).r;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let coord = vec2<i32>(gid.xy);
    if u32(coord.x) >= RES || u32(coord.y) >= RES {
        return;
    }

    let vel = textureLoad(velocity_in, coord).rg;

    let pr = load_p(coord + vec2<i32>(1, 0));
    let pl = load_p(coord - vec2<i32>(1, 0));
    let pt = load_p(coord + vec2<i32>(0, 1));
    let pb = load_p(coord - vec2<i32>(0, 1));

    let grad = vec2<f32>(pr - pl, pt - pb) * 0.5;
    let corrected = vel - grad;

    textureStore(velocity_out, coord, vec4<f32>(corrected, 0.0, 0.0));
}
