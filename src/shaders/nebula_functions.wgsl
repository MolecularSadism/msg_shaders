// Procedural nebula + starfield utility functions for import into materials.
// Generates fractal-noise nebula clouds and a hashed, twinkling star field.
// Pure WGSL (no #import), so it validates offline with naga.
//
// Usage in importing shader:
//   #import msg_shaders::nebula_functions as nb
//   let color = nb::nebula_color(p, tint_a, tint_b, background, density);
//   color += nb::star_layer(cell, spacing, time, seed, speed, cool, warm, density);

#define_import_path msg_shaders::nebula_functions

const NB_TAU: f32 = 6.28318530718;

// Hash a 2D point to a scalar in [0, 1). Dave Hoskins' hash without sine.
fn hash21(p: vec2<f32>) -> f32 {
    var p3 = fract(vec3<f32>(p.x, p.y, p.x) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

// Hash a 2D point to a 2D value, each component in [0, 1).
fn hash22(p: vec2<f32>) -> vec2<f32> {
    var p3 = fract(vec3<f32>(p.x, p.y, p.x) * vec3<f32>(0.1031, 0.1030, 0.0973));
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.xx + p3.yz) * p3.zy);
}

// Smoothed 2D value noise in [0, 1].
fn value_noise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let a = hash21(i);
    let b = hash21(i + vec2<f32>(1.0, 0.0));
    let c = hash21(i + vec2<f32>(0.0, 1.0));
    let d = hash21(i + vec2<f32>(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

// Fractal Brownian motion: summed octaves of value noise, each rotated to break
// axis alignment. Normalized back into [0, 1].
fn fbm(p: vec2<f32>, octaves: u32) -> f32 {
    let rot = mat2x2<f32>(0.8, 0.6, -0.6, 0.8);
    var value = 0.0;
    var total = 0.0;
    var amplitude = 0.5;
    var q = p;
    for (var i = 0u; i < octaves; i = i + 1u) {
        value = value + amplitude * value_noise(q);
        total = total + amplitude;
        q = rot * q * 2.0;
        amplitude = amplitude * 0.5;
    }
    return value / max(total, 1e-5);
}

// Nebula emission color (linear RGB, HDR-capable) at pattern position `p`.
// Domain-warps the cloud field into wispy filaments, tints between two palette
// colors by a low-frequency hue field, and adds brighter cores at dense knots.
fn nebula_color(
    p: vec2<f32>,
    color_a: vec3<f32>,
    color_b: vec3<f32>,
    background: vec3<f32>,
    density: f32,
) -> vec3<f32> {
    let warp = vec2<f32>(
        fbm(p + vec2<f32>(1.7, 9.2), 4u),
        fbm(p + vec2<f32>(8.3, 2.8), 4u),
    );
    let q = p + (warp - 0.5) * 3.0;

    var clouds = fbm(q, 5u);
    clouds = smoothstep(0.30, 0.85, clouds) * density;

    let hue = clamp(fbm(p * 0.5 + vec2<f32>(4.0, 4.0), 3u), 0.0, 1.0);
    let tint = mix(color_a, color_b, hue);

    let core = pow(clouds, 2.0) * 0.6;
    return background + tint * clouds + tint * core;
}

// Additive emission from one star layer for the art-pixel at integer coordinate
// `cell`. Each `spacing`x`spacing` block holds at most one star, lit on a single
// art-pixel, so stars stay crisp and pixel-perfect. Stars twinkle (a sharp
// brightness pulse that occasionally dips to full black) and shift color between
// `color_cool` and `color_warm` over time.
fn star_layer(
    cell: vec2<f32>,
    spacing: f32,
    time: f32,
    seed: f32,
    twinkle_speed: f32,
    color_cool: vec3<f32>,
    color_warm: vec3<f32>,
    density: f32,
) -> vec3<f32> {
    let block = floor(cell / spacing);
    let rnd = hash22(block + vec2<f32>(seed, seed * 1.7 + 3.1));
    let rnd2 = hash22(block + vec2<f32>(seed * 2.3 + 5.0, seed * 0.7 + 1.0));

    // Only a fraction of blocks contain a star.
    let present = step(1.0 - density, rnd2.x);

    // Place the star on one art-pixel inside the block, then test whether this
    // fragment's cell is exactly that pixel.
    let offset = floor(rnd * max(spacing - 1.0, 1.0));
    let star_cell = block * spacing + offset;
    let on_pixel = step(abs(cell.x - star_cell.x), 0.5) * step(abs(cell.y - star_cell.y), 0.5);

    // Twinkle: per-star phase and rate, sharpened for a sparkle, gated to black
    // near the trough so some stars wink fully out.
    let phase = rnd.x * NB_TAU;
    let rate = twinkle_speed * (0.5 + rnd.y);
    let osc = 0.5 + 0.5 * sin(time * rate + phase);
    let spark = pow(osc, 3.0);
    let alive = step(0.12, osc);
    let base = 0.55 + 0.45 * rnd2.y;
    let intensity = base * spark * alive;

    // Dynamic color: slow cool<->warm shift, independent of the brightness pulse.
    let temp = 0.5 + 0.5 * sin(time * rate * 0.37 + phase * 1.7);
    let color = mix(color_cool, color_warm, temp);

    return color * intensity * present * on_pixel;
}
