// Procedural nebula + starfield utility functions for import into materials.
// Generates fractal-noise nebula clouds and a hashed, twinkling star field.
// Pure WGSL (no #import), so it validates offline with naga.
//
// Each layer resolves to one of its own authored colors, so a stack of layers
// composites by decision rather than by blending — no layer's colors can shift
// where another layer's edges fall.
//
// Usage in importing shader:
//   #import msg_shaders::nebula_functions as nb
//   let d = nb::nebula_density(p, octaves, warp_octaves, warp, low, high);
//   let stop = nb::ramp3_level(d * 3.0, dither, c1, c2, c3);
//   let star = nb::star_layer(cell, spacing, time, seed, speed, density,
//                             twinkle_chance);

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

// Cloud density in [0, 1] for one nebula layer at pattern position `p`.
// Domain-warps the field into wispy filaments, then a smoothstep band sets the
// cloud contrast. Callers map this density through a color ramp and composite.
//
// `octaves` is the cloud fbm detail; `warp_octaves` the domain-warp detail;
// `warp_strength` how far the warp displaces the lookup; `cloud_low`/`cloud_high`
// the smoothstep band.
fn nebula_density(
    p: vec2<f32>,
    octaves: u32,
    warp_octaves: u32,
    warp_strength: f32,
    cloud_low: f32,
    cloud_high: f32,
) -> f32 {
    let warp = vec2<f32>(
        fbm(p + vec2<f32>(1.7, 9.2), warp_octaves),
        fbm(p + vec2<f32>(8.3, 2.8), warp_octaves),
    );
    let q = p + (warp - 0.5) * warp_strength;
    let clouds = fbm(q, octaves);
    return smoothstep(cloud_low, cloud_high, clouds);
}

// One layer's four-level resolve at `level`, the layer's density mapped onto
// [0, 3]. Returns `(color, painted)`: `.a` is 1 when the layer claims this
// art-pixel and `.rgb` is the stop it claims it with, 0 when the pixel belongs
// to whatever is behind. Level 1 is the faint edge stop `c1`, 2 the body `c2`,
// 3 the bright core `c3`.
//
// Background and the three stops sit one unit apart on one scale, so a single
// ordered-dither decision answers both "does this layer paint here" and "with
// which stop". `dither` is the art-pixel's ordered-dither offset in
// [-0.5, 0.5), so it can only reach the two levels bracketing `level`: the
// image is solid at every whole level and dithers across the gaps between
// them. A region therefore fills completely with one stop before the next
// starts appearing on top of it, and a region past level 1 never shows the
// background through.
//
// Interpolating between the stops mints colors belonging to no authored
// swatch, which a later palette match then has to guess its way back out of —
// that guess is what lets a layer's chosen colors decide where its visible
// edges land. Resolving to a whole level keeps a layer's output inside the
// three colors it was authored with, and leaves its edges a function of
// density alone.
fn ramp3_level(level: f32, dither: f32, c1: vec3<f32>, c2: vec3<f32>, c3: vec3<f32>) -> vec4<f32> {
    let stop = clamp(round(level + dither), 0.0, 3.0);
    if stop < 0.5 {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
    if stop < 1.5 {
        return vec4<f32>(c1, 1.0);
    }
    if stop < 2.5 {
        return vec4<f32>(c2, 1.0);
    }
    return vec4<f32>(c3, 1.0);
}

// One star layer's contribution for the art-pixel at integer coordinate `cell`,
// as `(pick, lit)`: `.y` is 1 when this pixel is a lit star and 0 otherwise,
// and `.x` is that star's draw in [0, 1), fixed for its lifetime, which the
// caller scales by its own color count to choose a swatch. Each
// `spacing`x`spacing` block holds at most one star, lit on a single art-pixel,
// so stars stay crisp and pixel-perfect.
//
// Returning the draw rather than a color keeps the layer's palette in the
// caller's uniform, where it can be any length, and keeps a star's color a
// single choice: the caller writes that swatch as-is instead of adding it to
// what is behind, so a lit pixel is exactly one authored color, with no
// brightness ramp and nothing swept between swatches over time.
//
// Twinkling is likewise a decision, not a fade: `twinkle_chance` of the stars
// blink between lit and dark at their own phase and rate, and the rest hold
// steady.
fn star_layer(
    cell: vec2<f32>,
    spacing: f32,
    time: f32,
    seed: f32,
    twinkle_speed: f32,
    density: f32,
    twinkle_chance: f32,
) -> vec2<f32> {
    let block = floor(cell / spacing);
    let rnd = hash22(block + vec2<f32>(seed, seed * 1.7 + 3.1));
    let rnd2 = hash22(block + vec2<f32>(seed * 2.3 + 5.0, seed * 0.7 + 1.0));
    // A third stream keeps the color draw and blink rate independent of
    // placement, so editing a layer's palette recolors its stars without
    // moving any of them.
    let rnd3 = hash22(block + vec2<f32>(seed * 3.9 + 7.3, seed * 1.3 + 2.4));

    // Only a fraction of blocks contain a star.
    let present = step(1.0 - density, rnd2.x);

    // Place the star on one art-pixel inside the block, then test whether this
    // fragment's cell is exactly that pixel.
    let offset = floor(rnd * max(spacing - 1.0, 1.0));
    let star_cell = block * spacing + offset;
    let on_pixel = step(abs(cell.x - star_cell.x), 0.5) * step(abs(cell.y - star_cell.y), 0.5);

    // Twinkle: the chosen fraction blinks on its own phase and rate; every
    // other star stays lit. The blink is a square wave, so a twinkling star is
    // only ever its own color or the sky behind it.
    let twinkles = step(1.0 - twinkle_chance, rnd2.y);
    let phase = rnd.x * NB_TAU;
    let rate = twinkle_speed * (0.5 + rnd3.y);
    let blink = step(0.0, sin(time * rate + phase));
    let lit = mix(1.0, blink, twinkles);

    return vec2<f32>(rnd3.x, present * on_pixel * lit);
}
