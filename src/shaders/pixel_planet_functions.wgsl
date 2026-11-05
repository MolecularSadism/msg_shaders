#define_import_path msg_shaders::pixel_planet_functions

// Shared noise / shaping helpers used by every pixel-planet layer.
//
// Ported from Deep-Fold/PixelPlanets (MIT), except `circle_noise`, which is an
// original implementation. The `tile` argument reproduces the per-shader
// wrapping factor (`mod(coord, tile * round(size))`) that makes the planet
// surface seamless across the sphere mapping.

fn gmod1(a: f32, b: f32) -> f32 {
    return a - b * floor(a / b);
}

fn gmod2(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return a - b * floor(a / b);
}

fn rand_v(coord: vec2<f32>, size: f32, seed: f32, tile: vec2<f32>) -> f32 {
    let c = gmod2(coord, tile * round(size));
    return fract(sin(dot(c, vec2<f32>(12.9898, 78.233))) * 15.5453 * seed);
}

fn noise_v(coord: vec2<f32>, size: f32, seed: f32, tile: vec2<f32>) -> f32 {
    let i = floor(coord);
    let f = fract(coord);

    let a = rand_v(i, size, seed, tile);
    let b = rand_v(i + vec2<f32>(1.0, 0.0), size, seed, tile);
    let c = rand_v(i + vec2<f32>(0.0, 1.0), size, seed, tile);
    let d = rand_v(i + vec2<f32>(1.0, 1.0), size, seed, tile);

    let cubic = f * f * (3.0 - 2.0 * f);

    return mix(a, b, cubic.x) + (c - a) * cubic.y * (1.0 - cubic.x) + (d - b) * cubic.x * cubic.y;
}

fn fbm_v(coord_in: vec2<f32>, octaves: i32, size: f32, seed: f32, tile: vec2<f32>) -> f32 {
    var value = 0.0;
    var scale = 0.5;
    var coord = coord_in;
    for (var i = 0; i < octaves; i = i + 1) {
        value += noise_v(coord, size, seed, tile) * scale;
        coord *= 2.0;
        scale *= 0.5;
    }
    return value;
}

// Bubble noise: every cell of a brick-staggered grid hosts one dark disc with
// a randomized centre and radius, kept fully inside its cell. Returns 1.0 in
// open space, falling to 0.0 toward each disc centre; callers sum shifted
// samples of it to build turbulence.
fn circle_noise(uv_in: vec2<f32>, size: f32, seed: f32, tile: vec2<f32>) -> f32 {
    var uv = uv_in;
    let row = floor(uv.y);
    // Shift odd rows by half a cell so the discs never line up in columns.
    uv.x += gmod1(row, 2.0) * 0.5;
    let cell = vec2<f32>(floor(uv.x), row);
    let local = fract(uv);

    // Independent draws for the disc's placement and its radius. The integer
    // offsets decorrelate the draws while staying compatible with the
    // `tile * round(size)` wrap inside `rand_v`.
    let jitter_x = rand_v(cell, size, seed, tile);
    let jitter_y = rand_v(cell + vec2<f32>(41.0, 289.0), size, seed, tile);
    let radius_draw = rand_v(cell + vec2<f32>(97.0, 113.0), size, seed, tile);

    // Squaring biases toward small discs; clamping the centre into
    // [radius, 1 - radius] keeps each disc inside its own cell. The full
    // centre-to-rim gradient keeps the summed field smooth enough that the
    // fbm domain warp built on it reads as streaks rather than rings.
    let radius = mix(0.08, 0.32, radius_draw * radius_draw);
    let centre = vec2<f32>(radius) + vec2<f32>(jitter_x, jitter_y) * (1.0 - 2.0 * radius);
    let dist = length(local - centre);
    return smoothstep(0.0, radius, dist);
}

fn spherify(uv: vec2<f32>) -> vec2<f32> {
    let centered = uv * 2.0 - 1.0;
    let z = sqrt(max(1.0 - dot(centered, centered), 0.0));
    let sphere = centered / (z + 1.0);
    return sphere * 0.5 + 0.5;
}

fn rotate_uv(coord_in: vec2<f32>, angle: f32) -> vec2<f32> {
    let coord = coord_in - 0.5;
    let c = cos(angle);
    let s = sin(angle);
    let r = vec2<f32>(coord.x * c - coord.y * s, coord.x * s + coord.y * c);
    return r + 0.5;
}

fn dither_bool(uv_pixel: vec2<f32>, uv_real: vec2<f32>, pixels: f32) -> bool {
    return gmod1(uv_pixel.x + uv_real.y, 2.0 / pixels) <= 1.0 / pixels;
}

// 4x4 ordered (Bayer) dither threshold in [0, 1) for an integer pixel-cell
// coordinate. Lets a soft coverage value resolve to a stable stipple pattern
// rather than a hard cutoff.
fn bayer4x4(cell: vec2<f32>) -> f32 {
    var matrix = array<f32, 16>(
        0.0,  8.0,  2.0,  10.0,
        12.0, 4.0,  14.0, 6.0,
        3.0,  11.0, 1.0,  9.0,
        15.0, 7.0,  13.0, 5.0,
    );
    let x = i32(gmod1(cell.x, 4.0));
    let y = i32(gmod1(cell.y, 4.0));
    return matrix[y * 4 + x] / 16.0;
}

// Resolves a soft `coverage` (1 fully inside the shape, 0 fully outside) into a
// hard 0/1 alpha through the ordered pattern, turning a smooth edge into a
// pixel-art dithered rim. Coverage that is already 0 or 1 passes through
// unchanged, so a zero-width band reproduces a plain step edge.
fn dither_coverage(coverage: f32, cell: vec2<f32>) -> f32 {
    return 1.0 - step(coverage, bayer4x4(cell));
}
