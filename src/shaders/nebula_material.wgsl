// Layered nebula + starfield background material.
//
// Composites a stack of procedural nebula layers (each its own noise settings,
// three-stop color ramp, and parallax) over a base color, then a stack of
// twinkling star layers. Everything is snapped to the game's art-pixel grid
// (pixel-perfect from screen-space derivatives) with ordered dithering. The
// pattern is authored in a bounded coordinate local to the quad, so it stays
// precise far from the world origin.
//
// Each layer's math finishes on its own: density picks a level on a scale that
// runs from the background through the layer's own three stops, one ordered
// dither resolves it, and the result is *written*, not blended. So
// every fragment carries exactly one authored swatch — a layer's background,
// one of a cloud layer's three stops, or one of the two star colors — and no
// layer's palette can move another layer's edges. Blending layers and matching
// the mixture to a shared palette afterwards would do exactly that: the
// nearest-swatch decision at every cloud edge would depend on which colors the
// other layers happened to be authored with.

#import bevy_sprite::mesh2d_vertex_output::VertexOutput
#import msg_shaders::pixelate_functions as px
#import msg_shaders::color_quantize_functions as cq
#import msg_shaders::nebula_functions as nb

// Keep in sync with MAX_NEBULA_LAYERS / MAX_STAR_LAYERS in nebula.rs.
const MAX_NEBULA_LAYERS: u32 = 6u;
const MAX_STAR_LAYERS: u32 = 4u;
const MAX_STAR_COLORS: u32 = 8u;

struct NebulaLayer {
    c1: vec4<f32>,
    c2: vec4<f32>,
    c3: vec4<f32>,
    offset: vec2<f32>,
    scale: f32,
    warp_strength: f32,
    cloud_low: f32,
    cloud_high: f32,
    intensity: f32,
    parallax: f32,
    octaves: u32,
    warp_octaves: u32,
    _pad0: f32,
    _pad1: f32,
};

struct StarLayer {
    // params: (spacing, density, first_color, parallax)
    params: vec4<f32>,
    // seed_pad: (seed, color_count, _, _)
    seed_pad: vec4<f32>,
};

struct NebulaSettings {
    layers: array<NebulaLayer, 6>,
    stars: array<StarLayer, 4>,
    background: vec4<f32>,
    // Every star layer's colors, concatenated; a layer reads the slice its
    // own params name.
    star_colors: array<vec4<f32>, 8>,
    world_size: vec2<f32>,
    scroll: vec2<f32>,
    num_layers: u32,
    num_stars: u32,
    dither_pattern: u32,
    twinkle_speed: f32,
    twinkle_chance: f32,
    _pad2: f32,
    rotation: f32,
    time: f32,
    pixel_size: f32,
    seed: f32,
    _pad0: f32,
    _pad1: f32,
};

@group(2) @binding(0)
var<uniform> nebula: NebulaSettings;

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    // Bounded pattern coordinate local to the quad, with the global rotation
    // (e.g. a day/night sky) applied once. Computed before `aps` below so the
    // pixel-size derivative reads this translation-independent magnitude
    // instead of the raw world position, which can be arbitrarily large (and
    // so lose derivative precision) far from the world origin.
    // `uv.y` runs top-down while world `y` runs bottom-up, so flip it: `local`
    // has to be a true world-space offset from the quad's center for the
    // parallax drift below to move a layer the same way the world moves.
    let local = (in.uv - vec2<f32>(0.5, 0.5)) * nebula.world_size * vec2<f32>(1.0, -1.0);
    let base = px::rotate2d(local, nebula.rotation);

    // World units per art pixel: pixel-perfect from screen derivatives (tracks
    // zoom), or an explicit override. The branch is on a uniform.
    var aps = nebula.pixel_size;
    if aps <= 0.0 {
        aps = px::art_pixel_size(px::world_units_per_pixel(base));
    }

    // Fragment centers sit at `(i + 0.5 - extent/2) * aps` from the quad
    // center, which is an exact cell boundary whenever the target's extent in
    // art pixels is odd — a `floor` tie that float error breaks per-fragment,
    // widening one art pixel and swallowing the next. A quarter-cell bias
    // clears the boundary for either parity, well under one art pixel.
    let grid = base + vec2<f32>(0.25 * aps);

    // One dither grid for the whole image (parallax-independent), so blocks
    // resolve consistently. Centered in [-0.5, 0.5): it shifts a layer's level
    // by at most half a stop, so a pixel only ever lands on one of the two
    // levels bracketing its density.
    let dither_cell = px::pixel_cell_index(grid, aps);
    let dither = cq::get_dither_threshold(dither_cell, nebula.dither_pattern);

    var color = nebula.background.rgb;

    // Nebula layers, back to front. Each has its own parallax drift, a
    // decorrelation offset, and a three-stop ramp keyed by density.
    for (var i = 0u; i < min(nebula.num_layers, MAX_NEBULA_LAYERS); i = i + 1u) {
        let layer = nebula.layers[i];
        // Drift quantized to whole art pixels so the pattern lands on the same
        // grid every frame instead of resampling onto a shifted one.
        let drift = round(nebula.scroll * layer.parallax / aps) * aps;
        let lp = px::pixelate_world(grid + drift, aps) + layer.offset;
        let d = nb::nebula_density(
            lp * layer.scale,
            layer.octaves,
            layer.warp_octaves,
            layer.warp_strength,
            layer.cloud_low,
            layer.cloud_high,
        );
        // The background and the layer's three stops are one 0..3 scale, so a
        // single dithered decision picks between them: the layer either writes
        // one of its own stops here or leaves what is behind untouched, and it
        // fills solid at each stop before the next dithers in over it.
        // `intensity` is the level the layer reaches where it is densest, so
        // below 1 it stops short of its core stop rather than stippling its
        // core down to the layer behind. Alpha-blending here is what would put
        // a between-layers color on screen for a palette match to resolve —
        // and resolve differently as soon as any layer's colors change.
        let level = clamp(d, 0.0, 1.0) * layer.intensity * 3.0;
        let painted = nb::ramp3_level(level, dither, layer.c1.rgb, layer.c2.rgb, layer.c3.rgb);
        if painted.a > 0.5 {
            color = painted.rgb;
        }
    }

    // Star layers on top, each drifting at its own parallax rate. A lit star
    // overwrites the cloud under it rather than adding to it, so its color
    // stays exactly the swatch it drew from its layer's palette.
    for (var j = 0u; j < min(nebula.num_stars, MAX_STAR_LAYERS); j = j + 1u) {
        let star = nebula.stars[j];
        // Star identity is keyed to the cell index, so the drift has to land on
        // whole cells; a fractional shift re-buckets the hash and the field
        // re-randomizes instead of moving.
        let sp = grid + round(nebula.scroll * star.params.w / aps) * aps;
        let cell = px::pixel_cell_index(sp, aps);
        let s = nb::star_layer(
            cell,
            star.params.x,
            nebula.time,
            nebula.seed + star.seed_pad.x,
            nebula.twinkle_speed,
            star.params.y,
            nebula.twinkle_chance,
        );
        // The layer's palette is the slice `[first, first + count)` of the
        // shared array; the star's own draw picks one swatch from it and holds
        // that swatch for as long as the layer does.
        let count = u32(star.seed_pad.y);
        if s.y > 0.5 && count > 0u {
            let pick = min(u32(s.x * f32(count)), count - 1u);
            color = nebula.star_colors[min(u32(star.params.z) + pick, MAX_STAR_COLORS - 1u)].rgb;
        }
    }

    // Already an authored swatch — nothing left to match to a palette.
    return vec4<f32>(color, 1.0);
}
