// Color quantization utility functions for import into materials.
// Uses Oklab color space for perceptually accurate nearest-color matching
// with optional Bayer dithering for retro aesthetics.
//
// Usage in importing shader:
//   #import msg_shaders::color_quantize_functions as cq
//   let quantized = cq::quantize_color(color, screen_pos, settings);

#define_import_path msg_shaders::color_quantize_functions

const MAX_PALETTE_SIZE: u32 = 64u;

/// Settings for color quantization.
/// Must be provided as a uniform in the importing shader.
struct QuantizationSettings {
    palette: array<vec4<f32>, 64>,
    palette_size: u32,
    alpha_cutoff: f32,
    dither_pattern: u32,
    transparency_floor: f32,
}

// Bayer 4x4 dithering matrix (normalized to 0-1 range)
const BAYER_4X4: array<f32, 16> = array<f32, 16>(
     0.0 / 16.0,  8.0 / 16.0,  2.0 / 16.0, 10.0 / 16.0,
    12.0 / 16.0,  4.0 / 16.0, 14.0 / 16.0,  6.0 / 16.0,
     3.0 / 16.0, 11.0 / 16.0,  1.0 / 16.0,  9.0 / 16.0,
    15.0 / 16.0,  7.0 / 16.0, 13.0 / 16.0,  5.0 / 16.0
);

// Bayer 8x8 dithering matrix (normalized to 0-1 range)
const BAYER_8X8: array<f32, 64> = array<f32, 64>(
     0.0 / 64.0, 32.0 / 64.0,  8.0 / 64.0, 40.0 / 64.0,  2.0 / 64.0, 34.0 / 64.0, 10.0 / 64.0, 42.0 / 64.0,
    48.0 / 64.0, 16.0 / 64.0, 56.0 / 64.0, 24.0 / 64.0, 50.0 / 64.0, 18.0 / 64.0, 58.0 / 64.0, 26.0 / 64.0,
    12.0 / 64.0, 44.0 / 64.0,  4.0 / 64.0, 36.0 / 64.0, 14.0 / 64.0, 46.0 / 64.0,  6.0 / 64.0, 38.0 / 64.0,
    60.0 / 64.0, 28.0 / 64.0, 52.0 / 64.0, 20.0 / 64.0, 62.0 / 64.0, 30.0 / 64.0, 54.0 / 64.0, 22.0 / 64.0,
     3.0 / 64.0, 35.0 / 64.0, 11.0 / 64.0, 43.0 / 64.0,  1.0 / 64.0, 33.0 / 64.0,  9.0 / 64.0, 41.0 / 64.0,
    51.0 / 64.0, 19.0 / 64.0, 59.0 / 64.0, 27.0 / 64.0, 49.0 / 64.0, 17.0 / 64.0, 57.0 / 64.0, 25.0 / 64.0,
    15.0 / 64.0, 47.0 / 64.0,  7.0 / 64.0, 39.0 / 64.0, 13.0 / 64.0, 45.0 / 64.0,  5.0 / 64.0, 37.0 / 64.0,
    63.0 / 64.0, 31.0 / 64.0, 55.0 / 64.0, 23.0 / 64.0, 61.0 / 64.0, 29.0 / 64.0, 53.0 / 64.0, 21.0 / 64.0
);

/// Convert linear RGB to Oklab color space.
/// Based on https://bottosson.github.io/posts/oklab/
fn linear_rgb_to_oklab(c: vec3<f32>) -> vec3<f32> {
    let l = 0.4122214708 * c.r + 0.5363325363 * c.g + 0.0514459929 * c.b;
    let m = 0.2119034982 * c.r + 0.6806995451 * c.g + 0.1073969566 * c.b;
    let s = 0.0883024619 * c.r + 0.2817188376 * c.g + 0.6299787005 * c.b;

    let l_ = pow(l, 1.0 / 3.0);
    let m_ = pow(m, 1.0 / 3.0);
    let s_ = pow(s, 1.0 / 3.0);

    return vec3<f32>(
        0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
        1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
        0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_
    );
}

/// Convert Oklab to linear RGB color space.
fn oklab_to_linear_rgb(c: vec3<f32>) -> vec3<f32> {
    let l_ = c.x + 0.3963377774 * c.y + 0.2158037573 * c.z;
    let m_ = c.x - 0.1055613458 * c.y - 0.0638541728 * c.z;
    let s_ = c.x - 0.0894841775 * c.y - 1.2914855480 * c.z;

    let l = l_ * l_ * l_;
    let m = m_ * m_ * m_;
    let s = s_ * s_ * s_;

    return vec3<f32>(
        4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
        -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
        -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s
    );
}

/// Non-negative remainder of `v / n` (floored division), computed entirely in
/// float space so it is correct for any `v` — negative, huge, or tiny —
/// without ever casting an out-of-range value to `u32`. `u32(v) % n` (the
/// previous approach) relied on callers pre-biasing `v` positive by a fixed
/// constant; that bias could be exceeded whenever the caller's grid cell size
/// shrank relative to its coordinate's extent, silently wrapping the `u32`
/// cast into a garbage index and reading a garbage matrix entry.
fn floor_mod_u32(v: f32, n: f32) -> u32 {
    // Clamp before the cast: floored division keeps this in [0, n) modulo
    // float rounding, but a `u32` cast of even a hair below zero is still an
    // out-of-range conversion, so never rely on that residual staying positive.
    return u32(max(v - floor(v / n) * n, 0.0));
}

/// Get dither threshold offset for color dithering (centered around 0).
fn get_dither_threshold(pos: vec2<f32>, pattern: u32) -> f32 {
    if pattern == 0u {
        return 0.0;
    } else if pattern == 1u {
        let x = floor_mod_u32(pos.x, 4.0);
        let y = floor_mod_u32(pos.y, 4.0);
        let idx = y * 4u + x;
        return BAYER_4X4[idx] - 0.5;
    } else {
        let x = floor_mod_u32(pos.x, 8.0);
        let y = floor_mod_u32(pos.y, 8.0);
        let idx = y * 8u + x;
        return BAYER_8X8[idx] - 0.5;
    }
}

/// Get raw dither threshold (0.0 to 1.0) for transparency dithering.
fn get_dither_threshold_raw(pos: vec2<f32>, pattern: u32) -> f32 {
    if pattern == 0u {
        return 0.5;
    } else if pattern == 1u {
        let x = floor_mod_u32(pos.x, 4.0);
        let y = floor_mod_u32(pos.y, 4.0);
        let idx = y * 4u + x;
        return BAYER_4X4[idx];
    } else {
        let x = floor_mod_u32(pos.x, 8.0);
        let y = floor_mod_u32(pos.y, 8.0);
        let idx = y * 8u + x;
        return BAYER_8X8[idx];
    }
}

/// Find the nearest color in the palette using Oklab distance.
///
/// `palette_oklab` must contain the Oklab representation of each palette entry,
/// precomputed on the CPU, so the per-pixel cost is a dot product loop rather
/// than `palette_size` cbrt/pow calls.
fn find_nearest_palette_color(
    color: vec3<f32>,
    dither_offset: f32,
    palette: ptr<function, array<vec4<f32>, 64>>,
    palette_oklab: ptr<function, array<vec4<f32>, 64>>,
    palette_size: u32
) -> vec3<f32> {
    let oklab = linear_rgb_to_oklab(color);

    // Apply the dither offset to all three Oklab channels, not just
    // lightness — a palette boundary that differs mainly in hue/chroma (a/b)
    // would otherwise get no dither at all and render as a hard edge.
    let dithered_oklab = oklab + vec3<f32>(dither_offset * 0.1);

    var best_distance = 1000000.0;
    var best_color = vec3<f32>(0.0, 0.0, 0.0);

    for (var i = 0u; i < palette_size; i++) {
        let diff = dithered_oklab - (*palette_oklab)[i].rgb;
        let distance = dot(diff, diff);

        if distance < best_distance {
            best_distance = distance;
            best_color = (*palette)[i].rgb;
        }
    }

    return best_color;
}

/// Nearest palette color from a precomputed lookup table.
///
/// The LUT bakes the Oklab nearest-palette search over the linear-RGB cube
/// `[0, 1]^3` (see `ColorQuantizeUniforms::build_lut`). A direct `textureLoad`
/// of the cell containing the color replaces the per-pixel `palette_size` loop
/// with one fetch — no sampler, so the unfilterable `Rgba32Float` LUT needs no
/// filtering-sampler binding and the read is always the exact stored entry.
///
/// Dithering is preserved exactly: the offset is applied to all three Oklab
/// channels and converted back to linear RGB before the lookup, so only the
/// nearest search is approximated (to the LUT's cell resolution), not the
/// dither.
fn find_nearest_palette_color_lut(
    color: vec3<f32>,
    dither_offset: f32,
    lut: texture_3d<f32>,
) -> vec3<f32> {
    let oklab = linear_rgb_to_oklab(color);
    let dithered_oklab = oklab + vec3<f32>(dither_offset * 0.1);
    let dithered_rgb = oklab_to_linear_rgb(dithered_oklab);

    // Integer index of the cell containing the (clamped) color. `floor(c * dims)`
    // maps [0,1] onto [0, dims], clamped to the last cell so c == 1 stays in range.
    let dims = vec3<f32>(textureDimensions(lut));
    let cf = clamp(dithered_rgb, vec3<f32>(0.0), vec3<f32>(1.0));
    let coord = vec3<i32>(clamp(floor(cf * dims), vec3<f32>(0.0), dims - vec3<f32>(1.0)));
    return textureLoad(lut, coord, 0).rgb;
}

/// LUT-backed equivalent of `quantize_color`.
///
/// Identical transparency / dither / alpha handling; only the per-pixel palette
/// loop is replaced by a `find_nearest_palette_color_lut` fetch. Pass the same
/// `palette_size`, `alpha_cutoff`, `transparency_floor`, and `dither_pattern`
/// the loop version would have used, plus the baked LUT and its nearest sampler.
fn quantize_color_lut(
    color: vec4<f32>,
    screen_pos: vec2<f32>,
    lut: texture_3d<f32>,
    palette_size: u32,
    alpha_cutoff: f32,
    transparency_floor: f32,
    dither_pattern: u32
) -> vec4<f32> {
    let dither_raw = get_dither_threshold_raw(screen_pos, dither_pattern);
    let dither_color = dither_raw - 0.5;

    let luminance = dot(color.rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
    let luminance_weight = 1.0 - color.a;
    let effective_alpha = mix(color.a, color.a * max(luminance, 0.05), luminance_weight);

    if effective_alpha < alpha_cutoff {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    let alpha_range = 1.0 - alpha_cutoff;
    let normalized_alpha = (effective_alpha - alpha_cutoff) / alpha_range;

    if normalized_alpha < transparency_floor {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    let floor_range = 1.0 - transparency_floor;
    let adjusted_alpha = (normalized_alpha - transparency_floor) / floor_range;

    if adjusted_alpha < dither_raw {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    let quantized = find_nearest_palette_color_lut(color.rgb, dither_color, lut);

    return vec4<f32>(quantized, 1.0);
}

/// Quantize a color with full transparency dithering support.
/// Returns vec4(0) for transparent pixels, vec4(quantized_rgb, 1.0) for opaque.
///
/// Arguments:
/// - color: Input color in linear RGB with alpha
/// - screen_pos: Pixel position for dithering pattern
/// - palette: Color palette array in linear RGB (pass by pointer)
/// - palette_oklab: Same palette pre-converted to Oklab on the CPU (pass by pointer)
/// - palette_size: Number of colors in palette
/// - alpha_cutoff: Alpha threshold below which pixels are transparent (0.0-1.0)
/// - transparency_floor: Minimum normalized alpha for dithering (0.0-1.0)
/// - dither_pattern: 0=none, 1=bayer4x4, 2=bayer8x8
fn quantize_color(
    color: vec4<f32>,
    screen_pos: vec2<f32>,
    palette: ptr<function, array<vec4<f32>, 64>>,
    palette_oklab: ptr<function, array<vec4<f32>, 64>>,
    palette_size: u32,
    alpha_cutoff: f32,
    transparency_floor: f32,
    dither_pattern: u32
) -> vec4<f32> {
    let dither_raw = get_dither_threshold_raw(screen_pos, dither_pattern);
    let dither_color = dither_raw - 0.5;

    // Luminance weighting for dark semi-transparent pixels
    let luminance = dot(color.rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
    let luminance_weight = 1.0 - color.a;
    let effective_alpha = mix(color.a, color.a * max(luminance, 0.05), luminance_weight);

    // Hard floor: below alpha_cutoff is always transparent
    if effective_alpha < alpha_cutoff {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    // Normalize alpha from [alpha_cutoff, 1.0] to [0.0, 1.0]
    let alpha_range = 1.0 - alpha_cutoff;
    let normalized_alpha = (effective_alpha - alpha_cutoff) / alpha_range;

    // Transparency floor: prevent scattered dots from very low alpha
    if normalized_alpha < transparency_floor {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    // Remap remaining alpha for dithering
    let floor_range = 1.0 - transparency_floor;
    let adjusted_alpha = (normalized_alpha - transparency_floor) / floor_range;

    // Dither-based transparency decision
    if adjusted_alpha < dither_raw {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    // Find nearest palette color
    let quantized = find_nearest_palette_color(color.rgb, dither_color, palette, palette_oklab, palette_size);

    return vec4<f32>(quantized, 1.0);
}

/// Simplified quantization for colors that are known to be opaque.
/// Skips transparency handling for better performance.
fn quantize_color_opaque(
    color: vec3<f32>,
    screen_pos: vec2<f32>,
    palette: ptr<function, array<vec4<f32>, 64>>,
    palette_oklab: ptr<function, array<vec4<f32>, 64>>,
    palette_size: u32,
    dither_pattern: u32
) -> vec3<f32> {
    let dither_offset = get_dither_threshold(screen_pos, dither_pattern);
    return find_nearest_palette_color(color, dither_offset, palette, palette_oklab, palette_size);
}
