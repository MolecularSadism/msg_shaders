//! Color quantization utilities for retro-style palette effects.
//!
//! Provides an importable shader module and utilities for applying
//! color quantization to materials. The quantization uses perceptually-accurate
//! Oklab color space matching with optional ordered dithering.
//!
//! # Usage
//!
//! For custom materials that want to integrate quantization:
//!
//! ```wgsl
//! #import msg_shaders::color_quantize_functions as cq
//!
//! // In fragment shader:
//! var palette = quantization.palette;
//! let quantized = cq::quantize_color(color, screen_pos, &palette, ...);
//! ```
//!
//! For standalone use with sprites/meshes, use [`crate::ColorQuantizeMaterial`].

use bevy::{
    asset::{embedded_asset, load_internal_asset, uuid_handle},
    prelude::*,
    shader::Shader,
};

/// Handle for the color quantize functions shader.
/// This shader is loaded as an internal asset so it can be imported by other shaders.
pub const COLOR_QUANTIZE_FUNCTIONS_SHADER_HANDLE: Handle<Shader> =
    uuid_handle!("8a7b6c5d-4e3f-2a1b-0c9d-8e7f6a5b4c3d");

/// Maximum number of colors in a quantization palette.
/// Limited by shader uniform buffer size.
pub const MAX_PALETTE_COLORS: usize = 64;

/// Dithering pattern for color quantization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Reflect)]
pub enum DitherPattern {
    /// No dithering - pure nearest color matching
    None,
    /// Classic 4x4 Bayer matrix (16 threshold levels)
    #[default]
    Bayer4x4,
    /// Smoother 8x8 Bayer matrix (64 threshold levels)
    Bayer8x8,
}

impl DitherPattern {
    /// Convert to shader uniform value.
    pub fn as_u32(self) -> u32 {
        match self {
            DitherPattern::None => 0,
            DitherPattern::Bayer4x4 => 1,
            DitherPattern::Bayer8x8 => 2,
        }
    }
}

/// Convert a linear RGB color to Oklab color space.
///
/// Mirrors `linear_rgb_to_oklab` in `color_quantize_functions.wgsl` so the
/// CPU-side palette and the shader's per-pixel conversions agree. Palettes are
/// pre-converted on the CPU so the per-pixel cost is a dot-product loop rather
/// than `palette_size` cbrt calls.
#[allow(clippy::excessive_precision)]
pub fn linear_rgb_to_oklab(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b;
    let m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b;
    let s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b;

    let l_ = l.cbrt();
    let m_ = m.cbrt();
    let s_ = s.cbrt();

    (
        0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
        1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
        0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_,
    )
}

/// Plugin that enables color quantization shader imports and materials.
///
/// # Example
///
/// ```
/// use bevy::prelude::*;
/// use msg_shaders::ColorQuantizationPlugin;
///
/// // Register the color quantization plugin with your app
/// fn register_plugin(app: &mut App) {
///     app.add_plugins(ColorQuantizationPlugin);
/// }
/// ```
pub struct ColorQuantizationPlugin;

impl Plugin for ColorQuantizationPlugin {
    fn build(&self, app: &mut App) {
        // Load the functions shader as an internal asset so it can be imported by other shaders.
        // This uses load_internal_asset! which registers the shader with the import system.
        load_internal_asset!(
            app,
            COLOR_QUANTIZE_FUNCTIONS_SHADER_HANDLE,
            "shaders/color_quantize_functions.wgsl",
            Shader::from_wgsl
        );

        // Embed the material shader (not imported by others, just used directly)
        embedded_asset!(app, "shaders/color_quantize_material.wgsl");

        // Register types
        app.register_type::<DitherPattern>();

        // Add standalone material plugin
        app.add_plugins(crate::quantize_material::plugin);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dither_pattern_default_is_bayer4x4() {
        assert_eq!(DitherPattern::default(), DitherPattern::Bayer4x4);
    }

    #[test]
    fn dither_pattern_as_u32() {
        assert_eq!(DitherPattern::None.as_u32(), 0);
        assert_eq!(DitherPattern::Bayer4x4.as_u32(), 1);
        assert_eq!(DitherPattern::Bayer8x8.as_u32(), 2);
    }

    #[test]
    fn oklab_white_is_lightness_one() {
        let (l, _, _) = linear_rgb_to_oklab(1.0, 1.0, 1.0);
        assert!((l - 1.0).abs() < 1e-3);
    }
}
