//! Standalone color quantization material for sprites and meshes.
//!
//! Use this material when you want to apply color quantization to entities
//! that don't have their own custom material.
//!
//! # Example
//!
//! ```rust
//! use bevy::prelude::*;
//! use msg_shaders::{ColorQuantizeMaterial, QuantizationConfig};
//!
//! fn setup(
//!     mut commands: Commands,
//!     mut meshes: ResMut<Assets<Mesh>>,
//!     mut materials: ResMut<Assets<ColorQuantizeMaterial>>,
//! ) {
//!     let quad_mesh = meshes.add(Rectangle::new(100.0, 100.0));
//!     commands.spawn((
//!         Mesh2d(quad_mesh),
//!         MeshMaterial2d(materials.add(ColorQuantizeMaterial::new(
//!             Handle::default(),
//!             QuantizationConfig::default(),
//!         ))),
//!     ));
//! }
//! ```

use bevy::{
    prelude::*,
    render::render_resource::{AsBindGroup, ShaderType},
    shader::ShaderRef,
    sprite_render::{AlphaMode2d, Material2d, Material2dPlugin},
};

use crate::quantize::{DitherPattern, MAX_PALETTE_COLORS, linear_rgb_to_oklab};

/// Configuration for color quantization.
#[derive(Clone)]
pub struct QuantizationConfig {
    /// Colors in the palette (max 64, linear RGB)
    pub palette: Vec<Color>,
    /// Alpha cutoff threshold (0.0-1.0)
    pub alpha_cutoff: f32,
    /// Dithering pattern
    pub dither_pattern: DitherPattern,
    /// Transparency floor (0.0-1.0)
    pub transparency_floor: f32,
}

impl Default for QuantizationConfig {
    fn default() -> Self {
        Self {
            palette: vec![Color::BLACK, Color::WHITE],
            alpha_cutoff: 0.03,
            dither_pattern: DitherPattern::Bayer4x4,
            transparency_floor: 0.06,
        }
    }
}

impl QuantizationConfig {
    /// Create a config from a palette of colors, using default dithering settings.
    #[must_use]
    pub fn from_palette(palette: Vec<Color>) -> Self {
        Self {
            palette,
            ..Default::default()
        }
    }

    /// Convert to shader uniforms.
    pub fn to_uniforms(&self) -> ColorQuantizeUniforms {
        let mut palette = [Vec4::ZERO; MAX_PALETTE_COLORS];
        let mut palette_oklab = [Vec4::ZERO; MAX_PALETTE_COLORS];
        for (i, color) in self.palette.iter().take(MAX_PALETTE_COLORS).enumerate() {
            let rgba = color.to_linear();
            palette[i] = Vec4::new(rgba.red, rgba.green, rgba.blue, rgba.alpha);
            let (l, a, b) = linear_rgb_to_oklab(rgba.red, rgba.green, rgba.blue);
            palette_oklab[i] = Vec4::new(l, a, b, 0.0);
        }

        ColorQuantizeUniforms {
            palette,
            palette_oklab,
            palette_size: self.palette.len().min(MAX_PALETTE_COLORS) as u32,
            alpha_cutoff: self.alpha_cutoff,
            dither_pattern: self.dither_pattern.as_u32(),
            transparency_floor: self.transparency_floor,
        }
    }
}

/// Shader uniforms for color quantization.
///
/// Field order matches the WGSL `QuantizationSettings` struct layout exactly.
#[derive(Clone, Copy, ShaderType)]
pub struct ColorQuantizeUniforms {
    /// Color palette (max 64 colors, linear RGB)
    pub palette: [Vec4; MAX_PALETTE_COLORS],
    /// Palette pre-converted to Oklab, computed once on CPU to avoid per-pixel cbrt in the shader.
    pub palette_oklab: [Vec4; MAX_PALETTE_COLORS],
    /// Number of colors in palette
    pub palette_size: u32,
    /// Alpha cutoff (0.0-1.0)
    pub alpha_cutoff: f32,
    /// Dither pattern (0=none, 1=bayer4x4, 2=bayer8x8)
    pub dither_pattern: u32,
    /// Transparency floor (0.0-1.0)
    pub transparency_floor: f32,
}

impl Default for ColorQuantizeUniforms {
    fn default() -> Self {
        Self {
            palette: [Vec4::ZERO; MAX_PALETTE_COLORS],
            palette_oklab: [Vec4::ZERO; MAX_PALETTE_COLORS],
            palette_size: 0,
            alpha_cutoff: 0.03,
            dither_pattern: 1,
            transparency_floor: 0.06,
        }
    }
}

/// Material that applies color quantization to a texture.
///
/// This is a standalone material for entities that don't have
/// their own custom material but want color quantization effects.
#[derive(Asset, TypePath, AsBindGroup, Clone)]
pub struct ColorQuantizeMaterial {
    /// The source texture to quantize
    #[texture(0)]
    #[sampler(1)]
    pub texture: Handle<Image>,
    /// Quantization settings
    #[uniform(2)]
    pub quantization: ColorQuantizeUniforms,
}

impl ColorQuantizeMaterial {
    /// Create a new material with the given texture and quantization config.
    #[must_use]
    pub fn new(texture: Handle<Image>, config: QuantizationConfig) -> Self {
        Self {
            texture,
            quantization: config.to_uniforms(),
        }
    }

}

impl Material2d for ColorQuantizeMaterial {
    fn fragment_shader() -> ShaderRef {
        "embedded://msg_shaders/shaders/color_quantize_material.wgsl".into()
    }

    fn alpha_mode(&self) -> AlphaMode2d {
        AlphaMode2d::Blend
    }
}

pub(crate) fn plugin(app: &mut App) {
    app.add_plugins(Material2dPlugin::<ColorQuantizeMaterial>::default());
}
