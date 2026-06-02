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
    asset::RenderAssetUsages,
    prelude::*,
    render::render_resource::{AsBindGroup, Extent3d, ShaderType, TextureDimension, TextureFormat},
    shader::ShaderRef,
    sprite_render::{AlphaMode2d, Material2d, Material2dPlugin},
};

use crate::quantize::{DitherPattern, MAX_PALETTE_COLORS, linear_rgb_to_oklab};

/// Default per-axis resolution for the palette lookup table.
///
/// A multiple of 16 keeps each texel row 256-byte aligned (`16 B/texel`), so the
/// 3D upload needs no row padding. 64³ holds the whole linear-RGB cube finely
/// enough that the nearest-palette result is visually indistinguishable from the
/// per-pixel Oklab search for palettes up to [`MAX_PALETTE_COLORS`].
pub const PALETTE_LUT_RESOLUTION: u32 = 64;

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

impl ColorQuantizeUniforms {
    /// Nearest palette color (linear RGB) to `color`, matched in Oklab with no
    /// dithering. Returns `color` unchanged when no palette is configured.
    ///
    /// Mirrors `find_nearest_palette_color` in `color_quantize_functions.wgsl`
    /// with a zero dither offset, so a solid fill can be pre-snapped on the CPU
    /// once instead of paying the per-pixel palette match in a fragment shader.
    #[must_use]
    pub fn nearest_palette_color(&self, color: Vec3) -> Vec3 {
        if self.palette_size == 0 {
            return color;
        }
        let (l, a, b) = linear_rgb_to_oklab(color.x, color.y, color.z);
        let target = Vec3::new(l, a, b);
        let mut best_distance = f32::MAX;
        let mut best = color;
        for i in 0..self.palette_size as usize {
            let diff = target - self.palette_oklab[i].truncate();
            let distance = diff.dot(diff);
            if distance < best_distance {
                best_distance = distance;
                best = self.palette[i].truncate();
            }
        }
        best
    }

    /// Stable hash of the active palette, used to skip rebuilding the LUT when
    /// the palette is unchanged. Only the matched colors and their count feed
    /// the lookup table, so dithering / alpha settings are deliberately excluded.
    #[must_use]
    pub fn palette_key(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.palette_size.hash(&mut hasher);
        for color in self.palette.iter().take(self.palette_size as usize) {
            color.x.to_bits().hash(&mut hasher);
            color.y.to_bits().hash(&mut hasher);
            color.z.to_bits().hash(&mut hasher);
        }
        hasher.finish()
    }

    /// Bakes the nearest-palette match over the linear-RGB cube into a 3D image.
    ///
    /// Each texel stores the palette color closest (in Oklab) to the linear RGB
    /// at its cell center, so a shader can replace the per-pixel palette loop
    /// with a single nearest-sampled fetch. The texture is `Rgba32Float` and
    /// must be sampled with a nearest (non-filtering) sampler — interpolation
    /// would blend two palette entries into an off-palette color.
    ///
    /// Inputs are clamped to `[0, 1]` by the sampler's clamp-to-edge addressing;
    /// HDR colors above the palette range resolve to the brightest palette entry,
    /// matching the per-pixel search whose palette is itself in `[0, 1]`.
    #[must_use]
    pub fn build_lut(&self, resolution: u32) -> Image {
        let n = resolution.max(1);
        let inv = 1.0 / n as f32;
        let mut data = Vec::with_capacity((n * n * n) as usize * 16);
        for z in 0..n {
            let b = (z as f32 + 0.5) * inv;
            for y in 0..n {
                let g = (y as f32 + 0.5) * inv;
                for x in 0..n {
                    let r = (x as f32 + 0.5) * inv;
                    let c = self.nearest_palette_color(Vec3::new(r, g, b));
                    data.extend_from_slice(&c.x.to_le_bytes());
                    data.extend_from_slice(&c.y.to_le_bytes());
                    data.extend_from_slice(&c.z.to_le_bytes());
                    data.extend_from_slice(&1.0f32.to_le_bytes());
                }
            }
        }
        Image::new(
            Extent3d {
                width: n,
                height: n,
                depth_or_array_layers: n,
            },
            TextureDimension::D3,
            data,
            TextureFormat::Rgba32Float,
            RenderAssetUsages::RENDER_WORLD,
        )
    }
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
