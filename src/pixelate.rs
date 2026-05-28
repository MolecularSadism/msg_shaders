//! Pixelation effect: snaps sampling to a grid so content renders as discrete
//! blocks independent of the output resolution.
//!
//! Provides an importable shader module ([`msg_shaders::pixelate_functions`](crate))
//! plus two standalone materials:
//! - [`PixelateMaterial`] — pixelation only.
//! - [`QuantizePixelateMaterial`] — pixelation combined with color quantization
//!   in a single pass.
//!
//! # Usage
//!
//! For custom materials that want to integrate pixelation:
//!
//! ```wgsl
//! #import msg_shaders::pixelate_functions as px
//!
//! // In fragment shader:
//! let snapped = px::pixelate_uv(in.uv, vec2<f32>(grid, grid));
//! ```

use bevy::{
    asset::{embedded_asset, load_internal_asset, uuid_handle},
    prelude::*,
    render::render_resource::{AsBindGroup, ShaderType},
    shader::{Shader, ShaderRef},
    sprite_render::{AlphaMode2d, Material2d, Material2dPlugin},
};

use crate::quantize_material::{ColorQuantizeUniforms, QuantizationConfig};

/// Handle for the pixelation functions shader, loaded as an internal asset so
/// other shaders can import it via `#import msg_shaders::pixelate_functions`.
pub const PIXELATE_FUNCTIONS_SHADER_HANDLE: Handle<Shader> =
    uuid_handle!("1f2e3d4c-5b6a-7980-1234-56789abcdef0");

/// Configuration for the pixelation effect.
#[derive(Clone, Copy, Debug)]
pub struct PixelateConfig {
    /// Number of cells across each axis over the sampled `[0, 1]` range.
    /// A component of `0.0` disables snapping on that axis, so `Vec2::ZERO`
    /// is a no-op.
    pub grid: Vec2,
}

impl Default for PixelateConfig {
    fn default() -> Self {
        Self { grid: Vec2::ZERO }
    }
}

impl PixelateConfig {
    /// Square grid of `cells` cells across both axes.
    #[must_use]
    pub fn square(cells: f32) -> Self {
        Self {
            grid: Vec2::splat(cells),
        }
    }

    /// Convert to shader uniforms.
    pub fn to_uniforms(&self) -> PixelateUniforms {
        PixelateUniforms {
            grid: self.grid,
            _pad: Vec2::ZERO,
        }
    }
}

/// Shader uniforms for pixelation.
///
/// Field order matches the WGSL `PixelateSettings` struct layout exactly.
#[derive(Clone, Copy, ShaderType)]
pub struct PixelateUniforms {
    /// Number of cells across each axis. A component of 0 disables snapping.
    pub grid: Vec2,
    pub _pad: Vec2,
}

impl Default for PixelateUniforms {
    fn default() -> Self {
        PixelateConfig::default().to_uniforms()
    }
}

/// Material that pixelates a texture by snapping its sample to a grid.
#[derive(Asset, TypePath, AsBindGroup, Clone)]
pub struct PixelateMaterial {
    /// The source texture to pixelate.
    #[texture(0)]
    #[sampler(1)]
    pub texture: Handle<Image>,
    /// Pixelation settings.
    #[uniform(2)]
    pub pixelate: PixelateUniforms,
}

impl PixelateMaterial {
    /// Create a new material with the given texture and pixelation config.
    #[must_use]
    pub fn new(texture: Handle<Image>, config: PixelateConfig) -> Self {
        Self {
            texture,
            pixelate: config.to_uniforms(),
        }
    }
}

impl Material2d for PixelateMaterial {
    fn fragment_shader() -> ShaderRef {
        "embedded://msg_shaders/shaders/pixelate_material.wgsl".into()
    }

    fn alpha_mode(&self) -> AlphaMode2d {
        AlphaMode2d::Blend
    }
}

/// Material that pixelates a texture and reduces it to a palette in one pass.
#[derive(Asset, TypePath, AsBindGroup, Clone)]
pub struct QuantizePixelateMaterial {
    /// The source texture to pixelate and quantize.
    #[texture(0)]
    #[sampler(1)]
    pub texture: Handle<Image>,
    /// Quantization settings.
    #[uniform(2)]
    pub quantization: ColorQuantizeUniforms,
    /// Pixelation settings.
    #[uniform(3)]
    pub pixelate: PixelateUniforms,
}

impl QuantizePixelateMaterial {
    /// Create a new material from a texture, quantization config, and pixelation config.
    #[must_use]
    pub fn new(
        texture: Handle<Image>,
        quantization: QuantizationConfig,
        pixelate: PixelateConfig,
    ) -> Self {
        Self {
            texture,
            quantization: quantization.to_uniforms(),
            pixelate: pixelate.to_uniforms(),
        }
    }
}

impl Material2d for QuantizePixelateMaterial {
    fn fragment_shader() -> ShaderRef {
        "embedded://msg_shaders/shaders/quantize_pixelate_material.wgsl".into()
    }

    fn alpha_mode(&self) -> AlphaMode2d {
        AlphaMode2d::Blend
    }
}

/// Plugin that enables pixelation shader imports and standalone materials.
///
/// Registering this also requires [`crate::ColorQuantizationPlugin`] for the
/// combined [`QuantizePixelateMaterial`], which imports the quantization
/// functions shader.
///
/// # Example
///
/// ```
/// use bevy::prelude::*;
/// use msg_shaders::PixelationPlugin;
///
/// fn register_plugin(app: &mut App) {
///     app.add_plugins(PixelationPlugin);
/// }
/// ```
pub struct PixelationPlugin;

impl Plugin for PixelationPlugin {
    fn build(&self, app: &mut App) {
        // The combined material imports the quantization functions shader, so
        // make sure that module is registered.
        if !app.is_plugin_added::<crate::ColorQuantizationPlugin>() {
            app.add_plugins(crate::ColorQuantizationPlugin);
        }

        load_internal_asset!(
            app,
            PIXELATE_FUNCTIONS_SHADER_HANDLE,
            "shaders/pixelate_functions.wgsl",
            Shader::from_wgsl
        );

        embedded_asset!(app, "shaders/pixelate_material.wgsl");
        embedded_asset!(app, "shaders/quantize_pixelate_material.wgsl");

        if !app.is_plugin_added::<Material2dPlugin<PixelateMaterial>>() {
            app.add_plugins(Material2dPlugin::<PixelateMaterial>::default());
        }
        if !app.is_plugin_added::<Material2dPlugin<QuantizePixelateMaterial>>() {
            app.add_plugins(Material2dPlugin::<QuantizePixelateMaterial>::default());
        }
    }
}
