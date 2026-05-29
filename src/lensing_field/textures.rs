// GPU textures for the lensing-field fluid simulation.
//
// Allocates the ping-pong velocity pair (Rg16Float) plus pressure and
// divergence scratch textures (R16Float), all at LENSING_FIELD_RES².
// The read-side velocity handle is the one LensingMaterial samples each
// frame; the write side is swapped by the node after each pass.

use bevy::{
    asset::RenderAssetUsages,
    prelude::*,
    render::render_resource::{Extent3d, TextureDimension, TextureFormat, TextureUsages},
};

use crate::lensing_field::LENSING_FIELD_RES;

/// Ping-pong velocity textures (Rg16Float) plus pressure and divergence
/// scratch textures (R16Float), all LENSING_FIELD_RES².
///
/// Inserted as a resource during plugin startup. The render node swaps
/// `velocity_ping` and `velocity_pong` each frame so the GPU never reads
/// and writes the same texture in one pass.
///
/// `velocity_read()` returns whichever handle `LensingMaterial` should
/// sample this frame; `velocity_write()` is the storage target.
#[derive(Resource, Clone)]
pub struct LensingFieldTextures {
    /// Velocity ping buffer (Rg16Float). Active read target on odd frames.
    pub velocity_ping: Handle<Image>,
    /// Velocity pong buffer (Rg16Float). Active read target on even frames.
    pub velocity_pong: Handle<Image>,
    /// Pressure scratch (R16Float). Ping buffer for Jacobi iteration.
    pub pressure_ping: Handle<Image>,
    /// Pressure scratch (R16Float). Pong buffer for Jacobi iteration.
    pub pressure_pong: Handle<Image>,
    /// Divergence of the velocity field (R16Float). Written once per frame.
    pub divergence: Handle<Image>,
    /// Which velocity buffer is currently the read target (`true` = ping).
    pub read_is_ping: bool,
}

impl LensingFieldTextures {
    /// Handle the fragment shader should sample this frame.
    pub fn velocity_read(&self) -> &Handle<Image> {
        if self.read_is_ping {
            &self.velocity_ping
        } else {
            &self.velocity_pong
        }
    }

    /// Handle the compute shader writes into this frame.
    pub fn velocity_write(&self) -> &Handle<Image> {
        if self.read_is_ping {
            &self.velocity_pong
        } else {
            &self.velocity_ping
        }
    }

    /// Swap read/write roles after all compute passes for this frame finish.
    pub fn swap(&mut self) {
        self.read_is_ping = !self.read_is_ping;
    }
}

/// Allocates and registers all four lensing-field textures into `images`.
///
/// Called once during plugin startup on the main-world `Assets<Image>`.
pub fn allocate_lensing_textures(images: &mut Assets<Image>) -> LensingFieldTextures {
    let res = LENSING_FIELD_RES;
    let extent = Extent3d {
        width: res,
        height: res,
        depth_or_array_layers: 1,
    };

    let velocity_usage =
        TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST;

    let make_rg = || {
        let bytes_per_pixel = 4u32; // Rg16Float = 2×f16
        let mut img = Image::new(
            extent,
            TextureDimension::D2,
            vec![0u8; (bytes_per_pixel * res * res) as usize],
            TextureFormat::Rg16Float,
            RenderAssetUsages::RENDER_WORLD,
        );
        img.texture_descriptor.usage = velocity_usage;
        img
    };

    let make_r = || {
        let bytes_per_pixel = 2u32; // R16Float = 1×f16
        let mut img = Image::new(
            extent,
            TextureDimension::D2,
            vec![0u8; (bytes_per_pixel * res * res) as usize],
            TextureFormat::R16Float,
            RenderAssetUsages::RENDER_WORLD,
        );
        img.texture_descriptor.usage = velocity_usage;
        img
    };

    LensingFieldTextures {
        velocity_ping: images.add(make_rg()),
        velocity_pong: images.add(make_rg()),
        pressure_ping: images.add(make_r()),
        pressure_pong: images.add(make_r()),
        divergence: images.add(make_r()),
        read_is_ping: true,
    }
}
