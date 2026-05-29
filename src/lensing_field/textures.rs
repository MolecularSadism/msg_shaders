// GPU textures for the lensing-field fluid simulation.
//
// Allocates a pair of Rg16Float velocity textures plus pressure and divergence
// scratch (R16Float), all at LENSING_FIELD_RES². The pair is not ping-ponged
// across frames: within each frame the compute passes shuttle data between
// them in a fixed order (inject → advect → divergence → Jacobi → gradient
// subtract), and the final solved velocity always lands in `velocity_write`.
// `LensingMaterial` samples `velocity_write`; inject reads `velocity_read` to
// recover last frame's post-advect state for its decay term.

use bevy::{
    asset::RenderAssetUsages,
    prelude::*,
    render::render_resource::{Extent3d, TextureDimension, TextureFormat, TextureUsages},
};

use crate::lensing_field::LENSING_FIELD_RES;

/// Velocity pair (Rg16Float) plus pressure and divergence scratch (R16Float),
/// all LENSING_FIELD_RES². Inserted as a resource during plugin startup.
#[derive(Resource, Clone)]
pub struct LensingFieldTextures {
    /// Intermediate velocity buffer. Inject reads this for decay; advect writes
    /// here; divergence and gradient-subtract read it.
    pub velocity_ping: Handle<Image>,
    /// Final velocity output. Gradient-subtract writes its result here, and
    /// `LensingMaterial` samples this handle each frame.
    pub velocity_pong: Handle<Image>,
    /// Pressure scratch (R16Float). Ping buffer for Jacobi iteration.
    pub pressure_ping: Handle<Image>,
    /// Pressure scratch (R16Float). Pong buffer for Jacobi iteration.
    pub pressure_pong: Handle<Image>,
    /// Divergence of the velocity field (R16Float). Written once per frame.
    pub divergence: Handle<Image>,
}

impl LensingFieldTextures {
    /// Inject's decay input; advect's output. Holds last frame's post-advect
    /// state at the start of the next frame's inject pass.
    pub fn velocity_read(&self) -> &Handle<Image> {
        &self.velocity_ping
    }

    /// Gradient-subtract's output: the divergence-free velocity field for the
    /// current frame. `LensingMaterial` samples this handle.
    pub fn velocity_write(&self) -> &Handle<Image> {
        &self.velocity_pong
    }
}

/// Allocates and registers all lensing-field textures into `images`.
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
    }
}
