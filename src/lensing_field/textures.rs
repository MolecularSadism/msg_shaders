// GPU textures for the lensing-field flow simulation.
//
// Two Rg16Float velocity textures at LENSING_FIELD_RES². Within each frame
// the inject pass writes the decayed-plus-force-injected field into one
// (the "intermediate"), advect reads it and writes the advected result into
// the other (the "persistent state"). The lensing display pass samples the
// persistent-state handle; the next frame's inject reads the same handle to
// recover last frame's field.

use bevy::{
    asset::RenderAssetUsages,
    prelude::*,
    render::render_resource::{Extent3d, TextureDimension, TextureFormat, TextureUsages},
};

use crate::lensing_field::LENSING_FIELD_RES;

/// Two Rg16Float velocity textures at LENSING_FIELD_RES². Inserted as a
/// resource during plugin startup.
#[derive(Resource, Clone)]
pub struct LensingFieldTextures {
    /// Intermediate buffer. Inject writes here; advect reads from here.
    pub velocity_ping: Handle<Image>,
    /// Persistent state. Advect writes here; the material samples this handle,
    /// and next frame's inject reads it to apply decay before adding new force.
    pub velocity_pong: Handle<Image>,
}

impl LensingFieldTextures {
    /// Inject's output / advect's input.
    pub fn velocity_read(&self) -> &Handle<Image> {
        &self.velocity_ping
    }

    /// Advect's output, and the handle `LensingMaterial` samples. Carries the
    /// field's persistent state across frames.
    pub fn velocity_write(&self) -> &Handle<Image> {
        &self.velocity_pong
    }
}

/// Allocates and registers the velocity texture pair into `images`.
pub fn allocate_lensing_textures(images: &mut Assets<Image>) -> LensingFieldTextures {
    let res = LENSING_FIELD_RES;
    let extent = Extent3d {
        width: res,
        height: res,
        depth_or_array_layers: 1,
    };

    let usage =
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
        img.texture_descriptor.usage = usage;
        img
    };

    LensingFieldTextures {
        velocity_ping: images.add(make_rg()),
        velocity_pong: images.add(make_rg()),
    }
}
