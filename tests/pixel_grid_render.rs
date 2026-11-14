//! Renders planets through the real shaders into an offscreen target and
//! checks the pixel grid of both [`PixelGrid`] modes.
//!
//! Needs a graphics adapter, so the test is ignored by default. On a machine
//! without a GPU it runs on a software Vulkan driver such as lavapipe:
//!
//! ```sh
//! VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json WGPU_BACKEND=vulkan \
//!     cargo test --test pixel_grid_render -- --ignored
//! ```

#![cfg(feature = "render_2d")]

use std::sync::{Arc, Mutex};

use bevy::app::PluginsState;
use bevy::camera::RenderTarget;
use bevy::prelude::*;
use bevy::render::RenderPlugin;
use bevy::render::gpu_readback::{Readback, ReadbackComplete};
use bevy::render::render_resource::{TextureFormat, TextureUsages};
use bevy::window::ExitCondition;
use bevy::winit::WinitPlugin;
use msg_shaders::{
    PixelGrid, PixelPlanet, PixelPlanetPlugin, PlanetPalette, PlanetSpec, PlanetType,
};

/// The latest frame read back from the render target, as RGBA8 bytes.
#[derive(Resource, Clone, Default)]
struct LatestFrame(Arc<Mutex<Option<Vec<u8>>>>);

/// One offscreen render of a single planet.
#[derive(Clone)]
struct PlanetRender {
    pixel_grid: PixelGrid,
    planet_type: PlanetType,
    /// Rotation of the planet entity, in degrees.
    angle_degrees: f32,
    /// World position of the planet entity.
    position: Vec2,
    /// `PlanetSpec::pixels`.
    pixels: f32,
}

/// The pixels a [`PlanetRender`] produced.
struct RenderedPlanet {
    /// RGBA8, `PlanetRender::TARGET` texels square.
    rgba: Vec<u8>,
}

impl PlanetRender {
    /// Diameter of the planet body in world units.
    const DIAMETER: f32 = 48.0;
    /// Texels per world unit: every world pixel covers a block this wide, so a
    /// world pixel that is not one flat colour shows.
    const TEXELS_PER_WORLD_UNIT: u32 = 4;
    /// Side of the square render target in texels; the camera sees
    /// `TARGET / TEXELS_PER_WORLD_UNIT` world units, centred on the origin.
    const TARGET: u32 = 512;
    /// Frames rendered before the target is read: the layers spawn on the
    /// first update and the readback lags the render by a few frames.
    const FRAMES: u32 = 12;

    fn spec(&self) -> PlanetSpec {
        PlanetSpec {
            planet_type: self.planet_type,
            diameter: Self::DIAMETER,
            palette: PlanetPalette::new([
                Color::srgb(0.95, 0.85, 0.55),
                Color::srgb(0.45, 0.7, 0.9),
                Color::srgb(0.1, 0.15, 0.4),
            ]),
            seed: 4.0,
            rotation: 0.3,
            rotation_speed: 0.0,
            light_origin: Vec2::new(0.35, 0.35),
            clouds: true,
            ring: true,
            dither: Some(1.5),
            pixels: self.pixels,
            octaves: 4,
            noise_size: 8.0,
        }
    }

    fn render(&self) -> RenderedPlanet {
        let mut app = App::new();
        app.add_plugins(
            DefaultPlugins
                .build()
                .disable::<WinitPlugin>()
                .set(WindowPlugin {
                    primary_window: None,
                    exit_condition: ExitCondition::DontExit,
                    ..default()
                })
                .set(RenderPlugin {
                    // Pipelines are ready on the frame that first needs them.
                    synchronous_pipeline_compilation: true,
                    ..default()
                }),
        );
        app.add_plugins(PixelPlanetPlugin {
            pixel_grid: self.pixel_grid,
        });
        let frame = LatestFrame::default();
        app.insert_resource(frame.clone());

        let mut target = Image::new_target_texture(
            Self::TARGET,
            Self::TARGET,
            TextureFormat::Rgba8UnormSrgb,
            None,
        );
        target.texture_descriptor.usage |= TextureUsages::COPY_SRC;
        let target = app.world_mut().resource_mut::<Assets<Image>>().add(target);

        let world = app.world_mut();
        world.spawn((
            Camera2d,
            Camera {
                clear_color: ClearColorConfig::Custom(Color::NONE),
                ..default()
            },
            RenderTarget::Image(target.clone().into()),
            Projection::Orthographic(OrthographicProjection {
                scale: 1.0 / Self::TEXELS_PER_WORLD_UNIT as f32,
                ..OrthographicProjection::default_2d()
            }),
            Msaa::Off,
        ));
        world.spawn((
            PixelPlanet(self.spec()),
            Transform::from_translation(self.position.extend(0.0))
                .with_rotation(Quat::from_rotation_z(self.angle_degrees.to_radians())),
            Visibility::Visible,
        ));
        world.spawn(Readback::texture(target)).observe(
            |event: On<ReadbackComplete>, frame: Res<LatestFrame>| {
                *frame.0.lock().expect("frame lock") = Some(event.data.clone());
            },
        );

        while app.plugins_state() == PluginsState::Adding {
            bevy::tasks::tick_global_task_pools_on_main_thread();
        }
        app.finish();
        app.cleanup();
        for _ in 0..Self::FRAMES {
            app.update();
        }

        let rgba = frame
            .0
            .lock()
            .expect("frame lock")
            .take()
            .expect("the render target was read back");
        assert_eq!(rgba.len(), (Self::TARGET * Self::TARGET * 4) as usize);
        RenderedPlanet { rgba }
    }
}

impl RenderedPlanet {
    fn texel(&self, x: u32, y: u32) -> [u8; 4] {
        let at = ((y * PlanetRender::TARGET + x) * 4) as usize;
        [
            self.rgba[at],
            self.rgba[at + 1],
            self.rgba[at + 2],
            self.rgba[at + 3],
        ]
    }

    /// World pixels in the view, per axis.
    fn world_pixels_across() -> u32 {
        PlanetRender::TARGET / PlanetRender::TEXELS_PER_WORLD_UNIT
    }

    /// The colours of one world pixel's block of texels, deduplicated.
    fn world_pixel_colours(&self, x: u32, y: u32) -> Vec<[u8; 4]> {
        let block = PlanetRender::TEXELS_PER_WORLD_UNIT;
        let mut colours = Vec::new();
        for texel_y in y * block..(y + 1) * block {
            for texel_x in x * block..(x + 1) * block {
                let colour = self.texel(texel_x, texel_y);
                if !colours.contains(&colour) {
                    colours.push(colour);
                }
            }
        }
        colours
    }

    /// How many world pixels are drawn at all, and how many of all world
    /// pixels are not one flat colour.
    fn world_pixel_census(&self) -> (u32, u32) {
        let mut drawn = 0;
        let mut split = 0;
        for y in 0..Self::world_pixels_across() {
            for x in 0..Self::world_pixels_across() {
                let colours = self.world_pixel_colours(x, y);
                if colours.iter().any(|colour| colour[3] != 0) {
                    drawn += 1;
                }
                if colours.len() > 1 {
                    split += 1;
                }
            }
        }
        (drawn, split)
    }

    /// Distinct opaque colours in the whole image.
    fn palette_size(&self) -> usize {
        let mut colours: Vec<[u8; 4]> = Vec::new();
        for texel in self.rgba.chunks_exact(4) {
            let colour = [texel[0], texel[1], texel[2], texel[3]];
            if colour[3] != 0 && !colours.contains(&colour) {
                colours.push(colour);
            }
        }
        colours.len()
    }

    /// Texels that differ from `other`.
    fn texels_differing_from(&self, other: &RenderedPlanet) -> usize {
        self.rgba
            .chunks_exact(4)
            .zip(other.rgba.chunks_exact(4))
            .filter(|(a, b)| a != b)
            .count()
    }
}

/// All assertions share one test: each render starts its own Bevy app, and
/// they must not run side by side.
#[test]
#[ignore = "needs a GPU or a software Vulkan driver; see the module docs"]
fn world_grid_draws_whole_world_pixels() {
    for planet_type in PlanetType::ALL {
        let rotated = PlanetRender {
            pixel_grid: PixelGrid::World,
            planet_type,
            angle_degrees: 33.0,
            position: Vec2::new(10.37, -4.81),
            pixels: PlanetRender::DIAMETER,
        };

        // World grid, turned by a non-right angle at a fractional position:
        // every world pixel is one flat colour, across all layers.
        let world = rotated.render();
        let (drawn, split) = world.world_pixel_census();
        println!(
            "{planet_type:?} world rotated: {drawn} world pixels drawn, {split} split, {} colours",
            world.palette_size()
        );
        // Most of the body's disc is drawn (a terminator or a dithered rim may
        // leave some of it out), in more than one colour.
        let disc_area = std::f32::consts::PI * (PlanetRender::DIAMETER / 2.0).powi(2);
        assert!(
            drawn as f32 > disc_area / 2.0,
            "{planet_type:?}: the planet was not drawn"
        );
        assert!(world.palette_size() > 1, "{planet_type:?}: no shading");
        assert_eq!(split, 0, "{planet_type:?}: world pixels with two colours");

        // The same planet on the quad grid does split world pixels, so the
        // census above can tell the modes apart.
        let quad = PlanetRender {
            pixel_grid: PixelGrid::Quad,
            ..rotated.clone()
        }
        .render();
        let (_, quad_split) = quad.world_pixel_census();
        println!("{planet_type:?} quad rotated: {quad_split} split");
        assert!(quad_split > 0, "{planet_type:?}: quad grid did not split");

        // A cell count that does not match the diameter changes nothing in
        // world mode: cells stay whole world units.
        let mismatched = PlanetRender {
            pixels: PlanetRender::DIAMETER * 0.4,
            ..rotated.clone()
        }
        .render();
        assert_eq!(
            mismatched.texels_differing_from(&world),
            0,
            "{planet_type:?}: `pixels` changed the world grid"
        );

        // Unrotated on whole world pixels with `pixels` equal to the diameter,
        // both modes draw the same picture.
        let aligned = PlanetRender {
            angle_degrees: 0.0,
            position: Vec2::new(6.0, -9.0),
            ..rotated.clone()
        };
        let aligned_world = aligned.render();
        let aligned_quad = PlanetRender {
            pixel_grid: PixelGrid::Quad,
            ..aligned
        }
        .render();
        let (aligned_drawn, aligned_split) = aligned_world.world_pixel_census();
        let differing = aligned_world.texels_differing_from(&aligned_quad);
        println!(
            "{planet_type:?} aligned: {aligned_drawn} drawn, {aligned_split} split, \
             {differing} texels differ between the modes"
        );
        assert_eq!(aligned_split, 0);
        assert_eq!(differing, 0, "{planet_type:?}: modes differ when aligned");
    }
}
