//! Pixel-art planet visuals with an injectable colour palette.
//!
//! A port/fork of the planet shaders from
//! [Deep-Fold/PixelPlanets](https://github.com/Deep-Fold/PixelPlanets) (MIT, see
//! `LICENSE-THIRD-PARTY`), by way of `l-kldzj/game-tools/pixel-gen`. Unlike upstream — which
//! ships fixed colour schemes — this fork takes the colour ramp from the caller via
//! [`PlanetPalette`], so a project can inject its own art-direction colours.
//!
//! Add [`PixelPlanetPlugin`] to a rendering [`App`], then spawn a [`PixelPlanet`] wherever you
//! want the planet (e.g. as a child of a level root). The plugin fills in the layered visual as
//! children on the next update: one `Mesh2d` per shader layer, composited back-to-front. Gate
//! [`PlanetSpinSystems`] to freeze the spin — while paused, say.
//!
//! The pixel grid lives in the planet quad's own space by default, so it turns and moves with
//! the quad. Add the plugin as `PixelPlanetPlugin { pixel_grid: PixelGrid::World }` to pin the
//! grid to whole world pixels instead; see [`PixelGrid`].
//!
//! ```
//! use bevy::prelude::*;
//! use msg_shaders::{PixelPlanet, PlanetPalette, PlanetSpec, PlanetType};
//!
//! let spec = PlanetSpec {
//!     planet_type: PlanetType::GasGiant,
//!     diameter: 256.0,
//!     palette: PlanetPalette::new([Color::srgb(0.9, 0.7, 0.4), Color::srgb(0.2, 0.1, 0.05)]),
//!     seed: 4.0,
//!     rotation: 0.3,
//!     rotation_speed: 0.05,
//!     light_origin: Vec2::new(0.35, 0.35),
//!     clouds: false,
//!     ring: true,
//!     dither: Some(1.5),
//!     pixels: 100.0,
//!     octaves: 4,
//!     noise_size: 8.0,
//! };
//!
//! // A real app adds PixelPlanetPlugin::default() here; it needs a render app, which
//! // MinimalPlugins does not provide.
//! let mut app = App::new();
//! app.add_plugins(MinimalPlugins);
//! app.world_mut().spawn((PixelPlanet(spec), Transform::default(), Visibility::Visible));
//! ```

mod grid;
mod material;
mod palette;
mod spawn;

use bevy::asset::{load_internal_asset, uuid_handle};
use bevy::prelude::*;
use bevy::shader::Shader;
use bevy::sprite_render::Material2dPlugin;

use material::{
    CloudsMaterial, GasLayersMaterial, NoAtmosphereMaterial, PlanetLandmassMaterial,
    PlanetUnderMaterial, RingMaterial,
};

pub use grid::PixelGrid;
pub use palette::PlanetPalette;
pub use spawn::{PixelPlanet, PlanetSpec, PlanetType};

/// System set containing the per-frame surface-animation systems that advance
/// each planet material's `time` uniform (the spin effect).
///
/// Grouped so consumers can gate the animation without knowing the individual
/// per-material systems — e.g. configure this set to run only while unpaused so
/// planets freeze in place while the game is paused.
#[derive(SystemSet, Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub struct PlanetSpinSystems;

/// Handle for the shared pixel-planet WGSL library (noise, dithering), imported as
/// `msg_shaders::pixel_planet_functions`.
pub const PIXEL_PLANET_FUNCTIONS_SHADER_HANDLE: Handle<Shader> =
    uuid_handle!("a1f0c2d4-3b6e-4a91-8c2f-1d5e7a9b0c31");

/// Source of the `msg_shaders::pixel_planet_functions` WGSL library, for offline validation
/// of downstream shaders that import it.
pub const PIXEL_PLANET_FUNCTIONS_WGSL: &str =
    include_str!("../shaders/pixel_planet_functions.wgsl");

pub(crate) const NO_ATMOSPHERE_SHADER: Handle<Shader> =
    uuid_handle!("b2e1d3c5-4c7f-4b82-9d3a-2e6f8b0c1d42");
pub(crate) const PLANET_UNDER_SHADER: Handle<Shader> =
    uuid_handle!("c3d2e4b6-5d80-4c73-ae4b-3f70a91c2e53");
pub(crate) const PLANET_LANDMASS_SHADER: Handle<Shader> =
    uuid_handle!("d4c3f5a7-6e91-4d64-bf5c-4081ba2d3f64");
pub(crate) const CLOUDS_SHADER: Handle<Shader> =
    uuid_handle!("e5b4a6b8-7f02-4e55-806d-5192cb3e4075");
pub(crate) const GAS_LAYERS_SHADER: Handle<Shader> =
    uuid_handle!("f6a5b7c9-8013-4f46-9d7e-62a3dc4f5186");
pub(crate) const RING_SHADER: Handle<Shader> = uuid_handle!("07b6c8da-9124-4037-a28f-73b4ed506297");
pub(crate) const WORLD_GRID_SHADER: Handle<Shader> =
    uuid_handle!("18c7d9eb-a235-4148-b39a-84c5fe617308");
pub(crate) const WORLD_GRID_VERTEX_SHADER: Handle<Shader> =
    uuid_handle!("29d8eafc-b346-4259-84ab-95d60f728419");

/// Source of the `msg_shaders::pixel_planet_world_grid` WGSL library: the world-pixel snap of
/// [`PixelGrid::World`], for offline validation.
pub const PIXEL_PLANET_WORLD_GRID_WGSL: &str =
    include_str!("../shaders/pixel_planet_world_grid.wgsl");

/// The plugin's settings as the layer-assembly system reads them.
#[derive(Resource, Clone, Copy)]
pub(crate) struct PixelPlanetSettings {
    pub(crate) pixel_grid: PixelGrid,
}

/// Registers the shared shader library `msg_shaders::pixel_planet_functions` alone (noise,
/// dithering), for shaders outside this module that import it. [`PixelPlanetPlugin`] adds it
/// when missing.
pub struct PixelPlanetShaderLibraryPlugin;

impl Plugin for PixelPlanetShaderLibraryPlugin {
    fn build(&self, app: &mut App) {
        load_internal_asset!(
            app,
            PIXEL_PLANET_FUNCTIONS_SHADER_HANDLE,
            "../shaders/pixel_planet_functions.wgsl",
            Shader::from_wgsl
        );
    }
}

/// Registers the planet shaders, materials, and the layer-assembly system.
///
/// `PixelPlanetPlugin::default()` draws the pixel grid in each planet quad's own space
/// ([`PixelGrid::Quad`]). A game that rotates its planets or places them off whole pixels, and
/// wants them on the world's pixel grid regardless, adds
/// `PixelPlanetPlugin { pixel_grid: PixelGrid::World }`.
#[derive(Default, Clone, Copy, Debug)]
pub struct PixelPlanetPlugin {
    /// The space the pixel grid of every planet lives in.
    pub pixel_grid: PixelGrid,
}

impl Plugin for PixelPlanetPlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<PixelPlanetShaderLibraryPlugin>() {
            app.add_plugins(PixelPlanetShaderLibraryPlugin);
        }
        app.insert_resource(PixelPlanetSettings {
            pixel_grid: self.pixel_grid,
        });
        load_internal_asset!(
            app,
            WORLD_GRID_SHADER,
            "../shaders/pixel_planet_world_grid.wgsl",
            Shader::from_wgsl
        );
        load_internal_asset!(
            app,
            WORLD_GRID_VERTEX_SHADER,
            "../shaders/pixel_planet_world_grid_vertex.wgsl",
            Shader::from_wgsl
        );
        load_internal_asset!(
            app,
            NO_ATMOSPHERE_SHADER,
            "../shaders/pixel_planet_no_atmosphere.wgsl",
            Shader::from_wgsl
        );
        load_internal_asset!(
            app,
            PLANET_UNDER_SHADER,
            "../shaders/pixel_planet_under.wgsl",
            Shader::from_wgsl
        );
        load_internal_asset!(
            app,
            PLANET_LANDMASS_SHADER,
            "../shaders/pixel_planet_landmass.wgsl",
            Shader::from_wgsl
        );
        load_internal_asset!(
            app,
            CLOUDS_SHADER,
            "../shaders/pixel_planet_clouds.wgsl",
            Shader::from_wgsl
        );
        load_internal_asset!(
            app,
            GAS_LAYERS_SHADER,
            "../shaders/pixel_planet_gas_layers.wgsl",
            Shader::from_wgsl
        );
        load_internal_asset!(
            app,
            RING_SHADER,
            "../shaders/pixel_planet_ring.wgsl",
            Shader::from_wgsl
        );

        app.add_plugins((
            Material2dPlugin::<NoAtmosphereMaterial>::default(),
            Material2dPlugin::<PlanetUnderMaterial>::default(),
            Material2dPlugin::<PlanetLandmassMaterial>::default(),
            Material2dPlugin::<CloudsMaterial>::default(),
            Material2dPlugin::<GasLayersMaterial>::default(),
            Material2dPlugin::<RingMaterial>::default(),
        ));

        app.add_systems(Update, spawn::build_pixel_planets);
        app.add_systems(
            Update,
            (
                spawn::advance_planet_spin::<NoAtmosphereMaterial>,
                spawn::advance_planet_spin::<PlanetUnderMaterial>,
                spawn::advance_planet_spin::<PlanetLandmassMaterial>,
                spawn::advance_planet_spin::<CloudsMaterial>,
                spawn::advance_planet_spin::<GasLayersMaterial>,
                spawn::advance_planet_spin::<RingMaterial>,
            )
                .in_set(PlanetSpinSystems),
        );
    }
}
