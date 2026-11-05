//! Planet assembly: turns a [`PixelPlanet`] component into a stack of layered
//! `Mesh2d` children, one per shader layer (plus an optional ring).

use bevy::prelude::*;

use bevy::sprite_render::Material2d;

use super::material::{
    CloudsMaterial, CommonParams, GasLayersMaterial, NoAtmosphereMaterial, PlanetLandmassMaterial,
    PlanetMaterial, PlanetUnderMaterial, RingMaterial,
};
use super::palette::PlanetPalette;

/// The kind of planet to generate. Each type maps to a distinct stack of
/// shader layers ported from Deep-Fold/PixelPlanets.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Reflect)]
pub enum PlanetType {
    /// Single airless rocky body.
    Barren,
    /// Ocean base + continents, optionally with a cloud layer.
    Terran,
    /// Banded gas giant, optionally with a ring.
    GasGiant,
}

impl PlanetType {
    /// Every supported planet type, for random selection.
    pub const ALL: [PlanetType; 3] = [PlanetType::Barren, PlanetType::Terran, PlanetType::GasGiant];

    /// Whether this type can carry a cloud layer.
    #[must_use]
    pub fn supports_clouds(self) -> bool {
        matches!(self, PlanetType::Terran)
    }

    /// Whether this type can carry a ring.
    #[must_use]
    pub fn supports_ring(self) -> bool {
        matches!(self, PlanetType::GasGiant)
    }
}

/// Full description of a planet to spawn. The light direction and any
/// randomization are decided by the caller; the crate just renders it.
#[derive(Clone)]
pub struct PlanetSpec {
    /// Which layer stack to build.
    pub planet_type: PlanetType,
    /// Diameter of the planet body in world units.
    pub diameter: f32,
    /// Injected colour ramp.
    pub palette: PlanetPalette,
    /// Noise seed (`1.0..=10.0`).
    pub seed: f32,
    /// Surface tilt in radians (handled inside the shader, not on the entity).
    pub rotation: f32,
    /// Surface rotation speed: how fast the shader `time` advances each second,
    /// scrolling the surface so the planet appears to spin. `0.0` is static;
    /// negative values reverse the spin direction.
    pub rotation_speed: f32,
    /// Highlight position in `[0, 1]` UV space (the lit side of the planet).
    pub light_origin: Vec2,
    /// Include the cloud layer (only meaningful when the type supports it).
    pub clouds: bool,
    /// Include the ring (only meaningful when the type supports it).
    pub ring: bool,
    /// Dithered colour banding and silhouette/ring-rim strength. `None` keeps
    /// hard edges and solid colour bands; `Some(strength)` enables both, with
    /// `strength` widening the stippled edge fade.
    pub dither: Option<f32>,
    /// Pixelation resolution.
    pub pixels: f32,
    /// fbm octave count.
    pub octaves: i32,
    /// Noise frequency.
    pub noise_size: f32,
}

impl PlanetSpec {
    fn common(&self) -> CommonParams {
        CommonParams {
            light_origin: self.light_origin,
            pixels: self.pixels,
            rotation: self.rotation,
            size: self.noise_size,
            seed: self.seed,
            octaves: self.octaves,
            dither: self.dither,
        }
    }
}

/// Component requesting a pixel-art planet. Spawn it on an entity (positioned
/// wherever you want the planet, e.g. as a child of a level root) and the crate
/// fills in the layered visual as children on the next update.
#[derive(Component, Clone)]
pub struct PixelPlanet(pub PlanetSpec);

/// Per-layer surface rotation speed, copied from [`PlanetSpec::rotation_speed`]
/// onto each spawned layer entity. [`advance_planet_spin`] reads it to advance
/// that layer material's `time` uniform every frame.
#[derive(Component, Clone, Copy)]
pub(crate) struct PlanetSpin {
    /// Shader `time` units added per second.
    pub(crate) speed: f32,
}

// Local-Z offsets so the layers composite back-to-front within one planet. The
// ring sits in *front* of the body: its shader clips the back arc to the planet
// silhouette and keeps the front arc drawn over the planet.
const Z_BODY: f32 = 0.0;
const Z_LAND: f32 = 0.01;
const Z_CLOUDS: f32 = 0.02;
const Z_RING: f32 = 0.03;

/// How much larger the ring quad is than the body, expressed as the
/// `scale_rel_to_planet` the ring shader expects.
const RING_SCALE_REL_TO_PLANET: f32 = 4.0;

/// Builds the layer children for every newly-added [`PixelPlanet`].
pub(crate) fn build_pixel_planets(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut m_barren: ResMut<Assets<NoAtmosphereMaterial>>,
    mut m_under: ResMut<Assets<PlanetUnderMaterial>>,
    mut m_land: ResMut<Assets<PlanetLandmassMaterial>>,
    mut m_clouds: ResMut<Assets<CloudsMaterial>>,
    mut m_gas: ResMut<Assets<GasLayersMaterial>>,
    mut m_ring: ResMut<Assets<RingMaterial>>,
    q: Query<(Entity, &PixelPlanet), Added<PixelPlanet>>,
) {
    for (entity, planet) in &q {
        let spec = &planet.0;
        let common = spec.common();
        let spin = PlanetSpin {
            speed: spec.rotation_speed,
        };
        let body_mesh = meshes.add(Rectangle::new(spec.diameter, spec.diameter));

        match spec.planet_type {
            PlanetType::Barren => {
                spawn_layer(
                    &mut commands,
                    entity,
                    body_mesh.clone(),
                    Z_BODY,
                    spin,
                    MeshMaterial2d(m_barren.add(NoAtmosphereMaterial::new(common, &spec.palette))),
                );
            }
            PlanetType::Terran => {
                spawn_layer(
                    &mut commands,
                    entity,
                    body_mesh.clone(),
                    Z_BODY,
                    spin,
                    MeshMaterial2d(m_under.add(PlanetUnderMaterial::new(common, &spec.palette))),
                );
                spawn_layer(
                    &mut commands,
                    entity,
                    body_mesh.clone(),
                    Z_LAND,
                    spin,
                    MeshMaterial2d(m_land.add(PlanetLandmassMaterial::new(common, &spec.palette))),
                );
                if spec.clouds {
                    spawn_layer(
                        &mut commands,
                        entity,
                        body_mesh.clone(),
                        Z_CLOUDS,
                        spin,
                        MeshMaterial2d(m_clouds.add(CloudsMaterial::new(
                            common,
                            &spec.palette,
                            0.45,
                        ))),
                    );
                }
            }
            PlanetType::GasGiant => {
                if spec.ring {
                    let ring_size = spec.diameter * RING_SCALE_REL_TO_PLANET / 2.0;
                    let ring_mesh = meshes.add(Rectangle::new(ring_size, ring_size));
                    // The ring quad is larger than the body, so its UV span covers
                    // more world units. Scale the pixelation by that same ratio to
                    // keep one ring pixel cell the same world size as a body cell.
                    let ring_common = CommonParams {
                        pixels: common.pixels * ring_size / spec.diameter,
                        ..common
                    };
                    spawn_layer(
                        &mut commands,
                        entity,
                        ring_mesh,
                        Z_RING,
                        spin,
                        MeshMaterial2d(m_ring.add(RingMaterial::new(
                            ring_common,
                            &spec.palette,
                            RING_SCALE_REL_TO_PLANET,
                        ))),
                    );
                }
                spawn_layer(
                    &mut commands,
                    entity,
                    body_mesh.clone(),
                    Z_BODY,
                    spin,
                    MeshMaterial2d(m_gas.add(GasLayersMaterial::new(common, &spec.palette, 1.0))),
                );
            }
        }
    }
}

fn spawn_layer<M: Material2d>(
    commands: &mut Commands,
    parent: Entity,
    mesh: Handle<Mesh>,
    z: f32,
    spin: PlanetSpin,
    material: MeshMaterial2d<M>,
) {
    commands.spawn((
        Name::new("PixelPlanetLayer"),
        Mesh2d(mesh),
        material,
        spin,
        Transform::from_xyz(0.0, 0.0, z),
        Visibility::Visible,
        ChildOf(parent),
    ));
}

/// Advances the `time` uniform of every animated planet layer of material type
/// `M`, scrolling its surface so the planet appears to spin. Registered once per
/// layer material in [`super::PixelPlanetPlugin`].
pub(crate) fn advance_planet_spin<M: Material2d + PlanetMaterial>(
    time: Res<Time>,
    mut materials: ResMut<Assets<M>>,
    q: Query<(&MeshMaterial2d<M>, &PlanetSpin)>,
) {
    let dt = time.delta_secs();
    for (handle, spin) in &q {
        if spin.speed.abs() < f32::EPSILON {
            continue;
        }
        if let Some(mut material) = materials.get_mut(&handle.0) {
            *material.time_mut() += spin.speed * dt;
        }
    }
}
