//! Planet assembly: turns a [`PixelPlanet`] component into a stack of layered
//! `Mesh2d` children, one per shader layer (plus an optional ring).

use bevy::prelude::*;

use bevy::sprite_render::Material2d;

use super::material::{
    CloudsMaterial, CommonParams, GasLayersMaterial, LayerView, NoAtmosphereMaterial,
    PlanetLandmassMaterial, PlanetMaterial, PlanetUnderMaterial, RingMaterial,
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
#[derive(Clone, Reflect)]
pub struct PlanetSpec {
    /// Which layer stack to build.
    pub planet_type: PlanetType,
    /// Diameter of the planet body in world units.
    pub diameter: f32,
    /// Injected colour ramp.
    pub palette: PlanetPalette,
    /// Noise seed (`1.0..=10.0`).
    pub seed: f32,
    /// Roll about the view axis in radians, turning the surface and ring inside
    /// the shader while the light stays put.
    pub rotation: f32,
    /// Spin about the pole in radians per second. `0.0` is static; negative
    /// values reverse the spin direction.
    pub rotation_speed: f32,
    /// Pole tilt toward the viewer in radians. `0.0` keeps the pole in the
    /// screen plane, so a ring is seen edge-on; `±π/2` looks straight down a pole.
    pub tilt: f32,
    /// Sine latitude in `0.0..=1.0` past which polar caps cover the surface.
    /// `None` leaves the poles bare.
    pub pole_cap: Option<f32>,
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
            tilt: self.tilt,
            pole_cap: self.pole_cap,
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
#[derive(Component, Reflect, Clone)]
#[reflect(Component)]
pub struct PixelPlanet(pub PlanetSpec);

/// Turns a [`PixelPlanet`] on top of its spec: a camera moving around it, say.
///
/// Optional, on the entity carrying the `PixelPlanet`. `tilt` adds to
/// [`PlanetSpec::tilt`] and `spin` to the angle the planet has spun by, both in
/// radians. A change redraws the layers once; a planet whose orientation stays
/// put costs nothing.
#[derive(Component, Reflect, Clone, Copy, Debug, Default, PartialEq)]
#[reflect(Component, Default)]
pub struct PlanetOrientation {
    /// Pole tilt toward the viewer, added to the spec's.
    pub tilt: f32,
    /// Spin about the pole, added to the accumulated spin.
    pub spin: f32,
}

/// Moves a [`PixelPlanet`]'s highlight after spawn, replacing
/// [`PlanetSpec::light_origin`] while present: a phase that follows a day
/// cycle, say. Optional, on the entity carrying the `PixelPlanet`; the light
/// stays in screen space.
#[derive(Component, Reflect, Clone, Copy, Debug, PartialEq)]
#[reflect(Component)]
pub struct PlanetLight {
    /// Highlight location in `[0, 1]` UV space.
    pub origin: Vec2,
}

/// Per-layer view state: the spec's spin speed, base tilt and light, and the
/// angle spun so far. [`update_planet_views`] writes it, with the planet's
/// [`PlanetOrientation`] and [`PlanetLight`], into the layer material.
#[derive(Component, Reflect, Clone, Copy)]
#[reflect(Component)]
pub(crate) struct PlanetSpin {
    /// Radians per second.
    pub(crate) speed: f32,
    /// Spin so far, in radians, within `[0, 2π)`.
    pub(crate) angle: f32,
    /// [`PlanetSpec::tilt`].
    pub(crate) tilt: f32,
    /// [`PlanetSpec::light_origin`].
    pub(crate) light_origin: Vec2,
}

impl PlanetSpin {
    fn for_spec(spec: &PlanetSpec) -> Self {
        Self {
            speed: spec.rotation_speed,
            angle: 0.0,
            tilt: spec.tilt,
            light_origin: spec.light_origin,
        }
    }

    /// The layer's view under `orientation` and `light`.
    pub(crate) fn view(
        &self,
        orientation: Option<&PlanetOrientation>,
        light: Option<&PlanetLight>,
    ) -> LayerView {
        let orientation = orientation.copied().unwrap_or_default();
        LayerView {
            // A whole revolution is a whole noise period, so wrapping keeps the surface seamless.
            spin: (self.angle + orientation.spin).rem_euclid(std::f32::consts::TAU),
            tilt: self.tilt + orientation.tilt,
            light_origin: light.map_or(self.light_origin, |light| light.origin),
        }
    }
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
        let spin = PlanetSpin::for_spec(spec);
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

/// Advances the spin of every planet layer of material type `M` and writes its
/// view into the material. A layer is written only while it spins, when it is
/// new, or when its planet's [`PlanetOrientation`] or [`PlanetLight`] changed,
/// so a still planet makes no asset writes. Registered once per layer material
/// in [`super::PixelPlanetPlugin`].
pub(crate) fn update_planet_views<M: Material2d + PlanetMaterial>(
    time: Res<Time>,
    mut materials: ResMut<Assets<M>>,
    mut q_layers: Query<(&MeshMaterial2d<M>, &mut PlanetSpin, &ChildOf)>,
    q_planets: Query<(Option<Ref<PlanetOrientation>>, Option<Ref<PlanetLight>>)>,
) {
    let dt = time.delta_secs();
    for (handle, mut spin, child_of) in &mut q_layers {
        let (orientation, light) = q_planets.get(child_of.parent()).unwrap_or((None, None));
        let spinning = spin.speed != 0.0;
        let moved = spin.is_added()
            || orientation.as_ref().is_some_and(Ref::is_changed)
            || light.as_ref().is_some_and(Ref::is_changed);
        if !spinning && !moved {
            continue;
        }
        if spinning {
            spin.angle = (spin.angle + spin.speed * dt).rem_euclid(std::f32::consts::TAU);
        }
        let view = spin.view(orientation.as_deref(), light.as_deref());
        if let Some(mut material) = materials.get_mut(&handle.0) {
            material.set_view(view);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use bevy::time::TimeUpdateStrategy;

    use super::*;

    /// Counts the frames the view system wrote `Assets<NoAtmosphereMaterial>`.
    #[derive(Resource, Default)]
    struct Writes(u32);

    fn count_writes(materials: Res<Assets<NoAtmosphereMaterial>>, mut writes: ResMut<Writes>) {
        if materials.is_changed() {
            writes.0 += 1;
        }
    }

    struct ViewTestPlugin;

    impl Plugin for ViewTestPlugin {
        fn build(&self, app: &mut App) {
            app.add_plugins(MinimalPlugins)
                .insert_resource(TimeUpdateStrategy::ManualDuration(Duration::from_millis(
                    100,
                )))
                .init_resource::<Assets<NoAtmosphereMaterial>>()
                .init_resource::<Writes>()
                .add_systems(
                    Update,
                    (update_planet_views::<NoAtmosphereMaterial>, count_writes).chain(),
                );
        }
    }

    fn spec(rotation_speed: f32) -> PlanetSpec {
        PlanetSpec {
            planet_type: PlanetType::Barren,
            diameter: 10.0,
            palette: PlanetPalette::new([Color::WHITE]),
            seed: 1.0,
            rotation: 0.0,
            rotation_speed,
            tilt: 0.25,
            pole_cap: None,
            light_origin: Vec2::splat(0.4),
            clouds: false,
            ring: false,
            dither: None,
            pixels: 10.0,
            octaves: 1,
            noise_size: 4.0,
        }
    }

    /// A planet entity with one barren layer; returns the planet and the layer's material.
    fn spawn_planet(app: &mut App, spec: &PlanetSpec) -> (Entity, Handle<NoAtmosphereMaterial>) {
        let material = app
            .world_mut()
            .resource_mut::<Assets<NoAtmosphereMaterial>>()
            .add(NoAtmosphereMaterial::new(spec.common(), &spec.palette));
        let planet = app.world_mut().spawn_empty().id();
        app.world_mut().spawn((
            MeshMaterial2d(material.clone()),
            PlanetSpin::for_spec(spec),
            ChildOf(planet),
        ));
        (planet, material)
    }

    fn uniform(
        app: &App,
        material: &Handle<NoAtmosphereMaterial>,
    ) -> super::super::material::NoAtmosphereUniform {
        app.world()
            .resource::<Assets<NoAtmosphereMaterial>>()
            .get(material)
            .expect("material")
            .u
    }

    fn writes(app: &App) -> u32 {
        app.world().resource::<Writes>().0
    }

    #[test]
    fn a_still_planet_is_written_once_and_then_left_alone() {
        let mut app = App::new();
        app.add_plugins(ViewTestPlugin);
        spawn_planet(&mut app, &spec(0.0));
        app.update();
        let first = writes(&app);
        for _ in 0..5 {
            app.update();
        }
        assert_eq!(writes(&app), first, "no writes after the first view");
    }

    #[test]
    fn an_orientation_change_rewrites_a_still_planet_once() {
        let mut app = App::new();
        app.add_plugins(ViewTestPlugin);
        let spec = spec(0.0);
        let (planet, material) = spawn_planet(&mut app, &spec);
        app.update();
        app.update();
        let before = writes(&app);

        let orientation = PlanetOrientation {
            tilt: 0.5,
            spin: 1.0,
        };
        app.world_mut().entity_mut(planet).insert(orientation);
        app.update();
        app.update();
        assert_eq!(writes(&app), before + 1);

        let u = uniform(&app, &material);
        assert_eq!(u.tilt_cs, Vec2::from_angle(spec.tilt + orientation.tilt));
        assert_eq!(u.time, orientation.spin);
    }

    #[test]
    fn a_planet_light_replaces_the_spec_highlight() {
        let mut app = App::new();
        app.add_plugins(ViewTestPlugin);
        let (planet, material) = spawn_planet(&mut app, &spec(0.0));
        app.update();
        let light = PlanetLight {
            origin: Vec2::new(0.9, 0.1),
        };
        app.world_mut().entity_mut(planet).insert(light);
        app.update();
        assert_eq!(uniform(&app, &material).light_origin, light.origin);
    }

    #[test]
    fn a_spinning_planet_advances_by_its_speed_and_wraps_a_revolution() {
        let mut app = App::new();
        app.add_plugins(ViewTestPlugin);
        let speed = 2.0;
        let (_, material) = spawn_planet(&mut app, &spec(speed));
        let mut previous: Option<f32> = None;
        for _ in 0..40 {
            app.update();
            let time = uniform(&app, &material).time;
            assert!((0.0..std::f32::consts::TAU).contains(&time), "{time}");
            if let Some(previous) = previous {
                let step = (time - previous).rem_euclid(std::f32::consts::TAU);
                let dt = app.world().resource::<Time>().delta_secs();
                assert!((step - speed * dt).abs() < 1e-4, "{step}");
            }
            previous = Some(time);
        }
    }
}
