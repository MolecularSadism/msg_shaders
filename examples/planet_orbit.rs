//! Turns pixel planets on all three axes, as a camera flying around them would see them.
//!
//! - Up / Down: tilt the pole toward or away from the viewer (passing over a pole).
//! - Left / Right: spin about the pole (circling the equator).
//! - Q / E: move the light around the planets.
//! - Z / X: roll the whole picture, light included.
//! - Space: pause or resume the planets' own spin.
//!
//! Every planet carries the same `PlanetOrientation` and `PlanetLight`, written only when a key
//! changes them.
//!
//! With `PLANET_ORBIT_SHOTS=<dir>` the example instead steps through a fixed set of views (tilt 0, a
//! small tilt, mid tilt, over the pole, past the pole), saves one screenshot of each into `<dir>` and exits.
//!
//! Run from this crate's directory: `cargo run --example planet_orbit`

use std::f32::consts::{FRAC_PI_2, PI};
use std::path::PathBuf;

use bevy::prelude::*;
use bevy::render::view::screenshot::{Screenshot, save_to_disk};
use msg_shaders::prelude::*;

/// Radians a second the arrow keys turn by.
const TURN_SPEED: f32 = 1.0;
/// Distance of the highlight from the disc centre, in UV units.
const LIGHT_REACH: f32 = 0.35;
/// The views `PLANET_ORBIT_SHOTS` captures: name, tilt and light angle.
const SHOTS: [(&str, f32, f32); 5] = [
    ("tilt_0", 0.0, 2.4),
    ("tilt_small", 0.08, 2.4),
    ("tilt_mid", 0.6, 2.4),
    ("tilt_pole", FRAC_PI_2, 2.4),
    ("tilt_past_pole", 2.3, 2.4),
];
/// Frames each shot waits for its view to be drawn.
const SHOT_FRAMES: u32 = 30;
/// Frames before the first shot, while the pipelines compile.
const WARM_UP_FRAMES: u32 = 300;

fn main() {
    let shots = std::env::var_os("PLANET_ORBIT_SHOTS").map(PathBuf::from);
    let mut app = App::new();
    app.add_plugins((DefaultPlugins, PixelPlanetPlugin))
        .insert_resource(ClearColor(Color::srgb(0.03, 0.02, 0.06)))
        .insert_resource(View {
            light_angle: 2.4,
            ..default()
        })
        .add_systems(Startup, setup)
        .add_systems(Update, (apply_view, show_view));
    match shots {
        Some(dir) => {
            app.insert_resource(ShotPlan { dir, frame: 0 })
                .add_systems(Update, take_shots.before(apply_view));
        }
        None => {
            app.add_systems(Update, steer.before(apply_view));
        }
    }
    app.run();
}

/// What the keys control.
#[derive(Resource, Default)]
struct View {
    tilt: f32,
    spin: f32,
    light_angle: f32,
    roll: f32,
    paused: bool,
}

impl View {
    /// The highlight in UV space (y down) for the light angle.
    fn light_origin(&self) -> Vec2 {
        let dir = Vec2::from_angle(self.light_angle);
        Vec2::splat(0.5) + Vec2::new(dir.x, -dir.y) * LIGHT_REACH
    }
}

#[derive(Component)]
struct Planet;

#[derive(Component)]
struct Readout;

#[derive(Resource)]
struct ShotPlan {
    dir: PathBuf,
    frame: u32,
}

fn spec(planet_type: PlanetType, diameter: f32, palette: PlanetPalette, seed: f32) -> PlanetSpec {
    PlanetSpec {
        planet_type,
        diameter,
        palette,
        seed,
        rotation: 0.0,
        rotation_speed: 0.15,
        tilt: 0.0,
        pole_cap: Some(0.82),
        light_origin: Vec2::splat(0.5),
        clouds: true,
        ring: true,
        dither: Some(1.5),
        pixels: diameter / 2.0,
        octaves: 5,
        noise_size: match planet_type {
            PlanetType::GasGiant => 8.4,
            PlanetType::Terran => 6.0,
            PlanetType::Barren => 4.8,
        },
    }
}

fn setup(mut commands: Commands) {
    commands.spawn(Camera2d);
    let planets = [
        (
            spec(
                PlanetType::Terran,
                200.0,
                PlanetPalette::new([
                    Color::srgb(0.55, 0.85, 0.6),
                    Color::srgb(0.25, 0.6, 0.35),
                    Color::srgb(0.15, 0.35, 0.55),
                    Color::srgb(0.05, 0.1, 0.25),
                ]),
                3.0,
            ),
            Vec2::new(-330.0, 140.0),
        ),
        (
            spec(
                PlanetType::Barren,
                160.0,
                PlanetPalette::new([
                    Color::srgb(0.75, 0.7, 0.65),
                    Color::srgb(0.45, 0.4, 0.38),
                    Color::srgb(0.18, 0.16, 0.17),
                ]),
                6.0,
            ),
            Vec2::new(-330.0, -170.0),
        ),
        (
            spec(
                PlanetType::GasGiant,
                240.0,
                PlanetPalette::new([
                    Color::srgb(0.95, 0.8, 0.55),
                    Color::srgb(0.8, 0.5, 0.3),
                    Color::srgb(0.45, 0.2, 0.15),
                ]),
                4.0,
            ),
            Vec2::new(250.0, 0.0),
        ),
    ];
    for (spec, at) in planets {
        commands.spawn((
            Planet,
            PixelPlanet(spec),
            PlanetOrientation::default(),
            PlanetLight {
                origin: Vec2::splat(0.5),
            },
            Transform::from_translation(at.extend(0.0)),
            Visibility::Visible,
        ));
    }
    commands.spawn((
        Readout,
        Text::new(""),
        Node {
            position_type: PositionType::Absolute,
            left: Val::Px(12.0),
            top: Val::Px(12.0),
            ..default()
        },
    ));
}

fn steer(keys: Res<ButtonInput<KeyCode>>, time: Res<Time>, mut view: ResMut<View>) {
    let step = TURN_SPEED * time.delta_secs();
    let axis = |minus: KeyCode, plus: KeyCode| {
        f32::from(u8::from(keys.pressed(plus))) - f32::from(u8::from(keys.pressed(minus)))
    };
    let tilt = axis(KeyCode::ArrowDown, KeyCode::ArrowUp);
    let spin = axis(KeyCode::ArrowLeft, KeyCode::ArrowRight);
    let light = axis(KeyCode::KeyE, KeyCode::KeyQ);
    let roll = axis(KeyCode::KeyX, KeyCode::KeyZ);
    if tilt != 0.0 {
        view.tilt = (view.tilt + tilt * step + PI).rem_euclid(2.0 * PI) - PI;
    }
    if spin != 0.0 {
        view.spin += spin * step;
    }
    if light != 0.0 {
        view.light_angle += light * step;
    }
    if roll != 0.0 {
        view.roll += roll * step;
    }
    if keys.just_pressed(KeyCode::Space) {
        view.paused = !view.paused;
    }
}

/// Writes the view into the planets only when it changed.
fn apply_view(
    view: Res<View>,
    mut spin_set: Local<Option<bool>>,
    mut q_planets: Query<(&mut PlanetOrientation, &mut PlanetLight, &mut Transform), With<Planet>>,
    mut time: ResMut<Time<Virtual>>,
) {
    if *spin_set != Some(view.paused) {
        *spin_set = Some(view.paused);
        if view.paused {
            time.pause();
        } else {
            time.unpause();
        }
    }
    if !view.is_changed() {
        return;
    }
    let orientation = PlanetOrientation {
        tilt: view.tilt,
        spin: view.spin,
    };
    let light = view.light_origin();
    for (mut planet_orientation, mut planet_light, mut transform) in &mut q_planets {
        planet_orientation.set_if_neq(orientation);
        planet_light.set_if_neq(PlanetLight { origin: light });
        transform.rotation = Quat::from_rotation_z(view.roll);
    }
}

fn show_view(view: Res<View>, mut readout: Single<&mut Text, With<Readout>>) {
    if view.is_changed() {
        readout.0 = format!(
            "tilt {:+.2} rad   spin {:+.2} rad   light {:.2} rad   roll {:+.2} rad{}\n\
             Up/Down tilt  Left/Right spin  Q/E light  Z/X roll  Space pause",
            view.tilt,
            view.spin,
            view.light_angle,
            view.roll,
            if view.paused { "   (paused)" } else { "" },
        );
    }
}

/// Steps through [`SHOTS`], saving each after [`SHOT_FRAMES`] frames, then exits.
fn take_shots(
    mut commands: Commands,
    mut plan: ResMut<ShotPlan>,
    mut view: ResMut<View>,
    mut exit: MessageWriter<AppExit>,
) {
    // The first shot also waits for every pipeline to compile.
    let Some(frame) = plan.frame.checked_sub(WARM_UP_FRAMES) else {
        plan.frame += 1;
        return;
    };
    let shot = (frame / SHOT_FRAMES) as usize;
    let frame = frame % SHOT_FRAMES;
    plan.frame += 1;
    let Some((name, tilt, light_angle)) = SHOTS.get(shot) else {
        // One more shot's worth of frames lets the last save finish.
        if frame == SHOT_FRAMES - 1 {
            exit.write(AppExit::Success);
        }
        return;
    };
    if frame == 0 {
        view.paused = true;
        view.tilt = *tilt;
        view.light_angle = *light_angle;
    } else if frame == SHOT_FRAMES - 1 {
        let path = plan.dir.join(format!("{name}.png"));
        commands
            .spawn(Screenshot::primary_window())
            .observe(save_to_disk(path));
    }
}
