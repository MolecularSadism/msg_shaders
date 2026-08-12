//! Screen-space gravitational lensing over a simple tile grid.
//!
//! A black hole orbits the center of the scene: its `LightDeflector` warps
//! everything the camera has rendered, and its `BlackHoleOverlay` draws the
//! photon ring and event horizon on top of the warp.
//!
//! Run from this crate's directory: `cargo run --example lensing`

use bevy::prelude::*;
use msg_shaders::prelude::*;

fn main() {
    App::new()
        .add_plugins((DefaultPlugins, LensingHolePlugin))
        .add_systems(Startup, setup)
        .add_systems(Update, orbit)
        .run();
}

/// Circular path around the world origin, driven by [`orbit`].
#[derive(Component)]
struct Orbit {
    radius: f32,
    /// Angular speed in radians per second.
    speed: f32,
}

/// Visible halo radius of the hole, in world units.
const HOLE_SIZE: f32 = 150.0;

fn setup(mut commands: Commands) {
    // The lensing display pass warps the view of the camera marked with
    // `LensingHoleCamera`.
    commands.spawn((Camera2d, LensingHoleCamera));

    // A colorful tile grid so the deflection is visible.
    for x in -10i32..=10 {
        for y in -6i32..=6 {
            let hue = ((x * 31 + y * 17).rem_euclid(36) * 10) as f32;
            commands.spawn((
                Sprite::from_color(Color::hsl(hue, 0.55, 0.45), Vec2::splat(56.0)),
                Transform::from_xyz(x as f32 * 64.0, y as f32 * 64.0, 0.0),
            ));
        }
    }

    // The hole itself: the `LightDeflector` bends the rendered scene around
    // the entity, and the `BlackHoleOverlay` draws the photon ring and event
    // horizon at the same spot. Each component also works on its own.
    commands.spawn((
        LightDeflector {
            shape: DeflectionShape::Lens {
                center: Vec2::ZERO,
                size: HOLE_SIZE,
                inner_radius: 0.15,
            },
            strength: 0.6,
        },
        BlackHoleOverlay {
            size: HOLE_SIZE,
            ..default()
        },
        Transform::default(),
        Orbit {
            radius: 170.0,
            speed: 0.4,
        },
    ));
}

fn orbit(time: Res<Time>, mut holes: Query<(&Orbit, &mut Transform)>) {
    for (params, mut transform) in &mut holes {
        let angle = time.elapsed_secs() * params.speed;
        transform.translation.x = angle.cos() * params.radius;
        transform.translation.y = angle.sin() * params.radius;
    }
}
