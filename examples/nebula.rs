//! Procedural nebula + twinkling starfield on the default 2D camera.
//!
//! The clouds are drawn in a retro palette with ordered dithering; the stars
//! pulse in brightness and shift color over time. The whole sky slowly rotates
//! and drifts to show the day/night and parallax controls.
//!
//! Run from this crate's directory: `cargo run --example nebula`

use bevy::prelude::*;
use msg_shaders::prelude::*;

fn main() {
    App::new()
        .add_plugins((DefaultPlugins, NebulaPlugin))
        .add_systems(Startup, setup)
        .add_systems(Update, drift)
        .run();
}

/// A small retro space palette (linear RGB) the nebula is quantized to.
fn palette() -> Vec<[f32; 4]> {
    // Deep-space darks, nebula purples/blues, and bright star tips.
    let srgb = [
        [0x0d, 0x0a, 0x18],
        [0x21, 0x14, 0x3a],
        [0x1b, 0x2e, 0x5c],
        [0x5a, 0x21, 0x9c],
        [0x00, 0x27, 0x7f],
        [0x00, 0x5d, 0xff],
        [0x80, 0x61, 0xe6],
        [0x00, 0xac, 0xff],
        [0x00, 0xdc, 0xff],
        [0xff, 0xbf, 0x00],
        [0xc1, 0xff, 0xe2],
        [0xff, 0xfc, 0xfc],
    ];
    srgb.iter()
        .map(|[r, g, b]| {
            let lin = Color::srgb_u8(*r, *g, *b).to_linear();
            [lin.red, lin.green, lin.blue, 1.0]
        })
        .collect()
}

fn setup(mut commands: Commands) {
    commands.spawn(Camera2d);
    commands.spawn((
        Nebula {
            size: Vec2::splat(1400.0),
            palette: palette(),
            background: [0.03, 0.02, 0.08, 1.0],
            // Two cloud layers pulling different color ramps, composited over.
            layers: vec![
                NebulaLayer {
                    colors: [
                        [0.10, 0.05, 0.25, 1.0],
                        [0.35, 0.12, 0.55, 1.0],
                        [0.55, 0.75, 1.00, 1.0],
                    ],
                    scale: 0.006,
                    intensity: 0.9,
                    parallax: 0.10,
                    ..default()
                },
                NebulaLayer {
                    colors: [
                        [0.06, 0.14, 0.30, 1.0],
                        [0.12, 0.45, 0.55, 1.0],
                        [1.00, 0.85, 0.50, 1.0],
                    ],
                    scale: 0.016,
                    cloud_low: 0.5,
                    intensity: 0.6,
                    parallax: 0.2,
                    offset: Vec2::new(400.0, -250.0),
                    ..default()
                },
            ],
            stars: vec![
                StarLayer {
                    spacing: 8.0,
                    parallax: 0.14,
                    ..default()
                },
                StarLayer {
                    spacing: 16.0,
                    density: 0.4,
                    brightness: 1.3,
                    parallax: 0.24,
                    seed: 5.0,
                },
            ],
            // Fix the art-pixel size so the demo is chunky without a
            // pixel-perfect game camera driving the zoom.
            pixel_size: 3.0,
            ..default()
        },
        Transform::default(),
    ));
}

/// Slowly rotate and drift the sky so the twinkle and parallax are visible.
fn drift(time: Res<Time>, mut q: Query<&mut Nebula>) {
    let t = time.elapsed_secs();
    for mut nebula in &mut q {
        nebula.rotation = t * 0.02;
        nebula.scroll = Vec2::new(t * 30.0, t * 12.0);
    }
}
