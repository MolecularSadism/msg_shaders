//! Procedural nebula + twinkling starfield on the default 2D camera.
//!
//! The clouds are drawn with ordered dithering; each star holds one swatch from
//! its own layer's palette, and the fraction set by `twinkle_chance` blinks. The
//! whole sky slowly rotates and drifts to show the day/night and parallax
//! controls.
//!
//! Every color on screen is one authored here: each layer resolves its own
//! density to one of its own three stops, so recoloring a layer never moves
//! another layer's edges.
//!
//! Run from this crate's directory: `cargo run --example nebula`

use bevy::prelude::*;
use msg_shaders::prelude::*;
use smallvec::smallvec;

fn main() {
    App::new()
        .add_plugins((DefaultPlugins, NebulaPlugin))
        .add_systems(Startup, setup)
        .add_systems(Update, drift)
        .run();
}

/// One retro-space swatch (sRGB hex) as linear RGBA.
fn swatch(r: u8, g: u8, b: u8) -> [f32; 4] {
    let lin = Color::srgb_u8(r, g, b).to_linear();
    [lin.red, lin.green, lin.blue, 1.0]
}

fn setup(mut commands: Commands) {
    commands.spawn(Camera2d);
    commands.spawn((
        Nebula {
            size: Vec2::splat(1400.0),
            background: swatch(0x0d, 0x0a, 0x18),
            // Two cloud layers, each painting only its own three stops.
            layers: vec![
                NebulaLayer {
                    colors: [
                        swatch(0x21, 0x14, 0x3a),
                        swatch(0x5a, 0x21, 0x9c),
                        swatch(0x80, 0x61, 0xe6),
                    ],
                    scale: 0.006,
                    intensity: 0.9,
                    parallax: 0.10,
                    ..default()
                },
                NebulaLayer {
                    colors: [
                        swatch(0x1b, 0x2e, 0x5c),
                        swatch(0x00, 0x5d, 0xff),
                        swatch(0x00, 0xdc, 0xff),
                    ],
                    scale: 0.016,
                    cloud_low: 0.5,
                    intensity: 0.85,
                    parallax: 0.2,
                    offset: Vec2::new(400.0, -250.0),
                    ..default()
                },
            ],
            stars: vec![
                // Near layer: the bright end of the palette.
                StarLayer {
                    spacing: 8.0,
                    colors: smallvec![
                        swatch(0xc1, 0xff, 0xe2),
                        swatch(0xff, 0xff, 0xff),
                        swatch(0xff, 0xbf, 0x00),
                    ],
                    parallax: 0.14,
                    ..default()
                },
                // Far layer: dimmer swatches, which is what reads as distance.
                StarLayer {
                    spacing: 16.0,
                    density: 0.4,
                    colors: smallvec![swatch(0x6d, 0x8f, 0x9c), swatch(0x8f, 0x7a, 0x5c)],
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
