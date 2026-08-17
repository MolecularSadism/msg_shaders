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
    srgb
        .iter()
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
            nebula_scale: 0.010,
            star_spacing: 8.0,
            star_density: 0.6,
            // Fix the art-pixel size so the demo is chunky without a
            // pixel-perfect game camera driving the zoom.
            pixel_size: 3.0,
            ..default()
        },
        Transform::default(),
    ));
}

/// Slowly rotate and drift the sky so the twinkle and motion are visible.
fn drift(time: Res<Time>, mut q: Query<&mut Nebula>) {
    let t = time.elapsed_secs();
    for mut nebula in &mut q {
        nebula.rotation = t * 0.02;
        nebula.scroll = Vec2::new(t * 6.0, t * 2.0);
    }
}
