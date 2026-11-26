//! CPU mirror of the pixel-planet sphere mapping in `src/shaders/pixel_planet_functions.wgsl`.
//!
//! `sphere_lonlat`, `lonlat_noise` and `noise_period` below restate the WGSL functions of the
//! same names line for line; they cover the mapping only, not the noise built on it.

use std::f32::consts::{FRAC_PI_2, PI, TAU};

use bevy::math::Vec2;

fn sphere_lonlat(uv: Vec2, tilt_cs: Vec2) -> Vec2 {
    let p = uv * 2.0 - 1.0;
    let z = (1.0 - p.dot(p)).max(0.0).sqrt();
    let y = p.y * tilt_cs.x - z * tilt_cs.y;
    let zt = p.y * tilt_cs.y + z * tilt_cs.x;
    Vec2::new(p.x.atan2(zt), y)
}

fn noise_period(size: f32) -> f32 {
    size.round().max(1.0)
}

fn lonlat_noise(ll: Vec2, spin: f32, size: f32, tile: Vec2) -> Vec2 {
    let x = (ll.x + spin) / TAU * tile.x * noise_period(size);
    Vec2::new(x, (ll.y * 0.5 + 0.5) * size)
}

/// The `gmod2(coord, tile * noise_period(size))` wrap `rand_v` applies before hashing.
fn wrapped(coord: Vec2, size: f32, tile: Vec2) -> Vec2 {
    let period = tile * noise_period(size);
    coord - period * (coord / period).floor()
}

/// Disc points on a grid, inside the disc.
fn disc_points() -> impl Iterator<Item = Vec2> {
    (0..=40)
        .flat_map(|i| (0..=40).map(move |j| Vec2::new(i as f32, j as f32) / 40.0))
        .filter(|uv| (*uv - 0.5).length() < 0.499)
}

#[test]
fn untilted_the_lift_is_the_plain_hemisphere() {
    for uv in disc_points() {
        let p = uv * 2.0 - 1.0;
        let z = (1.0 - p.dot(p)).sqrt();
        let ll = sphere_lonlat(uv, Vec2::from_angle(0.0));
        assert!((ll.y - p.y).abs() < 1e-6, "{uv}");
        assert!((ll.x - p.x.atan2(z)).abs() < 1e-6, "{uv}");
        assert!(
            ll.x.abs() <= FRAC_PI_2 + 1e-6,
            "front hemisphere only at {uv}"
        );
    }
}

#[test]
fn tilted_a_quarter_turn_the_disc_centre_is_a_pole() {
    let ll = sphere_lonlat(Vec2::splat(0.5), Vec2::from_angle(FRAC_PI_2));
    assert!((ll.y.abs() - 1.0).abs() < 1e-6, "{ll}");
}

#[test]
fn every_tilt_keeps_sine_latitude_on_the_unit_sphere() {
    for step in 0..=24 {
        let tilt_cs = Vec2::from_angle(step as f32 / 24.0 * TAU);
        for uv in disc_points() {
            let ll = sphere_lonlat(uv, tilt_cs);
            assert!(ll.y.abs() <= 1.0 + 1e-5);
            assert!(ll.x.abs() <= PI + 1e-6);
        }
    }
}

#[test]
fn longitude_is_seamless_across_the_back_for_every_tile_span() {
    for tile in [Vec2::new(2.0, 1.0), Vec2::new(1.0, 1.0)] {
        for size in [0.3_f32, 1.8, 2.0, 4.8, 6.0, 8.4, 11.5] {
            for sin_lat in [-0.9_f32, 0.0, 0.6] {
                for spin in [0.0_f32, 1.3, -5.0] {
                    let east = lonlat_noise(Vec2::new(PI, sin_lat), spin, size, tile);
                    let west = lonlat_noise(Vec2::new(-PI, sin_lat), spin, size, tile);
                    assert!(
                        wrapped(east, size, tile).abs_diff_eq(wrapped(west, size, tile), 1e-3),
                        "seam at tile {tile}, size {size}: {east} vs {west}"
                    );
                }
            }
        }
    }
}

#[test]
fn one_revolution_of_spin_returns_the_same_noise_cell() {
    let tile = Vec2::new(2.0, 1.0);
    for size in [2.0_f32, 6.0, 8.4] {
        let ll = Vec2::new(0.7, 0.2);
        let start = lonlat_noise(ll, 0.4, size, tile);
        let turned = lonlat_noise(ll, 0.4 + TAU, size, tile);
        assert!(
            wrapped(start, size, tile).abs_diff_eq(wrapped(turned, size, tile), 1e-3),
            "size {size}"
        );
    }
}
