//! Serde coverage for the `BlackHole` config types (`--features serde`):
//! a full round-trip and a partial deserialize backed by the `Default` impls.

#![cfg(feature = "serde")]

use msg_shaders::{BlackHole, BlackHoleColors, BlackHoleGeometry};

/// A config survives serialize → deserialize with every field intact.
#[test]
fn blackhole_round_trips_through_ron() {
    let original = BlackHole {
        size: 600.0,
        spin: 1.2,
        inclination: 0.9,
        colors: BlackHoleColors {
            disk_inner: [0.9, 0.8, 0.7, 1.0],
            disk_mid: [0.6, 0.5, 0.4, 1.0],
            disk_outer: [0.3, 0.2, 0.1, 1.0],
            glow: [1.0, 0.95, 0.9, 1.0],
            black: [0.01, 0.0, 0.02, 1.0],
        },
        geometry: BlackHoleGeometry {
            shadow_radius: 0.2,
            disk_inner_ratio: 1.6,
            disk_outer_ratio: 3.5,
            photon_ring_width: 0.004,
            photon_ring_intensity: 0.9,
            doppler_strength: 0.8,
            cloud_density: 1.3,
            axial_inner_ratio: 1.02,
            axial_outer_ratio: 1.3,
            secondary_brightness: 0.6,
        },
        pixel_grid: 96.0,
    };

    let text = ron::ser::to_string(&original).expect("BlackHole serializes");
    let parsed: BlackHole = ron::from_str(&text).expect("serialized BlackHole parses");

    assert_eq!(parsed.size, original.size);
    assert_eq!(parsed.spin, original.spin);
    assert_eq!(parsed.inclination, original.inclination);
    assert_eq!(parsed.pixel_grid, original.pixel_grid);
    assert_eq!(parsed.colors.disk_inner, original.colors.disk_inner);
    assert_eq!(parsed.colors.disk_mid, original.colors.disk_mid);
    assert_eq!(parsed.colors.disk_outer, original.colors.disk_outer);
    assert_eq!(parsed.colors.glow, original.colors.glow);
    assert_eq!(parsed.colors.black, original.colors.black);
    assert_eq!(
        parsed.geometry.shadow_radius,
        original.geometry.shadow_radius
    );
    assert_eq!(
        parsed.geometry.disk_inner_ratio,
        original.geometry.disk_inner_ratio
    );
    assert_eq!(
        parsed.geometry.disk_outer_ratio,
        original.geometry.disk_outer_ratio
    );
    assert_eq!(
        parsed.geometry.photon_ring_width,
        original.geometry.photon_ring_width
    );
    assert_eq!(
        parsed.geometry.photon_ring_intensity,
        original.geometry.photon_ring_intensity
    );
    assert_eq!(
        parsed.geometry.doppler_strength,
        original.geometry.doppler_strength
    );
    assert_eq!(
        parsed.geometry.cloud_density,
        original.geometry.cloud_density
    );
    assert_eq!(
        parsed.geometry.axial_inner_ratio,
        original.geometry.axial_inner_ratio
    );
    assert_eq!(
        parsed.geometry.axial_outer_ratio,
        original.geometry.axial_outer_ratio
    );
    assert_eq!(
        parsed.geometry.secondary_brightness,
        original.geometry.secondary_brightness
    );
}

/// A partial file sets only what it names; struct-level `#[serde(default)]`
/// fills the rest from the `Default` impls — including inside nested structs.
#[test]
fn partial_config_falls_back_to_defaults() {
    let parsed: BlackHole = ron::from_str(
        "(size: 250.0, geometry: (shadow_radius: 0.25), colors: (glow: (0.5, 0.5, 0.5, 1.0)))",
    )
    .expect("partial BlackHole parses");
    let default = BlackHole::default();

    // Named fields take the file's values.
    assert_eq!(parsed.size, 250.0);
    assert_eq!(parsed.geometry.shadow_radius, 0.25);
    assert_eq!(parsed.colors.glow, [0.5, 0.5, 0.5, 1.0]);

    // Everything else falls back to the Default impls.
    assert_eq!(parsed.spin, default.spin);
    assert_eq!(parsed.inclination, default.inclination);
    assert_eq!(parsed.pixel_grid, default.pixel_grid);
    assert_eq!(
        parsed.geometry.disk_outer_ratio,
        default.geometry.disk_outer_ratio
    );
    assert_eq!(
        parsed.geometry.secondary_brightness,
        default.geometry.secondary_brightness
    );
    assert_eq!(parsed.colors.disk_inner, default.colors.disk_inner);
    assert_eq!(parsed.colors.black, default.colors.black);
}

/// The empty document is exactly the `Default` configuration.
#[test]
fn empty_config_is_the_default() {
    let parsed: BlackHole = ron::from_str("()").expect("empty BlackHole parses");
    let default = BlackHole::default();
    assert_eq!(parsed.size, default.size);
    assert_eq!(parsed.spin, default.spin);
    assert_eq!(
        parsed.geometry.shadow_radius,
        default.geometry.shadow_radius
    );
    assert_eq!(parsed.colors.glow, default.colors.glow);
}
