//! Serde coverage for the `BlackHole` and `HoleQuantization` config types
//! (`--features serde`): full round-trips, partial deserializes backed by the
//! `Default` impls, and rejection of misspelled field names.

#![cfg(feature = "serde")]

use msg_shaders::{BlackHole, BlackHoleColors, BlackHoleGeometry, HoleQuantization};

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

    assert_eq!(parsed, original);
}

/// A partial file sets only what it names; struct-level `#[serde(default)]`
/// fills the rest from the `Default` impls — including inside nested structs.
#[test]
fn partial_config_falls_back_to_defaults() {
    let parsed: BlackHole = ron::from_str(
        "(size: 250.0, geometry: (shadow_radius: 0.25), colors: (glow: (0.5, 0.5, 0.5, 1.0)))",
    )
    .expect("partial BlackHole parses");

    let expected = BlackHole {
        size: 250.0,
        geometry: BlackHoleGeometry {
            shadow_radius: 0.25,
            ..BlackHoleGeometry::default()
        },
        colors: BlackHoleColors {
            glow: [0.5, 0.5, 0.5, 1.0],
            ..BlackHoleColors::default()
        },
        ..BlackHole::default()
    };
    assert_eq!(parsed, expected);
}

/// The empty document is exactly the `Default` configuration.
#[test]
fn empty_config_is_the_default() {
    let parsed: BlackHole = ron::from_str("()").expect("empty BlackHole parses");
    assert_eq!(parsed, BlackHole::default());
}

/// A misspelled field name is a parse error, not a silently ignored entry —
/// `deny_unknown_fields` at the top level and inside the nested structs.
#[test]
fn misspelled_field_is_rejected() {
    let top_level = ron::from_str::<BlackHole>("(spinn: 2.0)");
    assert!(top_level.is_err(), "typo'd top-level field must not parse");

    let nested = ron::from_str::<BlackHole>("(geometry: (shadow_radiu: 0.25))");
    assert!(nested.is_err(), "typo'd nested field must not parse");

    let quantization = ron::from_str::<HoleQuantization>("(alpha_cutof: 0.1)");
    assert!(
        quantization.is_err(),
        "typo'd HoleQuantization field must not parse"
    );
}

/// A palette config survives serialize → deserialize with every field intact.
#[test]
fn quantization_round_trips_through_ron() {
    let original = HoleQuantization {
        palette: vec![
            [0.1, 0.2, 0.3, 1.0],
            [0.4, 0.5, 0.6, 1.0],
            [0.7, 0.8, 0.9, 1.0],
        ],
        alpha_cutoff: 0.05,
        dither_pattern: 2,
        transparency_floor: 0.1,
    };

    let text = ron::ser::to_string(&original).expect("HoleQuantization serializes");
    let parsed: HoleQuantization =
        ron::from_str(&text).expect("serialized HoleQuantization parses");

    assert_eq!(parsed, original);
}

/// A partial palette file keeps the curated defaults — the same values
/// [`HoleQuantization::new`] applies — for the fields it does not name.
#[test]
fn partial_quantization_falls_back_to_curated_defaults() {
    let parsed: HoleQuantization = ron::from_str("(palette: [(1.0, 0.5, 0.0, 1.0)])")
        .expect("partial HoleQuantization parses");
    assert_eq!(parsed, HoleQuantization::new(vec![[1.0, 0.5, 0.0, 1.0]]));

    let empty: HoleQuantization = ron::from_str("()").expect("empty HoleQuantization parses");
    assert_eq!(empty, HoleQuantization::default());
    assert_eq!(empty, HoleQuantization::new(Vec::new()));
}
