//! Static validation of the lensing-field compute shaders.
//!
//! Both compute shaders are pure WGSL — no Bevy `#import` preprocessor
//! directives — so naga can parse and validate them offline. This catches
//! struct-layout, shape-`switch`, and type errors (the kind that would
//! otherwise only surface as a pipeline-compile error at runtime) without a GPU.

use naga::valid::{Capabilities, ValidationFlags, Validator};

fn validate(label: &str, src: &str) {
    let module = naga::front::wgsl::parse_str(src)
        .unwrap_or_else(|e| panic!("{label}: WGSL parse error\n{}", e.emit_to_string(src)));
    Validator::new(ValidationFlags::all(), Capabilities::all())
        .validate(&module)
        .unwrap_or_else(|e| panic!("{label}: WGSL validation error: {e:?}"));
}

#[test]
fn lensing_field_inject_is_valid_wgsl() {
    validate(
        "lensing_field_inject",
        include_str!("../src/shaders/lensing_field_inject.wgsl"),
    );
}

#[test]
fn lensing_field_advect_is_valid_wgsl() {
    validate(
        "lensing_field_advect",
        include_str!("../src/shaders/lensing_field_advect.wgsl"),
    );
}

#[test]
fn nebula_functions_is_valid_wgsl() {
    // The function library body is pure WGSL, so naga can validate the noise,
    // nebula, and star math offline once the Bevy `#define_import_path`
    // directive is stripped. The material shader that imports it is validated by
    // the pipeline at runtime, like the other `#import`-based materials.
    let src: String = include_str!("../src/shaders/nebula_functions.wgsl")
        .lines()
        .filter(|line| !line.trim_start().starts_with('#'))
        .collect::<Vec<_>>()
        .join("\n");
    validate("nebula_functions", &src);
}

#[test]
fn blackhole_functions_is_valid_wgsl() {
    // The black hole module imports the quantization library; inline it with its
    // `cq::` prefix dropped and strip the Bevy preprocessor directives so naga
    // can validate the pair offline.
    let strip = |src: &str| -> String {
        src.lines()
            .filter(|line| !line.trim_start().starts_with('#'))
            .collect::<Vec<_>>()
            .join("\n")
    };
    let src = format!(
        "{}\n{}",
        strip(include_str!("../src/shaders/color_quantize_functions.wgsl")),
        strip(include_str!("../src/shaders/blackhole_functions.wgsl")).replace("cq::", ""),
    );
    validate("blackhole_functions", &src);
}

/// Minimal stand-in for `bevy_sprite::mesh2d_vertex_output::VertexOutput`,
/// providing the `uv` field the planet fragments read.
const PLANET_VERTEX_OUTPUT_STUB: &str = "\
struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec4<f32>,
    @location(1) uv: vec2<f32>,
};
";

/// A planet layer shader with the shared library inlined and the naga-oil
/// `#import` / `#define_import_path` directives stripped, as plain WGSL.
fn compose_planet_layer(shader: &str) -> String {
    let common = include_str!("../src/shaders/pixel_planet_functions.wgsl");
    let mut out = String::from(PLANET_VERTEX_OUTPUT_STUB);
    for source in [common, shader] {
        for line in source.lines() {
            let trimmed = line.trim_start();
            if trimmed.starts_with("#import") || trimmed.starts_with("#define_import_path") {
                continue;
            }
            out.push_str(line);
            out.push('\n');
        }
    }
    out
}

#[test]
fn pixel_planet_shaders_are_valid_wgsl() {
    for (label, shader) in [
        (
            "pixel_planet_no_atmosphere",
            include_str!("../src/shaders/pixel_planet_no_atmosphere.wgsl"),
        ),
        (
            "pixel_planet_under",
            include_str!("../src/shaders/pixel_planet_under.wgsl"),
        ),
        (
            "pixel_planet_landmass",
            include_str!("../src/shaders/pixel_planet_landmass.wgsl"),
        ),
        (
            "pixel_planet_clouds",
            include_str!("../src/shaders/pixel_planet_clouds.wgsl"),
        ),
        (
            "pixel_planet_gas_layers",
            include_str!("../src/shaders/pixel_planet_gas_layers.wgsl"),
        ),
        (
            "pixel_planet_ring",
            include_str!("../src/shaders/pixel_planet_ring.wgsl"),
        ),
    ] {
        validate(label, &compose_planet_layer(shader));
    }
}
