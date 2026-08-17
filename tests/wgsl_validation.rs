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
