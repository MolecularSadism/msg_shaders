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
