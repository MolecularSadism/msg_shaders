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

/// Offline stand-in for the naga-oil preprocessing Bevy applies to the planet
/// shaders: inlines the libraries a layer imports, drops the `#import` /
/// `#define_import_path` directives and resolves `#ifdef` / `#else` / `#endif`
/// against a set of shader defs, leaving plain WGSL naga can validate.
struct PlanetShaderComposer {
    /// Shader defs that are set.
    defs: &'static [&'static str],
}

impl PlanetShaderComposer {
    /// The pipelines of `PixelGrid::Quad`: no shader def.
    const QUAD: Self = Self { defs: &[] };
    /// The pipelines of `PixelGrid::World`.
    const WORLD: Self = Self {
        defs: &["PIXEL_GRID_WORLD"],
    };

    /// Minimal stand-in for `bevy_sprite::mesh2d_vertex_output::VertexOutput`,
    /// providing the fields the planet fragments read.
    const VERTEX_OUTPUT_STUB: &str = "\
struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec4<f32>,
    @location(2) uv: vec2<f32>,
};
";

    /// Stand-ins for the `bevy_sprite::mesh2d_functions` the world-grid vertex
    /// stage calls (through the `mesh_functions::` alias, stripped on compose).
    const MESH_FUNCTIONS_STUB: &str = "\
fn get_world_from_local(instance_index: u32) -> mat4x4<f32> {
    let s = f32(instance_index);
    return mat4x4<f32>(
        vec4<f32>(s, 0.0, 0.0, 0.0),
        vec4<f32>(0.0, s, 0.0, 0.0),
        vec4<f32>(0.0, 0.0, 1.0, 0.0),
        vec4<f32>(0.0, 0.0, 0.0, 1.0),
    );
}
fn mesh2d_position_local_to_world(world_from_local: mat4x4<f32>, p: vec4<f32>) -> vec4<f32> {
    return world_from_local * p;
}
fn mesh2d_position_world_to_clip(world_position: vec4<f32>) -> vec4<f32> {
    return world_position;
}
";

    /// `source` with the directives resolved against `self.defs`.
    fn preprocess(&self, source: &str) -> String {
        let mut out = String::new();
        // One entry per open `#ifdef`: whether its current branch is kept.
        let mut branches: Vec<bool> = Vec::new();
        for line in source.lines() {
            let trimmed = line.trim();
            if let Some(def) = trimmed.strip_prefix("#ifdef") {
                branches.push(self.defs.contains(&def.trim()));
            } else if trimmed == "#else" {
                let kept = branches.pop().expect("#else without #ifdef");
                branches.push(!kept);
            } else if trimmed == "#endif" {
                branches.pop().expect("#endif without #ifdef");
            } else if trimmed.starts_with('#') {
                // `#import` / `#define_import_path`: the libraries are inlined.
            } else if branches.iter().all(|kept| *kept) {
                out.push_str(line);
                out.push('\n');
            }
        }
        assert!(branches.is_empty(), "unterminated #ifdef");
        out
    }

    /// A planet layer's fragment shader as plain WGSL.
    fn layer(&self, shader: &str) -> String {
        [
            Self::VERTEX_OUTPUT_STUB,
            include_str!("../src/shaders/pixel_planet_functions.wgsl"),
            include_str!("../src/shaders/pixel_planet_world_grid.wgsl"),
            shader,
        ]
        .map(|source| self.preprocess(source))
        .concat()
    }

    /// The world-grid vertex shader as plain WGSL.
    fn world_grid_vertex(&self) -> String {
        [
            Self::MESH_FUNCTIONS_STUB,
            include_str!("../src/shaders/pixel_planet_world_grid.wgsl"),
            include_str!("../src/shaders/pixel_planet_world_grid_vertex.wgsl"),
        ]
        .map(|source| self.preprocess(source))
        .concat()
        .replace("mesh_functions::", "")
    }
}

/// Every planet layer validates in both pixel-grid modes.
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
        for (mode, composer) in [
            ("quad", PlanetShaderComposer::QUAD),
            ("world", PlanetShaderComposer::WORLD),
        ] {
            let source = composer.layer(shader);
            // Each mode must read its own vertex output and nothing of the other's.
            assert_eq!(
                source.contains("world_grid_cell(mesh)"),
                mode == "world",
                "{label} ({mode}): wrong snap branch"
            );
            assert_eq!(
                source.contains("floor(mesh.uv * u.pixels)"),
                mode == "quad",
                "{label} ({mode}): wrong snap branch"
            );
            validate(&format!("{label} ({mode})"), &source);
        }
    }
}

#[test]
fn pixel_planet_world_grid_vertex_is_valid_wgsl() {
    validate(
        "pixel_planet_world_grid_vertex",
        &PlanetShaderComposer::WORLD.world_grid_vertex(),
    );
}
