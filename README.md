# msg_shaders

[![CI](https://github.com/MolecularSadism/msg_shaders/workflows/CI/badge.svg)](https://github.com/MolecularSadism/msg_shaders/actions)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](https://github.com/MolecularSadism/msg_shaders#license)
[![Bevy](https://img.shields.io/badge/Bevy-0.18-blue.svg)](https://bevyengine.org/)
[![Rust](https://img.shields.io/badge/rust-2024%20edition-orange.svg)](https://www.rust-lang.org/)

Shader effects for [Bevy](https://bevyengine.org): black holes, gravitational
lensing, palette quantization, and pixelation. Built for
[Gravitaria](https://github.com/ffmulks/gravitaria) and shared as-is — APIs
track the game's needs.

| Effect | Plugin | What it does |
|--------|--------|--------------|
| Black hole | `BlackHolePlugin` | Schwarzschild black-hole material on a quad: accretion disc, photon ring, Doppler beaming, relativistic lensing. Based on [Eric Bruneton's black_hole_shader](https://github.com/ebruneton/black_hole_shader). |
| Gravitational lensing | `LensingHolePlugin` | Screen-space warp of the camera's view, driven by a GPU-simulated deflection field. Generic `LightDeflector` sources (lens / ring / line); optional `BlackHoleOverlay` photon-ring + event-horizon discs. |
| Color quantization | `ColorQuantizationPlugin` | Nearest-palette quantization in Oklab space with Bayer dithering — a material plus a WGSL function library other shaders can `#import`. |
| Pixelation | `PixelationPlugin` | World-anchored pixelation — material plus WGSL function library. |
| Nebula + starfield | `NebulaPlugin` | A stack of procedural nebula cloud layers (each its own noise, three-stop color ramp, and parallax) composited over a base color, plus a stack of twinkling star layers. Snapped to the art-pixel grid and reduced to a palette with dithering; stars pulse (winking fully dark) and shift color over time. Reuses the quantization and pixelation modules. |

## Quick start

Black hole on the default 2D camera:

```rust
use bevy::prelude::*;
use msg_shaders::prelude::*;

fn main() {
    App::new()
        .add_plugins((DefaultPlugins, BlackHolePlugin))
        .add_systems(Startup, |mut commands: Commands| {
            commands.spawn(Camera2d);
            commands.spawn((BlackHole::default(), Transform::default()));
        })
        .run();
}
```

Screen-space lensing — mark the camera, then any entity with a `LightDeflector`
bends the rendered scene around itself:

```rust
use bevy::prelude::*;
use msg_shaders::prelude::*;

fn setup(mut commands: Commands) {
    commands.spawn((Camera2d, LensingHoleCamera));
    commands.spawn((
        LightDeflector {
            shape: DeflectionShape::Lens {
                center: Vec2::ZERO,
                size: 100.0,
                inner_radius: 0.2,
            },
            strength: 0.5,
        },
        Transform::default(),
    ));
}
// App: .add_plugins((DefaultPlugins, LensingHolePlugin)).add_systems(Startup, setup)
```

Tune the field via the `LensingFieldSettings` resource (force scale, falloff,
off-screen edge handling).

Layered nebula + twinkling starfield background on a quad:

```rust
use bevy::prelude::*;
use msg_shaders::prelude::*;

fn main() {
    App::new()
        .add_plugins((DefaultPlugins, NebulaPlugin))
        .add_systems(Startup, |mut commands: Commands| {
            commands.spawn(Camera2d);
            commands.spawn((
                Nebula {
                    // Two cloud layers + two star layers by default; override
                    // `layers` / `stars` for a custom stack. Linear-RGB palette;
                    // empty leaves the clouds un-quantized.
                    palette: vec![[0.05, 0.03, 0.1, 1.0], [0.35, 0.12, 0.55, 1.0], [1.0, 1.0, 1.0, 1.0]],
                    ..default()
                },
                Transform::default(),
            ));
        })
        .run();
}
```

Each [`NebulaLayer`] carries its own noise settings, a three-stop color ramp
(`c1` edge → `c2` body → `c3` core), and a `parallax` factor; layers composite
back-to-front, `Over` (default) or `Additive`. [`StarLayer`]s likewise carry
their own spacing/density/parallax. Set `Nebula::pixel_size` to `0.0` (the
default) to derive the art-pixel size from screen derivatives — pixel-perfect and
zoom-tracking. Set `scroll` to the camera's world position each frame; each layer
multiplies it by its own parallax factor. `rotation` spins the whole sky. A
runnable version is in [`examples/nebula.rs`](examples/nebula.rs):

```sh
cargo run --example nebula
```

A runnable version of this — an orbiting black hole warping a tile grid — is
in [`examples/lensing.rs`](examples/lensing.rs):

```sh
cargo run --example lensing
```

## Feature flags

- `render_2d` (default) — `Mesh2d` materials and the screen-space lensing pipeline
- `render_3d` — `Mesh3d` variant of the black-hole material
- `serde` — `Serialize`/`Deserialize` on the plain-data settings enums and the `BlackHole`,
  `BlackHoleColors`, `BlackHoleGeometry`, and `HoleQuantization` config types, so a game can
  author them straight from its own config files. The config structs use struct-level
  `#[serde(default, deny_unknown_fields)]`: a partial file sets only the fields it names (the
  rest fall back to the `Default` impls), while a misspelled field name is a parse error
  rather than being silently ignored

## Compatibility

| msg_shaders | Bevy | Rust |
|-------------|------|------|
| 0.2-0.4 | 0.18 | 1.85+ (edition 2024) |

## Installation

Not yet on crates.io — depend on the repository:

```toml
[dependencies]
msg_shaders = { git = "https://github.com/MolecularSadism/msg_shaders", tag = "v0.4.1" }
```

The one studio dependency is
[`msg_post_process`](https://github.com/MolecularSadism/msg_post_process), a
small render-graph ordering helper, itself pulled in as a `git` dependency. A
`Cargo.lock` is committed for reproducible builds.

## Tests

`cargo test` validates all WGSL shaders offline via naga (no GPU needed) and
runs the unit tests. The same checks run in CI
(`.github/workflows/ci.yml`).

## License

Dual-licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your
option. The black-hole shader derives from Eric Bruneton's black_hole_shader,
BSD-3-Clause — notice retained in [LICENSE-THIRD-PARTY](LICENSE-THIRD-PARTY).

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in this work, as defined in the Apache-2.0 license, shall be
dual-licensed as above, without any additional terms or conditions.
