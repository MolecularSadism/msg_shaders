# msg_shaders

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

## Feature flags

- `render_2d` (default) — `Mesh2d` materials and the screen-space lensing pipeline
- `render_3d` — `Mesh3d` variant of the black-hole material

## Compatibility

| msg_shaders | Bevy | Rust |
|-------------|------|------|
| 0.2 | 0.18 | 1.85+ (edition 2024) |

## Standalone use

This crate is developed inside the Gravitaria workspace and inherits two
dependencies from it. To build it outside that workspace:

1. `bevy = { workspace = true, ... }` → `bevy = { version = "0.18", ... }`
   (keep `default-features = false` and the listed features).
2. `bevy_post_process_2d = { workspace = true }` → a `path` or `git`
   dependency; ship that crate alongside this one (it's a ~100-line
   render-graph ordering helper).
3. Replace `[lints] workspace = true` with your own lint table, or delete it.

## Tests

`cargo test` validates all WGSL shaders offline via naga (no GPU needed) and
runs the unit tests. A ready-made GitHub Actions workflow for standalone repos
is in `.github/workflows/ci.yml`.

## License

Dual-licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your
option. The black-hole shader derives from Eric Bruneton's black_hole_shader,
BSD-3-Clause — notice retained in [LICENSE-THIRD-PARTY](LICENSE-THIRD-PARTY).

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in this work, as defined in the Apache-2.0 license, shall be
dual-licensed as above, without any additional terms or conditions.
