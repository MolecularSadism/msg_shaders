// Generic shaped deflection sources for the lensing velocity field.
//
// The inject pass (`lensing_field_inject.wgsl`) loops over an array of
// shape-tagged [`DeflectionSource`]s and sums each one's force into the
// velocity field. A gravitational lens is one shape (`Lens`); expanding
// explosion rings and directional shoves are others (`Ring`, `Line`). Every
// source contributes deflection only — the photon-ring / event-horizon disc is
// drawn solely from the visual `LensData[]` array (see `lensing_display.wgsl`),
// which this module does not touch. A `Lens` therefore needs no visual at all,
// and one that has a visual sizes it independently of its reach.
//
// Authoring is in world units via [`DeflectionShape`]. Gameplay injects sources
// two ways: a persistent [`LightDeflector`] component (transformed by the
// entity's `GlobalTransform` each frame), or a one-shot [`LightDeflectionRequest`]
// message resolved to world space by the caller. [`pack_deflection_sources`]
// gathers both, AABB-culls them against the lens canvas, and appends them to the
// field's source array after the black-hole `Lens` sources.

use bevy::prelude::*;
use bevy::render::render_resource::ShaderType;

use crate::LensingHoleCamera;
use crate::lensing_field::extract::LensingFieldExtractSource;

/// Strength magnitude below which a deflection source is skipped during the
/// gather. A resting `PulseValue`-driven deflector sits at zero between
/// pulses and contributes no force; dropping it frees a slot in the capped
/// source array (and lets the display pass idle when nothing is deflecting).
const MIN_DEFLECTION_STRENGTH: f32 = 1e-4;

/// Maximum number of deflection sources injected into the field in one frame.
///
/// The CPU gather appends sources up to this cap and `warn!`s on any drop, so a
/// full array never silently reads as complete coverage. At 48 B/source this is
/// a ~3 KB uniform array, well under the UBO size limit. To raise it past the
/// point where a fixed-size UBO array is comfortable, switch the binding to
/// `var<storage, read>` and drop the fixed array size — no shader logic changes.
pub const MAX_DEFLECTION_SOURCES: usize = 64;

/// Shape tag stored in `DeflectionSource::tag_strength.x`. Matches the `switch`
/// cases in `lensing_field_inject.wgsl`.
pub const TAG_LENS: f32 = 0.0;
/// Shape tag for an annular-sector outward push. See [`DeflectionShape::Ring`].
pub const TAG_RING: f32 = 1.0;
/// Shape tag for a directional band push. See [`DeflectionShape::Line`].
pub const TAG_LINE: f32 = 2.0;

/// One shape-tagged deflection source, packed for the inject shader's uniform
/// array. Three 16-byte rows (48 B); the shader reads `tag_strength.x` to
/// dispatch on shape and unpacks the geometry rows per shape.
#[derive(ShaderType, Clone, Copy, Default, Debug, PartialEq)]
pub struct DeflectionSource {
    /// `x` = shape tag (`TAG_LENS` / `TAG_RING` / `TAG_LINE`),
    /// `y` = strength, `zw` reserved.
    pub tag_strength: Vec4,
    /// Geometry row 0. Layout depends on the shape tag:
    /// - Lens: `(center.xy, size, core_radius)`
    /// - Ring: `(center.xy, inner_radius, thickness)`
    /// - Line: `(center.xy, half_length, thickness)`
    pub geom_a: Vec4,
    /// Geometry row 1. Layout depends on the shape tag:
    /// - Lens: unused
    /// - Ring: `(start_angle, arc, 0, 0)`
    /// - Line: `(rotation, 0, 0, 0)`
    pub geom_b: Vec4,
}

/// CPU-side authoring shape for a deflection source, in world units.
///
/// Author the local shape on a [`LightDeflector`] component or in a
/// [`LightDeflectionRequest`] message; [`DeflectionShape::pack`] lays it out for
/// the inject shader.
#[derive(Component, Reflect, Clone, Copy, Debug, PartialEq)]
pub enum DeflectionShape {
    /// Radial outward deflection that peaks across an annular plateau and fades
    /// to nothing at the rim — the gravitational-lens source.
    ///
    /// Three radii shape it, the outer two as fractions of `size`:
    ///
    /// - `size` — the reach in world units; the deflection is zero at and past it.
    /// - `core_radius` — outer edge of the peak plateau, where the outward taper begins. Widening
    ///   it carries full-strength bend further out.
    /// - `inner_radius` — inner edge of the plateau. Inside it the deflection ramps back down to
    ///   zero at the exact centre, so the centre carries no direction singularity. Set it to the
    ///   radius of whatever the lens sits behind — a black hole's event horizon — and the bend
    ///   peaks right where that becomes visible, wasting no gradient under an opaque disc.
    ///
    /// The peak is `strength * size` whatever the radii are, so they shape the
    /// warp without scaling it: a lens can be as small or as large as its
    /// owner's visual and still bend the scene across its full reach.
    Lens {
        center: Vec2,
        size: f32,
        core_radius: f32,
        inner_radius: f32,
    },

    /// Annular sector pushing radially outward with uniform magnitude inside the
    /// band and sector. A full explosion ring uses `arc = TAU`; a directional
    /// force-push fan uses a small arc. Gameplay animates `inner_radius` to make
    /// the wavefront expand.
    Ring {
        center: Vec2,
        inner_radius: f32,
        thickness: f32,
        /// Sector start angle in radians (`0` = +x, CCW).
        start_angle: f32,
        /// Sector sweep in radians; `TAU` is a closed ring.
        arc: f32,
    },

    /// Directed band: a uniform push along `rotation`. The band extends
    /// perpendicular to the push for `±half_length`, with depth `thickness`
    /// along the push axis (centered on the line). A flat shove or sword swipe.
    Line {
        center: Vec2,
        half_length: f32,
        /// Push direction in radians.
        rotation: f32,
        thickness: f32,
    },
}

impl DeflectionShape {
    /// Lays this shape out into a [`DeflectionSource`] for the inject shader,
    /// stamping the shape tag and `strength`.
    pub fn pack(&self, strength: f32) -> DeflectionSource {
        match *self {
            DeflectionShape::Lens {
                center,
                size,
                core_radius,
                inner_radius,
            } => DeflectionSource {
                tag_strength: Vec4::new(TAG_LENS, strength, 0.0, 0.0),
                geom_a: Vec4::new(center.x, center.y, size, core_radius),
                geom_b: Vec4::new(inner_radius, 0.0, 0.0, 0.0),
            },
            DeflectionShape::Ring {
                center,
                inner_radius,
                thickness,
                start_angle,
                arc,
            } => DeflectionSource {
                tag_strength: Vec4::new(TAG_RING, strength, 0.0, 0.0),
                geom_a: Vec4::new(center.x, center.y, inner_radius, thickness),
                geom_b: Vec4::new(start_angle, arc, 0.0, 0.0),
            },
            DeflectionShape::Line {
                center,
                half_length,
                rotation,
                thickness,
            } => DeflectionSource {
                tag_strength: Vec4::new(TAG_LINE, strength, 0.0, 0.0),
                geom_a: Vec4::new(center.x, center.y, half_length, thickness),
                geom_b: Vec4::new(rotation, 0.0, 0.0, 0.0),
            },
        }
    }

    /// Maps this locally-authored shape into world space through `transform`:
    /// the center is moved by the full affine, lengths scale by the transform's
    /// scale, and angles rotate by its Z rotation. Lets a [`LightDeflector`]
    /// follow a moving, rotating entity for free.
    pub fn transformed_by(&self, transform: &GlobalTransform) -> Self {
        let (scale, rotation, translation) = transform.to_scale_rotation_translation();
        let origin = translation.truncate();
        let z_rot = rotation.to_euler(EulerRot::ZYX).0;
        let s = scale.x;
        let to_world = |local: Vec2| origin + Vec2::from_angle(z_rot).rotate(local * s);
        match *self {
            DeflectionShape::Lens {
                center,
                size,
                core_radius,
                inner_radius,
            } => DeflectionShape::Lens {
                center: to_world(center),
                size: size * s,
                // Both are fractions of `size`, so they stay invariant under
                // scaling — the whole profile scales with the reach.
                core_radius,
                inner_radius,
            },
            DeflectionShape::Ring {
                center,
                inner_radius,
                thickness,
                start_angle,
                arc,
            } => DeflectionShape::Ring {
                center: to_world(center),
                inner_radius: inner_radius * s,
                thickness: thickness * s,
                start_angle: start_angle + z_rot,
                arc,
            },
            DeflectionShape::Line {
                center,
                half_length,
                rotation,
                thickness,
            } => DeflectionShape::Line {
                center: to_world(center),
                half_length: half_length * s,
                rotation: rotation + z_rot,
                thickness: thickness * s,
            },
        }
    }

    /// World-space bounding box of this shape's influence, for culling against
    /// the lens canvas.
    pub fn world_aabb(&self) -> Rect {
        match *self {
            DeflectionShape::Lens { center, size, .. } => {
                Rect::from_center_half_size(center, Vec2::splat(size))
            }
            DeflectionShape::Ring {
                center,
                inner_radius,
                thickness,
                ..
            } => Rect::from_center_half_size(center, Vec2::splat(inner_radius + thickness)),
            DeflectionShape::Line {
                center,
                half_length,
                rotation,
                thickness,
            } => {
                let dir = Vec2::from_angle(rotation);
                let perp = Vec2::new(-dir.y, dir.x);
                let half = (perp * half_length).abs() + (dir * (thickness * 0.5)).abs();
                Rect::from_center_half_size(center, half)
            }
        }
    }
}

/// Persistent deflection source carried by an entity. A system transforms the
/// local [`shape`](Self::shape) by the entity's `GlobalTransform` each frame,
/// AABB-culls it, and injects it into the field — so the source follows the
/// entity, and despawning the entity removes the source next frame. Animate an
/// expanding ring by mutating the shape's radius each frame from gameplay.
#[derive(Component, Reflect, Clone, Debug, Deref, DerefMut)]
#[reflect(Component)]
pub struct LightDeflector {
    /// Shape authored in the entity's local space.
    pub shape: DeflectionShape,
    /// Force magnitude passed to the shape's inject branch — for a `Lens` the
    /// peak deflection as a fraction of its `size`, for `Ring` / `Line` the
    /// world-space push. Dereferencing a `LightDeflector` yields this scalar, so
    /// a `PulseValue<LightDeflector>` can drive it directly.
    #[deref]
    pub strength: f32,
}

impl LightDeflector {
    /// Sets the reach of a `Lens` shape, leaving other shapes unchanged. Lets a
    /// size-animating system (a black hole, the level-ending hole) grow the
    /// deflection with whatever it drives.
    pub fn set_lens_size(&mut self, size: f32) {
        if let DeflectionShape::Lens { size: s, .. } = &mut self.shape {
            *s = size;
        }
    }

    /// Current reach of a `Lens` shape, or `None` for the other shapes. Lets a
    /// driver skip a write — and the change detection that comes with it — when
    /// the size it computed is the one already stored.
    pub fn lens_size(&self) -> Option<f32> {
        match self.shape {
            DeflectionShape::Lens { size, .. } => Some(size),
            _ => None,
        }
    }
}

/// One-shot, fire-and-forget deflection source for a single frame (e.g. a sword
/// swipe at impact), needing no entity. The [`shape`](Self::shape) is in world
/// space, fully resolved by the caller.
#[derive(Message, Clone, Debug)]
pub struct LightDeflectionRequest {
    /// Shape in world space.
    pub shape: DeflectionShape,
    /// Force magnitude passed to the shape's inject branch.
    pub strength: f32,
}

/// Axis-aligned bounding-box overlap test.
fn aabb_overlaps(a: Rect, b: Rect) -> bool {
    a.min.x <= b.max.x && a.max.x >= b.min.x && a.min.y <= b.max.y && a.max.y >= b.min.y
}

/// Gathers every deflection source into the field's source array each frame:
/// [`LightDeflector`] components (black-hole lenses, shock waves, …) and
/// one-shot [`LightDeflectionRequest`] messages. This is the single owner of the
/// inject `sources` array — `drive_lensing` only builds the visual disc/ring.
///
/// Reuses the [`LensingHoleCamera`] to derive the canvas square it AABB-culls
/// against. Components are transformed into world space by their
/// `GlobalTransform`; messages arrive already in world space. Over-cap sources
/// are dropped with a `warn!` (see [`LensingFieldExtractSource::set_sources`]).
pub fn pack_deflection_sources(
    q_camera: Query<(&Projection, &GlobalTransform), With<LensingHoleCamera>>,
    q_deflectors: Query<(&LightDeflector, &GlobalTransform)>,
    mut requests: MessageReader<LightDeflectionRequest>,
    field_source: Option<ResMut<LensingFieldExtractSource>>,
) {
    let Some(mut source) = field_source else {
        requests.clear();
        return;
    };
    let Ok((Projection::Orthographic(ortho), camera_gt)) = q_camera.single() else {
        requests.clear();
        return;
    };

    let canvas_center = camera_gt.translation().truncate();
    let canvas_extent = crate::lens_capture_extent(ortho.area.size());
    let canvas = Rect::from_center_half_size(canvas_center, canvas_extent * 0.5);

    let mut sources: Vec<DeflectionSource> = Vec::new();
    for (deflector, gt) in &q_deflectors {
        if deflector.strength.abs() < MIN_DEFLECTION_STRENGTH {
            continue;
        }
        let shape = deflector.shape.transformed_by(gt);
        if aabb_overlaps(shape.world_aabb(), canvas) {
            sources.push(shape.pack(deflector.strength));
        }
    }
    for req in requests.read() {
        if req.strength.abs() < MIN_DEFLECTION_STRENGTH {
            continue;
        }
        if aabb_overlaps(req.shape.world_aabb(), canvas) {
            sources.push(req.shape.pack(req.strength));
        }
    }

    source.set_sources(&sources);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::{FRAC_PI_2, PI, TAU};

    // WGSL `fract` is `x - floor(x)` (always in `[0, 1)`), unlike Rust's
    // sign-preserving `f32::fract`. The angle-wrap math relies on the WGSL
    // semantics, so the references below match it.
    fn wgsl_fract(x: f32) -> f32 {
        x - x.floor()
    }

    /// Rust port of `lens_force` in `lensing_field_inject.wgsl`, reading the
    /// packed geometry rows / strength.
    fn lens_force_ref(world: Vec2, geom_a: Vec4, geom_b: Vec4, strength: f32) -> Vec2 {
        let center = Vec2::new(geom_a.x, geom_a.y);
        let size = geom_a.z.max(1e-4);
        let core = geom_a.w.clamp(1e-3, 0.99);
        let inner = geom_b.x.clamp(1e-4, core);
        let delta = world - center;
        let dist = delta.length();
        let r = dist / size;
        if r >= 1.0 {
            return Vec2::ZERO;
        }
        let ramp = (r / inner).clamp(0.0, 1.0);
        let taper = (core / r.max(core) - core) / (1.0 - core);
        (delta / dist.max(1e-5)) * strength * size * ramp * taper
    }

    fn ring_force_ref(world: Vec2, geom_a: Vec4, geom_b: Vec4, strength: f32) -> Vec2 {
        let center = Vec2::new(geom_a.x, geom_a.y);
        let inner = geom_a.z;
        let thick = geom_a.w;
        let delta = world - center;
        let dist = delta.length();
        if dist < inner || dist > inner + thick {
            return Vec2::ZERO;
        }
        let ang = delta.y.atan2(delta.x);
        let from_start = wgsl_fract((ang - geom_b.x) / TAU + 1.0) * TAU;
        if from_start > geom_b.y {
            return Vec2::ZERO;
        }
        (delta / dist.max(1e-5)) * strength
    }

    fn line_force_ref(world: Vec2, geom_a: Vec4, geom_b: Vec4, strength: f32) -> Vec2 {
        let center = Vec2::new(geom_a.x, geom_a.y);
        let half_len = geom_a.z;
        let thick = geom_a.w;
        let rot = geom_b.x;
        let dir = Vec2::new(rot.cos(), rot.sin());
        let perp = Vec2::new(-dir.y, dir.x);
        let delta = world - center;
        let along = delta.dot(dir);
        let across = delta.dot(perp);
        if across.abs() > half_len || along.abs() > thick * 0.5 {
            return Vec2::ZERO;
        }
        dir * strength
    }

    #[test]
    fn lens_packs_its_three_radii() {
        let center = Vec2::new(10.0, -4.0);
        let size = 3.0;
        let core = 0.12;
        let inner = 0.04;
        let strength = 1.85;
        let src = DeflectionShape::Lens {
            center,
            size,
            core_radius: core,
            inner_radius: inner,
        }
        .pack(strength);

        assert_eq!(src.tag_strength.x, TAG_LENS);
        assert_eq!(src.tag_strength.y, strength);
        assert_eq!(src.geom_a, Vec4::new(center.x, center.y, size, core));
        assert_eq!(src.geom_b, Vec4::new(inner, 0.0, 0.0, 0.0));
    }

    fn lens_at(size: f32, core: f32, inner: f32, strength: f32) -> impl Fn(f32) -> Vec2 {
        let src = DeflectionShape::Lens {
            center: Vec2::ZERO,
            size,
            core_radius: core,
            inner_radius: inner,
        }
        .pack(strength);
        move |d: f32| lens_force_ref(Vec2::new(d, 0.0), src.geom_a, src.geom_b, strength)
    }

    /// The deflection holds `strength * size` across the whole plateau, points
    /// outward, and vanishes at both the exact centre and the rim.
    #[test]
    fn lens_force_holds_its_peak_across_the_plateau() {
        let (size, core, inner, strength) = (40.0, 0.5, 0.1, 0.1);
        let at = lens_at(size, core, inner, strength);
        let peak = Vec2::new(strength * size, 0.0);

        // Plateau: outward at full strength from the inner edge to the core.
        for r in [inner, 0.2, 0.35, core] {
            let force = at(size * r);
            assert!(
                (force - peak).length() < 1e-3,
                "r {r}: {force:?} is not the peak {peak:?}"
            );
        }
        // Centre and rim: nothing.
        assert!(at(0.0).length() < 1e-4);
        assert_eq!(at(size), Vec2::ZERO);
        assert_eq!(at(size * 1.5), Vec2::ZERO);
        // Inside the plateau the magnitude only rises, outside it only falls.
        let mut previous = 0.0;
        for step in 0..=10 {
            let current = at(size * inner * step as f32 / 10.0).length();
            assert!(
                current >= previous,
                "ramp step {step}: {current} < {previous}"
            );
            previous = current;
        }
        for step in 0..=10 {
            let current = at(size * (core + (1.0 - core) * step as f32 / 10.0)).length();
            assert!(
                current <= previous,
                "taper step {step}: {current} > {previous}"
            );
            previous = current;
        }
    }

    /// The inner radius moves where the bend peaks without touching how strong
    /// it is or how far it reaches: a lens can hug a pinhead event horizon and
    /// still carry its full plateau outward.
    #[test]
    fn inner_radius_moves_the_peak_without_scaling_it() {
        let (size, core, strength) = (40.0, 0.5, 0.1);
        for inner in [0.4, 0.1, 0.01, 0.001] {
            let at = lens_at(size, core, inner, strength);
            // Peak reached at the inner edge, and still held at the core.
            for r in [inner, core] {
                assert!(
                    (at(size * r).length() - strength * size).abs() < 1e-2,
                    "inner {inner} at r {r}: {} != {}",
                    at(size * r).length(),
                    strength * size
                );
            }
            // Just inside the inner edge the bend is already tapering off, so
            // nothing is spent under whatever visual the lens sits behind.
            assert!(at(size * inner * 0.5).length() < strength * size * 0.75);
        }
    }

    /// The peak deflection is set by `strength` and the reach alone: shrinking
    /// the plateau to a sliver leaves it untouched. This is what lets a tiny
    /// visual hole keep a full-strength, full-reach lens.
    #[test]
    fn lens_peak_is_independent_of_its_radii() {
        let size = 40.0;
        let strength = 0.1;
        for core in [0.5, 0.25, 0.05, 0.005] {
            let at = lens_at(size, core, core, strength);
            let peak = at(size * core).length();
            assert!(
                (peak - strength * size).abs() < 1e-2,
                "core {core}: peak {peak} != {}",
                strength * size
            );
        }
    }

    #[test]
    fn ring_full_pushes_radially_outward() {
        let center = Vec2::new(0.0, 0.0);
        let src = DeflectionShape::Ring {
            center,
            inner_radius: 4.0,
            thickness: 2.0,
            start_angle: 0.0,
            arc: TAU,
        }
        .pack(7.0);

        // Inside the band: outward push of magnitude `strength`.
        let f = ring_force_ref(Vec2::new(5.0, 0.0), src.geom_a, src.geom_b, 7.0);
        assert!((f - Vec2::new(7.0, 0.0)).length() < 1e-4);

        // Inside the inner hole and outside the band: no force.
        assert_eq!(
            ring_force_ref(Vec2::new(2.0, 0.0), src.geom_a, src.geom_b, 7.0),
            Vec2::ZERO
        );
        assert_eq!(
            ring_force_ref(Vec2::new(9.0, 0.0), src.geom_a, src.geom_b, 7.0),
            Vec2::ZERO
        );
    }

    #[test]
    fn ring_arc_gates_by_angle_across_seam() {
        // A fan that starts at 5π/3 and sweeps π/2 wraps past the 0-angle seam,
        // covering 5π/3 → 2π → π/6.
        let geom_a = Vec4::new(0.0, 0.0, 1.0, 2.0);
        let start = 5.0 * PI / 3.0;
        let arc = FRAC_PI_2;
        let geom_b = Vec4::new(start, arc, 0.0, 0.0);

        // At angle π/12 (just past the seam): inside the swept arc.
        let inside = Vec2::from_angle(PI / 12.0) * 2.0;
        assert_ne!(ring_force_ref(inside, geom_a, geom_b, 1.0), Vec2::ZERO);

        // At angle π (opposite side): outside the arc.
        let outside = Vec2::from_angle(PI) * 2.0;
        assert_eq!(ring_force_ref(outside, geom_a, geom_b, 1.0), Vec2::ZERO);
    }

    #[test]
    fn line_pushes_along_rotation_within_band() {
        // Push along +y (rotation = π/2); band extends ±5 along x, depth 2 along y.
        let src = DeflectionShape::Line {
            center: Vec2::ZERO,
            half_length: 5.0,
            rotation: FRAC_PI_2,
            thickness: 2.0,
        }
        .pack(3.0);

        // Inside the band: push points +y with magnitude `strength`.
        let f = line_force_ref(Vec2::new(2.0, 0.0), src.geom_a, src.geom_b, 3.0);
        assert!((f - Vec2::new(0.0, 3.0)).length() < 1e-4);

        // Beyond the perpendicular half-length: no force.
        assert_eq!(
            line_force_ref(Vec2::new(6.0, 0.0), src.geom_a, src.geom_b, 3.0),
            Vec2::ZERO
        );
        // Beyond the band depth: no force.
        assert_eq!(
            line_force_ref(Vec2::new(0.0, 2.0), src.geom_a, src.geom_b, 3.0),
            Vec2::ZERO
        );
    }

    #[test]
    fn transformed_by_moves_and_rotates() {
        let transform = GlobalTransform::from(
            Transform::from_translation(Vec3::new(100.0, 50.0, 0.0))
                .with_rotation(Quat::from_rotation_z(FRAC_PI_2)),
        );
        let local = DeflectionShape::Line {
            center: Vec2::new(10.0, 0.0),
            half_length: 4.0,
            rotation: 0.0,
            thickness: 1.0,
        };
        let DeflectionShape::Line {
            center,
            rotation,
            half_length,
            ..
        } = local.transformed_by(&transform)
        else {
            panic!("expected Line");
        };

        // Local +x point (10, 0) rotates 90° CCW to (0, 10) then translates.
        assert!((center - Vec2::new(100.0, 60.0)).length() < 1e-4);
        // Push direction rotates with the entity.
        assert!((rotation - FRAC_PI_2).abs() < 1e-5);
        // No scale → length preserved.
        assert!((half_length - 4.0).abs() < 1e-5);
    }

    #[test]
    fn world_aabb_bounds_each_shape() {
        let lens = DeflectionShape::Lens {
            center: Vec2::new(2.0, 3.0),
            size: 5.0,
            core_radius: 0.1,
            inner_radius: 0.05,
        }
        .world_aabb();
        assert_eq!(lens.min, Vec2::new(-3.0, -2.0));
        assert_eq!(lens.max, Vec2::new(7.0, 8.0));

        let ring = DeflectionShape::Ring {
            center: Vec2::ZERO,
            inner_radius: 4.0,
            thickness: 2.0,
            start_angle: 0.0,
            arc: TAU,
        }
        .world_aabb();
        assert_eq!(ring.min, Vec2::splat(-6.0));
        assert_eq!(ring.max, Vec2::splat(6.0));
    }
}
