//! The world-pixel snap of `PixelGrid::World`, checked on the CPU.
//!
//! The snap itself lives only in WGSL (`pixel_planet_world_grid.wgsl` and
//! `pixel_planet_world_grid_vertex.wgsl`). [`WorldGridMap`] is a line-by-line
//! Rust mirror of those two shaders, not the code the GPU runs: it pins the
//! math down, and a change to the shaders has to be repeated here by hand.
//! `tests/pixel_grid_render.rs` checks the real shaders on a GPU.

#![cfg(feature = "render_2d")]

use bevy::math::{Affine3A, Quat, Vec2, Vec3};

/// Mirror of `WORLD_GRID_QUAD_MARGIN`.
const WORLD_GRID_QUAD_MARGIN: f32 = 1.5;

/// Mirror of `WorldGridCell`.
#[derive(Debug, Clone, Copy, PartialEq)]
struct WorldGridCell {
    uv: Vec2,
    cell: Vec2,
    pixels: f32,
}

impl WorldGridCell {
    /// Mirror of `world_grid_checker`.
    fn checker(&self) -> bool {
        let sum = self.cell.x + self.cell.y;
        (sum - 2.0 * (sum / 2.0).floor()) < 1.0
    }
}

/// Mirror of the flat part of `WorldGridVertexOutput`, with the vertex stage
/// that fills it and the fragment snap that reads it.
#[derive(Debug, Clone, Copy)]
struct WorldGridMap {
    cell_from_world_x: Vec2,
    cell_from_world_y: Vec2,
    origin: Vec2,
    cells: f32,
}

impl WorldGridMap {
    /// Mirror of the vertex stage, run on one corner of a centred
    /// `Rectangle::new(size, size)`: `corner` is that corner's sign pair.
    fn from_corner(world_from_local: &Affine3A, size: f32, corner: Vec2) -> Self {
        let position = corner * size * 0.5;
        // `Rectangle` UVs: u runs with +x, v runs against +y.
        let uv = Vec2::new(0.5 + 0.5 * corner.x, 0.5 - 0.5 * corner.y);

        let axis_x = world_from_local.matrix3.x_axis.truncate();
        let axis_y = world_from_local.matrix3.y_axis.truncate();

        let extent = position / (uv - Vec2::splat(0.5));
        let cells = extent.x.abs() * axis_x.length();

        let det = axis_x.x * axis_y.y - axis_y.x * axis_x.y;
        let local_from_world_x = Vec2::new(axis_y.y, -axis_x.y) / det;
        let local_from_world_y = Vec2::new(-axis_y.x, axis_x.x) / det;
        let cell_from_local = Vec2::splat(cells) / extent;

        Self {
            cell_from_world_x: cell_from_local * local_from_world_x,
            cell_from_world_y: cell_from_local * local_from_world_y,
            origin: world_from_local.translation.truncate(),
            cells,
        }
    }

    /// The map as the fragments of the quad see it, checked to be the same
    /// from all four corners: it is passed flat, so a triangle takes it from
    /// whichever corner is its provoking vertex.
    fn from_quad(world_from_local: &Affine3A, size: f32) -> Self {
        let map = Self::from_corner(world_from_local, size, Vec2::new(1.0, 1.0));
        for corner in [
            Vec2::new(-1.0, 1.0),
            Vec2::new(-1.0, -1.0),
            Vec2::new(1.0, -1.0),
        ] {
            let other = Self::from_corner(world_from_local, size, corner);
            assert_eq!(map.cell_from_world_x, other.cell_from_world_x);
            assert_eq!(map.cell_from_world_y, other.cell_from_world_y);
            assert_eq!(map.cells, other.cells);
        }
        map
    }

    /// Mirror of `world_grid_cell`.
    fn cell(&self, world_position: Vec2) -> WorldGridCell {
        let world_cell = world_position.floor();
        let offset = world_cell + Vec2::splat(0.5) - self.origin;
        let lookup = self.cell_from_world_x * offset.x
            + self.cell_from_world_y * offset.y
            + Vec2::splat(0.5 * self.cells - 0.5);
        let anchor =
            (Vec2::new(-self.origin.x, self.origin.y) + Vec2::splat(0.5 * self.cells)).round();
        WorldGridCell {
            uv: lookup / self.cells,
            cell: Vec2::new(world_cell.x, -world_cell.y - 1.0) + anchor,
            pixels: self.cells,
        }
    }
}

/// A planet quad placed in the world.
#[derive(Debug, Clone, Copy)]
struct Quad {
    size: f32,
    angle: f32,
    position: Vec2,
    scale: f32,
}

impl Quad {
    fn world_from_local(&self) -> Affine3A {
        Affine3A::from_scale_rotation_translation(
            Vec3::splat(self.scale),
            Quat::from_rotation_z(self.angle),
            self.position.extend(0.0),
        )
    }

    /// The quad's true UV at a world position, by inverting its transform
    /// rather than through the mirrored map.
    fn uv_at(&self, world_position: Vec2) -> Vec2 {
        let local = self
            .world_from_local()
            .inverse()
            .transform_point3(world_position.extend(0.0));
        Vec2::new(0.5 + local.x / self.size, 0.5 - local.y / self.size)
    }

    /// Every placement the tests sweep: right and non-right angles, whole and
    /// fractional positions, with and without an entity scale.
    fn sweep() -> Vec<Quad> {
        let mut quads = Vec::new();
        for size in [64.0, 100.0, 37.0] {
            for angle_degrees in [0.0_f32, 90.0, 17.0, 45.0, 123.4, 200.0, -61.5, 359.0] {
                for position in [
                    Vec2::ZERO,
                    Vec2::new(0.5, 0.5),
                    Vec2::new(10.37, -4.81),
                    Vec2::new(-1234.56, 789.01),
                ] {
                    for scale in [1.0, 2.5] {
                        quads.push(Quad {
                            size,
                            angle: angle_degrees.to_radians(),
                            position,
                            scale,
                        });
                    }
                }
            }
        }
        quads
    }

    /// The world pixels (their minimum corners) the quad can touch, margin
    /// included.
    fn world_pixels(&self) -> Vec<Vec2> {
        let reach = (self.size * self.scale * 0.75 + WORLD_GRID_QUAD_MARGIN).ceil();
        let centre = self.position.floor();
        let mut pixels = Vec::new();
        let mut y = -reach;
        while y <= reach {
            let mut x = -reach;
            while x <= reach {
                pixels.push(centre + Vec2::new(x, y));
                x += 1.0;
            }
            y += 1.0;
        }
        pixels
    }
}

/// Sample points spread over a unit square, up to just inside its edges.
const SAMPLES_IN_PIXEL: [Vec2; 9] = [
    Vec2::new(0.0, 0.0),
    Vec2::new(0.999, 0.0),
    Vec2::new(0.0, 0.999),
    Vec2::new(0.999, 0.999),
    Vec2::new(0.5, 0.5),
    Vec2::new(0.125, 0.875),
    Vec2::new(0.875, 0.125),
    Vec2::new(0.3, 0.6),
    Vec2::new(0.71, 0.02),
];

/// At any rotation and any fractional position, every sample point inside one
/// world unit square maps to the very same lookup point and dither cell.
#[test]
fn every_point_of_a_world_pixel_maps_to_one_cell() {
    for quad in Quad::sweep() {
        let map = WorldGridMap::from_quad(&quad.world_from_local(), quad.size);
        for pixel in quad.world_pixels() {
            let reference = map.cell(pixel + SAMPLES_IN_PIXEL[0]);
            for sample in SAMPLES_IN_PIXEL {
                assert_eq!(
                    map.cell(pixel + sample),
                    reference,
                    "{quad:?}: pixel {pixel} splits at sample {sample}"
                );
            }
        }
    }
}

/// The lookup point is the quad's true UV at the world pixel's centre, moved
/// half a cell toward the UV origin.
#[test]
fn lookup_point_is_the_pixel_centre_in_quad_uv() {
    for quad in Quad::sweep() {
        let map = WorldGridMap::from_quad(&quad.world_from_local(), quad.size);
        let cells = quad.size * quad.scale;
        for pixel in quad.world_pixels() {
            let expected = quad.uv_at(pixel + Vec2::splat(0.5)) - Vec2::splat(0.5 / cells);
            let cell = map.cell(pixel + Vec2::splat(0.5));
            // In cell units, so the tolerance is a fraction of a pixel.
            let error = (cell.uv - expected) * cells;
            assert!(
                error.abs().max_element() < 0.02,
                "{quad:?}: pixel {pixel} looks up {} instead of {expected}",
                cell.uv
            );
        }
    }
}

/// A cell is one world unit whatever the quad's size and scale: the cell
/// count is the quad's world width, neighbouring world pixels look up points
/// exactly one cell apart, and their dither cells are neighbours. No `pixels`
/// setting enters the map at all.
#[test]
fn cells_are_whole_world_units() {
    for quad in Quad::sweep() {
        let map = WorldGridMap::from_quad(&quad.world_from_local(), quad.size);
        let world_width = quad.size * quad.scale;
        assert!(
            (map.cells - world_width).abs() < world_width * 1e-5,
            "{quad:?}: {} cells across a quad {world_width} world units wide",
            map.cells
        );
        for pixel in quad.world_pixels() {
            let here = map.cell(pixel);
            for step in [Vec2::X, Vec2::Y] {
                let there = map.cell(pixel + step);
                let apart = ((there.uv - here.uv) * map.cells).length();
                assert!(
                    (apart - 1.0).abs() < 0.02,
                    "{quad:?}: pixels {pixel} and {} look up {apart} cells apart",
                    pixel + step
                );
                assert_eq!((there.cell - here.cell).abs(), step);
                assert_ne!(there.checker(), here.checker());
            }
        }
    }
}

/// An unrotated quad on whole world pixels is where the quad-space grid is
/// pixel-perfect already; there the world grid reproduces the quad-space snap
/// (`floor(uv * pixels) / pixels`, `floor(uv * pixels)`, `dither_bool`) bit for
/// bit when `pixels` equals the diameter.
#[test]
fn aligned_quad_snaps_like_the_quad_grid() {
    // An odd diameter lands on whole pixels when centred on a half pixel.
    for (size, position) in [
        (64.0_f32, Vec2::new(12.0, -7.0)),
        (100.0, Vec2::new(-300.0, 41.0)),
        (37.0, Vec2::new(0.5, 8.5)),
    ] {
        let quad = Quad {
            size,
            angle: 0.0,
            position,
            scale: 1.0,
        };
        let map = WorldGridMap::from_quad(&quad.world_from_local(), size);
        let pixels = size;
        let mut compared = 0;
        for pixel in quad.world_pixels() {
            let mesh_uv = quad.uv_at(pixel + Vec2::splat(0.5));
            if mesh_uv.min_element() < 0.0 || mesh_uv.max_element() > 1.0 {
                continue;
            }
            // The quad-space snap, as the layer shaders write it.
            let quad_cell = (mesh_uv * pixels).floor();
            let quad_uv = quad_cell / pixels;
            let gap = 2.0 / pixels;
            let phase = quad_uv.x + mesh_uv.y;
            let quad_dither = (phase - gap * (phase / gap).floor()) <= 1.0 / pixels;

            let world = map.cell(pixel + Vec2::splat(0.5));
            assert_eq!(world.uv, quad_uv, "size {size}: pixel {pixel}");
            assert_eq!(world.cell, quad_cell, "size {size}: pixel {pixel}");
            assert_eq!(world.pixels, pixels);
            assert_eq!(world.checker(), quad_dither, "size {size}: pixel {pixel}");
            compared += 1;
        }
        assert_eq!(compared, (size * size) as usize);
    }
}
