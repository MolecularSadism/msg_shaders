// Pixelation utility functions for import into materials.
// Snaps sampling coordinates to a grid so procedural or textured content
// renders as discrete blocks, independent of the output resolution.
//
// Usage in importing shader:
//   #import msg_shaders::pixelate_functions as px
//   let snapped = px::pixelate_uv(in.uv, vec2<f32>(grid, grid));

#define_import_path msg_shaders::pixelate_functions

// Snap a UV to the center of a grid cell, producing a blocky sampling pattern.
// `grid` is the number of cells across each axis over the input's [0,1] range.
// A grid component <= 0 disables snapping on that axis (passes the coordinate
// through unchanged), so a zero grid is a no-op.
fn pixelate_uv(uv: vec2<f32>, grid: vec2<f32>) -> vec2<f32> {
    var out = uv;
    if grid.x > 0.0 {
        out.x = (floor(uv.x * grid.x) + 0.5) / grid.x;
    }
    if grid.y > 0.0 {
        out.y = (floor(uv.y * grid.y) + 0.5) / grid.y;
    }
    return out;
}

// Convenience for a square grid of `cells` cells across both axes.
fn pixelate_uv_square(uv: vec2<f32>, cells: f32) -> vec2<f32> {
    return pixelate_uv(uv, vec2<f32>(cells, cells));
}

// World units covered by one physical pixel at the current fragment, read from
// the screen-space derivatives of a world-space position. `1.0 / this` is the
// world-to-physical-pixel ratio; it tracks the camera's zoom and rotation, so
// callers never hardcode it. Call only in the fragment stage under uniform
// control flow (the derivative builtins require it).
fn world_units_per_pixel(world: vec2<f32>) -> f32 {
    let dx = vec2<f32>(dpdx(world.x), dpdy(world.x));
    let dy = vec2<f32>(dpdx(world.y), dpdy(world.y));
    return max(length(dx), length(dy));
}

// World-space size of one displayed art pixel. The scene is magnified by an
// integer number of physical pixels per world unit (`1 / wpp`, e.g. 4 at base
// zoom); one art pixel spans that many physical pixels, which is one world unit.
// Snapping to this grid keeps procedural detail at the same resolution as the
// surrounding pixel art instead of the finer display resolution.
fn art_pixel_size(wpp: f32) -> f32 {
    if wpp <= 0.0 {
        return 1.0;
    }
    return max(round(1.0 / wpp), 1.0) * wpp;
}

// Rotate a 2D vector by `angle` radians (counter-clockwise). Pass a negative
// angle to map a world-space offset into a frame rotated by `angle`, so a grid
// snapped on the result aligns to that rotated frame (e.g. a spinning hole's
// own axes) instead of the world axes.
fn rotate2d(v: vec2<f32>, angle: f32) -> vec2<f32> {
    let c = cos(angle);
    let s = sin(angle);
    return vec2<f32>(c * v.x - s * v.y, s * v.x + c * v.y);
}

// Snap a world-space position to the center of a grid cell `cell` world units
// across. A `cell <= 0` passes the position through unchanged.
fn pixelate_world(pos: vec2<f32>, cell: f32) -> vec2<f32> {
    if cell <= 0.0 {
        return pos;
    }
    return (floor(pos / cell) + vec2<f32>(0.5)) * cell;
}

// Grid-cell index of a world position, biased positive so it can index the Bayer
// dither matrices (which read it as u32) for negative world coordinates too. The
// bias is a multiple of both Bayer matrix sizes, leaving the pattern phase
// unchanged. One dither sample per cell makes each art-pixel block quantize as a
// unit.
fn pixel_cell_index(pos: vec2<f32>, cell: f32) -> vec2<f32> {
    if cell <= 0.0 {
        return pos;
    }
    return floor(pos / cell) + vec2<f32>(4096.0);
}
