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
