#define_import_path msg_shaders::pixel_planet_world_grid

// World-space pixel grid for the pixel-planet layers (`PixelGrid::World`).
//
// In this mode a planet pixel cell is a unit square of the integer world grid,
// whatever the quad's rotation or (fractional) position. The fragment snaps its
// world position to that square and maps the square's centre back into the
// quad's own rotated UV space, so everything a layer evaluates per cell is
// evaluated once per world pixel and the surface turns underneath a fixed grid.
//
// The map from world space into the quad is the same for every fragment of a
// quad, so the vertex stage (`pixel_planet_world_grid_vertex.wgsl`) derives it
// from the mesh transform and hands it over flat (uninterpolated). All
// fragments inside one world pixel therefore compute bit-identical results.

// How far, in world units, the vertex stage pushes the quad's edges outward. A
// world pixel is drawn whenever its lookup point lies on the planet, and that
// point sits up to half a cell diagonal from fragments up to half a pixel
// diagonal from the pixel centre, so the quad has to cover that much more than
// its own outline.
const WORLD_GRID_QUAD_MARGIN: f32 = 1.5;

struct WorldGridVertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) world_position: vec4<f32>,
    // Columns of the linear map from a world-space offset (relative to
    // `origin`) to an offset in cell units along the quad's UV axes.
    @location(1) @interpolate(flat) cell_from_world_x: vec2<f32>,
    @location(2) @interpolate(flat) cell_from_world_y: vec2<f32>,
    // World position of the quad's centre (UV 0.5, 0.5).
    @location(3) @interpolate(flat) origin: vec2<f32>,
    // Cells across the quad: its width in world units.
    @location(4) @interpolate(flat) cells: f32,
}

// The pixel cell a fragment belongs to.
struct WorldGridCell {
    // Lookup point in the quad's UV space, constant across the world pixel.
    uv: vec2<f32>,
    // Integer cell coordinate for the ordered dither patterns.
    cell: vec2<f32>,
    // Cells across the quad; stands in for the `pixels` uniform.
    pixels: f32,
}

fn world_grid_cell(mesh: WorldGridVertexOutput) -> WorldGridCell {
    let world_cell = floor(mesh.world_position.xy);
    let offset = world_cell + vec2<f32>(0.5) - mesh.origin;
    // The pixel centre in cell units, moved half a cell toward the UV origin:
    // the quad-space grid looks its cells up at their minimum-UV corner, and
    // for an unrotated quad on whole world pixels this lands on exactly the
    // same points.
    let lookup = mesh.cell_from_world_x * offset.x + mesh.cell_from_world_y * offset.y
        + vec2<f32>(0.5 * mesh.cells - 0.5);

    // Dither cells count along the world grid (y flipped to run the way UV
    // does), shifted by the whole number of pixels that follows the quad
    // around, so a drifting planet carries its stipple pattern with it.
    let anchor = round(vec2<f32>(-mesh.origin.x, mesh.origin.y) + vec2<f32>(0.5 * mesh.cells));

    var out: WorldGridCell;
    out.uv = lookup / mesh.cells;
    out.cell = vec2<f32>(world_cell.x, -world_cell.y - 1.0) + anchor;
    out.pixels = mesh.cells;
    return out;
}

// Checkerboard over the cells; the world-grid counterpart of `dither_bool`.
fn world_grid_checker(cell: vec2<f32>) -> bool {
    let sum = cell.x + cell.y;
    return (sum - 2.0 * floor(sum / 2.0)) < 1.0;
}
