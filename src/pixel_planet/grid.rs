//! The space a planet's pixel grid lives in.

use bevy::prelude::*;
use bevy::render::render_resource::RenderPipelineDescriptor;

/// The space the pixel grid of every [`PixelPlanet`](super::PixelPlanet) lives in. Chosen once,
/// where the plugin is added: [`PixelPlanetPlugin::pixel_grid`](super::PixelPlanetPlugin).
///
/// The choice compiles into the layer pipelines as a shader def, so neither mode costs anything
/// per frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Reflect)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum PixelGrid {
    /// Cells are squares of the planet quad's own UV space, [`PlanetSpec::pixels`] of them
    /// across the body. The grid turns and moves with the quad, which is pixel-perfect only
    /// while the quad is unrotated and lands on whole pixels of whatever it is drawn into.
    ///
    /// [`PlanetSpec::pixels`]: super::PlanetSpec::pixels
    #[default]
    Quad,
    /// Cells are the unit squares of the integer world grid, at any quad rotation and any
    /// fractional quad position: every world pixel gets exactly one flat colour, all layers of
    /// a planet share that grid, and the surface turns and drifts underneath it.
    ///
    /// Each world pixel is looked up at one point of the quad's rotated UV space — its centre,
    /// moved half a cell toward the UV origin, which is where [`Quad`](Self::Quad) looks up an
    /// unrotated planet on whole world pixels, so there the two modes draw the same picture.
    ///
    /// A cell is always one world unit, so the cell count across the body is its world-space
    /// width (`diameter` times the entity's scale) and [`PlanetSpec::pixels`] is not used: a
    /// value that does not match the diameter changes nothing. Dither widths, which are
    /// measured in cells, are therefore measured in world pixels. Under a non-uniform scale the
    /// cell count is taken along the quad's x axis.
    ///
    /// [`PlanetSpec::pixels`]: super::PlanetSpec::pixels
    World,
}

impl PixelGrid {
    /// Shader def that switches the layer shaders to the world grid.
    const WORLD_SHADER_DEF: &str = "PIXEL_GRID_WORLD";

    /// Applies this grid to a planet layer's pipeline. [`Quad`](Self::Quad) leaves the
    /// descriptor as Bevy built it.
    pub(crate) fn specialize(self, descriptor: &mut RenderPipelineDescriptor) {
        match self {
            PixelGrid::Quad => {}
            PixelGrid::World => {
                descriptor.vertex.shader = super::WORLD_GRID_VERTEX_SHADER;
                descriptor
                    .vertex
                    .shader_defs
                    .push(Self::WORLD_SHADER_DEF.into());
                if let Some(fragment) = descriptor.fragment.as_mut() {
                    fragment.shader_defs.push(Self::WORLD_SHADER_DEF.into());
                }
            }
        }
    }
}
