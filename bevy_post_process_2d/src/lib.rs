//! Thin ergonomics for ordering 2D full-screen post-process passes.
//!
//! A post-process pass is the standard Bevy pattern: a
//! [`ViewNode`] that reads the camera's
//! [`ViewTarget`](bevy::render::view::ViewTarget) via `post_process_write` and
//! draws a full-screen triangle (see Bevy's `custom_post_processing` example).
//! This crate adds nothing to that node — it only provides an extension trait so
//! a plugin can add its node to the `Core2d` graph and wire its ordering edges
//! in one place, reading like prose:
//!
//! ```
//! use bevy::prelude::*;
//! use bevy::core_pipeline::core_2d::graph::Node2d;
//! use bevy::render::render_graph::RenderLabel;
//! use bevy_post_process_2d::PostProcess2dAppExt;
//!
//! #[derive(Debug, Clone, PartialEq, Eq, Hash, RenderLabel)]
//! struct MyEffectLabel;
//!
//! let mut app = App::new();
//! app.add_plugins(MinimalPlugins);
//! // Place the pass in the post-main-pass band. (No render app under
//! // MinimalPlugins, so this is a no-op here — it shows the call shape.)
//! app.render_between(
//!     MyEffectLabel,
//!     Node2d::EndMainPass,
//!     Node2d::StartMainPassPostProcessing,
//! );
//! ```
//!
//! Edges are added immediately against the live render graph, so the referenced
//! nodes must already exist. Wire an edge from a plugin that runs after both
//! nodes are added — typically in `Plugin::finish`, which runs once every
//! plugin's `build` has completed.

use bevy::app::App;
use bevy::core_pipeline::core_2d::graph::Core2d;
use bevy::ecs::world::FromWorld;
use bevy::render::RenderApp;
use bevy::render::render_graph::{RenderGraphExt, RenderLabel, ViewNode, ViewNodeRunner};

/// Extension methods for adding and ordering 2D post-process render nodes.
///
/// Every method is a no-op when the app has no [`RenderApp`] (e.g. headless
/// tests), so callers don't need to guard for it.
pub trait PostProcess2dAppExt {
    /// Adds a [`ViewNode`] to the `Core2d` render graph (wrapped in a
    /// [`ViewNodeRunner`]). Order it with the `render_*` methods below.
    fn add_post_process_2d_node<N>(&mut self, label: impl RenderLabel) -> &mut Self
    where
        N: ViewNode + FromWorld + Send + Sync + 'static;

    /// Orders `node` to run after `after` (adds the edge `after → node`).
    fn render_after(&mut self, node: impl RenderLabel, after: impl RenderLabel) -> &mut Self;

    /// Orders `node` to run before `before` (adds the edge `node → before`).
    fn render_before(&mut self, node: impl RenderLabel, before: impl RenderLabel) -> &mut Self;

    /// Orders `node` to run between `after` and `before` (adds the edges
    /// `after → node → before`).
    fn render_between(
        &mut self,
        node: impl RenderLabel,
        after: impl RenderLabel,
        before: impl RenderLabel,
    ) -> &mut Self;
}

impl PostProcess2dAppExt for App {
    fn add_post_process_2d_node<N>(&mut self, label: impl RenderLabel) -> &mut Self
    where
        N: ViewNode + FromWorld + Send + Sync + 'static,
    {
        if let Some(render_app) = self.get_sub_app_mut(RenderApp) {
            render_app.add_render_graph_node::<ViewNodeRunner<N>>(Core2d, label);
        }
        self
    }

    fn render_after(&mut self, node: impl RenderLabel, after: impl RenderLabel) -> &mut Self {
        if let Some(render_app) = self.get_sub_app_mut(RenderApp) {
            render_app.add_render_graph_edge(Core2d, after, node);
        }
        self
    }

    fn render_before(&mut self, node: impl RenderLabel, before: impl RenderLabel) -> &mut Self {
        if let Some(render_app) = self.get_sub_app_mut(RenderApp) {
            render_app.add_render_graph_edge(Core2d, node, before);
        }
        self
    }

    fn render_between(
        &mut self,
        node: impl RenderLabel,
        after: impl RenderLabel,
        before: impl RenderLabel,
    ) -> &mut Self {
        let node = node.intern();
        if let Some(render_app) = self.get_sub_app_mut(RenderApp) {
            render_app.add_render_graph_edge(Core2d, after, node);
            render_app.add_render_graph_edge(Core2d, node, before);
        }
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::core_pipeline::core_2d::graph::Node2d;
    use bevy::render::render_graph::RenderLabel;

    #[derive(Debug, Clone, PartialEq, Eq, Hash, RenderLabel)]
    struct TestLabel;

    #[test]
    fn ordering_methods_are_noops_without_a_render_app() {
        let mut app = App::new();
        app.render_after(TestLabel, Node2d::EndMainPass)
            .render_before(TestLabel, Node2d::StartMainPassPostProcessing)
            .render_between(
                TestLabel,
                Node2d::EndMainPass,
                Node2d::StartMainPassPostProcessing,
            );
    }
}
