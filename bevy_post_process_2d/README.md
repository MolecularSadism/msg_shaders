# bevy_post_process_2d

Thin ergonomics for ordering 2D full-screen post-process passes in
[Bevy](https://bevyengine.org)'s `Core2d` render graph.

A post-process pass is the standard Bevy pattern: a `ViewNode` that reads the
camera's `ViewTarget` via `post_process_write` and draws a full-screen triangle
(see Bevy's `custom_post_processing` example). This crate adds nothing to that
node — it provides one extension trait, `PostProcess2dAppExt`, so a plugin can
add its node to the graph and wire its ordering edges in one place:

```rust,ignore
use bevy::core_pipeline::core_2d::graph::Node2d;
use bevy_post_process_2d::PostProcess2dAppExt;

app.add_post_process_2d_node::<MyEffectNode>(MyEffectLabel);
app.render_between(
    MyEffectLabel,
    Node2d::EndMainPass,
    Node2d::StartMainPassPostProcessing,
);
```

`render_after` / `render_before` / `render_between` add edges immediately
against the live render graph, so the referenced nodes must already exist —
wire edges from a plugin that runs after both nodes are added, typically in
`Plugin::finish`. Every method is a no-op when the app has no `RenderApp`
(e.g. headless tests), so callers don't need to guard for it.

## Compatibility

| bevy_post_process_2d | Bevy | Rust |
|----------------------|------|------|
| 0.1 | 0.18 | 1.85+ (edition 2024) |

## License

Dual-licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at
your option.

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in this work, as defined in the Apache-2.0 license, shall be
dual-licensed as above, without any additional terms or conditions.
