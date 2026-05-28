// Black Hole Vertex Shader
// Based on Eric Bruneton's black_hole_shader (BSD-3-Clause)
// https://github.com/ebruneton/black_hole_shader

#version 450

layout(location = 0) in vec3 Vertex_Position;
layout(location = 1) in vec3 Vertex_Normal;
layout(location = 2) in vec2 Vertex_Uv;

layout(location = 0) out vec2 v_Uv;
layout(location = 1) out vec3 v_WorldPosition;

layout(set = 0, binding = 0) uniform View {
    mat4 ViewProj;
    mat4 View;
    mat4 InverseView;
    mat4 Projection;
    vec3 WorldPosition;
    float _padding;
    float width;
    float height;
};

layout(set = 2, binding = 0) uniform Mesh {
    mat4 Model;
    mat4 InverseTransposeModel;
    uint flags;
};

void main() {
    v_Uv = Vertex_Uv;
    vec4 world_position = Model * vec4(Vertex_Position, 1.0);
    v_WorldPosition = world_position.xyz;
    gl_Position = ViewProj * world_position;
}
