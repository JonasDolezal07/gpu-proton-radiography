#version 450

// Full-screen triangle - no vertex buffer needed
// Generates a triangle that covers the entire screen

layout(location = 0) out vec2 uv;

void main() {
    // Generate vertices for a full-screen triangle
    // Vertex 0: (-1, -1), Vertex 1: (3, -1), Vertex 2: (-1, 3)
    vec2 positions[3] = vec2[](
        vec2(-1.0, -1.0),
        vec2( 3.0, -1.0),
        vec2(-1.0,  3.0)
    );

    vec2 pos = positions[gl_VertexIndex];
    gl_Position = vec4(pos, 0.0, 1.0);

    // UV coordinates: (0,0) bottom-left to (1,1) top-right
    uv = pos * 0.5 + 0.5;
}
