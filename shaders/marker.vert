#version 450

// Billboard marker vertex shader
// Draws a camera-facing quad at a world position

layout(location = 0) out vec2 uv;

layout(push_constant) uniform MarkerParams {
    mat4 view_proj;
    vec4 position;     // World position (xyz) + size (w)
    vec4 color;        // RGBA color
} params;

// Quad vertices
const vec2 positions[6] = vec2[](
    vec2(-1.0, -1.0),
    vec2( 1.0, -1.0),
    vec2(-1.0,  1.0),
    vec2(-1.0,  1.0),
    vec2( 1.0, -1.0),
    vec2( 1.0,  1.0)
);

void main() {
    vec2 pos2d = positions[gl_VertexIndex];
    uv = pos2d;

    // Get camera right and up from inverse view matrix columns
    // view_proj = proj * view, so we extract from the transpose
    vec3 right = vec3(params.view_proj[0][0], params.view_proj[1][0], params.view_proj[2][0]);
    vec3 up = vec3(params.view_proj[0][1], params.view_proj[1][1], params.view_proj[2][1]);

    // Normalize (they may be scaled by projection)
    right = normalize(right);
    up = normalize(up);

    float size = params.position.w;
    vec3 world_pos = params.position.xyz
        + right * pos2d.x * size
        + up * pos2d.y * size;

    gl_Position = params.view_proj * vec4(world_pos, 1.0);
}
