#version 450

// 3D detector plane vertex shader
// Positions a quad in world space based on detector parameters

layout(location = 0) out vec2 uv;

// Detector parameters (passed via push constants)
layout(push_constant) uniform DetectorParams {
    mat4 view_proj;           // Camera view-projection matrix
    vec4 detector_pos;        // Detector center position (world space)
    vec4 detector_normal;     // Detector facing direction
    vec4 detector_extent;     // Half-size in x, y (local space)
    // Display params follow...
    float max_count;
    float gamma;
    float exposure;
    uint use_log_scale;
} params;

// Generate quad vertices (2 triangles, 6 vertices)
// Vertex order: 0,1,2 and 2,1,3 for two triangles
const vec2 positions[6] = vec2[](
    vec2(-1.0, -1.0),  // 0: bottom-left
    vec2( 1.0, -1.0),  // 1: bottom-right
    vec2(-1.0,  1.0),  // 2: top-left
    vec2(-1.0,  1.0),  // 2: top-left
    vec2( 1.0, -1.0),  // 1: bottom-right
    vec2( 1.0,  1.0)   // 3: top-right
);

void main() {
    vec2 pos2d = positions[gl_VertexIndex];

    // UV coordinates [0,1] for texture sampling
    uv = pos2d * 0.5 + 0.5;

    // Build local coordinate system from detector normal
    vec3 normal = normalize(params.detector_normal.xyz);

    // Find perpendicular vectors (right and up)
    vec3 up = abs(normal.y) < 0.99 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    vec3 right = normalize(cross(up, normal));
    up = cross(normal, right);

    // Scale by detector extent
    vec3 local_pos = right * pos2d.x * params.detector_extent.x
                   + up * pos2d.y * params.detector_extent.y;

    // World position
    vec3 world_pos = params.detector_pos.xyz + local_pos;

    // Transform to clip space
    gl_Position = params.view_proj * vec4(world_pos, 1.0);
}
