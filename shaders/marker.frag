#version 450

// Billboard marker fragment shader
// Draws a glowing sphere/circle

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 fragColor;

layout(push_constant) uniform MarkerParams {
    mat4 view_proj;
    vec4 position;
    vec4 color;
} params;

void main() {
    float dist = length(uv);

    // Discard outside circle
    if (dist > 1.0) {
        discard;
    }

    // Soft edge with glow
    float alpha = 1.0 - smoothstep(0.6, 1.0, dist);

    // Add slight 3D shading (fake sphere lighting)
    float shade = 1.0 - dist * 0.3;

    // Glow effect
    float glow = exp(-dist * 2.0) * 0.5;

    vec3 col = params.color.rgb * shade + vec3(1.0) * glow;
    fragColor = vec4(col, alpha * params.color.a);
}
