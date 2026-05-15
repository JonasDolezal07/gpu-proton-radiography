#version 450

layout(push_constant) uniform PushConstants {
    vec2 screen_size;
} pc;

layout(location = 0) in vec2 in_position;
layout(location = 1) in vec2 in_uv;
layout(location = 2) in vec4 in_color;  // R8G8B8A8_UNORM -> vec4

layout(location = 0) out vec2 out_uv;
layout(location = 1) out vec4 out_color;

void main() {
    // Convert from screen coordinates to clip space
    vec2 pos = 2.0 * in_position / pc.screen_size - 1.0;
    gl_Position = vec4(pos.x, pos.y, 0.0, 1.0);

    out_uv = in_uv;
    // sRGB to linear conversion (egui uses sRGB colors)
    out_color = vec4(pow(in_color.rgb, vec3(2.2)), in_color.a);
}
