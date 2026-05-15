#version 450

layout(binding = 0) uniform sampler2D font_texture;

layout(location = 0) in vec2 in_uv;
layout(location = 1) in vec4 in_color;

layout(location = 0) out vec4 out_color;

void main() {
    vec4 tex_color = texture(font_texture, in_uv);
    // Premultiplied alpha output
    out_color = in_color * tex_color;
}
