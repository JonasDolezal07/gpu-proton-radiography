#version 450

// Display detector radiograph texture
// Reads from R32_UINT storage image, applies colormap

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 out_color;

// Detector hit counts (R32_UINT)
layout(binding = 0) uniform usampler2D detector_texture;

// Detector parameters (matches Detector3DParams in Rust)
layout(push_constant) uniform DetectorParams {
    mat4 view_proj;           // Camera view-projection matrix
    vec4 detector_pos;        // Detector center position
    vec4 detector_normal;     // Detector facing direction
    vec4 detector_extent;     // Half-size
    // Display params
    float max_count;          // For normalization
    float gamma;              // Gamma correction (e.g., 0.5 for sqrt)
    float exposure;           // Brightness multiplier
    uint  use_log_scale;      // 1 = log scale, 0 = linear
    uint  colormap_mode;      // 0 = RCF film, 1 = scientific (dark->light)
} params;

// Radiochromic Film (RCF) colormap - mimics EBT3 Gafchromic film
// Real film: unexposed = light cyan/green, increasing dose = darker blue/purple
// This is INVERTED compared to typical "hot" colormaps
vec3 colormap_rcf(float t) {
    t = clamp(t, 0.0, 1.0);

    // EBT3 Gafchromic film response:
    // - Unexposed: light cyan-ish (RGB ~0.7, 0.85, 0.8)
    // - Low dose: cyan-blue
    // - Medium dose: blue-purple
    // - High dose: dark purple/magenta
    // - Very high (saturation): dark magenta/brown

    vec3 c;
    if (t < 0.1) {
        // Very low exposure: cyan tint (unexposed film base)
        float s = t / 0.1;
        c = mix(vec3(0.75, 0.88, 0.82), vec3(0.6, 0.78, 0.8), s);
    } else if (t < 0.3) {
        // Low exposure: cyan to blue
        float s = (t - 0.1) / 0.2;
        c = mix(vec3(0.6, 0.78, 0.8), vec3(0.4, 0.55, 0.75), s);
    } else if (t < 0.5) {
        // Medium exposure: blue to blue-purple
        float s = (t - 0.3) / 0.2;
        c = mix(vec3(0.4, 0.55, 0.75), vec3(0.35, 0.35, 0.65), s);
    } else if (t < 0.7) {
        // Higher exposure: purple
        float s = (t - 0.5) / 0.2;
        c = mix(vec3(0.35, 0.35, 0.65), vec3(0.4, 0.25, 0.5), s);
    } else if (t < 0.9) {
        // High exposure: dark magenta
        float s = (t - 0.7) / 0.2;
        c = mix(vec3(0.4, 0.25, 0.5), vec3(0.35, 0.18, 0.35), s);
    } else {
        // Saturation: very dark (film burned through)
        float s = (t - 0.9) / 0.1;
        c = mix(vec3(0.35, 0.18, 0.35), vec3(0.2, 0.1, 0.15), s);
    }

    return c;
}

// Scientific colormap (dark->light, good for screen viewing)
vec3 colormap_scientific(float t) {
    t = clamp(t, 0.0, 1.0);

    vec3 c;
    if (t < 0.25) {
        float s = t / 0.25;
        c = mix(vec3(0.02, 0.02, 0.05), vec3(0.1, 0.2, 0.6), s);
    } else if (t < 0.5) {
        float s = (t - 0.25) / 0.25;
        c = mix(vec3(0.1, 0.2, 0.6), vec3(0.2, 0.6, 0.8), s);
    } else if (t < 0.75) {
        float s = (t - 0.5) / 0.25;
        c = mix(vec3(0.2, 0.6, 0.8), vec3(0.9, 0.9, 0.95), s);
    } else {
        float s = (t - 0.75) / 0.25;
        c = mix(vec3(0.9, 0.9, 0.95), vec3(1.0, 1.0, 0.7), s);
    }

    return c;
}

void main() {
    // Sample detector texture (uint)
    uint count = texture(detector_texture, uv).r;

    // Normalize
    float value;
    if (params.use_log_scale != 0 && count > 0) {
        value = log(float(count) + 1.0) / log(params.max_count + 1.0);
    } else {
        value = float(count) / params.max_count;
    }

    // Apply exposure and gamma
    value = pow(clamp(value * params.exposure, 0.0, 1.0), params.gamma);

    // Apply colormap based on mode:
    // 0 = RCF film, 1 = scientific, 2 = grayscale, 3 = hot, 4 = inverted grayscale
    vec3 color;
    if (params.colormap_mode == 0) {
        color = colormap_rcf(value);
    } else if (params.colormap_mode == 1) {
        color = colormap_scientific(value);
    } else if (params.colormap_mode == 2) {
        // Grayscale: 0 = black, 1 = white
        float v = clamp(value, 0.0, 1.0);
        color = vec3(v, v, v);
    } else if (params.colormap_mode == 3) {
        // Hot: black → red → yellow → white
        float t = clamp(value, 0.0, 1.0);
        vec3 c;
        if (t < 1.0 / 3.0)      c = mix(vec3(0.0), vec3(1.0, 0.0, 0.0), t * 3.0);
        else if (t < 2.0 / 3.0) c = mix(vec3(1.0, 0.0, 0.0), vec3(1.0, 1.0, 0.0), (t - 1.0/3.0) * 3.0);
        else                     c = mix(vec3(1.0, 1.0, 0.0), vec3(1.0, 1.0, 1.0), (t - 2.0/3.0) * 3.0);
        color = c;
    } else {
        // Inverted grayscale: 0 = white, 1 = black
        float v = 1.0 - clamp(value, 0.0, 1.0);
        color = vec3(v, v, v);
    }

    // Alpha: RCF and inverted grayscale are fully opaque; others fade low-hit regions
    float alpha;
    if (params.colormap_mode == 0 || params.colormap_mode == 4) {
        alpha = 1.0;
    } else {
        alpha = smoothstep(0.0, 0.1, value);
    }

    out_color = vec4(color, alpha);
}
