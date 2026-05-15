#version 450

// Volume rendering of 3D magnetic field texture
// Ray marches through the field, accumulating color based on |B|

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 out_color;

// 3D field texture (Bx, By, Bz, 0)
layout(binding = 0) uniform sampler3D field_texture;

// Camera and volume parameters
layout(push_constant) uniform VolumeParams {
    mat4 inv_view_proj;     // Inverse view-projection matrix
    mat4 view_proj;         // View-projection matrix (for depth calculation)
    vec4 camera_pos;        // Camera position in world space
    vec4 volume_min;        // Volume AABB min (world space)
    vec4 volume_max;        // Volume AABB max (world space)
    float step_size;        // Ray march step size
    float density_scale;    // Opacity multiplier
    float brightness;       // Color brightness
    uint num_steps;         // Max ray march steps
} params;

// Ray-box intersection (returns t_min, t_max)
// Returns false if no intersection
bool ray_box_intersect(vec3 ray_origin, vec3 ray_dir, vec3 box_min, vec3 box_max,
                       out float t_min, out float t_max) {
    vec3 inv_dir = 1.0 / ray_dir;
    vec3 t0 = (box_min - ray_origin) * inv_dir;
    vec3 t1 = (box_max - ray_origin) * inv_dir;

    vec3 t_near = min(t0, t1);
    vec3 t_far = max(t0, t1);

    t_min = max(max(t_near.x, t_near.y), t_near.z);
    t_max = min(min(t_far.x, t_far.y), t_far.z);

    return t_max > max(t_min, 0.0);
}

// Transfer function: |B| -> color
vec3 field_to_color(float magnitude) {
    // Blue (low) -> Cyan -> White -> Yellow (high)
    float t = clamp(magnitude, 0.0, 1.0);

    vec3 c;
    if (t < 0.25) {
        float s = t / 0.25;
        c = mix(vec3(0.0, 0.0, 0.2), vec3(0.0, 0.3, 0.8), s);
    } else if (t < 0.5) {
        float s = (t - 0.25) / 0.25;
        c = mix(vec3(0.0, 0.3, 0.8), vec3(0.2, 0.7, 0.9), s);
    } else if (t < 0.75) {
        float s = (t - 0.5) / 0.25;
        c = mix(vec3(0.2, 0.7, 0.9), vec3(0.9, 0.9, 1.0), s);
    } else {
        float s = (t - 0.75) / 0.25;
        c = mix(vec3(0.9, 0.9, 1.0), vec3(1.0, 1.0, 0.5), s);
    }

    return c;
}

// Transfer function: |B| -> opacity
float field_to_opacity(float magnitude) {
    // Low field = transparent, high field = opaque
    float t = clamp(magnitude, 0.0, 1.0);
    // Use smooth ramp with threshold
    return smoothstep(0.05, 0.5, t) * params.density_scale;
}

void main() {
    // Calculate ray direction from screen UV
    // Convert UV from [0,1] to NDC [-1,1]
    vec2 ndc = uv * 2.0 - 1.0;

    // Reconstruct world position from NDC (at near and far planes)
    vec4 near_world = params.inv_view_proj * vec4(ndc, -1.0, 1.0);
    vec4 far_world = params.inv_view_proj * vec4(ndc, 1.0, 1.0);
    near_world /= near_world.w;
    far_world /= far_world.w;

    vec3 ray_origin = params.camera_pos.xyz;
    vec3 ray_dir = normalize(far_world.xyz - near_world.xyz);

    // Intersect ray with volume bounding box
    float t_min, t_max;
    if (!ray_box_intersect(ray_origin, ray_dir, params.volume_min.xyz, params.volume_max.xyz, t_min, t_max)) {
        // No intersection - output transparent and far depth
        out_color = vec4(0.0);
        gl_FragDepth = 1.0;
        return;
    }

    // Clamp to volume bounds
    t_min = max(t_min, 0.0);

    // Calculate depth at volume entry point
    // Transform world position to clip space
    vec3 entry_pos = ray_origin + ray_dir * t_min;
    vec4 clip_pos = params.view_proj * vec4(entry_pos, 1.0);
    // Convert to normalized device coordinates and then to depth [0, 1]
    float depth = (clip_pos.z / clip_pos.w) * 0.5 + 0.5;
    gl_FragDepth = clamp(depth, 0.0, 1.0);

    // Ray march through volume (front-to-back)
    vec3 accumulated_color = vec3(0.0);
    float accumulated_alpha = 0.0;

    float t = t_min;
    vec3 volume_size = params.volume_max.xyz - params.volume_min.xyz;

    for (uint i = 0; i < params.num_steps && t < t_max; i++) {
        vec3 pos = ray_origin + ray_dir * t;

        // Convert world position to texture coordinates [0, 1]
        vec3 tex_coord = (pos - params.volume_min.xyz) / volume_size;

        // Sample field texture
        vec3 B = texture(field_texture, tex_coord).xyz;
        float magnitude = length(B);

        // Normalize magnitude (assume max field ~1T, adjust as needed)
        float norm_mag = magnitude * params.brightness;

        // Apply transfer function
        vec3 sample_color = field_to_color(norm_mag);
        float sample_alpha = field_to_opacity(norm_mag);

        // Front-to-back compositing
        float weight = sample_alpha * (1.0 - accumulated_alpha);
        accumulated_color += sample_color * weight;
        accumulated_alpha += weight;

        // Early termination if nearly opaque
        if (accumulated_alpha > 0.95) {
            break;
        }

        t += params.step_size;
    }

    out_color = vec4(accumulated_color, accumulated_alpha);
}
