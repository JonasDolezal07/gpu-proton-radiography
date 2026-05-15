# Proton Radiography Tracer - Architecture

## Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         Python Frontend                          │
│  - Load/generate fields (field_format.py)                       │
│  - Configure source (source_config.py)                          │
│  - Export binary data                                           │
│  - Post-processing & analysis                                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ Binary files + JSON config
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         Rust Backend                             │
│  - Load field data into GPU texture                             │
│  - Load particle data into GPU buffer                           │
│  - Dispatch compute shaders                                     │
│  - Render 3D visualization                                      │
│  - Handle user input (camera, params)                           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ Metal / Vulkan
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                            GPU                                   │
│                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐                    │
│  │ Compute Shader  │    │ Render Pipeline │                    │
│  │                 │    │                 │                    │
│  │ - Boris integr  │    │ - Field volume  │                    │
│  │ - Field sample  │    │ - Trajectories  │                    │
│  │ - Detector hit  │    │ - Detector quad │                    │
│  └─────────────────┘    └─────────────────┘                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow

### 1. Python → Binary Files

**Field data** (`*.bin`):
```
Header (64 bytes):
  - magic: "BFLD" (4 bytes)
  - version: u32
  - nx, ny, nz: 3× u32
  - bounds: 6× f32 (x_min, x_max, y_min, y_max, z_min, z_max)
  - padding

Data:
  - B-field: nx × ny × nz × 3 × f32 (contiguous, row-major)
```

**Particle data** (`*.bin`):
```
Header (32 bytes):
  - magic: "PRTC" (4 bytes)
  - version: u32
  - n_particles: u32
  - padding

Data:
  - positions: n × 3 × f32
  - velocities: n × 3 × f32
```

**Config** (`*.json`):
```json
{
  "field_path": "field.bin",
  "source": { ... },
  "dt": 1e-12,
  "max_steps": 5000,
  "detector_bins": [512, 512]
}
```

### 2. Rust GPU Buffers

```rust
// Field as 3D texture (for hardware trilinear interpolation)
let field_texture: Texture3D<f32x4>;  // RGBA = Bx, By, Bz, |B|

// Particle state (double-buffered for ping-pong)
struct ParticleState {
    position: [f32; 3],
    velocity: [f32; 3],
    active: u32,
    _pad: u32,
}
let particles_a: Buffer<ParticleState>;
let particles_b: Buffer<ParticleState>;

// Detector accumulator
let detector: Texture2D<u32>;  // Atomic histogram
```

### 3. Compute Shader (Boris Integrator)

```metal
kernel void boris_step(
    texture3d<float> field [[texture(0)]],
    device ParticleState* particles_in [[buffer(0)]],
    device ParticleState* particles_out [[buffer(1)]],
    device atomic_uint* detector [[buffer(2)]],
    constant SimParams& params [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    ParticleState p = particles_in[id];
    if (!p.active) return;

    // Sample field (hardware trilinear)
    float3 B = field.sample(sampler, normalize_pos(p.position)).xyz;

    // Boris algorithm
    float3 t = (Q_OVER_M * params.dt * 0.5) * B;
    float3 s = 2.0 * t / (1.0 + dot(t, t));
    float3 v_minus = p.velocity;
    float3 v_prime = v_minus + cross(v_minus, t);
    float3 v_plus = v_minus + cross(v_prime, s);

    p.velocity = v_plus;
    p.position += v_plus * params.dt;

    // Check detector hit
    if (crossed_detector(p.position, params.detector_plane)) {
        int2 bin = detector_coords(p.position, params);
        atomic_fetch_add_explicit(&detector[bin], 1, memory_order_relaxed);
        p.active = 0;
    }

    particles_out[id] = p;
}
```

### 4. Render Pipeline

**Field visualization** (volume rendering):
- Ray march through field volume
- Color by |B| (log scale)
- Semi-transparent, adjustable density

**Particle trajectories**:
- Store trajectory history in ring buffer (last N positions per particle)
- Render as line strips or points
- Color by energy or deflection

**Detector plane**:
- Textured quad at detector position
- Radiograph texture updated from histogram
- Grayscale or colormap

### 5. 3D View Controls

- Orbit camera (mouse drag)
- Zoom (scroll)
- Pan (middle mouse)
- Keyboard:
  - Space: pause/play simulation
  - R: reset particles
  - F: toggle field viz
  - T: toggle trajectories
  - D: toggle detector

## Directory Structure

```
proton_tracer/
├── python/
│   ├── field_format.py      # Field loading/generation
│   ├── source_config.py     # Source configuration
│   ├── export.py            # Export to binary
│   └── analysis.py          # Post-processing
│
├── rust/
│   ├── Cargo.toml
│   └── src/
│       ├── main.rs          # Entry point, window setup
│       ├── app.rs           # Application state
│       ├── gpu/
│       │   ├── mod.rs
│       │   ├── compute.rs   # Boris integrator dispatch
│       │   ├── render.rs    # 3D visualization
│       │   └── buffers.rs   # GPU buffer management
│       ├── loaders/
│       │   ├── mod.rs
│       │   ├── field.rs     # Load field binary
│       │   └── particles.rs # Load particle binary
│       └── camera.rs        # 3D camera controller
│
├── shaders/
│   ├── boris.metal          # Compute shader
│   ├── field_viz.metal      # Volume rendering
│   ├── trajectory.metal     # Line rendering
│   └── detector.metal       # Detector quad
│
└── data/
    ├── fields/              # Exported field binaries
    └── configs/             # Simulation configs
```

## Performance Targets

| Metric | Target |
|--------|--------|
| Particles | 500k - 1M |
| Integration steps/frame | 10-100 |
| Frame rate | 60 fps |
| Latency (param change → visible) | < 100ms |

## Dependencies

**Rust:**
- `wgpu` or `metal-rs` (GPU abstraction)
- `winit` (windowing)
- `bytemuck` (safe transmutes)
- `glam` (math)
- `serde` + `serde_json` (config loading)

**Python:**
- `numpy`
- `h5py` (GORGON loading)
- `matplotlib` (preview/analysis)
