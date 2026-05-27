# Input decks

Simulations are configured with TOML deck files. The canonical format since Phase 2.
Legacy JSON configs are still accepted but deprecated — prefer TOML for all new work.

## Creating a deck

```bash
# Start from a known-good preset
proton_tracer init zpinch -o my_run.toml        # zpinch instability
proton_tracer init kink-strong -o my_run.toml   # kink instability
proton_tracer init blank -o my_run.toml         # minimal blank template

# Check it parses before running
proton_tracer validate my_run.toml

# Inspect resolved geometry
proton_tracer explain my_run.toml
```

## Full schema

### `[field]`

```toml
[field]
path = "data/instabilities/zpinch.bfld"   # path to .bfld file (relative to deck)
e_path = "data/e_field.bfld"              # optional separate E-field file
scale_B = 1.0                             # multiplicative scale on B (default 1.0)
scale_E = 0.0                             # multiplicative scale on E (default 0.0)
```

`scale_B` and `scale_E` are applied before the simulation starts. Use them to scan field
strength without modifying the field file, or to disable one component entirely (`scale_E = 0`).

#### Superimposing multiple fields

Any number of extra field grids can be overlaid onto the primary grid using
`[[field.extra_b]]` entries.  Each extra field is resampled (CPU-side, trilinear
interpolation, zero outside its bounds) onto the primary grid before the GPU sees the data
— no shader or Vulkan changes are required.

```toml
[field]
path = "data/background.bfld"      # primary (defines the grid that the GPU sees)
scale_B = 1.0

[[field.extra_b]]
path = "data/insert_coil.bfld"     # resampled onto primary grid at load time
scale_B = 0.5                      # independent per-field scale factor
scale_E = 0.0

[[field.extra_b]]
path = "data/external_solenoid.bfld"
scale_B = 2.0
```

Rules:
- The first `[field]` entry sets the grid dimensions and bounds used by the GPU.
- Extra fields with different grids or bounds are interpolated onto the primary grid; voxels
  that fall outside an extra field's bounds contribute zero (no extrapolation).
- Each extra field can have its own `scale_B` / `scale_E`.
- `e_path` on an extra field works the same as on the primary field.
- JSON configs (legacy) do not support superimposed fields — use TOML decks.

### `[source]`

All source types share the same table, discriminated by `type`.

```toml
[source]
type = "parallel"           # parallel | disk | point | pencil
n_particles = 100000
energy_MeV = 14.7
energy_spread_percent = 0.0   # Gaussian σ = energy_MeV × energy_spread_percent / 100 (default 0)
seed = 42                     # optional RNG seed for reproducibility

# parallel-specific
direction = [1.0, 0.0, 0.0]
beam_radius_mm = 30.0
source_distance_mm = 100.0
angular_spread_deg = 0.0

# disk-specific
center_mm = [-80.0, 0.0, 0.0]
radius_um = 40.0
cone_half_angle_deg = 0.0

# point-specific
position_mm = [-80.0, 0.0, 0.0]
aim_at_mm = [0.0, 0.0, 0.0]

# pencil-specific
position_mm = [-80.0, 5.0, 0.0]
aim_at_mm = [0.0, 0.0, 0.0]
```

See [geometry.md](geometry.md) for a description of each source type.

### `[detector]`

```toml
[detector]
center_mm = [110.0, 0.0, 0.0]
normal = [1.0, 0.0, 0.0]       # beam-facing normal (default [1,0,0])
up = [0.0, 1.0, 0.0]           # for y-axis construction (default [0,1,0])
width_mm = 500.0
height_mm = 500.0
pixels = [512, 512]             # [cols, rows] (default [512,512])
```

### `[numerics]`

```toml
[numerics]
integrator = "boris"    # only boris is implemented
dt_ps = 1.0             # time step in picoseconds
max_steps = 20000       # hard cap — simulation ends when all particles exit or hit
```

The default `max_steps` is 10000 when not specified. For geometries larger than ~60 mm, or
with strong fields that increase path length, raise this. `explain` warns when the step budget
is likely insufficient.

### `[render]`

```toml
[render]
scale = "log"       # log | linear | sqrt
colormap = "rcf"    # rcf | scientific | grayscale | hot | inverted
exposure = 1.0      # multiplicative brightness
```

These control the PNG produced during `run`. The underlying count data is always saved
regardless of render settings — you can re-render later with different settings without
re-running the GPU simulation.

### `[output]`

```toml
[output]
write_raw_counts = true
write_processed_counts = true
write_png = true
write_metadata = true
```

All four default to true. Set to false to skip writing specific outputs.

### `[detector_response]` (optional)

```toml
[detector_response]
blur_sigma_um = 0.0       # Gaussian PSF blur on detector (0 = no blur)
background_counts = 0.0   # additive background
poisson_noise = false     # apply Poisson shot noise
noise_seed = 0            # RNG seed for noise (0 = random)
```

Default: identity (no blur, no noise, no background). Apply these to model detector physics
or film response when comparing to experimental data.

---

## `--set` overrides

Override any deck parameter at run time without editing the file:

```bash
proton_tracer run zpinch.toml --set source.energy_MeV=10
proton_tracer run zpinch.toml --set numerics.max_steps=30000
proton_tracer run zpinch.toml --set field.scale_B=2.0
```

Multiple overrides:
```bash
proton_tracer run zpinch.toml \
  --set source.energy_MeV=10 \
  --set numerics.dt_ps=0.5
```

Overrides are applied before SI unit conversion and recorded in `resolved_config.json` so the
run directory is still self-documenting.

Supported keys match the sweep parameter list — see [sweeps.md](sweeps.md#supported-parameters).

---

## Legacy JSON

JSON configs from earlier versions are still accepted. `SimConfig::load()` dispatches by file
extension (`.json` vs `.toml`). The JSON format uses flat keys (`dt_ps`, `max_steps`) rather
than nested TOML tables.

JSON configs are not recommended for new work. Use `proton_tracer init` to generate a TOML deck.
