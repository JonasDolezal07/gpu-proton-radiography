# Coordinate system and geometry

## Arbitrary geometry

Source positions, beam directions, and detector orientations are fully configurable —
there is no requirement to align anything with +x.  The examples throughout this page
use the conventional +x beam axis for clarity, but the engine is geometry-independent.

See [Tilted-geometry example](#tilted-geometry-example) below for a complete off-axis deck.

## Default beam axis convention

The default configuration places the beam along **+x**:

- Source is upstream: x < 0
- Detector is downstream: x > 0
- The detector plane is y–z

This is just a convention used by the built-in presets and examples, not a constraint of the engine.

## Source geometry

Four source types are available. Select with `[source] type = "..."`.

### `parallel`

A uniform parallel beam — all particles travel in `direction`, starting from a disk of radius
`beam_radius_mm` centred at `source_distance_mm` upstream of the field volume centre.

Use for: laser-driven proton sources in the far field, laboratory collimated beams.

```toml
[source]
type = "parallel"
direction = [1.0, 0.0, 0.0]
beam_radius_mm = 30.0
source_distance_mm = 100.0
energy_MeV = 14.7
n_particles = 100000
```

### `disk`

A point-like source (disk of finite radius `radius_um` at world position `center_mm`) emitting
a cone of protons aimed at `direction`. Models TNSA sources with a finite emission spot.

```toml
[source]
type = "disk"
center_mm = [-80.0, 0.0, 0.0]
direction = [1.0, 0.0, 0.0]
radius_um = 40.0
cone_half_angle_deg = 0.0
energy_MeV = 14.7
n_particles = 1000000
```

### `point`

Isotropic or cone-restricted emission from a single world-space point. Use for point-source
sensitivity tests or cases where the source radius is negligible.

```toml
[source]
type = "point"
position_mm = [-80.0, 0.0, 0.0]
aim_at_mm = [0.0, 0.0, 0.0]
cone_half_angle_deg = 5.0
energy_MeV = 14.7
n_particles = 100000
```

### `pencil`

A single pencil beam at position `position_mm`, aimed at `aim_at_mm`. Use for single-ray
tracing and geometry debugging.

```toml
[source]
type = "pencil"
position_mm = [-80.0, 5.0, 0.0]
aim_at_mm = [0.0, 0.0, 0.0]
energy_MeV = 14.7
n_particles = 1000
```

## Detector geometry

The detector is a rectangular plane defined by:

| Field | Meaning | Default |
|---|---|---|
| `center_mm` | World-space centre | required |
| `normal` | Outward normal (beam-facing) | `[1, 0, 0]` |
| `up` | Up direction for y-axis construction | `[0, 1, 0]` |
| `width_mm` | Physical width | required |
| `height_mm` | Physical height | required |
| `pixels` | Pixel grid `[cols, rows]` | `[512, 512]` |

### Detector-local axes

The detector constructs a local coordinate system from `normal` and `up` automatically.
If `up` is not perpendicular to `normal`, it is projected onto the detector plane first
(Gram–Schmidt):

```
u_y = normalize(up − dot(up, normal) · normal)
v_z = cross(normal, u_y)
```

With the defaults (`normal = [1,0,0]`, `up = [0,1,0]`):
- u\_y = [0, 1, 0]   (detector y is world y)
- v\_z = [0, 0, 1]   (detector z is world z)

### CSV hit positions

Hit positions are reported as `y_mm, z_mm` in detector-local coordinates. In the default
geometry, y\_mm corresponds to world y and z\_mm to world z.

**Pixel mapping:** row index corresponds to z, column index to y.

## Magnification

For a point-like source at distance `L_s` upstream of the field and a detector at distance
`L_d` downstream:

```
M = (L_s + L_d) / L_s
```

The radiograph is magnified by M relative to the plasma. `explain` prints this value.

## Step budget

The integrator must take enough steps to traverse the full source-to-detector path. The
minimum straight-line step count is:

```
min_steps = (source_to_detector distance) / (v × dt)
```

Strong field deflections increase path length, so you need margin above this — typically 25–50%.
`explain` prints a warning when `max_steps` is below the straight-line minimum.

The default `max_steps = 10000` with `dt_ps = 1.0` is adequate for most geometries within
~60 mm field volumes. Increase `max_steps` for larger geometries or strongly deflecting fields.

## Tilted-geometry example

The following deck models an OMEGA-style geometry: the plasma column is aligned with z,
the proton source enters at an oblique angle, and the detector is tilted.

```toml
# OMEGA-style tilted geometry
[field]
path = "data/instabilities/zpinch.bfld"

[source]
type              = "point"
position_mm       = [-60.0,  20.0, -80.0]   # off-axis source
aim_at_mm         = [  0.0,   0.0,   0.0]   # aimed at plasma centre
cone_half_angle_deg = 8.0
energy_MeV        = 14.7
n_particles       = 500000

[detector]
# Detector placed downstream of the field, rotated 20° away from +x
center_mm  = [120.0, -15.0, 80.0]
normal     = [-0.940,  0.116, -0.320]   # points back toward source (unit vector)
up         = [  0.0,   1.0,   0.0]
width_mm   = 400.0
height_mm  = 400.0

[numerics]
dt_ps     = 0.2
max_steps = 30000
```

Key points:

- `aim_at_mm` computes the `direction` vector automatically from `position_mm` → target.
- `normal` should point **toward the source** (the half-space the beam comes from).  If it
  points away, no hits will be recorded because particles cross from the wrong side.
- `up` only needs to be non-parallel to `normal`; it is Gram–Schmidt-projected onto the
  detector plane automatically.
- Hit positions in `hits.bin` and the CSV are in **detector-local coordinates** (`u_y`, `v_z`),
  regardless of world orientation.  Multiply by the detector pixel pitch to convert to mm on film.
