# Coordinate system and geometry

## Beam axis convention

**+x is the beam axis throughout.**

- Source is upstream: x < 0
- Detector is downstream: x > 0
- The detector plane is y–z

This convention is consistent across the config, shader, and CSV output. Do not assume any
other axis orientation.

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
