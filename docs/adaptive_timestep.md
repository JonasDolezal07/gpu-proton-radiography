# Adaptive timestep

By default prad automatically chooses a small timestep inside the field and a larger one
in vacuum on either side. For typical geometries this reduces the total number of GPU
dispatches by 5–20× with no change to the physics.

---

## The problem with a fixed timestep

A proton radiography geometry has three dynamically distinct regions:

1. **Pre-field vacuum** — source to field entry. No forces. Any step size is physically valid;
   small steps here are pure waste.
2. **Inside the field** — the physical region of interest. Steps must resolve the Larmor
   radius and the field grid spacing.
3. **Post-field vacuum** — field exit to detector. Again, no forces; small steps waste time.

With a fixed `dt_ps` tuned for accuracy inside the field, the vast majority of steps
(often > 95%) are spent in vacuum where nothing physically interesting happens.

---

## The three-phase schedule

When no explicit `dt_ps` is supplied in `[numerics]`, prad computes a schedule at load time
based on the source position, beam direction, and field bounds:

```
  source              field entry         field exit           detector
    │                     │                   │                    │
    ●──── dt_large ────────●─── dt_small ───── ●──── dt_large ──── ◎
    │← t_entry_s (vacuum) →│← field transit   →│← post-field vacuum→│
```

**dt_small** governs integration accuracy inside the field. Two constraints are applied, and
the tighter one wins:

| Constraint | Formula | Rationale |
|---|---|---|
| Larmor criterion | $dt_\text{small} \leq T_\text{Larmor} / 20$ | Resolves the Larmor orbit to 1/20th of a revolution |
| Grid-crossing criterion | $dt_\text{small} \leq \Delta x_\text{min} / (4v)$ | No particle crosses more than one voxel per 4 steps |

where $T_\text{Larmor} = 2\pi / (\vert q/m \vert \cdot B_\text{max})$ and $\Delta x_\text{min}$
is the smallest grid cell dimension.

**dt_large** governs the vacuum regions. It is the minimum of:

- $20 \times dt_\text{small}$ — keeps the ratio bounded so the field crossing is not missed
- $t_\text{entry} / 10$ — at least 10 steps before the field, so the entry transition is clean

**t_exit** gets a 30% safety margin over the geometric field-transit time to account for
deflected paths that spend longer inside the field than a straight-line estimate predicts.

---

## Phase boundary detection

The entry and exit times are computed from a ray-slab intersection of the beam direction
against the field bounding box:

```
t_entry = time for beam to travel from source to nearest field face
t_exit  = time to travel to farthest field face × 1.30
```

Both are computed in simulated time coordinates and compared to the accumulated simulation
time `t_sim` at each GPU dispatch. The schedule selection is:

```
dt = dt_small   if t_entry ≤ t_sim ≤ t_exit
dt = dt_large   otherwise
```

This is evaluated on the CPU before each dispatch and pushed as part of `SimParams.dt`.

---

## Speedup estimate

The approximate speedup over a fixed-dt run is:

$$
\text{speedup} \approx \frac{dt_\text{large}}{dt_\text{small}}
$$

which is capped at 20× by construction. In practice:

| Geometry | Source distance | dt ratio | Typical speedup |
|---|---|---|---|
| zpinch, source 80 mm | 80 mm from field | ~20× | 15–20× |
| zpinch, source 200 mm | 200 mm from field | capped at $t_\text{entry}/10$ | 5–10× |
| Source touching field | 0 mm | 1× (dt_small throughout) | 1× |

The speedup is most pronounced for large source-to-field distances relative to the field
transit length.

---

## Configuration

### Automatic (recommended)

Omit `dt_ps` from `[numerics]`:

```toml
[numerics]
max_steps = 25000
```

prad will log the computed schedule at startup:

```
dt adaptive: large=4.21e-12 s  small=2.11e-13 s  (≈20.0× speedup in vacuum)
  t_entry=3.81e-10 s  t_exit=4.19e-10 s
```

### Fixed (override)

Supply `dt_ps` to disable the adaptive schedule and use a single timestep throughout:

```toml
[numerics]
dt_ps     = 0.2
max_steps = 25000
```

Use fixed dt when:
- Benchmarking against a reference run
- The source is already inside or immediately adjacent to the field
- You need bit-exact reproducibility across different geometries

---

## Interaction with `max_steps`

The step budget (`max_steps`) counts individual integration steps, not dispatches. With the
adaptive schedule the vacuum regions now consume far fewer steps, so the same `max_steps`
covers a much larger total simulated time.

**Rule of thumb with adaptive dt:** `max_steps` needs to cover only the field-transit steps
(where dt is small). The vacuum budget is provided essentially for free. A value of 10 000–
20 000 is usually sufficient for most geometries.

With fixed dt, follow the guidance in `explain` which estimates the minimum required steps
from source-to-detector distance:

```bash
proton_tracer explain my_run.toml
```

---

## Validation — test15

A pencil beam in uniform B_z = 1 T is run twice: once with `dt_ps = 1.0` (fixed) and once
without (adaptive). The adaptive schedule selects dt_large ≈ 1 ns in vacuum and
dt_small ≈ 1 ps inside the 12 cm field cube.

Checks:
- Both runs produce ≥ 10 000 hits
- Mean y-deflection agrees to < 1 mm (same physics, same orbit)
- Energy conservation holds in both: std/mean < 10⁻⁴ (B-only → no energy loss)

Result: Δmean_y = 0.73 mm at 20× speedup.

---

## When not to use adaptive dt

- **Source inside or touching the field.** `t_entry ≈ 0`, so `dt_large = dt_small` and there is no
  benefit.
- **Very short vacuum gaps.** If the source-to-field distance is < 10 × dt_small × v, the phase
  transition happens in the first few steps and speedup is negligible.
- **Comparing to a reference simulation.** Fixed dt gives identical results independent of geometry;
  adaptive dt introduces a rounding difference (< 0.1% on deflection angle) from the phase
  transition point.
