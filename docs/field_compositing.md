# Field compositing

prad can superimpose any number of `.bfld` field grids onto the primary field at load time.
Each extra grid has its own scale factors and can have a different spatial extent or resolution
than the primary. Compositing is CPU-side and transparent to the GPU — no shader or Vulkan
changes are required.

---

## Motivation

Most experimental geometries are not a single monolithic field. Common cases:

- **Background field + plasma insert.** A background solenoid or Helmholtz coil provides a
  uniform base field; the plasma-generated turbulent or structured field is overlaid.
- **MHD simulation + external coil.** A simulated instability field superimposed on a
  separately-calculated external drive field.
- **Field strength scans.** One grid provides the spatial structure; `scale_B` sweeps
  the amplitude without re-running MHD.
- **Null-field baseline.** Set a zero-magnitude primary grid and add only the fields of
  interest, keeping the grid geometry independent from the physics.

---

## How it works

At load time, after the primary field is loaded and scaled:

1. Each `[[field.extra_b]]` entry is loaded from disk.
2. Its own `scale_B` / `scale_E` factors are applied.
3. Every voxel of the **primary** grid samples the extra field at the corresponding world
   position via trilinear interpolation.
4. The sampled value is added in-place to the primary grid.

Extra-field voxels that fall **outside** the extra field's declared bounds contribute zero —
the extra field is implicitly zero-padded outside its volume. There is no extrapolation.

The final composite grid is what the GPU sees. The process is $O(N_\text{extra} \times N_\text{primary})$
where $N$ is the number of voxels in the respective grid.

---

## TOML syntax

```toml
[field]
path    = "data/background.bfld"   # primary — defines the GPU grid
scale_B = 1.0

[[field.extra_b]]
path    = "data/plasma_insert.bfld"
scale_B = 1.0
scale_E = 0.0

[[field.extra_b]]
path    = "data/external_solenoid.bfld"
scale_B = 2.0
```

The `[[field.extra_b]]` syntax is a TOML array-of-tables — add as many blocks as needed.

### Available per-field keys

| Key | Default | Description |
|---|---|---|
| `path` | required | Path to `.bfld` file, relative to deck |
| `scale_B` | 1.0 | Multiplicative scale applied to B components |
| `scale_E` | 0.0 | Multiplicative scale applied to E components |
| `e_path` | — | Separate `.bfld` for E-field (version-1 field files) |

The primary `[field]` block also accepts `scale_B`, `scale_E`, and `e_path`.

---

## Grid independence

Each extra field can have a completely different resolution, grid dimensions, and spatial
extent from the primary. The resampling uses trilinear interpolation onto the primary grid's
world coordinates:

```
extra_B_at_primary_voxel = trilinear_interp(extra_field, world_pos)
```

If an extra field covers only part of the primary grid (e.g. a small plasma insert inside a
larger background solenoid), the regions outside the insert contribute zero.

The GPU grid dimensions, bounds, and voxel spacing are always determined by the primary field.

---

## Field amplitude scans

`scale_B` is applied before compositing. To scan the instability field amplitude while
keeping a fixed background:

```bash
# Fixed background at 1 T, scan plasma field from 10% to 100%
proton_tracer run deck.toml --set field.extra_b[0].scale_B=0.1
proton_tracer run deck.toml --set field.extra_b[0].scale_B=0.5
proton_tracer run deck.toml --set field.extra_b[0].scale_B=1.0
```

Or use a sweep:

```toml
# sweep.toml
[[sweep]]
param = "field.extra_b[0].scale_B"
values = [0.1, 0.25, 0.5, 0.75, 1.0]
```

```bash
proton_tracer sweep deck.toml sweep.toml -o runs/amplitude_scan/
```

---

## Worked example: z-pinch instability in an axial guide field

```toml
[field]
path    = "data/instabilities/zpinch.bfld"  # MHD simulation output
scale_B = 1.0

[[field.extra_b]]
path    = "data/guide_field_Bx.bfld"        # uniform axial Bx from external coil
scale_B = 0.5                               # 50% of coil's nominal amplitude
```

```python
import numpy as np
from validate import write_bfld   # or your own writer

# Guide field: uniform Bx = 1 T over the full simulation volume
B_guide = np.zeros((32, 32, 32, 3), dtype=np.float32)
B_guide[:, :, :, 0] = 1.0   # Bx component

bounds = (-0.06, 0.06, -0.06, 0.06, -0.06, 0.06)   # same extent as zpinch field
write_bfld("data/guide_field_Bx.bfld", B_guide, None, bounds)
```

---

## Generating extra fields

Any `.bfld` file is a valid extra field. Common approaches:

**Analytic fields** — generate in Python and write with the `.bfld` writer:

```python
import numpy as np, struct

def write_bfld(path, B, E, bounds):
    nx, ny, nz = B.shape[:3]
    version = 2 if E is not None and np.any(E) else 1
    xmn, xmx, ymn, ymx, zmn, zmx = bounds
    with open(path, "wb") as f:
        f.write(b"BFLD")
        f.write(struct.pack("<I", version))
        f.write(struct.pack("<III", nx, ny, nz))
        f.write(struct.pack("<6f", xmn, xmx, ymn, ymx, zmn, zmx))
        f.write(b"\x00" * 20)
        f.write(B.astype("<f4").tobytes())
        if version == 2:
            f.write(E.astype("<f4").tobytes())
```

**MHD simulation output** — convert your simulation's field array into a `.bfld` using the same writer, ensuring the bounds are in metres and the data is in Tesla.

**Resampling from a coarser grid** — the compositing step handles different resolutions automatically; no manual interpolation is required before writing.

---

## Validation — test14

The test verifies field addition by compositing:

- **Primary:** zero field (all components = 0)
- **Extra:** uniform $B_z = 1$ T

The composite must equal a standalone uniform $B_z = 1$ T field. Energy conservation is
checked (std/mean < 10⁻⁸ on impact kinetic energy — pure magnetic field does no work).

Result: std/mean = 6.5 × 10⁻⁸ (numerical noise only).

---

## Limitations

**CPU-side cost.** Compositing loops over the full primary grid once per extra field. For
a 256³ primary grid and 5 extra fields this takes ~1–2 seconds. It is a one-time load-time
cost, not a per-dispatch cost.

**Memory.** All extra fields must fit in CPU memory simultaneously. Each nx × ny × nz × 3
f32 grid uses `nx × ny × nz × 12` bytes (~750 MB for 256³).

**Grid orientation.** Fields are composited in the shared world coordinate frame defined by
the bounds in each header. If an extra field's coordinate system differs from the primary
(e.g. different x-axis convention), correct the field data before writing the `.bfld` file.

**Legacy JSON configs.** The `[[field.extra_b]]` syntax is TOML-only. JSON configs
use a flat `field_path` key and do not support compositing.
