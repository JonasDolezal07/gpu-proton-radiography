# File formats

## `.bfld` — field file

Binary format for B-field (and optionally E-field) data.

### Header (64 bytes, little-endian)

| Offset | Size | Type | Value |
|---|---|---|---|
| 0 | 4 | bytes | Magic: `BFLD` |
| 4 | 4 | u32 | Version: 1 (B only) or 2 (B + E) |
| 8 | 4 | u32 | nx — grid points in x |
| 12 | 4 | u32 | ny |
| 16 | 4 | u32 | nz |
| 20 | 4 | f32 | x\_min (metres) |
| 24 | 4 | f32 | x\_max |
| 28 | 4 | f32 | y\_min |
| 32 | 4 | f32 | y\_max |
| 36 | 4 | f32 | z\_min |
| 40 | 4 | f32 | z\_max |
| 44 | 20 | — | Padding (zeros) |

### Data

Immediately after the header:

- **B-field:** `nx × ny × nz × 3` values, `f32` little-endian, C-contiguous
  (x outermost, z innermost, components last: Bx, By, Bz)
- **E-field** (version 2 only): same layout, same size

Total data size: `nx × ny × nz × 3 × 4` bytes per field component.

### Axis convention

The +x axis is the beam axis. Field indexing is `B[ix, iy, iz, component]`.

### Field sampling

The shader clamps the normalised sampling coordinate to `[0.001, 0.999]`. Particles outside
the field volume see the boundary field value, not zero. Keep the field volume large enough
to cover the full proton path.

### Python reader / writer

`python/field_format.py` provides `read_bfld()` and `write_bfld()`. The validation suite
in `validate.py` uses `write_bfld()` directly.

```python
from python.field_format import read_bfld

header, B, E = read_bfld("plasma.bfld")
print(header)          # dict with version, nx, ny, nz, bounds
print(B.shape)         # (nx, ny, nz, 3)  float32
print(E)               # None for version 1
```

---

## `.dens` — density file

Binary format for scalar 3D density grids. Used by the `[density]` block for Bethe-Bloch
energy loss. See [Stopping power](stopping_power.md) for full documentation.

### Header (64 bytes, little-endian)

| Offset | Size | Type | Value |
|---|---|---|---|
| 0 | 4 | bytes | Magic: `DENS` |
| 4 | 4 | u32 | Version: 1 |
| 8 | 4 | u32 | nx |
| 12 | 4 | u32 | ny |
| 16 | 4 | u32 | nz |
| 20 | 4 | f32 | x\_min (metres) |
| 24 | 4 | f32 | x\_max |
| 28 | 4 | f32 | y\_min |
| 32 | 4 | f32 | y\_max |
| 36 | 4 | f32 | z\_min |
| 40 | 4 | f32 | z\_max |
| 44 | 20 | — | Padding (zeros) |

### Data

`nx × ny × nz` values, `f32` little-endian, C-contiguous (x outermost, z innermost).
Units: **g/cm³**.

---

## `hits.bin`

Per-hit binary export written to `counts/hits.bin` at the end of each run. Records the
detector impact position and kinetic energy for every particle that hit the detector.

### Format

```
4 bytes   — u32 little-endian: number of recorded hits (N)
N × 12    — f32 triples: (y_mm, z_mm, energy_MeV) per hit
```

- **y_mm, z_mm** — position on the detector plane in detector-local coordinates
  (y along the detector y-axis, z along the detector z-axis), in millimetres
- **energy_MeV** — kinetic energy at impact in MeV

The file contains at most `MAX_DETECTOR_HITS` records (currently 10⁶). Runs with more hits
than this cap will have a truncated buffer — the count field reflects the number of hits
that reached the detector, which may exceed the buffer size.

### Python reader

```python
import struct
import numpy as np

def read_hits(hits_bin_path):
    """Returns (y_mm, z_mm, energy_MeV) as numpy arrays."""
    raw = open(hits_bin_path, "rb").read()
    n = struct.unpack_from("<I", raw, 0)[0]
    if n == 0:
        return np.array([]), np.array([]), np.array([])
    data = np.frombuffer(raw, dtype="<f4", offset=4).reshape(-1, 3)
    y_mm       = data[:, 0]
    z_mm       = data[:, 1]
    energy_MeV = data[:, 2]
    return y_mm, z_mm, energy_MeV

y, z, E = read_hits("runs/my_run/counts/hits.bin")
print(f"{len(E)} hits, mean KE = {E.mean():.3f} MeV, σ = {E.std():.4f} MeV")
```

### Energy units

The GPU shader stores the particle energy as $(γ-1)c^2$ [m²/s²]. The export code converts to
MeV by multiplying by the proton mass (kg) and dividing by the joule-per-MeV conversion factor:

```
energy_MeV = (γ-1) × c² × m_p [kg] / (1.602 × 10⁻¹³ J/MeV)
           = (γ-1) × 938.272 MeV
```

which is the standard relativistic kinetic energy $T = (γ-1) m_p c^2$.

---

## `raw_counts.bin`

Raw detector hit counts produced by the GPU.

- **dtype:** `uint32` little-endian
- **shape:** `H × W` row-major (H = height in pixels, W = width in pixels)
- **values:** integer count of protons that hit each detector pixel

Pixel `[row, col]` corresponds to detector position `(col × pixel_width, row × pixel_height)`
in detector-local coordinates.

---

## `processed_counts.bin`

Counts after applying detector response (blur, background, noise).

- **dtype:** `float32` little-endian
- **shape:** `H × W` row-major, same as `raw_counts.bin`

If no detector response is configured (the default), processed counts are equal to raw counts
cast to f32.

### Loading in Python

```python
import numpy as np
import json

with open("runs/my_run/resolved_config.json") as f:
    cfg = json.load(f)

pixels = cfg["detector"]["pixels"]   # [W, H]
W, H = pixels[0], pixels[1]

raw = np.fromfile("runs/my_run/counts/raw_counts.bin", dtype="<u4").reshape(H, W)
proc = np.fromfile("runs/my_run/counts/processed_counts.bin", dtype="<f4").reshape(H, W)
```

---

## `metadata.json`

Schema version 1 (semi-frozen — additive fields are safe; removing or renaming fields or
changing binary format requires bumping `metadata_schema_version`).

Top-level structure:

```json
{
  "metadata_schema_version": 1,
  "status": "complete | running | failed",
  "run_id": "zpinch_20260515_143022",
  "created_at": "2026-05-15T14:30:22Z",
  "code": { ... },
  "hardware": { ... },
  "input_files": { ... },
  "output_files": { ... },
  "counts_format": { ... },
  "diagnostics": { ... },
  "performance": { ... },
  "render_provenance": { ... }
}
```

`counts_format` records the dtype, shape, and endianness of the binary count files so the
format is self-documenting.

---

## `sweep_manifest.json`

Written by `proton_tracer sweep` and updated after each run point.

```json
{
  "sweep_schema_version": 1,
  "deck_path": "zpinch.toml",
  "params": [
    {"source.energy_MeV": "5"},
    {"source.energy_MeV": "10"}
  ],
  "runs": [
    {"label": "energy_MeV_5",  "status": "complete", "run_dir": "energy_MeV_5/"},
    {"label": "energy_MeV_10", "status": "running",  "run_dir": "energy_MeV_10/"}
  ]
}
```

---

## `resolved_config.json`

The fully-resolved SI config written at the start of each run. All units are SI (metres,
seconds, joules). Useful for post-processing without re-parsing the deck.
