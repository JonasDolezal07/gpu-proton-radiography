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
