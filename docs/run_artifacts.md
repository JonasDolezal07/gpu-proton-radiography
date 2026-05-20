# Run artifacts

Every `proton_tracer run` produces a self-contained directory that records everything needed
to reproduce, analyse, or re-render the result — without re-running the GPU simulation.

## Directory layout

```
runs/my_experiment/
  input_deck.toml          ← exact copy of the deck used (or .json for legacy configs)
  resolved_config.json     ← fully resolved SI parameters after --set overrides
  metadata.json            ← provenance: git hash, hardware, field SHA-256, timing
  log.txt                  ← mirror of all terminal output during the run
  counts/
    raw_counts.bin         ← u32 little-endian [H×W] detector hit counts
    processed_counts.bin   ← f32 little-endian [H×W] after detector response
  images/
    radiograph.png
```

## Files in detail

### `input_deck.toml`

The deck file as it was read, copied verbatim before the simulation starts. If you used
`--set` overrides, they are *not* baked into this copy — they appear in `resolved_config.json`.

### `resolved_config.json`

The fully resolved configuration in SI units, after applying `--set` overrides, unit
conversions, and default expansion. This is the definitive record of what the simulation
actually ran. `metadata.json` references this file for reproducibility.

### `metadata.json`

Provenance record. Key fields:

```json
{
  "metadata_schema_version": 1,
  "status": "complete",
  "code": {
    "git_hash": "98a7bd3",
    "git_dirty": false,
    "binary_path": "/path/to/proton_tracer"
  },
  "hardware": {
    "gpu_name": "Apple M2 Pro",
    "os": "macOS 15.x"
  },
  "input_files": {
    "field_path": "data/instabilities/zpinch.bfld",
    "field_sha256": "a3b4c5..."
  },
  "diagnostics": {
    "n_particles": 100000,
    "hit_count": 87432,
    "exit_count": 12568
  },
  "performance": {
    "total_runtime_s": 4.2
  }
}
```

`status` is `"running"` while the simulation is in progress and updated to `"complete"` (or
`"failed"`) when it finishes. This lets you detect interrupted runs.

`field_sha256` is the SHA-256 hash of the field file at run time. If you re-run with a
modified field, the hash will differ — the run directory captures which version was used.

### `log.txt`

A verbatim mirror of all terminal output (stdout + stderr) via `TeeLogger`. Useful for
diagnosing issues without re-running.

### `counts/raw_counts.bin`

Raw detector hit counts: `u32` little-endian, `H × W` row-major. Each element is the number
of protons that landed in that detector pixel. No detector response applied.

H = height in pixels, W = width in pixels. Dimensions are in `resolved_config.json`.

### `counts/processed_counts.bin`

Counts after applying detector response (blur, background, noise): `f32` little-endian, same
`H × W` layout. This is what the PNG is rendered from.

If `[detector_response]` is identity (the default), processed equals raw.

### `images/radiograph.png`

The rendered radiograph — a visualisation of `processed_counts.bin` using the `[render]`
settings from the deck. This is a view, not the data. The PNG can be regenerated at any time:

```bash
proton_tracer render runs/my_experiment \
  --colormap scientific --scale sqrt
```

## Provenance and reproducibility

The run directory is the unit of reproducibility. Given:
- the run directory
- the field file referenced by `input_files.field_path`

you can verify the result by checking `field_sha256`, re-render with different settings, or
load `raw_counts.bin` into Python for custom analysis.

```python
import numpy as np
import json

with open("runs/my_experiment/resolved_config.json") as f:
    cfg = json.load(f)

H = cfg["detector"]["pixels"][1]
W = cfg["detector"]["pixels"][0]

raw = np.frombuffer(
    open("runs/my_experiment/counts/raw_counts.bin", "rb").read(),
    dtype="<u4"
).reshape(H, W)
```

See [file_formats.md](file_formats.md) for the full binary format specification.

## Naming

When you specify `-o runs/my_experiment`, the run directory is created at that exact path.
If the directory already exists and is non-empty, the run will fail unless you pass `--overwrite`
(removes and recreates) or `--resume` (reuses, creates missing subdirs).

## Sweep directories

Parameter sweeps produce a sweep directory containing one run directory per parameter point:

```
runs/sweep_001/
  sweep_manifest.json        ← live status, updated per run
  energy_MeV_5/              ← full run directory
  energy_MeV_10/
  energy_MeV_15/
```

See [sweeps.md](sweeps.md) for details.
