# Python API

`prad` is a subprocess wrapper around the Rust/Vulkan engine.
Install with `pip install prad`.

---

## `prad.run()`

Run a simulation and return parsed results.

```python
result = prad.run(field, **kwargs) -> RunResult
```

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `field` | `str \| Path \| GridField` | — | Path to `.bfld` file **or** a `GridField` object |
| `source` | `str` | `"parallel"` | Source type: `"parallel"`, `"point"`, `"disk"`, `"pencil"` |
| `energy_MeV` | `float` | `14.7` | Nominal proton kinetic energy [MeV] |
| `n_particles` | `int` | `100_000` | Number of protons to trace |
| `beam_radius_mm` | `float` | `40.0` | Beam radius at source plane [mm] |
| `source_distance_mm` | `float` | `80.0` | Source–origin distance along beam axis [mm] |
| `angular_spread_deg` | `float` | `20.0` | Half-angle cone spread for diverging sources [°] |
| `beam_direction` | `tuple[float,float,float]` | `(1,0,0)` | Unit vector along beam axis |
| `detector_distance_mm` | `float` | `100.0` | Detector–origin distance along beam axis [mm] |
| `detector_size_mm` | `tuple[float,float]` | `(500,500)` | Detector (width, height) [mm] |
| `detector_up` | `tuple[float,float,float]` | `(0,1,0)` | Detector y-axis ("up") direction |
| `dt_ps` | `float` | `0.2` | Boris integrator timestep [ps] |
| `max_steps` | `int` | `25_000` | Maximum steps per particle |
| `scale_B` | `float` | `1.0` | Multiplicative scale on B field |
| `scale_E` | `float` | `0.0` | Multiplicative scale on E field |
| `energy_spread_percent` | `float` | `0.0` | Gaussian σ as % of `energy_MeV`. Ignored when `temperature_MeV` is set. |
| `temperature_MeV` | `float \| None` | `None` | Exponential/TNSA spectrum temperature T [MeV]. Overrides `energy_spread_percent`. |
| `cutoff_MeV` | `float \| None` | `None` | Hard cutoff energy [MeV] for exponential spectrum. Default: 100 × T. |
| `output_dir` | `str \| Path \| None` | `None` | Run directory. If `None`, a temp directory is created and persists until deleted. |
| `overwrite` | `bool` | `True` | Overwrite existing `output_dir`. |
| `binary` | `str \| None` | `None` | Explicit path to the `proton_tracer` binary (optional; auto-detected by default). |
| `timeout` | `int` | `600` | Subprocess timeout [s]. |

### Returns

[`RunResult`](#runresult)

### Examples

=== "Monoenergetic"

    ```python
    import prad

    result = prad.run(
        "data/zpinch.bfld",
        energy_MeV=14.7,
        n_particles=200_000,
        source_distance_mm=80.0,
        detector_distance_mm=100.0,
    )
    print(result.diagnostics["hit_fraction"])  # e.g. 0.94
    counts = result.raw_counts                 # numpy uint32, 1024×1024
    ```

=== "Gaussian spread"

    ```python
    result = prad.run(
        field,
        energy_MeV=14.7,
        energy_spread_percent=5.0,   # σ = 5% of 14.7 MeV = 0.735 MeV
        n_particles=100_000,
    )
    ```

=== "TNSA exponential"

    ```python
    result = prad.run(
        field,
        temperature_MeV=3.0,    # dN/dE ∝ exp(−E / 3 MeV)
        cutoff_MeV=40.0,        # hard cutoff at 40 MeV
        n_particles=200_000,
        dt_ps=0.1,              # smaller dt for high-energy particles
        max_steps=50_000,
    )
    ```

=== "From numpy array"

    ```python
    import numpy as np, prad

    B = np.zeros((64, 64, 64, 3), dtype=np.float32)
    B[:, :, :, 2] = 5.0  # 5 T uniform Bz

    field = prad.Field.from_array(
        B, bounds_m=(-0.05, 0.05, -0.05, 0.05, -0.05, 0.05)
    )
    result = prad.run(field, n_particles=100_000)
    ```

---

## `prad.Field` / `prad.GridField`

`prad.Field` is an alias for `prad.GridField`.

### `GridField.from_array(B, bounds_m, E=None)`

Create a field from numpy arrays.

```python
field = prad.Field.from_array(
    B,          # ndarray (nx, ny, nz, 3) float32 — B field [T]
    bounds_m=(x_min, x_max, y_min, y_max, z_min, z_max),  # metres
    E=None,     # optional ndarray (nx, ny, nz, 3) float32 — E field [V/m]
)
```

### `GridField.load(path)`

Load a `.bfld` file.

```python
field = prad.Field.load("data/zpinch.bfld")
print(field.shape)   # (nx, ny, nz)
print(field.bounds)  # (x_min, x_max, y_min, y_max, z_min, z_max) in metres
```

### `GridField.save(path)`

Write to a `.bfld` file. Creates the file in version 2 format if E is non-zero.

```python
field.save("output/my_field.bfld")
```

---

## `RunResult`

Returned by `prad.run()`. All data is lazy-loaded on first access.

```python
result = prad.run(field, ...)
```

### Properties

| Property | Type | Description |
|---|---|---|
| `run_dir` | `Path` | Path to the run directory on disk |
| `raw_counts` | `ndarray uint32` | Raw detector hit counts, always `(1024, 1024)` |
| `processed_counts` | `ndarray float32` | Blur/noise-processed counts, `(1024, 1024)` |
| `metadata` | `dict` | Full `metadata.json` as a Python dict |
| `diagnostics` | `dict` | `metadata["diagnostics"]` — `n_particles`, `n_hits`, `hit_fraction` |
| `image` | `PIL.Image` | The rendered radiograph PNG as a PIL image |

### Methods

#### `result.show(scale="log", cmap="inferno")`

Display the radiograph inline (Jupyter / matplotlib).

```python
result.show()
result.show(scale="linear", cmap="viridis")
```

#### `result.save(path)`

Save the rendered PNG to disk.

```python
result.save("my_radiograph.png")
```

### Example: accessing raw data

```python
import numpy as np
import prad

result = prad.run("data/zpinch.bfld", n_particles=200_000)

# Raw hit counts
counts = result.raw_counts          # (1024, 1024) uint32
print(counts.sum())                 # total hits
print(counts.max())                 # peak pixel count

# Diagnostics
d = result.diagnostics
print(f"{d['n_hits']:,} hits  ({d['hit_fraction']:.1%} hit fraction)")

# Performance
perf = result.metadata["performance"]
print(f"Runtime: {perf['total_runtime_s']:.2f}s")
```

---

## Binary search order

When no `binary` argument is provided, `prad` finds the `proton_tracer` binary by:

1. `PROTON_TRACER_BIN` environment variable
2. Bundled binary inside the installed wheel (`prad/bin/proton_tracer`)
3. `proton_tracer` on `PATH`
4. `<repo>/rust/target/release/proton_tracer` (development layout)

On macOS, MoltenVK paths are set automatically if Homebrew is installed.
