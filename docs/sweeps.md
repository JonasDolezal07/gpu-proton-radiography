# Parameter sweeps

Run a parameter study without editing deck files — one full run directory per parameter point.

## Basic syntax

```bash
proton_tracer sweep <deck.toml> --param key=values [options]
```

## Value lists

### Explicit list

```bash
proton_tracer sweep zpinch.toml --param source.energy_MeV=5,10,15,20
```

Runs four simulations: energy\_MeV = 5, 10, 15, 20.

### Range syntax

```bash
proton_tracer sweep zpinch.toml --param source.energy_MeV=5:20:5
```

Equivalent to `5,10,15,20`. Format: `start:stop:step` (inclusive endpoints, step ≥ 1).

## Zipped multi-parameter sweep

Multiple `--param` flags are zipped — all lists must have the same length:

```bash
proton_tracer sweep zpinch.toml \
  --param source.energy_MeV=5,10,15 \
  --param numerics.max_steps=10000,20000,30000
```

Run 1: energy=5, max\_steps=10000  
Run 2: energy=10, max\_steps=20000  
Run 3: energy=15, max\_steps=30000

Cartesian product mode (`--product`) is planned for a future release.

## Supported parameters

| Key | Description |
|---|---|
| `source.energy_MeV` | Proton kinetic energy |
| `source.n_particles` | Particle count |
| `source.beam_radius_mm` | Beam radius (parallel source) |
| `source.angular_spread_deg` | Angular divergence |
| `source.energy_spread_percent` | Gaussian energy spread |
| `numerics.dt_ps` | Time step |
| `numerics.max_steps` | Maximum integration steps |
| `detector.width_mm` | Detector physical width |
| `detector.height_mm` | Detector physical height |
| `field.scale_B` | B-field multiplicative scale |
| `field.scale_E` | E-field multiplicative scale |

## Output layout

```
runs/sweep_001/
  sweep_manifest.json          ← live status, one entry per run point
  energy_MeV_5/                ← full run directory
    input_deck.toml
    resolved_config.json
    metadata.json
    counts/...
    images/...
  energy_MeV_10/
  energy_MeV_15/
  energy_MeV_20/
```

The sweep directory is auto-named `runs/sweep_NNN` (incrementing). Use `-o` to specify:

```bash
proton_tracer sweep zpinch.toml --param source.energy_MeV=5,10 -o runs/energy_scan
```

## `sweep_manifest.json`

Updated after each run completes. Contains status per point and the full parameter list.
Use `proton_tracer inspect runs/sweep_001` to print a summary.

## Options

```
--param key=values    Parameter to sweep (repeatable — zip mode)
-o <dir>              Output directory (default: auto runs/sweep_NNN)
--overwrite           Remove existing sweep directory before starting
```

## Fault tolerance

All runs continue even if one fails. Failed runs are marked in `sweep_manifest.json`.
The sweep exits non-zero if any run failed.

## Examples

```bash
# B-field strength scan
proton_tracer sweep zpinch.toml --param field.scale_B=0.5,1.0,2.0,5.0

# Energy scan with range syntax
proton_tracer sweep kink_strong.toml --param source.energy_MeV=3:15:3

# Step budget scan
proton_tracer sweep zpinch.toml --param numerics.max_steps=10000,20000,30000
```

## Inspecting a sweep

```bash
proton_tracer inspect runs/sweep_001
```

Prints status and diagnostics for each run point.
