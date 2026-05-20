# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build and run

```bash
# Build release binary (also recompiles any changed GLSL shaders)
cd rust && cargo build --release

# Run validation suite (12 physics tests — all should pass)
python3 validate.py
python3 validate.py --build     # cargo build --release first

# Phase 2 run-directory smoke test
python3 scripts/smoke_run_dir.py
python3 scripts/smoke_run_dir.py --build --keep   # keep output for inspection
```

**macOS/MoltenVK:** The binary requires these env vars when running outside of validate.py (which sets them automatically):
```bash
export VK_ICD_FILENAMES=/opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json
export DYLD_LIBRARY_PATH=/opt/homebrew/lib:$DYLD_LIBRARY_PATH
```

## CLI subcommands

```bash
./rust/target/release/proton_tracer run <deck> -o <dir> [--overwrite]
./rust/target/release/proton_tracer explain <deck>      # resolved geometry, no run
./rust/target/release/proton_tracer validate <deck>     # schema check only
./rust/target/release/proton_tracer gui [deck]
./rust/target/release/proton_tracer init [preset] [-o deck.toml]
```

Legacy invocation used by validate.py: `proton_tracer <config.json> --batch -o <dir>` — this uses a flat output path (no RunDir). Do not break this path.

## Architecture

### Two config systems — do not confuse them

**`rust/src/loaders/config.rs`** — canonical, used by the run pipeline:
- `RawConfig` parses JSON (with legacy field aliases + deprecation warnings)
- `DeckConfig` parses TOML (canonical new format)
- `SimConfig` is the internal SI representation — the only struct the engine sees
- `SimConfig::load(path)` accepts both `.json` and `.toml` by extension

**`rust/src/config.rs`** — legacy `SimulationConfig` used only by the GUI editing panels. Not wired into the run pipeline. When the two diverge, `SimConfig` is authoritative.

### Run pipeline (Phase 2)

`run` subcommand path in `main.rs`:
1. `RunDir::open()` — creates folder + subdirs (`counts/`, `images/`, `tables/`) before Vulkan starts
2. `RunMetadata::new_running()` — writes `metadata.json` with `status: "running"`
3. Vulkan init, `load_simulation()` — copies deck, writes `resolved_config.json`, hashes field file
4. Batch compute loop — GPU dispatches
5. `renderer.export_to_run_dir()` — writes `counts/raw_counts.bin`, `counts/processed_counts.bin`, `images/radiograph.png`
6. Finalise `metadata.json` — `status: "complete"`, diagnostics, perf

The run directory layout (`metadata_schema_version: 1`) is **semi-frozen**. Additive fields are safe; removing/renaming fields or changing binary format requires bumping `metadata_schema_version` in `run_dir.rs`.

### GPU completion condition

The detector buffer header (binding 2, `boris.comp`) is `[hit_count, exit_count, _pad2, _pad3]`.

Simulation is complete when `hit_count + exit_count >= particle_count`. Both counters are GPU atomics. Using only `hit_count` as the completion signal is wrong — particles that exit the domain increment `exit_count` without hitting the detector.

### Shader compilation

`build.rs` auto-compiles `shaders/*.{comp,vert,frag}` → `shaders/*.spv` via `glslangValidator` on every `cargo build` when sources change. The `.spv` files in the repo are a convenience copy, not authoritative. `glslangValidator` is at `/opt/homebrew/bin/glslangValidator`.

### Field format (`.bfld`)

```
Header (64 bytes): magic "BFLD" | version u32 | nx,ny,nz u32×3 | bounds 6×f32 | padding
Data: B-field nx×ny×nz×3 f32, C-contiguous (x outermost, z innermost, components last)
      [version 2 only] E-field, same layout
```

Version 1 = B-only. Version 2 = B+E. Field sampling in the shader returns **zero** for any particle position outside the declared `field_min`/`field_max` bounds. Inside the domain, trilinear interpolation is used; the hardware sampler is `CLAMP_TO_EDGE` but is never reached because of the explicit domain check.

### Coordinate convention

+x is the beam axis throughout. Source is upstream (x < 0); detector is downstream (x > 0). Detector plane is y–z. CSV hit positions are `y_mm, z_mm`. The detector basis is:
- `u_y = normalize(up − dot(up, n)·n)` (Gram-Schmidt)
- `v_z = cross(n, u_y)`

### Step budget warning

`build_experiment_summary()` in `main.rs` warns when `max_steps × dt × v < source-to-detector distance`. This is a straight-line lower bound. Strong field deflections increase path length, so configs need margin (25 000 steps for zpinch at dt=0.2 ps, not the straight-line minimum of ~17 200).

### Key files

| File | Role |
|---|---|
| `shaders/boris.comp` | GPU Boris integrator, detector hit + domain exit |
| `rust/src/loaders/config.rs` | `RawConfig`→`SimConfig` (JSON), `DeckConfig`→`SimConfig` (TOML) |
| `rust/src/loaders/particles.rs` | Particle generation for all 4 source types |
| `rust/src/loaders/field.rs` | `.bfld` loader, trilinear interpolation |
| `rust/src/gpu/compute.rs` | `SimParams` push constants, GPU dispatch |
| `rust/src/gpu/renderer.rs` | Batch/interactive loop, export methods |
| `rust/src/run_dir.rs` | `RunDir`, `RunMetadata`, `TeeLogger`, `sha256_file` |
| `rust/src/main.rs` | CLI dispatch, `App` struct, `load_simulation`, `build_experiment_summary` |
| `rust/src/config.rs` | Legacy GUI-only `SimulationConfig` (not used in run pipeline) |
| `validate.py` | 10-test headless suite using legacy `--batch` invocation |
| `scripts/smoke_run_dir.py` | End-to-end structured run directory test |
| `data/instabilities/*.json` | Real field configs for zpinch, kink, sausage, mixed |
