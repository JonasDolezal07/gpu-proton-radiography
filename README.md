<p align="center">
  <img src="docs/images/logo_text.png" alt="prad — GPU Proton Radiography" width="420" />
</p>

GPU-accelerated forward modelling of proton radiographs from magnetised plasma.

Runs the full **relativistic Boris** orbit through measured or simulated electromagnetic
fields — not a paraxial approximation — and produces synthetic radiographs that can be
directly compared to experimental RCF film data.

```
✓ 12/12 physics validation tests passing
✓ Reproducible, self-documenting run directories
✓ CLI + GUI workflows
✓ Re-render without re-tracing particles
```

---

## Example radiographs

Three MHD instability geometries, computed in seconds on a laptop GPU:

| z-pinch | kink instability | sausage instability |
|:---:|:---:|:---:|
| ![z-pinch radiograph](docs/images/zpinch.png) | ![kink instability radiograph](docs/images/kink_strong.png) | ![sausage instability radiograph](docs/images/sausage_strong.png) |

Each image is a synthetic proton radiograph — the spatial structure directly reflects the
path-integrated field topology.

### GUI

![GUI launcher showing sausage instability run complete](docs/images/gui.png)

Deck parameters, run status, and the 3D radiograph — all in one view.

---

## Why this tool

Proton radiography is sensitive to the path-integrated field, not just its peak value.
The mapping from field structure to film pattern is nonlinear and depends on geometry —
magnification, detector distance, source divergence. Paraxial approximations fail in the
strong-field, large-deflection regimes common in modern pulsed-power experiments.

This tool runs the **full relativistic Boris orbit**, so you can:

- See where paraxial approximations break down and by how much
- Forward-model field topologies and compare directly to experimental films
- Design detector geometry before committing to a shot

---

## Quick start

```bash
# Build (also compiles shaders via build.rs)
cd rust && cargo build --release && cd ..

# Scaffold a working deck from a preset
./rust/target/release/proton_tracer init zpinch -o my_run.toml

# Inspect resolved geometry before running
./rust/target/release/proton_tracer explain my_run.toml

# Schema check
./rust/target/release/proton_tracer validate my_run.toml

# Run — produces a self-contained output directory
./rust/target/release/proton_tracer run my_run.toml -o runs/zpinch_01
```

**macOS / MoltenVK** — set these before running:
```bash
export VK_ICD_FILENAMES=/opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json
export DYLD_LIBRARY_PATH=/opt/homebrew/lib:$DYLD_LIBRARY_PATH
```

See [docs/quickstart.md](docs/quickstart.md) for the full install walkthrough.

---

## Subcommands

| Command | Purpose |
|---|---|
| `run <deck> [-o dir]` | Batch run — GPU compute, full run directory output |
| `gui [deck]` | Interactive launcher with live progress |
| `explain <deck>` | Print resolved geometry and step budget — no GPU |
| `validate <deck>` | Schema check only — no GPU |
| `init [preset] [-o deck.toml]` | Emit a starter deck (`blank` / `zpinch` / `kink-strong`) |
| `demo [preset]` | Run a built-in preset without writing a deck |
| `render <run_dir>` | Re-render radiograph from saved counts — no GPU |
| `sweep <deck> --param k=v1,v2` | Parameter sweep — one run directory per point |
| `inspect <run_dir\|sweep_dir>` | Print run or sweep summary |
| `analyze <run_dir>` | Count statistics |

---

## Run directory layout

Every `run` produces a self-contained directory. Share it with a colleague and they can
re-render or analyse without re-running anything.

```
runs/zpinch_01/
  input_deck.toml          ← exact copy of deck used
  resolved_config.json     ← fully resolved SI parameters
  metadata.json            ← hardware, git hash, field SHA-256, timing
  log.txt                  ← full terminal output mirror
  counts/
    raw_counts.bin         ← u32 [H×W] detector hit counts
    processed_counts.bin   ← f32 [H×W] after detector response
  images/
    radiograph.png
```

---

## Parameter sweeps

```bash
# Energy scan — four runs, zipped
proton_tracer sweep zpinch.toml \
  --param source.energy_MeV=5,10,15,20

# Range syntax
proton_tracer sweep zpinch.toml \
  --param source.energy_MeV=5:20:5

# Paired sweep — same-length lists, zipped
proton_tracer sweep zpinch.toml \
  --param source.energy_MeV=5,10,15 \
  --param numerics.max_steps=10000,20000,30000
```

Output: `runs/sweep_001/` with one run directory per point and a live `sweep_manifest.json`.

---

## Validation

```bash
python3 validate.py           # uses existing binary
python3 validate.py --build   # build first, then validate
```

12 physics tests: B-only regression, zero-field straight-line projection, uniform E-field
deflection (sign and magnitude), relativistic Boris energy conservation (14.7000 MeV
recovered to sub-eV accuracy), pencil/point/disk source geometry, Gaussian energy spread,
exponential/TNSA spectrum (mean KE ≈ T, hard cutoff enforced), Gaussian blur, Poisson noise
reproducibility, and 60 MeV relativistic momentum initialisation (γ ≈ 1.064).

---

## Documentation

| Doc | Contents |
|---|---|
| [docs/quickstart.md](docs/quickstart.md) | Full install → first run walkthrough |
| [docs/geometry.md](docs/geometry.md) | Coordinate system, detector geometry, source types |
| [docs/input_decks.md](docs/input_decks.md) | TOML schema, all fields, `--set` overrides |
| [docs/run_artifacts.md](docs/run_artifacts.md) | Run directory anatomy and reproducibility |
| [docs/file_formats.md](docs/file_formats.md) | `.bfld`, binary count formats, metadata schema |
| [docs/rendering.md](docs/rendering.md) | Counts → PNG pipeline, re-render without GPU |
| [docs/sweeps.md](docs/sweeps.md) | Parameter sweeps, syntax, sweep manifest |
| [docs/validation.md](docs/validation.md) | Physics test descriptions and tolerances |
| [docs/gui.md](docs/gui.md) | Deck launcher workflow |
| [docs/limitations.md](docs/limitations.md) | Honest constraints and known gaps |

---

## License

See [LICENSE](LICENSE).
