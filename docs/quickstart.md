# Quickstart

Get from zero to a radiograph in five minutes.

## Prerequisites

- Rust toolchain (`rustup` — stable is fine)
- `glslangValidator` for shader compilation: `brew install glslang`
- MoltenVK (macOS): `brew install molten-vk`
- Python 3.9+ with numpy (for validation only)

## Build

```bash
cd rust
cargo build --release
cd ..
```

This also recompiles any changed GLSL shaders via `build.rs`.

## Set environment (macOS)

```bash
export VK_ICD_FILENAMES=/opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json
export DYLD_LIBRARY_PATH=/opt/homebrew/lib:$DYLD_LIBRARY_PATH
```

Add these to your shell profile so you don't need to set them each session.

## Optional: add binary to PATH

The commands below use `./rust/target/release/proton_tracer`. To use the bare name
`proton_tracer` from any directory (as shown in other docs), add the binary to your PATH:

```bash
export PATH="$PATH:/path/to/gpu_proton_tracer/rust/target/release"
```

## First run

### 1. Scaffold a deck

```bash
./rust/target/release/proton_tracer init zpinch -o my_run.toml
```

This writes a complete, valid deck for the z-pinch instability preset. Open `my_run.toml`
and read it — every field is documented by its name.

Available init presets: `blank`, `zpinch`, `kink-strong`.

### 2. Inspect before running

```bash
./rust/target/release/proton_tracer explain my_run.toml
```

Prints resolved geometry — source position, detector size, magnification factor, step budget
estimate, and field diagnostics — without touching the GPU. Use this to catch configuration
mistakes before committing compute time.

### 3. Schema check

```bash
./rust/target/release/proton_tracer validate my_run.toml
```

Parses and resolves the deck, reports any schema errors, exits 0 on success.

### 4. Run

```bash
./rust/target/release/proton_tracer run my_run.toml -o runs/first_run
```

Runs the GPU relativistic Boris integrator and writes a self-contained run directory:

```
runs/first_run/
  input_deck.toml
  resolved_config.json
  metadata.json
  log.txt
  counts/
    raw_counts.bin
    processed_counts.bin
  images/
    radiograph.png
```

### 5. Inspect the output

```bash
./rust/target/release/proton_tracer inspect runs/first_run
```

Prints a summary: status, particle counts, hit rate, timing.

### 6. Re-render without GPU

Change colourmap or scale without re-running the simulation:

```bash
./rust/target/release/proton_tracer render runs/first_run \
  --colormap scientific --scale sqrt --out runs/first_run/images/alt.png
```

Counts are the truth. The PNG is a view.

## Open the GUI

```bash
./rust/target/release/proton_tracer gui my_run.toml
```

The GUI launcher lets you select a deck, set the output directory, and watch hit/exit progress
in real time. The same backend runs as `proton_tracer run`.

## Run validation

```bash
python3 validate.py           # uses existing binary
python3 validate.py --build   # build first, then validate
```

10 physics tests — all should pass. See [validation.md](validation.md) for what each test checks.

## Common macOS MoltenVK issue

If the binary exits immediately with no output, the Vulkan ICD is not being found. Confirm:

```bash
echo $VK_ICD_FILENAMES
ls /opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json
```

Both should resolve. If `MoltenVK_icd.json` is missing, run `brew install molten-vk`.
