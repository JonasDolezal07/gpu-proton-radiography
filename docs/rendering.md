# Rendering

## Counts are truth. The PNG is a view.

The simulation produces integer detector hit counts stored in `counts/raw_counts.bin`.
These counts are the scientific output. The PNG radiograph is a visualisation of those counts
with a chosen colourmap, scale, and exposure.

You can re-render a different PNG at any time from the same counts — without re-running the
GPU simulation.

## Render settings in the deck

```toml
[render]
scale = "log"       # log | linear | sqrt
colormap = "rcf"    # rcf | scientific | grayscale | hot | inverted
exposure = 1.0      # multiplicative brightness multiplier
```

These settings apply when `proton_tracer run` produces the radiograph. They are recorded in
`metadata.json` under `render_provenance` so the exact settings used are always known.

## Re-rendering from a run directory

```bash
# Re-render with different settings (writes to images/alt.png by default)
proton_tracer render runs/my_run \
  --colormap scientific \
  --scale sqrt \
  --out runs/my_run/images/alt.png

# Other render flags
  --gamma 0.5           # gamma correction (default 0.5)
  --exposure 2.0        # brightness multiplier
  --width 1024          # output resolution
  --height 1024
```

The run directory form reads `counts/processed_counts.bin` directly — no CSV or hits file needed.

## Colourmap reference

| Name | Description |
|---|---|
| `rcf` / `rcf_film` | RCF film response — high contrast, good for sharp features |
| `scientific` | Perceptually uniform, suitable for publications |
| `grayscale` | Neutral, good for overlay compositing |
| `hot` | Blackbody — emphasises high-count regions |
| `inverted` | Inverted grayscale |

In the deck (`[render]`), use `rcf`. On the command line (`proton_tracer render`), use `rcf_film`.

## Scale modes

| Name | Use when |
|---|---|
| `log` | Dynamic range spans orders of magnitude (default for radiographs) |
| `sqrt` | Moderate dynamic range, less aggressive than log |
| `linear` | Flat, uniform illumination; only use when counts are narrow-range |

## Detector response

A detector response model can be applied before rendering. Configured in the deck:

```toml
[detector_response]
blur_sigma_um = 50.0      # Gaussian blur matching detector PSF
background_counts = 100   # additive background floor
poisson_noise = true      # Poisson shot noise
noise_seed = 42           # for reproducibility
```

Detector response is applied once at run time, producing `processed_counts.bin`. Changing
detector response settings requires a new run — only the render appearance (colourmap, scale,
exposure) can be changed without re-running.

## Re-render from legacy hits CSV

Older runs produced CSV files instead of binary count files. The `render` subcommand can also
read these:

```bash
proton_tracer render \
  --hits output/zpinch_1234.csv \
  --detector-width-mm 500 \
  --detector-height-mm 500 \
  --out zpinch.png
```

Prefer binary counts for new work — they are faster and retain the full detector resolution.
