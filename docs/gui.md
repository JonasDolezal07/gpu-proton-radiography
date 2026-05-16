# GUI launcher

The GUI provides an interactive launcher for deck-based runs with live progress and log output.
It uses the same backend as `proton_tracer run` — results are identical.

## Opening the GUI

```bash
proton_tracer gui                    # open without a deck
proton_tracer gui my_run.toml        # open with a deck pre-loaded
```

## Workflow

### 1. Select a deck

Use the file picker or drag a `.toml` file into the launcher. The deck preview shows the
resolved source type, detector position, and particle count.

### 2. Set output directory

Choose where the run directory will be created. Defaults to `output/`.

### 3. Launch

Click **Run**. The launcher transitions through the following states:

| State | Description |
|---|---|
| **Idle** | No run in progress |
| **Preparing** | Vulkan initialising, field loading |
| **Running** | GPU dispatching — hit and exit counters update live |
| **Complete** | Run finished — output path shown, open-folder button available |
| **Failed** | Error displayed inline |

### 4. Monitor progress

While **Running**, the launcher shows:
- particles hit detector
- particles exited domain
- combined count vs total (completion condition: hit + exit ≥ n\_particles)

### 5. Inspect output

When **Complete**, the run directory path is shown. Click **Open folder** to reveal it in Finder.

## What the GUI does not yet do

- Live radiograph preview during the run
- Re-render controls
- Sweep launching
- Deck editing

These remain CLI-only for now. The GUI is a launcher, not a full IDE.

## CLI vs GUI: same backend

`proton_tracer gui my_run.toml` and `proton_tracer run my_run.toml` run the same Vulkan
compute pipeline and produce identical run directories. The GUI adds the progress UI and
tees log output to `log.txt` the same way the CLI does.
