---
hide:
  - toc
---

<div class="hero-header">
  <img src="images/logo_text.png" alt="prad" class="hero-logo-side" />
  <div class="hero-text">
    <p><strong>GPU Proton Radiography — a forward model for laser-plasma and HEDP experiments.</strong></p>
    <p>prad traces synthetic proton beams through measured or simulated electromagnetic fields and produces radiographs for direct comparison with experimental RCF data. Full relativistic Boris orbit — not a paraxial approximation. 10⁶ particles in under 2 seconds on a laptop GPU.</p>
    <div class="install-block">pip install prad</div>
  </div>
</div>

---

<div class="grid cards" markdown>

-   :material-lightning-bolt:{ .lg .middle } **Relativistic Boris**

    ---

    Particles are pushed in `u = γv` momentum space.
    Energy is exact at all energies — no paraxial or non-relativistic shortcuts.

    [:octicons-arrow-right-24: Validation](validation.md)

-   :material-chart-bell-curve:{ .lg .middle } **TNSA Energy Spectra**

    ---

    Monoenergetic, Gaussian σ spread, or exponential TNSA spectrum
    with configurable temperature and hard cutoff.

    [:octicons-arrow-right-24: Energy spectra](spectra.md)

-   :material-check-all:{ .lg .middle } **16/16 Validation Tests**

    ---

    Energy conservation to < 10⁻¹² relative error.
    Correct impact KE. Spectrum shape and cutoff enforced.
    Bethe-Bloch range agrees with NIST PSTAR to < 2%.

    [:octicons-arrow-right-24: See tests](validation.md)

-   :material-cube-scan:{ .lg .middle } **GPU-Accelerated**

    ---

    ~9 B steps/s on Apple M4 · ~34 B steps/s on NVIDIA RTX 4090.
    1M particles traced end-to-end in under 2 s on a laptop GPU.

    [:octicons-arrow-right-24: Benchmarks](benchmark.md)

-   :material-chart-line:{ .lg .middle } **Numerical Robustness**

    ---

    Timestep, particle-count, and field-grid convergence studies.
    Field-grid resolution identified as the dominant sensitivity.
    One-command figure reproduction.

    [:octicons-arrow-right-24: Convergence studies](convergence.md)

-   :material-speedometer:{ .lg .middle } **Adaptive Timestep**

    ---

    Large dt in vacuum, small Larmor-constrained dt inside the field.
    5–20× speedup with no change to the physics. Automatic — no
    configuration required.

    [:octicons-arrow-right-24: How it works](adaptive_timestep.md)

-   :material-layers-plus:{ .lg .middle } **Field Compositing**

    ---

    Superimpose any number of `.bfld` grids with independent scale
    factors. Overlay plasma fields, coil fields, and background fields
    in a single TOML deck.

    [:octicons-arrow-right-24: Compositing guide](field_compositing.md)

-   :material-water:{ .lg .middle } **Bethe-Bloch Energy Loss**

    ---

    CSDA proton stopping via the relativistic Bethe-Bloch formula.
    Scalar density grids (`.dens`) drive spatially-varying energy loss.
    Built-in material presets for water, plastic, Be, Al.

    [:octicons-arrow-right-24: Stopping power](stopping_power.md)

</div>

---

## Tested platforms

| Platform | GPU | Backend | Status |
|---|---|---|---|
| macOS Apple Silicon | Apple M4 | MoltenVK/Vulkan | Validation suite passing; ~9 B steps/s peak |
| Ubuntu 22.04 Linux | NVIDIA RTX 4090 | NVIDIA Vulkan 1.3.277 | Validation suite passing; ~34 B steps/s peak |

---

## Synthetic radiographs from first principles

Three MHD instability geometries, computed in seconds on a laptop GPU:

<div class="radiograph-gallery">
  <figure>
    <img src="images/zpinch.png" alt="z-pinch radiograph">
    <figcaption><strong>z-pinch</strong> — axial compression, bright central caustic</figcaption>
  </figure>
  <figure>
    <img src="images/kink_strong.png" alt="kink instability radiograph">
    <figcaption><strong>kink (strong)</strong> — helical deflection pattern</figcaption>
  </figure>
  <figure>
    <img src="images/sausage_strong.png" alt="sausage instability radiograph">
    <figcaption><strong>sausage (strong)</strong> — periodic pinch–swell structure</figcaption>
  </figure>
</div>

Each image is a synthetic proton radiograph. The spatial structure directly encodes the
path-integrated field topology — no paraxial assumptions.

---

## 30-second quickstart

=== "Python API"

    ```python
    import numpy as np
    import prad

    # Build a field from numpy arrays — or load directly from a .bfld file
    B = np.zeros((64, 64, 64, 3), dtype=np.float32)
    B[:, :, :, 2] = 5.0   # 5 T uniform Bz

    field = prad.Field.from_array(
        B, bounds_m=(-0.05, 0.05, -0.05, 0.05, -0.05, 0.05)
    )

    # Monoenergetic — 14.7 MeV (D–³He fusion protons)
    result = prad.run(
        field,
        energy_MeV=14.7,
        n_particles=200_000,
        source_distance_mm=80.0,
        detector_distance_mm=100.0,
    )
    result.show()
    print(result.raw_counts.shape)   # (1024, 1024) uint32

    # TNSA broad spectrum — laser-plasma relevant
    result_tnsa = prad.run(
        field,
        temperature_MeV=3.0,   # dN/dE ∝ exp(−E / T)
        cutoff_MeV=40.0,
        n_particles=200_000,
    )
    result_tnsa.save("tnsa_radiograph.png")
    ```

=== "TOML deck (CLI)"

    ```toml
    [field]
    path = "data/zpinch.bfld"

    [source]
    type            = "parallel"
    energy_MeV      = 14.7
    temperature_MeV = 3.0    # TNSA exponential spectrum
    cutoff_mev      = 40.0
    n_particles     = 200000
    beam_radius_mm  = 40.0
    source_distance_mm = 80.0

    [detector]
    center_mm  = [100.0, 0.0, 0.0]
    width_mm   = 500.0
    height_mm  = 500.0

    [numerics]
    dt_ps      = 0.2
    max_steps  = 25000
    ```

    ```bash
    proton_tracer run deck.toml -o runs/my_run
    ```

---

## Why this tool?

**Speed that changes what's practical.** The full-orbit Boris integrator runs 10⁶ particles
in under 2 seconds on a laptop GPU and under 0.5 seconds on an RTX 4090. In a matched
simplified particle-tracing test (10,000 particles, uniform field, single core), prad is
**214× faster** than a CPU Boris implementation via PlasmaPy, with GPU utilisation
increasing further at larger particle counts. That gap makes workflows practical that
previously required paraxial shortcuts: broad parameter sweeps, interactive geometry design,
comparison of field topologies, and large synthetic dataset generation for ML inverse solvers.

**No approximations.** Every fast alternative uses the paraxial approximation — integrating
the field kick along an unperturbed straight-line trajectory. In the strong-field regimes
common in pulsed-power and high-intensity laser experiments, it fails badly. At just 20%
of the kink instability field amplitude the paraxial model produces streaks spanning the
entire detector; the full-orbit result shows the correct helical kink signature. prad traces
the complete relativistic orbit in all cases.

**A complete research environment.** prad ships with a Python API (`pip install prad`,
numpy arrays in and out), a parameter sweep engine, a GUI for interactive deck editing,
TNSA exponential spectrum sampling, and self-documenting run directories with SHA-256 field
hashes for exact reproducibility. You can go from a numpy field array to a synthetic
radiograph with three lines of Python, or run a multi-point energy sweep from the command line
in one command. Bethe-Bloch energy loss, field compositing, and adaptive timestepping are
available out of the box — no extra dependencies.

[Get started :octicons-arrow-right-24:](quickstart.md){ .md-button .md-button--primary }
[Python API :octicons-arrow-right-24:](python_api.md){ .md-button }
[Validation :octicons-arrow-right-24:](validation.md){ .md-button }

---

Built by [Jonas Dolezal](https://jonasdolezal.com).
