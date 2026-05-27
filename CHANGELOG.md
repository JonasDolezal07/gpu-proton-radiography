# Changelog

All notable changes are documented here. Follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [0.4.0] — 2026-05-28

### Summary

Five interconnected physics and infrastructure features that together make prad suitable for
modelling real composite-geometry experiments: arbitrary detector/source placement, per-hit
energy recording, faster vacuum traversal, overlapping field environments, and ionisation
energy loss.

### Added

**Arbitrary source and detector geometry** (test 13)
- Detector normal and up vectors are now fully configurable — the detector plane can face
  any direction, not just the beam axis.
- Gram-Schmidt orthonormalisation in the GLSL shader constructs the detector (y, z) basis
  from `normal` + `up` at runtime.
- Domain-exit condition rewritten with a detector-centred margin, correct for any orientation.
- `geometry.md` updated with tilted-geometry OMEGA-style example.

**Per-hit binary export (`counts/hits.bin`)**
- Every detector hit is written as a `(y_mm, z_mm, energy_MeV)` triple at end of run.
- Energy is converted from the shader's `(γ−1)c²` representation to MeV on export.
- Enables energy-resolved radiographs, exit KE spectra, and post-hoc material inference.
- File format documented in `file_formats.md`.

**Adaptive timestep** (test 15)
- Three-phase schedule: large dt in vacuum (before and after the field), small Larmor-constrained
  dt inside the field.
- `dt_small` = min(Larmor period/20, grid crossing time/4).
- `dt_large` = min(20 × dt_small, t_entry/10).
- Phase boundaries detected via ray-slab intersection of beam direction against field bounds.
- Activated automatically when no explicit `dt_ps` is supplied; fixed dt still available.
- Typical speedup: 5–20× with no change to physics (test 15: Δmean_y = 0.73 mm at 20× speedup).
- See `docs/adaptive_timestep.md`.

**Superimposed field grids** (test 14)
- Any number of `.bfld` files can be summed at load time via `[[field.extra_b]]` TOML blocks.
- Each extra grid has its own `scale_B` / `scale_E` and can have a different resolution or
  spatial extent from the primary.
- CPU-side trilinear resampling onto the primary grid; zero outside each extra field's bounds.
- No shader or Vulkan changes required.
- Test 14 verification: zero primary + 1 T extra = 1 T standalone (std/mean = 6.5 × 10⁻⁸).
- See `docs/field_compositing.md`.

**Bethe-Bloch CSDA energy loss** (test 16)
- New `[density]` TOML block: path to a `.dens` scalar density grid plus material preset.
- New `.dens` binary format (64-byte header + f32 scalar grid, units g/cm³).
- Relativistic Bethe-Bloch mass stopping power table (256 entries, log-spaced 0.1–1000 MeV)
  precomputed on CPU and uploaded as a GPU storage buffer (binding 6).
- Scalar density grid uploaded as `R32_SFLOAT` sampler3D (binding 5).
- CSDA energy update after each Boris step: sample ρ(x), look up S(T), rescale |u|.
- Built-in material presets: water, plastic (CH₂), beryllium, aluminium, hydrogen.
- Custom material supported via `z_over_a` + `i_ev`.
- Default is vacuum (no density grid, no energy loss) — all existing simulations unaffected.
- CSDA range at 14.7 MeV in water: 2.43 mm (NIST PSTAR: ~2.40 mm, +1.3%).
- Test 16: GPU vs analytic integral, relative error 0.7% (tolerance 5%).
- See `docs/stopping_power.md`.

### Changed

- Validation suite extended from 12 to 16 tests.
- `limitations.md`: removed stale "uniform timestep" entry; added CSDA approximation caveats.
- `index.md`: 3 new feature cards; test count updated to 16/16.

---

## [0.3.1] — 2026-05-15

Physics correctness and infrastructure release.

### Fixed

- **Domain-exit completion bug.** Particles that exit the simulation domain without hitting
  the detector now correctly increment `exit_count` (not `hit_count`). The simulation
  completion condition is `hit_count + exit_count >= n_particles`. Previously, large beams
  with significant background illumination would stall indefinitely.

### Added

- `hits.bin` binary per-hit export (y_mm, z_mm, energy_MeV) written to `counts/`.
- Energy conservation audit tooling; Python `sum()` artifact fixed in the test harness.
- `reproduce_paper.py`: one-command figure regeneration for the draft manuscript.
- Numerical robustness documentation page with convergence studies.

---

## [0.3.0] — 2026-05-01

Phase 2: structured run directories and TOML deck format.

### Added

- `proton_tracer run <deck> -o <dir>` subcommand with structured output layout
  (`counts/`, `images/`, `log.txt`, `metadata.json`, `resolved_config.json`).
- TOML deck format (`DeckConfig`) as the canonical configuration layer, replacing flat JSON.
- `proton_tracer init`, `explain`, `validate`, `render`, `sweep`, `inspect` subcommands.
- `RunDir`, `RunMetadata`, `TeeLogger` — self-documenting, SHA-256 hashed run provenance.
- GUI deck launcher: `RunState` machine, `DeckDisplay` preview, per-run log tee.
- Parameter sweeps with `sweep_manifest.json`.
- Detector response: Gaussian blur, Poisson noise, uniform background.
- TNSA exponential spectrum with inverse-CDF sampler and hard cutoff.

---

## [0.2.0] — 2026-04-10

Relativistic Boris integrator and core GPU pipeline.

### Added

- Relativistic Boris push in `u = γv` specific-momentum space (GLSL compute shader).
- B-field and E-field 3D texture sampling with trilinear interpolation.
- `.bfld` binary field format (version 1: B-only, version 2: B + E).
- Parallel, disk, point, and pencil proton source types.
- Flat rectangular detector plane with configurable pixel grid.
- `--batch` CLI mode (legacy, used by validate.py).
- Initial 12-test physics validation suite.
- Python API wrapper (`prad.run`, `prad.Field`).
