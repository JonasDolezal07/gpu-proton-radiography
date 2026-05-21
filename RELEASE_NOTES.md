# Release notes

## v0.3.0-rc1 — 2026-05-21

Feature freeze for the v0.3.0 manuscript submission.

### What's in this release

**Physics engine**
- Full relativistic Boris integrator (`u = γv`) — no paraxial or non-relativistic approximations
- Three proton source spectra: monoenergetic, Gaussian, exponential/TNSA (closed-form inverse-CDF)
- Three source geometries: point, disk, parallel beam
- B+E field support (`.bfld` v2), trilinear interpolation, explicit domain bounds check
- Detector model: Gaussian PSF blur, Poisson noise, background offset

**Validated platforms**
- Apple M4 / macOS (MoltenVK): 12/12 tests passing, ~9 B steps/s
- NVIDIA RTX 4090 / Ubuntu 22.04 (driver 550, Vulkan 1.3.277): 12/12 tests passing, ~34 B steps/s
- Headless Linux batch mode via `VulkanContext::new_headless()` (no display required once `DISPLAY` is set for NVIDIA ICD init)

**Validation suite** — 12 physics tests:
1. z-pinch regression (1M particles)
2. Zero-field straight-line projection
3. Uniform E-field parabolic deflection (sign + magnitude)
4. B-only energy conservation (14.7000 MeV, rel_std < 10⁻¹¹)
5. Tilted pencil beam (2°, rel_err < 10⁻³)
6. Point source full-cone (hit fraction = 1.000)
7. Disk source spatial spread (σ = 15.006 mm vs 15.000 mm expected)
8. Gaussian energy spread (5% reproduced in impact energies)
9. Gaussian blur count conservation + spot widening
10. Poisson RNG reproducibility (same seed → identical)
11. Exponential/TNSA spectrum (mean ≈ T, hard cutoff enforced)
12. 60 MeV relativistic momentum initialisation (γ ≈ 1.064, rel_std < 10⁻¹²)

**CLI / workflow**
- `run`, `explain`, `validate`, `init`, `demo`, `render`, `sweep`, `inspect`, `analyze`, `gui`
- Self-contained run directories with SHA-256 field hash and `metadata.json`
- Parameter sweeps with range syntax and multi-parameter zipped sweeps
- Re-render without re-tracing (`render` subcommand)

**Python API** — `pip install prad`
- `prad.run()`, `prad.Field.from_array()`, `prad.RunResult`
- NumPy array I/O

**Documentation** — MkDocs Material site (13 pages)

### Breaking changes from v0.2.x

None. JSON config format remains accepted (with deprecation warnings). TOML is canonical.

### Known limitations (see `docs/limitations.md`)

- Static fields only; no adaptive time step
- Flat detector plane only
- Python API is subprocess-based (PyO3 bindings are future work)
- AMD/Windows not yet validated
- No comparison against experimental shot data

### Cite as

```bibtex
@software{dolezal2026prad,
  author  = {Dolezal, Jonas},
  title   = {{prad}: {GPU}-accelerated relativistic proton radiography},
  year    = {2026},
  version = {0.3.0-rc1},
  url     = {https://github.com/JonasDolezal07/gpu-proton-radiography},
  license = {MIT},
}
```
