# Limitations

This is research-grade software. The following constraints are known and deliberate.

## Physics model

**Static fields only.** The field is loaded once at startup and does not evolve during the
simulation. There is no field–particle feedback.

**Single species.** Only protons are supported. No multi-species or ion mixture support.

**Relativistic Boris integrator.** Particles are pushed in specific-momentum space
(`u = γv`) with γ recomputed each step. This is exact at all energies — there is no
non-relativistic approximation in the integrator.

**Adaptive or fixed timestep.** By default, prad uses a large dt in vacuum and a
Larmor-constrained dt inside the field (see [adaptive timestep](adaptive_timestep.md)).
A fixed dt can be specified with `dt_ps`. There is no per-particle or sub-step
adaptive control.

**CSDA energy loss only.** When a `[density]` block is configured, proton energy loss
follows the Bethe-Bloch CSDA (mean energy loss per step). Energy-loss straggling
(Bohr / Landau-Vavilov fluctuations), nuclear stopping (relevant below ~100 keV), and
range straggling are not modelled. No energy loss occurs without a density grid —
the simulation is fully collisionless by default.

## Detector model

**Flat detector plane.** The detector is a rectangular plane — no curved or cylindrical
geometry.

**Simple detector response.** Blur, background, and Poisson noise are modelled. More
sophisticated response functions (energy-dependent efficiency, absolute dose calibration)
are not implemented. The energy deposited in RCF film layers is not computed — only the
geometric hit position and exit kinetic energy are recorded.

## Validation

**Not validated against experimental shot data.** The physics tests cover integrator
correctness and source geometry, but no comparison against real radiograph film data
has been performed.

**GPU hardware coverage.** The validation suite passes on Apple Silicon (MoltenVK) and
NVIDIA RTX 4090 (Ubuntu 22.04, driver 550, Vulkan 1.3.277). AMD hardware has not been
tested. The Vulkan compute path uses standard features with no vendor-specific extensions.

## Platform

**macOS / MoltenVK is the primary development platform.** Linux with NVIDIA Vulkan is
validated (see `benchmarks/validation/`). CI runs a build + schema-check pass on Linux
via Mesa software Vulkan; GPU compute tests require real hardware.

**Single GPU.** No multi-GPU or distributed simulation.

## Software architecture

**Legacy `config.rs` retained.** `rust/src/config.rs` contains a legacy GUI editing system
that is not part of the run pipeline. It is retained for compatibility while the deck override
system matures. See `rust/src/loaders/config.rs` for the authoritative configuration layer.

**JSON configs are legacy.** The JSON config format is accepted but not recommended. TOML
decks are the canonical format.

**`prad` Python package.** `pip install prad` gives a subprocess-based Python API
(`prad.run()`, `prad.Field`) wrapping the Rust CLI. Full pyo3/native bindings are not
implemented.
