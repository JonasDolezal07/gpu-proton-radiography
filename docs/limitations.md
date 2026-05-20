# Limitations

This is research-grade software. The following constraints are known and deliberate.

## Physics model

**Static fields only.** The field is loaded once at startup and does not evolve during the
simulation. There is no field–particle feedback.

**Collisionless.** No inter-particle interactions, scattering, or energy loss. Protons travel
through the field independently.

**Single species.** Only protons are supported. No multi-species or ion mixture support.

**Relativistic Boris integrator (implemented).** Particles are pushed in
specific-momentum space (`u = γv`) with γ recomputed each step. This is exact at all
energies — there is no non-relativistic approximation in the integrator.

**Uniform time step.** `dt_ps` is fixed for all particles throughout the simulation. There
is no adaptive step size control.

## Detector model

**Flat detector plane.** The detector is a rectangular plane — no curved or cylindrical
geometry.

**Simple detector response.** Blur, background, and Poisson noise are modelled. More
sophisticated response functions (energy-dependent efficiency, proton stopping) are not
implemented.

## Validation

**Not validated against experimental shot data.** The physics tests cover integrator
correctness and source geometry, but no comparison against real radiograph film data
has been performed.

**GPU hardware coverage is limited.** The validation suite runs on Apple Silicon via MoltenVK.
Behaviour on NVIDIA/AMD hardware has not been systematically tested, though the Vulkan compute
code uses standard features with no vendor-specific extensions.

## Platform

**macOS / MoltenVK is the primary development platform.** Linux with a native Vulkan driver
should work but has not been tested in CI.

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
