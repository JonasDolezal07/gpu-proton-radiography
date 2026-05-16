# Validation

## Running the suite

```bash
python3 validate.py           # uses existing binary
python3 validate.py --build   # cargo build --release first, then validate
```

Current status: **10/10 passing**.

Output: `output/validation/` (per-test run directories) and `output/validation_report.json`.

## Test descriptions

### Test 1 — B-only regression

Runs the zpinch preset (`data/instabilities/zpinch.json`) and asserts ≥ 10,000 detector hits.

Verifies that the core Boris + B-field path has not regressed. Also checks that PNG,
raw\_counts.bin, and processed\_counts.bin are produced with correct sizes.

### Test 2 — Zero fields, straight-line projection

B = E = 0. Protons travel straight. Asserts that the mean hit position is within ±1 mm of
the detector centre in both y and z.

Catches sign errors in the integrator or coordinate mapping that would cause systematic deflection
in the absence of fields.

### Test 3 — Uniform E only, sign and magnitude

B = 0, Ey = +10 MV/m everywhere. Protons (positive charge) must deflect in +y. Asserts:
- mean hit y > 0 (correct sign)
- magnitude is within a factor of 3 of the non-relativistic analytical estimate

Verifies that the E-field force term has the correct sign and is in the right order of magnitude.

### Test 4 — Uniform B only, energy conservation

B = Bz everywhere, E = 0. The Boris integrator must conserve kinetic energy exactly in a
pure magnetic field (magnetic force does no work). Asserts that the RMS fractional energy
deviation across all particles is < 0.01%.

### Test 5 — Pencil source, 2° tilt, zero field

A pencil beam aimed at 2° off-axis in zero field. Asserts that the mean hit position
corresponds to the expected geometric deflection at the detector plane.

Verifies pencil source geometry and detector coordinate mapping.

### Test 6 — Point source, cone covers full detector

A point source with a wide cone angle. Asserts that hits are distributed across the full
detector area and that the hit fraction exceeds a minimum threshold.

Verifies point source emission geometry and that the full detector is illuminated.

### Test 7 — Disk source, spatial spread matches radius

A disk source with known `radius_um`. In zero field, the spatial spread of hits on the
detector must match the magnified source radius within tolerance.

Verifies disk source geometry and the spatial–angular relationship at the detector.

### Test 8 — Energy spread

A source with `energy_spread_percent > 0`. Asserts that the distribution of hit kinetic
energies matches a Gaussian with the expected standard deviation.

Verifies that the energy spread sampling is applied correctly.

### Test 9 — Gaussian blur, count conservation and spot widening

Applies detector blur (`blur_sigma_um`) to a pencil-like source. Asserts:
- total counts are conserved (blur redistributes, not removes, particles)
- the spatial spread of processed counts is wider than raw counts

Verifies the detector response pipeline.

### Test 10 — Poisson noise reproducibility

Applies Poisson noise with a fixed `noise_seed`. Two runs with the same seed must produce
identical processed counts. A run with a different seed must differ.

Verifies noise seeding and deterministic output.

## Tolerances

Tests use physically motivated tolerances rather than arbitrary percentages:

- Energy conservation (test 4): < 0.01% RMS deviation
- Zero-field centring (test 2): < 1 mm mean offset
- E-field magnitude (test 3): within factor of 3 of analytical estimate (broad, covers relativistic corrections)
- Spatial spread (test 7): within 10% of expected magnified radius

## What is not validated

- GPU numerical precision differences across hardware
- Relativistic accuracy at energies above ~100 MeV
- Multi-species or non-proton particles
- Collisional or radiative processes (the integrator is collisionless)
- Comparison against experimental shot data
- MoltenVK-specific behaviour on non-Apple Silicon hardware
