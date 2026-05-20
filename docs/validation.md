# Validation

## Running the suite

```bash
python3 validate.py           # uses existing binary
python3 validate.py --build   # cargo build --release first, then validate
```

Current status: **12/12 passing**.

Output: `output/validation/` (per-test run directories) and `output/validation_report.json`.

## Test descriptions

### Test 1 — B-only regression

Runs the zpinch preset (`data/instabilities/zpinch.json`) and asserts ≥ 10,000 detector hits.

Verifies that the core relativistic Boris + B-field path has not regressed. Also checks
that PNG, raw\_counts.bin, and processed\_counts.bin are produced with correct sizes.

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

B = Bz everywhere, E = 0. The relativistic Boris integrator must conserve kinetic energy
exactly in a pure magnetic field (magnetic force does no work — the rotation preserves `|u|`
to machine precision). Asserts that the RMS fractional energy deviation across all particles
is < 0.01%.

This test validates relativistic magnetic-only energy conservation. With the relativistic
Boris, reported impact KE equals the input energy (14.7000 MeV) to sub-eV accuracy.

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

### Test 11 — Exponential / TNSA energy spectrum

A pencil source with `temperature_MeV = 3.0` and `cutoff_MeV = 40.0` in zero field.
Checks:
- All impact energies ≤ cutoff (hard cutoff enforced)
- Mean KE within 20 % of T (correct exponential shape — mean ≈ T when cutoff ≫ T)
- Significant energy spread (std/mean > 0.3, confirming non-monoenergetic sampling)

Verifies the inverse-CDF exponential sampler and that the cutoff is respected exactly.

### Test 12 — Relativistic energy conservation at 60 MeV

A pencil source at 60 MeV in zero field. At this energy γ ≈ 1.064, a 6.4% relativistic
correction over the non-relativistic approximation. Wrong momentum initialisation
(e.g. using KE = ½mv²) would give an impact KE of ~58.17 MeV, a ~1.8 MeV shortfall. Checks:
- mean(KE) = 60.000 ± 0.1 MeV (within 0.17%)
- std / mean < 10⁻⁴ (monoenergetic — no spread introduced in zero field)

Verifies that the u = γv momentum initialisation is correct in the fully relativistic regime.

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
- Multi-species or non-proton particles
- Collisional or radiative processes (the integrator is collisionless)
- Comparison against experimental shot data
- MoltenVK-specific behaviour on non-Apple Silicon hardware
