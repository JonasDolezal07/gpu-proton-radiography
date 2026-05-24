# Energy Spectra

prad supports three proton source energy distributions.
All three are fully compatible with the relativistic Boris integrator —
`u = γv` is computed correctly regardless of the energy drawn.

---

## Overview

| Mode | Key parameter | Typical use |
|---|---|---|
| `mono` | `energy_MeV` | D–³He fusion protons (14.7 MeV), accelerator beams |
| `gaussian` | `energy_spread_percent` | Slightly impure mono sources, calibration runs |
| `exponential` | `temperature_MeV`, `cutoff_MeV` | Laser-accelerated (TNSA) proton sources |

The mode is selected automatically from the parameters you provide:

- `temperature_MeV` set → **exponential** (overrides everything else)
- `energy_spread_percent > 0` → **Gaussian**
- Neither → **monoenergetic** (default)

---

## Monoenergetic

All particles start with the same kinetic energy `energy_MeV`.
This is the default and the simplest case.

=== "Python"

    ```python
    result = prad.run(field, energy_MeV=14.7)
    ```

=== "TOML"

    ```toml
    [source]
    energy_MeV = 14.7
    # no energy_spread_percent or temperature_MeV → monoenergetic
    ```

**Typical energies:**

| Source | Energy |
|---|---|
| D–³He fusion protons | 14.7 MeV and 3.0 MeV |
| OMEGA-EP, NIF backlighter | 14.7 MeV |
| Accelerator protons | varies |

---

## Gaussian spread

Samples each particle energy from a normal distribution:

```
E ~ Normal(μ, σ)    σ = μ × spread / 100
```

where μ is `energy_MeV` and spread is `energy_spread_percent`. Negative draws are
rejected and resampled.

=== "Python"

    ```python
    result = prad.run(
        field,
        energy_MeV=14.7,
        energy_spread_percent=5.0,   # σ = 0.735 MeV
    )
    ```

=== "TOML"

    ```toml
    [source]
    energy_MeV            = 14.7
    energy_spread_percent = 5.0
    ```

A [seeded RNG](input_decks.md) makes the spread reproducible:

```toml
[source]
energy_MeV            = 14.7
energy_spread_percent = 5.0
seed                  = 42
```

---

## Exponential / TNSA

The Target Normal Sheath Acceleration (TNSA) mechanism in laser-plasma experiments produces
a quasi-exponential proton energy spectrum:

```
dN/dE ∝ exp(−E / T)    for 0 < E ≤ E_cut
```

where **T** is the effective proton temperature and **E_cut** is the maximum (cutoff) energy,
set by the sheath potential. For a given laser pulse, T and E_cut are experimentally
measurable from time-of-flight or RCF stack data.

For a pure exponential (E_cut ≫ T), the mean energy equals T:

```
⟨E⟩ = T  (when E_cut / T → ∞)
```

### Sampling

prad uses **closed-form inverse-CDF sampling** — no rejection loops:

```
E = −T · ln(1 − u · (1 − exp(−E_cut / T)))    u ~ Uniform(0, 1)
```

This is O(1) per particle and preserves the exact distribution including the effect of
the finite cutoff.

=== "Python"

    ```python
    result = prad.run(
        field,
        temperature_MeV=3.0,    # T = 3 MeV
        cutoff_MeV=40.0,        # E_cut = 40 MeV
        n_particles=200_000,
        dt_ps=0.1,              # smaller dt for the high-energy tail
        max_steps=50_000,
    )
    ```

=== "TOML"

    ```toml
    [source]
    energy_MeV      = 14.7    # nominal — overridden by spectrum
    temperature_MeV = 3.0
    cutoff_mev      = 40.0
    n_particles     = 200000
    ```

!!! note "Key casing: TOML vs Python API"
    TOML keys are case-sensitive. Use `cutoff_mev` in deck files.
    The Python API uses `cutoff_MeV` as the keyword argument name.

!!! tip "dt and max_steps for broad spectra"
    The high-energy tail of a TNSA spectrum moves faster than the nominal energy.
    Use a smaller `dt_ps` (e.g. `0.1`) and more `max_steps` (e.g. `50_000`) when
    the cutoff energy is much larger than the typical 14.7 MeV mono case.
    Run `proton_tracer explain deck.toml` to see the recommended dt for your config.

### Typical TNSA parameters

| Experiment type | T [MeV] | E_cut [MeV] |
|---|---|---|
| Short-pulse (< 1 ps), moderate intensity | 1–3 | 10–30 |
| High-intensity, long-pulse | 3–8 | 30–80 |
| PW-class lasers | 5–15 | 50–100+ |

### Validation

Test 11 in the validation suite verifies:

- All impact energies ≤ E_cut + 0.05 MeV (hard cutoff enforced)
- Mean KE within 20 % of T (correct exponential shape)
- `std / mean > 0.3` (confirmed non-monoenergetic)

```
Test 11: Exponential / TNSA spectrum  (T=3 MeV, cutoff=40 MeV)
   hits=20000  mean=2.995 MeV  std=2.97 MeV  max=30.4 MeV  → PASS
```

---

## Interaction with relativistic Boris

All three spectrum modes feed into the **relativistic Boris integrator**.
Regardless of the energy sampled, `prad` initialises the particle with the exact
specific momentum:

```
|u| = c · √(γ² − 1)    γ = 1 + E / (m_p c²)
```

and the shader stores `u = γv` throughout the trajectory.
Impact kinetic energy is recovered as `KE = (γ − 1) m_p c²`.

This means even a 60 MeV TNSA particle is pushed relativistically — the 6 % γ correction
matters at that energy.
