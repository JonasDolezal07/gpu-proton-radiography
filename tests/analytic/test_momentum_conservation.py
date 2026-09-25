"""
Direct |p| conservation in a static magnetic field.

Magnetic forces do no work, so a proton moving through a pure B field must
preserve its momentum magnitude |p| exactly. Test 4 in `validate.py` pins
kinetic-energy conservation; for pure B the two quantities are equivalent,
but `PLAN.md` §1 explicitly names |p|, so this test measures |p| directly
from the detector-hit kinetic energy and compares against the initial value.

Setup: pencil beam at 14.7 MeV through uniform Bz = 0.1 T, matching test 18
so the geometry is a proven-good configuration (particles reliably reach the
detector).

|p|c in MeV/c is recovered from KE via the exact relativistic identity

    (|p|c)² = (KE + m c²)² − (m c²)²

Tolerances:
  - Mean:  |mean(|p|) − |p|₀| / |p|₀  <  1 × 10⁻⁴.
  - Std:   std(|p|) / |p|₀             <  1 × 10⁻⁴.

Why 10⁻⁴: matches test 4's kinetic-energy tolerance, which is the empirical
Boris floor with a pencil beam (all particles execute identical ops → true
std ≈ 0, residual std ≈ f64 precision). The relation Δ|p|/|p| ≈ Δ(KE)/(β KE)
means a 10⁻⁴ tolerance on |p| here is slightly *stricter* than 10⁻⁴ on KE
at 14.7 MeV (β ≈ 0.175), so this bar is meaningful, not free.
"""

from __future__ import annotations

import math

import numpy as np


MP_C2 = 938.272046   # MeV


def _p_c_from_ke(ke_MeV: np.ndarray | float) -> np.ndarray | float:
    """|p|·c in MeV from kinetic energy in MeV (exact relativistic form)."""
    total = np.asarray(ke_MeV, dtype=np.float64) + MP_C2
    return np.sqrt(total * total - MP_C2 * MP_C2)


def test_momentum_magnitude_conserved_in_pure_B(
    bin_path,             # noqa: ARG001
    vulkan_env,           # noqa: ARG001
    tmp_out_dir,
    val_helpers,
) -> None:
    energy_MeV = 14.7
    B_T        = 0.1

    p0_c_MeV = float(_p_c_from_ke(energy_MeV))

    work = tmp_out_dir.parent / "pmag"
    work.mkdir(parents=True, exist_ok=True)

    B = np.zeros((16, 16, 16, 3), dtype=np.float32)
    B[..., 2] = float(B_T)
    E = np.zeros_like(B)
    bfld_path = work / "pmag_Bz.bfld"
    val_helpers.write_bfld(bfld_path, B, E,
                           (-0.06, 0.06, -0.06, 0.06, -0.06, 0.06))

    deck_path = work / "pmag.toml"
    deck_path.write_text(f"""\
[field]
path = "{bfld_path}"
scale_B = 1.0
scale_E = 0.0

[source]
type = "pencil"
n_particles = 10000
energy_MeV = {energy_MeV}
position_mm = [-100.0, 0.0, 0.0]
aim_at_mm = [0.0, 0.0, 0.0]

[detector]
center_mm = [110.0, 0.0, 0.0]
normal = [1.0, 0.0, 0.0]
up = [0.0, 1.0, 0.0]
width_mm = 500.0
height_mm = 500.0
pixels = [256, 256]

[numerics]
dt_ps = 0.5
max_steps = 20000
""")

    out_dir = tmp_out_dir / "pmag"
    assert val_helpers.run_deck(deck_path, out_dir), "tracer failed"

    hits = val_helpers.read_hits_bin(out_dir)
    assert len(hits) > 1000, f"too few hits: {len(hits)}"

    p_c = _p_c_from_ke(hits[:, 2])
    mean_p = float(np.mean(p_c))
    std_p  = float(np.std(p_c))

    rel_mean_err = abs(mean_p - p0_c_MeV) / p0_c_MeV
    rel_std      = std_p / p0_c_MeV
    tol = 1e-4

    assert rel_mean_err < tol, (
        f"|p| mean drift: |p|₀·c = {p0_c_MeV:.6f} MeV, "
        f"measured mean = {mean_p:.6f} MeV, rel_err = {rel_mean_err:.2e}"
    )
    assert rel_std < tol, (
        f"|p| spread: std/|p|₀ = {rel_std:.2e} > {tol:.0e}"
    )


def test_momentum_helper_is_self_consistent() -> None:
    """
    The |p| ↔ KE identity is critical to the assertion above; double-check
    against an independent Lorentz-factor calculation so a bug in the helper
    can't mask an integrator failure.
    """
    ke = 14.7
    gamma = 1.0 + ke / MP_C2
    beta_gamma = math.sqrt(gamma * gamma - 1.0)
    expected = beta_gamma * MP_C2
    got = float(_p_c_from_ke(ke))
    assert math.isclose(got, expected, rel_tol=1e-12), (got, expected)
