"""
Tight-tolerance uniform-E deflection.

A proton entering a uniform E_y field traces a hyperbolic trajectory (the
relativistic analogue of the classical qE t²/2 m parabola). Test 3 in
`validate.py` only checks the *sign* of the deflection — it tolerates a
factor of 3 on the magnitude — because it uses a non-relativistic magnitude
estimate. This test uses the exact relativistic closed form and pins the
deflection to 1 % relative.

Closed form. Take the beam moving in +x with initial momentum p_x = γ₀ m v₀
into a uniform E_y over field length L. Since F_x = 0 the x-momentum is
conserved; only γ and p_y evolve. Let

    ξ  =  q E_y L / (p_x c)      (dimensionless)
    E₀ =  γ₀ m c²

Then at the field exit (position L along x, measured from the field entry):

    y_exit           =  (E₀ / q E_y) · (cosh ξ − 1)
    tan θ_exit       =  sinh ξ / β₀

Outside the field the trajectory is straight, so at a downstream detector at
x = x_det (measured from field exit at x = x_field_exit):

    y_det  =  y_exit  +  tan θ_exit · (x_det − x_field_exit)

`sinh` and `cosh` are exact — no small-angle approximation — so any residual
is pure integrator error.

Tolerance: 1 % relative on y_det, matching test 18's tolerance on the Larmor
cycloid (the same Boris integrator applies to both). The smallest sweep
point has y_det ≈ 1 mm, so 1 % ≈ 10 μm, comfortably above the ~μm-level
noise floor from float32 position accumulation over 10³-10⁴ steps.
"""

from __future__ import annotations

import math

import numpy as np
import pytest


# ── physical constants ──────────────────────────────────────────────────────
C     = 2.99792458e8
Q     = 1.602176634e-19
MP_C2_MEV = 938.272046
MEV_TO_J  = 1.602176634e-13


# ── geometry ────────────────────────────────────────────────────────────────
# Source outside the field (x = -100 mm), field domain half-width 60 mm in x,
# detector 50 mm past the field exit. Mirrors t18's layout so the extrapolation
# arithmetic is identical between the two tests.
X_SRC_MM         = -100.0
X_FIELD_MIN_MM   =  -60.0
X_FIELD_MAX_MM   =  +60.0
X_DET_MM         = +110.0
L_FIELD_M        = (X_FIELD_MAX_MM - X_FIELD_MIN_MM) * 1e-3   # 0.12 m
DELTA_X_POST_M   = (X_DET_MM - X_FIELD_MAX_MM) * 1e-3         # 0.05 m


SWEEP = [
    # (Ey_MV_per_m, energy_MeV)
    (5.0,   14.7),
    (10.0,  14.7),
    (5.0,   30.0),
    (10.0,  30.0),
]

TOL_REL = 0.01


def _analytic_y_det_mm(Ey_MV_per_m: float, energy_MeV: float) -> float:
    """Exact relativistic uniform-E deflection at the detector plane (mm)."""
    Ey_SI  = Ey_MV_per_m * 1e6                          # V/m
    gamma0 = 1.0 + energy_MeV / MP_C2_MEV
    beta0  = math.sqrt(1.0 - 1.0 / gamma0**2)
    E0_J   = gamma0 * MP_C2_MEV * MEV_TO_J              # γ₀ m c²  [J]
    pxc_J  = beta0 * E0_J                               # p_x c    [J]
    qE     = Q * Ey_SI                                  # force    [N]

    xi = qE * L_FIELD_M / pxc_J
    y_exit_m   = (E0_J / qE) * (math.cosh(xi) - 1.0)
    tan_theta  = math.sinh(xi) / beta0
    y_det_m    = y_exit_m + tan_theta * DELTA_X_POST_M
    return y_det_m * 1000.0


@pytest.mark.parametrize(
    "Ey_MV_per_m,energy_MeV", SWEEP,
    ids=[f"Ey{ey:g}MVpm-E{e:g}MeV" for ey, e in SWEEP],
)
def test_uniform_E_parabolic_deflection_matches_closed_form(
    Ey_MV_per_m: float,
    energy_MeV: float,
    bin_path,             # noqa: ARG001
    vulkan_env,           # noqa: ARG001
    tmp_out_dir,
    val_helpers,
) -> None:
    y_det_analytic_mm = _analytic_y_det_mm(Ey_MV_per_m, energy_MeV)
    assert y_det_analytic_mm > 0.1, (
        f"sweep point too small to resolve: analytic y_det = {y_det_analytic_mm} mm"
    )

    work = tmp_out_dir.parent / "eparab"
    work.mkdir(parents=True, exist_ok=True)
    tag = f"Ey{Ey_MV_per_m:g}_E{energy_MeV:g}"

    B = np.zeros((16, 16, 16, 3), dtype=np.float32)
    E = np.zeros_like(B)
    E[..., 1] = float(Ey_MV_per_m * 1e6)      # E_y in V/m
    bfld_path = work / f"eparab_{tag}.bfld"
    val_helpers.write_bfld(bfld_path, B, E,
                           (X_FIELD_MIN_MM * 1e-3, X_FIELD_MAX_MM * 1e-3,
                            -0.06, 0.06, -0.06, 0.06))

    deck_path = work / f"eparab_{tag}.toml"
    deck_path.write_text(f"""\
[field]
path = "{bfld_path}"
scale_B = 0.0
scale_E = 1.0

[source]
type = "pencil"
n_particles = 10000
energy_MeV = {energy_MeV}
position_mm = [{X_SRC_MM}, 0.0, 0.0]
aim_at_mm = [0.0, 0.0, 0.0]

[detector]
center_mm = [{X_DET_MM}, 0.0, 0.0]
normal = [1.0, 0.0, 0.0]
up = [0.0, 1.0, 0.0]
width_mm = 500.0
height_mm = 500.0
pixels = [256, 256]

[numerics]
dt_ps = 0.2
max_steps = 40000
""")

    out_dir = tmp_out_dir / f"eparab_{tag}"
    assert val_helpers.run_deck(deck_path, out_dir), (
        f"tracer failed for Ey={Ey_MV_per_m} MV/m, E={energy_MeV} MeV"
    )

    hits = val_helpers.read_hits_bin(out_dir)
    assert len(hits) > 1000, f"too few hits: {len(hits)}"

    y_gpu_mm = float(np.mean(hits[:, 0]))
    assert y_gpu_mm > 0.0, (
        f"deflection sign wrong: mean_y = {y_gpu_mm:+.4f} mm "
        f"(expected +y for proton in +Ey)"
    )

    rel_err = abs(y_gpu_mm - y_det_analytic_mm) / y_det_analytic_mm
    assert rel_err < TOL_REL, (
        f"Ey={Ey_MV_per_m} MV/m, E={energy_MeV} MeV: "
        f"analytic y_det = {y_det_analytic_mm:.4f} mm, "
        f"GPU = {y_gpu_mm:.4f} mm, rel_err = {rel_err:.4%} > {TOL_REL:.2%}"
    )
