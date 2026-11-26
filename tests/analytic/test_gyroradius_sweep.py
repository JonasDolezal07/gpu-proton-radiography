"""
Uniform-B gyroradius sweep — direct r_L measurement.

For a proton moving perpendicular to a uniform B field, the trajectory is a
circular arc of radius r_L = γ m v / (q B). Test 18 in `validate.py` pins this
at one point (14.7 MeV, 0.1 T) via a mixed cycloid + straight-line detector
formula. This test sweeps three energies × three field strengths and inverts
the geometry to recover r_L directly from the detector hit, which is what
`PLAN.md` §1 actually asks for.

Geometry: source and detector are both placed **inside** a large uniform-Bz
domain, so the whole trajectory is a single circular arc — no straight-line
piece before or after. The source is a pencil beam along +x at
(x_src, 0, 0); the detector plane is at x = x_det, normal +x. With chord
length L = x_det - x_src and detector deflection magnitude d = |y_hit|, the
geometry of the arc yields

    r_L_measured = (L² + d²) / (2 d)

which we compare against r_L_theory = γ m v / (q B).

Tolerance: 1 % relative on r_L. Rationale:
  - Test 18 already passes at 1 % relative on the closed-form deflection with
    dt = 0.5 ps; here we use dt = 0.2 ps (2.5× tighter), and the same Boris
    integrator applies. The tolerance therefore does not need to be looser
    than t18's — 1 % is the physics-relevant bar the paper already relies on.
  - Above 1 % the test would stop being a check on the pusher and start
    tolerating drift that the single-point t18 would already flag.
"""

from __future__ import annotations

import math

import numpy as np
import pytest


# ── physical constants ──────────────────────────────────────────────────────
C     = 2.99792458e8            # m/s
Q     = 1.602176634e-19         # C
MP_C2 = 938.272046              # MeV
MP_KG = MP_C2 * 1.602176634e-13 / C**2   # kg


# ── geometry ────────────────────────────────────────────────────────────────
# Source and detector both sit inside a uniformly-Bz domain that extends
# beyond the trajectory in every direction. Chord length L = 100 mm.
X_SRC_MM = -50.0
X_DET_MM = +50.0
L_CHORD_MM = X_DET_MM - X_SRC_MM

# Field domain: much larger than the arc so trilinear interpolation returns
# the uniform value everywhere the particle can visit. In -y the max
# deflection is ~35 mm (worst case: 5 MeV, 2 T); we leave >2× margin.
FIELD_BOUNDS = (-0.20, 0.20, -0.10, 0.02, -0.02, 0.02)


# ── sweep grid ──────────────────────────────────────────────────────────────
# Three energies span the range the code is used in (a few MeV → tens of MeV).
# Three field strengths span typical HED-plasma diagnostic regimes.
SWEEP = [
    (5.0,   0.5),
    (5.0,   1.0),
    (5.0,   2.0),
    (14.7,  0.5),
    (14.7,  1.0),
    (14.7,  2.0),
    (60.0,  0.5),
    (60.0,  1.0),
    (60.0,  2.0),
]

TOL_REL = 0.01


def _larmor_radius(energy_MeV: float, B_T: float) -> float:
    """Relativistic Larmor radius r_L = γ m v / (q B) [m]."""
    gamma = 1.0 + energy_MeV / MP_C2
    beta  = math.sqrt(1.0 - 1.0 / gamma**2)
    v     = beta * C
    p_SI  = gamma * MP_KG * v
    return p_SI / (Q * B_T)


@pytest.mark.parametrize(
    "energy_MeV,B_T", SWEEP,
    ids=[f"E{e:g}MeV-B{b:g}T" for e, b in SWEEP],
)
def test_gyroradius_matches_relativistic_formula(
    energy_MeV: float,
    B_T: float,
    bin_path,             # noqa: ARG001 — used via val_helpers.run_deck
    vulkan_env,           # noqa: ARG001 — configured inside run_deck
    tmp_out_dir,
    val_helpers,
) -> None:
    r_L_theory = _larmor_radius(energy_MeV, B_T)

    # Sanity: the chord must be shorter than the diameter, else the proton
    # would loop back before reaching x_det.
    assert L_CHORD_MM * 1e-3 < 2.0 * r_L_theory, (
        "sweep point ill-posed: chord exceeds 2 r_L"
    )

    # Write uniform-Bz field. Uniform → grid resolution is irrelevant.
    work = tmp_out_dir.parent / "gyro"
    work.mkdir(parents=True, exist_ok=True)
    tag = f"E{energy_MeV:g}_B{B_T:g}"
    bfld_path = work / f"gyro_{tag}.bfld"

    B = np.zeros((8, 8, 8, 3), dtype=np.float32)
    B[..., 2] = float(B_T)
    E = np.zeros_like(B)
    val_helpers.write_bfld(bfld_path, B, E, FIELD_BOUNDS)

    deck_path = work / f"gyro_{tag}.toml"
    deck_path.write_text(f"""\
[field]
path = "{bfld_path}"
scale_B = 1.0
scale_E = 0.0

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

    out_dir = tmp_out_dir / f"gyro_{tag}"
    assert val_helpers.run_deck(deck_path, out_dir), (
        f"tracer failed for E={energy_MeV} MeV, B={B_T} T"
    )

    hits = val_helpers.read_hits_bin(out_dir)
    assert len(hits) > 1000, (
        f"too few hits ({len(hits)}) for E={energy_MeV} MeV, B={B_T} T"
    )

    # For +Bz and a proton moving in +x, F = qv × B is in -y. Hits must
    # therefore lie in -y; a positive mean would mean a sign error.
    y_mean_mm = float(np.mean(hits[:, 0]))
    assert y_mean_mm < 0.0, (
        f"deflection sign wrong: mean_y = {y_mean_mm:+.3f} mm "
        f"(expected negative for +Bz proton)"
    )

    d_mm = -y_mean_mm
    r_L_measured_m = ((L_CHORD_MM**2 + d_mm**2) / (2.0 * d_mm)) * 1e-3

    rel_err = abs(r_L_measured_m - r_L_theory) / r_L_theory
    assert rel_err < TOL_REL, (
        f"r_L mismatch for E={energy_MeV} MeV, B={B_T} T: "
        f"theory {r_L_theory:.5f} m, measured {r_L_measured_m:.5f} m, "
        f"rel_err {rel_err:.4%} > {TOL_REL:.2%}"
    )
